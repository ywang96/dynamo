// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use anyhow::Result;
use futures::StreamExt;
use tokio::sync::OwnedSemaphorePermit;
use tracing::Instrument;

use dynamo_runtime::{
    pipeline::{ManyOut, SingleIn},
    protocols::{annotated::Annotated, maybe_error::MaybeError},
};

use super::{PrefillCompletion, PrefillError, PrefillRouter};
use crate::{
    http::service::error::HttpError,
    kv_router::KvPushRouter,
    protocols::common::{
        FinishReason,
        llm_backend::{LLMEngineOutput, PreprocessedRequest},
        timing::RequestTracker,
    },
    session_affinity::{AffinityTarget, SessionAffinityPushRouter},
};

/// Extract a status-bearing rejection from a prefill worker data frame.
///
/// The vLLM worker uses `finish_reason: Error({"message", "code"})` for
/// request validation failures. Since that error is carried by the data frame
/// instead of the annotated SSE envelope, `Annotated::err()` cannot see it.
fn prefill_worker_error(output: &LLMEngineOutput) -> Option<HttpError> {
    let Some(FinishReason::Error(raw_message)) = output.finish_reason.as_ref() else {
        return None;
    };

    let Ok(payload) = serde_json::from_str::<serde_json::Value>(raw_message) else {
        return Some(HttpError {
            code: 500,
            message: raw_message.clone(),
        });
    };

    let message = payload
        .get("message")
        .and_then(serde_json::Value::as_str)
        .unwrap_or(raw_message)
        .to_string();
    let code = payload
        .get("code")
        .and_then(serde_json::Value::as_u64)
        .and_then(|code| u16::try_from(code).ok())
        .unwrap_or(500);

    Some(HttpError { code, message })
}

pub(super) enum InnerPrefillRouter {
    KvRouter(Arc<KvPushRouter>),
    SimpleRouter(Arc<SessionAffinityPushRouter>),
}

impl InnerPrefillRouter {
    pub(super) async fn select_and_dispatch_prefill<M, F>(
        &self,
        request: SingleIn<PreprocessedRequest>,
        prepare: F,
    ) -> Result<(M, ManyOut<Annotated<LLMEngineOutput>>)>
    where
        F: FnOnce(&mut PreprocessedRequest, AffinityTarget) -> Result<M>,
    {
        match self {
            InnerPrefillRouter::KvRouter(router) => {
                router.select_and_dispatch_prefill(request, prepare).await
            }
            InnerPrefillRouter::SimpleRouter(router) => {
                router.select_and_dispatch_prefill(request, prepare).await
            }
        }
    }
}

impl PrefillRouter {
    pub(super) async fn consume_prefill_stream(
        mut prefill_response: ManyOut<Annotated<LLMEngineOutput>>,
        tracker: Option<Arc<RequestTracker>>,
    ) -> Result<PrefillCompletion, PrefillError> {
        let Some(first_output) = prefill_response.next().await else {
            return Err(PrefillError::PrefillError(
                "Prefill router returned no output (stream ended)".to_string(),
                None,
            ));
        };

        if let Some(error) = first_output.err() {
            return Err(PrefillError::PrefillError(
                "Prefill router returned error in output".to_string(),
                Some(Box::new(error)),
            ));
        }

        if let Some(error) = first_output.data.as_ref().and_then(prefill_worker_error) {
            return Err(PrefillError::WorkerError(error));
        }

        if let Some(ref tracker) = tracker {
            tracker.record_prefill_complete();
        }

        let mut prompt_tokens_details = first_output
            .data
            .as_ref()
            .and_then(|output| output.completion_usage.as_ref())
            .and_then(|usage| usage.prompt_tokens_details.clone());

        // For SGLang, check if the first output is a bootstrap message.
        let is_bootstrap = first_output
            .data
            .as_ref()
            .and_then(|o| o.disaggregated_params.as_ref())
            .and_then(|p| p.as_object())
            .is_some_and(|obj| {
                obj.contains_key("bootstrap_host")
                    && obj.contains_key("bootstrap_port")
                    && obj.contains_key("bootstrap_room")
            });

        if !is_bootstrap {
            while let Some(next) = prefill_response.next().await {
                if let Some(error) = next.err() {
                    return Err(PrefillError::PrefillError(
                        "Prefill router returned error in output stream".to_string(),
                        Some(Box::new(error)),
                    ));
                }
                if let Some(output) = next.data.as_ref() {
                    if let Some(error) = prefill_worker_error(output) {
                        return Err(PrefillError::WorkerError(error));
                    }
                    if prompt_tokens_details.is_none() {
                        prompt_tokens_details = output
                            .completion_usage
                            .as_ref()
                            .and_then(|usage| usage.prompt_tokens_details.clone());
                    }
                }
            }
        } else {
            tokio::spawn(async move { while prefill_response.next().await.is_some() {} });
        }

        let Some(output) = &first_output.data else {
            return Err(PrefillError::NoDisaggregatedParams(
                "Prefill router output has no data field".to_string(),
            ));
        };
        let Some(disaggregated_params) = output.disaggregated_params.clone() else {
            return Err(PrefillError::NoDisaggregatedParams(
                "Prefill router output missing disaggregated_params".to_string(),
            ));
        };

        Ok(PrefillCompletion {
            result: crate::protocols::common::preprocessor::PrefillResult {
                disaggregated_params,
                prompt_tokens_details,
            },
            worker_link: output.worker_trace_link.clone(),
        })
    }

    pub(super) fn spawn_prefill_task(
        &self,
        prefill_stream: ManyOut<Annotated<LLMEngineOutput>>,
        tracker: Option<Arc<RequestTracker>>,
        phase_transition_permit: OwnedSemaphorePermit,
    ) {
        let span = tracing::Span::current();
        tokio::spawn(
            async move {
                drop(phase_transition_permit);
                match Self::consume_prefill_stream(prefill_stream, tracker).await {
                    Ok(_) => tracing::debug!("Prefill background task completed"),
                    Err(error) => tracing::warn!("Prefill background task error: {error:?}"),
                }
            }
            .instrument(span),
        );
    }
}

#[cfg(test)]
mod tests {
    use futures::stream;
    use serde_json::json;

    use dynamo_runtime::pipeline::{ResponseStream, context::Controller};

    use super::*;

    fn prefill_stream(
        items: Vec<Annotated<LLMEngineOutput>>,
    ) -> ManyOut<Annotated<LLMEngineOutput>> {
        ResponseStream::new(
            Box::pin(stream::iter(items)),
            Arc::new(Controller::default()),
        )
    }

    fn valid_prefill_output() -> Annotated<LLMEngineOutput> {
        Annotated::from_data(LLMEngineOutput {
            disaggregated_params: Some(json!({})),
            ..Default::default()
        })
    }

    fn rejected_prefill_output(message: &str, code: u16) -> Annotated<LLMEngineOutput> {
        Annotated::from_data(LLMEngineOutput::error(
            json!({"message": message, "code": code}).to_string(),
        ))
    }

    #[tokio::test]
    async fn first_output_error_does_not_record_prefill_complete() {
        let tracker = Arc::new(RequestTracker::new());
        let result = PrefillRouter::consume_prefill_stream(
            prefill_stream(vec![Annotated::from_error("prefill failed")]),
            Some(tracker.clone()),
        )
        .await;

        assert!(result.is_err());
        assert!(tracker.record_prefill_complete());
    }

    #[tokio::test]
    async fn later_output_error_is_propagated_after_prefill_arrival() {
        let tracker = Arc::new(RequestTracker::new());
        let result = PrefillRouter::consume_prefill_stream(
            prefill_stream(vec![
                valid_prefill_output(),
                Annotated::from_error("prefill stream failed"),
            ]),
            Some(tracker.clone()),
        )
        .await;

        assert!(result.is_err());
        assert!(!tracker.record_prefill_complete());
    }

    #[tokio::test]
    async fn first_output_worker_rejection_preserves_status_and_message() {
        let result = PrefillRouter::consume_prefill_stream(
            prefill_stream(vec![rejected_prefill_output("invalid image", 400)]),
            None,
        )
        .await;

        let error = match result {
            Err(PrefillError::WorkerError(error)) => error,
            _ => panic!("expected worker error"),
        };
        assert_eq!(error.code, 400);
        assert_eq!(error.message, "invalid image");
    }

    #[tokio::test]
    async fn later_worker_rejection_preserves_status_and_message() {
        let result = PrefillRouter::consume_prefill_stream(
            prefill_stream(vec![
                valid_prefill_output(),
                rejected_prefill_output("image too large", 413),
            ]),
            None,
        )
        .await;

        let error = match result {
            Err(PrefillError::WorkerError(error)) => error,
            _ => panic!("expected worker error"),
        };
        assert_eq!(error.code, 413);
        assert_eq!(error.message, "image too large");
    }
}
