// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use super::{NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse};
use crate::{
    protocols::{
        common::{self, extensions::NvExtProvider, timing::RequestTracker},
        openai::{
            convert_backend_top_logprobs,
            delta_common::{self, DeltaGeneratorOptions},
            token_to_utf8_bytes,
        },
    },
    types::TokenIdType,
};

impl NvCreateChatCompletionRequest {
    pub fn enable_usage_for_nonstreaming(&mut self, original_stream_flag: bool) {
        delta_common::enable_usage_for_nonstreaming(
            &mut self.inner.stream_options,
            original_stream_flag,
        );
    }

    pub fn response_generator(&self, request_id: String) -> DeltaGenerator {
        let enable_logprobs =
            self.inner.logprobs.unwrap_or(false) || self.inner.top_logprobs.unwrap_or(0) > 0;
        let options = DeltaGeneratorOptions::new(
            self.inner.stream_options.as_ref(),
            self.return_tokens_as_token_ids,
            enable_logprobs,
            self.nvext(),
        );
        DeltaGenerator::new(self.inner.model.clone(), options, request_id)
    }
}

/// Generates incremental chat completion responses in a streaming fashion.
pub struct DeltaGenerator {
    /// Unique identifier for the chat completion session.
    id: String,
    /// Object type, representing a streamed chat completion response.
    object: String,
    /// Timestamp (Unix epoch) when the response was created.
    created: u32,
    model: String,
    /// Optional system fingerprint for version tracking.
    system_fingerprint: Option<String>,
    /// Optional service tier information for the response.
    service_tier: Option<dynamo_protocols::types::ServiceTierResponse>,
    /// Tracks token usage for the completion request.
    usage: dynamo_protocols::types::CompletionUsage,
    /// Counter tracking the number of messages issued.
    msg_counter: u64,
    /// Configuration options for response generation.
    options: DeltaGeneratorOptions,
    /// Request tracker for per-request metrics (shared with PreprocessedRequest).
    tracker: Arc<RequestTracker>,
    /// Prompt tokens sent to the model but excluded from API usage for protocol
    /// compatibility. Applied to both the frontend ISL and any authoritative
    /// prompt count later reported by the worker.
    prompt_tokens_adjustment: u32,
    /// Token ids of the reasoning-start marker, resolved from the tokenizer by
    /// the preprocessor. Used to derive `reasoning_tokens` when the backend
    /// reports none.
    reasoning_start_ids: Option<Vec<u32>>,
    /// Token ids of the reasoning-end marker.
    reasoning_end_ids: Option<Vec<u32>>,
    /// Whether the prompt ended inside the reasoning channel.
    is_reasoning_started_in_prompt: bool,
    /// Full generated token-id sequence, accumulated across streamed chunks, used
    /// to locate the configured reasoning span.
    reasoning_output_ids: Vec<u32>,
    /// Completion tokens per candidate index, for the per-candidate usage
    /// snapshot on each candidate's end frame (streaming spec P0.4 / §5.5).
    per_candidate_completion_tokens: std::collections::HashMap<u32, u32>,
}

impl DeltaGenerator {
    pub fn new(model: String, options: DeltaGeneratorOptions, request_id: String) -> Self {
        let (now, usage, tracker) = delta_common::initial_state();
        Self {
            id: format!("chatcmpl-{request_id}"),
            object: "chat.completion.chunk".to_string(),
            created: now,
            model,
            system_fingerprint: None,
            service_tier: None,
            usage,
            msg_counter: 0,
            prompt_tokens_adjustment: 0,
            options,
            tracker,
            reasoning_start_ids: None,
            reasoning_end_ids: None,
            is_reasoning_started_in_prompt: false,
            reasoning_output_ids: Vec::new(),
            per_candidate_completion_tokens: std::collections::HashMap::new(),
        }
    }

    /// Derive the reasoning-token count from the generated token-id stream, used
    /// when the backend reports none (vLLM/TRT-LLM/MLX). Reasoning tokens are those
    /// between the configured marker sequences. If the prompt opened the reasoning
    /// channel, counting starts at token 0; seeing only an end marker retains the
    /// legacy prefill auto-detection. An open reasoning span with no end marker runs
    /// to the end (truncated mid-thinking). Returns 0 when no span is present.
    fn derive_reasoning_tokens(
        output_ids: &[u32],
        start_ids: Option<&[u32]>,
        end_ids: Option<&[u32]>,
        is_reasoning_started_in_prompt: bool,
    ) -> u32 {
        let find_marker = |marker: &[u32], from: usize| -> Option<usize> {
            if marker.is_empty() {
                return None;
            }
            output_ids
                .get(from..)?
                .windows(marker.len())
                .position(|window| window == marker)
                .map(|relative| from + relative)
        };

        let explicit_start = start_ids
            .and_then(|marker| find_marker(marker, 0).map(|position| position + marker.len()));
        let start = explicit_start.unwrap_or(0);
        let end = end_ids.and_then(|marker| find_marker(marker, start));
        let has_reasoning_started =
            explicit_start.is_some() || is_reasoning_started_in_prompt || end.is_some();
        if !has_reasoning_started {
            return 0;
        }

        end.unwrap_or(output_ids.len()).saturating_sub(start) as u32
    }

    /// Returns the request tracker. Tracking is enabled. For sharing with PreprocessedRequest.
    pub fn tracker(&self) -> Arc<RequestTracker> {
        self.tracker.clone()
    }

    fn adjusted_prompt_tokens(&self, prompt_tokens: u32) -> u32 {
        prompt_tokens.saturating_sub(self.prompt_tokens_adjustment)
    }

    /// Exclude a request-specific suffix from API prompt usage without changing
    /// the token IDs sent to the model.
    pub(crate) fn set_prompt_tokens_adjustment(&mut self, adjustment: u32) {
        self.prompt_tokens_adjustment = adjustment;
    }

    /// Updates the prompt token usage count.
    ///
    /// # Arguments
    /// * `isl` - Input Sequence Length. The number of prompt tokens used.
    pub fn update_isl(&mut self, isl: u32) {
        self.usage.prompt_tokens = self.adjusted_prompt_tokens(isl);
    }

    pub fn create_logprobs(
        &self,
        tokens: Vec<common::llm_backend::TokenType>,
        token_ids: &[TokenIdType],
        logprobs: Option<common::llm_backend::LogProbs>,
        top_logprobs: Option<common::llm_backend::TopLogprobs>,
    ) -> Option<dynamo_protocols::types::ChatChoiceLogprobs> {
        if !self.options.enable_logprobs || logprobs.is_none() {
            return None;
        }

        let toks = tokens
            .into_iter()
            .zip(token_ids)
            .map(|(token, token_id)| (token.unwrap_or_default(), *token_id))
            .collect::<Vec<(String, TokenIdType)>>();
        let tok_lps = toks
            .iter()
            .zip(logprobs.unwrap())
            .map(|(_, lp)| lp as f32)
            .collect::<Vec<f32>>();

        let return_as_ids = self.options.return_tokens_as_token_ids;
        let content = top_logprobs.map(|top_logprobs| {
            toks.iter()
                .zip(tok_lps)
                .zip(top_logprobs)
                .map(|(((t, tid), lp), top_lps)| {
                    let token_str = if return_as_ids {
                        format!("token_id:{}", tid)
                    } else {
                        t.clone()
                    };
                    let converted =
                        convert_backend_top_logprobs(&top_lps, t, *tid, lp, return_as_ids);
                    dynamo_protocols::types::ChatCompletionTokenLogprob {
                        token: token_str.clone(),
                        logprob: lp,
                        bytes: token_to_utf8_bytes(&token_str),
                        top_logprobs: converted,
                    }
                })
                .collect()
        });

        Some(dynamo_protocols::types::ChatChoiceLogprobs {
            content,
            refusal: None,
        })
    }

    #[allow(deprecated)]
    pub fn create_choice(
        &mut self,
        index: u32,
        text: Option<String>,
        finish_reason: Option<dynamo_protocols::types::FinishReason>,
        logprobs: Option<dynamo_protocols::types::ChatChoiceLogprobs>,
    ) -> NvCreateChatCompletionStreamResponse {
        let delta = dynamo_protocols::types::ChatCompletionStreamResponseDelta {
            content: text.map(dynamo_protocols::types::ChatCompletionMessageContent::Text),
            function_call: None,
            tool_calls: None,
            role: if self.msg_counter == 0 {
                Some(dynamo_protocols::types::Role::Assistant)
            } else {
                None
            },
            refusal: None,
            reasoning_content: None,
        };

        let choice = dynamo_protocols::types::ChatChoiceStream {
            index,
            delta,
            finish_reason,
            logprobs,
        };

        let choices = vec![choice];

        // According to OpenAI spec: when stream_options.include_usage is true,
        // all intermediate chunks should have usage: null
        // The final usage chunk will be sent separately with empty choices
        NvCreateChatCompletionStreamResponse {
            inner: dynamo_protocols::types::CreateChatCompletionStreamResponse {
                id: self.id.clone(),
                object: self.object.clone(),
                created: self.created,
                model: self.model.clone(),
                system_fingerprint: self.system_fingerprint.clone(),
                choices,
                usage: if self.options.enable_usage && self.options.continuous_usage_stats {
                    Some(self.get_usage())
                } else {
                    None
                },
                service_tier: self.service_tier.clone(),
            },
            nvext: None, // Will be populated by router layer if needed
            llm_metrics: None,
            choice_usage: None,
        }
    }

    /// Creates a final usage-only chunk for OpenAI compliance.
    /// This should be sent after the last content chunk when stream_options.include_usage is true.
    ///
    /// # Returns
    /// * A [`CreateChatCompletionStreamResponse`] with empty choices and usage stats.
    pub fn create_usage_chunk(&self) -> NvCreateChatCompletionStreamResponse {
        let usage = self.get_usage();

        NvCreateChatCompletionStreamResponse {
            inner: dynamo_protocols::types::CreateChatCompletionStreamResponse {
                id: self.id.clone(),
                object: self.object.clone(),
                created: self.created,
                model: self.model.clone(),
                system_fingerprint: self.system_fingerprint.clone(),
                choices: vec![], // Empty choices for usage-only chunk
                usage: Some(usage),
                service_tier: self.service_tier.clone(),
            },
            nvext: None,
            llm_metrics: None,
            choice_usage: None,
        }
    }

    /// Check if usage tracking is enabled
    pub fn is_usage_enabled(&self) -> bool {
        self.options.enable_usage
    }

    /// Check if continuous usage tracking is enabled
    pub fn is_continuous_usage_enabled(&self) -> bool {
        self.options.continuous_usage_stats
    }

    pub fn get_usage(&self) -> dynamo_protocols::types::CompletionUsage {
        let mut usage = self.usage.clone();
        usage.total_tokens = usage.prompt_tokens.saturating_add(usage.completion_tokens);

        // Streaming spec P0.2: the usage object must carry `reasoning_tokens` and
        // `cached_tokens` as integers (0 when absent), never null or omitted. The
        // upstream (async-openai) `CompletionUsage` skips the `*_details` objects
        // entirely when `None`, so populate both and zero-fill every numeric field
        // to keep the whole usage object null-free.
        let mut ctd = usage.completion_tokens_details.take().unwrap_or_default();
        // Prefer a backend-reported reasoning-token count; otherwise derive it from
        // the generated token stream (vLLM/TRT-LLM/MLX carry no such count, so
        // `reasoning_tokens` would otherwise be a hardcoded 0 even when the model
        // reasoned).
        let reasoning_tokens = match ctd.reasoning_tokens {
            Some(n) if n > 0 => n,
            _ => Self::derive_reasoning_tokens(
                &self.reasoning_output_ids,
                self.reasoning_start_ids.as_deref(),
                self.reasoning_end_ids.as_deref(),
                self.is_reasoning_started_in_prompt,
            ),
        };
        ctd.reasoning_tokens = Some(reasoning_tokens);
        ctd.audio_tokens = Some(ctd.audio_tokens.unwrap_or(0));
        ctd.accepted_prediction_tokens = Some(ctd.accepted_prediction_tokens.unwrap_or(0));
        ctd.rejected_prediction_tokens = Some(ctd.rejected_prediction_tokens.unwrap_or(0));
        usage.completion_tokens_details = Some(ctd);

        let mut ptd = usage.prompt_tokens_details.take().unwrap_or_default();
        ptd.cached_tokens = Some(ptd.cached_tokens.unwrap_or(0));
        ptd.audio_tokens = Some(ptd.audio_tokens.unwrap_or(0));
        usage.prompt_tokens_details = Some(ptd);

        usage
    }
}

/// Implements the [`crate::protocols::openai::DeltaGeneratorExt`] trait for [`DeltaGenerator`], allowing
/// it to transform backend responses into OpenAI-style streaming responses.
impl crate::protocols::openai::DeltaGeneratorExt<NvCreateChatCompletionStreamResponse>
    for DeltaGenerator
{
    fn set_reasoning_markers(
        &mut self,
        start_ids: Option<Vec<u32>>,
        end_ids: Option<Vec<u32>>,
        is_reasoning_started_in_prompt: bool,
    ) {
        self.reasoning_start_ids = start_ids;
        self.reasoning_end_ids = end_ids;
        self.is_reasoning_started_in_prompt = is_reasoning_started_in_prompt;
    }

    /// Converts a backend response into a structured OpenAI-style streaming response.
    ///
    /// * `delta` - The backend response containing generated text and metadata.
    fn choice_from_postprocessor(
        &mut self,
        delta: crate::protocols::common::llm_backend::BackendOutput,
    ) -> anyhow::Result<NvCreateChatCompletionStreamResponse> {
        // Aggregate token usage even if usage tracking is disabled for metrics tracking
        // SAFETY: Casting from `usize` to `u32` could lead to precision loss after `u32::MAX`,
        // but this will not be an issue until context lengths exceed 4_294_967_295.
        let token_length: u32 = delta
            .token_ids
            .len()
            .try_into()
            .expect("token_ids length exceeds u32::MAX");

        self.usage.completion_tokens += token_length;
        let candidate_index = delta.index.unwrap_or(0);
        *self
            .per_candidate_completion_tokens
            .entry(candidate_index)
            .or_default() += token_length;

        // Accumulate the full generated token-id stream so `get_usage` can locate
        // the configured reasoning-boundary sequences.
        self.reasoning_output_ids
            .extend_from_slice(&delta.token_ids);

        // If backend provides completion_usage, use it to update usage stats
        // This is critical for prompt embeddings where prompt_tokens comes from
        // the embedding sequence length computed by the worker
        if let Some(completion_usage) = delta.completion_usage.as_ref() {
            // Update prompt_tokens from worker if provided (e.g., for embeddings)
            self.usage.prompt_tokens = self.adjusted_prompt_tokens(completion_usage.prompt_tokens);

            // Propagate prompt token details if provided
            if let Some(prompt_details) = completion_usage.prompt_tokens_details.as_ref() {
                self.usage.prompt_tokens_details = Some(prompt_details.clone());
            }

            // Propagate completion token details (e.g. reasoning_tokens) if the
            // backend reports them. The chat path previously dropped these, so a
            // backend-reported reasoning_tokens count never reached the usage
            // frame (the completions path already does this).
            if let Some(completion_details) = completion_usage.completion_tokens_details.as_ref() {
                self.usage.completion_tokens_details = Some(completion_details.clone());
            }
        }

        let logprobs = self.create_logprobs(
            delta.tokens,
            &delta.token_ids,
            delta.log_probs,
            delta.top_logprobs,
        );

        // Map backend finish reasons to OpenAI's finish reasons.
        let finish_reason = match delta.finish_reason {
            Some(common::FinishReason::EoS) => Some(dynamo_protocols::types::FinishReason::Stop),
            Some(common::FinishReason::Stop) => Some(dynamo_protocols::types::FinishReason::Stop),
            Some(common::FinishReason::Length) => {
                Some(dynamo_protocols::types::FinishReason::Length)
            }
            Some(common::FinishReason::Cancelled) => {
                Some(dynamo_protocols::types::FinishReason::Stop)
            }
            Some(common::FinishReason::ContentFilter) => {
                Some(dynamo_protocols::types::FinishReason::ContentFilter)
            }
            Some(common::FinishReason::Error(err_msg)) => {
                return Err(anyhow::anyhow!(err_msg));
            }
            None => None,
        };
        let stop_reason = delta.stop_reason.clone();

        // Create the streaming response.
        let index = delta.index.unwrap_or(0);
        let mut stream_response = self.create_choice(index, delta.text, finish_reason, logprobs);

        // Streaming spec P0.4 / §5.5: the candidate's end frame carries a
        // per-candidate usage snapshot (same null-free shape as the summary
        // frame, completion tokens scoped to this candidate). Serialized as
        // `choices[0].usage` by the manual Serialize impl on the response.
        if finish_reason.is_some() {
            let mut candidate_usage = self.get_usage();
            let candidate_completion = self
                .per_candidate_completion_tokens
                .get(&index)
                .copied()
                .unwrap_or(candidate_usage.completion_tokens);
            candidate_usage.completion_tokens = candidate_completion;
            candidate_usage.total_tokens = candidate_usage
                .prompt_tokens
                .saturating_add(candidate_completion);
            stream_response.choice_usage = Some(candidate_usage);
        }

        // Record finish for timing/ITL accounting even when timing is not returned to the client.
        // Kept at call site because it's a side effect on the tracker — not a gating decision.
        if finish_reason.is_some() {
            self.tracker.record_finish();
        }

        // Build the nvext response payload via the shared gating helper on
        // `NvExtResponseFieldSelection` (see `nvext.rs`). Both chat and
        // completions delta generators go through the same helper so the gating
        // rules stay in one place.
        let prompt_logprobs_payload =
            common::llm_backend::prompt_logprobs_from_engine_data(delta.engine_data.as_ref());
        let completion_token_ids_slice: &[u32] = &delta.token_ids;
        if let Some(nvext_response) = self.options.response_fields.build_response_nvext(
            Some(&self.tracker),
            finish_reason.is_some(),
            delta.engine_data,
            stop_reason,
            Some(completion_token_ids_slice),
            prompt_logprobs_payload,
        ) && let Ok(nvext_json) = serde_json::to_value(&nvext_response)
        {
            stream_response.nvext = Some(nvext_json);
            if let Some(ref info) = nvext_response.worker_id {
                tracing::debug!(
                    "Injected worker_id into chat completion nvext: prefill={:?}, decode={:?}",
                    info.prefill_worker_id,
                    info.decode_worker_id
                );
            }
            if let Some(ref tokens) = nvext_response.token_ids {
                tracing::debug!(
                    "Injected token_ids into chat completion nvext: {} tokens",
                    tokens.len()
                );
            }
            if let Some(ref tokens) = nvext_response.completion_token_ids {
                tracing::debug!(
                    "Injected completion_token_ids into chat completion nvext: {} tokens",
                    tokens.len()
                );
            }
        }

        Ok(stream_response)
    }

    fn get_isl(&self) -> Option<u32> {
        Some(self.usage.prompt_tokens)
    }

    fn create_usage_chunk(&self) -> NvCreateChatCompletionStreamResponse {
        DeltaGenerator::create_usage_chunk(self)
    }

    fn is_usage_enabled(&self) -> bool {
        DeltaGenerator::is_usage_enabled(self)
    }

    fn is_continuous_usage_enabled(&self) -> bool {
        DeltaGenerator::is_continuous_usage_enabled(self)
    }

    fn get_usage(&self) -> dynamo_protocols::types::CompletionUsage {
        DeltaGenerator::get_usage(self)
    }

    fn tracker(&self) -> Option<Arc<RequestTracker>> {
        Some(self.tracker.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::common::{self, llm_backend::BackendOutput, timing::WORKER_TYPE_PREFILL};
    use crate::protocols::openai::DeltaGeneratorExt;

    #[test]
    fn derive_reasoning_tokens_counts_the_think_span() {
        let start = [100u32]; // <think>
        let end = [200u32]; // </think>
        let derive = |output: &[u32]| {
            DeltaGenerator::derive_reasoning_tokens(output, Some(&start), Some(&end), false)
        };

        // Toggleable model, thinking on: explicit <think> in the output; reasoning
        // is between the markers. [c, <think>, r, r, </think>, c] -> 2.
        assert_eq!(derive(&[9, 100, 1, 2, 200, 9]), 2);
        // Toggleable, <think> emitted but truncated before </think>: to the end.
        // [c, <think>, r, r, r] -> 3.
        assert_eq!(derive(&[9, 100, 1, 2, 3]), 3);
        // Prefill model: <think> pre-opened (absent from output), </think> present.
        // [r, r, r, </think>, c] -> 3.
        assert_eq!(derive(&[1, 2, 3, 200, 9]), 3);
        // No reasoning (thinking off): neither marker -> 0.
        assert_eq!(derive(&[9, 9, 9]), 0);
        // <think> present but </think> id unresolved -> counts to the end.
        assert_eq!(
            DeltaGenerator::derive_reasoning_tokens(&[9, 100, 1, 2], Some(&start), None, false,),
            2
        );
    }

    #[test]
    fn derive_reasoning_tokens_supports_multi_token_markers() {
        let start = [100, 101, 102]; // <|open|>think<|sep|>
        let end = [200, 101, 102]; // <|close|>think<|sep|>

        assert_eq!(
            DeltaGenerator::derive_reasoning_tokens(
                &[9, 100, 101, 102, 1, 2, 200, 101, 102, 9],
                Some(&start),
                Some(&end),
                false,
            ),
            2
        );
    }

    #[test]
    fn derive_reasoning_tokens_counts_prompt_prefilled_truncation() {
        let start = [100, 101, 102]; // <|open|>think<|sep|>
        let end = [200, 101, 102]; // <|close|>think<|sep|>

        assert_eq!(
            DeltaGenerator::derive_reasoning_tokens(&[1, 2, 3], Some(&start), Some(&end), true,),
            3
        );
    }

    use dynamo_protocols::types::{
        ChatCompletionRequestMessage, ChatCompletionRequestUserMessage,
        ChatCompletionRequestUserMessageContent, CreateChatCompletionRequest,
    };

    fn create_test_request() -> NvCreateChatCompletionRequest {
        let messages = vec![ChatCompletionRequestMessage::User(
            ChatCompletionRequestUserMessage {
                content: ChatCompletionRequestUserMessageContent::Text("test".to_string()),
                name: None,
            },
        )];

        NvCreateChatCompletionRequest {
            inner: CreateChatCompletionRequest {
                model: "test-model".to_string(),
                messages,
                stream: Some(false),
                stream_options: None,
                ..Default::default()
            },
            common: Default::default(),
            nvext: None,
            chat_template_args: None,
            thinking: None,
            media_io_kwargs: None,
            return_tokens_as_token_ids: None,
            unsupported_fields: Default::default(),
        }
    }

    #[test]
    fn test_enable_usage_for_nonstreaming_enables_usage() {
        // Test that non-streaming requests get usage enabled
        let mut request = create_test_request();
        assert!(request.inner.stream_options.is_none());

        request.enable_usage_for_nonstreaming(false); // false = non-streaming

        assert!(
            request.inner.stream_options.is_some(),
            "Non-streaming request should have stream_options created"
        );
        assert!(
            request.inner.stream_options.unwrap().include_usage,
            "Non-streaming request should have include_usage=true for OpenAI compliance"
        );
        assert!(
            !request.inner.stream_options.unwrap().continuous_usage_stats,
            "Non-streaming request should have continuous_usage_stats=false for OpenAI compliance"
        );
    }

    #[test]
    fn test_enable_usage_for_nonstreaming_ignores_streaming() {
        // Test that streaming requests are not modified
        let mut request = create_test_request();
        assert!(request.inner.stream_options.is_none());

        request.enable_usage_for_nonstreaming(true); // true = streaming

        assert!(
            request.inner.stream_options.is_none(),
            "Streaming request should not have stream_options modified"
        );
    }

    fn make_request_with_nvext(
        nvext: crate::protocols::common::extensions::NvExt,
    ) -> NvCreateChatCompletionRequest {
        let mut request = create_test_request();
        request.nvext = Some(nvext);
        request
    }

    fn final_backend_output() -> BackendOutput {
        BackendOutput {
            token_ids: vec![1],
            tokens: vec![Some("hello".to_string())],
            text: Some("hello".to_string()),
            cum_log_probs: None,
            log_probs: None,
            top_logprobs: None,
            finish_reason: Some(common::FinishReason::Stop),
            stop_reason: None,
            index: Some(0),
            completion_usage: None,
            disaggregated_params: None,
            worker_trace_link: None,
            // routed_experts rides the engine's opaque passthrough.
            engine_data: Some(serde_json::json!({
                "routed_experts": {"layer_0": [1, 3]}
            })),
            encoder_result: None,
            routing_data: None,
        }
    }

    #[test]
    fn usage_always_includes_reasoning_and_cached_tokens_as_integers() {
        // Streaming spec P0.2: reasoning_tokens + cached_tokens must be present
        // integers (0 when absent), never null/omitted — even when the backend
        // reports no token details.
        let request = create_test_request();
        let mut generator = request.response_generator("req-usage-zero".to_string());
        generator
            .choice_from_postprocessor(final_backend_output())
            .unwrap();
        let usage = generator.get_usage();

        assert_eq!(
            usage
                .completion_tokens_details
                .as_ref()
                .expect("completion_tokens_details must be present (P0.2)")
                .reasoning_tokens,
            Some(0),
            "reasoning_tokens must be integer 0, not null/omitted"
        );
        assert_eq!(
            usage
                .prompt_tokens_details
                .as_ref()
                .expect("prompt_tokens_details must be present (P0.2)")
                .cached_tokens,
            Some(0),
            "cached_tokens must be integer 0, not null/omitted"
        );

        // And the serialized JSON must carry them as integers (never null).
        let v = serde_json::to_value(&usage).unwrap();
        assert!(
            v["completion_tokens_details"]["reasoning_tokens"].is_u64(),
            "reasoning_tokens must serialize as an integer, got {}",
            v["completion_tokens_details"]["reasoning_tokens"]
        );
        assert!(
            v["prompt_tokens_details"]["cached_tokens"].is_u64(),
            "cached_tokens must serialize as an integer, got {}",
            v["prompt_tokens_details"]["cached_tokens"]
        );
    }

    #[test]
    fn backend_reasoning_tokens_propagate_to_usage() {
        // A backend-reported reasoning_tokens / cached_tokens count must reach the
        // usage frame (the chat path previously dropped completion_tokens_details).
        let request = create_test_request();
        let mut generator = request.response_generator("req-usage-prop".to_string());
        let mut output = final_backend_output();
        output.completion_usage = Some(dynamo_protocols::types::CompletionUsage {
            prompt_tokens: 5,
            completion_tokens: 0,
            total_tokens: 5,
            prompt_tokens_details: Some(dynamo_protocols::types::PromptTokensDetails {
                cached_tokens: Some(3),
                ..Default::default()
            }),
            completion_tokens_details: Some(dynamo_protocols::types::CompletionTokensDetails {
                reasoning_tokens: Some(7),
                ..Default::default()
            }),
        });
        generator.choice_from_postprocessor(output).unwrap();
        let usage = generator.get_usage();

        assert_eq!(
            usage.completion_tokens_details.unwrap().reasoning_tokens,
            Some(7)
        );
        assert_eq!(usage.prompt_tokens_details.unwrap().cached_tokens, Some(3));
    }

    #[test]
    fn k3_marker_sequences_populate_reasoning_tokens_in_usage() {
        let request = create_test_request();
        let mut generator = request.response_generator("req-k3-usage".to_string());
        generator.set_reasoning_markers(Some(vec![100, 101, 102]), Some(vec![200, 101, 102]), true);
        let mut output = final_backend_output();
        output.token_ids = vec![1, 2, 200, 101, 102, 9];
        generator.choice_from_postprocessor(output).unwrap();

        let usage = generator.get_usage();
        assert_eq!(usage.completion_tokens, 6);
        assert_eq!(
            usage.completion_tokens_details.unwrap().reasoning_tokens,
            Some(2)
        );
    }

    #[test]
    fn prompt_token_adjustment_survives_backend_usage_override() {
        let request = create_test_request();
        let mut generator = request.response_generator("req-prompt-adjustment".to_string());

        generator.set_prompt_tokens_adjustment(3);
        generator.update_isl(39);
        assert_eq!(generator.get_usage().prompt_tokens, 36);

        // Workers can replace the frontend ISL with their own prompt count (for
        // example after multimodal placeholder expansion). The API-only
        // adjustment must still apply to that authoritative backend count.
        let mut output = final_backend_output();
        output.completion_usage = Some(dynamo_protocols::types::CompletionUsage {
            prompt_tokens: 122,
            completion_tokens: 0,
            total_tokens: 122,
            ..Default::default()
        });
        generator.choice_from_postprocessor(output).unwrap();

        let usage = generator.get_usage();
        assert_eq!(usage.prompt_tokens, 119);
        assert_eq!(
            usage.total_tokens,
            usage.prompt_tokens + usage.completion_tokens
        );
    }

    fn create_test_request_with_extra_fields(fields: Vec<String>) -> NvCreateChatCompletionRequest {
        let messages = vec![ChatCompletionRequestMessage::User(
            ChatCompletionRequestUserMessage {
                content: ChatCompletionRequestUserMessageContent::Text("test".to_string()),
                name: None,
            },
        )];

        NvCreateChatCompletionRequest {
            inner: CreateChatCompletionRequest {
                model: "test-model".to_string(),
                messages,
                stream: Some(true),
                stream_options: None,
                ..Default::default()
            },
            common: Default::default(),
            nvext: Some(
                crate::protocols::common::extensions::NvExt::builder()
                    .extra_fields(fields)
                    .build()
                    .unwrap(),
            ),
            chat_template_args: None,
            thinking: None,
            media_io_kwargs: None,
            return_tokens_as_token_ids: None,
            unsupported_fields: Default::default(),
        }
    }

    fn make_backend_output_with_engine_data() -> crate::protocols::common::llm_backend::BackendOutput
    {
        crate::protocols::common::llm_backend::BackendOutput {
            token_ids: vec![42],
            tokens: vec![Some("hello".to_string())],
            text: Some("hello".to_string()),
            cum_log_probs: None,
            log_probs: None,
            top_logprobs: None,
            finish_reason: Some(crate::protocols::common::FinishReason::Stop),
            stop_reason: None,
            index: Some(0),
            completion_usage: None,
            disaggregated_params: None,
            encoder_result: None,
            worker_trace_link: None,
            engine_data: Some(serde_json::json!({
                "kv_transfer_time_ms": 12.3,
                "disaggregated_kv_transfer_time_ms": 8.1,
                "prefill_compute_time_ms": 45.6
            })),
            routing_data: None,
        }
    }

    #[test]
    fn finish_frame_carries_per_candidate_usage() {
        // Streaming spec P0.4: the end frame gets a per-candidate usage
        // snapshot; increment frames get none.
        let request = create_test_request();
        let mut generator = request.response_generator("req-p04".to_string());

        let mut increment = final_backend_output();
        increment.finish_reason = None;
        increment.token_ids = vec![1, 2, 3];
        let response = generator
            .choice_from_postprocessor(increment)
            .expect("increment");
        assert!(response.choice_usage.is_none());

        let mut end = final_backend_output();
        end.token_ids = vec![4];
        let response = generator.choice_from_postprocessor(end).expect("finish");
        let usage = response.choice_usage.expect("end frame usage");
        assert_eq!(usage.completion_tokens, 4, "per-candidate completion tokens");
        assert!(
            usage.completion_tokens_details.is_some(),
            "P0.2 detail counters present on the per-candidate snapshot"
        );
    }

    #[test]
    fn test_plain_request_without_extra_fields_omits_nvext() {
        let request = create_test_request();
        let mut generator = request.response_generator("req-no-nvext".to_string());
        generator
            .tracker()
            .record_worker(42, Some(0), WORKER_TYPE_PREFILL);

        let response = generator
            .choice_from_postprocessor(final_backend_output())
            .expect("choice generation");

        assert!(response.nvext.is_none());
    }

    #[test]
    fn test_backend_choice_index_is_preserved() {
        let request = create_test_request();
        let mut generator = request.response_generator("req-choice-index".to_string());
        let mut output = final_backend_output();
        output.index = Some(2);

        let response = generator
            .choice_from_postprocessor(output)
            .expect("choice generation");

        assert_eq!(response.inner.choices[0].index, 2);
    }

    #[test]
    fn test_stop_reason_emits_in_nvext_when_requested() {
        let request = create_test_request_with_extra_fields(vec!["stop_reason".to_string()]);
        let mut generator = request.response_generator("req-stop-reason-nvext".to_string());
        let mut output = final_backend_output();
        output.stop_reason = Some(dynamo_protocols::types::StopReason::String(
            "END".to_string(),
        ));

        let response = generator
            .choice_from_postprocessor(output)
            .expect("choice generation");

        let response_json = serde_json::to_value(&response).expect("serialize response");
        assert!(response_json["choices"][0].get("stop_reason").is_none());
        assert_eq!(response_json["nvext"]["stop_reason"], "END");
    }

    #[test]
    fn test_timing_extra_field_emits_timing_on_final_chunk() {
        use crate::protocols::common::extensions::NvExt;
        let nvext = NvExt::builder()
            .extra_fields(vec!["timing".to_string()])
            .build()
            .unwrap();
        let mut generator =
            make_request_with_nvext(nvext).response_generator("req-timing".to_string());

        let response = generator
            .choice_from_postprocessor(final_backend_output())
            .expect("choice generation");

        let nvext_json = response.nvext.expect("nvext present for timing request");
        assert!(
            nvext_json.get("timing").is_some(),
            "timing should be emitted when extra_fields=[\"timing\"]"
        );
        assert!(nvext_json.get("worker_id").is_none());
        assert!(nvext_json.get("token_ids").is_none());
        assert!(nvext_json.get("routed_experts").is_none());
    }

    #[test]
    fn test_query_instance_id_emits_worker_id_and_token_ids() {
        use crate::protocols::common::extensions::NvExt;
        let nvext = NvExt::builder()
            .annotations(vec!["query_instance_id:abc".to_string()])
            .build()
            .unwrap();
        let mut generator =
            make_request_with_nvext(nvext).response_generator("req-qid".to_string());
        generator
            .tracker()
            .record_worker(42, Some(0), WORKER_TYPE_PREFILL);
        // The query-only tokenized prompt reaches the delta generator via the tracker,
        // mirroring the standalone-router round-trip the preprocessor drains.
        generator
            .tracker()
            .set_external_query_token_ids(vec![11, 22, 33]);

        let response = generator
            .choice_from_postprocessor(final_backend_output())
            .expect("choice generation");

        let nvext_json = response
            .nvext
            .expect("nvext present for query_instance_id flow");
        assert!(nvext_json.get("worker_id").is_some());
        assert_eq!(
            nvext_json.get("token_ids"),
            Some(&serde_json::json!([11, 22, 33]))
        );
        // timing is NOT auto-enabled for query_instance_id — it is gated by `extra_fields: ["timing"]`.
        assert!(nvext_json.get("timing").is_none());
        assert!(nvext_json.get("routed_experts").is_none());
    }

    #[test]
    fn test_routed_experts_extra_field_emits_routed_experts() {
        use crate::protocols::common::extensions::NvExt;
        let nvext = NvExt::builder()
            .extra_fields(vec!["routed_experts".to_string()])
            .build()
            .unwrap();
        let mut generator =
            make_request_with_nvext(nvext).response_generator("req-experts".to_string());

        let response = generator
            .choice_from_postprocessor(final_backend_output())
            .expect("choice generation");

        let nvext_json = response
            .nvext
            .expect("nvext present for routed_experts request");
        assert_eq!(
            nvext_json.get("routed_experts"),
            Some(&serde_json::json!({"layer_0": [1, 3]}))
        );
        assert!(nvext_json.get("worker_id").is_none());
        assert!(nvext_json.get("timing").is_none());
        assert!(nvext_json.get("token_ids").is_none());
    }

    #[test]
    fn test_engine_data_included_when_requested_via_extra_fields() {
        let request = create_test_request_with_extra_fields(vec!["engine_data".to_string()]);
        let mut generator = request.response_generator("req-engine-1".to_string());

        let backend_output = make_backend_output_with_engine_data();
        let response = generator
            .choice_from_postprocessor(backend_output)
            .expect("should produce a response");

        let nvext = response.nvext.expect("nvext should be present");
        let engine_data = nvext
            .get("engine_data")
            .expect("engine_data should be present");
        assert_eq!(engine_data["kv_transfer_time_ms"], 12.3);
        assert_eq!(engine_data["prefill_compute_time_ms"], 45.6);
    }

    #[test]
    fn test_engine_data_excluded_when_not_requested() {
        let request = create_test_request();
        let mut generator = request.response_generator("req-engine-2".to_string());

        let backend_output = make_backend_output_with_engine_data();
        let response = generator
            .choice_from_postprocessor(backend_output)
            .expect("should produce a response");

        // nvext may or may not be present (tracker may inject worker_id),
        // but engine_data specifically must be absent
        if let Some(nvext) = &response.nvext {
            assert!(
                nvext.get("engine_data").is_none() || nvext.get("engine_data").unwrap().is_null(),
                "engine_data should not be present when not requested"
            );
        }
    }

    #[test]
    fn test_engine_data_excluded_when_other_extra_fields_requested() {
        let request = create_test_request_with_extra_fields(vec!["timing".to_string()]);
        let mut generator = request.response_generator("req-engine-3".to_string());

        let backend_output = make_backend_output_with_engine_data();
        let response = generator
            .choice_from_postprocessor(backend_output)
            .expect("should produce a response");

        if let Some(nvext) = &response.nvext {
            assert!(
                nvext.get("engine_data").is_none() || nvext.get("engine_data").unwrap().is_null(),
                "engine_data should not be present when only timing is requested"
            );
        }
    }

    #[test]
    fn test_engine_data_none_from_backend_no_nvext_noise() {
        let request = create_test_request_with_extra_fields(vec!["engine_data".to_string()]);
        let mut generator = request.response_generator("req-engine-4".to_string());

        let backend_output = crate::protocols::common::llm_backend::BackendOutput {
            token_ids: vec![42],
            tokens: vec![Some("hello".to_string())],
            text: Some("hello".to_string()),
            cum_log_probs: None,
            log_probs: None,
            top_logprobs: None,
            finish_reason: Some(crate::protocols::common::FinishReason::Stop),
            stop_reason: None,
            index: Some(0),
            completion_usage: None,
            disaggregated_params: None,
            encoder_result: None,
            worker_trace_link: None,
            engine_data: None, // engine didn't provide any data
            routing_data: None,
        };

        let response = generator
            .choice_from_postprocessor(backend_output)
            .expect("should produce a response");

        // engine_data is None from backend, so nvext.engine_data should be absent
        if let Some(nvext) = &response.nvext {
            assert!(
                nvext.get("engine_data").is_none() || nvext.get("engine_data").unwrap().is_null(),
                "engine_data should not appear when backend provides None"
            );
        }
    }
}
