// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The `disconnect` module provides a mechanism for our axum http services to monitoring and responding
//! to disconnects from the client.
//!
//! There are two potential phases in any request where we need to handle the disconnect.
//!
//! For unary, request-response, there is just a single phase where the primary task that axum kicks off
//! to handle the request will be dropped if the client disconnects. In order for us to have a long running
//! task, like an LLM request, we need to spawn our long running task in a separate task and then spawn
//! a second task that will monitor for disconnects from the client. The primary task which spawned the
//! two tasks will hold an "armed" [`ConnectionHandle`] which will issue a [`ConnectionStatus::ClosedUnexpectedly`]
//! if the task is dropped before it is [`ConnectionHandle::disarm`]ed.
//!
//! For the streaming case, request in - stream out, we need a second [`ConnectionHandle`] which will be owned
//! by the stream. A streaming response is when the [`axum::response::Response]] is a [axum::response::Sse] stream.
//! This means the primary task handle will go out of scope when it returns the stream. When we create our
//! SSE stream, we capture the second [`ConnectionHandle`] and arm it. If the stream closes gracefully, the
//! second handle will be disarmed, otherwise, the stream was dropped and the [`Drop`] trait on the [`ConnectionHandle`]
//! triggers a [`ConnectionStatus::ClosedUnexpectedly`] signal.
//!
//! The [`ConnectionHandle`] is a simple wrapper around a [`tokio::sync::oneshot::Sender`] which will send a
//! [`ConnectionStatus`] enum to the primary task. The primary task will then use this to determine if it should
//! cancel the request or not.
//!
//! The [`ConnectionHandle`] is also used to signal to the client that the request has been cancelled. This is
//! done by sending a [`axum::response::sse::Event`] with the event type "error" and the data "[DONE]".
//!

use axum::response::sse::Event;
use dynamo_runtime::engine::AsyncEngineContext;
use futures::{Stream, StreamExt};
use std::sync::Arc;
use std::time::Duration;

use crate::http::service::error::SanitizedError;
use crate::http::service::metrics::{CancellationLabels, ErrorType, InflightGuard, Metrics};
use crate::protocols::common::FinishReason;
use dynamo_runtime::error::{DynamoError, ErrorType as DynamoErrorType};

use dynamo_runtime::config::environment_names::llm::{
    DYN_HTTP_BACKEND_STREAM_TIMEOUT_SECS as BACKEND_STREAM_TIMEOUT_ENV,
    DYN_HTTP_BACKEND_TTFT_STREAM_TIMEOUT_SECS as BACKEND_TTFT_TIMEOUT_ENV,
    DYN_HTTP_SSE_INACTIVITY_TIMEOUT_SECS as SSE_INACTIVITY_TIMEOUT_ENV,
};

fn read_positive_secs(env_var: &str) -> Option<Duration> {
    std::env::var(env_var)
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .filter(|&secs| secs > 0)
        .map(Duration::from_secs)
}

/// Read the backend stream inactivity timeout from the environment.
/// Returns `None` if unset or zero (timeout disabled).
///
/// The HTTP-layer timeout uses a 2x multiplier over the configured value so that
/// the request-plane timeout in `push_router` (which uses the raw value) always
/// fires first and triggers `report_instance_down()` for worker quarantine.
/// This layer is strictly a safety net for gauge cleanup.
pub fn backend_stream_timeout() -> Option<Duration> {
    if let Some(timeout) = read_positive_secs(SSE_INACTIVITY_TIMEOUT_ENV) {
        return Some(timeout);
    }
    read_positive_secs(BACKEND_STREAM_TIMEOUT_ENV)
        .map(|d| Duration::from_secs(d.as_secs().saturating_mul(2)))
}

/// HTTP-layer TTFT safety-net timeout. The explicit SSE override wins;
/// otherwise this is twice the request-plane TTFT timeout.
pub fn backend_ttft_timeout() -> Option<Duration> {
    if let Some(timeout) = read_positive_secs(SSE_INACTIVITY_TIMEOUT_ENV) {
        return Some(timeout);
    }
    read_positive_secs(BACKEND_TTFT_TIMEOUT_ENV)
        .map(|d| Duration::from_secs(d.as_secs().saturating_mul(2)))
}

/// Reject a safety-net override that would fire before either request-plane
/// timer and therefore bypass worker quarantine.
pub fn validate_backend_stream_timeouts() -> anyhow::Result<()> {
    let Some(safety) = read_positive_secs(SSE_INACTIVITY_TIMEOUT_ENV) else {
        return Ok(());
    };

    for (primary_env, phase) in [
        (BACKEND_STREAM_TIMEOUT_ENV, "inter-token"),
        (BACKEND_TTFT_TIMEOUT_ENV, "TTFT"),
    ] {
        if let Some(primary) = read_positive_secs(primary_env)
            && safety <= primary
        {
            anyhow::bail!(
                "{SSE_INACTIVITY_TIMEOUT_ENV}={}s must be > {primary_env}={}s \
                 (HTTP safety net would beat the request-plane {phase} timer)",
                safety.as_secs(),
                primary.as_secs(),
            );
        }
    }
    Ok(())
}

fn downcast_dynamo_error(err: &axum::Error) -> Option<&DynamoError> {
    let mut source: Option<&(dyn std::error::Error + 'static)> = Some(err);
    while let Some(error) = source {
        if let Some(dynamo_error) = error.downcast_ref::<DynamoError>() {
            return Some(dynamo_error);
        }
        source = error.source();
    }
    None
}

/// Convert a typed request-plane timeout into its metrics and protocol labels.
pub(super) fn timeout_classification(
    error: &DynamoError,
) -> Option<(ErrorType, &'static str, String)> {
    match error.error_type() {
        DynamoErrorType::FirstTokenTimeout => Some((
            ErrorType::FirstTokenTimeout,
            "first_token_timeout",
            error.message().to_string(),
        )),
        DynamoErrorType::IntraTokenTimeout => Some((
            ErrorType::IntraTokenTimeout,
            "intra_token_timeout",
            error.message().to_string(),
        )),
        DynamoErrorType::ResponseTimeout => Some((
            ErrorType::ResponseTimeout,
            "timeout",
            error.message().to_string(),
        )),
        _ => None,
    }
}

#[derive(Clone, Copy)]
pub enum ConnectionStatus {
    Disabled,
    ClosedUnexpectedly,
    ClosedGracefully,
}

pub struct ConnectionHandle {
    sender: Option<tokio::sync::oneshot::Sender<ConnectionStatus>>,
    on_drop: ConnectionStatus,
}

impl ConnectionHandle {
    /// Handle which by default will issue a [`ConnectionStatus::ClosedGracefully`] signal when dropped.
    pub fn create_disarmed(sender: tokio::sync::oneshot::Sender<ConnectionStatus>) -> Self {
        Self {
            sender: Some(sender),
            on_drop: ConnectionStatus::ClosedGracefully,
        }
    }

    /// Handle which will issue a [`ConnectionStatus::ClosedUnexpectedly`] signal when dropped.
    pub fn create_armed(sender: tokio::sync::oneshot::Sender<ConnectionStatus>) -> Self {
        Self {
            sender: Some(sender),
            on_drop: ConnectionStatus::ClosedUnexpectedly,
        }
    }

    /// Handle which will not issue a signal when dropped.
    pub fn create_disabled(sender: tokio::sync::oneshot::Sender<ConnectionStatus>) -> Self {
        Self {
            sender: Some(sender),
            on_drop: ConnectionStatus::Disabled,
        }
    }

    /// Handle which will issue a [`ConnectionStatus::ClosedGracefully`] signal when dropped.
    pub fn disarm(&mut self) {
        self.on_drop = ConnectionStatus::ClosedGracefully;
    }

    /// Handle which will issue a [`ConnectionStatus::ClosedUnexpectedly`] signal when dropped.
    pub fn arm(&mut self) {
        self.on_drop = ConnectionStatus::ClosedUnexpectedly;
    }
}

impl Drop for ConnectionHandle {
    fn drop(&mut self) {
        if let Some(sender) = self.sender.take() {
            let _ = sender.send(self.on_drop);
        }
    }
}

/// Creates a pair of handles which will monitor for disconnects from the client.
///
/// The first handle is armed and will issue a [`ConnectionStatus::ClosedUnexpectedly`] signal when dropped.
/// The second handle is disarmed and will issue a [`ConnectionStatus::ClosedGracefully`] signal when dropped.
///
/// The handles are returned in the order of the first being armed and the second being disarmed.
pub async fn create_connection_monitor(
    engine_context: Arc<dyn AsyncEngineContext>,
    metrics: Option<Arc<Metrics>>,
    cancellation_labels: CancellationLabels,
) -> (ConnectionHandle, ConnectionHandle) {
    // these oneshot channels monitor possible disconnects from the client in two different scopes:
    // - the local task (connection_handle)
    // - an optionally streaming response (stream_handle)
    let (connection_tx, connection_rx) = tokio::sync::oneshot::channel();
    let (stream_tx, stream_rx) = tokio::sync::oneshot::channel();

    // detached task that will naturally close when both handles are dropped
    tokio::spawn(connection_monitor(
        engine_context.clone(),
        connection_rx,
        stream_rx,
        metrics,
        cancellation_labels,
    ));

    // Two handles, the first is armed, the second is disarmed
    (
        ConnectionHandle::create_armed(connection_tx),
        ConnectionHandle::create_disabled(stream_tx),
    )
}

#[tracing::instrument(level = "trace", skip_all, fields(request_id = %engine_context.id()))]
async fn connection_monitor(
    engine_context: Arc<dyn AsyncEngineContext>,
    connection_rx: tokio::sync::oneshot::Receiver<ConnectionStatus>,
    stream_rx: tokio::sync::oneshot::Receiver<ConnectionStatus>,
    metrics: Option<Arc<Metrics>>,
    cancellation_labels: CancellationLabels,
) {
    match connection_rx.await {
        Err(_) | Ok(ConnectionStatus::ClosedUnexpectedly) => {
            // the client has disconnected, no need to gracefully cancel, just kill the context
            tracing::warn!("Connection closed unexpectedly; issuing cancellation");
            if let Some(metrics) = &metrics {
                metrics.inc_client_disconnect();
                metrics.inc_cancellation(&cancellation_labels);
            }
            engine_context.kill();
        }
        Ok(ConnectionStatus::ClosedGracefully) => {
            tracing::trace!("Connection closed gracefully");
        }
        Ok(ConnectionStatus::Disabled) => {}
    }

    match stream_rx.await {
        Err(_) | Ok(ConnectionStatus::ClosedUnexpectedly) => {
            tracing::warn!("Stream closed unexpectedly; issuing cancellation");
            if let Some(metrics) = &metrics {
                metrics.inc_client_disconnect();
                metrics.inc_cancellation(&cancellation_labels);
            }
            engine_context.kill();
        }
        Ok(ConnectionStatus::ClosedGracefully) => {
            tracing::trace!("Stream closed gracefully");
        }
        Ok(ConnectionStatus::Disabled) => {}
    }
}

/// This method will consume a stream of SSE events and monitor for disconnects or context cancellation.
///
/// Uses `tokio::select!` to choose between receiving events from the source stream or detecting when
/// the context is stopped. If the context is stopped, we break the stream. If the source stream ends
/// naturally, we mark the request as successful and send the final `[DONE]` event.
///
/// A configurable inactivity timeout (see [`BACKEND_STREAM_TIMEOUT_ENV`]) adds a third arm: if no
/// SSE event is received from the backend within the timeout window, the engine context is killed and
/// the inflight guard is dropped, preventing permanent gauge inflation caused by zombie workers that
/// hold a live TCP connection but produce no output.
pub fn monitor_for_disconnects(
    stream: impl Stream<Item = Result<Event, axum::Error>>,
    context: Arc<dyn AsyncEngineContext>,
    inflight_guard: InflightGuard,
    stream_handle: ConnectionHandle,
) -> impl Stream<Item = Result<Event, axum::Error>> {
    let (_finish_reason_tx, finish_reason_rx) =
        tokio::sync::watch::channel(Some(FinishReason::Stop));
    monitor_for_disconnects_inner(
        stream,
        context,
        inflight_guard,
        stream_handle,
        finish_reason_rx,
        None,
        backend_stream_timeout(),
        backend_ttft_timeout(),
    )
}

/// Monitor an LLM stream and classify a clean close from its terminal finish
/// reason. `None` means the stream was truncated; `Error` is a backend error.
pub fn monitor_for_disconnects_with_finish_reason(
    stream: impl Stream<Item = Result<Event, axum::Error>>,
    context: Arc<dyn AsyncEngineContext>,
    inflight_guard: InflightGuard,
    stream_handle: ConnectionHandle,
    finish_reason_rx: tokio::sync::watch::Receiver<Option<FinishReason>>,
) -> impl Stream<Item = Result<Event, axum::Error>> {
    monitor_for_disconnects_inner(
        stream,
        context,
        inflight_guard,
        stream_handle,
        finish_reason_rx,
        None,
        backend_stream_timeout(),
        backend_ttft_timeout(),
    )
}

/// Token-aware LLM monitor. The TTFT window remains active across leading
/// annotation or handshake events until the producer sets `first_token_seen`.
pub fn monitor_for_disconnects_tracking_first_token(
    stream: impl Stream<Item = Result<Event, axum::Error>>,
    context: Arc<dyn AsyncEngineContext>,
    inflight_guard: InflightGuard,
    stream_handle: ConnectionHandle,
    finish_reason_rx: tokio::sync::watch::Receiver<Option<FinishReason>>,
    first_token_seen: Arc<std::sync::atomic::AtomicBool>,
) -> impl Stream<Item = Result<Event, axum::Error>> {
    monitor_for_disconnects_inner(
        stream,
        context,
        inflight_guard,
        stream_handle,
        finish_reason_rx,
        Some(first_token_seen),
        backend_stream_timeout(),
        backend_ttft_timeout(),
    )
}

#[cfg(test)]
fn monitor_for_disconnects_with_timeout(
    stream: impl Stream<Item = Result<Event, axum::Error>>,
    context: Arc<dyn AsyncEngineContext>,
    inflight_guard: InflightGuard,
    stream_handle: ConnectionHandle,
    inactivity_timeout: Option<Duration>,
) -> impl Stream<Item = Result<Event, axum::Error>> {
    let (_finish_reason_tx, finish_reason_rx) =
        tokio::sync::watch::channel(Some(FinishReason::Stop));
    monitor_for_disconnects_inner(
        stream,
        context,
        inflight_guard,
        stream_handle,
        finish_reason_rx,
        None,
        inactivity_timeout,
        inactivity_timeout,
    )
}

#[allow(clippy::too_many_arguments)]
fn monitor_for_disconnects_inner(
    stream: impl Stream<Item = Result<Event, axum::Error>>,
    context: Arc<dyn AsyncEngineContext>,
    mut inflight_guard: InflightGuard,
    mut stream_handle: ConnectionHandle,
    finish_reason_rx: tokio::sync::watch::Receiver<Option<FinishReason>>,
    first_token_seen: Option<Arc<std::sync::atomic::AtomicBool>>,
    stream_timeout: Option<Duration>,
    ttft_timeout: Option<Duration>,
) -> impl Stream<Item = Result<Event, axum::Error>> {
    stream_handle.arm();

    // Default to Cancelled: if the stream is dropped unexpectedly (e.g. client
    // disconnect causing a broken-pipe on the SSE write), the guard will report
    // "cancelled" instead of "internal". The happy path overrides this via mark_ok().
    inflight_guard.mark_error(ErrorType::Cancelled);

    let pre_first_timeout = ttft_timeout.or(stream_timeout);

    async_stream::try_stream! {
        tokio::pin!(stream);
        let mut first_received = false;
        // Keep the context's watch-backed cancellation future alive across body frames.
        // Recreating it for every token repeatedly clones a receiver and churns Notify state.
        let stopped = context.stopped();
        tokio::pin!(stopped);
        loop {
            let content_seen = match &first_token_seen {
                Some(flag) => flag.load(std::sync::atomic::Ordering::Acquire),
                None => first_received,
            };
            let inactivity_timeout = if content_seen {
                stream_timeout
            } else {
                pre_first_timeout
            };
            tokio::select! {
                // Prefer a response that became ready in the same poll as the timer.
                biased;
                event = stream.next() => {
                    match event {
                        Some(Ok(event)) => {
                            first_received = true;
                            yield event;
                        }
                        Some(Err(err)) => {
                            let typed_timeout =
                                downcast_dynamo_error(&err).and_then(timeout_classification);
                            let backend_error = matches!(
                                &*finish_reason_rx.borrow(),
                                Some(FinishReason::Error(_))
                            );
                            let (error_type, envelope_type, envelope_code, envelope_message) =
                                if let Some((metric, error_type, message)) = typed_timeout {
                                    (metric, error_type, 504u16, message)
                                } else {
                                    let metric = if backend_error {
                                        ErrorType::BackendError
                                    } else {
                                        ErrorType::Internal
                                    };
                                    tracing::error!(details = %err, "Streaming error");
                                    let sanitized = SanitizedError::Internal;
                                    (
                                        metric,
                                        sanitized.openai_type_slug(),
                                        sanitized.status().as_u16(),
                                        sanitized.to_string(),
                                    )
                                };
                            inflight_guard.mark_error(error_type);
                            // We're terminating the stream intentionally here with a
                            // structured error + [DONE]; disarm so the stream handle
                            // doesn't later record this as ClosedUnexpectedly (which
                            // would mis-attribute the fault as a client disconnect).
                            stream_handle.disarm();
                            // A merged batch stream uses one parent context linked to
                            // every prompt. Terminating on any substream error must fan
                            // cancellation out to the still-running siblings.
                            context.kill();
                            let err_json = serde_json::json!({
                                "error": {
                                    "message": envelope_message,
                                    "type": envelope_type,
                                    "code": envelope_code,
                                }
                            });
                            yield Event::default().data(err_json.to_string());
                            yield Event::default().data("[DONE]");
                            // Break to prevent any subsequent mark_ok() from overwriting the error
                            break;
                        }
                        None => {
                            match &*finish_reason_rx.borrow() {
                                Some(FinishReason::Error(_)) => {
                                    inflight_guard.mark_error(ErrorType::BackendError);
                                }
                                Some(_) => inflight_guard.mark_ok(),
                                None => inflight_guard.mark_error(ErrorType::TruncatedStream),
                            }
                            stream_handle.disarm();

                            // todo: if we yield a dynamo sentinel event, we need to do it before the done or the
                            // async-openai client will chomp it.
                            yield Event::default().data("[DONE]");
                            break;
                        }
                    }
                }
                _ = &mut stopped => {
                    // Mark as cancelled when context is stopped (client disconnect or timeout)
                    inflight_guard.mark_error(ErrorType::Cancelled);
                    // Token counts (input_tokens, output_tokens) are recorded on
                    // the enclosing span by ResponseMetricCollector::Drop.
                    tracing::warn!(
                        request_id = %inflight_guard.request_id(),
                        model = %inflight_guard.model(),
                        endpoint = %inflight_guard.endpoint(),
                        request_type = %inflight_guard.request_type(),
                        error_type = "cancelled",
                        elapsed_ms = %inflight_guard.elapsed_ms(),
                        "request cancelled"
                    );
                    break;
                }
                // Circuit breaker for zombie backend workers: if the backend holds a live TCP
                // connection but produces no output for `inactivity_timeout`, kill the engine
                // context so that InflightGuard::drop() fires and dec() corrects the gauge.
                // The sleep is re-created each iteration so it acts as an *inactivity* timeout
                // (resets whenever a token is received), not a hard total-request deadline.
                // When inactivity_timeout is None the pending() future never resolves.
                _ = async {
                    match inactivity_timeout {
                        Some(d) => tokio::time::sleep(d).await,
                        None => std::future::pending::<()>().await,
                    }
                } => {
                    inflight_guard.mark_error(ErrorType::ResponseTimeout);
                    stream_handle.disarm();
                    let phase = if content_seen { "stream" } else { "ttft" };
                    tracing::warn!(
                        request_id = %inflight_guard.request_id(),
                        model = %inflight_guard.model(),
                        endpoint = %inflight_guard.endpoint(),
                        request_type = %inflight_guard.request_type(),
                        phase,
                        error_type = "response_timeout",
                        elapsed_ms = %inflight_guard.elapsed_ms(),
                        timeout_secs = ?inactivity_timeout.map(|d| d.as_secs()),
                        "backend stream inactivity timeout; killing engine context to release inflight gauge"
                    );
                    context.kill();
                    let err_json = serde_json::json!({
                        "error": {
                            "message": "backend stream inactivity timeout",
                            "type": "timeout",
                            "code": 504,
                        }
                    });
                    yield Event::default().data(err_json.to_string());
                    yield Event::default().data("[DONE]");
                    break;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::http::service::metrics::{Endpoint, ErrorType, RequestType, Status};
    use futures::StreamExt;
    use serial_test::serial;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Debug, Default)]
    struct MockContext {
        stopped_polls: AtomicUsize,
        killed: std::sync::atomic::AtomicBool,
        track_kill: bool,
    }

    impl MockContext {
        fn new() -> Self {
            Self::default()
        }

        fn with_kill_tracking() -> Self {
            Self {
                track_kill: true,
                ..Default::default()
            }
        }
    }

    #[async_trait::async_trait]
    impl dynamo_runtime::engine::AsyncEngineContext for MockContext {
        fn id(&self) -> &str {
            "test"
        }
        fn stop(&self) {}
        fn stop_generating(&self) {}
        fn kill(&self) {
            if self.track_kill {
                self.killed.store(true, std::sync::atomic::Ordering::SeqCst);
            }
        }
        fn is_stopped(&self) -> bool {
            false
        }
        fn is_killed(&self) -> bool {
            self.track_kill && self.killed.load(std::sync::atomic::Ordering::SeqCst)
        }
        async fn stopped(&self) {
            self.stopped_polls.fetch_add(1, Ordering::Relaxed);
            std::future::pending::<()>().await;
        }
        async fn killed(&self) {
            std::future::pending::<()>().await;
        }
        fn link_child(&self, _: Arc<dyn dynamo_runtime::engine::AsyncEngineContext>) {}
    }

    fn hanging_stream()
    -> impl futures::Stream<Item = Result<axum::response::sse::Event, axum::Error>> {
        async_stream::try_stream! {
            std::future::pending::<()>().await;
            yield axum::response::sse::Event::default().data("unreachable");
        }
    }

    fn timed_token_stream(
        count: usize,
        interval: Duration,
    ) -> impl futures::Stream<Item = Result<axum::response::sse::Event, axum::Error>> {
        async_stream::try_stream! {
            for i in 0..count {
                tokio::time::sleep(interval).await;
                yield axum::response::sse::Event::default().data(format!("token-{i}"));
            }
        }
    }

    fn setup_test(
        model: &str,
        req_id: &str,
    ) -> (
        Arc<Metrics>,
        InflightGuard,
        Arc<dyn AsyncEngineContext>,
        ConnectionHandle,
    ) {
        let metrics = Arc::new(Metrics::new());
        let guard =
            metrics
                .clone()
                .create_inflight_guard(model, Endpoint::ChatCompletions, true, req_id);
        let context: Arc<dyn AsyncEngineContext> = Arc::new(MockContext::new());
        let (tx, _rx) = tokio::sync::oneshot::channel();
        let handle = ConnectionHandle::create_disabled(tx);
        (metrics, guard, context, handle)
    }

    #[tokio::test]
    async fn test_monitor_reuses_stopped_future_across_events() {
        let model = "reuse-stopped-future";
        let metrics = Arc::new(Metrics::new());
        let guard = metrics.clone().create_inflight_guard(
            model,
            Endpoint::ChatCompletions,
            true,
            "req-reuse",
        );
        let context = Arc::new(MockContext::new());
        let engine_context: Arc<dyn AsyncEngineContext> = context.clone();
        let (tx, _rx) = tokio::sync::oneshot::channel();
        let handle = ConnectionHandle::create_disabled(tx);
        let stream = futures::stream::unfold(0, |index| async move {
            tokio::task::yield_now().await;
            (index < 4).then(|| {
                (
                    Ok(Event::default().data(format!("token-{index}"))),
                    index + 1,
                )
            })
        });

        let monitored =
            monitor_for_disconnects_with_timeout(stream, engine_context, guard, handle, None);
        tokio::pin!(monitored);
        while monitored.next().await.is_some() {}

        assert_eq!(
            context.stopped_polls.load(Ordering::Relaxed),
            1,
            "the same stopped future should remain pending across all response events"
        );
    }

    fn generate_cancellation_labels() -> CancellationLabels {
        CancellationLabels {
            model: "test-model".to_string(),
            endpoint: Endpoint::Generate.to_string(),
            request_type: "unary".to_string(),
        }
    }

    async fn wait_for_kill(context: &Arc<MockContext>) {
        for _ in 0..100 {
            if context.is_killed() {
                return;
            }
            tokio::task::yield_now().await;
        }
    }

    #[tokio::test]
    async fn armed_handle_drop_kills_generate_context() {
        let context = Arc::new(MockContext::with_kill_tracking());
        let engine_context: Arc<dyn AsyncEngineContext> = context.clone();
        let (connection_handle, stream_handle) =
            create_connection_monitor(engine_context, None, generate_cancellation_labels()).await;

        drop(connection_handle);
        drop(stream_handle);

        wait_for_kill(&context).await;
        assert!(context.is_killed());
    }

    #[tokio::test]
    async fn disarmed_handle_does_not_kill_generate_context() {
        let context = Arc::new(MockContext::with_kill_tracking());
        let engine_context: Arc<dyn AsyncEngineContext> = context.clone();
        let (mut connection_handle, stream_handle) =
            create_connection_monitor(engine_context, None, generate_cancellation_labels()).await;

        connection_handle.disarm();
        drop(connection_handle);
        drop(stream_handle);

        for _ in 0..100 {
            tokio::task::yield_now().await;
        }
        assert!(!context.is_killed());
    }

    /// Zombie backend with hanging stream is terminated by inactivity timeout.
    #[tokio::test(start_paused = true)]
    async fn test_backend_inactivity_timeout_releases_inflight_gauge() {
        let model = "zombie-model";
        // Config value "1" → HTTP-layer timeout is 2s (2x safety-net multiplier)
        let (metrics, guard, context, handle) = setup_test(model, "req-zombie");
        assert_eq!(metrics.get_inflight_count(model), 1);

        let monitored = monitor_for_disconnects_with_timeout(
            hanging_stream(),
            context,
            guard,
            handle,
            Some(Duration::from_secs(2)),
        );
        tokio::pin!(monitored);

        tokio::time::advance(Duration::from_secs(3)).await;

        let completed = tokio::time::timeout(Duration::from_secs(2), async move {
            while monitored.next().await.is_some() {}
        })
        .await;

        completed.expect("stream did not terminate — backend inactivity timeout is broken");
        assert_eq!(
            metrics.get_inflight_count(model),
            0,
            "inflight gauge leaked"
        );

        // Verify the error was categorized as ResponseTimeout, not Cancelled
        assert_eq!(
            metrics.get_request_counter(
                model,
                &Endpoint::ChatCompletions,
                &RequestType::Stream,
                &Status::Error,
                &ErrorType::ResponseTimeout,
            ),
            1,
            "inactivity timeout should be recorded as ResponseTimeout"
        );
        assert_eq!(
            metrics.get_request_counter(
                model,
                &Endpoint::ChatCompletions,
                &RequestType::Stream,
                &Status::Error,
                &ErrorType::Cancelled,
            ),
            0,
            "inactivity timeout should NOT be recorded as Cancelled"
        );
    }

    /// Inactivity timeout resets on each token; only fires after a true gap.
    #[tokio::test(start_paused = true)]
    async fn test_inactivity_timeout_resets_on_each_token() {
        let model = "reset-model";

        // Phase 1: tokens arrive every 2s with a 5s config (10s HTTP timeout after 2x multiplier)
        // — stream completes normally because each token resets the timer.
        let (metrics, guard_1, ctx_1, handle_1) = setup_test(model, "phase1");
        assert_eq!(metrics.get_inflight_count(model), 1);

        let token_count = 5;
        let monitored_1 = monitor_for_disconnects_with_timeout(
            timed_token_stream(token_count, Duration::from_secs(2)),
            ctx_1,
            guard_1,
            handle_1,
            Some(Duration::from_secs(10)),
        );
        tokio::pin!(monitored_1);

        let mut received = Vec::new();
        let phase1 = tokio::time::timeout(Duration::from_secs(30), async {
            while let Some(event) = monitored_1.next().await {
                received.push(event);
            }
        })
        .await;

        assert!(
            phase1.is_ok(),
            "inactivity timeout incorrectly fired as a hard deadline"
        );
        assert_eq!(received.len(), token_count + 1); // tokens + [DONE]
        assert_eq!(metrics.get_inflight_count(model), 0);

        // Phase 2: hanging stream — timeout DOES fire.
        let guard_2 =
            metrics
                .clone()
                .create_inflight_guard(model, Endpoint::ChatCompletions, true, "phase2");
        assert_eq!(metrics.get_inflight_count(model), 1);

        let ctx_2: Arc<dyn AsyncEngineContext> = Arc::new(MockContext::new());
        let (tx_2, _rx_2) = tokio::sync::oneshot::channel();
        let handle_2 = ConnectionHandle::create_disabled(tx_2);

        let monitored_2 = monitor_for_disconnects_with_timeout(
            hanging_stream(),
            ctx_2,
            guard_2,
            handle_2,
            Some(Duration::from_secs(10)),
        );
        tokio::pin!(monitored_2);

        // Config "5" → HTTP timeout 10s (2x multiplier). Advance past it.
        tokio::time::advance(Duration::from_secs(11)).await;

        let phase2 = tokio::time::timeout(Duration::from_secs(10), async {
            while monitored_2.next().await.is_some() {}
        })
        .await;

        assert!(
            phase2.is_ok(),
            "hanging stream was not terminated by inactivity timeout"
        );
        assert_eq!(
            metrics.get_inflight_count(model),
            0,
            "inflight gauge leaked in phase 2"
        );
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // mid-stream fault SSE contract
    //
    // When the upstream stream yields `Err(_)` mid-stream — e.g. an upstream
    // worker dies and the mpsc channel reports
    // `Disconnected: Stream ended before generation completed`, or the Python
    // chat-processor raises and the Rust→Python `tx.send()` fails with
    // `Failed to send response: SendError { .. }` — the client MUST receive:
    //   1. a structured `data: {"error":{"message":..., "type":... or "code":...}}` frame, then
    //   2. a `data: [DONE]` terminator.
    // Before the fix, the code emitted the bare SSE trailer
    // `event: error\n: <comment>\n\n` with no `[DONE]`, which violates the
    // OpenAI SSE contract and is silently skipped by naive `data:`-line parsers.
    // The two tests below pin the post-fix contract.
    // ─────────────────────────────────────────────────────────────────────────────

    /// Builds a stream that yields `data_chunks` successful events, then yields an
    /// `Err` carrying `err_msg`, simulating a mid-stream upstream fault.
    fn simulate_mid_stream_error(
        data_chunks: usize,
        err_msg: &'static str,
    ) -> impl futures::Stream<Item = Result<axum::response::sse::Event, axum::Error>> {
        async_stream::try_stream! {
            for i in 0..data_chunks {
                yield axum::response::sse::Event::default().data(format!("chunk-{i}"));
            }
            Err(axum::Error::new(err_msg))?;
        }
    }

    /// Collect the wire-format SSE body from a monitored stream.
    async fn collect_sse_body(
        stream: impl Stream<Item = Result<Event, axum::Error>> + Send + 'static,
    ) -> String {
        use axum::body::to_bytes;
        use axum::response::{IntoResponse, Sse};
        let response = Sse::new(stream).into_response();
        let body = to_bytes(response.into_body(), 1 << 20)
            .await
            .expect("body bytes");
        String::from_utf8(body.to_vec()).expect("utf8 body")
    }

    /// Assert the post-fix SSE fault contract: a parsed structured error frame
    /// carrying the sanitized static message/type/code, positioned before
    /// `[DONE]`, with no bare `event: error` trailer, and crucially with no trace
    /// of `leaked_detail` (the raw backend error) anywhere in the body.
    fn assert_fault_contract(case: &str, text: &str, leaked_detail: &str) {
        let done_pos = text.find("data: [DONE]").unwrap_or_else(|| {
            panic!("[{case}] body does not terminate with `data: [DONE]`. Body:\n{text}")
        });

        let (error_line, error_frame) = text
            .lines()
            .find_map(|line| {
                let payload = line.strip_prefix("data: ")?;
                serde_json::from_str::<serde_json::Value>(payload)
                    .ok()
                    .filter(|v| v.get("error").is_some())
                    .map(|v| (line, v))
            })
            .unwrap_or_else(|| {
                panic!(
                    "[{case}] body missing structured JSON `data: {{\"error\":{{...}}}}` frame. Body:\n{text}"
                )
            });

        let error_pos = text.find(error_line).unwrap_or_default();
        assert!(
            error_pos < done_pos,
            "[{case}] structured error frame must precede `data: [DONE]`. Body:\n{text}"
        );

        let error = error_frame
            .get("error")
            .and_then(|v| v.as_object())
            .unwrap_or_else(|| panic!("[{case}] `error` field is not an object. Body:\n{text}"));
        let expected = SanitizedError::Internal;
        let expected_message = expected.to_string();
        assert_eq!(
            error.get("message").and_then(|v| v.as_str()),
            Some(expected_message.as_str()),
            "[{case}] structured error `message` must be the sanitized static string. Body:\n{text}"
        );
        assert_eq!(
            error.get("type").and_then(|v| v.as_str()),
            Some(expected.openai_type_slug()),
            "[{case}] structured error `type` mismatch. Body:\n{text}"
        );
        assert_eq!(
            error.get("code").and_then(|v| v.as_i64()),
            Some(i64::from(expected.status().as_u16())),
            "[{case}] structured error `code` mismatch. Body:\n{text}"
        );
        assert!(
            !text.contains("event: error\n: "),
            "[{case}] body contains bare `event: error\\n: <comment>` trailer (pre-fix bug). Body:\n{text}"
        );
        assert!(
            !text.contains(leaked_detail),
            "[{case}] SSE body leaked raw backend error detail to the client. \
             Expected `{leaked_detail}` to be absent. Body:\n{text}"
        );
    }

    /// Upstream worker killed mid-stream → mpsc channel reports `Disconnected` to the
    /// HTTP layer. Client MUST receive structured error + `[DONE]`.
    #[tokio::test]
    async fn test_simulate_worker_kill_emits_structured_error_and_done() {
        let (_metrics, guard, ctx, handle) = setup_test("worker-kill-model", "req-wk");
        let backend_detail = "Disconnected: Stream ended before generation completed";
        let stream = simulate_mid_stream_error(3, backend_detail);
        let monitored = monitor_for_disconnects_with_timeout(stream, ctx, guard, handle, None);
        let body = collect_sse_body(monitored).await;
        assert_fault_contract("worker_kill", &body, backend_detail);
    }

    /// Python chat-processor raises mid-stream → Rust→Python `tx.send()` fails with
    /// `SendError`. Client MUST receive structured error + `[DONE]`.
    #[tokio::test]
    async fn test_simulate_python_consumer_drop_emits_structured_error_and_done() {
        let (_metrics, guard, ctx, handle) = setup_test("py-drop-model", "req-py");
        let backend_detail = "Failed to send response: SendError { .. }";
        let stream = simulate_mid_stream_error(3, backend_detail);
        let monitored = monitor_for_disconnects_with_timeout(stream, ctx, guard, handle, None);
        let body = collect_sse_body(monitored).await;
        assert_fault_contract("python_consumer_drop", &body, backend_detail);
    }

    /// A backend error carrying sensitive internals (file paths, panic text,
    /// Python exception details) MUST NOT reach the streaming client. The client
    /// receives only the sanitized static frame; the detail stays server-side.
    #[tokio::test]
    async fn test_mid_stream_error_does_not_leak_internal_details() {
        let (_metrics, guard, ctx, handle) = setup_test("leak-model", "req-leak");
        let backend_detail = "panicked at '/opt/dynamo/lib/python3.12/site-packages/engine/worker.py:512: ValueError: secret tensor shape mismatch'";
        let stream = simulate_mid_stream_error(2, backend_detail);
        let monitored = monitor_for_disconnects_with_timeout(stream, ctx, guard, handle, None);
        let body = collect_sse_body(monitored).await;
        assert_fault_contract("internal_detail_leak", &body, backend_detail);
        // Spot-check the most damaging fragments explicitly.
        assert!(!body.contains("site-packages"), "leaked a filesystem path");
        assert!(!body.contains("panicked at"), "leaked panic text");
        assert!(!body.contains("ValueError"), "leaked exception type");
    }

    struct TimeoutEnvGuard {
        saved: Vec<(&'static str, Option<std::ffi::OsString>)>,
    }

    impl TimeoutEnvGuard {
        fn cleared() -> Self {
            let vars = [
                BACKEND_STREAM_TIMEOUT_ENV,
                BACKEND_TTFT_TIMEOUT_ENV,
                SSE_INACTIVITY_TIMEOUT_ENV,
            ];
            let saved = vars
                .into_iter()
                .map(|name| (name, std::env::var_os(name)))
                .collect();
            for name in vars {
                // SAFETY: timeout env tests are serialized and the original
                // process values are restored by this guard.
                unsafe { std::env::remove_var(name) };
            }
            Self { saved }
        }
    }

    impl Drop for TimeoutEnvGuard {
        fn drop(&mut self) {
            for (name, value) in &self.saved {
                // SAFETY: timeout env tests are serialized and this restores
                // the exact process state captured at test entry.
                unsafe {
                    match value {
                        Some(value) => std::env::set_var(name, value),
                        None => std::env::remove_var(name),
                    }
                }
            }
        }
    }

    fn set_timeout_env(name: &str, value: &str) {
        // SAFETY: callers hold the serial timeout-env test lock.
        unsafe { std::env::set_var(name, value) };
    }

    #[test]
    #[serial]
    fn test_sse_timeout_precedence_and_zero_fallback() {
        let _env = TimeoutEnvGuard::cleared();
        assert_eq!(backend_stream_timeout(), None);

        set_timeout_env(BACKEND_STREAM_TIMEOUT_ENV, "5");
        assert_eq!(backend_stream_timeout(), Some(Duration::from_secs(10)));

        set_timeout_env(SSE_INACTIVITY_TIMEOUT_ENV, "45");
        assert_eq!(backend_stream_timeout(), Some(Duration::from_secs(45)));

        set_timeout_env(SSE_INACTIVITY_TIMEOUT_ENV, "0");
        assert_eq!(backend_stream_timeout(), Some(Duration::from_secs(10)));
    }

    #[test]
    #[serial]
    fn test_sse_timeout_override_governs_both_phases() {
        let _env = TimeoutEnvGuard::cleared();
        set_timeout_env(BACKEND_STREAM_TIMEOUT_ENV, "10");
        set_timeout_env(BACKEND_TTFT_TIMEOUT_ENV, "5");
        set_timeout_env(SSE_INACTIVITY_TIMEOUT_ENV, "60");

        assert_eq!(backend_stream_timeout(), Some(Duration::from_secs(60)));
        assert_eq!(backend_ttft_timeout(), Some(Duration::from_secs(60)));
        assert!(validate_backend_stream_timeouts().is_ok());
    }

    #[test]
    #[serial]
    fn test_sse_timeout_validation_rejects_primary_timer_races() {
        let _env = TimeoutEnvGuard::cleared();
        set_timeout_env(BACKEND_STREAM_TIMEOUT_ENV, "30");
        set_timeout_env(SSE_INACTIVITY_TIMEOUT_ENV, "30");
        let err = validate_backend_stream_timeouts().expect_err("equal timers must be rejected");
        assert!(err.to_string().contains(BACKEND_STREAM_TIMEOUT_ENV));

        set_timeout_env(BACKEND_STREAM_TIMEOUT_ENV, "4");
        set_timeout_env(BACKEND_TTFT_TIMEOUT_ENV, "10");
        set_timeout_env(SSE_INACTIVITY_TIMEOUT_ENV, "5");
        let err = validate_backend_stream_timeouts().expect_err("TTFT inversion must be rejected");
        assert!(err.to_string().contains(BACKEND_TTFT_TIMEOUT_ENV));
    }

    #[tokio::test]
    async fn test_clean_close_metrics_distinguish_backend_error_and_truncation() {
        for (case, finish_reason, expected) in [
            (
                "backend-error",
                Some(FinishReason::Error("worker failed".to_string())),
                ErrorType::BackendError,
            ),
            ("truncated", None, ErrorType::TruncatedStream),
        ] {
            let (metrics, guard, context, handle) = setup_test(case, case);
            let (_finish_reason_tx, finish_reason_rx) = tokio::sync::watch::channel(finish_reason);
            let stream = futures::stream::empty::<Result<Event, axum::Error>>();
            let monitored = monitor_for_disconnects_inner(
                stream,
                context,
                guard,
                handle,
                finish_reason_rx,
                None,
                None,
                None,
            );
            let body = collect_sse_body(monitored).await;

            assert!(body.contains("data: [DONE]"));
            assert_eq!(
                metrics.get_request_counter(
                    case,
                    &Endpoint::ChatCompletions,
                    &RequestType::Stream,
                    &Status::Error,
                    &expected,
                ),
                1,
                "{case} metric was not recorded",
            );
        }
    }

    #[tokio::test]
    async fn test_typed_timeout_errors_emit_specific_sse_envelopes_and_metrics() {
        for (case, dynamo_type, metric, wire_type, message) in [
            (
                "first-token",
                DynamoErrorType::FirstTokenTimeout,
                ErrorType::FirstTokenTimeout,
                "first_token_timeout",
                "backend time-to-first-token timeout",
            ),
            (
                "intra-token",
                DynamoErrorType::IntraTokenTimeout,
                ErrorType::IntraTokenTimeout,
                "intra_token_timeout",
                "backend response inactivity timeout",
            ),
        ] {
            let (metrics, guard, context, handle) = setup_test(case, case);
            let error = DynamoError::builder()
                .error_type(dynamo_type)
                .message(message)
                .build();
            let stream = futures::stream::iter([Err::<Event, _>(axum::Error::new(error))]);
            let (_finish_reason_tx, finish_reason_rx) = tokio::sync::watch::channel(None);
            let monitored = monitor_for_disconnects_inner(
                stream,
                context,
                guard,
                handle,
                finish_reason_rx,
                None,
                None,
                None,
            );
            let body = collect_sse_body(monitored).await;
            let error_payload = body
                .lines()
                .filter_map(|line| line.strip_prefix("data: "))
                .find_map(|payload| serde_json::from_str::<serde_json::Value>(payload).ok())
                .expect("structured timeout payload");

            assert_eq!(error_payload["error"]["type"], wire_type);
            assert_eq!(error_payload["error"]["code"], 504);
            assert_eq!(error_payload["error"]["message"], message);
            assert!(body.contains("data: [DONE]"));
            assert_eq!(
                metrics.get_request_counter(
                    case,
                    &Endpoint::ChatCompletions,
                    &RequestType::Stream,
                    &Status::Error,
                    &metric,
                ),
                1,
                "{case} timeout metric was not recorded",
            );
        }
    }

    #[tokio::test]
    async fn test_terminal_stream_error_kills_context() {
        let model = "stream-error-cancellation";
        let metrics = Arc::new(Metrics::new());
        let guard = metrics.clone().create_inflight_guard(
            model,
            Endpoint::ChatCompletions,
            true,
            "stream-error-cancellation",
        );
        let context = Arc::new(MockContext::with_kill_tracking());
        let engine_context: Arc<dyn AsyncEngineContext> = context.clone();
        let (tx, _rx) = tokio::sync::oneshot::channel();
        let handle = ConnectionHandle::create_disabled(tx);
        let error = DynamoError::builder()
            .error_type(DynamoErrorType::IntraTokenTimeout)
            .message("one batch prompt timed out")
            .build();
        let stream = futures::stream::iter([Err::<Event, _>(axum::Error::new(error))]);
        let (_finish_reason_tx, finish_reason_rx) = tokio::sync::watch::channel(None);
        let monitored = monitor_for_disconnects_inner(
            stream,
            engine_context,
            guard,
            handle,
            finish_reason_rx,
            None,
            None,
            None,
        );

        let _body = collect_sse_body(monitored).await;

        assert!(
            context.is_killed(),
            "terminal stream errors must cancel linked sibling requests"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn test_ttft_window_survives_leading_non_content_event() {
        let (metrics, guard, context, handle) = setup_test("ttft-leading", "ttft-leading");
        let first_token_seen = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let stream = async_stream::try_stream! {
            yield Event::default().data("annotation");
            std::future::pending::<()>().await;
        };
        let (_finish_reason_tx, finish_reason_rx) = tokio::sync::watch::channel(None);
        let monitored = monitor_for_disconnects_inner(
            stream,
            context,
            guard,
            handle,
            finish_reason_rx,
            Some(first_token_seen),
            None,
            Some(Duration::from_secs(2)),
        );
        tokio::pin!(monitored);

        assert!(monitored.next().await.is_some(), "leading event missing");
        tokio::time::timeout(Duration::from_secs(3), async {
            while monitored.next().await.is_some() {}
        })
        .await
        .expect("TTFT safety net did not fire after a non-content event");

        assert_eq!(
            metrics.get_request_counter(
                "ttft-leading",
                &Endpoint::ChatCompletions,
                &RequestType::Stream,
                &Status::Error,
                &ErrorType::ResponseTimeout,
            ),
            1,
        );
    }

    #[tokio::test(start_paused = true)]
    async fn test_ttft_window_releases_after_first_real_token() {
        let (_metrics, guard, context, handle) = setup_test("ttft-release", "ttft-release");
        let first_token_seen = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let producer_flag = first_token_seen.clone();
        let stream = async_stream::try_stream! {
            yield Event::default().data("annotation");
            producer_flag.store(true, std::sync::atomic::Ordering::Release);
            yield Event::default().data("real-token");
            std::future::pending::<()>().await;
        };
        let (_finish_reason_tx, finish_reason_rx) = tokio::sync::watch::channel(None);
        let monitored = monitor_for_disconnects_inner(
            stream,
            context,
            guard,
            handle,
            finish_reason_rx,
            Some(first_token_seen),
            None,
            Some(Duration::from_secs(2)),
        );
        tokio::pin!(monitored);

        assert!(monitored.next().await.is_some(), "leading event missing");
        assert!(monitored.next().await.is_some(), "real token missing");
        let next = tokio::time::timeout(Duration::from_secs(3), monitored.next()).await;
        assert!(
            next.is_err(),
            "TTFT timer remained armed after real content"
        );
    }
}
