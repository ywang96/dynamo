// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Content-scoped stop sequences (streaming spec P1.1).
//!
//! Engine-side stop scanning matches over *everything* the model generates —
//! a stop string appearing inside reasoning or tool-call markup kills the
//! request before any answer is produced. Per the spec, stop sequences apply
//! only to the visible answer (`delta.content`).
//!
//! When a reasoning or tool-call parser is active, the preprocessor strips
//! `stop` strings from the backend request and this stage enforces them
//! frontend-side instead, downstream of the wire-shape stage (so content
//! increments arrive as single-kind frames), per candidate index:
//!
//! - Only `content` increments are scanned. Reasoning and tool-call frames
//!   pass through untouched.
//! - Matches spanning chunk boundaries are caught by holding back any content
//!   tail that is a prefix of a stop string until it resolves.
//! - On a match: content before the match is emitted, a synthetic end frame
//!   (`delta: {}`, `finish_reason: "stop"`) closes the candidate, and later
//!   frames for that candidate are suppressed. The stop string itself is
//!   never emitted (matching engine-side semantics).
//! - Once every expected candidate has stopped or finished, remaining engine
//!   output is drained silently (no cancellation: the cancel path suppresses
//!   the usage chunk and `[DONE]`, but a stop match is a normal termination).
//!
//! Usage-only summary chunks and annotation events/errors pass through.
//! `usage`/`nvext`/`llm_metrics` on suppressed frames are folded into the
//! next emitted frame so accounting is never lost.

use std::collections::{HashMap, HashSet, VecDeque};
use std::pin::Pin;
use std::sync::Arc;

use futures::Stream;
use futures::stream::{self, StreamExt};

use dynamo_protocols::types::{
    ChatChoiceStream, ChatCompletionMessageContent, ChatCompletionStreamResponseDelta,
    FinishReason,
};
use dynamo_runtime::engine::AsyncEngineContext;

use super::wire_shape::{Frame, bare, empty_delta, frame_from_template};
use crate::protocols::openai::chat_completions::NvCreateChatCompletionStreamResponse;

/// Enforce content-scoped stop sequences on a shaped chat stream.
pub(crate) fn scan_content_stop<S>(
    stream: S,
    stop_sequences: Vec<String>,
    expected_candidates: usize,
    context: Arc<dyn AsyncEngineContext>,
) -> impl Stream<Item = Frame> + Send
where
    S: Stream<Item = Frame> + Send + 'static,
{
    let stop_sequences: Vec<String> = stop_sequences
        .into_iter()
        .filter(|s| !s.is_empty())
        .collect();
    let state = ScanState {
        stream: Box::pin(stream),
        pending: VecDeque::new(),
        candidates: HashMap::new(),
        done_candidates: HashSet::new(),
        expected_candidates: expected_candidates.max(1),
        cancelled: false,
        stop_sequences,
        context,
        carry: super::wire_shape::PendingCarry::default(),
    };

    stream::unfold(state, |mut state| async move {
        loop {
            if let Some(frame) = state.pending.pop_front() {
                return Some((frame, state));
            }
            match state.stream.next().await {
                Some(frame) => {
                    let outs = state.process(frame);
                    state.pending.extend(outs);
                }
                None => {
                    // Flush any held-back content tails (no match completed).
                    let outs = state.flush_all();
                    if outs.is_empty() {
                        return None;
                    }
                    state.pending.extend(outs);
                }
            }
        }
    })
    .fuse()
}

#[derive(Default)]
struct CandidateScan {
    /// Content tail held back because it is a prefix of a stop string.
    held: String,
    /// Frame template for flushing held content at stream end.
    template: Option<NvCreateChatCompletionStreamResponse>,
    /// Candidate was terminated by a frontend stop match.
    stopped: bool,
}

struct ScanState {
    stream: Pin<Box<dyn Stream<Item = Frame> + Send>>,
    pending: VecDeque<Frame>,
    candidates: HashMap<u32, CandidateScan>,
    /// Candidates that finished (naturally or via stop match).
    done_candidates: HashSet<u32>,
    expected_candidates: usize,
    cancelled: bool,
    stop_sequences: Vec<String>,
    context: Arc<dyn AsyncEngineContext>,
    carry: super::wire_shape::PendingCarry,
}

impl ScanState {
    /// Earliest match of any stop sequence in `text`.
    fn find_stop(&self, text: &str) -> Option<usize> {
        self.stop_sequences
            .iter()
            .filter_map(|stop| text.find(stop.as_str()))
            .min()
    }

    /// Length of the longest suffix of `text` that is a proper prefix of any
    /// stop sequence (a potential match still waiting for more bytes).
    fn pending_prefix_len(&self, text: &str) -> usize {
        let mut longest = 0;
        for stop in &self.stop_sequences {
            let max = stop.len().saturating_sub(1).min(text.len());
            for take in (longest + 1..=max).rev() {
                if !text.is_char_boundary(text.len() - take) {
                    continue;
                }
                if stop.starts_with(&text[text.len() - take..]) {
                    longest = take;
                    break;
                }
            }
        }
        longest
    }

    fn content_frame(
        template: &NvCreateChatCompletionStreamResponse,
        index: u32,
        text: String,
    ) -> Frame {
        #[allow(deprecated)]
        let delta = ChatCompletionStreamResponseDelta {
            content: Some(ChatCompletionMessageContent::Text(text)),
            ..empty_delta()
        };
        bare(frame_from_template(
            template,
            ChatChoiceStream {
                index,
                delta,
                finish_reason: None,
                logprobs: None,
            },
        ))
    }

    fn end_frame(template: &NvCreateChatCompletionStreamResponse, index: u32) -> Frame {
        bare(frame_from_template(
            template,
            ChatChoiceStream {
                index,
                delta: empty_delta(),
                finish_reason: Some(FinishReason::Stop),
                logprobs: None,
            },
        ))
    }

    fn mark_done(&mut self, index: u32) {
        self.done_candidates.insert(index);
        if !self.cancelled && self.done_candidates.len() >= self.expected_candidates {
            self.cancelled = true;
            // Deliberately NOT calling context.stop_generating(): the
            // cancellation path is treated as a client disconnect by the HTTP
            // framing layer, which suppresses the usage summary chunk and the
            // SSE `[DONE]` terminator (P1.2). A stop-sequence termination is a
            // NORMAL end of stream, so we let the engine run to its natural
            // finish (bounded by max_tokens) and suppress the frames instead.
            // TODO: early-cancel would need a graceful-stop signal the
            // disconnect machinery distinguishes from client aborts.
            tracing::debug!(
                request_id = self.context.id(),
                "all candidates stopped/finished; draining engine output"
            );
        }
    }

    /// Drain held content for a candidate into `outs` (no pending match).
    fn flush_held(&mut self, index: u32, outs: &mut Vec<Frame>) {
        if let Some(scan) = self.candidates.get_mut(&index)
            && !scan.held.is_empty()
            && let Some(template) = scan.template.clone()
        {
            let held = std::mem::take(&mut scan.held);
            outs.push(Self::content_frame(&template, index, held));
        }
    }

    fn flush_all(&mut self) -> Vec<Frame> {
        let mut outs = Vec::new();
        let indexes: Vec<u32> = self
            .candidates
            .iter()
            .filter(|(_, scan)| !scan.stopped && !scan.held.is_empty())
            .map(|(index, _)| *index)
            .collect();
        for index in indexes {
            self.flush_held(index, &mut outs);
        }
        outs
    }

    fn process(&mut self, mut frame: Frame) -> Vec<Frame> {
        // Annotation events/errors pass through untouched.
        if frame.event.is_some() || frame.error.is_some() {
            return vec![frame];
        }
        let Some(mut data) = frame.data.take() else {
            return vec![frame];
        };
        // Usage-only summary chunks (and any multi-choice frame) pass through;
        // fold buffered accounting payloads from suppressed frames into them.
        if data.inner.choices.len() != 1 {
            let mut usage = data.inner.usage.take();
            let mut nvext = data.nvext.take();
            let mut llm_metrics = data.llm_metrics.take();
            self.carry
                .drain_into(&mut usage, &mut nvext, &mut llm_metrics);
            data.inner.usage = usage;
            data.nvext = nvext;
            data.llm_metrics = llm_metrics;
            frame.data = Some(data);
            return vec![frame];
        }

        let index = data.inner.choices[0].index;
        let scan = self.candidates.entry(index).or_default();
        if scan.template.is_none() {
            let mut template = data.clone();
            template.inner.usage = None;
            template.nvext = None;
            template.llm_metrics = None;
            scan.template = Some(template);
        }

        // Suppressed candidate: swallow the frame, keep its accounting.
        if scan.stopped {
            self.carry.stash(
                data.inner.usage.take(),
                data.nvext.take(),
                data.llm_metrics.take(),
            );
            return Vec::new();
        }

        let choice = &data.inner.choices[0];
        // Only non-empty text content is scannable. Empty-content frames — the
        // wire-shape stage's canonical first frame {"role":"assistant",
        // "content":""} in particular — pass through untouched.
        let is_content = matches!(
            &choice.delta.content,
            Some(ChatCompletionMessageContent::Text(text)) if !text.is_empty()
        );
        let finish = choice.finish_reason;

        if !is_content {
            let mut outs = Vec::new();
            // Preserve ordering: held content precedes any other increment or
            // the end frame.
            self.flush_held(index, &mut outs);
            if finish.is_some() {
                self.mark_done(index);
            }
            frame.data = Some(data);
            outs.push(frame);
            return outs;
        }

        // Content increment: scan (held + text) for stop sequences.
        let Some(ChatCompletionMessageContent::Text(text)) =
            data.inner.choices[0].delta.content.clone()
        else {
            unreachable!("is_content checked above");
        };
        let scan = self.candidates.get_mut(&index).expect("entry created");
        let template = scan.template.clone().expect("template captured");
        let combined = format!("{}{}", scan.held, text);
        scan.held.clear();

        let mut outs = Vec::new();
        if let Some(pos) = self.find_stop(&combined) {
            // Stop match: emit content before the match, close the candidate.
            if pos > 0 {
                outs.push(Self::content_frame(&template, index, combined[..pos].to_string()));
            }
            outs.push(Self::end_frame(&template, index));
            let scan = self.candidates.get_mut(&index).expect("entry created");
            scan.stopped = true;
            // The original frame's accounting must survive even though its
            // content is replaced by the truncated emission.
            self.carry.stash(
                data.inner.usage.take(),
                data.nvext.take(),
                data.llm_metrics.take(),
            );
            self.mark_done(index);
            return outs;
        }

        let hold = self.pending_prefix_len(&combined);
        let emit_len = combined.len() - hold;
        let scan = self.candidates.get_mut(&index).expect("entry created");
        scan.held = combined[emit_len..].to_string();
        if emit_len > 0 {
            // Reuse the original frame so usage/nvext/metrics stay attached.
            data.inner.choices[0].delta.content = Some(ChatCompletionMessageContent::Text(
                combined[..emit_len].to_string(),
            ));
            if finish.is_some() {
                self.mark_done(index);
            }
            frame.data = Some(data);
            outs.push(frame);
        } else {
            // Entire increment held back: stash accounting, emit nothing yet
            // (unless this was the end frame, which never carries content
            // post-wire-shape — defensive).
            self.carry.stash(
                data.inner.usage.take(),
                data.nvext.take(),
                data.llm_metrics.take(),
            );
            if finish.is_some() {
                self.mark_done(index);
                frame.data = Some(data);
                outs.push(frame);
            }
        }
        outs
    }
}

#[cfg(test)]
mod tests {
    use super::super::wire_shape::Frame;
    use super::*;
    use dynamo_protocols::types::Role;
    use futures::executor::block_on;
    use std::sync::atomic::{AtomicBool, Ordering};

    #[derive(Debug)]
    struct MockContext(AtomicBool);

    #[async_trait::async_trait]
    impl AsyncEngineContext for MockContext {
        fn id(&self) -> &str {
            "test"
        }
        fn stop(&self) {}
        fn stop_generating(&self) {
            self.0.store(true, Ordering::SeqCst);
        }
        fn kill(&self) {}
        fn is_stopped(&self) -> bool {
            false
        }
        fn is_killed(&self) -> bool {
            false
        }
        async fn stopped(&self) {}
        async fn killed(&self) {}
        fn link_child(&self, _: Arc<dyn AsyncEngineContext>) {}
    }

    fn base_response() -> NvCreateChatCompletionStreamResponse {
        NvCreateChatCompletionStreamResponse {
            inner: dynamo_protocols::types::CreateChatCompletionStreamResponse {
                id: "chatcmpl-test".to_string(),
                object: "chat.completion.chunk".to_string(),
                created: 1_700_000_000,
                model: "test-model".to_string(),
                system_fingerprint: None,
                choices: vec![],
                usage: None,
                service_tier: None,
            },
            nvext: None,
            llm_metrics: None,
            choice_usage: None,
        }
    }

    #[allow(deprecated)]
    fn frame(
        index: u32,
        role: Option<Role>,
        content: Option<&str>,
        reasoning: Option<&str>,
        finish: Option<FinishReason>,
    ) -> Frame {
        let mut response = base_response();
        response.inner.choices = vec![ChatChoiceStream {
            index,
            delta: ChatCompletionStreamResponseDelta {
                role,
                content: content.map(|c| ChatCompletionMessageContent::Text(c.to_string())),
                function_call: None,
                tool_calls: None,
                refusal: None,
                reasoning_content: reasoning.map(str::to_string),
            },
            finish_reason: finish,
            logprobs: None,
        }];
        bare(response)
    }

    fn run(
        frames: Vec<Frame>,
        stops: &[&str],
        n: usize,
    ) -> (Vec<(Option<String>, Option<String>, Option<FinishReason>)>, bool) {
        let ctx = Arc::new(MockContext(AtomicBool::new(false)));
        let out = block_on(
            scan_content_stop(
                stream::iter(frames),
                stops.iter().map(|s| s.to_string()).collect(),
                n,
                ctx.clone(),
            )
            .collect::<Vec<_>>(),
        );
        let rows = out
            .into_iter()
            .filter_map(|f| f.data)
            .filter(|d| !d.inner.choices.is_empty())
            .map(|d| {
                let c = &d.inner.choices[0];
                let content = match &c.delta.content {
                    Some(ChatCompletionMessageContent::Text(t)) => Some(t.clone()),
                    _ => None,
                };
                (content, c.delta.reasoning_content.clone(), c.finish_reason)
            })
            .collect();
        (rows, ctx.0.load(Ordering::SeqCst))
    }

    #[test]
    fn stop_truncates_content_and_closes_candidate() {
        let (rows, cancelled) = run(
            vec![
                frame(0, Some(Role::Assistant), Some(""), None, None),
                frame(0, None, Some("ONE TWO "), None, None),
                frame(0, None, Some("THREE FOUR FIVE"), None, None),
                frame(0, None, Some(" SIX"), None, None),
                frame(0, None, None, None, Some(FinishReason::Length)),
            ],
            &["FOUR"],
            1,
        );
        let contents: String = rows.iter().filter_map(|r| r.0.clone()).collect();
        assert_eq!(contents, "ONE TWO THREE ");
        assert_eq!(
            rows.last().unwrap().2,
            Some(FinishReason::Stop),
            "candidate must close with finish_reason=stop"
        );
        // The post-stop content frame and the natural end frame are suppressed.
        assert_eq!(rows.iter().filter(|r| r.2.is_some()).count(), 1);
        assert!(
            !cancelled,
            "must NOT cancel: the cancel path suppresses usage/[DONE]; drain instead"
        );
    }

    #[test]
    fn canonical_first_frame_passes_through() {
        // Regression: the wire-shape first frame {"role","content":""} was
        // routed into the scanner and swallowed (empty text held forever).
        let (rows, _) = run(
            vec![
                frame(0, Some(Role::Assistant), Some(""), None, None),
                frame(0, None, Some("hello"), None, None),
                frame(0, None, None, None, Some(FinishReason::Stop)),
            ],
            &["FOUR"],
            1,
        );
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].0.as_deref(), Some(""), "first frame must survive");
        assert_eq!(rows[1].0.as_deref(), Some("hello"));
        assert_eq!(rows[2].2, Some(FinishReason::Stop));
    }

    #[test]
    fn stop_split_across_chunks_is_caught() {
        let (rows, _) = run(
            vec![
                frame(0, None, Some("alpha FO"), None, None),
                frame(0, None, Some("UR beta"), None, None),
            ],
            &["FOUR"],
            1,
        );
        let contents: String = rows.iter().filter_map(|r| r.0.clone()).collect();
        assert_eq!(contents, "alpha ");
        assert_eq!(rows.last().unwrap().2, Some(FinishReason::Stop));
    }

    #[test]
    fn stop_in_reasoning_does_not_fire() {
        let (rows, cancelled) = run(
            vec![
                frame(0, None, None, Some("thinking about FOUR"), None),
                frame(0, None, None, Some(""), None),
                frame(0, None, Some("the answer is 4"), None, None),
                frame(0, None, None, None, Some(FinishReason::Stop)),
            ],
            &["FOUR"],
            1,
        );
        let reasoning: String = rows.iter().filter_map(|r| r.1.clone()).collect();
        assert_eq!(reasoning, "thinking about FOUR");
        let contents: String = rows.iter().filter_map(|r| r.0.clone()).collect();
        assert_eq!(contents, "the answer is 4");
        assert!(!cancelled, "never cancels; drains to natural finish");
    }

    #[test]
    fn held_prefix_is_flushed_when_no_match_completes() {
        let (rows, _) = run(
            vec![
                frame(0, None, Some("count FO"), None, None),
                frame(0, None, Some("X done"), None, None),
                frame(0, None, None, None, Some(FinishReason::Stop)),
            ],
            &["FOUR"],
            1,
        );
        let contents: String = rows.iter().filter_map(|r| r.0.clone()).collect();
        assert_eq!(contents, "count FOX done", "held prefix must not be lost");
    }

    #[test]
    fn held_tail_flushes_at_stream_end() {
        // Stream ends (no finish frame) while a potential match is held.
        let (rows, _) = run(
            vec![frame(0, None, Some("tail FOU"), None, None)],
            &["FOUR"],
            1,
        );
        let contents: String = rows.iter().filter_map(|r| r.0.clone()).collect();
        assert_eq!(contents, "tail FOU");
    }

    #[test]
    fn candidates_stop_independently() {
        let (rows, cancelled) = run(
            vec![
                frame(0, None, Some("hit FOUR now"), None, None),
                frame(1, None, Some("no match here"), None, None),
                frame(1, None, Some(" more"), None, None),
                frame(1, None, None, None, Some(FinishReason::Stop)),
            ],
            &["FOUR"],
            2,
        );
        // Candidate 0 truncated + stopped; candidate 1 streams fully.
        let c1: String = rows
            .iter()
            .filter(|r| r.0.is_some())
            .filter_map(|r| r.0.clone())
            .collect();
        assert_eq!(c1, "hit no match here more");
        assert_eq!(rows.iter().filter(|r| r.2.is_some()).count(), 2);
        assert!(!cancelled, "never cancels; drains to natural finish");
    }
}
