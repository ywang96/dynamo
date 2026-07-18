// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Wire-shape enforcement for streamed chat completions.
//!
//! Kimi's streaming output spec (and the OpenAI reference behavior) pins down
//! the chunk shapes the SSE stream must follow; the shapes are enforced here,
//! as the last transform before frames leave the parsing pipeline:
//!
//! - P0.9 / P1.5 / P1.6: the first frame per candidate is exactly
//!   `{"role": "assistant", "content": ""}`, and `role` appears on no other
//!   frame.
//! - P0.10 / P0.12: a non-end frame carries exactly one increment kind
//!   (`content` | `reasoning_content` | `tool_calls`); mixed frames are split
//!   into one frame per kind, in reasoning → content → tool_calls order.
//! - P1.10 / P1.12: the end frame's `delta` is `{}` — an increment arriving
//!   together with `finish_reason` is split into an increment frame followed
//!   by a bare end frame.
//! - Frames left with no increment and no `finish_reason` are dropped rather
//!   than emitted as empty deltas.
//!
//! Frames carrying annotation events or errors, usage-only summary chunks
//! (empty `choices`), and multi-choice frames pass through untouched.

use std::collections::{HashSet, VecDeque};
use std::pin::Pin;

use futures::Stream;
use futures::stream::{self, StreamExt};

use dynamo_runtime::protocols::annotated::Annotated;

use crate::protocols::openai::chat_completions::NvCreateChatCompletionStreamResponse;
use dynamo_protocols::types::{
    ChatChoiceStream, ChatCompletionMessageContent, ChatCompletionMessageToolCallChunk,
    ChatCompletionStreamResponseDelta, FinishReason, Role,
};

pub(crate) type Frame = Annotated<NvCreateChatCompletionStreamResponse>;

/// Enforce the streaming wire shape on a chat completion stream.
pub(crate) fn shape_chat_stream<S>(stream: S) -> impl Stream<Item = Frame> + Send
where
    S: Stream<Item = Frame> + Send + 'static,
{
    let state = ShapeState {
        stream: Box::pin(stream),
        pending: VecDeque::new(),
        role_sent: HashSet::new(),
        reasoning_open: HashSet::new(),
        carry: PendingCarry::default(),
    };

    stream::unfold(state, |mut state| async move {
        loop {
            if let Some(frame) = state.pending.pop_front() {
                return Some((frame, state));
            }
            match state.stream.next().await {
                Some(frame) => {
                    let ShapeState {
                        role_sent,
                        reasoning_open,
                        carry,
                        ..
                    } = &mut state;
                    for out in shape_frame(frame, role_sent, reasoning_open, carry) {
                        state.pending.push_back(out);
                    }
                }
                None => return None,
            }
        }
    })
    .fuse()
}

struct ShapeState {
    stream: Pin<Box<dyn Stream<Item = Frame> + Send>>,
    pending: VecDeque<Frame>,
    role_sent: HashSet<u32>,
    /// Candidates that have emitted reasoning increments but not yet the
    /// P1.7 `{"reasoning_content": ""}` end-of-thinking boundary frame.
    reasoning_open: HashSet<u32>,
    carry: PendingCarry,
}

/// Payloads buffered from dropped frames so shaping never loses accounting
/// data. `llm_metrics.chunk_tokens` is cumulative across drops (mirroring the
/// tool jail's buffering of the same field); `usage`/`nvext` are
/// carried only until a later frame provides its own.
#[derive(Default)]
pub(crate) struct PendingCarry {
    usage: Option<dynamo_protocols::types::CompletionUsage>,
    nvext: Option<serde_json::Value>,
    llm_metrics: Option<crate::protocols::common::metrics::LLMMetricAnnotation>,
}

impl PendingCarry {
    pub(crate) fn stash(
        &mut self,
        usage: Option<dynamo_protocols::types::CompletionUsage>,
        nvext: Option<serde_json::Value>,
        llm_metrics: Option<crate::protocols::common::metrics::LLMMetricAnnotation>,
    ) {
        if usage.is_some() {
            self.usage = usage;
        }
        if nvext.is_some() {
            self.nvext = nvext;
        }
        if let Some(metrics) = llm_metrics {
            match self.llm_metrics.as_mut() {
                Some(pending) => {
                    // Latest cumulative fields win; per-chunk counts add up.
                    let chunk_tokens = pending.chunk_tokens.saturating_add(metrics.chunk_tokens);
                    *pending = metrics;
                    pending.chunk_tokens = chunk_tokens;
                }
                None => self.llm_metrics = Some(metrics),
            }
        }
    }

    /// Fold the pending payloads into a frame's own (the frame's own values
    /// win for `usage`/`nvext`; metrics accumulate).
    pub(crate) fn drain_into(
        &mut self,
        usage: &mut Option<dynamo_protocols::types::CompletionUsage>,
        nvext: &mut Option<serde_json::Value>,
        llm_metrics: &mut Option<crate::protocols::common::metrics::LLMMetricAnnotation>,
    ) {
        if usage.is_none() {
            *usage = self.usage.take();
        } else {
            self.usage = None;
        }
        if nvext.is_none() {
            *nvext = self.nvext.take();
        } else {
            self.nvext = None;
        }
        if let Some(pending) = self.llm_metrics.take() {
            match llm_metrics.as_mut() {
                Some(own) => own.chunk_tokens = own.chunk_tokens.saturating_add(pending.chunk_tokens),
                None => *llm_metrics = Some(pending),
            }
        }
    }
}

#[allow(deprecated)]
pub(crate) fn empty_delta() -> ChatCompletionStreamResponseDelta {
    ChatCompletionStreamResponseDelta {
        role: None,
        content: None,
        function_call: None,
        tool_calls: None,
        refusal: None,
        reasoning_content: None,
    }
}

/// A bare response carrying one choice, cloned from `template` for the
/// constant top-level fields (id/object/created/model/...). Usage, nvext and
/// internal metrics never ride these synthesized frames.
pub(crate) fn frame_from_template(
    template: &NvCreateChatCompletionStreamResponse,
    choice: ChatChoiceStream,
) -> NvCreateChatCompletionStreamResponse {
    let mut response = template.clone();
    response.inner.usage = None;
    response.inner.choices = vec![choice];
    response.nvext = None;
    response.llm_metrics = None;
    response.choice_usage = None;
    response
}

pub(crate) fn bare(data: NvCreateChatCompletionStreamResponse) -> Frame {
    Annotated {
        data: Some(data),
        id: None,
        event: None,
        comment: None,
        error: None,
    }
}

fn is_empty_content(content: &ChatCompletionMessageContent) -> bool {
    matches!(content, ChatCompletionMessageContent::Text(text) if text.is_empty())
}

fn is_empty_tool_calls(tool_calls: &[ChatCompletionMessageToolCallChunk]) -> bool {
    tool_calls.is_empty()
}

/// Normalize one tool-call chunk to the spec's streaming shape.
///
/// A "complete" chunk (a name together with non-empty arguments — the shape
/// the tool jail emits after parsing a jailed call) becomes a header chunk
/// with `arguments: ""` plus a continuation chunk carrying only
/// index + arguments. A header chunk with missing arguments gets `""` filled
/// in (the `arguments` key must always be present); already-split chunks pass
/// through unchanged.
fn normalize_tool_call_chunk(
    mut chunk: ChatCompletionMessageToolCallChunk,
) -> Vec<ChatCompletionMessageToolCallChunk> {
    let Some(function) = chunk.function.as_mut() else {
        return vec![chunk];
    };

    if function.name.is_some() {
        // Header-bearing chunk: the spec requires its arguments to be "".
        match function.arguments.take() {
            Some(arguments) if !arguments.is_empty() => {
                function.arguments = Some(String::new());
                let index = chunk.index;
                let continuation = ChatCompletionMessageToolCallChunk {
                    index,
                    id: None,
                    r#type: None,
                    function: Some(dynamo_protocols::types::FunctionCallStream {
                        name: None,
                        arguments: Some(arguments),
                    }),
                };
                vec![chunk, continuation]
            }
            _ => {
                function.arguments = Some(String::new());
                vec![chunk]
            }
        }
    } else {
        // Continuation chunk: ensure the arguments key is present.
        if function.arguments.is_none() {
            function.arguments = Some(String::new());
        }
        vec![chunk]
    }
}

/// Shape one incoming frame into zero or more compliant frames.
fn shape_frame(
    mut frame: Frame,
    role_sent: &mut HashSet<u32>,
    reasoning_open: &mut HashSet<u32>,
    carry: &mut PendingCarry,
) -> Vec<Frame> {
    // Events and errors ride annotations; never reshape those frames.
    if frame.event.is_some() || frame.error.is_some() {
        return vec![frame];
    }
    let Some(mut data) = frame.data.take() else {
        return vec![frame];
    };
    // Usage-only summary chunks (empty choices) and multi-choice frames pass
    // through unchanged; the per-candidate shaping below assumes the
    // one-choice-per-frame layout every dynamo delta generator produces.
    if data.inner.choices.len() != 1 {
        frame.data = Some(data);
        return vec![frame];
    }

    let index = data.inner.choices[0].index;

    let mut outputs: Vec<NvCreateChatCompletionStreamResponse> = Vec::new();

    if role_sent.insert(index) {
        #[allow(deprecated)]
        let first_delta = ChatCompletionStreamResponseDelta {
            role: Some(Role::Assistant),
            content: Some(ChatCompletionMessageContent::Text(String::new())),
            ..empty_delta()
        };
        outputs.push(frame_from_template(
            &data,
            ChatChoiceStream {
                index,
                delta: first_delta,
                finish_reason: None,
                logprobs: None,
            },
        ));
    }

    // Decompose the incoming choice into its increment kinds. Empty
    // increments (empty text, empty tool-call list) are treated as absent —
    // the spec forbids empty increment frames. Scoped so the mutable borrow
    // ends before `data` is used as an immutable frame template below.
    let (reasoning, content, tool_calls, function_call, refusal, finish_reason, logprobs) = {
        let choice = &mut data.inner.choices[0];
        choice.delta.role = None;
        (
            choice
                .delta
                .reasoning_content
                .take()
                .filter(|text| !text.is_empty()),
            choice.delta.content.take().filter(|c| !is_empty_content(c)),
            choice
                .delta
                .tool_calls
                .take()
                .filter(|calls| !is_empty_tool_calls(calls)),
            choice.delta.function_call.take(),
            choice.delta.refusal.take(),
            choice.finish_reason.take(),
            choice.logprobs.take(),
        )
    };
    let mut usage = data.inner.usage.take();
    let mut nvext = data.nvext.take();
    let mut llm_metrics = data.llm_metrics.take();
    // Per-candidate usage (P0.4) travels with finish_reason; reattached to
    // the end frame below.
    let choice_usage = data.choice_usage.take();

    if let Some(reasoning) = reasoning {
        #[allow(deprecated)]
        let delta = ChatCompletionStreamResponseDelta {
            reasoning_content: Some(reasoning),
            ..empty_delta()
        };
        outputs.push(frame_from_template(
            &data,
            ChatChoiceStream {
                index,
                delta,
                finish_reason: None,
                logprobs: None,
            },
        ));
        reasoning_open.insert(index);
    }
    // P1.7: once a candidate's thinking ends — the first content/tool
    // increment arrives, or the candidate finishes — emit the standalone
    // `{"reasoning_content": ""}` end-of-thinking boundary frame, after the
    // last reasoning increment and before anything else.
    let thinking_ends = content.is_some()
        || function_call.is_some()
        || refusal.is_some()
        || tool_calls.is_some()
        || finish_reason.is_some();
    if thinking_ends && reasoning_open.remove(&index) {
        #[allow(deprecated)]
        let delta = ChatCompletionStreamResponseDelta {
            reasoning_content: Some(String::new()),
            ..empty_delta()
        };
        outputs.push(frame_from_template(
            &data,
            ChatChoiceStream {
                index,
                delta,
                finish_reason: None,
                logprobs: None,
            },
        ));
    }
    if content.is_some() || function_call.is_some() || refusal.is_some() {
        #[allow(deprecated)]
        let delta = ChatCompletionStreamResponseDelta {
            content,
            function_call,
            refusal,
            ..empty_delta()
        };
        outputs.push(frame_from_template(
            &data,
            ChatChoiceStream {
                index,
                delta,
                // Logprobs describe the generated tokens; keep them on the
                // content increment they belong to.
                finish_reason: None,
                logprobs,
            },
        ));
    }
    if let Some(tool_calls) = tool_calls {
        // P0.13/P0.14/P1.8/P1.9: the jail emits each tool call as one complete
        // chunk (id + name + full arguments), but the spec requires a header
        // chunk carrying `arguments: ""` followed by argument continuations
        // holding only index + arguments. Explode every chunk into that shape,
        // one frame per chunk, in array order so per-index frames stay
        // contiguous.
        for chunk in tool_calls {
            for normalized in normalize_tool_call_chunk(chunk) {
                #[allow(deprecated)]
                let delta = ChatCompletionStreamResponseDelta {
                    tool_calls: Some(vec![normalized]),
                    ..empty_delta()
                };
                outputs.push(frame_from_template(
                    &data,
                    ChatChoiceStream {
                        index,
                        delta,
                        finish_reason: None,
                        logprobs: None,
                    },
                ));
            }
        }
    }
    if finish_reason.is_some() {
        let mut end = frame_from_template(
            &data,
            ChatChoiceStream {
                index,
                delta: empty_delta(),
                finish_reason,
                logprobs: None,
            },
        );
        end.choice_usage = choice_usage;
        outputs.push(end);
    }

    if outputs.is_empty() {
        // Nothing survived (e.g. a role-only heartbeat after the first
        // frame): drop the frame, but never its accounting payloads — buffer
        // them for the next emitted frame.
        carry.stash(usage, nvext, llm_metrics);
        return Vec::new();
    }

    // Usage, nvext and internal metrics ride the last emitted frame — for a
    // finished candidate that is the end frame, matching where the final-chunk
    // extension payloads (timing, token_ids) are expected. Payloads buffered
    // from dropped frames are folded in first.
    carry.drain_into(&mut usage, &mut nvext, &mut llm_metrics);
    if let Some(last) = outputs.last_mut() {
        last.inner.usage = usage;
        last.nvext = nvext;
        last.llm_metrics = llm_metrics;
    }

    let mut frames: Vec<Frame> = Vec::with_capacity(outputs.len());
    for (position, output) in outputs.into_iter().enumerate() {
        if position == 0 {
            // Keep the original annotation metadata (id/comment) on the first
            // emitted frame only, so split frames don't duplicate SSE ids.
            frames.push(Annotated {
                data: Some(output),
                id: frame.id.take(),
                event: None,
                comment: frame.comment.take(),
                error: None,
            });
        } else {
            frames.push(bare(output));
        }
    }
    frames
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;

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
                content: content
                    .map(|c| ChatCompletionMessageContent::Text(c.to_string())),
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

    fn collect(frames: Vec<Frame>) -> Vec<NvCreateChatCompletionStreamResponse> {
        block_on(
            shape_chat_stream(stream::iter(frames))
                .map(|frame| frame.data.expect("data frame"))
                .collect::<Vec<_>>(),
        )
    }

    fn delta(response: &NvCreateChatCompletionStreamResponse) -> &ChatCompletionStreamResponseDelta {
        &response.inner.choices[0].delta
    }

    fn assert_is_first_frame(response: &NvCreateChatCompletionStreamResponse, index: u32) {
        let choice = &response.inner.choices[0];
        assert_eq!(choice.index, index);
        assert_eq!(delta(response).role, Some(Role::Assistant));
        assert_eq!(
            delta(response).content,
            Some(ChatCompletionMessageContent::Text(String::new()))
        );
        assert!(delta(response).reasoning_content.is_none());
        assert!(delta(response).tool_calls.is_none());
        assert!(choice.finish_reason.is_none());
    }

    #[test]
    fn injects_first_frame_before_reasoning_and_strips_role() {
        // Mirrors the live P0.9 failure: stream opens with a
        // role+reasoning_content delta and ends with a role-carrying end frame.
        let out = collect(vec![
            frame(0, Some(Role::Assistant), None, Some("Th"), None),
            frame(0, Some(Role::Assistant), None, Some("inking"), None),
            frame(0, Some(Role::Assistant), Some("Answer"), None, None),
            frame(0, Some(Role::Assistant), None, None, Some(FinishReason::Stop)),
        ]);

        assert_eq!(out.len(), 6);
        assert_is_first_frame(&out[0], 0);
        assert_eq!(delta(&out[1]).reasoning_content.as_deref(), Some("Th"));
        assert!(delta(&out[1]).role.is_none());
        assert_eq!(delta(&out[2]).reasoning_content.as_deref(), Some("inking"));
        // P1.7 boundary: standalone reasoning_content == "" after the last
        // reasoning increment and before the first content.
        assert_eq!(delta(&out[3]).reasoning_content.as_deref(), Some(""));
        assert!(delta(&out[3]).content.is_none());
        assert_eq!(
            delta(&out[4]).content,
            Some(ChatCompletionMessageContent::Text("Answer".to_string()))
        );
        // End frame: empty delta, finish_reason only.
        let end = &out[5];
        assert_eq!(*delta(end), empty_delta());
        assert_eq!(
            end.inner.choices[0].finish_reason,
            Some(FinishReason::Stop)
        );
    }

    #[test]
    fn splits_increment_off_end_frame() {
        let out = collect(vec![frame(
            0,
            Some(Role::Assistant),
            Some("hi"),
            None,
            Some(FinishReason::Stop),
        )]);

        assert_eq!(out.len(), 3);
        assert_is_first_frame(&out[0], 0);
        assert_eq!(
            delta(&out[1]).content,
            Some(ChatCompletionMessageContent::Text("hi".to_string()))
        );
        assert!(out[1].inner.choices[0].finish_reason.is_none());
        assert_eq!(*delta(&out[2]), empty_delta());
        assert_eq!(
            out[2].inner.choices[0].finish_reason,
            Some(FinishReason::Stop)
        );
    }

    #[test]
    fn splits_mixed_increment_kinds() {
        let out = collect(vec![
            frame(0, Some(Role::Assistant), Some("text"), Some("thought"), None),
            frame(0, None, None, None, Some(FinishReason::Stop)),
        ]);

        assert_eq!(out.len(), 5);
        assert_is_first_frame(&out[0], 0);
        // Reasoning precedes content, with the P1.7 boundary between them.
        assert_eq!(delta(&out[1]).reasoning_content.as_deref(), Some("thought"));
        assert!(delta(&out[1]).content.is_none());
        assert_eq!(delta(&out[2]).reasoning_content.as_deref(), Some(""));
        assert_eq!(
            delta(&out[3]).content,
            Some(ChatCompletionMessageContent::Text("text".to_string()))
        );
        assert!(delta(&out[3]).reasoning_content.is_none());
        assert_eq!(*delta(&out[4]), empty_delta());
    }

    #[test]
    fn emits_reasoning_boundary_before_finish_without_content() {
        // A candidate that only thinks and then finishes still gets the P1.7
        // boundary, before the end frame; a candidate that never reasons gets
        // no boundary at all.
        let out = collect(vec![
            frame(0, Some(Role::Assistant), None, Some("only thinking"), None),
            frame(0, None, None, None, Some(FinishReason::Stop)),
        ]);
        assert_eq!(out.len(), 4);
        assert_eq!(delta(&out[1]).reasoning_content.as_deref(), Some("only thinking"));
        assert_eq!(delta(&out[2]).reasoning_content.as_deref(), Some(""));
        assert_eq!(
            out[3].inner.choices[0].finish_reason,
            Some(FinishReason::Stop)
        );

        let out = collect(vec![
            frame(0, Some(Role::Assistant), Some("no thinking"), None, None),
            frame(0, None, None, None, Some(FinishReason::Stop)),
        ]);
        assert_eq!(out.len(), 3, "no reasoning → no boundary frame");
        assert!(out.iter().all(|r| delta(r).reasoning_content.is_none()));
    }

    #[test]
    fn splits_mixed_content_and_tool_calls() {
        // Mirrors the live mutual-exclusion failure: one frame carrying both
        // leftover content and the first tool-call chunk.
        #[allow(deprecated)]
        let tool_chunk = ChatCompletionMessageToolCallChunk {
            index: 0,
            id: Some("call_1".to_string()),
            r#type: Some(dynamo_protocols::types::FunctionType::Function),
            function: Some(dynamo_protocols::types::FunctionCallStream {
                name: Some("get_weather".to_string()),
                arguments: Some(String::new()),
            }),
        };
        let mut mixed = frame(0, Some(Role::Assistant), Some("lead-in"), None, None);
        mixed.data.as_mut().unwrap().inner.choices[0].delta.tool_calls =
            Some(vec![tool_chunk]);

        let out = collect(vec![mixed]);

        assert_eq!(out.len(), 3);
        assert_is_first_frame(&out[0], 0);
        assert!(delta(&out[1]).tool_calls.is_none());
        assert_eq!(
            delta(&out[1]).content,
            Some(ChatCompletionMessageContent::Text("lead-in".to_string()))
        );
        assert!(delta(&out[2]).content.is_none());
        assert_eq!(
            delta(&out[2]).tool_calls.as_ref().map(|calls| calls.len()),
            Some(1)
        );
    }

    #[test]
    fn explodes_complete_tool_calls_into_header_and_arguments_frames() {
        // The jail emits complete tool calls (id + name + full arguments) in a
        // single chunk; P0.13/P0.14 require header (arguments:"") + argument
        // continuation, and P1.9 requires per-index frames to stay contiguous.
        #[allow(deprecated)]
        fn complete_chunk(index: u32, name: &str, args: &str) -> ChatCompletionMessageToolCallChunk {
            ChatCompletionMessageToolCallChunk {
                index,
                id: Some(format!("call_{index}")),
                r#type: Some(dynamo_protocols::types::FunctionType::Function),
                function: Some(dynamo_protocols::types::FunctionCallStream {
                    name: Some(name.to_string()),
                    arguments: Some(args.to_string()),
                }),
            }
        }
        let mut jail_frame = frame(
            0,
            Some(Role::Assistant),
            None,
            None,
            Some(FinishReason::ToolCalls),
        );
        jail_frame.data.as_mut().unwrap().inner.choices[0].delta.tool_calls = Some(vec![
            complete_chunk(0, "get_weather", "{\"city\":\"SF\"}"),
            complete_chunk(1, "get_time", "{}"),
        ]);

        let out = collect(vec![jail_frame]);

        // first frame + (header+args) per call + end frame
        assert_eq!(out.len(), 6);
        assert_is_first_frame(&out[0], 0);
        let chunk_of = |response: &NvCreateChatCompletionStreamResponse| {
            delta(response).tool_calls.as_ref().expect("tool_calls")[0].clone()
        };
        let header0 = chunk_of(&out[1]);
        assert_eq!(header0.index, 0);
        assert_eq!(header0.id.as_deref(), Some("call_0"));
        assert_eq!(
            header0.function.as_ref().unwrap().arguments.as_deref(),
            Some("")
        );
        let args0 = chunk_of(&out[2]);
        assert_eq!(args0.index, 0);
        assert!(args0.id.is_none() && args0.r#type.is_none());
        assert!(args0.function.as_ref().unwrap().name.is_none());
        assert_eq!(
            args0.function.as_ref().unwrap().arguments.as_deref(),
            Some("{\"city\":\"SF\"}")
        );
        // Second call's frames are contiguous after the first call's.
        assert_eq!(chunk_of(&out[3]).index, 1);
        assert_eq!(chunk_of(&out[4]).index, 1);
        // End frame is a bare delta with the finish reason.
        assert_eq!(*delta(&out[5]), empty_delta());
        assert_eq!(
            out[5].inner.choices[0].finish_reason,
            Some(FinishReason::ToolCalls)
        );
        // Every frame carries at most one increment kind.
        for response in &out[1..5] {
            assert!(delta(response).content.is_none());
            assert!(delta(response).reasoning_content.is_none());
            assert_eq!(delta(response).tool_calls.as_ref().unwrap().len(), 1);
        }
    }

    #[test]
    fn drops_empty_frames_and_empty_increments() {
        let out = collect(vec![
            frame(0, Some(Role::Assistant), Some("hi"), None, None),
            // Role-only heartbeat after the first frame: dropped.
            frame(0, Some(Role::Assistant), None, None, None),
            // Empty-string increments: dropped.
            frame(0, None, Some(""), Some(""), None),
            frame(0, None, None, None, Some(FinishReason::Stop)),
        ]);

        assert_eq!(out.len(), 3);
        assert_is_first_frame(&out[0], 0);
        assert_eq!(
            delta(&out[1]).content,
            Some(ChatCompletionMessageContent::Text("hi".to_string()))
        );
        assert_eq!(
            out[2].inner.choices[0].finish_reason,
            Some(FinishReason::Stop)
        );
    }

    #[test]
    fn n_gt_1_gets_first_frame_per_candidate() {
        let out = collect(vec![
            frame(0, Some(Role::Assistant), Some("a"), None, None),
            frame(1, Some(Role::Assistant), Some("b"), None, None),
            frame(0, None, None, None, Some(FinishReason::Stop)),
            frame(1, None, None, None, Some(FinishReason::Stop)),
        ]);

        assert_eq!(out.len(), 6);
        assert_is_first_frame(&out[0], 0);
        assert_eq!(out[1].inner.choices[0].index, 0);
        assert_is_first_frame(&out[2], 1);
        assert_eq!(out[3].inner.choices[0].index, 1);
        assert!(out[4].inner.choices[0].finish_reason.is_some());
        assert!(out[5].inner.choices[0].finish_reason.is_some());
    }

    #[test]
    fn usage_and_nvext_ride_the_last_split_frame() {
        let mut finished = frame(
            0,
            Some(Role::Assistant),
            Some("hi"),
            None,
            Some(FinishReason::Stop),
        );
        {
            let data = finished.data.as_mut().unwrap();
            data.inner.usage = Some(Default::default());
            data.nvext = Some(serde_json::json!({"timing": {}}));
        }

        let out = collect(vec![finished]);

        assert_eq!(out.len(), 3);
        // First frame and content increment carry neither usage nor nvext.
        assert!(out[0].inner.usage.is_none() && out[0].nvext.is_none());
        assert!(out[1].inner.usage.is_none() && out[1].nvext.is_none());
        // The end frame carries both.
        assert!(out[2].inner.usage.is_some());
        assert!(out[2].nvext.is_some());
    }

    #[test]
    fn dropped_frames_carry_metrics_forward() {
        use crate::protocols::common::metrics::LLMMetricAnnotation;
        // A marker-only chunk parsed down to nothing still carries
        // chunk_tokens; dropping the frame must not lose them.
        let mut consumed = frame(0, Some(Role::Assistant), Some("hi"), None, None);
        consumed.data.as_mut().unwrap().llm_metrics = Some(LLMMetricAnnotation {
            chunk_tokens: 3,
            ..Default::default()
        });
        let mut dropped = frame(0, None, Some(""), None, None);
        dropped.data.as_mut().unwrap().llm_metrics = Some(LLMMetricAnnotation {
            chunk_tokens: 2,
            ..Default::default()
        });
        let mut end = frame(0, None, None, None, Some(FinishReason::Stop));
        end.data.as_mut().unwrap().llm_metrics = Some(LLMMetricAnnotation {
            chunk_tokens: 1,
            ..Default::default()
        });

        let out = collect(vec![consumed, dropped, end]);

        assert_eq!(out.len(), 3);
        let total: usize = out
            .iter()
            .filter_map(|r| r.llm_metrics.as_ref())
            .map(|m| m.chunk_tokens)
            .sum();
        assert_eq!(total, 6, "chunk_tokens from the dropped frame must survive");
    }

    #[test]
    fn choice_usage_rides_the_split_end_frame() {
        // P0.4: per-candidate usage attached by the delta generator to a
        // finish frame must land on the end frame after splitting, and never
        // on increment frames or synthesized first frames.
        let mut finished = frame(
            0,
            Some(Role::Assistant),
            Some("hi"),
            None,
            Some(FinishReason::Stop),
        );
        finished.data.as_mut().unwrap().choice_usage =
            Some(dynamo_protocols::types::CompletionUsage {
                completion_tokens: 7,
                ..Default::default()
            });

        let out = collect(vec![finished]);

        assert_eq!(out.len(), 3);
        assert!(out[0].choice_usage.is_none(), "first frame carries no usage");
        assert!(out[1].choice_usage.is_none(), "increment frame carries no usage");
        assert_eq!(
            out[2].choice_usage.as_ref().map(|u| u.completion_tokens),
            Some(7),
            "end frame carries the per-candidate usage"
        );
        // And it serializes as choices[0].usage on the wire.
        let json = serde_json::to_value(&out[2]).unwrap();
        assert_eq!(json["choices"][0]["usage"]["completion_tokens"], 7);
        // Increment frames keep no usage key.
        let json1 = serde_json::to_value(&out[1]).unwrap();
        assert!(json1["choices"][0].get("usage").is_none());
    }

    #[test]
    fn usage_only_summary_chunk_passes_through() {
        let mut summary = base_response();
        summary.inner.usage = Some(Default::default());
        let out = collect(vec![
            frame(0, Some(Role::Assistant), None, None, Some(FinishReason::Stop)),
            bare(summary),
        ]);

        // first frame + end frame + untouched summary
        assert_eq!(out.len(), 3);
        assert!(out[2].inner.choices.is_empty());
        assert!(out[2].inner.usage.is_some());
    }

    /// Fixture emitter for offline cross-validation against Moonshot's
    /// harness validators (see kimi-vv `harness/streaming`). No-op unless
    /// `WIRE_SHAPE_FIXTURE_DIR` is set; then each scenario's shaped stream is
    /// serialized with the real serde config, one SSE `data:` payload per
    /// line, into `<dir>/<scenario>.jsonl`.
    #[test]
    fn write_cross_validation_fixtures() {
        let Ok(dir) = std::env::var("WIRE_SHAPE_FIXTURE_DIR") else {
            return;
        };
        let dir = std::path::PathBuf::from(dir);
        std::fs::create_dir_all(&dir).expect("create fixture dir");

        #[allow(deprecated)]
        fn complete_chunk(index: u32, name: &str, args: &str) -> ChatCompletionMessageToolCallChunk {
            ChatCompletionMessageToolCallChunk {
                index,
                id: Some(format!("call_{index}")),
                r#type: Some(dynamo_protocols::types::FunctionType::Function),
                function: Some(dynamo_protocols::types::FunctionCallStream {
                    name: Some(name.to_string()),
                    arguments: Some(args.to_string()),
                }),
            }
        }

        // Each scenario mirrors what the upstream pipeline hands the shaper
        // today (role on every frame, jail-style complete tool chunks, ...).
        let mut scenarios: Vec<(&str, Vec<Frame>)> = vec![
            (
                "thinking_then_content",
                vec![
                    frame(0, Some(Role::Assistant), None, Some("Let me think"), None),
                    frame(0, Some(Role::Assistant), None, Some(" about it."), None),
                    frame(0, Some(Role::Assistant), Some("The"), None, None),
                    frame(0, Some(Role::Assistant), Some(" answer."), None, None),
                    frame(0, Some(Role::Assistant), None, None, Some(FinishReason::Stop)),
                ],
            ),
            (
                "mixed_increments_with_finish",
                vec![
                    frame(0, Some(Role::Assistant), Some("text"), Some("thought"), None),
                    frame(
                        0,
                        Some(Role::Assistant),
                        Some("tail"),
                        None,
                        Some(FinishReason::Stop),
                    ),
                ],
            ),
            (
                "n_gt_1",
                vec![
                    frame(0, Some(Role::Assistant), Some("alpha"), None, None),
                    frame(1, Some(Role::Assistant), Some("beta"), None, None),
                    frame(0, None, Some(" one"), None, None),
                    frame(1, None, Some(" two"), None, None),
                    frame(0, None, None, None, Some(FinishReason::Stop)),
                    frame(1, None, None, None, Some(FinishReason::Stop)),
                ],
            ),
        ];

        // Jail-style tool stream: role + empty content + complete tool chunks
        // + finish in a single frame.
        let mut jail_frame = frame(
            0,
            Some(Role::Assistant),
            Some(""),
            None,
            Some(FinishReason::ToolCalls),
        );
        jail_frame.data.as_mut().unwrap().inner.choices[0].delta.tool_calls = Some(vec![
            complete_chunk(0, "get_weather", "{\"city\":\"SF\"}"),
            complete_chunk(1, "get_time", "{\"tz\":\"PST\"}"),
        ]);
        scenarios.push(("jail_tool_calls", vec![jail_frame]));

        for (name, mut frames) in scenarios {
            // Mirror the delta generator: finish frames carry the
            // per-candidate usage snapshot (P0.4).
            for f in frames.iter_mut() {
                if let Some(data) = f.data.as_mut()
                    && data.inner.choices.first().is_some_and(|c| c.finish_reason.is_some())
                {
                    data.choice_usage = Some(dynamo_protocols::types::CompletionUsage {
                        prompt_tokens: 10,
                        completion_tokens: 5,
                        total_tokens: 15,
                        completion_tokens_details: Some(
                            dynamo_protocols::types::CompletionTokensDetails {
                                reasoning_tokens: Some(2),
                                ..Default::default()
                            },
                        ),
                        prompt_tokens_details: Some(
                            dynamo_protocols::types::PromptTokensDetails {
                                cached_tokens: Some(0),
                                ..Default::default()
                            },
                        ),
                        ..Default::default()
                    });
                }
            }
            let shaped = block_on(
                shape_chat_stream(stream::iter(frames)).collect::<Vec<_>>(),
            );
            let mut lines = String::new();
            for f in shaped {
                let data = f.data.expect("data frame");
                lines.push_str(&serde_json::to_string(&data).expect("serialize"));
                lines.push('\n');
            }
            std::fs::write(dir.join(format!("{name}.jsonl")), lines).expect("write fixture");
        }
    }

    #[test]
    fn error_annotations_pass_through_unshaped() {
        let error_frame: Frame = Annotated {
            data: None,
            id: None,
            event: Some("error".to_string()),
            comment: Some(vec!["boom".to_string()]),
            error: None,
        };
        let shaped = block_on(
            shape_chat_stream(stream::iter(vec![error_frame]))
                .collect::<Vec<_>>(),
        );
        assert_eq!(shaped.len(), 1);
        assert_eq!(shaped[0].event.as_deref(), Some("error"));
    }
}
