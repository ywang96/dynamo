// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unified reasoning and tool-call output parsing.

pub(super) mod kimi_k3;

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;

use anyhow::Context as _;
use async_stream::stream;
use dynamo_protocols::types::{
    ChatChoiceLogprobs, ChatChoiceStream, ChatCompletionMessageContent,
    ChatCompletionMessageToolCallChunk, ChatCompletionStreamResponseDelta, ChatCompletionTool,
    FinishReason, FunctionCallStream, FunctionType,
};
use dynamo_runtime::protocols::annotated::Annotated;
use futures::{Stream, StreamExt as _};
use uuid::Uuid;
use vllm_parser::tool::{Tool, ToolCallDelta};
use vllm_parser::unified::{UnifiedParser, UnifiedParserEvent, UnifiedParserOutput};
use vllm_tokenizer::{DynTokenizer, TokenizerError};

use crate::protocols::openai::chat_completions::NvCreateChatCompletionStreamResponse;
use crate::tokenizers::traits::Tokenizer as DynamoTokenizer;

type UnifiedParserCreator =
    fn(&[Tool], DynTokenizer) -> vllm_parser::unified::Result<Box<dyn UnifiedParser>>;

type UnifiedOutputStream =
    Pin<Box<dyn Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send>>;

/// Static metadata and constructor for one vLLM unified parser.
#[derive(Clone, Copy)]
struct UnifiedParserSpec {
    name: &'static str,
    create: UnifiedParserCreator,
}

impl UnifiedParserSpec {
    /// Create and initialize one parser for a response choice.
    fn create_initialized(
        self,
        tools: &[Tool],
        tokenizer: DynTokenizer,
        prompt_token_ids: &[u32],
    ) -> anyhow::Result<Box<dyn UnifiedParser>> {
        let mut parser = (self.create)(tools, tokenizer)
            .with_context(|| format!("failed to create {} unified parser", self.name))?;
        parser
            .initialize(prompt_token_ids)
            .with_context(|| format!("failed to initialize {} unified parser", self.name))?;
        Ok(parser)
    }
}

/// Adapt Dynamo's tokenizer contract to the vLLM parser tokenizer contract.
struct VllmTokenizerAdapter {
    inner: Arc<dyn DynamoTokenizer>,
}

impl vllm_tokenizer::Tokenizer for VllmTokenizerAdapter {
    fn encode(&self, text: &str, _add_special_tokens: bool) -> vllm_tokenizer::Result<Vec<u32>> {
        self.inner
            .encode(text)
            .map(|encoding| encoding.token_ids().to_vec())
            .map_err(|error| TokenizerError(format!("{error:#}")))
    }

    fn encode_ordinary(&self, _text: &str) -> vllm_tokenizer::Result<Vec<u32>> {
        // The contract is `encode(text, false)` with every added, special, and
        // control-token matcher bypassed. Dynamo's `Encoder` exposes only
        // `encode(&str)`, which always applies those matchers, so there is no
        // faithful delegation: returning matched IDs would let prompt text forge
        // control tokens (K3 XTML markers in particular).
        //
        // This adapter only backs the unified output parser, and `vllm-parser`
        // has no `encode_ordinary` call site outside its own test doubles, so
        // this is unreachable today. Fail loudly rather than silently
        // mis-encoding if that ever changes.
        Err(TokenizerError(
            "encode_ordinary is not supported by the Dynamo tokenizer adapter: \
             Dynamo's Encoder cannot bypass special-token matching"
                .to_string(),
        ))
    }

    fn decode(
        &self,
        token_ids: &[u32],
        skip_special_tokens: bool,
    ) -> vllm_tokenizer::Result<String> {
        self.inner
            .decode(token_ids, skip_special_tokens)
            .map(String::from)
            .map_err(|error| TokenizerError(format!("{error:#}")))
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        let encoding = self.inner.encode(token).ok()?;
        let [token_id] = encoding.token_ids() else {
            return None;
        };
        Some(*token_id)
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.inner.decode(&[id], false).ok().map(String::from)
    }

    fn is_special_id(&self, token_id: u32) -> bool {
        let visible = self.inner.decode(&[token_id], false).ok().map(String::from);
        let stripped = self.inner.decode(&[token_id], true).ok().map(String::from);
        visible.is_some_and(|text| !text.is_empty()) && stripped.is_some_and(|text| text.is_empty())
    }
}

/// Complete Kimi K3 XTML channel markers that must never reach the client
/// inside `reasoning_content`.
///
/// By the time the unified parser emits a `Reasoning` event it has already
/// consumed the markers that delimit the think channel, so anything
/// marker-shaped still present is noise the model wrote *inside* its own
/// reasoning -- typically a drafted turn ending such as
/// `<|close|>response<|sep|><|close|>message<|sep|>`. Measured on the BEAM 1M
/// benchmark against a K3 endpoint: 9 of 117 answers (7.7%) carried that exact
/// sequence in `reasoning_content`; `content` was unaffected.
const K3_REASONING_NOISE: &[&str] = &[
    "<|open|>think<|sep|>",
    "<|close|>think<|sep|>",
    "<|open|>response<|sep|>",
    "<|close|>response<|sep|>",
    "<|open|>tools<|sep|>",
    "<|close|>tools<|sep|>",
    "<|close|>call<|sep|>",
    "<|close|>argument<|sep|>",
    "<|close|>json<|sep|>",
    "<|close|>message<|sep|>",
    "<|end_of_msg|>",
];

/// Strips [`K3_REASONING_NOISE`] from a streamed reasoning channel.
///
/// Stateful on purpose: a marker arrives split across deltas (one token at a
/// time in the worst case), so a per-delta match would pass every fragment
/// through. Text whose tail is a proper prefix of some marker is held back
/// until the next delta decides it, and [`Self::flush`] releases whatever is
/// still pending when the turn ends.
#[derive(Default)]
struct ReasoningNoiseFilter {
    pending: String,
}

impl ReasoningNoiseFilter {
    /// Longest suffix of `text` that is a proper prefix of some marker.
    fn held_suffix_len(text: &str) -> usize {
        let mut hold = 0;
        for marker in K3_REASONING_NOISE {
            // Proper prefixes only: a *complete* marker is removed by
            // `earliest_marker`, never held. Inclusive range -- `1..max` is
            // empty when only one byte has arrived, which would let a lone
            // leading `<` escape and defeat the whole hold-back.
            let max = (marker.len() - 1).min(text.len());
            for take in 1..=max {
                if text.is_char_boundary(text.len() - take)
                    && text[text.len() - take..] == marker[..take]
                {
                    hold = hold.max(take);
                }
            }
        }
        hold
    }

    fn earliest_marker(text: &str) -> Option<(usize, usize)> {
        K3_REASONING_NOISE
            .iter()
            .filter_map(|marker| text.find(marker).map(|at| (at, marker.len())))
            .min_by_key(|(at, _)| *at)
    }

    fn push(&mut self, delta: &str) -> String {
        self.pending.push_str(delta);
        let mut out = String::new();
        while let Some((at, len)) = Self::earliest_marker(&self.pending) {
            out.push_str(&self.pending[..at]);
            self.pending.drain(..at + len);
        }
        let hold = Self::held_suffix_len(&self.pending);
        let split = self.pending.len() - hold;
        out.push_str(&self.pending[..split]);
        self.pending.drain(..split);
        out
    }

    /// Release whatever is still held when the turn ends.
    ///
    /// What remains is by construction a proper prefix of some marker, so two
    /// or more bytes is marker debris the model was mid-way through and is
    /// dropped -- bounded by the longest marker. A single `<` is far more
    /// likely to be real text than the start of a marker, so it is kept.
    fn flush(&mut self) -> String {
        let pending = std::mem::take(&mut self.pending);
        if pending.len() >= 2 {
            String::new()
        } else {
            pending
        }
    }
}

/// One choice's parser and wire-emission state.
struct ChoiceState {
    choice_index: u32,
    parser_name: &'static str,
    parser: Box<dyn UnifiedParser>,
    parser_failed: bool,
    emitted_tool_calls: bool,
    pending_logprobs: Option<ChatChoiceLogprobs>,
    finished: bool,
    reasoning_noise: Option<ReasoningNoiseFilter>,
}

impl ChoiceState {
    fn new(parser: Box<dyn UnifiedParser>, choice_index: u32, parser_name: &'static str) -> Self {
        Self {
            choice_index,
            parser_name,
            parser,
            parser_failed: false,
            emitted_tool_calls: false,
            pending_logprobs: None,
            finished: false,
            // K3 is the only parser whose channel markers can appear inside a
            // reasoning delta; leave every other parser's stream untouched.
            reasoning_noise: (parser_name == "kimi_k3").then(ReasoningNoiseFilter::default),
        }
    }

    fn append_logprobs(&mut self, logprobs: Option<ChatChoiceLogprobs>) {
        let Some(logprobs) = logprobs else {
            return;
        };
        let pending = self.pending_logprobs.get_or_insert(ChatChoiceLogprobs {
            content: None,
            refusal: None,
        });
        if let Some(content) = logprobs.content {
            pending.content.get_or_insert_with(Vec::new).extend(content);
        }
        if let Some(refusal) = logprobs.refusal {
            pending.refusal.get_or_insert_with(Vec::new).extend(refusal);
        }
    }

    fn process_delta(&mut self, delta: String) -> UnifiedParserOutput {
        if self.parser_failed {
            let mut output = UnifiedParserOutput::default();
            output.push_text(delta);
            return output;
        }

        let mut output = UnifiedParserOutput::default();
        if let Err(error) = self.parser.parse_into(&delta, &mut output) {
            tracing::warn!(
                choice_index = self.choice_index,
                parser = self.parser_name,
                error = %error,
                "unified parser failed; falling back to visible text"
            );
            self.parser_failed = true;
            let recovered = self.parser.reset();
            if recovered.is_empty() && output.events.is_empty() {
                output.push_text(delta);
            } else {
                output.push_text(recovered);
            }
        }
        output
    }

    fn finish(&mut self) -> UnifiedParserOutput {
        if self.parser_failed {
            return UnifiedParserOutput::default();
        }

        match self.parser.finish() {
            Ok(mut output) => {
                if let Some(filter) = self.reasoning_noise.as_mut() {
                    output.push_reasoning(filter.flush());
                }
                output
            }
            Err(error) => {
                tracing::warn!(
                    choice_index = self.choice_index,
                    parser = self.parser_name,
                    error = %error,
                    "unified parser finish failed; recovering buffered text"
                );
                self.parser_failed = true;
                let mut output = UnifiedParserOutput::default();
                output.push_text(self.parser.reset());
                output
            }
        }
    }

    fn event_choice(&mut self, event: UnifiedParserEvent) -> Option<ChatChoiceStream> {
        let mut choice = empty_choice(self.choice_index);
        match event {
            UnifiedParserEvent::Text(text) => {
                choice.delta.content = Some(ChatCompletionMessageContent::Text(text));
            }
            UnifiedParserEvent::Reasoning(reasoning) => {
                let reasoning = match self.reasoning_noise.as_mut() {
                    Some(filter) => filter.push(&reasoning),
                    None => reasoning,
                };
                // Everything in this delta was marker or held back; emitting an
                // empty reasoning delta would be a visible no-op chunk.
                if reasoning.is_empty() {
                    return None;
                }
                choice.delta.reasoning_content = Some(reasoning);
            }
            UnifiedParserEvent::ToolCall(call) => {
                let chunk = self.tool_call_chunk(call)?;
                choice.delta.tool_calls = Some(vec![chunk]);
            }
        }
        Some(choice)
    }

    fn tool_call_chunk(
        &mut self,
        call: ToolCallDelta,
    ) -> Option<ChatCompletionMessageToolCallChunk> {
        let Ok(index) = u32::try_from(call.tool_index) else {
            tracing::warn!(
                choice_index = self.choice_index,
                tool_index = call.tool_index,
                "tool index exceeds u32"
            );
            return None;
        };
        if call.name.is_none() && call.arguments.is_empty() {
            return None;
        }

        let first_delta = call.name.is_some();
        let id = first_delta.then(|| {
            self.parser
                .tool_call_id(call.tool_index)
                .map(str::to_string)
                .unwrap_or_else(|| format!("call-{}", Uuid::new_v4()))
        });
        self.emitted_tool_calls = true;

        Some(ChatCompletionMessageToolCallChunk {
            index,
            id,
            r#type: first_delta.then_some(FunctionType::Function),
            function: Some(FunctionCallStream {
                name: call.name,
                arguments: Some(call.arguments),
            }),
        })
    }
}

/// Request-scoped unified parser factory and per-choice state.
struct UnifiedOutputProcessor {
    parser_spec: UnifiedParserSpec,
    tools: Arc<Vec<Tool>>,
    tokenizer: DynTokenizer,
    prompt_token_ids: Arc<Vec<u32>>,
    choices: HashMap<u32, ChoiceState>,
    spare_parser: Option<Box<dyn UnifiedParser>>,
    last_response: Option<Annotated<NvCreateChatCompletionStreamResponse>>,
}

impl UnifiedOutputProcessor {
    fn new(
        parser_spec: UnifiedParserSpec,
        tools: Vec<Tool>,
        tokenizer: DynTokenizer,
        prompt_token_ids: Vec<u32>,
    ) -> anyhow::Result<Self> {
        let parser =
            parser_spec.create_initialized(&tools, tokenizer.clone(), &prompt_token_ids)?;
        Ok(Self {
            parser_spec,
            tools: Arc::new(tools),
            tokenizer,
            prompt_token_ids: Arc::new(prompt_token_ids),
            choices: HashMap::new(),
            spare_parser: Some(parser),
            last_response: None,
        })
    }

    fn create_choice_state(&mut self, choice_index: u32) -> Option<ChoiceState> {
        let parser = if let Some(parser) = self.spare_parser.take() {
            parser
        } else {
            match self.parser_spec.create_initialized(
                &self.tools,
                self.tokenizer.clone(),
                &self.prompt_token_ids,
            ) {
                Ok(parser) => parser,
                Err(error) => {
                    tracing::warn!(
                        choice_index,
                        parser = self.parser_spec.name,
                        error = %error,
                        "failed to prepare per-choice unified parser"
                    );
                    return None;
                }
            }
        };
        Some(ChoiceState::new(
            parser,
            choice_index,
            self.parser_spec.name,
        ))
    }

    fn process_response(
        &mut self,
        response: Annotated<NvCreateChatCompletionStreamResponse>,
    ) -> Vec<Annotated<NvCreateChatCompletionStreamResponse>> {
        let Some(data) = response.data.as_ref() else {
            return vec![response];
        };
        self.last_response = Some(response.clone());

        let source_choices = data.inner.choices.clone();
        if source_choices.is_empty() {
            return vec![response];
        }

        let mut emitted = Vec::new();
        for choice in source_choices {
            let choice_index = choice.index;
            if !self.choices.contains_key(&choice_index) {
                let Some(state) = self.create_choice_state(choice_index) else {
                    emitted.push(choice);
                    continue;
                };
                self.choices.insert(choice_index, state);
            }
            let state = self
                .choices
                .get_mut(&choice_index)
                .expect("choice parser state was inserted");
            emitted.extend(process_choice(choice, state));
        }

        emit_choices(response, emitted)
    }

    fn finish_eof(&mut self) -> Vec<Annotated<NvCreateChatCompletionStreamResponse>> {
        let Some(response) = self.last_response.clone() else {
            return Vec::new();
        };
        let mut emitted = Vec::new();
        let mut choice_indexes = self.choices.keys().copied().collect::<Vec<_>>();
        choice_indexes.sort_unstable();
        for choice_index in choice_indexes {
            let state = self
                .choices
                .get_mut(&choice_index)
                .expect("choice parser state exists");
            debug_assert_eq!(choice_index, state.choice_index);
            if state.finished {
                continue;
            }
            let output = state.finish();
            let mut choices = output
                .events
                .into_iter()
                .filter_map(|event| state.event_choice(event))
                .collect::<Vec<_>>();
            if let Some(first) = choices.first_mut() {
                first.logprobs = state.pending_logprobs.take();
            }
            emitted.extend(choices);
            state.finished = true;
        }
        emit_choices(response, emitted)
    }
}

fn process_choice(mut source: ChatChoiceStream, state: &mut ChoiceState) -> Vec<ChatChoiceStream> {
    debug_assert_eq!(source.index, state.choice_index);
    let choice_index = state.choice_index;
    let finish_reason = source.finish_reason.take();
    state.append_logprobs(source.logprobs.take());

    let mut output = UnifiedParserOutput::default();
    match source.delta.content.take() {
        Some(ChatCompletionMessageContent::Text(text)) => {
            output.append(state.process_delta(text));
        }
        content => source.delta.content = content,
    }
    if finish_reason.is_some() {
        output.append(state.finish());
    }

    let mut emitted = output
        .events
        .into_iter()
        .filter_map(|event| state.event_choice(event))
        .collect::<Vec<_>>();

    let has_passthrough_delta = source.delta.role.is_some()
        || source.delta.content.is_some()
        || source.delta.tool_calls.is_some()
        || source.delta.function_call.is_some()
        || source.delta.refusal.is_some()
        || source.delta.reasoning_content.is_some();
    if has_passthrough_delta {
        if let Some(first) = emitted.first_mut() {
            first.delta.role = source.delta.role;
            first.delta.function_call = source.delta.function_call;
            first.delta.refusal = source.delta.refusal;
            if first.delta.content.is_none() {
                first.delta.content = source.delta.content;
            }
            if first.delta.tool_calls.is_none() {
                first.delta.tool_calls = source.delta.tool_calls;
            }
            if first.delta.reasoning_content.is_none() {
                first.delta.reasoning_content = source.delta.reasoning_content;
            }
        } else {
            emitted.push(source);
        }
    }

    if finish_reason.is_some() && emitted.is_empty() {
        emitted.push(empty_choice(choice_index));
    }
    if let Some(first) = emitted.first_mut() {
        first.logprobs = state.pending_logprobs.take();
    }
    if let Some(mut finish_reason) = finish_reason {
        if finish_reason == FinishReason::Stop && state.emitted_tool_calls {
            finish_reason = FinishReason::ToolCalls;
        }
        emitted
            .last_mut()
            .expect("terminal choice emission exists")
            .finish_reason = Some(finish_reason);
        state.finished = true;
    }

    emitted
}

fn empty_choice(index: u32) -> ChatChoiceStream {
    #[allow(deprecated)]
    ChatChoiceStream {
        index,
        delta: ChatCompletionStreamResponseDelta {
            content: None,
            function_call: None,
            tool_calls: None,
            role: None,
            refusal: None,
            reasoning_content: None,
        },
        finish_reason: None,
        logprobs: None,
    }
}

fn emit_choices(
    response: Annotated<NvCreateChatCompletionStreamResponse>,
    choices: Vec<ChatChoiceStream>,
) -> Vec<Annotated<NvCreateChatCompletionStreamResponse>> {
    if choices.is_empty() {
        return Vec::new();
    }
    let last = choices.len() - 1;
    choices
        .into_iter()
        .enumerate()
        .map(|(position, choice)| {
            let mut emitted = response.clone();
            let data = emitted
                .data
                .as_mut()
                .expect("source response contains data");
            data.inner.choices = vec![choice];
            if position != last {
                data.inner.usage = None;
                data.nvext = None;
                data.llm_metrics = None;
            }
            emitted
        })
        .collect()
}

fn convert_tools(tools: &[ChatCompletionTool]) -> Vec<Tool> {
    tools
        .iter()
        .map(|tool| Tool {
            name: tool.function.name.clone(),
            description: tool.function.description.clone(),
            parameters: tool
                .function
                .parameters
                .clone()
                .unwrap_or_else(|| serde_json::json!({"type": "object"})),
            strict: tool.function.strict,
        })
        .collect()
}

/// Parse one OpenAI delta stream through a vLLM unified parser.
fn unified_output_stream<S>(
    input: S,
    parser_spec: UnifiedParserSpec,
    tools: &[ChatCompletionTool],
    tokenizer: Arc<dyn DynamoTokenizer>,
    prompt_token_ids: &[u32],
) -> anyhow::Result<UnifiedOutputStream>
where
    S: Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send + 'static,
{
    let tokenizer: DynTokenizer = Arc::new(VllmTokenizerAdapter { inner: tokenizer });
    let mut processor = UnifiedOutputProcessor::new(
        parser_spec,
        convert_tools(tools),
        tokenizer,
        prompt_token_ids.to_vec(),
    )?;

    Ok(Box::pin(stream! {
        let mut input = Box::pin(input);
        while let Some(response) = input.next().await {
            for emitted in processor.process_response(response) {
                yield emitted;
            }
        }
        for emitted in processor.finish_eof() {
            yield emitted;
        }
    }))
}

#[cfg(test)]
mod reasoning_noise_tests {
    use super::{K3_REASONING_NOISE, ReasoningNoiseFilter};

    /// Feed `text` through the filter in fixed-size chunks (0 = one shot).
    fn run(text: &str, chunk: usize) -> String {
        let mut filter = ReasoningNoiseFilter::default();
        let mut out = String::new();
        if chunk == 0 {
            out.push_str(&filter.push(text));
        } else {
            let chars: Vec<char> = text.chars().collect();
            for piece in chars.chunks(chunk) {
                out.push_str(&filter.push(&piece.iter().collect::<String>()));
            }
        }
        out.push_str(&filter.flush());
        out
    }

    /// The shape reported from production and reproduced by BEAM 1M: the model
    /// drafts a turn ending inside its own reasoning.
    #[test]
    fn strips_drafted_turn_end_at_every_chunk_size() {
        let text = "在工作目录里的项目任务，都可以直接告诉我。\
                    <|close|>response<|sep|><|close|>message<|sep|>\
                    想让我帮您做点什么，直接说就行。\
                    <|close|>response<|sep|><|close|>message<|sep|>";
        let want = "在工作目录里的项目任务，都可以直接告诉我。想让我帮您做点什么，直接说就行。";
        for chunk in [0usize, 1, 2, 3, 5, 7, 11] {
            assert_eq!(run(text, chunk), want, "chunk {chunk}");
        }
    }

    /// Every marker, split every way, must be removed and nothing else lost.
    #[test]
    fn strips_every_marker_without_touching_surrounding_text() {
        for marker in K3_REASONING_NOISE {
            let text = format!("head{marker}tail");
            for chunk in [0usize, 1, 2, 3, 7] {
                assert_eq!(run(&text, chunk), "headtail", "{marker} at chunk {chunk}");
            }
        }
    }

    /// Ordinary reasoning must pass through byte-identical.
    #[test]
    fn leaves_plain_reasoning_untouched() {
        let text = "The user asks who I am. Respond briefly. 1 < 2 and a|b.";
        for chunk in [0usize, 1, 3, 7] {
            assert_eq!(run(text, chunk), text, "chunk {chunk}");
        }
    }

    /// A dangling partial marker at end of turn is dropped (bounded debris),
    /// but a lone `<` is kept -- far more likely to be real text.
    #[test]
    fn drops_dangling_marker_debris_but_keeps_a_lone_angle() {
        assert_eq!(run("answer<|close|>mess", 0), "answer");
        assert_eq!(run("compare 5 <", 0), "compare 5 <");
    }

    /// No panic and no lost bytes on adversarial marker soup.
    #[test]
    fn never_panics_on_marker_soup() {
        let soup = "<|<||>>|<|close|><|sep|>x<|open|>y<|end_of_msg|<|close|>message<|sep|";
        for chunk in [0usize, 1, 2, 3, 5] {
            let _ = run(soup, chunk);
        }
    }
}
