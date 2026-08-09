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
    ChatCompletionMessageToolCallChunk, ChatCompletionResponseContentPart,
    ChatCompletionResponseContentPartText, ChatCompletionStreamResponseDelta, ChatCompletionTool,
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

/// Complete, attribute-free Kimi K3 XTML structural tags.
///
/// Filtering the three bare delimiter tokens (`<|open|>`, `<|close|>`, and
/// `<|sep|>`) would silently alter assistant text that quotes the protocol.
/// Containment therefore removes only complete tags. Attribute-bearing open
/// tags (`call`, `argument`, `json`, and `message`) are recognized separately
/// after their attributes have been validated.
const K3_FIXED_STRUCTURAL_TAGS: &[&str] = &[
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

const K3_SEP: &str = "<|sep|>";
const K3_MAX_DYNAMIC_TAG_LEN: usize = 4096;
const K3_DYNAMIC_OPEN_TAGS: &[(&str, &str)] = &[
    ("call", "<|open|>call"),
    ("argument", "<|open|>argument"),
    ("json", "<|open|>json"),
    ("message", "<|open|>message"),
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum K3TagMatch {
    Complete(usize),
    Prefix,
    None,
}

/// Strips complete K3 structural tags from one streamed output channel.
///
/// Each API channel owns a separate instance. A tag can arrive one token (or
/// one byte) at a time, so a possible tag prefix is held until another delta
/// proves or completes it. Incomplete and near-marker text is released
/// byte-identical at end of stream; containment must not guess that it was a
/// control token merely because it began with `<|`.
#[derive(Default)]
struct K3StructuralTagFilter {
    pending: String,
}

impl K3StructuralTagFilter {
    fn push(&mut self, delta: &str) -> String {
        self.pending.push_str(delta);
        let mut out = String::new();
        loop {
            let Some(at) = self.pending.find('<') else {
                out.push_str(&self.pending);
                self.pending.clear();
                break;
            };
            if at != 0 {
                out.push_str(&self.pending[..at]);
                self.pending.drain(..at);
            }
            match classify_k3_structural_tag(&self.pending) {
                K3TagMatch::Complete(len) => {
                    self.pending.drain(..len);
                }
                K3TagMatch::Prefix => break,
                K3TagMatch::None => {
                    // Every candidate starts with the one-byte ASCII '<'.
                    out.push('<');
                    self.pending.drain(..1);
                }
            }
        }
        out
    }

    fn flush(&mut self) -> String {
        std::mem::take(&mut self.pending)
    }
}

fn classify_k3_structural_tag(text: &str) -> K3TagMatch {
    debug_assert!(text.starts_with('<'));

    for tag in K3_FIXED_STRUCTURAL_TAGS {
        if text.starts_with(tag) {
            return K3TagMatch::Complete(tag.len());
        }
        if tag.starts_with(text) {
            return K3TagMatch::Prefix;
        }
    }

    for &(name, prefix) in K3_DYNAMIC_OPEN_TAGS {
        if prefix.starts_with(text) {
            return K3TagMatch::Prefix;
        }
        let Some(after_name) = text.strip_prefix(prefix) else {
            continue;
        };
        if after_name.is_empty() {
            return K3TagMatch::Prefix;
        }
        if !matches!(after_name.as_bytes()[0], b' ' | b'\t') {
            continue;
        }
        let Some(sep_at) = text.find(K3_SEP) else {
            return if text.len() <= K3_MAX_DYNAMIC_TAG_LEN {
                K3TagMatch::Prefix
            } else {
                K3TagMatch::None
            };
        };
        if sep_at + K3_SEP.len() > K3_MAX_DYNAMIC_TAG_LEN {
            return K3TagMatch::None;
        }
        let attrs = &text[prefix.len()..sep_at];
        if valid_k3_open_tag(name, attrs) {
            return K3TagMatch::Complete(sep_at + K3_SEP.len());
        }
    }

    K3TagMatch::None
}

fn valid_k3_open_tag(name: &str, attrs: &str) -> bool {
    let Some(attrs) = parse_k3_attrs(attrs) else {
        return false;
    };
    let unique = |allowed: &[&str]| {
        attrs.iter().all(|(key, _)| allowed.contains(key))
            && attrs
                .iter()
                .enumerate()
                .all(|(index, (key, _))| !attrs[..index].iter().any(|(prior, _)| prior == key))
    };
    let value = |key: &str| {
        attrs
            .iter()
            .find_map(|(attr, value)| (*attr == key).then_some(*value))
    };
    let positive_index = |index: &str| {
        index.as_bytes().first().is_some_and(|digit| *digit != b'0')
            && index.bytes().all(|byte| byte.is_ascii_digit())
    };

    match name {
        "call" => {
            unique(&["tool", "index"])
                && attrs.len() == 2
                && value("tool").is_some_and(|tool| !tool.is_empty())
                && value("index").is_some_and(positive_index)
        }
        "argument" => {
            unique(&["key", "type"])
                && attrs.len() == 2
                && value("key").is_some_and(|key| !key.is_empty())
                && value("type").is_some_and(|kind| {
                    matches!(
                        kind,
                        "string" | "number" | "boolean" | "null" | "object" | "array"
                    )
                })
        }
        "json" => unique(&["type"]) && attrs.len() == 1 && value("type") == Some("object"),
        "message" => {
            let Some(role) = value("role") else {
                return false;
            };
            match role {
                "assistant" | "user" => {
                    unique(&["role", "name"])
                        && matches!(attrs.len(), 1 | 2)
                        && value("name").is_none_or(|name| !name.is_empty())
                }
                "system" => {
                    let name = value("name");
                    let kind = value("type");
                    unique(&["role", "name", "type"])
                        && matches!(attrs.len(), 1 | 2)
                        && name.is_none_or(|name| !name.is_empty())
                        && kind.is_none_or(|kind| !kind.is_empty())
                        && !(name.is_some() && kind.is_some())
                }
                "tool" => {
                    unique(&["role", "tool", "index"])
                        && attrs.len() == 3
                        && value("tool").is_some_and(|tool| !tool.is_empty())
                        && value("index").is_some_and(positive_index)
                }
                _ => false,
            }
        }
        _ => false,
    }
}

fn parse_k3_attrs(mut input: &str) -> Option<Vec<(&str, &str)>> {
    let mut attrs = Vec::new();
    while !input.is_empty() {
        let trimmed = input.trim_start_matches([' ', '\t']);
        if trimmed.len() == input.len() {
            return None;
        }
        input = trimmed;

        let name_len = input
            .bytes()
            .take_while(|byte| byte.is_ascii_alphanumeric() || *byte == b'_')
            .count();
        if name_len == 0 || !input[name_len..].starts_with("=\"") {
            return None;
        }
        let name = &input[..name_len];
        input = &input[name_len + 2..];
        let quote = input.find('"')?;
        let value = &input[..quote];
        if value.chars().any(char::is_control) {
            return None;
        }
        attrs.push((name, value));
        input = &input[quote + 1..];
    }
    Some(attrs)
}

fn contains_complete_k3_structural_tag(mut text: &str) -> bool {
    while let Some(at) = text.find('<') {
        text = &text[at..];
        if matches!(classify_k3_structural_tag(text), K3TagMatch::Complete(_)) {
            return true;
        }
        text = &text[1..];
    }
    false
}

fn validate_k3_json_strings(value: &serde_json::Value) -> Result<(), &'static str> {
    match value {
        serde_json::Value::String(text) => {
            if contains_complete_k3_structural_tag(text) {
                return Err("structural tag in an argument string value");
            }
            Ok(())
        }
        serde_json::Value::Array(values) => {
            for value in values {
                validate_k3_json_strings(value)?;
            }
            Ok(())
        }
        serde_json::Value::Object(values) => {
            if values
                .keys()
                .any(|key| contains_complete_k3_structural_tag(key))
            {
                return Err("structural tag in an argument key");
            }
            for value in values.values() {
                validate_k3_json_strings(value)?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

fn validate_k3_tool_arguments(arguments: &str) -> Result<(), &'static str> {
    let value: serde_json::Value =
        serde_json::from_str(arguments).map_err(|_| "invalid arguments JSON")?;
    if !value.is_object() {
        return Err("arguments JSON is not an object");
    }
    validate_k3_json_strings(&value)
}

#[derive(Default)]
struct K3OutputFilters {
    reasoning: K3StructuralTagFilter,
    content: K3StructuralTagFilter,
    /// Once a backend emits typed response parts, keep subsequently released
    /// text in that representation. The unary aggregator intentionally prefers
    /// Parts over scalar Text and would otherwise discard a held prefix when it
    /// is flushed at the end of the stream.
    content_as_parts: bool,
}

fn response_text(text: String, as_parts: bool) -> ChatCompletionMessageContent {
    if as_parts {
        ChatCompletionMessageContent::Parts(vec![ChatCompletionResponseContentPart::Text(
            ChatCompletionResponseContentPartText { text },
        )])
    } else {
        ChatCompletionMessageContent::Text(text)
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
    k3_output_filters: Option<K3OutputFilters>,
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
            // K3 is the only parser whose XTML tags need containment. Keep
            // content and reasoning state independent so a prefix in one API
            // channel can never consume bytes from another.
            k3_output_filters: (parser_name == "kimi_k3").then(K3OutputFilters::default),
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
            Ok(output) => output,
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
                let (text, as_parts) = match self.k3_output_filters.as_mut() {
                    Some(filters) => (filters.content.push(&text), filters.content_as_parts),
                    None => (text, false),
                };
                if text.is_empty() {
                    return None;
                }
                choice.delta.content = Some(response_text(text, as_parts));
            }
            UnifiedParserEvent::Reasoning(reasoning) => {
                let reasoning = match self.k3_output_filters.as_mut() {
                    Some(filters) => filters.reasoning.push(&reasoning),
                    None => reasoning,
                };
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

    /// Release incomplete candidate tags without sending them through the
    /// filter a second time. They were never proven structural and therefore
    /// remain literal output.
    fn flush_output_filters(&mut self) -> Vec<ChatChoiceStream> {
        let Some(filters) = self.k3_output_filters.as_mut() else {
            return Vec::new();
        };
        let reasoning = filters.reasoning.flush();
        let content = filters.content.flush();
        let mut choices = Vec::new();
        if !reasoning.is_empty() {
            let mut choice = empty_choice(self.choice_index);
            choice.delta.reasoning_content = Some(reasoning);
            choices.push(choice);
        }
        if !content.is_empty() {
            let mut choice = empty_choice(self.choice_index);
            choice.delta.content = Some(response_text(content, filters.content_as_parts));
            choices.push(choice);
        }
        choices
    }

    /// Filter text-bearing response parts while preserving every non-text part
    /// and its position. A structural tag may span adjacent text parts, but it
    /// cannot span an intervening image, video, or audio part, so pending text
    /// is released before a media boundary.
    fn filter_content_parts(
        &mut self,
        parts: Vec<ChatCompletionResponseContentPart>,
    ) -> Vec<ChatCompletionResponseContentPart> {
        let Some(filters) = self.k3_output_filters.as_mut() else {
            return parts;
        };
        filters.content_as_parts = true;

        let mut filtered = Vec::with_capacity(parts.len());
        for part in parts {
            match part {
                ChatCompletionResponseContentPart::Text(mut text_part) => {
                    text_part.text = filters.content.push(&text_part.text);
                    if !text_part.text.is_empty() {
                        filtered.push(ChatCompletionResponseContentPart::Text(text_part));
                    }
                }
                media => {
                    let pending = filters.content.flush();
                    if !pending.is_empty() {
                        filtered.push(ChatCompletionResponseContentPart::Text(
                            ChatCompletionResponseContentPartText { text: pending },
                        ));
                    }
                    filtered.push(media);
                }
            }
        }
        filtered
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
        if self.parser_name == "kimi_k3"
            && let Err(reason) = validate_k3_tool_arguments(&call.arguments)
        {
            tracing::warn!(
                choice_index = self.choice_index,
                tool_index = call.tool_index,
                reason,
                "suppressing unsafe K3 tool call"
            );
            return None;
        }
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
            choices.extend(state.flush_output_filters());
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
        Some(ChatCompletionMessageContent::Parts(parts)) => {
            let parts = state.filter_content_parts(parts);
            if !parts.is_empty() {
                source.delta.content = Some(ChatCompletionMessageContent::Parts(parts));
            }
        }
        content => source.delta.content = content,
    }
    if let Some(reasoning) = source.delta.reasoning_content.take() {
        output.push_reasoning(reasoning);
    }
    if finish_reason.is_some() {
        output.append(state.finish());
    }

    let mut emitted = output
        .events
        .into_iter()
        .filter_map(|event| state.event_choice(event))
        .collect::<Vec<_>>();

    // The unified K3 parser owns tool-call reconstruction. Accepting an
    // independently parsed backend field here would bypass both its state
    // machine and the complete-JSON/tag validation in `tool_call_chunk`.
    if state.parser_name == "kimi_k3" {
        if source.delta.tool_calls.take().is_some() {
            tracing::warn!(
                choice_index,
                parser = state.parser_name,
                "suppressing unexpected K3 passthrough tool_calls"
            );
        }
        #[allow(deprecated)]
        if source.delta.function_call.take().is_some() {
            tracing::warn!(
                choice_index,
                parser = state.parser_name,
                "suppressing unexpected K3 passthrough function_call"
            );
        }
    }

    let has_passthrough_delta = source.delta.role.is_some()
        || source.delta.content.is_some()
        || source.delta.tool_calls.is_some()
        || source.delta.function_call.is_some()
        || source.delta.refusal.is_some()
        || source.delta.reasoning_content.is_some();
    if has_passthrough_delta {
        if let Some(first) = emitted.first_mut() {
            if source.delta.role.is_some() {
                first.delta.role = source.delta.role.take();
            }
            if source.delta.function_call.is_some() {
                first.delta.function_call = source.delta.function_call.take();
            }
            if source.delta.refusal.is_some() {
                first.delta.refusal = source.delta.refusal.take();
            }
            if first.delta.content.is_none() {
                first.delta.content = source.delta.content.take();
            }
            if first.delta.tool_calls.is_none() {
                first.delta.tool_calls = source.delta.tool_calls.take();
            }
            if first.delta.reasoning_content.is_none() {
                first.delta.reasoning_content = source.delta.reasoning_content.take();
            }
        }
        let has_remaining_delta = source.delta.role.is_some()
            || source.delta.content.is_some()
            || source.delta.tool_calls.is_some()
            || source.delta.function_call.is_some()
            || source.delta.refusal.is_some()
            || source.delta.reasoning_content.is_some();
        if emitted.is_empty() || has_remaining_delta {
            emitted.push(source);
        }
    }

    // Flush after passthrough parts so a held literal prefix retains stream
    // order (`visible part`, then `pending prefix`) and both remain Parts for
    // non-streaming aggregation.
    if finish_reason.is_some() {
        emitted.extend(state.flush_output_filters());
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
mod k3_output_containment_tests {
    use dynamo_protocols::types::{
        ChatCompletionMessageToolCallChunk, ChatCompletionResponseContentPart,
        ChatCompletionResponseContentPartText, ChatCompletionStreamResponseDeltaFunctionCall,
        FinishReason, FunctionCallStream, FunctionType,
    };
    use vllm_parser::tool::{Tool, ToolCallDelta};
    use vllm_parser::unified::{
        UnifiedParser, UnifiedParserError, UnifiedParserEvent, UnifiedParserOutput,
    };
    use vllm_tokenizer::DynTokenizer;

    use super::{
        ChatCompletionMessageContent, ChoiceState, K3_FIXED_STRUCTURAL_TAGS, K3OutputFilters,
        K3StructuralTagFilter, empty_choice, process_choice, validate_k3_tool_arguments,
    };

    fn text_part(text: impl Into<String>) -> ChatCompletionResponseContentPart {
        ChatCompletionResponseContentPart::Text(ChatCompletionResponseContentPartText {
            text: text.into(),
        })
    }

    fn part_text(content: &ChatCompletionMessageContent) -> String {
        match content {
            ChatCompletionMessageContent::Text(text) => text.clone(),
            ChatCompletionMessageContent::Parts(parts) => parts
                .iter()
                .filter_map(|part| match part {
                    ChatCompletionResponseContentPart::Text(text) => Some(text.text.as_str()),
                    _ => None,
                })
                .collect(),
        }
    }

    /// Feed `text` through the filter in fixed-size character chunks (0 = one shot).
    fn run(text: &str, chunk: usize) -> String {
        let mut filter = K3StructuralTagFilter::default();
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

    #[test]
    fn strips_beam_drafted_turn_end_at_every_chunk_size() {
        let text = "在工作目录里的项目任务，都可以直接告诉我。\
                    <|close|>response<|sep|><|close|>message<|sep|>\
                    想让我帮您做点什么，直接说就行。";
        let want = "在工作目录里的项目任务，都可以直接告诉我。想让我帮您做点什么，直接说就行。";
        for chunk in [0usize, 1, 2, 3, 5, 7, 11] {
            assert_eq!(run(text, chunk), want, "chunk {chunk}");
        }
    }

    #[test]
    fn strips_deepswe_content_sequences_but_preserves_malformed_tail() {
        assert_eq!(
            run("<|close|>think<|sep|>Now I'll add the test.", 1),
            "Now I'll add the test."
        );
        let observed = "<|close|>call<|sep|><|close|>tools<|sep|>\
                        <|close|>message<|close|>```python stays literal";
        assert_eq!(
            run(observed, 1),
            "<|close|>message<|close|>```python stays literal"
        );
    }

    #[test]
    fn strips_every_fixed_tag_without_touching_surrounding_text() {
        for tag in K3_FIXED_STRUCTURAL_TAGS {
            let text = format!("head{tag}tail");
            for chunk in [0usize, 1, 2, 3, 7] {
                assert_eq!(run(&text, chunk), "headtail", "{tag} at chunk {chunk}");
            }
        }
    }

    #[test]
    fn strips_valid_attribute_tags_split_every_way() {
        let tags = [
            r#"<|open|>call tool="Bash" index="1"<|sep|>"#,
            r#"<|open|>argument key="command" type="string"<|sep|>"#,
            r#"<|open|>json type="object"<|sep|>"#,
            r#"<|open|>message role="assistant"<|sep|>"#,
            r#"<|open|>message role="assistant" name="coder"<|sep|>"#,
            r#"<|open|>message role="user" name="alice"<|sep|>"#,
            r#"<|open|>message role="system" name="policy"<|sep|>"#,
            r#"<|open|>message role="system" type="thinking-effort"<|sep|>"#,
            r#"<|open|>message role="tool" tool="Bash" index="1"<|sep|>"#,
        ];
        for tag in tags {
            let text = format!("head{tag}tail");
            for chunk in [0usize, 1, 2, 3, 7, 11] {
                assert_eq!(run(&text, chunk), "headtail", "{tag} at chunk {chunk}");
            }
        }
    }

    #[test]
    fn preserves_bare_partial_near_and_invalid_attribute_markers() {
        let literals = [
            "constants: <|open|>, <|close|>, and <|sep|>",
            "incomplete <|close|>mess",
            "near <|closer|>message<|sep|>",
            r#"example <|open|>call tool="Bash" index="0"<|sep|>"#,
            r#"example <|open|>argument key="x" type="imaginary"<|sep|>"#,
            r#"example <|open|>message role="developer"<|sep|>"#,
            r#"example <|open|>message role="user" name=""<|sep|>"#,
            r#"example <|open|>message role="assistant" unknown="x"<|sep|>"#,
            r#"example <|open|>message role="system" name="x" type="y"<|sep|>"#,
            "example <|open|>message\n role=\"user\"<|sep|>",
            "example <|open|>message role=\"user\"\n name=\"alice\"<|sep|>",
            "compare 1 < 2",
        ];
        for literal in literals {
            for chunk in [0usize, 1, 3, 7] {
                assert_eq!(run(literal, chunk), literal, "{literal} at chunk {chunk}");
            }
        }
    }

    #[test]
    fn content_and_reasoning_prefixes_never_share_state() {
        let mut filters = K3OutputFilters::default();
        assert_eq!(filters.content.push("<|close|>res"), "");
        assert_eq!(filters.reasoning.push("ponse<|sep|>"), "ponse<|sep|>");
        assert_eq!(filters.content.flush(), "<|close|>res");
        assert_eq!(filters.reasoning.flush(), "");
    }

    #[test]
    fn suppresses_observed_deepswe_bash_argument_instead_of_mutating_command() {
        // Same marker counts as the observed Bash leak: 2 open, 1 close, 3 sep.
        let arguments = r#"{"command":"<|open|>call tool=\"Bash\" index=\"1\"<|sep|><|open|>argument key=\"command\" type=\"string\"<|sep|>pytest -q<|close|>argument<|sep|>"}"#;
        assert!(validate_k3_tool_arguments(arguments).is_err());
    }

    #[test]
    fn tool_argument_guard_preserves_literals_and_rejects_unsafe_shapes() {
        let literal = r#"{"command":"print('<|open|>', '<|close|>', '<|sep|>', '<|close|>mess')"}"#;
        assert!(validate_k3_tool_arguments(literal).is_ok());
        assert!(validate_k3_tool_arguments("not-json").is_err());
        assert!(validate_k3_tool_arguments(r#"["not", "an", "object"]"#).is_err());
        assert!(validate_k3_tool_arguments(r#"{"<|close|>message<|sep|>":"value"}"#).is_err());
        assert!(
            validate_k3_tool_arguments(
                r#"{"text":"<|open|>message role=\"user\" name=\"alice\"<|sep|>"}"#
            )
            .is_err()
        );
    }

    struct FailingParser {
        recovered: String,
    }

    impl UnifiedParser for FailingParser {
        fn create(
            _tools: &[Tool],
            _tokenizer: DynTokenizer,
        ) -> vllm_parser::unified::Result<Box<dyn UnifiedParser>> {
            unreachable!("test parser is constructed directly")
        }

        fn parse_into(
            &mut self,
            _delta: &str,
            _output: &mut UnifiedParserOutput,
        ) -> vllm_parser::unified::Result<()> {
            Err(UnifiedParserError::ParsingFailed {
                message: "synthetic failure".to_string(),
            })
        }

        fn reset(&mut self) -> String {
            std::mem::take(&mut self.recovered)
        }
    }

    struct NoopParser;

    impl UnifiedParser for NoopParser {
        fn create(
            _tools: &[Tool],
            _tokenizer: DynTokenizer,
        ) -> vllm_parser::unified::Result<Box<dyn UnifiedParser>> {
            unreachable!("test parser is constructed directly")
        }

        fn parse_into(
            &mut self,
            _delta: &str,
            _output: &mut UnifiedParserOutput,
        ) -> vllm_parser::unified::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn content_parts_filter_complete_and_split_tags_without_empty_text_parts() {
        let mut state = ChoiceState::new(Box::new(NoopParser), 0, "kimi_k3");
        let filtered = state.filter_content_parts(vec![
            text_part("before<|close|>res"),
            text_part("ponse<|sep|>"),
            text_part("after"),
            text_part("<|close|>message<|sep|>"),
        ]);

        assert_eq!(filtered, vec![text_part("before"), text_part("after")]);
    }

    #[test]
    fn content_parts_flush_partial_text_before_media_and_preserve_media_order() {
        let media: Vec<ChatCompletionResponseContentPart> =
            serde_json::from_value(serde_json::json!([
                {"type": "image_url", "image_url": {"url": "https://example.test/i"}},
                {"type": "video_url", "video_url": {"url": "https://example.test/v"}},
                {"type": "audio_url", "audio_url": {"url": "https://example.test/a"}}
            ]))
            .unwrap();
        let mut input = vec![text_part("head<|close|>mess")];
        input.extend(media.clone());
        input.push(text_part("tail <|open|> <|close|> <|sep|>"));

        let mut state = ChoiceState::new(Box::new(NoopParser), 0, "kimi_k3");
        let filtered = state.filter_content_parts(input);

        assert_eq!(filtered[0], text_part("head"));
        assert_eq!(filtered[1], text_part("<|close|>mess"));
        assert_eq!(&filtered[2..5], media.as_slice());
        assert_eq!(filtered[5], text_part("tail <|open|> <|close|> <|sep|>"));
    }

    #[test]
    fn terminal_partial_from_content_parts_stays_parts_and_keeps_stream_order() {
        let literal = "visible<|close|>mess";
        let mut source = empty_choice(0);
        source.delta.content = Some(ChatCompletionMessageContent::Parts(vec![text_part(
            literal,
        )]));
        source.finish_reason = Some(FinishReason::Stop);
        let mut state = ChoiceState::new(Box::new(NoopParser), 0, "kimi_k3");

        let emitted = process_choice(source, &mut state);

        assert_eq!(emitted.len(), 2);
        assert!(emitted.iter().all(|choice| matches!(
            &choice.delta.content,
            Some(ChatCompletionMessageContent::Parts(_))
        )));
        assert_eq!(
            emitted
                .iter()
                .filter_map(|choice| choice.delta.content.as_ref())
                .map(part_text)
                .collect::<String>(),
            literal
        );
        assert_eq!(emitted[1].finish_reason, Some(FinishReason::Stop));
    }

    #[allow(deprecated)]
    fn passthrough_tool_delta() -> super::ChatChoiceStream {
        let mut source = empty_choice(0);
        source.delta.tool_calls = Some(vec![ChatCompletionMessageToolCallChunk {
            index: 0,
            id: Some("call-1".to_string()),
            r#type: Some(FunctionType::Function),
            function: Some(FunctionCallStream {
                name: Some("Bash".to_string()),
                arguments: Some(r#"{"command":"echo safe"}"#.to_string()),
            }),
        }]);
        source.delta.function_call = Some(ChatCompletionStreamResponseDeltaFunctionCall {
            name: Some("legacy".to_string()),
            arguments: Some("{}".to_string()),
        });
        source
    }

    #[test]
    fn k3_suppresses_unparsed_passthrough_tool_fields() {
        let mut state = ChoiceState::new(Box::new(NoopParser), 0, "kimi_k3");
        assert!(process_choice(passthrough_tool_delta(), &mut state).is_empty());
        assert!(!state.emitted_tool_calls);
    }

    #[test]
    #[allow(deprecated)]
    fn non_k3_preserves_passthrough_tool_fields() {
        let source = passthrough_tool_delta();
        let expected_tools = source.delta.tool_calls.clone();
        let expected_function = source.delta.function_call.clone();
        let mut state = ChoiceState::new(Box::new(NoopParser), 0, "other");

        let emitted = process_choice(source, &mut state);

        assert_eq!(emitted.len(), 1);
        assert_eq!(emitted[0].delta.tool_calls, expected_tools);
        assert_eq!(emitted[0].delta.function_call, expected_function);
    }

    #[test]
    fn parser_recovery_text_uses_the_same_containment_and_suppresses_empty_deltas() {
        let parser = FailingParser {
            recovered: "<|close|>think<|sep|>safe".to_string(),
        };
        let mut state = ChoiceState::new(Box::new(parser), 0, "kimi_k3");
        let output = state.process_delta("ignored".to_string());
        let choices = output
            .events
            .into_iter()
            .filter_map(|event| state.event_choice(event))
            .collect::<Vec<_>>();
        assert_eq!(choices.len(), 1);
        assert_eq!(
            choices[0].delta.content,
            Some(ChatCompletionMessageContent::Text("safe".to_string()))
        );
        assert!(
            state
                .event_choice(UnifiedParserEvent::Text(
                    "<|close|>response<|sep|>".to_string()
                ))
                .is_none()
        );
        assert!(
            state
                .event_choice(UnifiedParserEvent::Reasoning(
                    "<|close|>think<|sep|>".to_string()
                ))
                .is_none()
        );
    }

    struct SplitFinishParser;

    impl UnifiedParser for SplitFinishParser {
        fn create(
            _tools: &[Tool],
            _tokenizer: DynTokenizer,
        ) -> vllm_parser::unified::Result<Box<dyn UnifiedParser>> {
            unreachable!("test parser is constructed directly")
        }

        fn parse_into(
            &mut self,
            _delta: &str,
            output: &mut UnifiedParserOutput,
        ) -> vllm_parser::unified::Result<()> {
            output.push_text("<|close|>res");
            Ok(())
        }

        fn finish(&mut self) -> vllm_parser::unified::Result<UnifiedParserOutput> {
            let mut output = UnifiedParserOutput::default();
            output.push_text("ponse<|sep|>safe");
            Ok(output)
        }
    }

    #[test]
    fn marker_split_across_parser_finish_is_filtered_before_channel_flush() {
        let mut state = ChoiceState::new(Box::new(SplitFinishParser), 0, "kimi_k3");
        let first = state.process_delta("ignored".to_string());
        assert!(
            first
                .events
                .into_iter()
                .filter_map(|event| state.event_choice(event))
                .next()
                .is_none()
        );

        let final_output = state.finish();
        let mut choices = final_output
            .events
            .into_iter()
            .filter_map(|event| state.event_choice(event))
            .collect::<Vec<_>>();
        choices.extend(state.flush_output_filters());
        assert_eq!(choices.len(), 1);
        assert_eq!(
            choices[0].delta.content,
            Some(ChatCompletionMessageContent::Text("safe".to_string()))
        );
    }

    #[test]
    fn non_k3_parser_output_is_untouched() {
        let parser = FailingParser {
            recovered: String::new(),
        };
        let mut state = ChoiceState::new(Box::new(parser), 0, "other");
        let tag = "<|close|>response<|sep|>";
        let choice = state
            .event_choice(UnifiedParserEvent::Text(tag.to_string()))
            .unwrap();
        assert_eq!(
            choice.delta.content,
            Some(ChatCompletionMessageContent::Text(tag.to_string()))
        );
    }

    #[test]
    fn invalid_k3_tool_call_is_suppressed_before_emission() {
        let parser = FailingParser {
            recovered: String::new(),
        };
        let mut state = ChoiceState::new(Box::new(parser), 0, "kimi_k3");
        assert!(
            state
                .tool_call_chunk(ToolCallDelta {
                    tool_index: 0,
                    name: Some("Bash".to_string()),
                    arguments: "not-json".to_string(),
                })
                .is_none()
        );
        assert!(!state.emitted_tool_calls);
    }
}
