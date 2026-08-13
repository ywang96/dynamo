// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unified reasoning and tool-call output parsing.

pub(super) mod kimi_k3;

use std::collections::{HashMap, HashSet};
use std::fmt;
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
use serde::de::{DeserializeSeed, Deserializer as _, Error as _, MapAccess, SeqAccess, Visitor};
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

/// Channel names, for recognizing an open marker that lost its `<|open|>`.
///
/// Only meaningful at a channel boundary the parser just crossed -- see
/// `K3StructuralTagFilter::hold_bare_channel_open`. Mid-channel these are
/// ordinary words and must stay untouched.
const K3_CHANNEL_NAMES: &[&str] = &[
    "think", "response", "tools", "call", "argument", "json", "message",
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
/// proves or completes it.
#[derive(Default)]
struct K3StructuralTagFilter {
    pending: String,
    /// The parser just crossed into this channel, so a bare channel name here
    /// is the tail of an open marker rather than prose. Cleared by the first
    /// text that resolves either way.
    at_channel_start: bool,
}

impl K3StructuralTagFilter {
    /// Drop a channel name and separator whose `<|open|>` never arrived.
    ///
    /// Reaching a new channel means the parser consumed a *complete* marker to
    /// get here, so a leading `response<|sep|>` at that boundary is the tail of
    /// a second, malformed one -- the shape behind the reasoning-only stops of
    /// 2026-08-09. The same bytes mid-channel are ordinary text and are left
    /// alone, which is why this is keyed on an observed transition and never on
    /// start of stream: a non-thinking prompt ends at `<|open|>response<|sep|>`,
    /// so the first generated text was never preceded by a marker at all.
    ///
    /// Returns true while the candidate is still incomplete and must be held.
    fn hold_bare_channel_open(&mut self) -> bool {
        if self.pending.is_empty() {
            return true;
        }
        for name in K3_CHANNEL_NAMES {
            let Some(rest) = self.pending.strip_prefix(name) else {
                if name.starts_with(self.pending.as_str()) {
                    return true;
                }
                continue;
            };
            if let Some(after) = rest.strip_prefix(K3_SEP) {
                self.pending = after.to_string();
                self.at_channel_start = false;
                return false;
            }
            if K3_SEP.starts_with(rest) {
                return true;
            }
        }
        self.at_channel_start = false;
        false
    }

    fn push(&mut self, delta: &str) -> String {
        self.pending.push_str(delta);
        if self.at_channel_start && self.hold_bare_channel_open() {
            return String::new();
        }
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

    /// Release what is left, dropping marker debris.
    ///
    /// Whatever is still pending is a proper prefix of a structural tag by
    /// construction -- that is the only thing `push` holds -- so at end of
    /// stream it is a marker the model started and never finished, not text.
    /// A lone `<` is the exception: it is far more likely to be real output
    /// than the start of a tag, so it is kept.
    fn flush(&mut self) -> String {
        let pending = std::mem::take(&mut self.pending);
        if pending.len() >= 2 {
            String::new()
        } else {
            pending
        }
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

#[derive(Clone, Copy)]
struct K3JsonValueSeed;

impl<'de> DeserializeSeed<'de> for K3JsonValueSeed {
    type Value = ();

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_any(K3JsonValueVisitor)
    }
}

struct K3JsonValueVisitor;

impl<'de> Visitor<'de> for K3JsonValueVisitor {
    type Value = ();

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a JSON value without K3 structural tags")
    }

    fn visit_bool<E>(self, _value: bool) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_i64<E>(self, _value: i64) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_u64<E>(self, _value: u64) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_f64<E>(self, _value: f64) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        if contains_complete_k3_structural_tag(value) {
            return Err(E::custom("structural tag in an argument string value"));
        }
        Ok(())
    }

    fn visit_string<E>(self, value: String) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        self.visit_str(&value)
    }

    fn visit_none<E>(self) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_seq<A>(self, mut values: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        while values.next_element_seed(K3JsonValueSeed)?.is_some() {}
        Ok(())
    }

    fn visit_map<A>(self, values: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        validate_k3_json_map(values)
    }
}

fn validate_k3_json_map<'de, A>(mut values: A) -> Result<(), A::Error>
where
    A: MapAccess<'de>,
{
    let mut keys = HashSet::new();
    while let Some(key) = values.next_key::<String>()? {
        if contains_complete_k3_structural_tag(&key) {
            return Err(A::Error::custom("structural tag in an argument key"));
        }
        if !keys.insert(key.clone()) {
            return Err(A::Error::custom(format!("duplicate argument key {key:?}")));
        }
        values.next_value_seed(K3JsonValueSeed)?;
    }
    Ok(())
}

struct K3JsonObjectVisitor;

impl<'de> Visitor<'de> for K3JsonObjectVisitor {
    type Value = ();

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("an object-shaped arguments JSON value")
    }

    fn visit_map<A>(self, values: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        validate_k3_json_map(values)
    }
}

fn validate_k3_tool_arguments(arguments: &str) -> Result<(), String> {
    let mut deserializer = serde_json::Deserializer::from_str(arguments);
    deserializer
        .deserialize_map(K3JsonObjectVisitor)
        .map_err(|error| error.to_string())?;
    deserializer.end().map_err(|error| error.to_string())
}

fn k3_tool_recovery_should_be_suppressed(error: &dyn fmt::Display, recovered: &str) -> bool {
    let reason = error.to_string().to_ascii_lowercase();
    reason.contains("kimi k3 call")
        || reason.contains("tool call")
        || [
            "<|open|>tools",
            "<|close|>tools",
            "<|open|>call",
            "<|close|>call",
            "<|open|>argument",
            "<|close|>argument",
            "<|open|>json",
            "<|close|>json",
        ]
        .iter()
        .any(|marker| recovered.contains(marker))
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
    /// Channel the previous output belonged to. A change means the parser
    /// consumed a complete channel marker to get here.
    last_channel: Option<K3Channel>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum K3Channel {
    Reasoning,
    Content,
}

impl K3OutputFilters {
    /// Note which channel output is about to be written to.
    ///
    /// A change of channel means the parser consumed a complete marker to get
    /// here, so a channel name at that boundary is the tail of a second,
    /// malformed one.
    ///
    /// The first channel is a transition only when it is `Reasoning`. Thinking
    /// mode is the only way generation starts there — the prompt ends at
    /// `<|open|>think<|sep|>` — so a leading `response<|sep|>` in reasoning is
    /// unambiguously the marker the model failed to form. A stream that opens
    /// on `Content` is the non-thinking shape, where the prompt already ended
    /// at `<|open|>response<|sep|>`; no marker preceded that text and a leading
    /// channel name there is prose.
    fn enter(&mut self, channel: K3Channel) {
        let first = self.last_channel.is_none();
        let switched = self
            .last_channel
            .replace(channel)
            .is_some_and(|last| last != channel);
        if switched || (first && channel == K3Channel::Reasoning) {
            match channel {
                K3Channel::Reasoning => self.reasoning.at_channel_start = true,
                K3Channel::Content => self.content.at_channel_start = true,
            }
        }
    }
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
    suppress_parser_fallback: bool,
    emitted_tool_calls: bool,
    suppressed_tool_calls: bool,
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
            suppress_parser_fallback: false,
            emitted_tool_calls: false,
            suppressed_tool_calls: false,
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

    /// Return the backend's raw generated-token logprobs before parser filtering.
    fn take_pending_logprobs(&mut self) -> Option<ChatChoiceLogprobs> {
        self.pending_logprobs.take()
    }

    fn process_delta(&mut self, delta: String) -> UnifiedParserOutput {
        if self.parser_failed {
            if self.suppress_parser_fallback {
                return UnifiedParserOutput::default();
            }
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
            let suppress_recovery = self.parser_name == "kimi_k3"
                && k3_tool_recovery_should_be_suppressed(&error, &recovered);
            if suppress_recovery {
                self.suppressed_tool_calls = true;
                self.suppress_parser_fallback = true;
                tracing::warn!(
                    choice_index = self.choice_index,
                    parser = self.parser_name,
                    "suppressing K3 tool-protocol parser recovery"
                );
            } else if recovered.is_empty() && output.events.is_empty() {
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
                let recovered = self.parser.reset();
                let mut output = UnifiedParserOutput::default();
                if self.parser_name == "kimi_k3"
                    && k3_tool_recovery_should_be_suppressed(&error, &recovered)
                {
                    self.suppressed_tool_calls = true;
                    self.suppress_parser_fallback = true;
                    tracing::warn!(
                        choice_index = self.choice_index,
                        parser = self.parser_name,
                        "suppressing incomplete K3 tool call at end of stream"
                    );
                } else {
                    output.push_text(recovered);
                }
                output
            }
        }
    }

    fn event_choice(&mut self, event: UnifiedParserEvent) -> Option<ChatChoiceStream> {
        let mut choice = empty_choice(self.choice_index);
        match event {
            UnifiedParserEvent::Text(text) => {
                let (text, as_parts) = match self.k3_output_filters.as_mut() {
                    Some(filters) => {
                        filters.enter(K3Channel::Content);
                        let text = filters.content.push(&text);
                        (text, filters.content_as_parts)
                    }
                    None => (text, false),
                };
                if text.is_empty() {
                    return None;
                }
                choice.delta.content = Some(response_text(text, as_parts));
            }
            UnifiedParserEvent::Reasoning(reasoning) => {
                let reasoning = match self.k3_output_filters.as_mut() {
                    Some(filters) => {
                        filters.enter(K3Channel::Reasoning);
                        filters.reasoning.push(&reasoning)
                    }
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

    fn filter_passthrough_reasoning(&mut self, reasoning: String) -> Option<String> {
        let Some(filters) = self.k3_output_filters.as_mut() else {
            return Some(reasoning);
        };
        filters.enter(K3Channel::Reasoning);
        let reasoning = filters.reasoning.push(&reasoning);
        (!reasoning.is_empty()).then_some(reasoning)
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
        filters.enter(K3Channel::Content);
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
            if self.parser_name == "kimi_k3" {
                self.suppressed_tool_calls = true;
            }
            tracing::warn!(
                choice_index = self.choice_index,
                tool_index = call.tool_index,
                "tool index exceeds u32"
            );
            return None;
        };
        if self.parser_name == "kimi_k3" {
            let unsafe_name = call
                .name
                .as_deref()
                .is_some_and(contains_complete_k3_structural_tag);
            let invalid_arguments = validate_k3_tool_arguments(&call.arguments).err();
            if unsafe_name || invalid_arguments.is_some() {
                self.suppressed_tool_calls = true;
                tracing::warn!(
                    choice_index = self.choice_index,
                    tool_index = call.tool_index,
                    reason = invalid_arguments
                        .as_deref()
                        .unwrap_or("structural tag in tool name"),
                    "suppressing unsafe K3 tool call"
                );
                return None;
            }
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
        if self.parser_name == "kimi_k3"
            && id
                .as_deref()
                .is_some_and(contains_complete_k3_structural_tag)
        {
            self.suppressed_tool_calls = true;
            tracing::warn!(
                choice_index = self.choice_index,
                tool_index = call.tool_index,
                "suppressing K3 tool call with structural tag in id"
            );
            return None;
        }
        // Kimi streaming spec P2: tool-call ids are `{function_name}_{index}`
        // (e.g. `get_weather_0`); vLLM parsers emit `{name}:{index}`. Rewritten
        // after the structural-tag suppression above so suppression semantics
        // are unchanged.
        let id = id.map(spec_tool_call_id);
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

/// Kimi streaming spec P2: tool-call ids are `{function_name}_{global_index}`
/// (matching `^[A-Za-z_][A-Za-z0-9_]*_\d+$`). vLLM parsers generate
/// `{name}:{index}`; rewrite only a trailing `:<digits>` suffix so other id
/// shapes (e.g. the `call-{uuid}` fallback) pass through unchanged.
fn spec_tool_call_id(id: String) -> String {
    let Some(colon) = id.rfind(':') else {
        return id;
    };
    let (name, index) = id.split_at(colon);
    let digits = &index[1..];
    if name.is_empty() || digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return id;
    }
    format!("{name}_{digits}")
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
    /// P0.5: token ids behind increments the parser has swallowed but not yet
    /// re-emitted, in stream order. Attached to the first emitted frame that
    /// carries an increment (see `wire_shape::attach_token_ids_to_increment`).
    pending_token_ids: Vec<u32>,
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
            pending_token_ids: Vec::new(),
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
        mut response: Annotated<NvCreateChatCompletionStreamResponse>,
    ) -> Vec<Annotated<NvCreateChatCompletionStreamResponse>> {
        let Some(data) = response.data.as_mut() else {
            return vec![response];
        };
        // P0.5: buffer the ids behind this frame's increment; they attach to
        // whichever emitted frame finally carries the corresponding increment.
        if let Some(token_ids) = data.internal_token_ids.take() {
            self.pending_token_ids.extend(token_ids);
        }
        let source_choices = data.inner.choices.clone();
        self.last_response = Some(response.clone());

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

        let mut out = emit_choices(response, emitted);
        super::wire_shape::attach_token_ids_to_increment(&mut self.pending_token_ids, &mut out);
        out
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
                first.logprobs = state.take_pending_logprobs();
            }
            emitted.extend(choices);
            state.finished = true;
        }
        let mut out = emit_choices(response, emitted);
        super::wire_shape::attach_token_ids_to_increment(&mut self.pending_token_ids, &mut out);
        out
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
        source.delta.reasoning_content = state.filter_passthrough_reasoning(reasoning);
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
            state.suppressed_tool_calls = true;
            tracing::warn!(
                choice_index,
                parser = state.parser_name,
                "suppressing unexpected K3 passthrough tool_calls"
            );
        }
        #[allow(deprecated)]
        if source.delta.function_call.take().is_some() {
            state.suppressed_tool_calls = true;
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
            if let Some(reasoning) = source.delta.reasoning_content.take() {
                if let Some(parsed) = first.delta.reasoning_content.as_mut() {
                    let mut combined = reasoning;
                    combined.push_str(parsed);
                    *parsed = combined;
                } else {
                    first.delta.reasoning_content = Some(reasoning);
                }
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
        first.logprobs = state.take_pending_logprobs();
    }
    if let Some(mut finish_reason) = finish_reason {
        if finish_reason == FinishReason::Stop && state.emitted_tool_calls {
            finish_reason = FinishReason::ToolCalls;
        } else if state.suppressed_tool_calls
            && !state.emitted_tool_calls
            && matches!(
                finish_reason,
                FinishReason::ToolCalls | FinishReason::FunctionCall
            )
        {
            finish_reason = FinishReason::Stop;
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
    use std::sync::Arc;

    use dynamo_protocols::types::{
        ChatChoiceLogprobs, ChatCompletionMessageToolCallChunk, ChatCompletionResponseContentPart,
        ChatCompletionResponseContentPartText, ChatCompletionStreamResponseDeltaFunctionCall,
        ChatCompletionTokenLogprob, FinishReason, FunctionCallStream, FunctionType,
    };
    use vllm_parser::tool::{Tool, ToolCallDelta};
    use vllm_parser::unified::{
        KimiK3UnifiedParser, UnifiedParser, UnifiedParserError, UnifiedParserEvent,
        UnifiedParserOutput,
    };
    use vllm_tokenizer::DynTokenizer;
    use vllm_tokenizer::test_utils::TestTokenizer;

    use super::{
        ChatCompletionMessageContent, ChoiceState, K3_FIXED_STRUCTURAL_TAGS, K3Channel,
        K3OutputFilters, K3StructuralTagFilter, UnifiedOutputProcessor, UnifiedParserSpec,
        empty_choice, process_choice, spec_tool_call_id, validate_k3_tool_arguments,
    };
    use crate::protocols::openai::chat_completions::NvCreateChatCompletionStreamResponse;
    use dynamo_runtime::protocols::annotated::Annotated;

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

    fn actual_k3_parser() -> KimiK3UnifiedParser {
        let tokenizer = TestTokenizer::new()
            .with_special_token("<|open|>", 256)
            .with_special_token("<|close|>", 257)
            .with_special_token("<|sep|>", 258)
            .with_special_token("<|end_of_msg|>", 259);
        KimiK3UnifiedParser::new(&[], Arc::new(tokenizer)).unwrap()
    }

    fn token_logprobs(tokens: &[&str]) -> ChatChoiceLogprobs {
        ChatChoiceLogprobs {
            content: Some(
                tokens
                    .iter()
                    .map(|token| ChatCompletionTokenLogprob {
                        token: (*token).to_string(),
                        logprob: -0.25,
                        bytes: Some(token.as_bytes().to_vec()),
                        top_logprobs: Vec::new(),
                    })
                    .collect(),
            ),
            refusal: None,
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
        assert_eq!(filters.content.flush(), "");
        assert_eq!(filters.reasoning.flush(), "");
    }

    #[test]
    fn drops_a_marker_the_model_never_finished() {
        // Whatever survives to end of stream is a proper prefix of a structural
        // tag -- that is the only thing `push` holds -- so it is a marker the
        // model started and abandoned, not text. A lone `<` is the exception.
        for chunk in [0usize, 1, 3, 7] {
            assert_eq!(run("incomplete <|close|>mess", chunk), "incomplete ");
            assert_eq!(run("truncated <|open|>tools", chunk), "truncated ");
            assert_eq!(run("dangling <", chunk), "dangling <");
        }
    }

    #[test]
    fn strips_a_channel_open_that_lost_its_open_marker() {
        // The reasoning-only stops of 2026-08-09: the model closed think, then
        // emitted `response<|sep|>` with no `<|open|>`. Crossing into a channel
        // means a complete marker was consumed to get there, so a channel name
        // at that boundary is the tail of a second, malformed one.
        for chunk in [0usize, 1, 3, 7] {
            let mut filters = K3OutputFilters::default();
            filters.enter(K3Channel::Reasoning);
            assert_eq!(
                filters.reasoning.push("thinking out loud"),
                "thinking out loud"
            );

            filters.enter(K3Channel::Content);
            let text = "response<|sep|>the answer";
            let mut out = String::new();
            if chunk == 0 {
                out.push_str(&filters.content.push(text));
            } else {
                let chars: Vec<char> = text.chars().collect();
                for piece in chars.chunks(chunk) {
                    out.push_str(&filters.content.push(&piece.iter().collect::<String>()));
                }
            }
            out.push_str(&filters.content.flush());
            assert_eq!(out, "the answer", "chunk {chunk}");
        }
    }

    #[test]
    fn keeps_a_channel_name_that_is_ordinary_prose() {
        let mut filters = K3OutputFilters::default();
        filters.enter(K3Channel::Reasoning);
        assert_eq!(filters.reasoning.push("weighing it"), "weighing it");

        // Same word, same boundary, but no separator follows: plain text.
        filters.enter(K3Channel::Content);
        assert_eq!(
            filters.content.push("response times look fine"),
            "response times look fine"
        );

        // And mid-channel the sequence is never touched.
        assert_eq!(
            filters.content.push(" -- `response<|sep|>` is the marker"),
            " -- `response<|sep|>` is the marker"
        );
    }

    #[test]
    fn a_stream_opening_on_content_is_not_a_transition() {
        // A non-thinking prompt ends at `<|open|>response<|sep|>`, so the first
        // generated text was never preceded by a marker in the output stream.
        let mut filters = K3OutputFilters::default();
        filters.enter(K3Channel::Content);
        assert_eq!(filters.content.push("response<|sep|>"), "response<|sep|>");
    }

    #[test]
    fn a_stream_opening_on_reasoning_strips_a_bare_channel_open() {
        // Generation only starts in reasoning in thinking mode, where the
        // prompt ends at `<|open|>think<|sep|>`. A leading `response<|sep|>`
        // there is the marker the model failed to form -- observed in a user
        // session on 2026-08-09 as reasoning consisting of a bare channel name.
        for chunk in [0usize, 1, 3, 7] {
            let mut filters = K3OutputFilters::default();
            filters.enter(K3Channel::Reasoning);
            let text = "response<|sep|>weighing the options";
            let mut out = String::new();
            if chunk == 0 {
                out.push_str(&filters.reasoning.push(text));
            } else {
                let chars: Vec<char> = text.chars().collect();
                for piece in chars.chunks(chunk) {
                    out.push_str(&filters.reasoning.push(&piece.iter().collect::<String>()));
                }
            }
            out.push_str(&filters.reasoning.flush());
            assert_eq!(out, "weighing the options", "chunk {chunk}");
        }
    }

    #[test]
    fn a_stream_opening_on_reasoning_keeps_ordinary_prose() {
        let mut filters = K3OutputFilters::default();
        filters.enter(K3Channel::Reasoning);
        assert_eq!(
            filters.reasoning.push("response times look fine to me"),
            "response times look fine to me"
        );
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

        let overwritten_fixed = r#"{"command":"<|close|>message<|sep|>","command":"echo safe"}"#;
        assert!(validate_k3_tool_arguments(overwritten_fixed).is_err());
        let overwritten_dynamic =
            r#"{"command":"<|open|>message role=\"user\"<|sep|>","command":"echo safe"}"#;
        assert!(validate_k3_tool_arguments(overwritten_dynamic).is_err());
        assert!(validate_k3_tool_arguments(r#"{"x":1,"x":2}"#).is_err());
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

    struct EchoParser;

    impl UnifiedParser for EchoParser {
        fn create(
            _tools: &[Tool],
            _tokenizer: DynTokenizer,
        ) -> vllm_parser::unified::Result<Box<dyn UnifiedParser>> {
            Ok(Box::new(EchoParser))
        }

        fn parse_into(
            &mut self,
            delta: &str,
            output: &mut UnifiedParserOutput,
        ) -> vllm_parser::unified::Result<()> {
            output.push_text(delta.to_string());
            Ok(())
        }
    }

    struct ToolIdParser {
        id: String,
    }

    impl UnifiedParser for ToolIdParser {
        fn create(
            _tools: &[Tool],
            _tokenizer: DynTokenizer,
        ) -> vllm_parser::unified::Result<Box<dyn UnifiedParser>> {
            unreachable!("test parser is constructed directly")
        }

        fn tool_call_id(&self, _tool_index: usize) -> Option<&str> {
            Some(&self.id)
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
    fn suppressed_passthrough_tool_call_normalizes_separate_terminal_reason() {
        let mut state = ChoiceState::new(Box::new(NoopParser), 0, "kimi_k3");
        assert!(process_choice(passthrough_tool_delta(), &mut state).is_empty());

        let mut terminal = empty_choice(0);
        terminal.finish_reason = Some(FinishReason::ToolCalls);
        let emitted = process_choice(terminal, &mut state);

        assert_eq!(emitted.len(), 1);
        assert!(emitted[0].delta.tool_calls.is_none());
        assert_eq!(emitted[0].finish_reason, Some(FinishReason::Stop));
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

    #[test]
    fn actual_k3_parser_drops_incomplete_tool_body_at_terminal_eof() {
        let parser = actual_k3_parser();
        let mut state = ChoiceState::new(Box::new(parser), 0, "kimi_k3");
        let mut source = empty_choice(0);
        source.delta.content = Some(ChatCompletionMessageContent::Text(
            "<|open|>tools<|sep|>\
             <|open|>call tool=\"Bash\" index=\"1\"<|sep|>\
             <|open|>argument key=\"command\" type=\"string\"<|sep|>\
             pytest -q<|close|>argument<|sep|>"
                .to_string(),
        ));
        source.finish_reason = Some(FinishReason::Length);

        let emitted = process_choice(source, &mut state);

        assert_eq!(emitted.len(), 1);
        assert!(emitted[0].delta.content.is_none());
        assert!(emitted[0].delta.reasoning_content.is_none());
        assert!(emitted[0].delta.tool_calls.is_none());
        assert_eq!(emitted[0].finish_reason, Some(FinishReason::Length));
        assert!(state.suppressed_tool_calls);
    }

    #[test]
    fn actual_k3_parser_drops_malformed_complete_tool_body_on_parse_failure() {
        let parser = actual_k3_parser();
        let mut state = ChoiceState::new(Box::new(parser), 0, "kimi_k3");
        let output = state.process_delta(
            "<|open|>tools<|sep|>\
             <|open|>call tool=\"Bash\" index=\"1\"<|sep|>\
             not-an-argument<|close|>call<|sep|>"
                .to_string(),
        );

        assert!(output.events.is_empty());
        assert!(state.parser_failed);
        assert!(state.suppressed_tool_calls);

        state.append_logprobs(Some(token_logprobs(&["pytest", " -q"])));
        let followup = state.process_delta("pytest -q".to_string());
        assert!(followup.events.is_empty());
        assert_eq!(
            state
                .take_pending_logprobs()
                .and_then(|logprobs| logprobs.content)
                .expect("raw parser-failure logprobs")
                .into_iter()
                .map(|entry| entry.token)
                .collect::<Vec<_>>(),
            ["pytest", " -q"]
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
    fn non_k3_mixed_delta_and_logprobs_remain_byte_identical() {
        let tag = "<|close|>response<|sep|>";
        let expected_logprobs = token_logprobs(&["<|close|>", "response", "<|sep|>"]);
        let mut source = empty_choice(0);
        source.delta.content = Some(ChatCompletionMessageContent::Text(tag.to_string()));
        source.delta.reasoning_content = Some(tag.to_string());
        source.logprobs = Some(expected_logprobs.clone());
        let mut state = ChoiceState::new(Box::new(EchoParser), 0, "other");

        let emitted = process_choice(source, &mut state);

        assert_eq!(emitted.len(), 1);
        assert_eq!(
            emitted[0].delta.content,
            Some(ChatCompletionMessageContent::Text(tag.to_string()))
        );
        assert_eq!(emitted[0].delta.reasoning_content.as_deref(), Some(tag));
        assert_eq!(emitted[0].logprobs, Some(expected_logprobs));
    }

    #[test]
    fn k3_mixed_delta_keeps_reasoning_and_content_in_the_same_choice() {
        let mut source = empty_choice(0);
        source.delta.content = Some(ChatCompletionMessageContent::Text("answer".to_string()));
        source.delta.reasoning_content = Some("thought".to_string());
        let mut state = ChoiceState::new(Box::new(EchoParser), 0, "kimi_k3");

        let emitted = process_choice(source, &mut state);

        assert_eq!(emitted.len(), 1);
        assert_eq!(
            emitted[0].delta.reasoning_content.as_deref(),
            Some("thought")
        );
        assert_eq!(
            emitted[0].delta.content,
            Some(ChatCompletionMessageContent::Text("answer".to_string()))
        );
    }

    #[test]
    fn k3_structural_prefix_logprobs_keep_raw_positions() {
        let structural = [
            "<|close|>",
            "think",
            "<|sep|>",
            "<|open|>",
            "response",
            "<|sep|>",
        ];
        let mut state = ChoiceState::new(Box::new(NoopParser), 0, "kimi_k3");
        state.append_logprobs(Some(token_logprobs(&structural)));

        assert!(
            state
                .event_choice(UnifiedParserEvent::Reasoning(structural.concat()))
                .is_none()
        );

        state.append_logprobs(Some(token_logprobs(&["OK"])));
        let choice = state
            .event_choice(UnifiedParserEvent::Text("OK".to_string()))
            .expect("visible response token");
        assert_eq!(
            choice.delta.content,
            Some(ChatCompletionMessageContent::Text("OK".to_string()))
        );

        let tokens = state
            .take_pending_logprobs()
            .and_then(|logprobs| logprobs.content)
            .expect("raw generated-token logprobs")
            .into_iter()
            .map(|entry| entry.token)
            .collect::<Vec<_>>();
        assert_eq!(
            tokens,
            structural
                .into_iter()
                .chain(["OK"])
                .map(str::to_string)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn marker_split_across_held_prefix_preserves_raw_logprobs() {
        let mut state = ChoiceState::new(Box::new(NoopParser), 0, "kimi_k3");
        state.append_logprobs(Some(token_logprobs(&["<|close|>", "res"])));
        assert!(
            state
                .event_choice(UnifiedParserEvent::Text("<|close|>res".to_string()))
                .is_none()
        );

        state.append_logprobs(Some(token_logprobs(&["ponse", "<|sep|>", "safe"])));
        let choice = state
            .event_choice(UnifiedParserEvent::Text("ponse<|sep|>safe".to_string()))
            .unwrap();

        assert_eq!(
            choice.delta.content,
            Some(ChatCompletionMessageContent::Text("safe".to_string()))
        );
        assert_eq!(
            state
                .take_pending_logprobs()
                .and_then(|logprobs| logprobs.content)
                .expect("raw split-marker logprobs")
                .into_iter()
                .map(|entry| entry.token)
                .collect::<Vec<_>>(),
            ["<|close|>", "res", "ponse", "<|sep|>", "safe"]
        );
    }

    #[test]
    fn complete_structural_tag_in_sampled_logprobs_is_preserved() {
        let mut state = ChoiceState::new(Box::new(NoopParser), 0, "kimi_k3");
        state.append_logprobs(Some(token_logprobs(&["<|close|>", "message", "<|sep|>"])));
        assert_eq!(
            state
                .take_pending_logprobs()
                .and_then(|logprobs| logprobs.content)
                .expect("raw structural-tag logprobs")
                .into_iter()
                .map(|entry| entry.token)
                .collect::<Vec<_>>(),
            ["<|close|>", "message", "<|sep|>"]
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

    #[test]
    fn k3_tool_call_name_and_parser_id_cannot_carry_structural_tags() {
        let mut unsafe_name = ChoiceState::new(Box::new(NoopParser), 0, "kimi_k3");
        assert!(
            unsafe_name
                .tool_call_chunk(ToolCallDelta {
                    tool_index: 0,
                    name: Some("<|end_of_msg|>".to_string()),
                    arguments: "{}".to_string(),
                })
                .is_none()
        );
        assert!(unsafe_name.suppressed_tool_calls);

        let parser = ToolIdParser {
            id: "Bash:<|end_of_msg|>".to_string(),
        };
        let mut unsafe_id = ChoiceState::new(Box::new(parser), 0, "kimi_k3");
        assert!(
            unsafe_id
                .tool_call_chunk(ToolCallDelta {
                    tool_index: 0,
                    name: Some("Bash".to_string()),
                    arguments: "{}".to_string(),
                })
                .is_none()
        );
        assert!(unsafe_id.suppressed_tool_calls);
    }

    #[test]
    fn tool_call_id_rewrites_trailing_colon_index() {
        // P2: `{name}:{index}` from vLLM parsers becomes `{name}_{index}`.
        assert_eq!(
            spec_tool_call_id("get_weather:0".to_string()),
            "get_weather_0"
        );
        assert_eq!(spec_tool_call_id("ns:tool:12".to_string()), "ns:tool_12");
        // Only a trailing `:<digits>` suffix is rewritten; every other shape
        // (including the `call-{uuid}` fallback) passes through unchanged.
        assert_eq!(spec_tool_call_id("call-abc".to_string()), "call-abc");
        assert_eq!(
            spec_tool_call_id("get_weather:x".to_string()),
            "get_weather:x"
        );
        assert_eq!(spec_tool_call_id(":0".to_string()), ":0");
        assert_eq!(
            spec_tool_call_id("get_weather:".to_string()),
            "get_weather:"
        );
    }

    #[test]
    fn tool_call_chunk_emits_spec_shaped_id() {
        let parser = ToolIdParser {
            id: "get_weather:3".to_string(),
        };
        let mut state = ChoiceState::new(Box::new(parser), 0, "other");
        let chunk = state
            .tool_call_chunk(ToolCallDelta {
                tool_index: 3,
                name: Some("get_weather".to_string()),
                arguments: "{}".to_string(),
            })
            .expect("tool call chunk");
        assert_eq!(chunk.id.as_deref(), Some("get_weather_3"));
    }

    /// Parser that swallows every delta and regurgitates the buffered text at
    /// `finish`, exercising the token-id carry across re-chunked increments.
    struct BufferingParser {
        buffered: String,
    }

    impl UnifiedParser for BufferingParser {
        fn create(
            _tools: &[Tool],
            _tokenizer: DynTokenizer,
        ) -> vllm_parser::unified::Result<Box<dyn UnifiedParser>> {
            Ok(Box::new(Self {
                buffered: String::new(),
            }))
        }

        fn parse_into(
            &mut self,
            delta: &str,
            _output: &mut UnifiedParserOutput,
        ) -> vllm_parser::unified::Result<()> {
            self.buffered.push_str(delta);
            Ok(())
        }

        fn finish(&mut self) -> vllm_parser::unified::Result<UnifiedParserOutput> {
            let mut output = UnifiedParserOutput::default();
            output.push_text(std::mem::take(&mut self.buffered));
            Ok(output)
        }

        fn reset(&mut self) -> String {
            std::mem::take(&mut self.buffered)
        }
    }

    fn stream_response(
        choice: super::ChatChoiceStream,
        token_ids: Option<Vec<u32>>,
    ) -> Annotated<NvCreateChatCompletionStreamResponse> {
        Annotated::from_data(NvCreateChatCompletionStreamResponse {
            inner: dynamo_protocols::types::CreateChatCompletionStreamResponse {
                id: "chatcmpl-test".to_string(),
                object: "chat.completion.chunk".to_string(),
                created: 0,
                model: "test-model".to_string(),
                system_fingerprint: None,
                choices: vec![choice],
                usage: None,
                service_tier: None,
            },
            nvext: None,
            llm_metrics: None,
            choice_usage: None,
            internal_token_ids: token_ids,
        })
    }

    fn content_choice(text: &str) -> super::ChatChoiceStream {
        let mut choice = empty_choice(0);
        choice.delta.content = Some(ChatCompletionMessageContent::Text(text.to_string()));
        choice
    }

    #[test]
    fn internal_token_ids_follow_swallowed_increments() {
        // P0.5: ids behind increments the parser swallows must attach to the
        // frame that finally emits the corresponding increment.
        let spec = UnifiedParserSpec {
            name: "buffering",
            create: BufferingParser::create,
        };
        let tokenizer: DynTokenizer = Arc::new(TestTokenizer::new());
        let mut processor = UnifiedOutputProcessor::new(spec, vec![], tokenizer, vec![]).unwrap();

        // Swallowed by the parser: nothing emitted, ids buffered.
        let out = processor.process_response(stream_response(content_choice("hel"), Some(vec![1])));
        assert!(out.is_empty(), "buffering parser emits nothing mid-stream");

        // The finish frame flushes the buffered text; the emitted increment
        // carries the swallowed frame's ids plus this frame's own.
        let mut end = empty_choice(0);
        end.finish_reason = Some(FinishReason::Stop);
        let out = processor.process_response(stream_response(end, Some(vec![2])));
        let increment = out
            .iter()
            .filter_map(|a| a.data.as_ref())
            .find(|data| data.inner.choices.iter().any(|c| c.delta.content.is_some()))
            .expect("flushed increment frame");
        assert_eq!(
            increment.inner.choices[0].delta.content,
            Some(ChatCompletionMessageContent::Text("hel".to_string()))
        );
        assert_eq!(increment.internal_token_ids.as_deref(), Some(&[1, 2][..]));
        // And it serializes as choices[0].delta.internal_content on the wire.
        let json = serde_json::to_value(increment).unwrap();
        assert_eq!(
            json["choices"][0]["delta"]["internal_content"]["token_ids"],
            serde_json::json!([1, 2])
        );
    }

    #[test]
    fn internal_token_ids_pass_through_unswallowed_frames() {
        let spec = UnifiedParserSpec {
            name: "echo",
            create: EchoParser::create,
        };
        let tokenizer: DynTokenizer = Arc::new(TestTokenizer::new());
        let mut processor = UnifiedOutputProcessor::new(spec, vec![], tokenizer, vec![]).unwrap();

        let out = processor.process_response(stream_response(content_choice("hi"), Some(vec![7])));
        assert_eq!(out.len(), 1);
        assert_eq!(
            out[0].data.as_ref().unwrap().internal_token_ids.as_deref(),
            Some(&[7][..])
        );
    }
}
