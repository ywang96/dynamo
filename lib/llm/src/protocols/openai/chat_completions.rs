// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use dynamo_runtime::protocols::annotated::{Annotated, AnnotationsProvider};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;
use validator::Validate;

use crate::engines::ValidateRequest;
use crate::preprocessor::media::MediaDecoder;

use super::{
    OpenAIOutputOptionsProvider, OpenAISamplingOptionsProvider, OpenAIStopConditionsProvider,
    common_ext::{CommonExt, CommonExtProvider},
    validate,
};
use crate::protocols::common::extensions::{
    NvExt, NvExtProvider, validate_completion_token_ids_single_choice,
};

pub mod aggregator;
mod delta;
pub mod tool_parser_v2;

pub use aggregator::DeltaAggregator;
pub use delta::DeltaGenerator;

use dynamo_parsers::tool_calling::{ToolCallResponse, ToolCallResponseChunk};
use dynamo_protocols::types::{
    ChatChoiceStream, ChatCompletionMessageContent, ChatCompletionMessageToolCall,
    ChatCompletionMessageToolCallChunk, ChatCompletionRequestMessage,
    ChatCompletionStreamResponseDelta, ChatCompletionTool, FinishReason, FunctionCall,
    FunctionCallStream, FunctionType,
};

/// Map a parser-native [`ToolCallResponse`] onto the protocol/wire
/// [`ChatCompletionMessageToolCall`].
///
/// `dynamo-parsers` is decoupled from `dynamo-protocols`, so this consumer —
/// which already depends on both — owns the mapping between the parser-native
/// types and the OpenAI wire types. The field shapes are identical, so this is
/// a straight re-map that preserves the previous wire output.
pub(crate) fn tool_call_response_to_protocol(
    parsed: ToolCallResponse,
) -> ChatCompletionMessageToolCall {
    ChatCompletionMessageToolCall {
        id: parsed.id,
        r#type: FunctionType::Function,
        function: FunctionCall {
            name: parsed.function.name,
            arguments: parsed.function.arguments,
        },
    }
}

/// Map a parser-native [`ToolCallResponseChunk`] onto the protocol/wire
/// [`ChatCompletionMessageToolCallChunk`]. See
/// [`tool_call_response_to_protocol`] for the rationale.
///
/// Exposed so consumers of the decoupled streaming parser entrypoint
/// ([`dynamo_parsers::tool_calling::try_tool_call_parse_stream`]) can recover
/// the wire type without `dynamo-parsers` depending on `dynamo-protocols`.
#[allow(dead_code)]
pub(crate) fn tool_call_response_chunk_to_protocol(
    parsed: ToolCallResponseChunk,
) -> ChatCompletionMessageToolCallChunk {
    ChatCompletionMessageToolCallChunk {
        index: parsed.index,
        id: parsed.id,
        r#type: parsed.tp.map(|_| FunctionType::Function),
        function: parsed.function.map(|f| FunctionCallStream {
            name: f.name,
            arguments: f.arguments,
        }),
    }
}

/// A request structure for creating a chat completion, extending OpenAI's
/// `CreateChatCompletionRequest` with [`NvExt`] extensions and common fields.
///
/// # Fields
/// - `inner`: The base OpenAI chat completion request, embedded using `serde(flatten)`.
/// - `common`: Common extension fields (ignore_eos, min_tokens) at root level, embedded using `serde(flatten)`.
/// - `nvext`: The optional NVIDIA extension field. See [`NvExt`] for more details.
///   Note: If ignore_eos is specified in both common and nvext, the common (root-level) value takes precedence.
#[derive(ToSchema, Serialize, Deserialize, Validate, Debug, Clone)]
pub struct NvCreateChatCompletionRequest {
    #[serde(flatten)]
    #[schema(value_type = Object)]
    pub inner: dynamo_protocols::types::CreateChatCompletionRequest,

    #[serde(flatten, default)]
    pub common: CommonExt,

    #[serde(skip_serializing_if = "Option::is_none")]
    #[schema(value_type = Object)]
    pub nvext: Option<NvExt>,

    /// Extra args to pass to the chat template rendering context
    /// Also accepts "chat_template_kwargs" as an alias for compatibility
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "chat_template_kwargs"
    )]
    pub chat_template_args: Option<std::collections::HashMap<String, serde_json::Value>>,

    /// OpenAI-style thinking control from client request payloads.
    /// Normalized into `chat_template_args` before preprocessing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<serde_json::Value>,

    /// Runtime media decoding parameters.
    /// When provided, these override the MDC defaults
    /// Example: `{"video": {"num_frames": 16}}`
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub media_io_kwargs: Option<MediaDecoder>,

    /// When true, logprob token fields are returned as "token_id:<id>" instead
    /// of decoded text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub return_tokens_as_token_ids: Option<bool>,

    /// Catch-all for unsupported fields - checked during validation
    #[serde(flatten, default, skip_serializing)]
    pub unsupported_fields: std::collections::HashMap<String, serde_json::Value>,
}

impl NvCreateChatCompletionRequest {
    /// Dynamic tools in message order.
    pub(crate) fn dynamic_tools(&self) -> impl Iterator<Item = &ChatCompletionTool> {
        self.inner
            .messages
            .iter()
            .flat_map(|message| match message {
                ChatCompletionRequestMessage::System(system) => {
                    system.tools.as_deref().unwrap_or_default()
                }
                _ => &[],
            })
    }

    /// Global and dynamic tools available to validation and output parsing.
    ///
    /// Prompt rendering still reads each system message directly, preserving
    /// the declaration's position in the tokenized context.
    pub(crate) fn effective_tools(&self) -> Vec<ChatCompletionTool> {
        self.inner
            .tools
            .iter()
            .flatten()
            .chain(self.dynamic_tools())
            .cloned()
            .collect()
    }

    /// Normalize OpenAI-style DS-V4 reasoning controls into the template kwargs
    /// consumed by the SGLang/DeepSeek-V4 prompt formatter.
    pub fn normalize_reasoning_template_args(&mut self) -> anyhow::Result<()> {
        let thinking_mode = self
            .thinking
            .as_ref()
            .map(openai_thinking_mode)
            .transpose()?
            .flatten();
        // spec §4: `thinking.effort` / `thinking.keep` apply only when thinking is
        // enabled and are IGNORED when disabled. Skip their validation entirely in
        // the disabled case, so an invalid effort/keep under `type: disabled` is
        // ignored rather than rejected with a 400.
        let is_disabled = matches!(thinking_mode, Some(OpenAiThinkingMode::Disabled));
        let thinking_effort = if is_disabled {
            None
        } else {
            self.thinking
                .as_ref()
                .map(openai_thinking_effort)
                .transpose()?
                .flatten()
        };
        let thinking_keep = if is_disabled {
            None
        } else {
            self.thinking
                .as_ref()
                .map(openai_thinking_keep)
                .transpose()?
                .flatten()
        };
        let reasoning_effort = self
            .inner
            .reasoning_effort
            .as_ref()
            .and_then(|effort| serde_json::to_value(effort).ok())
            .or(thinking_effort);
        // Kimi K3 renderer reads `thinking_effort` + `preserve_thinking` from the
        // chat-template args. Resolve the request-level effort with priority
        //   thinking.effort  >  OpenAI reasoning_effort
        // applied only when thinking is not disabled, so effort reaches the K3
        // renderer even when the request uses the OpenAI `reasoning_effort` field.
        // The spec §4 default (effort=max when enabled) is applied at insertion
        // time, and only when the caller pinned no explicit `thinking_effort`
        // template arg, so it never clobbers a user-supplied value. The renderer
        // validates the {low,medium,high,max} set and renders nothing otherwise.
        let thinking_effort_from_obj = self
            .thinking
            .as_ref()
            .and_then(|value| value.as_object())
            .and_then(|obj| obj.get("effort"))
            .and_then(|v| v.as_str())
            .map(str::to_string);
        let oai_reasoning_effort = self
            .inner
            .reasoning_effort
            .as_ref()
            .and_then(|effort| serde_json::to_value(effort).ok())
            .and_then(|v| v.as_str().map(str::to_string));
        let thinking_keep_raw = self
            .thinking
            .as_ref()
            .and_then(|value| value.as_object())
            .and_then(|obj| obj.get("keep").cloned());
        // Explicit request-level effort (thinking.effort > reasoning_effort);
        // None when thinking is disabled. Authoritative over template kwargs.
        // The `max` default is NOT folded in here — it is applied at insertion
        // time only when the caller supplied no `thinking_effort` template arg.
        let request_thinking_effort = match thinking_mode {
            Some(OpenAiThinkingMode::Disabled) => None,
            _ => thinking_effort_from_obj.or(oai_reasoning_effort),
        };

        if thinking_mode.is_none() && reasoning_effort.is_none() && thinking_keep.is_none() {
            return Ok(());
        }

        let args = self.chat_template_args.get_or_insert_with(HashMap::new);
        if let Some(mode) = thinking_mode {
            match mode {
                OpenAiThinkingMode::Enabled => {
                    args.insert("thinking".to_string(), serde_json::Value::Bool(true));
                    args.insert(
                        "thinking_mode".to_string(),
                        serde_json::Value::String("enabled".to_string()),
                    );
                }
                OpenAiThinkingMode::Disabled => {
                    args.insert("thinking".to_string(), serde_json::Value::Bool(false));
                    args.insert(
                        "thinking_mode".to_string(),
                        serde_json::Value::String("disabled".to_string()),
                    );
                }
            }
        }
        if let Some(effort) = reasoning_effort {
            args.entry("enable_thinking".to_string())
                .or_insert_with(|| serde_json::Value::Bool(effort.as_str() != Some("none")));
            args.insert("reasoning_effort".to_string(), effort);
        }
        if let Some(keep) = thinking_keep {
            args.insert("thinking_keep".to_string(), keep);
        }

        // Request-level `thinking.effort` / `thinking.keep` are authoritative:
        // inserted after the user-supplied chat_template_args merge above so
        // they overwrite any user-provided `thinking_effort`/`preserve_thinking`.
        // When the request carries no effort, fall back to the spec §4 default
        // (`max`, thinking enabled) ONLY when the caller did not already pin an
        // explicit `thinking_effort` template arg — never clobber their value.
        if let Some(effort) = request_thinking_effort {
            args.insert(
                "thinking_effort".to_string(),
                serde_json::Value::String(effort),
            );
        } else if matches!(thinking_mode, Some(OpenAiThinkingMode::Enabled))
            && !args.contains_key("thinking_effort")
        {
            args.insert(
                "thinking_effort".to_string(),
                serde_json::Value::String("max".to_string()),
            );
        }
        // K3 spec §4 keep normalization: `thinking.keep` is honored only when the
        // effective thinking type is enabled (i.e. NOT disabled) and is IGNORED
        // when disabled. When enabled, `interleaved` keeps only the most recent
        // reasoning (preserve_thinking = false); `all` or unset keeps all prior
        // reasoning (preserve_thinking = true). Skipping the disabled case is what
        // stops `thinking:{type:disabled, keep:all}` from wrongly preserving
        // history (the renderer applies preserve_thinking regardless of the
        // thinking flag). Unknown keep values are already rejected upstream by
        // openai_thinking_keep, so they never reach here.
        if !matches!(thinking_mode, Some(OpenAiThinkingMode::Disabled)) {
            let preserve_thinking = !matches!(
                thinking_keep_raw.as_ref().and_then(|v| v.as_str()),
                Some("interleaved")
            );
            args.insert(
                "preserve_thinking".to_string(),
                serde_json::Value::Bool(preserve_thinking),
            );
        }

        // The raw `thinking` payload has been folded into `chat_template_args`;
        // drop it so it isn't double-shipped downstream (and so it can't be
        // re-interpreted with different precedence by the worker preprocessor).
        self.thinking = None;
        Ok(())
    }
}

#[derive(Clone, Copy)]
enum OpenAiThinkingMode {
    Enabled,
    Disabled,
}

fn openai_thinking_effort(value: &serde_json::Value) -> anyhow::Result<Option<serde_json::Value>> {
    if value.as_bool().is_some() {
        return Ok(None);
    }

    let Some(thinking_object) = value.as_object() else {
        anyhow::bail!(
            "`thinking` must be a boolean or an object with `type` set to `enabled` or `disabled`"
        );
    };
    let Some(effort) = thinking_object.get("effort") else {
        return Ok(None);
    };
    if !effort.is_string() {
        anyhow::bail!("`thinking.effort` must be a string");
    }
    Ok(Some(effort.clone()))
}

fn openai_thinking_keep(value: &serde_json::Value) -> anyhow::Result<Option<serde_json::Value>> {
    if value.as_bool().is_some() {
        return Ok(None);
    }

    let Some(thinking_object) = value.as_object() else {
        anyhow::bail!(
            "`thinking` must be a boolean or an object with `type` set to `enabled` or `disabled`"
        );
    };
    let Some(keep) = thinking_object.get("keep") else {
        return Ok(None);
    };
    match keep.as_str() {
        Some("all" | "interleaved") => Ok(Some(keep.clone())),
        _ => anyhow::bail!("`thinking.keep` must be `all` or `interleaved`"),
    }
}

fn openai_thinking_mode(value: &serde_json::Value) -> anyhow::Result<Option<OpenAiThinkingMode>> {
    if let Some(enabled) = value.as_bool() {
        return Ok(Some(if enabled {
            OpenAiThinkingMode::Enabled
        } else {
            OpenAiThinkingMode::Disabled
        }));
    }

    let Some(thinking_object) = value.as_object() else {
        anyhow::bail!(
            "`thinking` must be a boolean or an object with `type` set to `enabled` or `disabled`"
        );
    };
    // spec §4: `thinking.type` is optional and defaults to `enabled`. An object
    // with no `type` (e.g. only `keep`/`effort`) is enabled thinking; a present
    // but unrecognized value (e.g. `adaptive`) or non-string is still rejected.
    let Some(thinking_type) = thinking_object.get("type") else {
        return Ok(Some(OpenAiThinkingMode::Enabled));
    };
    match thinking_type.as_str() {
        Some("enabled") => Ok(Some(OpenAiThinkingMode::Enabled)),
        Some("disabled") => Ok(Some(OpenAiThinkingMode::Disabled)),
        _ => anyhow::bail!("`thinking.type` must be `enabled` or `disabled`"),
    }
}

/// A response structure for unary chat completion responses, embedding OpenAI's
/// `CreateChatCompletionResponse` with optional NVIDIA extension metadata.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct NvCreateChatCompletionResponse {
    #[serde(flatten)]
    pub inner: dynamo_protocols::types::CreateChatCompletionResponse,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nvext: Option<serde_json::Value>,
}

/// A response structure for streamed chat completions, embedding OpenAI's
/// `CreateChatCompletionStreamResponse` with optional NVIDIA extension metadata.
///
/// `Serialize` is implemented manually: tool-call continuation chunks (index +
/// arguments only, produced by the wire-shape stage) must not serialize their
/// unset `id`/`type`/`function.name` as JSON nulls — the streaming spec allows
/// only `index` and `function.arguments` keys on continuations — and the chunk
/// types live in the external `dynamo-protocols` crate where we cannot add
/// `skip_serializing_if`. Frames without sparse chunks take a zero-cost path
/// identical to the former derive.
#[derive(Deserialize, Debug, Clone, PartialEq)]
pub struct NvCreateChatCompletionStreamResponse {
    #[serde(flatten)]
    pub inner: dynamo_protocols::types::CreateChatCompletionStreamResponse,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nvext: Option<serde_json::Value>,
    /// Internal frontend metrics payload. This must never be serialized to
    /// client-facing OpenAI-compatible streams.
    #[serde(skip)]
    pub llm_metrics: Option<crate::protocols::common::metrics::LLMMetricAnnotation>,
    /// Per-candidate usage for this frame's (single) choice, populated on the
    /// candidate's end frame (Kimi streaming spec P0.4 / §5.5). Carried out of
    /// band because `ChatChoiceStream` lives in the external dynamo-protocols
    /// crate (also constructed by dynamo-parsers, so adding a field there
    /// breaks external constructors); the manual `Serialize` impl below
    /// injects it as `choices[0].usage` on the wire.
    #[serde(skip)]
    pub choice_usage: Option<dynamo_protocols::types::CompletionUsage>,
    /// Token ids behind this frame's increment (Kimi streaming spec P0.5,
    /// `stream_options.include_internal_content`). Only set when the request
    /// opted in; carried out of band for the same external-type reason as
    /// `choice_usage` and injected by the manual `Serialize` impl as
    /// `choices[0].delta.internal_content.token_ids`.
    #[serde(skip)]
    pub internal_token_ids: Option<Vec<crate::types::TokenIdType>>,
}

impl Serialize for NvCreateChatCompletionStreamResponse {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        #[derive(Serialize)]
        struct Shadow<'a> {
            #[serde(flatten)]
            inner: &'a dynamo_protocols::types::CreateChatCompletionStreamResponse,
            #[serde(skip_serializing_if = "Option::is_none")]
            nvext: &'a Option<serde_json::Value>,
        }

        let shadow = Shadow {
            inner: &self.inner,
            nvext: &self.nvext,
        };

        let has_sparse_tool_call_chunks = self.inner.choices.iter().any(|choice| {
            choice.delta.tool_calls.iter().flatten().any(|chunk| {
                chunk.id.is_none()
                    || chunk.r#type.is_none()
                    || chunk
                        .function
                        .as_ref()
                        .is_some_and(|function| function.name.is_none())
            })
        });
        if !has_sparse_tool_call_chunks
            && self.choice_usage.is_none()
            && self.internal_token_ids.is_none()
        {
            return shadow.serialize(serializer);
        }

        let mut value = serde_json::to_value(&shadow).map_err(serde::ser::Error::custom)?;
        // Streaming spec P0.5: increment frames carry the token ids behind
        // their increment as `choices[0].delta.internal_content.token_ids`.
        // The ids live out of band on this wrapper (see `internal_token_ids`)
        // because `ChatCompletionStreamResponseDelta` is an external type.
        if let Some(token_ids) = &self.internal_token_ids
            && !token_ids.is_empty()
            && let Some(choice) = value
                .get_mut("choices")
                .and_then(|v| v.as_array_mut())
                .and_then(|choices| choices.first_mut())
            && let Some(delta) = choice.get_mut("delta").and_then(|d| d.as_object_mut())
        {
            delta.insert(
                "internal_content".to_string(),
                serde_json::json!({ "token_ids": token_ids }),
            );
        }
        // Streaming spec P0.4 / §5.5: the candidate's end frame carries
        // `choices[0].usage`. The field lives out of band on this wrapper
        // (see `choice_usage`) because `ChatChoiceStream` is an external type.
        if let Some(choice_usage) = &self.choice_usage
            && let Some(choice) = value
                .get_mut("choices")
                .and_then(|v| v.as_array_mut())
                .and_then(|choices| choices.first_mut())
            && let Some(obj) = choice.as_object_mut()
        {
            obj.insert(
                "usage".to_string(),
                serde_json::to_value(choice_usage).map_err(serde::ser::Error::custom)?,
            );
        }
        if let Some(choices) = value.get_mut("choices").and_then(|v| v.as_array_mut()) {
            for choice in choices {
                let Some(tool_calls) = choice
                    .get_mut("delta")
                    .and_then(|d| d.get_mut("tool_calls"))
                    .and_then(|v| v.as_array_mut())
                else {
                    continue;
                };
                for chunk in tool_calls {
                    if let Some(obj) = chunk.as_object_mut() {
                        obj.retain(|key, v| !(v.is_null() && matches!(key.as_str(), "id" | "type")));
                        if let Some(function) = obj.get_mut("function").and_then(|f| f.as_object_mut())
                        {
                            function.retain(|key, v| !(v.is_null() && key == "name"));
                        }
                    }
                }
            }
        }
        value.serialize(serializer)
    }
}

/// Build one synthetic stream choice from an existing response template.
///
/// Both streaming tool-call paths use this constructor when an engine omits a
/// terminal choice. Accounting data belongs only on the usage chunk and must
/// not be copied onto the synthetic choice.
pub(super) fn stream_choice_chunk_from_template(
    template: &NvCreateChatCompletionStreamResponse,
    index: u32,
    content: Option<ChatCompletionMessageContent>,
    tool_calls: Option<Vec<ChatCompletionMessageToolCallChunk>>,
    finish_reason: Option<FinishReason>,
) -> Annotated<NvCreateChatCompletionStreamResponse> {
    let mut response = template.clone();
    response.inner.usage = None;
    response.llm_metrics = None;
    response.internal_token_ids = None;
    #[allow(deprecated)]
    let choice = ChatChoiceStream {
        index,
        delta: ChatCompletionStreamResponseDelta {
            role: None,
            content,
            tool_calls,
            function_call: None,
            refusal: None,
            reasoning_content: None,
        },
        finish_reason,
        logprobs: None,
    };
    response.inner.choices = vec![choice];
    Annotated {
        data: Some(response),
        id: None,
        event: None,
        comment: None,
        error: None,
    }
}

/// Implements `NvExtProvider` for `NvCreateChatCompletionRequest`,
/// providing access to NVIDIA-specific extensions.
impl NvExtProvider for NvCreateChatCompletionRequest {
    /// Returns a reference to the optional `NvExt` extension, if available.
    fn nvext(&self) -> Option<&NvExt> {
        self.nvext.as_ref()
    }

    /// Returns `None`, as raw prompt extraction is not implemented.
    fn raw_prompt(&self) -> Option<String> {
        None
    }

    fn unsupported_fields(&self) -> Option<&std::collections::HashMap<String, serde_json::Value>> {
        Some(&self.unsupported_fields)
    }
}

/// Implements `AnnotationsProvider` for `NvCreateChatCompletionRequest`,
/// enabling retrieval and management of request annotations.
impl AnnotationsProvider for NvCreateChatCompletionRequest {
    /// Retrieves the list of annotations from `NvExt`, if present.
    fn annotations(&self) -> Option<Vec<String>> {
        self.nvext
            .as_ref()
            .and_then(|nvext| nvext.annotations.clone())
    }

    /// Checks whether a specific annotation exists in the request.
    fn has_annotation(&self, annotation: &str) -> bool {
        self.nvext
            .as_ref()
            .and_then(|nvext| nvext.annotations.as_ref())
            .map(|annotations| annotations.contains(&annotation.to_string()))
            .unwrap_or(false)
    }
}

/// Implements `OpenAISamplingOptionsProvider` for `NvCreateChatCompletionRequest`,
/// exposing OpenAI's sampling parameters for chat completion.
impl OpenAISamplingOptionsProvider for NvCreateChatCompletionRequest {
    /// Retrieves the temperature parameter for sampling, if set.
    fn get_temperature(&self) -> Option<f32> {
        self.inner.temperature
    }

    /// Retrieves the top-p (nucleus sampling) parameter, if set.
    fn get_top_p(&self) -> Option<f32> {
        self.inner.top_p
    }

    /// Retrieves the frequency penalty parameter, if set.
    fn get_frequency_penalty(&self) -> Option<f32> {
        self.inner.frequency_penalty
    }

    /// Retrieves the presence penalty parameter, if set.
    fn get_presence_penalty(&self) -> Option<f32> {
        self.inner.presence_penalty
    }

    /// Returns a reference to the optional `NvExt` extension, if available.
    fn nvext(&self) -> Option<&NvExt> {
        self.nvext.as_ref()
    }
    /// Retrieves the seed value for random number generation, if set.
    fn get_seed(&self) -> Option<i64> {
        self.inner.seed
    }

    /// Retrieves the number of completions to generate for each prompt, if set.
    fn get_n(&self) -> Option<u8> {
        self.inner.n
    }

    /// Retrieves the best_of parameter, if set.
    fn get_best_of(&self) -> Option<u8> {
        None // Not supported in chat completions
    }
}

/// Implements `CommonExtProvider` for `NvCreateChatCompletionRequest`,
/// providing access to common extension fields.
impl CommonExtProvider for NvCreateChatCompletionRequest {
    /// Returns a reference to the CommonExt struct.
    fn common_ext(&self) -> Option<&CommonExt> {
        Some(&self.common)
    }

    /// Guided Decoding Options
    fn get_guided_json(&self) -> Option<serde_json::Value> {
        if let Some(value) = self.common.guided_json.clone() {
            return Some(value);
        }

        if let Some(response_format) = self.inner.response_format.as_ref() {
            use dynamo_protocols::types::ResponseFormat;
            match response_format {
                ResponseFormat::Text => {}
                ResponseFormat::JsonObject => {
                    // Minimal JSON Schema for "any JSON object"
                    return Some(serde_json::json!({
                        "type": "object"
                    }));
                }
                ResponseFormat::JsonSchema { json_schema } => {
                    // validate_response_format ensures schema is present when type=json_schema
                    if let Some(schema) = json_schema.schema.clone() {
                        return Some(schema);
                    }
                }
            }
        }

        None
    }

    fn get_guided_regex(&self) -> Option<String> {
        self.common.guided_regex.clone()
    }

    fn get_guided_grammar(&self) -> Option<String> {
        self.common.guided_grammar.clone()
    }

    fn get_guided_choice(&self) -> Option<Vec<String>> {
        self.common.guided_choice.clone()
    }

    fn get_guided_decoding_backend(&self) -> Option<String> {
        self.common.guided_decoding_backend.clone()
    }

    fn get_guided_whitespace_pattern(&self) -> Option<String> {
        self.common.guided_whitespace_pattern.clone()
    }

    fn get_top_k(&self) -> Option<i32> {
        self.common.top_k
    }

    fn get_min_p(&self) -> Option<f32> {
        self.common.min_p
    }

    fn get_repetition_penalty(&self) -> Option<f32> {
        self.common.repetition_penalty
    }

    fn get_include_stop_str_in_output(&self) -> Option<bool> {
        self.common.include_stop_str_in_output
    }

    fn get_skip_special_tokens(&self) -> Option<bool> {
        self.common.skip_special_tokens
    }

    fn get_prompt_logprobs_count(&self) -> Option<u32> {
        self.common.prompt_logprobs
    }
}

/// Implements `OpenAIStopConditionsProvider` for `NvCreateChatCompletionRequest`,
/// providing access to stop conditions that control chat completion behavior.
impl OpenAIStopConditionsProvider for NvCreateChatCompletionRequest {
    /// Retrieves the maximum number of tokens allowed in the response.
    #[allow(deprecated)]
    fn get_max_tokens(&self) -> Option<u32> {
        self.inner.max_completion_tokens.or(self.inner.max_tokens)
    }

    /// Retrieves the minimum number of tokens required in the response.
    /// Returns `min_tokens` Value
    /// `min_tokens` is not an OpenAI-supported parameter.
    fn get_min_tokens(&self) -> Option<u32> {
        self.common.min_tokens
    }

    /// Retrieves the stop conditions that terminate the chat completion response.
    ///
    /// Converts OpenAI's `Stop` enum to a `Vec<String>`, normalizing the representation.
    ///
    /// # Returns
    /// * `Some(Vec<String>)` if stop conditions are set.
    /// * `None` if no stop conditions are defined.
    fn get_stop(&self) -> Option<Vec<String>> {
        self.inner.stop.as_ref().and_then(|stop| stop.strings())
    }

    fn get_stop_token_ids(&self) -> Option<Vec<crate::types::TokenIdType>> {
        // Token IDs may be provided in the standard OpenAI `stop` array.
        if let Some(ids) = self.inner.stop.as_ref().and_then(|stop| stop.token_ids()) {
            return Some(ids);
        }
        // Also accept top-level `stop_token_ids` from passthrough clients.
        self.unsupported_fields
            .get("stop_token_ids")
            .and_then(|v| serde_json::from_value::<Vec<crate::types::TokenIdType>>(v.clone()).ok())
    }

    /// Returns a reference to the optional `NvExt` extension, if available.
    fn nvext(&self) -> Option<&NvExt> {
        self.nvext.as_ref()
    }

    /// Get ignore_eos from CommonExt.
    fn get_common_ignore_eos(&self) -> Option<bool> {
        self.common.ignore_eos
    }

    /// Get the effective ignore_eos value from CommonExt.
    fn get_ignore_eos(&self) -> Option<bool> {
        self.common.ignore_eos
    }
}

impl OpenAIOutputOptionsProvider for NvCreateChatCompletionRequest {
    fn get_logprobs(&self) -> Option<u32> {
        match self.inner.logprobs {
            Some(true) => match self.inner.top_logprobs {
                Some(top_logprobs) => Some(top_logprobs as u32),
                None => Some(1_u32),
            },
            Some(false) => None,
            None => None,
        }
    }

    fn get_prompt_logprobs(&self) -> Option<u32> {
        // Top-level `prompt_logprobs` is carried through CommonExt.
        self.common.prompt_logprobs
    }

    fn get_skip_special_tokens(&self) -> Option<bool> {
        CommonExtProvider::get_skip_special_tokens(self)
    }

    fn get_formatted_prompt(&self) -> Option<bool> {
        None
    }

    fn get_return_tokens_as_token_ids(&self) -> Option<bool> {
        self.return_tokens_as_token_ids
    }
}

impl NvCreateChatCompletionRequest {
    // `max_tokens` is deprecated upstream but still honored as a fallback in
    // `get_max_tokens` (max_completion_tokens.or(max_tokens)), so it must be validated.
    #[allow(deprecated)]
    fn validate_with_options(
        &self,
        allow_unparseable_tool_arguments: bool,
    ) -> Result<(), anyhow::Error> {
        validate::validate_no_unsupported_fields(&self.unsupported_fields)?;
        validate::validate_chat_template_args(self.chat_template_args.as_ref())?;
        validate::validate_messages(&self.inner.messages, allow_unparseable_tool_arguments)?;
        validate::validate_model(&self.inner.model)?;
        // none for store
        validate::validate_reasoning_effort(&self.inner.reasoning_effort)?;
        // none for metadata
        validate::validate_frequency_penalty(self.inner.frequency_penalty)?;
        validate::validate_logit_bias(&self.inner.logit_bias)?;
        // none for logprobs
        validate::validate_top_logprobs(self.inner.top_logprobs)?;
        // Honor the deprecated `max_tokens` fallback: reject 0 at the frontend (400)
        // instead of letting the vLLM worker reject it as a 500.
        validate::validate_max_tokens(self.inner.max_tokens)?;
        validate::validate_max_completion_tokens(self.inner.max_completion_tokens)?;
        validate::validate_n(self.inner.n)?;
        validate_completion_token_ids_single_choice(
            self.inner.n.unwrap_or(1) as usize,
            self.nvext.as_ref(),
        )?;
        // none for modalities
        // none for prediction
        // none for audio
        validate::validate_presence_penalty(self.inner.presence_penalty)?;
        validate::validate_response_format(&self.inner.response_format)?;
        // none for seed
        validate::validate_service_tier(&self.inner.service_tier)?;
        validate::validate_stop(&self.inner.stop)?;
        // none for stream
        // none for stream_options
        validate::validate_temperature(self.inner.temperature)?;
        validate::validate_top_p(self.inner.top_p)?;
        let effective_tools = self.effective_tools();
        validate::validate_tools(&Some(effective_tools.as_slice()))?;
        validate::validate_dynamic_tool_messages(&self.inner.messages, &effective_tools)?;
        validate::validate_tool_choice(&self.inner.tool_choice, Some(&effective_tools))?;
        // none for parallel_tool_calls
        validate::validate_user(self.inner.user.as_deref())?;
        // none for function call
        // none for functions
        // Common Ext
        validate::validate_repetition_penalty(self.get_repetition_penalty())?;
        validate::validate_min_p(self.get_min_p())?;
        validate::validate_top_k(self.get_top_k())?;
        // Cross-field validation
        validate::validate_n_with_temperature(self.inner.n, self.inner.temperature)?;

        Ok(())
    }

    pub(crate) fn validate_with_kimi_api_compliance(&self) -> Result<(), anyhow::Error> {
        self.validate_with_options(true)
    }
}

/// Implements `ValidateRequest` for `NvCreateChatCompletionRequest`,
/// allowing us to validate the data.
impl ValidateRequest for NvCreateChatCompletionRequest {
    fn validate(&self) -> Result<(), anyhow::Error> {
        self.validate_with_options(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engines::ValidateRequest;
    use crate::protocols::common::{OutputOptionsProvider, StopConditionsProvider};
    use dynamo_protocols::types::{ChatCompletionTool, ChatCompletionToolType, FunctionObject};
    use serde_json::json;

    fn tool(name: &str) -> serde_json::Value {
        json!({
            "type": "function",
            "function": {
                "name": name,
                "description": "A test tool",
                "parameters": {"type": "object", "properties": {}}
            }
        })
    }

    fn dynamic_request(
        dynamic_messages: Vec<serde_json::Value>,
        global_tools: Option<Vec<serde_json::Value>>,
    ) -> NvCreateChatCompletionRequest {
        let mut messages = dynamic_messages;
        messages.push(json!({"role": "user", "content": "Use a tool"}));

        let mut request = json!({
            "model": "test-model",
            "messages": messages,
            "tool_choice": "required"
        });
        if let Some(tools) = global_tools {
            request["tools"] = json!(tools);
        }

        serde_json::from_value(request).expect("dynamic-tool request should deserialize")
    }

    fn validate_request(request: &NvCreateChatCompletionRequest) -> anyhow::Result<()> {
        ValidateRequest::validate(request)
    }

    #[test]
    fn test_dynamic_tools_satisfy_required_tool_choice() {
        let request = dynamic_request(
            vec![json!({
                "role": "system",
                "content": "",
                "tools": [tool("get_weather")]
            })],
            None,
        );

        validate_request(&request).expect("dynamic tools should be available");
    }

    #[test]
    fn test_dynamic_tools_require_empty_system_content() {
        let request = dynamic_request(
            vec![json!({
                "role": "system",
                "content": "not empty",
                "tools": [tool("get_weather")]
            })],
            None,
        );

        assert!(validate_request(&request).is_err());
    }

    #[test]
    fn test_dynamic_tools_require_complete_definitions_and_valid_names() {
        let mut missing_parameters = tool("missing_parameters");
        missing_parameters["function"]
            .as_object_mut()
            .unwrap()
            .remove("parameters");

        for invalid_tool in [tool("1bad_name"), missing_parameters] {
            let request = dynamic_request(
                vec![json!({
                    "role": "system",
                    "content": "",
                    "tools": [invalid_tool]
                })],
                None,
            );
            assert!(validate_request(&request).is_err());
        }
    }

    #[test]
    fn test_dynamic_tools_reject_malformed_wire_definitions() {
        let malformed_tools = [
            json!({"function": {"name": "missing_type", "parameters": {}}}),
            json!({"type": "function"}),
            json!({"type": "function", "function": {"parameters": {}}}),
            json!({"type": "bogus", "function": {"name": "bad_type"}}),
        ];

        for invalid_tool in malformed_tools {
            let request = json!({
                "model": "test-model",
                "messages": [
                    {
                        "role": "system",
                        "content": "",
                        "tools": [invalid_tool]
                    },
                    {"role": "user", "content": "hello"}
                ]
            });
            assert!(serde_json::from_value::<NvCreateChatCompletionRequest>(request).is_err());
        }
    }

    #[test]
    fn test_dynamic_tool_names_must_be_unique() {
        let cases = [
            (
                vec![json!({
                    "role": "system",
                    "content": "",
                    "tools": [tool("dup"), tool("dup")]
                })],
                None,
            ),
            (
                vec![
                    json!({"role": "system", "content": "", "tools": [tool("dup")]}),
                    json!({"role": "system", "content": "", "tools": [tool("dup")]}),
                ],
                None,
            ),
            (
                vec![json!({
                    "role": "system",
                    "content": "",
                    "tools": [tool("dup")]
                })],
                Some(vec![tool("dup")]),
            ),
        ];

        for (messages, global_tools) in cases {
            let request = dynamic_request(messages, global_tools);
            assert!(validate_request(&request).is_err());
        }
    }

    #[test]
    fn test_global_and_dynamic_tools_can_coexist() {
        let request = dynamic_request(
            vec![json!({
                "role": "system",
                "content": "",
                "tools": [tool("get_weather")]
            })],
            Some(vec![tool("get_stock_price")]),
        );

        validate_request(&request).expect("distinct global and dynamic tools should coexist");
    }
    #[test]
    fn test_skip_special_tokens_none() {
        let json_str = json!({
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "Hello"}
            ]
        });

        let request: NvCreateChatCompletionRequest =
            serde_json::from_value(json_str).expect("Failed to deserialize request");

        assert_eq!(request.common.skip_special_tokens, None);

        let output_options = request
            .extract_output_options()
            .expect("Failed to extract output options");

        assert_eq!(output_options.skip_special_tokens, None);
    }

    #[test]
    fn test_skip_special_tokens_propagates() {
        for skip_value in [true, false] {
            let json_str = json!({
                "model": "test-model",
                "messages": [
                    {"role": "user", "content": "Hello"}
                ],
                "skip_special_tokens": skip_value
            });

            let request: NvCreateChatCompletionRequest =
                serde_json::from_value(json_str).expect("Failed to deserialize request");

            let output_options = request
                .extract_output_options()
                .expect("Failed to extract output options");

            assert_eq!(output_options.skip_special_tokens, Some(skip_value));
        }
    }

    #[test]
    fn test_stop_contract() {
        let one_stop = json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stop": " The"
        });
        let request: NvCreateChatCompletionRequest =
            serde_json::from_value(one_stop).expect("Failed to deserialize request");
        assert_eq!(request.get_stop(), Some(vec![" The".to_string()]));
        assert_eq!(request.get_stop_token_ids(), None);

        let many_stops = json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stop": ["A", "B"]
        });
        let request: NvCreateChatCompletionRequest =
            serde_json::from_value(many_stops).expect("Failed to deserialize request");
        assert_eq!(
            request.get_stop(),
            Some(vec!["A".to_string(), "B".to_string()])
        );
        assert_eq!(request.get_stop_token_ids(), None);

        let token_id_stops = json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stop": [32, 34]
        });
        let request: NvCreateChatCompletionRequest =
            serde_json::from_value(token_id_stops).expect("Failed to deserialize request");
        assert_eq!(request.get_stop(), None);
        assert_eq!(request.get_stop_token_ids(), Some(vec![32, 34]));

        let stop_conditions = request
            .extract_stop_conditions()
            .expect("extract stop conditions");
        assert_eq!(stop_conditions.stop, None);
        assert_eq!(stop_conditions.stop_token_ids, Some(vec![32, 34]));

        let token_id_display_string_stop = json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stop": "token_id:576"
        });
        let request: NvCreateChatCompletionRequest =
            serde_json::from_value(token_id_display_string_stop)
                .expect("Failed to deserialize request");
        assert_eq!(request.get_stop(), Some(vec!["token_id:576".to_string()]));
        assert_eq!(request.get_stop_token_ids(), None);

        let token_id_display_string_array_stop = json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stop": ["token_id:576"]
        });
        let request: NvCreateChatCompletionRequest =
            serde_json::from_value(token_id_display_string_array_stop)
                .expect("Failed to deserialize request");
        assert_eq!(request.get_stop(), Some(vec!["token_id:576".to_string()]));
        assert_eq!(request.get_stop_token_ids(), None);

        let scalar_token_id_stop = json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stop": 576
        });
        let result: Result<NvCreateChatCompletionRequest, _> =
            serde_json::from_value(scalar_token_id_stop);
        assert!(result.is_err());

        // `stop_token_ids` is accepted and plumbed by the provider trait.
        let whitelisted_stop_token_ids = json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stop_token_ids": [576]
        });
        let request: NvCreateChatCompletionRequest =
            serde_json::from_value(whitelisted_stop_token_ids)
                .expect("Failed to deserialize request");
        assert_eq!(request.get_stop_token_ids(), Some(vec![576]));
        assert!(
            ValidateRequest::validate(&request).is_ok(),
            "stop_token_ids must be accepted via PASSTHROUGH_EXTRA_FIELDS"
        );

        let invalid_stop_token_ids = json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stop_token_ids": "bad"
        });
        let request: NvCreateChatCompletionRequest =
            serde_json::from_value(invalid_stop_token_ids).expect("Failed to deserialize request");
        let err = ValidateRequest::validate(&request).expect_err("invalid stop_token_ids");
        assert!(err.to_string().contains("stop_token_ids"));
    }

    #[test]
    fn test_passthrough_token_constraints_validate() {
        let request_json = json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "allowed_token_ids": [10, 11],
            "bad_words_token_ids": [[12, 13]]
        });
        let request: NvCreateChatCompletionRequest =
            serde_json::from_value(request_json).expect("Failed to deserialize request");

        assert_eq!(
            request.unsupported_fields.get("allowed_token_ids"),
            Some(&serde_json::json!([10, 11]))
        );
        assert_eq!(
            request.unsupported_fields.get("bad_words_token_ids"),
            Some(&serde_json::json!([[12, 13]]))
        );
        assert!(ValidateRequest::validate(&request).is_ok());
    }

    #[test]
    fn test_max_tokens_zero_rejected() {
        // Deprecated `max_tokens` is still honored as a fallback in `get_max_tokens`,
        // so 0 must be rejected at the frontend (-> 400) instead of reaching the worker (-> 500).
        let request_json = json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 0
        });
        let request: NvCreateChatCompletionRequest =
            serde_json::from_value(request_json).expect("Failed to deserialize request");
        let err = ValidateRequest::validate(&request).expect_err("max_tokens 0 must be rejected");
        assert!(
            err.to_string().contains("Max tokens"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_completion_token_ids_rejected_for_multi_choice() {
        let request_json = json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "n": 2,
            "nvext": {
                "extra_fields": ["completion_token_ids"]
            }
        });
        let request: NvCreateChatCompletionRequest =
            serde_json::from_value(request_json).expect("Failed to deserialize request");

        let err = ValidateRequest::validate(&request).expect_err("multi-choice token ids");
        assert!(err.to_string().contains("completion_token_ids"));
    }

    #[test]
    fn test_validate_tool_choice_required_rejects_empty_tools() {
        let request_json = json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "tool_choice": "required"
        });
        let request: NvCreateChatCompletionRequest =
            serde_json::from_value(request_json).expect("Failed to deserialize request");

        let err = ValidateRequest::validate(&request).expect_err("required needs tools");
        assert!(
            err.to_string()
                .contains("tool_choice is \"required\" but tools is empty")
        );
    }

    #[test]
    fn test_validate_tool_choice_named_rejects_missing_tool() {
        let request_json = json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {"type": "object", "properties": {}}
                }
            }],
            "tool_choice": {
                "type": "function",
                "function": {"name": "search"}
            }
        });
        let request: NvCreateChatCompletionRequest =
            serde_json::from_value(request_json).expect("Failed to deserialize request");

        let err = ValidateRequest::validate(&request).expect_err("named tool must exist");
        assert!(
            err.to_string()
                .contains("tool named \"search\" in tool_choice is not present in tools")
        );
    }

    #[test]
    fn test_truncate_prompt_tokens_rejected_until_supported() {
        let request_json = json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "truncate_prompt_tokens": 2
        });
        let request: NvCreateChatCompletionRequest =
            serde_json::from_value(request_json).expect("Failed to deserialize request");

        assert!(ValidateRequest::validate(&request).is_err());
    }

    #[test]
    fn test_prompt_cache_key_ignored() {
        // `prompt_cache_key` is a standard OpenAI prompt-cache hint Dynamo does
        // not implement. It must be accepted (not 400'd) and silently ignored.
        let request_json = json!({
            "model": "MiniMaxAI/MiniMax-M3",
            "messages": [
                {"role": "system", "content": "[Session: nKYaXRvj7uff0LYT] You are a helpful assistant for cache testing."},
                {"role": "user", "content": "Say hi in one word."}
            ],
            "max_tokens": 64,
            "stream": false,
            "prompt_cache_key": "NbrnTP3fAbnFbmOH"
        });
        let request: NvCreateChatCompletionRequest =
            serde_json::from_value(request_json).expect("Failed to deserialize request");

        assert!(
            ValidateRequest::validate(&request).is_ok(),
            "prompt_cache_key must be accepted and ignored, not rejected"
        );
        // It is captured by the catch-all but `unsupported_fields` is
        // `skip_serializing`, so it is dropped before forwarding downstream.
        assert!(request.unsupported_fields.contains_key("prompt_cache_key"));
    }

    // -----------------------------------------------------------------------
    // Parser -> protocol mapping (decoupling guard).
    //
    // `dynamo-parsers` no longer depends on `dynamo-protocols`; the mapping
    // moved into this consumer. These tests pin the mapper output to the
    // *exact* struct + serialized JSON the old protocol-typed parser path
    // produced, proving the wire output is unchanged.
    // -----------------------------------------------------------------------
    use dynamo_parsers::tool_calling::{
        CalledFunction, CalledFunctionStream, ToolCallResponse, ToolCallResponseChunk, ToolCallType,
    };

    fn native_call(id: &str, name: &str, args: &str) -> ToolCallResponse {
        ToolCallResponse {
            id: id.to_string(),
            tp: ToolCallType::Function,
            function: CalledFunction {
                name: name.to_string(),
                arguments: args.to_string(),
            },
        }
    }

    fn native_chunk(index: u32, id: &str, name: &str, args: &str) -> ToolCallResponseChunk {
        ToolCallResponseChunk {
            index,
            id: Some(id.to_string()),
            tp: Some(ToolCallType::Function),
            function: Some(CalledFunctionStream {
                name: Some(name.to_string()),
                arguments: Some(args.to_string()),
            }),
        }
    }

    /// Reference reconstruction of the pre-decoupling unary mapping that lived
    /// inside `dynamo-parsers`. Kept inline so a divergence in the live mapper
    /// fails the test.
    fn legacy_unary(id: &str, name: &str, args: &str) -> ChatCompletionMessageToolCall {
        ChatCompletionMessageToolCall {
            id: id.to_string(),
            r#type: FunctionType::Function,
            function: FunctionCall {
                name: name.to_string(),
                arguments: args.to_string(),
            },
        }
    }

    /// Reference reconstruction of the pre-decoupling streaming mapping.
    fn legacy_chunk(
        index: u32,
        id: &str,
        name: &str,
        args: &str,
    ) -> ChatCompletionMessageToolCallChunk {
        ChatCompletionMessageToolCallChunk {
            index,
            id: Some(id.to_string()),
            r#type: Some(FunctionType::Function),
            function: Some(FunctionCallStream {
                name: Some(name.to_string()),
                arguments: Some(args.to_string()),
            }),
        }
    }

    #[test]
    fn unary_mapping_matches_legacy_struct_and_json() {
        for (id, name, args) in [
            (
                "call_1",
                "get_weather",
                r#"{"location":"SF","unit":"celsius"}"#,
            ),
            ("call_2", "ping", "{}"), // empty arguments
        ] {
            let mapped = tool_call_response_to_protocol(native_call(id, name, args));
            let legacy = legacy_unary(id, name, args);
            assert_eq!(mapped, legacy, "struct mismatch for {name}");
            assert_eq!(
                serde_json::to_string(&mapped).unwrap(),
                serde_json::to_string(&legacy).unwrap(),
                "serialized JSON mismatch for {name}"
            );
        }
    }

    #[test]
    fn unary_mapping_multi_call_matches_legacy() {
        let inputs = [
            ("a", "first", r#"{"k":"v1"}"#),
            ("b", "second", r#"{"k":"v2"}"#),
        ];
        let mapped: Vec<_> = inputs
            .iter()
            .map(|(id, n, a)| tool_call_response_to_protocol(native_call(id, n, a)))
            .collect();
        let legacy: Vec<_> = inputs
            .iter()
            .map(|(id, n, a)| legacy_unary(id, n, a))
            .collect();
        assert_eq!(mapped, legacy);
        assert_eq!(
            serde_json::to_string(&mapped).unwrap(),
            serde_json::to_string(&legacy).unwrap()
        );
    }

    #[test]
    fn stream_mapping_matches_legacy_struct_and_json() {
        for (idx, id, name, args) in [
            (0u32, "call_1", "get_weather", r#"{"location":"SF"}"#),
            (1u32, "call_2", "ping", "{}"), // empty arguments
        ] {
            let mapped = tool_call_response_chunk_to_protocol(native_chunk(idx, id, name, args));
            let legacy = legacy_chunk(idx, id, name, args);
            assert_eq!(mapped, legacy, "struct mismatch for {name}");
            assert_eq!(
                serde_json::to_string(&mapped).unwrap(),
                serde_json::to_string(&legacy).unwrap(),
                "serialized JSON mismatch for {name}"
            );
        }
    }

    #[test]
    fn stream_mapping_multi_call_indexes_and_matches_legacy() {
        let inputs = [
            (0u32, "a", "first", r#"{"k":"v1"}"#),
            (1u32, "b", "second", r#"{"k":"v2"}"#),
        ];
        let mapped: Vec<_> = inputs
            .iter()
            .map(|(i, id, n, a)| tool_call_response_chunk_to_protocol(native_chunk(*i, id, n, a)))
            .collect();
        let legacy: Vec<_> = inputs
            .iter()
            .map(|(i, id, n, a)| legacy_chunk(*i, id, n, a))
            .collect();
        assert_eq!(mapped, legacy);
        assert_eq!(
            serde_json::to_string(&mapped).unwrap(),
            serde_json::to_string(&legacy).unwrap()
        );
    }

    #[test]
    fn test_validate_messages_rejects_bad_tool_call_arguments() {
        for arguments in ["{invalid json}", "[]", "null", "\"not an object\""] {
            let request_json = json!({
                "model": "test-model",
                "messages": [
                    {"role": "user", "content": "weather?"},
                    {
                        "role": "assistant",
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": arguments
                            }
                        }]
                    },
                    {"role": "tool", "tool_call_id": "call_1", "content": "sunny"}
                ],
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {"type": "object", "properties": {}}
                    }
                }]
            });

            let request: NvCreateChatCompletionRequest =
                serde_json::from_value(request_json).expect("Failed to deserialize request");
            let err = ValidateRequest::validate(&request)
                .expect_err("bad tool_call arguments should fail validation");
            let err = err.to_string();
            assert!(
                err.contains("`messages[1].tool_calls[0].function.arguments`"),
                "unexpected error for {arguments:?}: {err}"
            );
            assert!(
                err.contains("valid JSON object string"),
                "unexpected error for {arguments:?}: {err}"
            );
        }
    }

    #[test]
    fn kimi_compliance_accepts_unparseable_tool_call_arguments() {
        let request: NvCreateChatCompletionRequest = serde_json::from_value(json!({
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "weather?"},
                {
                    "role": "assistant",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": "{invalid json}"
                        }
                    }]
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "sunny"}
            ],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {"type": "object", "properties": {}}
                }
            }]
        }))
        .expect("Failed to deserialize request");

        request
            .validate_with_kimi_api_compliance()
            .expect("Kimi renderer accepts unparseable arguments as a raw JSON block");
    }

    #[test]
    fn test_validate_messages_accepts_empty_tool_call_arguments() {
        for arguments in ["", " \n\t ", "{}"] {
            let request_json = json!({
                "model": "test-model",
                "messages": [
                    {"role": "user", "content": "weather?"},
                    {
                        "role": "assistant",
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": arguments
                            }
                        }]
                    },
                    {"role": "tool", "tool_call_id": "call_1", "content": "sunny"}
                ],
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {"type": "object", "properties": {}}
                    }
                }]
            });

            let request: NvCreateChatCompletionRequest =
                serde_json::from_value(request_json).expect("Failed to deserialize request");
            ValidateRequest::validate(&request)
                .unwrap_or_else(|err| panic!("empty tool_call arguments should validate: {err}"));
        }
    }

    #[test]
    fn test_validate_tools_valid_names() {
        fn make_tool(name: &str) -> ChatCompletionTool {
            ChatCompletionTool {
                r#type: ChatCompletionToolType::Function,
                function: FunctionObject {
                    name: name.to_string(),
                    description: None,
                    parameters: Some(json!({"type": "object", "properties": {}})),
                    strict: None,
                },
            }
        }

        let tools = vec![
            make_tool("func_name"),
            make_tool("func-name_v2"),
            make_tool("FuncName"),
            make_tool("Func_Name-123"),
        ];
        assert!(validate::validate_tools(&Some(&tools)).is_ok());
    }

    #[test]
    fn test_validate_tools_invalid_names() {
        for name in ["<func_name>", "func name", "func@name", "func,name", ""] {
            let tools = vec![ChatCompletionTool {
                r#type: ChatCompletionToolType::Function,
                function: FunctionObject {
                    name: name.to_string(),
                    description: None,
                    parameters: Some(json!({"type": "object", "properties": {}})),
                    strict: None,
                },
            }];
            assert!(
                validate::validate_tools(&Some(&tools)).is_err(),
                "expected error for name: {name:?}"
            );
        }
    }

    #[test]
    fn test_openai_thinking_payload_normalizes_to_template_args() {
        let json_str = json!({
            "model": "deepseek-ai/DeepSeek-V4-Pro",
            "messages": [
                {"role": "user", "content": "Hello"}
            ],
            "reasoning_effort": "max",
            "thinking": {"type": "enabled"}
        });

        let mut request: NvCreateChatCompletionRequest =
            serde_json::from_value(json_str).expect("Failed to deserialize request");
        request
            .normalize_reasoning_template_args()
            .expect("thinking payload should normalize");

        let args = request
            .chat_template_args
            .as_ref()
            .expect("chat_template_args should be populated");
        assert_eq!(args.get("thinking"), Some(&json!(true)));
        assert_eq!(args.get("thinking_mode"), Some(&json!("enabled")));
        assert_eq!(args.get("reasoning_effort"), Some(&json!("max")));
    }

    #[test]
    fn test_openai_thinking_adaptive_is_rejected() {
        let json_str = json!({
            "model": "MiniMaxAI/MiniMax-M3",
            "messages": [
                {"role": "user", "content": "Hello"}
            ],
            "thinking": {"type": "adaptive"}
        });

        let mut request: NvCreateChatCompletionRequest =
            serde_json::from_value(json_str).expect("Failed to deserialize request");
        let err = request
            .normalize_reasoning_template_args()
            .expect_err("adaptive thinking payload should be rejected");
        assert!(err.to_string().contains("enabled` or `disabled"));
    }

    fn normalize_thinking_object(thinking: serde_json::Value) -> NvCreateChatCompletionRequest {
        let json_str = json!({
            "model": "moonshotai/Kimi-K3",
            "messages": [{"role": "user", "content": "Hello"}],
            "thinking": thinking
        });
        let mut request: NvCreateChatCompletionRequest =
            serde_json::from_value(json_str).expect("Failed to deserialize request");
        request
            .normalize_reasoning_template_args()
            .expect("thinking payload should normalize");
        request
    }

    #[test]
    fn test_kimi_thinking_effort_and_keep_all_normalize() {
        let request = normalize_thinking_object(json!({
            "type": "enabled", "effort": "high", "keep": "all"
        }));
        let args = request.chat_template_args.as_ref().unwrap();
        assert_eq!(args.get("thinking"), Some(&json!(true)));
        assert_eq!(args.get("thinking_mode"), Some(&json!("enabled")));
        assert_eq!(args.get("thinking_effort"), Some(&json!("high")));
        // spec §4: keep=all keeps all reasoning history => preserve_thinking=true.
        assert_eq!(args.get("preserve_thinking"), Some(&json!(true)));
        assert!(request.thinking.is_none());
    }

    #[test]
    fn test_kimi_thinking_keep_interleaved_drops_history() {
        let request = normalize_thinking_object(json!({
            "type": "enabled", "keep": "interleaved"
        }));
        let args = request.chat_template_args.as_ref().unwrap();
        // spec §4: interleaved keeps only the latest turn => preserve_thinking=false.
        assert_eq!(args.get("preserve_thinking"), Some(&json!(false)));
        // No request effort + no caller kwarg + enabled => spec §4 default max.
        assert_eq!(args.get("thinking_effort"), Some(&json!("max")));
    }

    #[test]
    fn test_kimi_thinking_keep_absent_defaults_to_preserve_all() {
        let request = normalize_thinking_object(json!({"type": "enabled", "effort": "low"}));
        let args = request.chat_template_args.as_ref().unwrap();
        // spec §4: keep unset normalizes to all => preserve_thinking=true.
        assert_eq!(args.get("preserve_thinking"), Some(&json!(true)));
        assert_eq!(args.get("thinking_effort"), Some(&json!("low")));
    }

    #[test]
    fn test_kimi_thinking_keep_unknown_value_rejected() {
        // spec §4 allows only all/interleaved; anything else is a 400 at ingress
        // (keep must map to a preserve_thinking bool, so there is no passthrough).
        let json_str = json!({
            "model": "moonshotai/Kimi-K3",
            "messages": [{"role": "user", "content": "Hello"}],
            "thinking": {"type": "enabled", "keep": "sometimes"}
        });
        let mut request: NvCreateChatCompletionRequest =
            serde_json::from_value(json_str).expect("Failed to deserialize request");
        assert!(
            request.normalize_reasoning_template_args().is_err(),
            "unknown thinking.keep must be rejected"
        );
    }

    #[test]
    fn test_kimi_thinking_type_absent_defaults_enabled() {
        // spec §4: `thinking.type` is optional; an object with no type (here only
        // `keep`) is enabled thinking, not a 400.
        let request = normalize_thinking_object(json!({"keep": "all"}));
        let args = request.chat_template_args.as_ref().unwrap();
        assert_eq!(args.get("thinking"), Some(&json!(true)));
        assert_eq!(args.get("thinking_mode"), Some(&json!("enabled")));
        // keep=all + enabled => preserve all reasoning; effort defaults to max.
        assert_eq!(args.get("preserve_thinking"), Some(&json!(true)));
        assert_eq!(args.get("thinking_effort"), Some(&json!("max")));
    }

    #[test]
    fn test_kimi_thinking_disabled_ignores_invalid_keep_and_effort() {
        // spec §4: keep/effort are ignored when disabled — an otherwise-invalid
        // keep must NOT 400 (it is not validated in the disabled path).
        let json_str = json!({
            "model": "moonshotai/Kimi-K3",
            "messages": [{"role": "user", "content": "Hello"}],
            "thinking": {"type": "disabled", "keep": "none", "effort": "high"}
        });
        let mut request: NvCreateChatCompletionRequest =
            serde_json::from_value(json_str).expect("Failed to deserialize request");
        request
            .normalize_reasoning_template_args()
            .expect("disabled thinking must ignore (not validate) keep/effort");
        let args = request.chat_template_args.as_ref().unwrap();
        assert_eq!(args.get("thinking"), Some(&json!(false)));
        assert_eq!(args.get("thinking_mode"), Some(&json!("disabled")));
        assert_eq!(args.get("thinking_keep"), None);
        assert_eq!(args.get("preserve_thinking"), None);
        assert_eq!(args.get("thinking_effort"), None);
    }

    #[test]
    fn test_kimi_thinking_effort_passes_through_verbatim() {
        // The renderer owns effort validation ({low,medium,high,max} renders,
        // anything else renders nothing) — normalization must not filter it.
        let request = normalize_thinking_object(json!({
            "type": "enabled", "effort": "ultra"
        }));
        let args = request.chat_template_args.as_ref().unwrap();
        assert_eq!(args.get("thinking_effort"), Some(&json!("ultra")));
    }

    #[test]
    fn test_kimi_request_level_effort_wins_over_user_kwargs() {
        let json_str = json!({
            "model": "moonshotai/Kimi-K3",
            "messages": [{"role": "user", "content": "Hello"}],
            "chat_template_args": {"thinking_effort": "low", "preserve_thinking": true},
            "thinking": {"type": "enabled", "effort": "high", "keep": "interleaved"}
        });
        let mut request: NvCreateChatCompletionRequest =
            serde_json::from_value(json_str).expect("Failed to deserialize request");
        request
            .normalize_reasoning_template_args()
            .expect("thinking payload should normalize");
        let args = request.chat_template_args.as_ref().unwrap();
        // Authoritative request fields overwrite user-supplied kwargs.
        assert_eq!(args.get("thinking_effort"), Some(&json!("high")));
        // keep=interleaved => preserve_thinking=false, overriding the caller's true.
        assert_eq!(args.get("preserve_thinking"), Some(&json!(false)));
    }

    #[test]
    fn test_kimi_user_kwargs_survive_when_thinking_object_has_no_extras() {
        let json_str = json!({
            "model": "moonshotai/Kimi-K3",
            "messages": [{"role": "user", "content": "Hello"}],
            "chat_template_args": {"thinking_effort": "low"},
            "thinking": {"type": "enabled"}
        });
        let mut request: NvCreateChatCompletionRequest =
            serde_json::from_value(json_str).expect("Failed to deserialize request");
        request
            .normalize_reasoning_template_args()
            .expect("thinking payload should normalize");
        let args = request.chat_template_args.as_ref().unwrap();
        // Absent request-level extras leave user kwargs untouched.
        assert_eq!(args.get("thinking_effort"), Some(&json!("low")));
    }

    #[test]
    fn test_openai_thinking_disabled_normalizes_to_template_mode() {
        let json_str = json!({
            "model": "MiniMaxAI/MiniMax-M3",
            "messages": [
                {"role": "user", "content": "Hello"}
            ],
            "thinking": {"type": "disabled"}
        });

        let mut request: NvCreateChatCompletionRequest =
            serde_json::from_value(json_str).expect("Failed to deserialize request");
        request
            .normalize_reasoning_template_args()
            .expect("disabled thinking payload should normalize");

        let args = request
            .chat_template_args
            .as_ref()
            .expect("chat_template_args should be populated");
        assert_eq!(args.get("thinking"), Some(&json!(false)));
        assert_eq!(args.get("thinking_mode"), Some(&json!("disabled")));
    }

    #[test]
    fn test_openai_thinking_effort_normalizes_to_reasoning_effort() {
        let json_str = json!({
            "model": "moonshotai/Kimi-K2.7-Code",
            "messages": [
                {"role": "user", "content": "Hello"}
            ],
            "thinking": {"type": "enabled", "effort": "high"}
        });

        let mut request: NvCreateChatCompletionRequest =
            serde_json::from_value(json_str).expect("Failed to deserialize request");
        request
            .normalize_reasoning_template_args()
            .expect("thinking effort should normalize");

        let args = request
            .chat_template_args
            .as_ref()
            .expect("chat_template_args should be populated");
        assert_eq!(args.get("thinking"), Some(&json!(true)));
        assert_eq!(args.get("thinking_mode"), Some(&json!("enabled")));
        assert_eq!(args.get("reasoning_effort"), Some(&json!("high")));
        assert!(request.thinking.is_none());
    }

    #[test]
    fn test_openai_thinking_effort_is_ignored_when_thinking_disabled() {
        let json_str = json!({
            "model": "moonshotai/Kimi-K2.7-Code",
            "messages": [
                {"role": "user", "content": "Hello"}
            ],
            "thinking": {"type": "disabled", "effort": "high"}
        });

        let mut request: NvCreateChatCompletionRequest =
            serde_json::from_value(json_str).expect("Failed to deserialize request");
        request
            .normalize_reasoning_template_args()
            .expect("disabled thinking effort should normalize");

        let args = request
            .chat_template_args
            .as_ref()
            .expect("chat_template_args should be populated");
        assert_eq!(args.get("thinking"), Some(&json!(false)));
        assert_eq!(args.get("thinking_mode"), Some(&json!("disabled")));
        assert_eq!(args.get("reasoning_effort"), None);
        assert!(request.thinking.is_none());
    }

    #[test]
    fn test_openai_thinking_keep_normalizes_to_template_args() {
        let json_str = json!({
            "model": "moonshotai/Kimi-K2.7-Code",
            "messages": [
                {"role": "user", "content": "Hello"}
            ],
            "thinking": {"type": "enabled", "keep": "interleaved"}
        });

        let mut request: NvCreateChatCompletionRequest =
            serde_json::from_value(json_str).expect("Failed to deserialize request");
        request
            .normalize_reasoning_template_args()
            .expect("thinking keep should normalize");

        let args = request
            .chat_template_args
            .as_ref()
            .expect("chat_template_args should be populated");
        assert_eq!(args.get("thinking"), Some(&json!(true)));
        assert_eq!(args.get("thinking_mode"), Some(&json!("enabled")));
        assert_eq!(args.get("thinking_keep"), Some(&json!("interleaved")));
        assert!(request.thinking.is_none());
    }

    #[test]
    fn test_openai_thinking_keep_is_ignored_when_thinking_disabled() {
        let json_str = json!({
            "model": "moonshotai/Kimi-K2.7-Code",
            "messages": [
                {"role": "user", "content": "Hello"}
            ],
            "thinking": {"type": "disabled", "keep": "all"}
        });

        let mut request: NvCreateChatCompletionRequest =
            serde_json::from_value(json_str).expect("Failed to deserialize request");
        request
            .normalize_reasoning_template_args()
            .expect("disabled thinking keep should normalize");

        let args = request
            .chat_template_args
            .as_ref()
            .expect("chat_template_args should be populated");
        assert_eq!(args.get("thinking"), Some(&json!(false)));
        assert_eq!(args.get("thinking_mode"), Some(&json!("disabled")));
        assert_eq!(args.get("thinking_keep"), None);
        assert!(request.thinking.is_none());
    }

    #[test]
    fn test_openai_thinking_top_level_overrides_stale_template_args() {
        let json_str = json!({
            "model": "MiniMaxAI/MiniMax-M3",
            "messages": [
                {"role": "user", "content": "Hello"}
            ],
            "chat_template_args": {
                "thinking": true,
                "thinking_mode": "thinking",
                "reasoning_effort": "high"
            },
            "reasoning_effort": "none",
            "thinking": {"type": "disabled"}
        });

        let mut request: NvCreateChatCompletionRequest =
            serde_json::from_value(json_str).expect("Failed to deserialize request");
        request
            .normalize_reasoning_template_args()
            .expect("top-level thinking payload should normalize");

        let args = request
            .chat_template_args
            .as_ref()
            .expect("chat_template_args should be populated");
        assert_eq!(args.get("thinking"), Some(&json!(false)));
        assert_eq!(args.get("thinking_mode"), Some(&json!("disabled")));
        assert_eq!(args.get("reasoning_effort"), Some(&json!("none")));
        assert!(request.thinking.is_none());
    }

    #[test]
    fn test_reasoning_effort_controls_enable_thinking() {
        for (effort, expected) in [("none", false), ("low", true), ("high", true)] {
            let mut request: NvCreateChatCompletionRequest = serde_json::from_value(json!({
                "model": "zai-org/GLM-5.2",
                "messages": [{"role": "user", "content": "Hello"}],
                "reasoning_effort": effort
            }))
            .expect("request should deserialize");

            request
                .normalize_reasoning_template_args()
                .expect("reasoning effort should normalize");

            let args = request
                .chat_template_args
                .as_ref()
                .expect("chat_template_args should be populated");
            assert_eq!(args.get("enable_thinking"), Some(&json!(expected)));
            assert_eq!(args.get("reasoning_effort"), Some(&json!(effort)));
        }
    }

    #[test]
    fn test_explicit_enable_thinking_overrides_reasoning_effort() {
        let mut request: NvCreateChatCompletionRequest = serde_json::from_value(json!({
            "model": "zai-org/GLM-5.2",
            "messages": [{"role": "user", "content": "Hello"}],
            "reasoning_effort": "none",
            "chat_template_args": {"enable_thinking": true}
        }))
        .expect("request should deserialize");

        request
            .normalize_reasoning_template_args()
            .expect("reasoning effort should normalize");

        let args = request
            .chat_template_args
            .as_ref()
            .expect("chat_template_args should be populated");
        assert_eq!(args.get("enable_thinking"), Some(&json!(true)));
        assert_eq!(args.get("reasoning_effort"), Some(&json!("none")));
    }

    #[test]
    fn test_invalid_openai_thinking_payload_is_rejected() {
        for invalid_thinking in [
            json!("enabled"),
            json!({"type": "auto"}),
            json!({"type": true}),
            json!({}),
        ] {
            let json_str = json!({
                "model": "deepseek-ai/DeepSeek-V4-Pro",
                "messages": [
                    {"role": "user", "content": "Hello"}
                ],
                "thinking": invalid_thinking
            });

            let mut request: NvCreateChatCompletionRequest =
                serde_json::from_value(json_str).expect("Failed to deserialize request");
            assert!(request.normalize_reasoning_template_args().is_err());
        }
    }
}
