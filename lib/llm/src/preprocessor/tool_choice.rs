// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tool-choice guided decoding policy for OpenAI chat requests.

use crate::local_model::runtime_config::StructuralTagMode;
use crate::preprocessor::prompt::kimi_k3::structural_tag::build_kimi_k3_structural_tag;
use crate::preprocessor::{OpenAIPreprocessor, PreprocessedRequest, unified};
use crate::protocols::openai::chat_completions::NvCreateChatCompletionRequest;
use crate::protocols::openai::tools::get_json_schema_from_tools;

use dynamo_parsers::tool_calling::{ToolChoice, ToolDefinition};
use dynamo_protocols::types::{ChatCompletionTool, ChatCompletionToolChoiceOption, ResponseFormat};
use dynamo_runtime::error::{DynamoError, ErrorType};

fn invalid_argument(message: impl Into<String>) -> DynamoError {
    DynamoError::builder()
        .error_type(ErrorType::InvalidArgument)
        .message(message)
        .build()
}

impl OpenAIPreprocessor {
    /// Apply guided decoding for OpenAI tool-choice requests.
    ///
    /// Structural tags are preferred when enabled and supported by the configured
    /// tool-call parser. Forced tool-choice requests fall back to the legacy
    /// JSON-schema constraint when structural tags are not applied.
    pub(super) fn apply_tool_choice_guided_decoding(
        &self,
        request: &NvCreateChatCompletionRequest,
        common_request: &mut PreprocessedRequest,
        prompt_injected_reasoning: bool,
    ) -> Result<bool, DynamoError> {
        let tool_choice = request
            .inner
            .tool_choice
            .as_ref()
            .unwrap_or(&ChatCompletionToolChoiceOption::Auto);

        let tools = request.effective_tools();
        let is_forced_tool_choice = matches!(
            tool_choice,
            ChatCompletionToolChoiceOption::Required | ChatCompletionToolChoiceOption::Named(_)
        );
        let has_explicit_guided_decoding = has_explicit_guided_decoding(request);
        let has_response_format_constraint = has_response_format_constraint(request);

        if is_forced_tool_choice && has_explicit_guided_decoding {
            return Err(invalid_argument(concat!(
                "guided decoding cannot be used in the same request as ",
                "tool_choice=\"required\" or a named tool_choice.",
            )));
        }

        // For non-forced tool choice, explicit guided decoding and response_format
        // constrain assistant content, so tool-choice guided decoding stays inactive.
        let has_assistant_constraint =
            has_explicit_guided_decoding || has_response_format_constraint;
        if !is_forced_tool_choice && has_assistant_constraint {
            return Ok(false);
        }

        if is_forced_tool_choice
            && has_response_format_constraint
            && let Some(gd) = common_request.sampling_options.guided_decoding.as_mut()
        {
            // OpenAI `response_format` applies to assistant content, not tool calls.
            gd.json = None;
        }

        if unified::kimi_k3::is_selected(
            self.runtime_config.reasoning_parser.as_deref(),
            self.tool_call_parser.as_deref(),
        ) {
            return apply_kimi_k3_structural_tag(
                self.runtime_config.structural_tag_mode,
                &convert_tool_choice(tool_choice),
                &convert_tools(&tools),
                prompt_injected_reasoning,
                common_request,
            );
        }

        if self.apply_tool_choice_structural_tag(
            &convert_tool_choice(tool_choice),
            &convert_tools(&tools),
            request.inner.parallel_tool_calls,
            prompt_injected_reasoning,
            common_request,
        )? {
            return Ok(true);
        }

        match get_json_schema_from_tools(Some(tool_choice), Some(&tools)) {
            Ok(Some(schema)) => {
                let gd = common_request
                    .sampling_options
                    .guided_decoding
                    .get_or_insert_default();
                gd.json = Some(schema);
            }
            Ok(None) => {}
            Err(err) => {
                return Err(invalid_argument(err.to_string()));
            }
        }

        // Auto/None requests can reach here when neither structural tags nor a
        // tool-choice JSON fallback were needed.
        Ok(false)
    }
}

/// `prompt_injected_reasoning` is K3's effective thinking flag: the renderer
/// ends the prompt at `<|open|>think<|sep|>` when it is set and
/// `<|open|>response<|sep|>` when it is not. The grammar needs it to know which
/// marker the model still owes -- see `build_kimi_k3_structural_tag`.
fn apply_kimi_k3_structural_tag(
    structural_tag_mode: StructuralTagMode,
    tool_choice: &ToolChoice,
    tools: &[ToolDefinition],
    prompt_injected_reasoning: bool,
    common_request: &mut PreprocessedRequest,
) -> Result<bool, DynamoError> {
    if matches!(tool_choice, ToolChoice::Named(_)) {
        return Err(invalid_argument(
            "Named tool choice is not supported for Kimi K3. \
             Use `tool_choice` set to \"auto\", \"required\", or \"none\" instead.",
        ));
    }
    if structural_tag_mode == StructuralTagMode::Off {
        return Ok(false);
    }

    let Some(structural_tag) =
        build_kimi_k3_structural_tag(tool_choice, tools, prompt_injected_reasoning)
            .map_err(|err| invalid_argument(err.to_string()))?
    else {
        return Ok(false);
    };
    common_request
        .sampling_options
        .guided_decoding
        .get_or_insert_default()
        .structural_tag = Some(structural_tag);
    Ok(true)
}

fn has_explicit_guided_decoding(request: &NvCreateChatCompletionRequest) -> bool {
    request.common.guided_json.is_some()
        || request.common.guided_regex.is_some()
        || request
            .common
            .guided_choice
            .as_ref()
            .is_some_and(|v| !v.is_empty())
        || request.common.guided_grammar.is_some()
}

fn has_response_format_constraint(request: &NvCreateChatCompletionRequest) -> bool {
    request
        .inner
        .response_format
        .as_ref()
        .is_some_and(|format| !matches!(format, ResponseFormat::Text))
}

fn convert_tool_choice(tool_choice: &ChatCompletionToolChoiceOption) -> ToolChoice {
    match tool_choice {
        ChatCompletionToolChoiceOption::None => ToolChoice::None,
        ChatCompletionToolChoiceOption::Auto => ToolChoice::Auto,
        ChatCompletionToolChoiceOption::Required => ToolChoice::Required,
        ChatCompletionToolChoiceOption::Named(named) => {
            ToolChoice::Named(named.function.name.clone())
        }
    }
}

fn convert_tools(tools: &[ChatCompletionTool]) -> Vec<ToolDefinition> {
    tools
        .iter()
        .map(|tool| ToolDefinition {
            name: tool.function.name.clone(),
            parameters: tool.function.parameters.clone(),
            strict: tool.function.strict,
        })
        .collect()
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::common::{OutputOptions, SamplingOptions, StopConditions};
    use serde_json::json;

    fn preprocessed_request() -> PreprocessedRequest {
        PreprocessedRequest::builder()
            .model("kimi-k3".to_string())
            .token_ids(vec![1])
            .stop_conditions(StopConditions::default())
            .sampling_options(SamplingOptions::default())
            .output_options(OutputOptions::default())
            .build()
            .unwrap()
    }

    fn tool(name: &str, strict: Option<bool>) -> ToolDefinition {
        ToolDefinition {
            name: name.to_string(),
            parameters: Some(json!({"type": "object"})),
            strict,
        }
    }

    /// Every marker a branch may legally begin with. A mandatory element ends
    /// the walk: nothing after it can be the first thing the model emits.
    fn first_markers(format: &serde_json::Value, out: &mut Vec<String>) {
        match format["type"].as_str() {
            Some("sequence") => {
                for element in format["elements"].as_array().unwrap() {
                    first_markers(element, out);
                    if element["type"] != "optional" {
                        return;
                    }
                }
            }
            Some("or") => {
                for element in format["elements"].as_array().unwrap() {
                    first_markers(element, out);
                }
            }
            Some("optional") => first_markers(&format["content"], out),
            Some("const_string") => out.push(format["value"].as_str().unwrap().to_string()),
            Some("tag") => out.push(format["begin"].as_str().unwrap().to_string()),
            other => out.push(format!("<free text: {}>", other.unwrap_or("?"))),
        }
    }

    fn applied_tag(
        choice: &ToolChoice,
        tools: &[ToolDefinition],
        thinking: bool,
    ) -> Option<serde_json::Value> {
        let mut request = preprocessed_request();
        let applied = apply_kimi_k3_structural_tag(
            StructuralTagMode::On,
            choice,
            tools,
            thinking,
            &mut request,
        )
        .unwrap();
        if !applied {
            return None;
        }
        Some(
            request
                .sampling_options
                .guided_decoding
                .unwrap()
                .structural_tag
                .unwrap(),
        )
    }

    /// A thinking-on turn may only begin at a channel-open marker. If free text
    /// or `<|close|>message<|sep|>` were reachable first, a model reaching to
    /// end the turn would commit part of that marker before the exclusion
    /// masked its final token, and strand mid-marker.
    #[test]
    fn k3_thinking_on_only_ever_starts_at_a_channel_open_marker() {
        let tools = [tool("lookup", None)];
        for (label, choice, tools) in [
            ("auto", ToolChoice::Auto, &tools[..]),
            ("required", ToolChoice::Required, &tools[..]),
            ("none", ToolChoice::None, &tools[..]),
            ("auto/no-tools", ToolChoice::Auto, &[][..]),
            ("none/no-tools", ToolChoice::None, &[][..]),
        ] {
            let tag = applied_tag(&choice, tools, true)
                .unwrap_or_else(|| panic!("{label} must attach a structural tag"));
            let mut markers = Vec::new();
            first_markers(&tag["format"], &mut markers);
            assert!(
                markers
                    .iter()
                    .all(|m| m == "<|open|>response<|sep|>" || m == "<|open|>tools<|sep|>"),
                "{label}: only channel-open markers may start the turn, got {markers:?}"
            );
        }
    }

    /// The 2026-08-08 BEAM shape: no tools, thinking on. Without a constraint
    /// the model could answer inside the think channel and stop -- 49 of 700
    /// rows, none of which sent a tool.
    #[test]
    fn k3_empty_tools_thinking_on_constrains_the_response_channel() {
        let tag = applied_tag(&ToolChoice::Auto, &[], true)
            .expect("no-tools thinking-on must attach a response-only constraint");
        let elements = tag["format"]["elements"].as_array().unwrap();

        assert_eq!(elements.len(), 3, "{tag}");
        assert_eq!(elements[0]["value"], "<|open|>response<|sep|>");
        assert_eq!(elements[1]["end"], "<|close|>response<|sep|>");
        assert_eq!(elements[2]["content"]["value"], "<|close|>message<|sep|>");
        assert!(
            elements
                .iter()
                .all(|e| e["begin"] != "<|open|>tools<|sep|>"),
            "no tools channel should be emitted: {tag}"
        );
    }

    /// `required` with nothing to call is unsatisfiable, and with thinking off
    /// the prompt has already opened the response channel. Both build nothing,
    /// exactly as before.
    #[test]
    fn k3_empty_tools_builds_nothing_when_it_cannot_help() {
        assert!(applied_tag(&ToolChoice::Required, &[], true).is_none());
        for choice in [ToolChoice::Auto, ToolChoice::None, ToolChoice::Required] {
            assert!(
                applied_tag(&choice, &[], false).is_none(),
                "thinking-off empty tools must stay unconstrained"
            );
        }
    }

    /// Thinking off keeps the permissive shape on every tool policy: the prompt
    /// already emitted the response-open marker, so it stays optional.
    #[test]
    fn k3_thinking_off_keeps_the_optional_response_open_marker() {
        let tools = [tool("lookup", None)];
        for (label, choice) in [
            ("auto", ToolChoice::Auto),
            ("required", ToolChoice::Required),
            ("none", ToolChoice::None),
        ] {
            let tag = applied_tag(&choice, &tools, false)
                .unwrap_or_else(|| panic!("{label} must attach a structural tag"));
            assert!(
                tag.to_string()
                    .contains(r#"{"type":"optional","content":{"type":"const_string","value":"<|open|>response<|sep|>"}}"#),
                "{label}: response-open must stay optional with thinking off: {tag}"
            );
        }
    }

    #[test]
    fn k3_required_attaches_structural_tag_for_backend_xgrammar() {
        let mut request = preprocessed_request();

        let applied = apply_kimi_k3_structural_tag(
            StructuralTagMode::On,
            &ToolChoice::Required,
            &[tool("lookup", None)],
            true,
            &mut request,
        )
        .unwrap();

        assert!(applied);
        let guided = request.sampling_options.guided_decoding.unwrap();
        assert!(guided.json.is_none());
        assert_eq!(
            guided.structural_tag.as_ref().unwrap()["type"],
            "structural_tag"
        );
    }

    #[test]
    fn k3_none_attaches_response_only_structural_tag() {
        let mut request = preprocessed_request();

        let applied = apply_kimi_k3_structural_tag(
            StructuralTagMode::On,
            &ToolChoice::None,
            &[tool("lookup", None)],
            true,
            &mut request,
        )
        .unwrap();

        assert!(applied);
        let structural_tag = request
            .sampling_options
            .guided_decoding
            .unwrap()
            .structural_tag
            .unwrap();
        let elements = structural_tag["format"]["elements"].as_array().unwrap();
        assert_eq!(elements.len(), 3);
        assert!(
            elements
                .iter()
                .all(|element| element["begin"] != "<|open|>tools<|sep|>")
        );
    }

    #[test]
    fn k3_named_is_rejected_even_when_structural_tags_are_disabled() {
        let mut request = preprocessed_request();

        let err = apply_kimi_k3_structural_tag(
            StructuralTagMode::Off,
            &ToolChoice::Named("second".into()),
            &[tool("first", None), tool("second", None)],
            true,
            &mut request,
        )
        .unwrap_err();

        assert!(
            err.to_string()
                .contains("Named tool choice is not supported for Kimi K3")
        );
        assert!(request.sampling_options.guided_decoding.is_none());
    }

    #[test]
    fn k3_required_leaves_xgrammar_inactive_when_structural_tags_are_disabled() {
        let mut request = preprocessed_request();

        assert!(
            !apply_kimi_k3_structural_tag(
                StructuralTagMode::Off,
                &ToolChoice::Required,
                &[tool("lookup", None)],
            true,
                &mut request,
            )
            .unwrap()
        );
        assert!(request.sampling_options.guided_decoding.is_none());
    }

    #[test]
    fn k3_auto_without_strict_tool_attaches_structural_tag() {
        let mut request = preprocessed_request();

        assert!(
            apply_kimi_k3_structural_tag(
                StructuralTagMode::On,
                &ToolChoice::Auto,
                &[tool("lookup", None)],
            true,
                &mut request,
            )
            .unwrap()
        );
        assert!(
            request
                .sampling_options
                .guided_decoding
                .unwrap()
                .structural_tag
                .is_some()
        );
    }

    #[test]
    fn k3_required_constraint_uses_dynamic_tools() {
        let request: NvCreateChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "kimi-k3",
            "messages": [{
                "role": "system",
                "content": "",
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {"type": "object"}
                    }
                }]
            }],
            "tool_choice": "required"
        }))
        .unwrap();
        let mut preprocessed = preprocessed_request();

        assert!(
            apply_kimi_k3_structural_tag(
                StructuralTagMode::On,
                &convert_tool_choice(request.inner.tool_choice.as_ref().unwrap()),
                &convert_tools(&request.effective_tools()),
            true,
                &mut preprocessed,
            )
            .unwrap()
        );
        let structural_tag = preprocessed
            .sampling_options
            .guided_decoding
            .unwrap()
            .structural_tag
            .unwrap();
        assert!(
            serde_json::to_string(&structural_tag)
                .unwrap()
                .contains(r#"tool=\"get_weather\""#)
        );
    }
}
