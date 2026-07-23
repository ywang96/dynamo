// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_protocols::types::ReasoningEffort;

use crate::frontend_config::KimiApiComplianceConfig;
use crate::protocols::openai::chat_completions::NvCreateChatCompletionRequest;

const FLOAT_TOLERANCE: f32 = 1e-6;
const THINKING_TEMPERATURE: f32 = 1.0;
const NON_THINKING_TEMPERATURE: f32 = 0.6;

#[derive(Debug, thiserror::Error)]
#[error("Kimi request field {field} {requirement}")]
pub(crate) struct KimiComplianceError {
    field: &'static str,
    requirement: String,
}

impl KimiComplianceError {
    fn new(field: &'static str, requirement: impl Into<String>) -> Self {
        Self {
            field,
            requirement: requirement.into(),
        }
    }
}

pub(crate) fn normalize_request_json(
    request: &mut serde_json::Value,
    config: &KimiApiComplianceConfig,
) {
    if !config.enabled() {
        return;
    }
    let Some(messages) = request
        .get_mut("messages")
        .and_then(serde_json::Value::as_array_mut)
    else {
        return;
    };
    for message in messages {
        let Some(message) = message.as_object_mut() else {
            continue;
        };
        let is_dynamic_tool_message = message.get("role").and_then(serde_json::Value::as_str)
            == Some("system")
            && message.get("tools").is_some_and(|tools| !tools.is_null());
        if is_dynamic_tool_message && !message.contains_key("content") {
            message.insert(
                "content".to_string(),
                serde_json::Value::String(String::new()),
            );
        }
    }
}

pub(crate) fn apply(
    request: &mut NvCreateChatCompletionRequest,
    config: &KimiApiComplianceConfig,
) -> Result<(), KimiComplianceError> {
    if !config.enabled() {
        return Ok(());
    }

    let reasoning_effort_none = matches!(
        request.inner.reasoning_effort.as_ref(),
        Some(ReasoningEffort::None)
    );
    if reasoning_effort_none && request.thinking.is_none() {
        request.thinking = Some(serde_json::json!({"type": "disabled"}));
    }

    let thinking = match request.thinking.as_ref() {
        None => None,
        Some(serde_json::Value::Object(thinking)) => Some(thinking),
        Some(_) => {
            return Err(KimiComplianceError::new("thinking", "must be an object"));
        }
    };
    let thinking_type = match thinking.and_then(|thinking| thinking.get("type")) {
        None => "enabled",
        Some(serde_json::Value::String(thinking_type)) => thinking_type.as_str(),
        Some(_) => {
            return Err(allowed_values_error(
                "thinking.type",
                config.allowed_thinking_types(),
            ));
        }
    };
    if !config
        .allowed_thinking_types()
        .iter()
        .any(|allowed| allowed == thinking_type)
    {
        return Err(allowed_values_error(
            "thinking.type",
            config.allowed_thinking_types(),
        ));
    }

    let thinking_enabled = thinking_type == "enabled";
    if thinking_enabled {
        if let Some(effort) = request.inner.reasoning_effort.as_ref() {
            validate_effort("reasoning_effort", reasoning_effort_name(effort), config)?;
        }
        let nested_effort = thinking.and_then(|thinking| thinking.get("effort"));
        if let Some(effort) = nested_effort {
            let effort = effort
                .as_str()
                .ok_or_else(|| KimiComplianceError::new("thinking.effort", "must be a string"))?;
            validate_effort("thinking.effort", effort, config)?;
        }
        if let Some(keep) = thinking.and_then(|thinking| thinking.get("keep")) {
            if !matches!(keep.as_str(), Some("all" | "interleaved")) {
                return Err(KimiComplianceError::new(
                    "thinking.keep",
                    "must be one of: all, interleaved",
                ));
            }
        }
        if request.inner.reasoning_effort.is_none() && nested_effort.is_none() {
            request.inner.reasoning_effort = Some(configured_reasoning_effort(config)?);
        }
    } else if reasoning_effort_none {
        request.inner.reasoning_effort = None;
    } else if request.inner.reasoning_effort.is_some() {
        return Err(KimiComplianceError::new(
            "reasoning_effort",
            "requires thinking.type=enabled",
        ));
    }

    let expected_temperature = if thinking_enabled {
        THINKING_TEMPERATURE
    } else {
        NON_THINKING_TEMPERATURE
    };
    validate_float(
        "temperature",
        request.inner.temperature,
        expected_temperature,
    )?;
    if let Some(top_p) = request.inner.top_p {
        if !config
            .allowed_top_p()
            .iter()
            .any(|allowed| nearly_equal(top_p, *allowed))
        {
            return Err(KimiComplianceError::new(
                "top_p",
                format!("must be one of: {}", join_values(config.allowed_top_p())),
            ));
        }
    }
    validate_float("presence_penalty", request.inner.presence_penalty, 0.0)?;
    validate_float("frequency_penalty", request.inner.frequency_penalty, 0.0)?;
    if request.inner.n.is_some_and(|n| n != 1) {
        return Err(KimiComplianceError::new("n", "must be 1"));
    }

    let default_top_p = config
        .allowed_top_p()
        .iter()
        .copied()
        .find(|top_p| nearly_equal(*top_p, 1.0))
        .or_else(|| config.allowed_top_p().first().copied())
        .ok_or_else(|| {
            KimiComplianceError::new("top_p", "requires a non-empty configured allowlist")
        })?;
    request
        .inner
        .temperature
        .get_or_insert(expected_temperature);
    request.inner.top_p.get_or_insert(default_top_p);
    request.inner.presence_penalty.get_or_insert(0.0);
    request.inner.frequency_penalty.get_or_insert(0.0);
    request.inner.n.get_or_insert(1);
    #[allow(deprecated)]
    if request.inner.max_tokens.is_none() && request.inner.max_completion_tokens.is_none() {
        request.inner.max_completion_tokens = Some(config.default_max_completion_tokens());
    }
    if request.thinking.is_none() {
        request.thinking = Some(serde_json::json!({"type": "enabled"}));
    }
    Ok(())
}

fn validate_float(
    field: &'static str,
    value: Option<f32>,
    expected: f32,
) -> Result<(), KimiComplianceError> {
    if value.is_some_and(|value| !nearly_equal(value, expected)) {
        return Err(KimiComplianceError::new(
            field,
            format!("must be {expected}"),
        ));
    }
    Ok(())
}

fn validate_effort(
    field: &'static str,
    effort: &str,
    config: &KimiApiComplianceConfig,
) -> Result<(), KimiComplianceError> {
    if !config
        .allowed_reasoning_efforts()
        .iter()
        .any(|allowed| allowed == effort)
    {
        return Err(allowed_values_error(
            field,
            config.allowed_reasoning_efforts(),
        ));
    }
    Ok(())
}

fn configured_reasoning_effort(
    config: &KimiApiComplianceConfig,
) -> Result<ReasoningEffort, KimiComplianceError> {
    let effort = match config.default_reasoning_effort() {
        "low" => ReasoningEffort::Low,
        "high" => ReasoningEffort::High,
        "max" => ReasoningEffort::Max,
        _ => {
            return Err(allowed_values_error(
                "reasoning_effort",
                config.allowed_reasoning_efforts(),
            ));
        }
    };
    Ok(effort)
}

fn reasoning_effort_name(effort: &ReasoningEffort) -> &'static str {
    match effort {
        ReasoningEffort::None => "none",
        ReasoningEffort::Minimal => "minimal",
        ReasoningEffort::Low => "low",
        ReasoningEffort::Medium => "medium",
        ReasoningEffort::High => "high",
        ReasoningEffort::Xhigh => "xhigh",
        ReasoningEffort::Max => "max",
    }
}

fn allowed_values_error<T: std::fmt::Display>(
    field: &'static str,
    allowed: &[T],
) -> KimiComplianceError {
    KimiComplianceError::new(field, format!("must be one of: {}", join_values(allowed)))
}

fn join_values<T: std::fmt::Display>(values: &[T]) -> String {
    values
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(", ")
}

fn nearly_equal(left: f32, right: f32) -> bool {
    (left - right).abs() < FLOAT_TOLERANCE
}

#[cfg(test)]
mod tests {
    use dynamo_protocols::types::ReasoningEffort;
    use serde_json::{Value, json};

    use crate::frontend_config::KimiApiComplianceConfig;
    use crate::protocols::openai::chat_completions::NvCreateChatCompletionRequest;

    use super::*;

    fn request(extra: Value) -> NvCreateChatCompletionRequest {
        let mut body = json!({
            "model": "moonshotai/Kimi-K3-Instruct",
            "messages": [{"role": "user", "content": "hello"}]
        });
        body.as_object_mut()
            .expect("base request is an object")
            .extend(extra.as_object().expect("extra is an object").clone());
        serde_json::from_value(body).expect("valid chat request")
    }

    fn config(
        max_completion_tokens: Option<u32>,
        thinking_types: Option<Vec<&str>>,
        default_effort: Option<&str>,
        efforts: Option<Vec<&str>>,
        top_p: Option<Vec<f32>>,
    ) -> KimiApiComplianceConfig {
        KimiApiComplianceConfig::from_optional_flags(
            Some(true),
            max_completion_tokens,
            thinking_types.map(|values| values.into_iter().map(str::to_string).collect()),
            default_effort.map(str::to_string),
            efforts.map(|values| values.into_iter().map(str::to_string).collect()),
            top_p,
        )
        .expect("valid config")
        .expect("explicit enabled flag creates config")
    }

    fn enabled_config() -> KimiApiComplianceConfig {
        config(None, None, None, None, None)
    }

    fn narrowed_config() -> KimiApiComplianceConfig {
        config(
            Some(131_072),
            Some(vec!["enabled"]),
            Some("max"),
            Some(vec!["max"]),
            Some(vec![0.95]),
        )
    }

    #[test]
    fn omitted_fields_receive_pr85_defaults() {
        let mut request = request(json!({}));

        apply(&mut request, &enabled_config()).unwrap();

        assert_eq!(request.inner.temperature, Some(1.0));
        assert_eq!(request.inner.top_p, Some(1.0));
        assert_eq!(request.inner.presence_penalty, Some(0.0));
        assert_eq!(request.inner.frequency_penalty, Some(0.0));
        assert_eq!(request.inner.n, Some(1));
        assert_eq!(request.inner.max_completion_tokens, Some(32_768));
        assert_eq!(request.inner.reasoning_effort, Some(ReasoningEffort::Max));
        assert_eq!(request.thinking, Some(json!({"type": "enabled"})));
    }

    #[test]
    fn narrowed_policy_defaults_to_deployment_values() {
        let mut request = request(json!({}));

        apply(&mut request, &narrowed_config()).unwrap();

        assert_eq!(request.inner.top_p, Some(0.95));
        assert_eq!(request.inner.max_completion_tokens, Some(131_072));
        assert_eq!(request.inner.reasoning_effort, Some(ReasoningEffort::Max));
    }

    #[test]
    fn narrowed_policy_rejects_disabled_and_other_top_p() {
        let config = narrowed_config();
        let mut disabled = request(json!({"thinking": {"type": "disabled"}}));
        assert!(
            apply(&mut disabled, &config)
                .unwrap_err()
                .to_string()
                .contains("thinking.type")
        );

        let mut top_p = request(json!({"top_p": 1.0}));
        assert!(
            apply(&mut top_p, &config)
                .unwrap_err()
                .to_string()
                .contains("top_p")
        );
    }

    #[test]
    fn explicit_allowed_zero_top_p_is_preserved() {
        let mut request = request(json!({"top_p": 0.0}));
        let config = config(None, None, None, None, Some(vec![0.0, 1.0]));

        apply(&mut request, &config).unwrap();

        assert_eq!(request.inner.top_p, Some(0.0));
    }

    #[test]
    fn disabled_thinking_ignores_nested_extras() {
        let mut request = request(json!({
            "thinking": {"type": "disabled", "effort": 17, "keep": "invalid"}
        }));

        apply(&mut request, &enabled_config()).unwrap();

        assert_eq!(request.inner.temperature, Some(0.6));
        assert_eq!(request.inner.reasoning_effort, None);
    }

    #[test]
    fn reasoning_effort_none_selects_non_thinking() {
        let mut request = request(json!({"reasoning_effort": "none"}));

        apply(&mut request, &enabled_config()).unwrap();

        assert_eq!(request.inner.temperature, Some(0.6));
        assert_eq!(request.inner.reasoning_effort, None);
        assert_eq!(request.thinking, Some(json!({"type": "disabled"})));
    }

    #[test]
    fn explicit_disabled_accepts_redundant_none_effort() {
        let mut request = request(json!({
            "thinking": {"type": "disabled"},
            "reasoning_effort": "none"
        }));

        apply(&mut request, &enabled_config()).unwrap();

        assert_eq!(request.inner.reasoning_effort, None);
        assert_eq!(request.inner.temperature, Some(0.6));
    }

    #[test]
    fn disabled_thinking_rejects_top_level_reasoning_effort() {
        let mut request = request(json!({
            "thinking": {"type": "disabled"},
            "reasoning_effort": "max"
        }));

        let error = apply(&mut request, &enabled_config()).unwrap_err();

        assert!(error.to_string().contains("reasoning_effort"));
        assert!(error.to_string().contains("thinking.type=enabled"));
    }

    #[test]
    #[allow(deprecated)]
    fn explicit_completion_token_aliases_are_preserved() {
        let mut legacy = request(json!({"max_tokens": 7}));
        apply(&mut legacy, &enabled_config()).unwrap();
        assert_eq!(legacy.inner.max_tokens, Some(7));
        assert_eq!(legacy.inner.max_completion_tokens, None);

        let mut current = request(json!({"max_completion_tokens": 9}));
        apply(&mut current, &enabled_config()).unwrap();
        assert_eq!(current.inner.max_completion_tokens, Some(9));
        assert_eq!(current.inner.max_tokens, None);
    }

    #[test]
    fn explicit_sampling_values_are_validated() {
        for (payload, field) in [
            (json!({"temperature": 0.6}), "temperature"),
            (json!({"top_p": 0.5}), "top_p"),
            (json!({"presence_penalty": 1.0}), "presence_penalty"),
            (json!({"frequency_penalty": -1.0}), "frequency_penalty"),
            (json!({"n": 2}), "n"),
        ] {
            let mut request = request(payload);
            let error = apply(&mut request, &enabled_config()).unwrap_err();
            assert!(error.to_string().contains(field), "{error}");
        }
    }

    #[test]
    fn enabled_thinking_validates_both_effort_aliases_and_keep() {
        for (payload, field) in [
            (json!({"reasoning_effort": "medium"}), "reasoning_effort"),
            (
                json!({"thinking": {"type": "enabled", "effort": "medium"}}),
                "thinking.effort",
            ),
            (
                json!({
                    "reasoning_effort": "low",
                    "thinking": {"type": "enabled", "effort": "medium"}
                }),
                "thinking.effort",
            ),
            (
                json!({"thinking": {"type": "enabled", "keep": "invalid"}}),
                "thinking.keep",
            ),
        ] {
            let mut request = request(payload);
            let error = apply(&mut request, &enabled_config()).unwrap_err();
            assert!(error.to_string().contains(field), "{error}");
        }
    }

    #[test]
    fn thinking_must_be_an_object() {
        for payload in [json!({"thinking": true}), json!({"thinking": "enabled"})] {
            let mut request = request(payload);
            let error = apply(&mut request, &enabled_config()).unwrap_err();
            assert!(error.to_string().contains("thinking"), "{error}");
            assert!(error.to_string().contains("object"), "{error}");
        }
    }

    #[test]
    fn disabled_config_is_a_no_op() {
        let mut request = request(json!({
            "temperature": 0.2,
            "top_p": 0.3,
            "thinking": {"type": "bogus"}
        }));
        let original = serde_json::to_value(&request).unwrap();

        apply(&mut request, &KimiApiComplianceConfig::default()).unwrap();

        assert_eq!(serde_json::to_value(&request).unwrap(), original);
    }

    #[test]
    fn dynamic_tool_messages_get_empty_content_only_when_enabled() {
        let original = json!({
            "messages": [
                {"role": "system", "tools": []},
                {"role": "system", "content": "keep", "tools": []},
                {"role": "system"},
                {"role": "user", "content": "hello"}
            ]
        });

        let mut enabled = original.clone();
        normalize_request_json(&mut enabled, &enabled_config());
        assert_eq!(enabled["messages"][0]["content"], "");
        assert_eq!(enabled["messages"][1]["content"], "keep");
        assert!(enabled["messages"][2].get("content").is_none());

        let mut disabled = original.clone();
        normalize_request_json(&mut disabled, &KimiApiComplianceConfig::default());
        assert_eq!(disabled, original);
    }
}
