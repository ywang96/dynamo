// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Frontend-owned configuration groups shared by the Python entrypoint,
//! `LocalModel`, and HTTP/gRPC service setup.
//!
//! Python may expose these as flat CLI flags for compatibility, but Rust stores
//! them by domain so each service consumes an explicit typed contract. Defaults
//! read the legacy environment variables only for direct Rust/non-Python callers.

use dynamo_runtime::config::{
    env_is_truthy,
    environment_names::llm::{self as env_llm, metrics as env_metrics},
};

/// Metrics naming controls for frontend-owned services.
///
/// Contains the optional metric name prefix resolved from `--metrics-prefix` or
/// `DYN_METRICS_PREFIX`. HTTP services use it when constructing
/// `http::service::metrics::Metrics`; gRPC mode also exposes the prefix through
/// the existing `LocalModel::metrics_prefix()` compatibility accessor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetricsConfig {
    prefix: Option<String>,
}

impl MetricsConfig {
    pub fn new(prefix: Option<String>) -> Self {
        Self { prefix }
    }

    pub fn prefix(&self) -> Option<String> {
        self.prefix.clone()
    }

    pub fn set_prefix(&mut self, prefix: Option<String>) {
        self.prefix = prefix;
    }
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            prefix: std::env::var(env_metrics::DYN_METRICS_PREFIX).ok(),
        }
    }
}

/// Anthropic API surface controls.
///
/// Contains whether the experimental Anthropic Messages API routes are exposed
/// and whether Anthropic billing preambles are stripped from requests. The HTTP
/// service uses these values to choose Anthropic vs OpenAI model routes, enable
/// `/v1/messages`, and drive Anthropic request handling.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnthropicApiConfig {
    enabled: bool,
    strip_preamble: bool,
}

impl AnthropicApiConfig {
    pub fn new(enabled: bool, strip_preamble: bool) -> Self {
        Self {
            enabled,
            strip_preamble,
        }
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    pub fn strip_preamble(&self) -> bool {
        self.strip_preamble
    }

    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    pub fn set_strip_preamble(&mut self, strip_preamble: bool) {
        self.strip_preamble = strip_preamble;
    }
}

impl Default for AnthropicApiConfig {
    fn default() -> Self {
        Self {
            enabled: env_is_truthy(env_llm::DYN_ENABLE_ANTHROPIC_API),
            strip_preamble: env_is_truthy(env_llm::DYN_STRIP_ANTHROPIC_PREAMBLE),
        }
    }
}

/// Streaming-specific response dispatch controls.
///
/// Contains the OpenAI-compatible streaming toggles for tool-call dispatch and
/// reasoning dispatch events. HTTP request handlers read these values from
/// shared service state when deciding whether to emit the extra SSE events.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamingDispatchConfig {
    tool_dispatch: bool,
    reasoning_dispatch: bool,
}

impl StreamingDispatchConfig {
    pub fn new(tool_dispatch: bool, reasoning_dispatch: bool) -> Self {
        Self {
            tool_dispatch,
            reasoning_dispatch,
        }
    }

    pub fn tool_dispatch(&self) -> bool {
        self.tool_dispatch
    }

    pub fn reasoning_dispatch(&self) -> bool {
        self.reasoning_dispatch
    }

    pub fn set_tool_dispatch(&mut self, tool_dispatch: bool) {
        self.tool_dispatch = tool_dispatch;
    }

    pub fn set_reasoning_dispatch(&mut self, reasoning_dispatch: bool) {
        self.reasoning_dispatch = reasoning_dispatch;
    }
}

impl Default for StreamingDispatchConfig {
    fn default() -> Self {
        Self {
            tool_dispatch: env_is_truthy(env_llm::DYN_ENABLE_STREAMING_TOOL_DISPATCH),
            reasoning_dispatch: env_is_truthy(env_llm::DYN_ENABLE_STREAMING_REASONING_DISPATCH),
        }
    }
}

pub const KIMI_DEFAULT_MAX_COMPLETION_TOKENS: u32 = 32_768;
pub const KIMI_DEFAULT_REASONING_EFFORT: &str = "max";
pub const KIMI_ALLOWED_THINKING_TYPES: &[&str] = &["enabled", "disabled"];
pub const KIMI_ALLOWED_REASONING_EFFORTS: &[&str] = &["low", "high", "max"];
pub const KIMI_ALLOWED_TOP_P: &[f32] = &[0.95, 1.0];

/// Kimi API request defaults and allowlists.
///
/// The frontend enables this policy explicitly. It never infers Kimi behavior
/// from the requested model name.
#[derive(Debug, Clone, PartialEq)]
pub struct KimiApiComplianceConfig {
    enabled: bool,
    default_max_completion_tokens: u32,
    allowed_thinking_types: Vec<String>,
    default_reasoning_effort: String,
    allowed_reasoning_efforts: Vec<String>,
    allowed_top_p: Vec<f32>,
}

impl Default for KimiApiComplianceConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            default_max_completion_tokens: KIMI_DEFAULT_MAX_COMPLETION_TOKENS,
            allowed_thinking_types: KIMI_ALLOWED_THINKING_TYPES
                .iter()
                .map(|value| (*value).to_string())
                .collect(),
            default_reasoning_effort: KIMI_DEFAULT_REASONING_EFFORT.to_string(),
            allowed_reasoning_efforts: KIMI_ALLOWED_REASONING_EFFORTS
                .iter()
                .map(|value| (*value).to_string())
                .collect(),
            allowed_top_p: KIMI_ALLOWED_TOP_P.to_vec(),
        }
    }
}

impl KimiApiComplianceConfig {
    #[allow(clippy::too_many_arguments)]
    pub fn from_optional_flags(
        enabled: Option<bool>,
        default_max_completion_tokens: Option<u32>,
        allowed_thinking_types: Option<Vec<String>>,
        default_reasoning_effort: Option<String>,
        allowed_reasoning_efforts: Option<Vec<String>>,
        allowed_top_p: Option<Vec<f32>>,
    ) -> Result<Option<Self>, String> {
        if enabled.is_none()
            && default_max_completion_tokens.is_none()
            && allowed_thinking_types.is_none()
            && default_reasoning_effort.is_none()
            && allowed_reasoning_efforts.is_none()
            && allowed_top_p.is_none()
        {
            return Ok(None);
        }

        let defaults = Self::default();
        let config = Self {
            enabled: enabled.unwrap_or(defaults.enabled),
            default_max_completion_tokens: default_max_completion_tokens
                .unwrap_or(defaults.default_max_completion_tokens),
            allowed_thinking_types: allowed_thinking_types
                .unwrap_or_else(|| defaults.allowed_thinking_types.clone()),
            default_reasoning_effort: default_reasoning_effort
                .unwrap_or_else(|| defaults.default_reasoning_effort.clone()),
            allowed_reasoning_efforts: allowed_reasoning_efforts
                .unwrap_or_else(|| defaults.allowed_reasoning_efforts.clone()),
            allowed_top_p: allowed_top_p.unwrap_or_else(|| defaults.allowed_top_p.clone()),
        };
        config.validate()?;
        Ok(Some(config))
    }

    fn validate(&self) -> Result<(), String> {
        if self.default_max_completion_tokens == 0 {
            return Err("--kimi-default-max-completion-tokens must be >= 1".to_string());
        }
        if self.allowed_thinking_types.is_empty()
            || !self
                .allowed_thinking_types
                .iter()
                .any(|value| value == "enabled")
            || self
                .allowed_thinking_types
                .iter()
                .any(|value| !KIMI_ALLOWED_THINKING_TYPES.contains(&value.as_str()))
        {
            return Err(
                "--kimi-allowed-thinking-types must contain enabled and only enabled,disabled"
                    .to_string(),
            );
        }
        if self.allowed_reasoning_efforts.is_empty()
            || self
                .allowed_reasoning_efforts
                .iter()
                .any(|value| !KIMI_ALLOWED_REASONING_EFFORTS.contains(&value.as_str()))
            || !self
                .allowed_reasoning_efforts
                .contains(&self.default_reasoning_effort)
        {
            return Err(
                "--kimi-default-reasoning-effort must belong to a non-empty valid allowlist"
                    .to_string(),
            );
        }
        if self.allowed_top_p.is_empty()
            || self
                .allowed_top_p
                .iter()
                .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
        {
            return Err(
                "--kimi-allowed-top-p must contain finite values between 0 and 1".to_string(),
            );
        }
        Ok(())
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    pub fn default_max_completion_tokens(&self) -> u32 {
        self.default_max_completion_tokens
    }

    pub fn allowed_thinking_types(&self) -> &[String] {
        &self.allowed_thinking_types
    }

    pub fn default_reasoning_effort(&self) -> &str {
        &self.default_reasoning_effort
    }

    pub fn allowed_reasoning_efforts(&self) -> &[String] {
        &self.allowed_reasoning_efforts
    }

    pub fn allowed_top_p(&self) -> &[f32] {
        &self.allowed_top_p
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutoToolChoiceOverrideMode {
    All,
    Strict,
}

impl std::str::FromStr for AutoToolChoiceOverrideMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "all" => Ok(Self::All),
            "strict" => Ok(Self::Strict),
            _ => Err(
                "--override-auto-tool-choice-to-required must be one of: all, strict".to_string(),
            ),
        }
    }
}

/// Frontend API behavior consumed by the HTTP service.
///
/// Groups endpoint-surface and streaming-behavior settings that originate from
/// the frontend CLI/env contract. `EntrypointArgs` builds this from flat Python
/// kwargs, `LocalModel` carries it, and `HttpServiceConfig` installs it into
/// request-handler state.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FrontendApiConfig {
    anthropic: AnthropicApiConfig,
    streaming_dispatch: StreamingDispatchConfig,
    kimi_api_compliance: KimiApiComplianceConfig,
    override_auto_tool_choice_to_required: Option<AutoToolChoiceOverrideMode>,
}

impl FrontendApiConfig {
    pub fn new(
        anthropic: AnthropicApiConfig,
        streaming_dispatch: StreamingDispatchConfig,
        kimi_api_compliance: KimiApiComplianceConfig,
        override_auto_tool_choice_to_required: Option<AutoToolChoiceOverrideMode>,
    ) -> Self {
        Self {
            anthropic,
            streaming_dispatch,
            kimi_api_compliance,
            override_auto_tool_choice_to_required,
        }
    }

    pub fn from_flags(
        enable_anthropic_api: bool,
        strip_anthropic_preamble: bool,
        enable_streaming_tool_dispatch: bool,
        enable_streaming_reasoning_dispatch: bool,
        kimi_api_compliance: KimiApiComplianceConfig,
        override_auto_tool_choice_to_required: Option<AutoToolChoiceOverrideMode>,
    ) -> Self {
        Self {
            anthropic: AnthropicApiConfig::new(enable_anthropic_api, strip_anthropic_preamble),
            streaming_dispatch: StreamingDispatchConfig::new(
                enable_streaming_tool_dispatch,
                enable_streaming_reasoning_dispatch,
            ),
            kimi_api_compliance,
            override_auto_tool_choice_to_required,
        }
    }
    pub fn from_optional_flags(
        enable_anthropic_api: Option<bool>,
        strip_anthropic_preamble: Option<bool>,
        enable_streaming_tool_dispatch: Option<bool>,
        enable_streaming_reasoning_dispatch: Option<bool>,
        kimi_api_compliance: Option<KimiApiComplianceConfig>,
        override_auto_tool_choice_to_required: Option<AutoToolChoiceOverrideMode>,
    ) -> Option<Self> {
        if enable_anthropic_api.is_none()
            && strip_anthropic_preamble.is_none()
            && enable_streaming_tool_dispatch.is_none()
            && enable_streaming_reasoning_dispatch.is_none()
            && kimi_api_compliance.is_none()
            && override_auto_tool_choice_to_required.is_none()
        {
            return None;
        }
        let defaults = Self::default();
        Some(Self::from_flags(
            enable_anthropic_api.unwrap_or_else(|| defaults.anthropic().enabled()),
            strip_anthropic_preamble.unwrap_or_else(|| defaults.anthropic().strip_preamble()),
            enable_streaming_tool_dispatch
                .unwrap_or_else(|| defaults.streaming_dispatch().tool_dispatch()),
            enable_streaming_reasoning_dispatch
                .unwrap_or_else(|| defaults.streaming_dispatch().reasoning_dispatch()),
            kimi_api_compliance.unwrap_or_else(|| defaults.kimi_api_compliance().clone()),
            override_auto_tool_choice_to_required
                .or(defaults.override_auto_tool_choice_to_required()),
        ))
    }
    pub fn anthropic(&self) -> &AnthropicApiConfig {
        &self.anthropic
    }

    pub fn anthropic_mut(&mut self) -> &mut AnthropicApiConfig {
        &mut self.anthropic
    }

    pub fn streaming_dispatch(&self) -> &StreamingDispatchConfig {
        &self.streaming_dispatch
    }

    pub fn streaming_dispatch_mut(&mut self) -> &mut StreamingDispatchConfig {
        &mut self.streaming_dispatch
    }

    pub fn kimi_api_compliance(&self) -> &KimiApiComplianceConfig {
        &self.kimi_api_compliance
    }

    pub fn override_auto_tool_choice_to_required(&self) -> Option<AutoToolChoiceOverrideMode> {
        self.override_auto_tool_choice_to_required
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn optional_flags_return_none_when_all_values_are_unspecified() {
        let config = FrontendApiConfig::from_optional_flags(None, None, None, None, None, None);

        assert_eq!(config, None);
    }
    #[test]
    fn optional_flags_preserve_explicit_values() {
        let config = FrontendApiConfig::from_optional_flags(
            Some(false),
            Some(true),
            Some(false),
            Some(true),
            None,
            Some(AutoToolChoiceOverrideMode::Strict),
        )
        .expect("explicit flags should produce a config");
        assert!(!config.anthropic().enabled());
        assert!(config.anthropic().strip_preamble());
        assert!(!config.streaming_dispatch().tool_dispatch());
        assert!(config.streaming_dispatch().reasoning_dispatch());
        assert_eq!(
            config.override_auto_tool_choice_to_required(),
            Some(AutoToolChoiceOverrideMode::Strict)
        );
    }

    #[test]
    fn optional_flags_use_env_defaults_for_unspecified_values() {
        temp_env::with_vars(
            [
                (env_llm::DYN_ENABLE_ANTHROPIC_API, Some("1")),
                (env_llm::DYN_STRIP_ANTHROPIC_PREAMBLE, Some("1")),
                (env_llm::DYN_ENABLE_STREAMING_TOOL_DISPATCH, Some("1")),
                (env_llm::DYN_ENABLE_STREAMING_REASONING_DISPATCH, Some("1")),
            ],
            || {
                let config = FrontendApiConfig::from_optional_flags(
                    Some(false),
                    None,
                    None,
                    Some(false),
                    None,
                    None,
                )
                .expect("partial flags should produce a config");

                assert!(!config.anthropic().enabled());
                assert!(config.anthropic().strip_preamble());
                assert!(config.streaming_dispatch().tool_dispatch());
                assert!(!config.streaming_dispatch().reasoning_dispatch());
            },
        );
    }

    #[test]
    fn kimi_config_preserves_narrowed_values() {
        let config = KimiApiComplianceConfig::from_optional_flags(
            Some(true),
            Some(131_072),
            Some(vec!["enabled".into()]),
            Some("max".into()),
            Some(vec!["max".into()]),
            Some(vec![0.95]),
        )
        .expect("valid Kimi config")
        .expect("explicit values should produce a config");

        assert!(config.enabled());
        assert_eq!(config.default_max_completion_tokens(), 131_072);
        assert_eq!(config.allowed_thinking_types(), &["enabled"]);
        assert_eq!(config.default_reasoning_effort(), "max");
        assert_eq!(config.allowed_reasoning_efforts(), &["max"]);
        assert_eq!(config.allowed_top_p(), &[0.95]);
    }

    #[test]
    fn kimi_config_returns_none_when_unspecified() {
        let config =
            KimiApiComplianceConfig::from_optional_flags(None, None, None, None, None, None)
                .expect("unspecified config is valid");

        assert_eq!(config, None);
    }

    #[test]
    fn kimi_config_rejects_invalid_values() {
        assert!(
            KimiApiComplianceConfig::from_optional_flags(
                Some(true),
                Some(0),
                None,
                None,
                None,
                None,
            )
            .is_err()
        );
        assert!(
            KimiApiComplianceConfig::from_optional_flags(
                Some(true),
                None,
                Some(vec!["disabled".into()]),
                None,
                None,
                None,
            )
            .is_err()
        );
        assert!(
            KimiApiComplianceConfig::from_optional_flags(
                Some(true),
                None,
                None,
                Some("max".into()),
                Some(vec!["high".into()]),
                None,
            )
            .is_err()
        );
        assert!(
            KimiApiComplianceConfig::from_optional_flags(
                Some(true),
                None,
                None,
                None,
                None,
                Some(Vec::new()),
            )
            .is_err()
        );
    }
}
