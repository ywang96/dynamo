// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! MFJS (Moonshot-Flavored JSON Schema) validation via MoonshotAI/walle.
//!
//! Moonshot's server validates every `tools[].function.parameters` schema with
//! walle at API ingress. We match that by FFI'ing walle's c-shared library
//! (`libwalle.so`, exported `ValidateSchema` / `FreeErrString`).
//!
//! Gated behind the `walle-validation` cargo feature: when it is off (the
//! default), the frontend is not linked against walle; [`validate_schema`] is a
//! no-op that accepts everything — so normal builds need neither the native lib
//! nor Go. The runtime `--kimi-schema-validation` flag only has an effect when
//! the feature is compiled in (see [`is_available`]).
//!
//! Schema *canonicalization* for constrained decoding (walle `CanonicalSchema`)
//! is intentionally not ported here: Dynamo emits decode constraints to the
//! backend engine rather than compiling a grammar in-process, so the
//! canonicalize + bound-clamp path has no consumer on this side yet.

use crate::preprocessor::OpenAIPreprocessor;
use crate::protocols::openai::chat_completions::NvCreateChatCompletionRequest;
use dynamo_runtime::error::{DynamoError, ErrorType};

/// `true` if walle validation was compiled into this binary.
pub(crate) const fn is_available() -> bool {
    cfg!(feature = "walle-validation")
}

/// Walle validation level applied at ingress.
///
/// `strict` is the level required by the Moonshot AI server; `lite` accepts a
/// strict superset of it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum WalleLevel {
    #[default]
    Strict,
    Lite,
}

impl WalleLevel {
    /// Parse a config string ("strict" / "lite", case-insensitive).
    pub(crate) fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "strict" => Some(Self::Strict),
            "lite" => Some(Self::Lite),
            _ => None,
        }
    }

    #[cfg(feature = "walle-validation")]
    fn config_json(self) -> &'static str {
        match self {
            Self::Strict => r#"{"validateLevel":"strict"}"#,
            Self::Lite => r#"{"validateLevel":"lite"}"#,
        }
    }
}

impl std::fmt::Display for WalleLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Strict => "strict",
            Self::Lite => "lite",
        })
    }
}

impl OpenAIPreprocessor {
    /// Reject tool `parameters` schemas that walle rejects at the configured
    /// level, matching Moonshot's API-ingress behavior.
    ///
    /// No-op unless `--kimi-schema-validation` is set AND the frontend was built
    /// with the `walle-validation` feature. Returns a 400-mapping [`DynamoError`]
    /// (`ErrorType::InvalidArgument`) on the first schema walle rejects.
    pub(super) fn validate_kimi_tool_schemas(
        &self,
        request: &NvCreateChatCompletionRequest,
    ) -> Result<(), DynamoError> {
        if self.runtime_config.kimi_schema_validation != Some(true) || !is_available() {
            return Ok(());
        }

        let level = self
            .runtime_config
            .kimi_schema_validation_level
            .as_deref()
            .and_then(WalleLevel::parse)
            .unwrap_or_default();

        for tool in request
            .inner
            .tools
            .iter()
            .flatten()
            .chain(request.dynamic_tools())
        {
            let Some(params) = tool.function.parameters.as_ref() else {
                continue;
            };
            if let Err(msg) = validate_schema(&params.to_string(), level) {
                return Err(DynamoError::builder()
                    .error_type(ErrorType::InvalidArgument)
                    .message(format!(
                        "Invalid tool `parameters` schema (walle {level}): {msg}"
                    ))
                    .build());
            }
        }
        Ok(())
    }
}

#[cfg(feature = "walle-validation")]
#[allow(unsafe_code)] // FFI into libwalle.so; safety is documented at the call site.
mod imp {
    use std::{
        ffi::{CStr, CString},
        os::raw::c_char,
    };

    use super::WalleLevel;

    // Exported by libwalle.so (walle/python/c-shared/main.go).
    unsafe extern "C" {
        fn ValidateSchema(schema: *const c_char, config: *const c_char) -> *mut c_char;
        fn FreeErrString(s: *mut c_char);
    }

    /// Validate a JSON-schema string with walle at `level`. `Ok(())` if walle
    /// accepts it; `Err(message)` with walle's diagnostic if it rejects it.
    pub(crate) fn validate_schema(schema_json: &str, level: WalleLevel) -> Result<(), String> {
        let schema = CString::new(schema_json)
            .map_err(|_| "schema contains an interior NUL byte".to_string())?;
        let config = CString::new(level.config_json()).expect("static config has no NUL byte");

        // SAFETY: ValidateSchema copies both inputs (they need only be valid for
        // the duration of the call) and returns a heap C string that we own and
        // free via FreeErrString.
        unsafe {
            let ptr = ValidateSchema(schema.as_ptr(), config.as_ptr());
            if ptr.is_null() {
                return Ok(());
            }
            let msg = CStr::from_ptr(ptr).to_string_lossy().into_owned();
            FreeErrString(ptr);
            if msg.is_empty() { Ok(()) } else { Err(msg) }
        }
    }
}

#[cfg(not(feature = "walle-validation"))]
mod imp {
    use super::WalleLevel;

    /// No-op stub: the frontend was built without walle, so schema validation is
    /// skipped (accept everything). Keeps call sites feature-agnostic.
    pub(crate) fn validate_schema(_schema_json: &str, _level: WalleLevel) -> Result<(), String> {
        Ok(())
    }
}

pub(crate) use imp::validate_schema;

#[cfg(test)]
mod level_tests {
    use super::WalleLevel;

    #[test]
    fn parse_levels() {
        assert_eq!(WalleLevel::parse("strict"), Some(WalleLevel::Strict));
        assert_eq!(WalleLevel::parse("LITE"), Some(WalleLevel::Lite));
        assert_eq!(WalleLevel::parse("ultra"), None);
        assert_eq!(WalleLevel::default(), WalleLevel::Strict);
        assert_eq!(WalleLevel::Lite.to_string(), "lite");
    }
}
