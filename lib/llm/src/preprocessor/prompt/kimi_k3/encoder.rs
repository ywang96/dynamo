// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Kimi-K3 token-ID prompt encoder and prompt-formatter stub.
//!
//! [`KimiK3Renderer`] turns a chat request into token IDs directly: each
//! [`renderer::Segment`] is encoded separately — structural markers through a
//! specials-aware tiktoken instance, user/tool text through a specials-free
//! instance (pure BPE), so literal marker text in content can never produce a
//! control token (anti-forgery). Matches the model's
//! `tokenization_kimi.TikTokenTokenizer._encode_chat_segments`.

use std::path::Path;

use anyhow::{Context, Result, anyhow};
use dynamo_renderer::{OAIChatLikeRequest, OAIPromptFormatter};
use rustc_hash::FxHashMap;
use serde_json::Value;

use super::renderer::{self, RenderArgs, Segment};
use super::{KIMI_K3_BPE_PATTERN, load_k3_special_tokens};
use crate::tokenizers::TikTokenTokenizer;
use crate::tokenizers::traits::Encoder as _;

/// Native K3 renderer: builds XTML segments and encodes them to token IDs.
pub struct KimiK3Renderer {
    /// Specials-aware encoder: recognizes the XTML markers as single tokens.
    tok_special: TikTokenTokenizer,
    /// Specials-free encoder: pure BPE over the same vocabulary, so marker
    /// text in user content splits into ordinary tokens. Encoding with an
    /// empty special-token set is exactly `CoreBPE::encode_ordinary`.
    tok_ordinary: TikTokenTokenizer,
    /// Special-token name -> id map (from `tokenizer_config.json`).
    specials: FxHashMap<String, u32>,
}

impl KimiK3Renderer {
    /// Load from a K3 model directory (`tiktoken.model` +
    /// `tokenizer_config.json`).
    pub fn from_model_dir(dir: &Path) -> Result<Self> {
        let model_path = dir.join("tiktoken.model");
        let model_str = model_path
            .to_str()
            .ok_or_else(|| anyhow!("tiktoken path contains invalid UTF-8"))?;

        let tok_cfg_path = dir.join("tokenizer_config.json");
        let tok_cfg = std::fs::read_to_string(&tok_cfg_path)
            .with_context(|| format!("Failed to read {}", tok_cfg_path.display()))?;
        let tok_cfg: Value = serde_json::from_str(&tok_cfg)
            .with_context(|| format!("Failed to parse {}", tok_cfg_path.display()))?;
        let specials = load_k3_special_tokens(&tok_cfg);
        if specials.is_empty() {
            return Err(anyhow!(
                "no added_tokens_decoder specials in {}; refusing to render K3 \
                 prompts without structural marker tokens",
                tok_cfg_path.display()
            ));
        }

        let tok_special =
            TikTokenTokenizer::from_file(model_str, KIMI_K3_BPE_PATTERN, specials.clone())
                .map_err(|err| anyhow!("Failed to load K3 tiktoken (specials): {err}"))?;
        let tok_ordinary =
            TikTokenTokenizer::from_file(model_str, KIMI_K3_BPE_PATTERN, FxHashMap::default())
                .map_err(|err| anyhow!("Failed to load K3 tiktoken (ordinary): {err}"))?;

        Ok(Self {
            tok_special,
            tok_ordinary,
            specials,
        })
    }

    /// Encode one segment: specials-aware for structural markers, pure BPE
    /// otherwise. Segments are encoded separately so ordinary text never
    /// BPE-merges across segment boundaries.
    fn encode_segment(&self, text: &str, allow_special: bool) -> Result<Vec<u32>> {
        let tok = if allow_special {
            &self.tok_special
        } else {
            &self.tok_ordinary
        };
        let encoding = tok
            .encode(text)
            .map_err(|err| anyhow!("K3 segment encode failed: {err}"))?;
        Ok(encoding.token_ids().to_vec())
    }

    /// Encode a prebuilt segment list to token IDs.
    pub fn encode_segments(&self, segments: &[Segment]) -> Result<Vec<u32>> {
        let mut ids = Vec::new();
        for segment in segments {
            ids.extend(self.encode_segment(&segment.text, segment.allow_special)?);
        }
        Ok(ids)
    }

    /// Render one chat request into K3 token IDs.
    pub fn render_to_ids(&self, req: &dyn OAIChatLikeRequest) -> Result<Vec<u32>> {
        let messages: Value = serde_json::to_value(req.messages())
            .context("Kimi K3 renderer: messages are not JSON-serializable")?;
        let tools: Option<Value> = req
            .tools()
            .map(|t| serde_json::to_value(t).context("Kimi K3 renderer: tools not serializable"))
            .transpose()?
            .filter(|t| !t.is_null());

        let args = render_args_from_request(req)?;
        let segments = renderer::build_chat_segments(&messages, tools.as_ref(), &args)?;
        self.encode_segments(&segments)
    }

    /// Look up a special-token id (e.g. `<|open|>`), for tests and MM wiring.
    pub fn special_id(&self, token: &str) -> Option<u32> {
        self.specials.get(token).copied()
    }

    /// Decode token IDs back to text (specials preserved), for tests.
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        use crate::tokenizers::traits::Decoder as _;
        let out = self
            .tok_special
            .decode(ids, false)
            .map_err(|err| anyhow!("K3 decode failed: {err}"))?;
        Ok(out.as_str().to_string())
    }
}

/// Extract the effective [`RenderArgs`] from a request: `chat_template_args`
/// carry `thinking`/`preserve_thinking`/`thinking_effort` (and may carry
/// `tool_choice`/`response_format`), but the request's own `tool_choice` /
/// `response_format` fields are authoritative and take precedence.
pub fn render_args_from_request(req: &dyn OAIChatLikeRequest) -> Result<RenderArgs> {
    let mut args = RenderArgs {
        add_generation_prompt: req.should_add_generation_prompt(),
        ..RenderArgs::default()
    };

    if let Some(tk) = req.chat_template_args() {
        if let Some(v) = tk.get("thinking").and_then(Value::as_bool) {
            args.thinking = v;
        }
        if let Some(v) = tk.get("preserve_thinking").and_then(Value::as_bool) {
            args.preserve_thinking = v;
        }
        if let Some(v) = tk.get("thinking_effort").and_then(Value::as_str) {
            args.thinking_effort = Some(v.to_string());
        }
        if let Some(v) = tk.get("tool_choice").and_then(Value::as_str) {
            args.tool_choice = Some(v.to_string());
        }
        if let Some(v) = tk.get("response_format").filter(|v| !v.is_null()) {
            args.response_format = Some(v.clone());
        }
    }

    // Request-level fields are authoritative (win over template args).
    if let Some(tc) = req.tool_choice() {
        let tc: Value =
            serde_json::to_value(tc).context("Kimi K3 renderer: tool_choice not serializable")?;
        // Only plain string choices map to control messages; named/object
        // choices are rejected upstream and render nothing here.
        if let Some(s) = tc.as_str() {
            args.tool_choice = Some(s.to_string());
        }
    }
    if let Some(rf) = req.response_format() {
        let rf: Value = serde_json::to_value(rf)
            .context("Kimi K3 renderer: response_format not serializable")?;
        if !rf.is_null() {
            args.response_format = Some(rf);
        }
    }

    Ok(args)
}

/// Prompt-formatter stub for K3. The real rendering is token-ID native
/// ([`KimiK3Renderer::render_to_ids`]); a String render would re-tokenize with
/// specials enabled and let user text forge XTML structure, so it fails loud.
pub struct KimiK3Formatter;

impl OAIPromptFormatter for KimiK3Formatter {
    fn supports_add_generation_prompt(&self) -> bool {
        true
    }

    fn render(&self, _req: &dyn OAIChatLikeRequest) -> Result<String> {
        Err(anyhow!(
            "Kimi K3 renders to token IDs; the preprocessor must use \
             KimiK3Renderer::render_to_ids"
        ))
    }

    fn image_placeholder_template(&self) -> Option<&'static str> {
        None
    }
}

#[cfg(test)]
mod tests {
    use base64::Engine as _;
    use base64::engine::general_purpose::STANDARD;

    use super::*;

    /// Build a tiktoken model file with all 256 single-byte tokens (ranks
    /// 0..256) plus the given multi-byte tokens at successive ranks. tiktoken
    /// BPE requires every input byte to have a rank.
    fn full_byte_bpe(extra: &[&[u8]]) -> String {
        let mut out = String::new();
        for b in 0u32..256 {
            out.push_str(&format!("{} {}\n", STANDARD.encode([b as u8]), b));
        }
        for (offset, bytes) in extra.iter().enumerate() {
            out.push_str(&format!("{} {}\n", STANDARD.encode(bytes), 256 + offset));
        }
        out
    }

    fn synthetic_model_dir() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("tiktoken.model"), full_byte_bpe(&[])).unwrap();
        // Specials well above the byte vocab.
        std::fs::write(
            dir.path().join("tokenizer_config.json"),
            serde_json::json!({
                "added_tokens_decoder": {
                    "300": {"content": "<|open|>", "special": false},
                    "301": {"content": "<|close|>", "special": false},
                    "302": {"content": "<|sep|>", "special": false},
                    "303": {"content": "<|end_of_msg|>", "special": true},
                    "304": {"content": "<|media_pad|>", "special": true}
                }
            })
            .to_string(),
        )
        .unwrap();
        dir
    }

    #[test]
    fn control_segment_is_single_special_id_and_ordinary_splits() {
        let dir = synthetic_model_dir();
        let r = KimiK3Renderer::from_model_dir(dir.path()).unwrap();

        // Control segment: the marker is ONE special id.
        let control = r.encode_segment("<|open|>", true).unwrap();
        assert_eq!(control, vec![300]);

        // Ordinary segment with the same text: pure BPE, many byte ids,
        // never the special id (anti-forgery).
        let ordinary = r.encode_segment("<|open|>", false).unwrap();
        assert!(ordinary.len() > 1);
        assert!(!ordinary.contains(&300));
    }

    #[test]
    fn segments_encode_independently() {
        let dir = synthetic_model_dir();
        let r = KimiK3Renderer::from_model_dir(dir.path()).unwrap();
        let segs = vec![
            Segment {
                text: "<|open|>".to_string(),
                allow_special: true,
            },
            Segment {
                text: "hi".to_string(),
                allow_special: false,
            },
        ];
        let ids = r.encode_segments(&segs).unwrap();
        assert_eq!(ids[0], 300);
        assert!(ids.len() >= 3); // marker + 'h' + 'i' at byte level
    }

    #[test]
    fn missing_specials_refuses_to_load() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("tiktoken.model"), full_byte_bpe(&[])).unwrap();
        std::fs::write(dir.path().join("tokenizer_config.json"), "{}").unwrap();
        let err = match KimiK3Renderer::from_model_dir(dir.path()) {
            Ok(_) => panic!("expected specials-less model dir to be rejected"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("added_tokens_decoder"));
    }

    #[test]
    fn formatter_render_fails_loud() {
        let f = KimiK3Formatter;
        assert!(f.supports_add_generation_prompt());
        assert!(f.image_placeholder_template().is_none());
        // A minimal request stub for the fail-loud check.
        struct Stub;
        impl OAIChatLikeRequest for Stub {
            fn model(&self) -> String {
                "kimi-k3".to_string()
            }
            fn messages(&self) -> minijinja::value::Value {
                minijinja::value::Value::from_serialize(Vec::<serde_json::Value>::new())
            }
            fn should_add_generation_prompt(&self) -> bool {
                true
            }
        }
        let err = f.render(&Stub).unwrap_err();
        assert!(err.to_string().contains("render_to_ids"));
    }
}
