//! Kimi-K3 (XTML) model plumbing: model-type detection and tiktoken
//! special-token loading for the native K3 frontend.
//!
//! K3 ships a `tiktoken.model` vocabulary but its `model_type` (`kimi_k3`) is
//! not recognized by `dynamo_tokenizers::TikTokenTokenizer::from_file_auto`,
//! so the model card constructs the tokenizer through the explicit
//! `TikTokenTokenizer::from_file` path using the helpers here.

pub mod encoder;
pub mod renderer;

use rustc_hash::FxHashMap;

/// BPE regex for the Kimi tiktoken vocabulary.
///
/// Byte-identical to dynamo-tokenizers' private `KIMI_PATTERN` and to the
/// K3 model snapshot's `tokenization_kimi.py` `pat_str` (verified 2026-07-17
/// by executing the module and string-comparing); the crate const is private,
/// so the K3 path carries its own copy.
pub const KIMI_K3_BPE_PATTERN: &str = r#"[\p{Han}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#;

/// Mirrors dynamo-tokenizers' private `DEFAULT_NUM_RESERVED_SPECIAL_TOKENS`:
/// `from_file_auto` registers this many reserved special-token slots above the
/// base vocabulary so every id in the added-token range decodes atomically.
const NUM_RESERVED_SPECIAL_TOKENS: u32 = 256;

/// The K3 model family selector, from `config.json` `model_type`.
pub fn is_kimi_k3_model_type(model_type: &str) -> bool {
    model_type == "kimi_k3"
}

/// Read the special-token map (`content` → id) from a parsed
/// `tokenizer_config.json`'s `added_tokens_decoder`.
///
/// Mirrors the added-token parsing of `from_file_auto` (entries with an empty
/// or missing `content` are skipped with a warning).
pub fn load_k3_special_tokens(tokenizer_config: &serde_json::Value) -> FxHashMap<String, u32> {
    let mut special_tokens = FxHashMap::default();
    let Some(added_tokens) = tokenizer_config
        .get("added_tokens_decoder")
        .and_then(|v| v.as_object())
    else {
        return special_tokens;
    };

    for (id_str, token_def) in added_tokens {
        let Ok(id) = id_str.parse::<u32>() else {
            tracing::warn!(id = %id_str, "invalid token id in added_tokens_decoder; skipping");
            continue;
        };
        match token_def.get("content").and_then(|v| v.as_str()) {
            Some(content) if !content.is_empty() => {
                special_tokens.insert(content.to_string(), id);
            }
            _ => tracing::warn!(id, "missing 'content' for added token; skipping"),
        }
    }
    special_tokens
}

/// Fill unassigned ids in `[num_base_tokens, num_base_tokens + 256)` with
/// `<|reserved_token_{id}|>` placeholders, matching `from_file_auto`'s
/// gap-filling so unexpected ids in the added-token range still decode.
pub fn fill_reserved_special_tokens(
    special_tokens: &mut FxHashMap<String, u32>,
    num_base_tokens: u32,
) {
    let used: std::collections::HashSet<u32> = special_tokens.values().copied().collect();
    for i in 0..NUM_RESERVED_SPECIAL_TOKENS {
        let id = num_base_tokens + i;
        if !used.contains(&id) {
            special_tokens.insert(format!("<|reserved_token_{id}|>"), id);
        }
    }
}

/// Count the base-vocabulary size of a tiktoken model file as
/// `max rank + 1` (ranks may be sparse), matching `from_file_auto`.
///
/// Only the rank column is parsed; token payloads are not base64-decoded.
pub fn count_base_tokens(path: &str) -> anyhow::Result<u32> {
    let contents = std::fs::read_to_string(path)
        .map_err(|err| anyhow::anyhow!("Failed to read tiktoken file '{path}': {err}"))?;
    let mut max_rank: Option<u32> = None;
    for line in contents.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let rank_str = line
            .split_whitespace()
            .nth(1)
            .ok_or_else(|| anyhow::anyhow!("Invalid tiktoken line (no rank): {line}"))?;
        let rank: u32 = rank_str
            .parse()
            .map_err(|err| anyhow::anyhow!("Invalid rank in tiktoken file: {err}"))?;
        max_rank = Some(max_rank.map_or(rank, |m| m.max(rank)));
    }
    max_rank
        .map(|m| m + 1)
        .ok_or_else(|| anyhow::anyhow!("Empty tiktoken file: {path}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn k3_model_type_detected() {
        assert!(is_kimi_k3_model_type("kimi_k3"));
        assert!(!is_kimi_k3_model_type("kimi_k25"));
        assert!(!is_kimi_k3_model_type("kimi"));
    }

    #[test]
    fn k3_special_tokens_loaded_from_tokenizer_config() {
        let cfg = serde_json::json!({
            "added_tokens_decoder": {
                "163586": {"content": "<|end_of_msg|>", "special": true},
                "163587": {"content": "<|open|>", "special": false},
                "163605": {"content": "<|media_pad|>", "special": true}
            }
        });
        let m = load_k3_special_tokens(&cfg);
        assert_eq!(m.get("<|open|>"), Some(&163587));
        assert_eq!(m.get("<|end_of_msg|>"), Some(&163586));
        assert_eq!(m.get("<|media_pad|>"), Some(&163605));
    }

    #[test]
    fn reserved_tokens_fill_gaps_only() {
        let mut m = FxHashMap::default();
        m.insert("<|open|>".to_string(), 1001);
        fill_reserved_special_tokens(&mut m, 1000);
        // 1001 is taken by a real token — no reserved placeholder minted for it.
        assert_eq!(m.get("<|reserved_token_1000|>"), Some(&1000));
        assert_eq!(m.get("<|reserved_token_1002|>"), Some(&1002));
        assert!(!m.contains_key("<|reserved_token_1001|>"));
        // Full range covered: 255 reserved slots plus the real token.
        assert_eq!(m.len(), 256);
    }

    #[test]
    fn count_base_tokens_uses_max_rank() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("tiktoken.model");
        // Sparse ranks: max is 41, so the base vocab size is 42.
        std::fs::write(&p, "YQ== 0\nYg== 41\nYw== 7\n").unwrap();
        assert_eq!(count_base_tokens(p.to_str().unwrap()).unwrap(), 42);
    }
}
