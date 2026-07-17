//! Kimi-K3 preprocessor token-ID path.
//!
//! K3 prompts must be rendered by the native XTML renderer (token IDs with
//! per-segment special-token control) and NEVER by the String-template +
//! re-tokenize path. These tests run against a synthetic K3-shaped model dir
//! (tiny byte-level tiktoken vocab), so they are CI-safe (no downloads).

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;
use dynamo_llm::model_card::ModelDeploymentCard;
use dynamo_llm::preprocessor::OpenAIPreprocessor;
use dynamo_llm::preprocessor::prompt::kimi_k3::encoder::KimiK3Renderer;
use dynamo_llm::protocols::openai::chat_completions::NvCreateChatCompletionRequest;
use std::collections::HashMap;

const OPEN_ID: u32 = 300;
const SEP_ID: u32 = 302;

/// tiktoken model with all 256 single-byte ranks (BPE needs every byte).
fn full_byte_bpe() -> String {
    let mut out = String::new();
    for b in 0u32..256 {
        out.push_str(&format!("{} {}\n", STANDARD.encode([b as u8]), b));
    }
    out
}

/// Synthetic K3 model dir: config.json (model_type kimi_k3) +
/// tokenizer_config.json (wire-marker specials) + tiktoken.model.
fn synthetic_k3_dir() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("tiktoken.model"), full_byte_bpe()).unwrap();
    std::fs::write(
        dir.path().join("config.json"),
        serde_json::json!({
            "model_type": "kimi_k3",
            "architectures": ["KimiK3ForConditionalGeneration"],
            "eos_token_id": 303,
            "max_position_embeddings": 8192,
            "vocab_size": 512,
            // Mirrors the real K3 config (163605 there): the renderer emits
            // this id for image content parts; MM routing resolves it from
            // config.json (see lightseek_mm chat-placeholder fallback).
            "media_placeholder_token_id": 306
        })
        .to_string(),
    )
    .unwrap();
    std::fs::write(
        dir.path().join("tokenizer_config.json"),
        serde_json::json!({
            "tokenizer_class": "TikTokenTokenizer",
            "added_tokens_decoder": {
                "300": {"content": "<|open|>", "special": false},
                "301": {"content": "<|close|>", "special": false},
                "302": {"content": "<|sep|>", "special": false},
                "303": {"content": "<|end_of_msg|>", "special": true},
                "304": {"content": "<|media_begin|>", "special": true},
                "305": {"content": "<|media_content|>", "special": true},
                "306": {"content": "<|media_pad|>", "special": true},
                "307": {"content": "<|media_end|>", "special": true}
            }
        })
        .to_string(),
    )
    .unwrap();
    dir
}

fn chat_request(
    chat_template_args: Option<HashMap<String, serde_json::Value>>,
    model: String,
) -> NvCreateChatCompletionRequest {
    let messages: Vec<dynamo_protocols::types::ChatCompletionRequestMessage> =
        serde_json::from_str(r#"[{"role":"user","content":"hello"}]"#).unwrap();
    let mut inner = dynamo_protocols::types::CreateChatCompletionRequestArgs::default();
    inner.model(model);
    inner.messages(messages);
    let inner = inner.build().unwrap();
    NvCreateChatCompletionRequest {
        inner,
        common: Default::default(),
        nvext: None,
        chat_template_args,
        thinking: None,
        media_io_kwargs: None,
        return_tokens_as_token_ids: None,
        unsupported_fields: Default::default(),
    }
}

#[tokio::test]
async fn k3_preprocess_uses_renderer_token_ids() {
    let dir = synthetic_k3_dir();
    let mdc = ModelDeploymentCard::load_from_disk(dir.path(), None).expect("load K3 MDC");
    let preprocessor = OpenAIPreprocessor::new(mdc.clone()).expect("build preprocessor");

    let request = chat_request(None, mdc.slug().to_string());
    let (preprocessed, annotations, _injected) = preprocessor
        .preprocess_request(&request, None)
        .await
        .expect("preprocess K3 request");

    // Byte-for-byte the renderer's output — never a String re-tokenize.
    let renderer = KimiK3Renderer::from_model_dir(dir.path()).expect("build renderer");
    let expected = renderer.render_to_ids(&request).expect("render ids");
    assert!(!expected.is_empty());
    assert_eq!(preprocessed.token_ids, expected);

    // Structural markers arrive as atomic special ids (message open + sep).
    assert!(preprocessed.token_ids.contains(&OPEN_ID));
    assert!(preprocessed.token_ids.contains(&SEP_ID));

    // The template path never ran: no formatted_prompt/token_ids annotations.
    assert!(
        annotations.is_empty(),
        "unexpected annotations: {annotations:?}"
    );
}

#[tokio::test]
async fn k3_preprocess_marks_prompt_injected_reasoning() {
    let dir = synthetic_k3_dir();
    let mut mdc = ModelDeploymentCard::load_from_disk(dir.path(), None).expect("load K3 MDC");
    mdc.runtime_config.reasoning_parser = Some("kimi_k3".to_string());
    let preprocessor = OpenAIPreprocessor::new(mdc.clone()).expect("build preprocessor");

    // Thinking defaults ON: the rendered prompt ends inside the think channel.
    let request = chat_request(None, mdc.slug().to_string());
    let (_, _, injected) = preprocessor
        .preprocess_request(&request, None)
        .await
        .expect("preprocess");
    assert!(
        injected,
        "K3 with thinking on must mark prompt-injected reasoning"
    );

    // Thinking explicitly disabled: instruct mode, no think prefill.
    let mut args = HashMap::new();
    args.insert("thinking".to_string(), serde_json::Value::Bool(false));
    let request = chat_request(Some(args), mdc.slug().to_string());
    let (_, _, injected) = preprocessor
        .preprocess_request(&request, None)
        .await
        .expect("preprocess");
    assert!(
        !injected,
        "K3 with thinking off must not mark prompt-injected reasoning"
    );
}
/// Image-bearing requests render exactly one `<|media_pad|>` (id 306 in the
/// synthetic vocab; 163605 on the real model) per image content part —
/// downstream MM processing expands each placeholder into the full media
/// sequence. Text parts around the images encode as ordinary tokens.
#[tokio::test]
async fn k3_preprocess_image_parts_render_one_media_pad_each() {
    const MEDIA_PAD_ID: u32 = 306;

    let dir = synthetic_k3_dir();
    let mdc = ModelDeploymentCard::load_from_disk(dir.path(), None).expect("load K3 MDC");
    let preprocessor = OpenAIPreprocessor::new(mdc.clone()).expect("build preprocessor");

    let messages: Vec<dynamo_protocols::types::ChatCompletionRequestMessage> =
        serde_json::from_value(serde_json::json!([{
            "role": "user",
            "content": [
                {"type": "text", "text": "compare these"},
                {"type": "image_url", "image_url": {"url": "http://example/a.png"}},
                {"type": "text", "text": "with"},
                {"type": "image_url", "image_url": {"url": "http://example/b.png"}}
            ]
        }]))
        .expect("content-part message deserializes");
    let mut inner = dynamo_protocols::types::CreateChatCompletionRequestArgs::default();
    inner.model(mdc.slug().to_string());
    inner.messages(messages);
    let request = NvCreateChatCompletionRequest {
        inner: inner.build().unwrap(),
        common: Default::default(),
        nvext: None,
        chat_template_args: None,
        thinking: None,
        media_io_kwargs: None,
        return_tokens_as_token_ids: None,
        unsupported_fields: Default::default(),
    };

    let (preprocessed, _, _) = preprocessor
        .preprocess_request(&request, None)
        .await
        .expect("preprocess image-bearing K3 request");

    let pad_count = preprocessed
        .token_ids
        .iter()
        .filter(|&&id| id == MEDIA_PAD_ID)
        .count();
    assert_eq!(
        pad_count, 2,
        "exactly one <|media_pad|> per image part (got {pad_count}); ids: {:?}",
        preprocessed.token_ids
    );
}

/// Decode synthetic-vocab token ids back to text: ids < 256 are raw bytes,
/// 300..=307 are the K3 wire markers registered in `synthetic_k3_dir`.
fn decode_synthetic(ids: &[u32]) -> String {
    let mut bytes: Vec<u8> = Vec::new();
    for &id in ids {
        match id {
            0..=255 => bytes.push(id as u8),
            300 => bytes.extend_from_slice(b"<|open|>"),
            301 => bytes.extend_from_slice(b"<|close|>"),
            302 => bytes.extend_from_slice(b"<|sep|>"),
            303 => bytes.extend_from_slice(b"<|end_of_msg|>"),
            304 => bytes.extend_from_slice(b"<|media_begin|>"),
            305 => bytes.extend_from_slice(b"<|media_content|>"),
            306 => bytes.extend_from_slice(b"<|media_pad|>"),
            307 => bytes.extend_from_slice(b"<|media_end|>"),
            other => panic!("unexpected token id {other}"),
        }
    }
    String::from_utf8(bytes).expect("synthetic decode is utf-8")
}

/// End-to-end request-gate flow: `thinking {type, effort, keep}` normalized at
/// ingress (mirroring the HTTP handler) must reach the renderer as
/// `thinking_effort` / `preserve_thinking` — the thinking-effort control
/// message renders and prior-turn think blocks are preserved.
#[tokio::test]
async fn k3_thinking_effort_and_keep_render_end_to_end() {
    let dir = synthetic_k3_dir();
    let mut mdc = ModelDeploymentCard::load_from_disk(dir.path(), None).expect("load K3 MDC");
    mdc.runtime_config.reasoning_parser = Some("kimi_k3".to_string());
    let preprocessor = OpenAIPreprocessor::new(mdc.clone()).expect("build preprocessor");

    let mut request: NvCreateChatCompletionRequest = serde_json::from_value(serde_json::json!({
        "model": mdc.slug().to_string(),
        "messages": [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "reasoning_content": "old think", "content": "a1"},
            {"role": "user", "content": "q2"}
        ],
        "thinking": {"type": "enabled", "effort": "high", "keep": "interleaved"}
    }))
    .expect("request deserializes");

    // The HTTP ingress normalizes `thinking` into chat_template_args before
    // the preprocessor runs; mirror that here.
    request
        .normalize_reasoning_template_args()
        .expect("normalize thinking");

    let (preprocessed, _, injected) = preprocessor
        .preprocess_request(&request, None)
        .await
        .expect("preprocess K3 request");
    let text = decode_synthetic(&preprocessed.token_ids);

    // REQ4: the thinking-effort control message renders with the request value.
    assert!(
        text.contains("thinking_effort=high"),
        "missing thinking-effort control message: {text}"
    );
    // keep=interleaved => preserve_thinking=true => prior-turn think survives.
    assert!(
        text.contains("old think"),
        "history think dropped despite keep=interleaved: {text}"
    );
    // Thinking enabled: generation prefix ends inside the think channel.
    assert!(
        injected,
        "thinking enabled must mark prompt-injected reasoning"
    );
    assert!(text.ends_with("<|open|>think<|sep|>"), "prefix: {text}");
}

/// keep=all (the default history rule) drops prior-turn think blocks.
#[tokio::test]
async fn k3_thinking_keep_all_drops_history_think() {
    let dir = synthetic_k3_dir();
    let mdc = ModelDeploymentCard::load_from_disk(dir.path(), None).expect("load K3 MDC");
    let preprocessor = OpenAIPreprocessor::new(mdc.clone()).expect("build preprocessor");

    let mut request: NvCreateChatCompletionRequest = serde_json::from_value(serde_json::json!({
        "model": mdc.slug().to_string(),
        "messages": [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "reasoning_content": "old think", "content": "a1"},
            {"role": "user", "content": "q2"}
        ],
        "thinking": {"type": "enabled", "keep": "all"}
    }))
    .expect("request deserializes");
    request
        .normalize_reasoning_template_args()
        .expect("normalize thinking");

    let (preprocessed, _, _) = preprocessor
        .preprocess_request(&request, None)
        .await
        .expect("preprocess K3 request");
    let text = decode_synthetic(&preprocessed.token_ids);
    assert!(
        !text.contains("old think"),
        "keep=all must drop pre-final-assistant think blocks: {text}"
    );
    // a1 (the response content) still renders.
    assert!(text.contains("a1"), "history response missing: {text}");
}
