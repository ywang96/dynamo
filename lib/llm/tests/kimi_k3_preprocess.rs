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
            // Real K3 config carries this (163605). The vLLM backend uses it as
            // the per-patch <|media_pad|> id inside its OWN placeholder expansion;
            // the frontend renderer does NOT emit it — it emits the unexpanded
            // <|kimi_image_placeholder|>.
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
async fn k3_preprocess_preserves_dynamic_tools_in_message_order() {
    let dir = synthetic_k3_dir();
    let mdc = ModelDeploymentCard::load_from_disk(dir.path(), None).expect("load K3 MDC");
    let preprocessor = OpenAIPreprocessor::new(mdc.clone()).expect("build preprocessor");

    let request: NvCreateChatCompletionRequest = serde_json::from_value(serde_json::json!({
        "model": mdc.slug(),
        "messages": [
            {"role": "user", "content": "before dynamic declaration"},
            {
                "role": "system",
                "content": "",
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"]
                        }
                    }
                }]
            },
            {"role": "user", "content": "after dynamic declaration"}
        ],
        "tool_choice": "required"
    }))
    .expect("deserialize dynamic-tool request");

    let (preprocessed, _, _) = preprocessor
        .preprocess_request(&request, None)
        .await
        .expect("preprocess dynamic-tool request");
    let renderer = KimiK3Renderer::from_model_dir(dir.path()).expect("build renderer");
    let decoded = renderer
        .decode(&preprocessed.token_ids)
        .expect("decode rendered prompt");

    let before = decoded.find("before dynamic declaration").unwrap();
    let declaration = decoded.find("## New Tools Available").unwrap();
    let after = decoded.find("after dynamic declaration").unwrap();
    assert!(before < declaration && declaration < after);
    assert!(decoded.contains("\"name\":\"get_weather\""));
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

#[test]
fn k3_renderer_detects_the_terminal_generation_prefill() {
    let dir = synthetic_k3_dir();
    let mdc = ModelDeploymentCard::load_from_disk(dir.path(), None).expect("load K3 MDC");
    let renderer = KimiK3Renderer::from_model_dir(dir.path()).expect("build renderer");

    let request = chat_request(None, mdc.slug().to_string());
    let ids = renderer.render_to_ids(&request).expect("render ids");
    assert_eq!(
        renderer
            .trailing_generation_prefill_token_count(&ids)
            .expect("detect think prefill"),
        7,
        "the synthetic byte-level tokenizer encodes open + 'think' + sep as 7 tokens"
    );

    let mut args = HashMap::new();
    args.insert("thinking".to_string(), serde_json::Value::Bool(false));
    let request = chat_request(Some(args), mdc.slug().to_string());
    let ids = renderer.render_to_ids(&request).expect("render ids");
    assert_eq!(
        renderer
            .trailing_generation_prefill_token_count(&ids)
            .expect("detect response prefill"),
        10,
        "the synthetic byte-level tokenizer encodes open + 'response' + sep as 10 tokens"
    );
}

/// Image-bearing requests render one `<|kimi_image_placeholder|>` per image
/// content part — the UNEXPANDED placeholder the vLLM Kimi-K3 backend searches
/// for and expands (`KimiK3ForConditionalGeneration._get_prompt_updates`). It
/// is not a special token, so it BPE-splits into ordinary ids; it must NOT be
/// pre-substituted to a `<|media_pad|>` special (the token the backend never
/// searches for), which caused
/// `Failed to apply prompt replacement for mm_items['image'][0]`.
#[tokio::test]
async fn k3_preprocess_image_parts_render_image_placeholder_each() {
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

    // The placeholder is not a special token, so it BPE-splits into ordinary
    // (byte) ids; decode and count the literal placeholder, once per image.
    let decoded = decode_synthetic(&preprocessed.token_ids);
    let placeholder_count = decoded.matches("<|kimi_image_placeholder|>").count();
    assert_eq!(
        placeholder_count, 2,
        "one <|kimi_image_placeholder|> per image part (got {placeholder_count}); decoded: {decoded}"
    );

    // The frontend must NOT emit a `<|media_pad|>` special — the backend does
    // its own expansion and searches only for `<|kimi_image_placeholder|>`.
    assert!(
        !preprocessed.token_ids.contains(&MEDIA_PAD_ID),
        "frontend must not emit <|media_pad|> ({MEDIA_PAD_ID}); ids: {:?}",
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
/// message renders and (keep=interleaved) prior-turn think blocks are dropped.
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
    // keep=interleaved => preserve_thinking=false => prior-turn think dropped.
    assert!(
        !text.contains("old think"),
        "history think survived despite keep=interleaved: {text}"
    );
    // Thinking enabled: generation prefix ends inside the think channel.
    assert!(
        injected,
        "thinking enabled must mark prompt-injected reasoning"
    );
    assert!(text.ends_with("<|open|>think<|sep|>"), "prefix: {text}");
}

/// keep=all (the default history rule) keeps prior-turn think blocks.
#[tokio::test]
async fn k3_thinking_keep_all_keeps_history_think() {
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
        text.contains("old think"),
        "keep=all must keep pre-final-assistant think blocks: {text}"
    );
    // a1 (the response content) still renders.
    assert!(text.contains("a1"), "history response missing: {text}");
}
