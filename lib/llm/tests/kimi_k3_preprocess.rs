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
            "vocab_size": 512
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
