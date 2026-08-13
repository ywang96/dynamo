// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use dynamo_llm::model_card::ModelDeploymentCard;
use dynamo_llm::preprocessor::OpenAIPreprocessor;
use dynamo_llm::preprocessor::prompt::prompt_formatter_from_mdc;
use dynamo_llm::protocols::openai::chat_completions::NvCreateChatCompletionRequest;
use dynamo_llm::tokenizers::traits::{DecodeResult, Decoder, Encoder};
use dynamo_llm::tokenizers::{Encoding, Tokenizer as DynamoTokenizer};
use dynamo_protocols::types::FinishReason;
use dynamo_renderer::PromptFormatter;
use dynamo_runtime::protocols::annotated::Annotated;
use futures::{StreamExt, stream};

use super::{get_text, mock_content_chunk, mock_final_chunk, mock_multi_choice_content_chunk};

struct KimiK3TestTokenizer;

impl Encoder for KimiK3TestTokenizer {
    fn encode(&self, input: &str) -> anyhow::Result<Encoding> {
        let ids = match input {
            "<|open|>" => vec![1],
            "<|sep|>" => vec![2],
            "think" => vec![3],
            "response" => vec![4],
            _ => input.bytes().map(|byte| u32::from(byte) + 100).collect(),
        };
        Ok(Encoding::Sp(ids))
    }

    fn encode_batch(&self, inputs: &[&str]) -> anyhow::Result<Vec<Encoding>> {
        inputs.iter().map(|input| self.encode(input)).collect()
    }
}

impl Decoder for KimiK3TestTokenizer {
    fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> anyhow::Result<DecodeResult> {
        let mut output = String::new();
        for token_id in token_ids {
            match *token_id {
                1 if !skip_special_tokens => output.push_str("<|open|>"),
                2 if !skip_special_tokens => output.push_str("<|sep|>"),
                1 | 2 => {}
                3 => output.push_str("think"),
                4 => output.push_str("response"),
                id if id >= 100 => output.push((id - 100) as u8 as char),
                _ => {}
            }
        }
        Ok(DecodeResult::Complete(output))
    }
}

impl dynamo_llm::tokenizers::traits::Tokenizer for KimiK3TestTokenizer {}

fn build_preprocessor() -> Arc<OpenAIPreprocessor> {
    let model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/sample-models/mock-llama-3.1-8b-instruct");
    let mut mdc = ModelDeploymentCard::load_from_disk(model_path, None).unwrap();
    mdc.runtime_config.reasoning_parser = Some("kimi_k3".to_string());
    mdc.runtime_config.tool_call_parser = Some("kimi_k3".to_string());
    let PromptFormatter::OAI(formatter) = prompt_formatter_from_mdc(&mdc).unwrap();
    let tokenizer = DynamoTokenizer::from(Arc::new(KimiK3TestTokenizer));
    OpenAIPreprocessor::new_with_parts(mdc, formatter, tokenizer).unwrap()
}

#[tokio::test]
async fn preserves_unified_event_order() {
    let preprocessor = build_preprocessor();
    let request: NvCreateChatCompletionRequest = serde_json::from_value(serde_json::json!({
        "messages": [{"role": "user", "content": "Check Paris weather and current time."}],
        "model": "moonshotai/Kimi-K3-Instruct",
        "stream": true,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_time",
                    "parameters": {
                        "type": "object",
                        "properties": {"zone": {"type": "string"}},
                        "required": ["zone"]
                    }
                }
            }
        ],
        "tool_choice": "required"
    }))
    .unwrap();

    let tool_output = concat!(
        "<|close|>response<|sep|><|open|>tools<|sep|>",
        "<|open|>call tool=\"get_weather\" index=\"1\"<|sep|>",
        "<|open|>argument key=\"city\" type=\"string\"<|sep|>Paris",
        "<|close|>argument<|sep|><|close|>call<|sep|>",
        "<|open|>call tool=\"get_time\" index=\"2\"<|sep|>",
        "<|open|>argument key=\"zone\" type=\"string\"<|sep|>UTC",
        "<|close|>argument<|sep|><|close|>call<|sep|>",
        "<|close|>tools<|sep|><|close|>message<|sep|>"
    );
    let input_chunks = vec![
        mock_content_chunk("Inspect "),
        mock_content_chunk("forecast<|clo"),
        mock_content_chunk("se|>think<|sep|><|open|>response<|sep|>I will check."),
        mock_content_chunk(tool_output),
        mock_final_chunk(),
    ];

    let output = preprocessor
        .postprocessor_parsing_stream_with_prompt_tokens(
            stream::iter(input_chunks.into_iter().map(Annotated::from_data)),
            &request,
            true,
            false,
            &[1, 3, 2],
        )
        .unwrap()
        .collect::<Vec<_>>()
        .await;

    let mut events = Vec::new();
    let mut finish_reasons = Vec::new();
    for response in output {
        let Some(data) = response.data else {
            continue;
        };
        for choice in data.inner.choices {
            // Skip the wire-shape stage's canonical first frame
            // ({"role":"assistant","content":""}) — it carries no increment.
            if choice.delta.role.is_some() {
                continue;
            }
            if let Some(reasoning) = choice.delta.reasoning_content {
                events.push(format!("reasoning:{reasoning}"));
            }
            if let Some(content) = choice.delta.content {
                events.push(format!("content:{}", get_text(&content)));
            }
            if let Some(tool_calls) = choice.delta.tool_calls {
                for tool_call in tool_calls {
                    let function = tool_call.function.unwrap();
                    events.push(format!(
                        "tool:{}:{}:{}:{}",
                        tool_call.index,
                        tool_call.id.as_deref().unwrap_or("-"),
                        function.name.as_deref().unwrap_or("-"),
                        function.arguments.unwrap_or_default()
                    ));
                }
            }
            finish_reasons.extend(choice.finish_reason);
        }
    }

    // The wire-shape stage splits each complete tool call into a header chunk
    // (id + name + empty arguments) followed by an arguments-only continuation
    // (P0.13/P0.14).
    assert_eq!(
        events,
        [
            "reasoning:Inspect ",
            "reasoning:forecast",
            // P1.7 end-of-thinking boundary: standalone reasoning_content "".
            "reasoning:",
            "content:I will check.",
            "tool:0:get_weather_0:get_weather:",
            "tool:0:-:-:{\"city\":\"Paris\"}",
            "tool:1:get_time_1:get_time:",
            "tool:1:-:-:{\"zone\":\"UTC\"}",
        ]
    );
    assert_eq!(finish_reasons, [FinishReason::ToolCalls]);
    assert!(events.iter().all(|event| !event.contains("<|")));
}

#[tokio::test]
async fn keeps_choice_state_isolated() {
    let preprocessor = build_preprocessor();
    let request: NvCreateChatCompletionRequest = serde_json::from_value(serde_json::json!({
        "messages": [{"role": "user", "content": "Answer twice."}],
        "model": "moonshotai/Kimi-K3-Instruct",
        "stream": true
    }))
    .unwrap();
    let input = vec![
        mock_multi_choice_content_chunk(&[(0, "alpha<|cl"), (1, "beta")]),
        mock_multi_choice_content_chunk(&[
            (0, "ose|>response<|sep|><|close|>message<|sep|>"),
            (1, " gamma"),
        ]),
    ];

    let output = preprocessor
        .postprocessor_parsing_stream_with_prompt_tokens(
            stream::iter(input.into_iter().map(Annotated::from_data)),
            &request,
            false,
            false,
            &[1, 4, 2],
        )
        .unwrap()
        .collect::<Vec<_>>()
        .await;

    let mut content = BTreeMap::<u32, String>::new();
    for response in output {
        if let Some(data) = response.data {
            for choice in data.inner.choices {
                if let Some(delta) = choice.delta.content {
                    content
                        .entry(choice.index)
                        .or_default()
                        .push_str(get_text(&delta));
                }
            }
        }
    }
    assert_eq!(content.get(&0).map(String::as_str), Some("alpha"));
    assert_eq!(content.get(&1).map(String::as_str), Some("beta gamma"));
}

#[tokio::test]
async fn does_not_emit_incomplete_tool_call() {
    let preprocessor = build_preprocessor();
    let request: NvCreateChatCompletionRequest = serde_json::from_value(serde_json::json!({
        "messages": [{"role": "user", "content": "Check Paris weather."}],
        "model": "moonshotai/Kimi-K3-Instruct",
        "stream": true,
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {"type": "object"}
            }
        }]
    }))
    .unwrap();
    let incomplete = concat!(
        "<|close|>response<|sep|><|open|>tools<|sep|>",
        "<|open|>call tool=\"get_weather\" index=\"1\"<|sep|>",
        "<|open|>argument key=\"city\" type=\"string\"<|sep|>Par"
    );

    let output = preprocessor
        .postprocessor_parsing_stream_with_prompt_tokens(
            stream::iter(
                [mock_content_chunk(incomplete), mock_final_chunk()]
                    .into_iter()
                    .map(Annotated::from_data),
            ),
            &request,
            false,
            false,
            &[1, 4, 2],
        )
        .unwrap()
        .collect::<Vec<_>>()
        .await;

    let mut tool_calls = 0;
    let mut recovered = String::new();
    let mut finish_reasons = Vec::new();
    for response in output {
        if let Some(data) = response.data {
            for choice in data.inner.choices {
                tool_calls += choice.delta.tool_calls.map_or(0, |calls| calls.len());
                if let Some(content) = choice.delta.content {
                    recovered.push_str(get_text(&content));
                }
                finish_reasons.extend(choice.finish_reason);
            }
        }
    }
    assert_eq!(tool_calls, 0);
    assert!(recovered.contains("Par"));
    assert_eq!(finish_reasons, [FinishReason::Stop]);
}
