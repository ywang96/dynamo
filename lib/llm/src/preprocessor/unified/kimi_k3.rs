// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Kimi K3 unified parser selection and construction.

use std::sync::Arc;

use dynamo_protocols::types::ChatCompletionTool;
use dynamo_runtime::protocols::annotated::Annotated;
use futures::Stream;
use vllm_parser::unified::{KimiK3UnifiedParser, UnifiedParser as _};

use super::{UnifiedOutputStream, UnifiedParserSpec, unified_output_stream};
use crate::protocols::openai::chat_completions::NvCreateChatCompletionStreamResponse;
use crate::tokenizers::traits::Tokenizer as DynamoTokenizer;

/// Return whether the configured split parser names select Kimi K3's unified parser.
pub(in crate::preprocessor) fn is_selected(
    reasoning_parser: Option<&str>,
    tool_call_parser: Option<&str>,
) -> bool {
    fn is_kimi_k3(name: Option<&str>) -> bool {
        matches!(name, Some("kimi_k3" | "kimi-k3"))
    }

    is_kimi_k3(reasoning_parser) && is_kimi_k3(tool_call_parser)
}

/// Parse one OpenAI delta stream through vLLM's Kimi K3 unified parser.
pub(in crate::preprocessor) fn output_stream<S>(
    input: S,
    tools: &[ChatCompletionTool],
    allow_tool_calls: bool,
    tokenizer: Arc<dyn DynamoTokenizer>,
    prompt_token_ids: &[u32],
) -> anyhow::Result<UnifiedOutputStream>
where
    S: Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send + 'static,
{
    unified_output_stream(
        input,
        UnifiedParserSpec {
            name: "kimi_k3",
            create: KimiK3UnifiedParser::create,
        },
        tools,
        allow_tool_calls,
        tokenizer,
        prompt_token_ids,
    )
}
