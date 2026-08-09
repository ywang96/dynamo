// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Kimi K3 XTML structural-tag constraints for xgrammar.

use anyhow::{Context, bail};
use dynamo_parsers::tool_calling::{ToolChoice, ToolDefinition};
use serde_json::Value;
use vllm_parser::unified::KimiK3StructuralTagBuilder;
use xgrammar_structural_tag::builders::StructuralTagOptions;
use xgrammar_structural_tag::{
    FunctionDefinition, FunctionToolParam, ToolChoice as StructuralTagToolChoice, ToolParam,
    build_structural_tag,
};

/// Build the K3 XTML constraint when the request's tool policy needs one.
///
/// `thinking` must match what the renderer put at the end of the prompt: it
/// prefills `<|open|>think<|sep|>` when thinking is on and
/// `<|open|>response<|sep|>` when it is off, and the grammar has to agree about
/// which of the two markers the model still owes. Pass the preprocessor's
/// `prompt_injected_reasoning`, which is derived for K3 from the same effective
/// thinking flag.
pub fn build_kimi_k3_structural_tag(
    tool_choice: &ToolChoice,
    tools: &[ToolDefinition],
    thinking: bool,
) -> anyhow::Result<Option<Value>> {
    if matches!(tool_choice, ToolChoice::Named(_)) {
        bail!("named tool choice is not supported for Kimi K3");
    }
    if tools.is_empty() {
        return Ok(None);
    }

    let tool_choice = match tool_choice {
        ToolChoice::None => StructuralTagToolChoice::none(),
        ToolChoice::Auto => StructuralTagToolChoice::auto(),
        ToolChoice::Required => StructuralTagToolChoice::required(),
        ToolChoice::Named(_) => unreachable!("handled above"),
    };
    let tools = tools
        .iter()
        .map(|tool| {
            ToolParam::Function(FunctionToolParam::new(FunctionDefinition {
                name: tool.name.clone(),
                description: None,
                parameters: tool.parameters.clone(),
                // KVV omits `strict`; K3 still expects its parameter schema to
                // constrain arguments. Only an explicit false opts out.
                strict: Some(tool.strict.unwrap_or(true)),
            }))
        })
        .collect::<Vec<_>>();
    let structural_tag = build_structural_tag(
        // `enable_in_reasoning` is left at its default: the engine hands the
        // grammar to the model only once `<|close|>think<|sep|>` has landed, so
        // the grammar must not try to model the think channel itself.
        KimiK3StructuralTagBuilder::new(),
        &tools,
        tool_choice,
        StructuralTagOptions::default().with_reasoning(thinking),
    )?;

    serde_json::to_value(structural_tag)
        .context("failed to serialize Kimi K3 structural tag")
        .map(Some)
}
