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

/// Build the K3 XTML constraint for the request's tool policy.
///
/// Requests without tools are constrained too. The tag is a whole-turn channel
/// grammar, and what it buys on a no-tools request is the part that has nothing
/// to do with tool calling: the model cannot end the turn without opening the
/// response channel. Leaving those requests unconstrained is why a plain chat
/// completion can answer inside `reasoning_content` and then stop with empty
/// `content` -- 49 of 700 rows on the 2026-08-08 BEAM run.
pub fn build_kimi_k3_structural_tag(
    tool_choice: &ToolChoice,
    tools: &[ToolDefinition],
) -> anyhow::Result<Option<Value>> {
    if matches!(tool_choice, ToolChoice::Named(_)) {
        bail!("named tool choice is not supported for Kimi K3");
    }
    // `required` with no tools has nothing to require, and the builder rejects
    // it (`Error::RequiredWithoutTools`). Keep returning an unconstrained
    // request rather than turning a contradictory one into a 400 it did not
    // get before.
    if tools.is_empty() && matches!(tool_choice, ToolChoice::Required) {
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
        KimiK3StructuralTagBuilder,
        &tools,
        tool_choice,
        StructuralTagOptions::default().with_reasoning(false),
    )?;

    serde_json::to_value(structural_tag)
        .context("failed to serialize Kimi K3 structural tag")
        .map(Some)
}
