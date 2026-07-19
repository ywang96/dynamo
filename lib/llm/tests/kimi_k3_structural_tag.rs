// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_llm::preprocessor::prompt::kimi_k3::structural_tag::build_kimi_k3_structural_tag;
use dynamo_parsers::tool_calling::{ToolChoice, ToolDefinition};
use serde_json::{Value, json};

fn tool(name: &str, parameters: Value, strict: Option<bool>) -> ToolDefinition {
    ToolDefinition {
        name: name.to_string(),
        parameters: Some(parameters),
        strict,
    }
}

fn build(tool_choice: ToolChoice, tools: &[ToolDefinition]) -> Option<Value> {
    build_kimi_k3_structural_tag(&tool_choice, tools).unwrap()
}

fn tools_part(tag: &Value) -> &Value {
    &tag["format"]["elements"][2]
}

#[test]
fn required_builds_a_mandatory_k3_tools_channel() {
    let tools = [tool(
        "lookup",
        json!({
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"]
        }),
        None,
    )];

    let tag = build(ToolChoice::Required, &tools).expect("required must build a tag");

    assert_eq!(tag["type"], "structural_tag");
    assert_eq!(tag["format"]["type"], "sequence");
    assert_eq!(tag["format"]["elements"].as_array().unwrap().len(), 4);
    assert_eq!(tools_part(&tag)["type"], "tag");
    assert_eq!(tools_part(&tag)["begin"], "<|open|>tools<|sep|>");
    assert_eq!(tools_part(&tag)["end"], "<|close|>tools<|sep|>");
    assert_eq!(tools_part(&tag)["content"]["type"], "tags_with_separator");
    assert_eq!(tools_part(&tag)["content"]["at_least_one"], true);
    assert_eq!(
        tools_part(&tag)["content"]["tags"][0]["begin"],
        "<|open|>call tool=\"lookup\" index=\""
    );
}

#[test]
fn auto_only_builds_for_strict_tools_and_keeps_tools_optional() {
    let loose = [tool("loose", json!({"type": "object"}), None)];
    assert!(build(ToolChoice::Auto, &loose).is_none());

    let strict = [tool("strict", json!({"type": "object"}), Some(true))];
    let tag = build(ToolChoice::Auto, &strict).expect("strict auto must build a tag");

    assert_eq!(tools_part(&tag)["type"], "optional");
    assert_eq!(tools_part(&tag)["content"]["type"], "tag");
    assert_eq!(tools_part(&tag)["content"]["begin"], "<|open|>tools<|sep|>");
}

#[test]
fn named_choice_selects_one_tool_and_rejects_an_unknown_name() {
    let tools = [
        tool("first", json!({"type": "object"}), None),
        tool("second", json!({"type": "object"}), None),
    ];

    let tag =
        build(ToolChoice::Named("second".into()), &tools).expect("named choice must build a tag");
    let serialized = serde_json::to_string(&tag).unwrap();

    assert!(serialized.contains("tool=\\\"second\\\""));
    assert!(!serialized.contains("tool=\\\"first\\\""));
    assert!(build_kimi_k3_structural_tag(&ToolChoice::Named("missing".into()), &tools).is_err());
}

#[test]
fn argument_formats_match_k3_raw_string_and_json_channels() {
    let tools = [tool(
        "typed",
        json!({
            "type": "object",
            "$defs": {"payload": {"type": "object", "properties": {"id": {"type": "integer"}}}},
            "properties": {
                "mode": {"type": "string", "enum": ["fast", "safe"]},
                "label": {"type": "string", "minLength": 2, "maxLength": 8},
                "payload": {"$ref": "#/$defs/payload", "type": "object"}
            }
        }),
        None,
    )];

    let tag = build(ToolChoice::Required, &tools).unwrap();
    let serialized = serde_json::to_string(&tag).unwrap();

    assert!(serialized.contains(r#""type":"or""#));
    assert!(serialized.contains(r#""value":"fast""#));
    assert!(serialized.contains(r#""pattern":"(?:[^<]|<[^|]){2,8}""#));
    assert!(serialized.contains(r#""$defs":{"payload""#));
}

#[test]
fn none_and_empty_tools_do_not_build_a_constraint() {
    let tools = [tool("lookup", json!({"type": "object"}), Some(true))];

    assert!(build(ToolChoice::None, &tools).is_none());
    assert!(build(ToolChoice::Required, &[]).is_none());
}
