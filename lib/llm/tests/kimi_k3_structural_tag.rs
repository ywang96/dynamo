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

/// Both choices lower to `[response, tools, optional(message-close)]`, so the
/// tools channel is always the middle element. It is a bare tag under
/// `required` and wrapped in `optional` under `auto`.
fn tools_part(tag: &Value) -> &Value {
    &tag["format"]["elements"][1]
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
    let tools_part = tools_part(&tag);

    assert_eq!(tag["type"], "structural_tag");
    assert_eq!(tag["format"]["type"], "sequence");
    assert_eq!(tag["format"]["elements"].as_array().unwrap().len(), 3);
    assert_eq!(tools_part["type"], "tag");
    assert_eq!(tools_part["begin"], "<|open|>tools<|sep|>");
    assert_eq!(tools_part["end"], "<|close|>tools<|sep|>");
    assert_eq!(tools_part["content"]["type"], "tags_with_separator");
    assert_eq!(tools_part["content"]["at_least_one"], true);
    assert_eq!(
        tools_part["content"]["tags"][0]["begin"],
        "<|open|>call tool=\"lookup\" index=\""
    );
}

#[test]
fn auto_builds_for_all_strict_values_and_keeps_tools_optional() {
    // Every element here is optional, so the empty string derives and the turn
    // may end producing nothing. That is a known defect, not an oversight --
    // requiring a forward channel instead strands the model mid-marker on
    // `<|close|>message` and it derails into arbitrary text, which is worse
    // because the client accepts it. See the note on `KimiK3StructuralTagBuilder
    // ::build` in vllm-parser.
    for strict in [None, Some(true), Some(false)] {
        let tools = [tool("lookup", json!({"type": "object"}), strict)];
        let tag = build(ToolChoice::Auto, &tools).expect("auto must build a tag");
        let elements = tag["format"]["elements"].as_array().unwrap();

        assert_eq!(elements.len(), 3);
        assert_eq!(tools_part(&tag)["type"], "optional");
        assert_eq!(tools_part(&tag)["content"]["begin"], "<|open|>tools<|sep|>");
        assert_eq!(elements[2]["content"]["value"], "<|close|>message<|sep|>");
    }
}

#[test]
fn absent_strict_uses_parameters_but_explicit_false_opts_out() {
    let parameters = json!({
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"]
    });

    for strict in [None, Some(true)] {
        let tools = [tool("lookup", parameters.clone(), strict)];
        let tag = build(ToolChoice::Auto, &tools).expect("auto must build a tag");
        assert!(
            serde_json::to_string(&tag)
                .unwrap()
                .contains(r#""required":["query"]"#)
        );
    }

    let tools = [tool("lookup", parameters, Some(false))];
    let tag = build(ToolChoice::Auto, &tools).expect("auto must build a tag");
    assert!(
        !serde_json::to_string(&tag)
            .unwrap()
            .contains(r#""required":["query"]"#)
    );
}

#[test]
fn named_choice_is_not_part_of_the_k3_api_contract() {
    let tools = [
        tool("first", json!({"type": "object"}), None),
        tool("second", json!({"type": "object"}), None),
    ];

    let err = build_kimi_k3_structural_tag(&ToolChoice::Named("second".into()), &tools)
        .expect_err("K3 only supports auto, required, and none");

    assert!(
        err.to_string()
            .contains("named tool choice is not supported")
    );
}

#[test]
fn argument_formats_use_mke_typed_and_raw_json_channels() {
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
    let call_content = &tools_part(&tag)["content"]["tags"][0]["content"];

    assert_eq!(call_content["elements"][0]["pattern"], "[1-9][0-9]*");
    assert_eq!(call_content["elements"][2]["type"], "or");
    assert_eq!(
        call_content["elements"][2]["elements"][1]["begin"],
        "<|open|>json type=\"object\"<|sep|>"
    );
    assert!(serialized.contains(r#""value":"fast""#));
    assert!(serialized.contains(r#""$defs":{"payload""#));
}

#[test]
fn none_builds_a_response_only_constraint() {
    let tools = [tool("lookup", json!({"type": "object"}), Some(true))];

    let tag = build(ToolChoice::None, &tools).expect("none must build a ban tag");
    let elements = tag["format"]["elements"].as_array().unwrap();

    assert_eq!(elements.len(), 3);
    assert!(
        elements
            .iter()
            .all(|element| element["begin"] != "<|open|>tools<|sep|>")
    );
}

#[test]
fn empty_tools_do_not_build_a_constraint() {
    assert!(build(ToolChoice::Required, &[]).is_none());
}
