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

/// Build with thinking on, which is K3's default and the production shape: the
/// prompt prefix ends at `<|open|>think<|sep|>`, so the model still owes
/// `<|open|>response<|sep|>`.
fn build(tool_choice: ToolChoice, tools: &[ToolDefinition]) -> Option<Value> {
    build_kimi_k3_structural_tag(&tool_choice, tools, true).unwrap()
}

/// Build with thinking off: the prompt prefix already opened the response
/// channel.
fn build_without_thinking(tool_choice: ToolChoice, tools: &[ToolDefinition]) -> Option<Value> {
    build_kimi_k3_structural_tag(&tool_choice, tools, false).unwrap()
}

/// Under `required` the lowering is `[optional(response), tools,
/// optional(message-close)]`, so the tools channel is the middle element.
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
fn auto_requires_a_forward_channel_but_keeps_tools_reachable() {
    // With thinking on the turn cannot end without opening a channel, and both
    // branches start at a marker so `<|close|>` is masked outright rather than
    // three tokens into `<|close|>message<|sep|>`.
    for strict in [None, Some(true), Some(false)] {
        let tools = [tool("lookup", json!({"type": "object"}), strict)];
        let tag = build(ToolChoice::Auto, &tools).expect("auto must build a tag");
        let elements = tag["format"]["elements"].as_array().unwrap();

        assert_eq!(elements.len(), 2);
        assert_eq!(elements[0]["type"], "or");
        assert_eq!(elements[1]["content"]["value"], "<|close|>message<|sep|>");

        // `or([sequence[optional(response), tools], response])`.
        let branches = elements[0]["elements"].as_array().unwrap();
        assert_eq!(branches.len(), 2);
        assert_eq!(
            branches[0]["elements"][0]["content"]["elements"][0]["value"],
            "<|open|>response<|sep|>"
        );
        assert_eq!(branches[0]["elements"][1]["begin"], "<|open|>tools<|sep|>");
        assert_eq!(branches[1]["elements"][0]["value"], "<|open|>response<|sep|>");
    }
}

#[test]
fn auto_without_thinking_keeps_the_permissive_shape() {
    // The prompt already opened the response channel, so its marker stays
    // optional and there is no marker to anchor a requirement on. Unchanged
    // from before the forward-channel fix.
    let tools = [tool("lookup", json!({"type": "object"}), None)];
    let tag = build_without_thinking(ToolChoice::Auto, &tools).expect("auto must build a tag");
    let elements = tag["format"]["elements"].as_array().unwrap();

    assert_eq!(elements.len(), 3);
    assert_eq!(
        elements[0]["content"]["elements"][0]["content"]["value"],
        "<|open|>response<|sep|>"
    );
    assert_eq!(tools_part(&tag)["type"], "optional");
    assert_eq!(tools_part(&tag)["content"]["begin"], "<|open|>tools<|sep|>");
    assert_eq!(elements[2]["content"]["value"], "<|close|>message<|sep|>");
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

    let err = build_kimi_k3_structural_tag(&ToolChoice::Named("second".into()), &tools, true)
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
fn empty_tools_still_constrain_the_response_channel_when_thinking() {
    // The 2026-08-08 BEAM shape. Without this the model may answer inside the
    // think channel and stop: 49 of 700 rows first-pass empty, none of which
    // sent a tool. An empty tool set lowers to a mandatory response channel
    // with no tools channel at all.
    for choice in [ToolChoice::Auto, ToolChoice::None] {
        let tag = build(choice, &[]).expect("thinking-on no-tools must build a tag");
        let elements = tag["format"]["elements"].as_array().unwrap();

        assert_eq!(elements.len(), 3, "{tag}");
        assert_eq!(elements[0]["value"], "<|open|>response<|sep|>");
        assert_eq!(elements[1]["end"], "<|close|>response<|sep|>");
        assert_eq!(elements[2]["content"]["value"], "<|close|>message<|sep|>");
        assert!(
            elements
                .iter()
                .all(|element| element["begin"] != "<|open|>tools<|sep|>"),
            "no tools channel should be emitted: {tag}"
        );
    }
}

#[test]
fn empty_tools_build_nothing_when_a_constraint_cannot_help() {
    // Nothing to force when `required` has no tools, and with thinking off the
    // prompt has already opened the response channel.
    assert!(build(ToolChoice::Required, &[]).is_none());
    for choice in [ToolChoice::Auto, ToolChoice::None, ToolChoice::Required] {
        assert!(build_without_thinking(choice, &[]).is_none());
    }
}

/// Every marker a branch may legally begin with, in grammar order. A mandatory
/// element ends the walk: nothing after it can be the first thing emitted.
fn first_markers(format: &Value, out: &mut Vec<String>) {
    match format["type"].as_str() {
        Some("sequence") => {
            for element in format["elements"].as_array().unwrap() {
                first_markers(element, out);
                if element["type"] != "optional" {
                    return;
                }
            }
        }
        Some("or") => {
            for element in format["elements"].as_array().unwrap() {
                first_markers(element, out);
            }
        }
        Some("optional") => first_markers(&format["content"], out),
        Some("const_string") => out.push(format["value"].as_str().unwrap().to_string()),
        Some("tag") => out.push(format["begin"].as_str().unwrap().to_string()),
        other => out.push(format!("<free text: {}>", other.unwrap_or("?"))),
    }
}

#[test]
fn a_thinking_turn_can_only_start_at_a_channel_open_marker() {
    // The regression this guards: with an optional response-open marker the
    // response body's free text is reachable at the first constrained position,
    // so a model reaching for `<|close|>message<|sep|>` commits
    // `<|close|>message` before `<|sep|>` is masked -- `any_text_excluding`
    // masks only the token that COMPLETES an excluded string -- and strands
    // mid-marker. Anchoring every branch at a marker moves the mask onto
    // `<|close|>` itself.
    let tools = [tool("lookup", json!({"type": "object"}), None)];
    for choice in [ToolChoice::Auto, ToolChoice::Required, ToolChoice::None] {
        let tag = build(choice, &tools).expect("must build a tag");
        let mut markers = Vec::new();
        first_markers(&tag["format"], &mut markers);

        assert!(
            markers
                .iter()
                .all(|m| m == "<|open|>response<|sep|>" || m == "<|open|>tools<|sep|>"),
            "only channel-open markers may start the turn, got {markers:?} for {tag}"
        );
    }
}
