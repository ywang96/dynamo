// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Kimi K3 XTML structural-tag constraints for xgrammar.

use anyhow::{Context, bail};
use dynamo_parsers::tool_calling::{ToolChoice, ToolDefinition};
use serde_json::{Map, Value};
use xgrammar_structural_tag::format::{
    Format, OptionalFormat, StarFormat, StructuralTag, TagFormat,
};

const OPEN: &str = "<|open|>";
const CLOSE: &str = "<|close|>";
const SEP: &str = "<|sep|>";
const RESPONSE_OPEN: &str = "<|open|>response<|sep|>";
const RESPONSE_CLOSE: &str = "<|close|>response<|sep|>";
const TOOLS_OPEN: &str = "<|open|>tools<|sep|>";
const TOOLS_CLOSE: &str = "<|close|>tools<|sep|>";
const CALL_CLOSE: &str = "<|close|>call<|sep|>";
const ARGUMENT_CLOSE: &str = "<|close|>argument<|sep|>";
const MESSAGE_CLOSE: &str = "<|close|>message<|sep|>";
const STRING_ATOM: &str = "(?:[^<]|<[^|])";

/// Build the K3 XTML constraint when the request's tool policy needs one.
///
/// This mirrors vLLM's K3 structural-tag activation: none, empty tools, and
/// non-strict auto do not constrain generation; strict auto is optional;
/// required uses a mandatory tools channel. Named choice is outside the K3 API.
pub fn build_kimi_k3_structural_tag(
    tool_choice: &ToolChoice,
    tools: &[ToolDefinition],
) -> anyhow::Result<Option<Value>> {
    if matches!(tool_choice, ToolChoice::Named(_)) {
        bail!("named tool choice is not supported for Kimi K3");
    }
    if tools.is_empty() || matches!(tool_choice, ToolChoice::None) {
        return Ok(None);
    }
    if matches!(tool_choice, ToolChoice::Auto)
        && !tools.iter().any(|tool| tool.strict == Some(true))
    {
        return Ok(None);
    }

    let tools_channel = Format::Tag(k3_tools_channel(tools));
    let tools_part = if matches!(tool_choice, ToolChoice::Auto) {
        Format::Optional(OptionalFormat {
            content: Box::new(tools_channel),
        })
    } else {
        tools_channel
    };

    let format = Format::sequence(vec![
        Format::Optional(OptionalFormat {
            content: Box::new(Format::const_string(RESPONSE_OPEN)),
        }),
        Format::tag("", Format::any_text(), RESPONSE_CLOSE),
        tools_part,
        Format::Optional(OptionalFormat {
            content: Box::new(Format::const_string(MESSAGE_CLOSE)),
        }),
    ]);

    serde_json::to_value(StructuralTag::new(format))
        .context("failed to serialize Kimi K3 structural tag")
        .map(Some)
}

fn k3_tools_channel(tools: &[ToolDefinition]) -> TagFormat {
    TagFormat::new(
        TOOLS_OPEN,
        Format::tags_with_separator(tools.iter().map(k3_call_tag).collect(), "", true, false),
        TOOLS_CLOSE,
    )
}

fn k3_call_tag(tool: &ToolDefinition) -> TagFormat {
    let parameters = if tool.strict == Some(false) {
        Value::Bool(true)
    } else {
        tool.parameters.clone().unwrap_or(Value::Bool(true))
    };
    TagFormat::new(
        format!(
            "{OPEN}call tool=\"{}\" index=\"",
            escape_attribute(&tool.name)
        ),
        Format::sequence(vec![
            Format::regex("[0-9]+"),
            Format::const_string(format!("\"{SEP}")),
            k3_arguments_block(&parameters),
        ]),
        CALL_CLOSE,
    )
}

fn k3_arguments_block(parameters: &Value) -> Format {
    let Some(parameters) = parameters.as_object() else {
        return star(Format::Tag(k3_permissive_argument_tag()));
    };
    let Some(properties) = parameters.get("properties").and_then(Value::as_object) else {
        return star(Format::Tag(k3_permissive_argument_tag()));
    };
    if properties.is_empty() {
        return star(Format::Tag(k3_permissive_argument_tag()));
    }

    let root_definitions: Map<String, Value> = ["$defs", "definitions"]
        .into_iter()
        .filter_map(|key| {
            parameters
                .get(key)
                .filter(|value| value.is_object())
                .cloned()
                .map(|value| (key.to_string(), value))
        })
        .collect();
    let formats: Vec<Format> = properties
        .iter()
        .map(|(key, schema)| {
            k3_argument_tag(key, schema, &root_definitions)
                .map(Format::Tag)
                .unwrap_or_else(|| Format::Tag(k3_permissive_argument_tag()))
        })
        .collect();
    let arguments = if formats.len() == 1 {
        formats.into_iter().next().expect("one argument format")
    } else {
        Format::or(formats)
    };
    star(arguments)
}

fn k3_argument_tag(
    key: &str,
    schema: &Value,
    root_definitions: &Map<String, Value>,
) -> Option<TagFormat> {
    let schema = schema.as_object()?;
    let json_type = schema.get("type")?.as_str()?;
    let xtml_type = match json_type {
        "string" => "string",
        "integer" | "number" => "number",
        "boolean" => "boolean",
        "null" => "null",
        "object" => "object",
        "array" => "array",
        _ => return None,
    };
    let begin = format!(
        "{OPEN}argument key=\"{}\" type=\"{xtml_type}\"{SEP}",
        escape_attribute(key)
    );

    let content = if xtml_type == "string" {
        k3_string_format(schema)
    } else {
        let mut embedded = schema.clone();
        for (key, value) in root_definitions {
            embedded.entry(key.clone()).or_insert_with(|| value.clone());
        }
        Format::json_schema(Value::Object(embedded))
    };
    Some(TagFormat::new(begin, content, ARGUMENT_CLOSE))
}

fn k3_string_format(schema: &Map<String, Value>) -> Format {
    let enum_values = schema
        .get("enum")
        .and_then(Value::as_array)
        .cloned()
        .or_else(|| schema.get("const").cloned().map(|value| vec![value]));
    if let Some(values) = enum_values
        && !values.is_empty()
        && values.len() <= 256
        && values
            .iter()
            .all(|value| value.as_str().is_some_and(|value| !value.contains("<|")))
    {
        let mut branches = values
            .into_iter()
            .map(|value| Format::const_string(value.as_str().expect("checked string")))
            .collect::<Vec<_>>();
        return if branches.len() == 1 {
            branches.remove(0)
        } else {
            Format::or(branches)
        };
    }

    if let Some(pattern) = bounded_string_regex(schema) {
        Format::regex(pattern)
    } else {
        Format::any_text_excluding(&[CLOSE])
    }
}

fn bounded_string_regex(schema: &Map<String, Value>) -> Option<String> {
    let max = schema.get("maxLength")?.as_u64()?;
    if max > 4096 {
        return None;
    }
    let min = schema
        .get("minLength")
        .and_then(Value::as_u64)
        .filter(|min| min <= &max)
        .unwrap_or(0);
    Some(format!("{STRING_ATOM}{{{min},{max}}}"))
}

fn k3_permissive_argument_tag() -> TagFormat {
    TagFormat::new(
        format!("{OPEN}argument "),
        Format::sequence(vec![
            Format::regex(r"[^<]*<\|sep\|>"),
            Format::any_text_excluding(&[CLOSE]),
        ]),
        ARGUMENT_CLOSE,
    )
}

fn star(content: Format) -> Format {
    Format::Star(StarFormat {
        content: Box::new(content),
    })
}

fn escape_attribute(value: &str) -> String {
    value.replace('&', "&amp;").replace('"', "&quot;")
}
