// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_llm::preprocessor::DELEGATED_UNIFIED_PARSERS;
use dynamo_parsers::reasoning::get_available_reasoning_parsers;
use dynamo_parsers::tool_calling::parsers::get_available_tool_parsers;
use pyo3::prelude::*;

/// Get list of available tool parser names.
///
/// Combines the published `dynamo-parsers` registry with delegated unified
/// parsers (e.g. `kimi_k3`). Callers that gate on this list — notably the
/// `--dyn-tool-call-parser` CLI `choices` — must accept both.
#[pyfunction]
pub fn get_tool_parser_names() -> Vec<&'static str> {
    let mut names = get_available_tool_parsers();
    for &name in DELEGATED_UNIFIED_PARSERS {
        if !names.contains(&name) {
            names.push(name);
        }
    }
    names
}

/// Get list of available reasoning parser names.
///
/// Combines the published `dynamo-parsers` registry with delegated unified
/// parsers (e.g. `kimi_k3`); see [`get_tool_parser_names`].
#[pyfunction]
pub fn get_reasoning_parser_names() -> Vec<&'static str> {
    let mut names = get_available_reasoning_parsers();
    for &name in DELEGATED_UNIFIED_PARSERS {
        if !names.contains(&name) {
            names.push(name);
        }
    }
    names
}

/// Add parsers module functions to the Python module
pub fn add_to_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_tool_parser_names, m)?)?;
    m.add_function(wrap_pyfunction!(get_reasoning_parser_names, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // These names feed the `--dyn-reasoning-parser` / `--dyn-tool-call-parser`
    // argparse `choices`, so a delegated parser missing here is rejected at the CLI.
    #[test]
    fn delegated_kimi_k3_is_an_exposed_reasoning_parser() {
        let names = get_reasoning_parser_names();
        assert!(names.contains(&"kimi_k3"), "reasoning names: {names:?}");
        // Appended, not duplicated, if the registry ever adds it too.
        assert_eq!(names.iter().filter(|n| **n == "kimi_k3").count(), 1);
    }

    #[test]
    fn delegated_kimi_k3_is_an_exposed_tool_parser() {
        let names = get_tool_parser_names();
        assert!(names.contains(&"kimi_k3"), "tool names: {names:?}");
        assert_eq!(names.iter().filter(|n| **n == "kimi_k3").count(), 1);
    }
}
