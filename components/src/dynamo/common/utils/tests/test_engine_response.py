# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for normalize_finish_reason().

The helper centralizes the mapping from engine-emitted finish_reason strings
to the Dynamo-compatible wire format. Critically, it converts bare and
prefixed "error[: ...]" strings into the `{"error": "<msg>"}` object form
that matches the Rust FinishReason::Error(String) serde newtype variant.

Without that conversion the FE drops the chunk with
`invalid type: unit variant, expected newtype variant` (see production
trace 5cf40fe6-b935-4aa0-9e39-158fe54ea178 and ai-dynamo/dynamo#8549),
so these tests are the regression boundary for that bug.

The module is loaded with importlib so the test runs without pulling in
the full dynamo package (which requires CUDA / dynamo.llm). Matches the
sibling tests under this directory (e.g. test_topology.py).
"""

import importlib.util
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Module loading: import engine_response.py directly so the test does not
# require the full dynamo package import chain.
# ---------------------------------------------------------------------------
_ENGINE_RESPONSE_PY = (
    Path(__file__).resolve().parents[2] / "utils" / "engine_response.py"
)


def _load_engine_response_module():
    spec = importlib.util.spec_from_file_location(
        "engine_response", _ENGINE_RESPONSE_PY
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


engine_response = _load_engine_response_module()
normalize_finish_reason = engine_response.normalize_finish_reason


# Marker categories required by tests/report_pytest_markers.py
# Matches sibling tests under this directory (e.g. test_topology.py).
pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_normalize_abort_returns_cancelled():
    """Regression: existing "abort..." -> "cancelled" mapping is preserved."""
    assert normalize_finish_reason("abort") == "cancelled"
    assert normalize_finish_reason("abort: client disconnect") == "cancelled"


def test_normalize_bare_error_returns_dict():
    """Bare "error" must become the FE-compatible object form."""
    assert normalize_finish_reason("error") == {"error": "backend error"}


def test_normalize_error_prefix_returns_dict():
    """`"error: <msg>"` -> `{"error": "<msg>"}` with whitespace stripped."""
    assert normalize_finish_reason("error: KV load failure") == {
        "error": "KV load failure"
    }
    # Also covers the four worker emission sites we route through the helper.
    assert normalize_finish_reason("error: No outputs from vLLM engine") == {
        "error": "No outputs from vLLM engine"
    }
    assert normalize_finish_reason(
        "error: vllm engine returned empty outputs or internal abort; check worker logs"
    ) == {
        "error": "vllm engine returned empty outputs or internal abort; check worker logs"
    }


def test_normalize_error_prefix_empty_message_falls_back():
    """`"error:"` (no message) should not emit `{"error": ""}`."""
    assert normalize_finish_reason("error:") == {"error": "backend error"}
    assert normalize_finish_reason("error:   ") == {"error": "backend error"}


@pytest.mark.parametrize(
    "value",
    ["stop", "length", "tool_calls", "content_filter", None],
)
def test_normalize_passes_through_known_reasons(value):
    """Successful / known terminal reasons (and None) must pass through unchanged."""
    assert normalize_finish_reason(value) == value


def test_normalize_passes_through_unknown_reasons():
    """Arbitrary unknown strings pass through; helper is not a whitelist."""
    assert normalize_finish_reason("weird_unknown") == "weird_unknown"
    assert normalize_finish_reason("") == ""
