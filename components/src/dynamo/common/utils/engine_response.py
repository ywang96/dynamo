#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""Utilities for engine response processing."""

import logging
from typing import Optional, Union

NormalizedFinishReason = Union[str, dict[str, str], None]


def normalize_finish_reason(finish_reason: Optional[str]) -> NormalizedFinishReason:
    """
    Normalize engine finish reasons to Dynamo-compatible values.

    Engine may return finish reasons that aren't recognized by Dynamo's Rust
    layer (`FinishReason` enum at `lib/llm/src/protocols/common.rs`). This
    helper centralizes the mapping so all emission sites in the worker stay
    consistent.

    Maps:
      - "abort..."     -> "cancelled"
      - "error"        -> {"error": "backend error"}
      - "error: <msg>" -> {"error": "<msg>"}
      - everything else passed through unchanged

    The dict form matches the Rust `FinishReason::Error(String)` serde newtype
    variant, which deserializes from `{"error": "<msg>"}`. Emitting a bare
    `"error"` string from the worker would fail serde with
    `invalid type: unit variant, expected newtype variant` and be dropped by
    the frontend (see production trace 5cf40fe6-b935-4aa0-9e39-158fe54ea178
    and upstream ai-dynamo/dynamo#8549).

    Centralized here so it can be removed when the Rust layer accepts bare
    error strings; until then every worker emission site MUST route through
    this helper.
    """
    if finish_reason and finish_reason.startswith("abort"):
        logging.debug(f"Normalizing finish reason: {finish_reason} to cancelled")
        return "cancelled"
    if finish_reason == "error":
        return {"error": "backend error"}
    if finish_reason and finish_reason.startswith("error:"):
        msg = finish_reason.split(":", 1)[1].strip()
        return {"error": msg or "backend error"}
    return finish_reason
