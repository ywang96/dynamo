# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for local vLLM prefill health probes."""

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import dynamo.vllm.handlers as mod
from dynamo.health_check import HEALTH_CHECK_KEY
from dynamo.vllm.multimodal_utils.request_processor import PreparedMultimodalInput

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.core,
]


@asynccontextmanager
async def _noop_abort_monitor(*_args, **_kwargs):
    yield


def _engine_response(kv_transfer_params):
    return SimpleNamespace(
        kv_transfer_params=kv_transfer_params,
        outputs=[SimpleNamespace(token_ids=[42])],
        prompt_token_ids=[1],
        num_cached_tokens=0,
    )


def _make_handler(prepared_request, response):
    captured = {}

    async def generate(_prompt, sampling_params, _request_id, **_kwargs):
        captured["sampling_params"] = sampling_params
        yield response

    processor = SimpleNamespace(
        prepare_input=AsyncMock(
            return_value=PreparedMultimodalInput(
                request=prepared_request,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        ),
        build_prefill_handoff=MagicMock(return_value=None),
    )
    handler = mod.PrefillWorkerHandler.__new__(mod.PrefillWorkerHandler)
    handler._multimodal_request_processor = processor
    handler._build_prompt_from_request = MagicMock(
        return_value=({"prompt_token_ids": [1]}, None, None)
    )
    handler._abort_monitor = _noop_abort_monitor
    handler._resolve_lora_request = MagicMock(return_value=None)
    handler._to_local_dp_rank = MagicMock(return_value=None)
    handler._log_with_lora_context = MagicMock()
    handler.default_sampling_params = {}
    handler.model_max_len = 4096
    handler.config = SimpleNamespace(enable_rl=False)
    handler.engine_client = SimpleNamespace(
        vllm_config=object(),
        generate=generate,
    )
    return handler, captured


@pytest.mark.asyncio
async def test_prefill_health_probe_skips_remote_kv_handoff():
    prepared_request = {
        "token_ids": [1],
        "sampling_options": {
            "extra_args": {
                "kv_transfer_params": {"do_remote_decode": True},
                "unrelated": "preserved",
            }
        },
        "stop_conditions": {"max_tokens": 1},
        "output_options": {},
    }
    response = _engine_response({"remote_engine_id": "must-not-leak"})
    handler, captured = _make_handler(prepared_request, response)
    request = {**prepared_request, HEALTH_CHECK_KEY: True}
    context = MagicMock()
    context.trace_headers.return_value = {}

    with patch.object(mod, "make_kv_connector_protocol") as make_protocol:
        chunks = [
            chunk
            async for chunk in handler._generate_token_mode(
                request, context, "health-probe"
            )
        ]

    make_protocol.assert_not_called()
    sampling_params = captured["sampling_params"]
    assert sampling_params.max_tokens == 1
    assert sampling_params.min_tokens == 1
    assert "kv_transfer_params" not in sampling_params.extra_args
    assert sampling_params.extra_args["unrelated"] == "preserved"
    assert chunks[0]["token_ids"] == [42]
    assert chunks[0]["disaggregated_params"] is None


@pytest.mark.asyncio
async def test_prefill_request_preserves_remote_kv_handoff():
    request = {
        "token_ids": [1],
        "sampling_options": {},
        "stop_conditions": {"max_tokens": 1},
        "output_options": {},
    }
    response = _engine_response({"remote_engine_id": "prefill-engine"})
    handler, captured = _make_handler(request, response)
    context = MagicMock()
    context.trace_headers.return_value = {}
    prefill_params = {"do_remote_decode": True}
    decode_params = {"remote_engine_id": "prefill-engine"}
    protocol = MagicMock()
    protocol.prefill_request_kv_transfer_params.return_value = prefill_params
    protocol.decode_request_kv_transfer_params.return_value = decode_params

    with patch.object(
        mod, "make_kv_connector_protocol", return_value=protocol
    ) as make_protocol:
        chunks = [
            chunk
            async for chunk in handler._generate_token_mode(
                request, context, "ordinary-prefill"
            )
        ]

    make_protocol.assert_called_once_with(handler.engine_client.vllm_config)
    protocol.prefill_request_kv_transfer_params.assert_called_once_with()
    protocol.decode_request_kv_transfer_params.assert_called_once_with(response)
    sampling_params = captured["sampling_params"]
    assert sampling_params.max_tokens == 1
    assert sampling_params.min_tokens == 1
    assert sampling_params.extra_args["kv_transfer_params"] is prefill_params
    assert chunks[0]["token_ids"] == [42]
    assert chunks[0]["disaggregated_params"] == {"kv_transfer_params": decode_params}
