# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for worker-side pre-registration warmup."""

import argparse
import asyncio
from types import SimpleNamespace

import pytest

from dynamo.vllm.backend_args import DynamoVllmArgGroup, DynamoVllmConfig
from dynamo.vllm.warmup import (
    WARMUP_DECODE_FILL,
    WARMUP_TEMPERATURE,
    WARMUP_TOP_P,
    WarmupResult,
    build_warmup_shapes,
    maybe_run_warmup,
    parse_input_lens,
    run_warmup,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
]


def test_parse_input_lens():
    assert parse_input_lens("128,2048,8192") == [128, 2048, 8192]
    assert parse_input_lens(" 64 , 512 ") == [64, 512]


def test_build_warmup_shapes_covers_sampling_kernels():
    shapes = build_warmup_shapes("128,2048,8192", 16, WARMUP_TEMPERATURE, WARMUP_TOP_P)

    assert [shape.input_tokens for shape in shapes] == [128, 2048, 8192]
    assert all(shape.temperature > 0.0 and shape.top_p < 1.0 for shape in shapes)
    assert all(shape.output_tokens >= 4 for shape in shapes)


class _FakeEngine:
    """Record generated shapes and optionally fail or block requests."""

    def __init__(self, raise_exc=None, hang=False):
        self.calls = []
        self._raise = raise_exc
        self._hang = hang

    async def generate(self, prompt, sampling_params, request_id, **kwargs):
        self.calls.append(
            (
                list(prompt["prompt_token_ids"]),
                sampling_params.temperature,
                sampling_params.top_p,
                sampling_params.max_tokens,
            )
        )
        if self._raise is not None:
            raise self._raise
        if self._hang:
            await asyncio.Event().wait()
        yield object()


def test_run_warmup_drives_each_shape():
    engine = _FakeEngine()
    shapes = build_warmup_shapes("128,2048", 8, WARMUP_TEMPERATURE, WARMUP_TOP_P)

    result = asyncio.run(
        run_warmup(
            engine,
            shapes,
            bos_token_id=1,
            concurrency=2,
            iterations=1,
            timeout_s=30,
        )
    )

    assert result.completed == 2
    assert result.failed == 0
    assert result.timed_out is False
    assert sorted(len(call[0]) for call in engine.calls) == [128, 2048]
    assert all(
        temperature > 0 and top_p < 1 and max_tokens == 8
        for _, temperature, top_p, max_tokens in engine.calls
    )


def test_run_warmup_fail_open_on_request_exception():
    engine = _FakeEngine(raise_exc=RuntimeError("boom"))

    result = asyncio.run(
        run_warmup(
            engine,
            build_warmup_shapes("128", 8, WARMUP_TEMPERATURE, WARMUP_TOP_P),
            bos_token_id=1,
            concurrency=1,
            iterations=1,
            timeout_s=30,
        )
    )

    assert result.failed == 1
    assert result.completed == 0
    assert result.timed_out is False


def test_run_warmup_propagates_cancellation():
    engine = _FakeEngine(raise_exc=asyncio.CancelledError())

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            run_warmup(
                engine,
                build_warmup_shapes("128", 8, WARMUP_TEMPERATURE, WARMUP_TOP_P),
                bos_token_id=1,
                concurrency=1,
                iterations=1,
                timeout_s=30,
            )
        )


def test_run_warmup_fail_open_on_timeout():
    result = asyncio.run(
        run_warmup(
            _FakeEngine(hang=True),
            build_warmup_shapes("128", 8, WARMUP_TEMPERATURE, WARMUP_TOP_P),
            bos_token_id=1,
            concurrency=1,
            iterations=1,
            timeout_s=0,
        )
    )

    assert result.timed_out is True


def test_warmup_disabled_by_default():
    config = DynamoVllmConfig()

    assert config.warmup_enabled is False
    assert config.warmup_input_lens == "128,2048,8192"
    assert config.warmup_output_tokens == 16
    assert config.warmup_concurrency == 12
    assert config.warmup_iterations == 1
    assert config.warmup_timeout == 300


def test_warmup_args_parse():
    parser = argparse.ArgumentParser()
    DynamoVllmArgGroup().add_arguments(parser)

    args = parser.parse_args(
        [
            "--warmup",
            "--warmup-input-lens",
            "64,512",
            "--warmup-output-tokens",
            "8",
            "--warmup-concurrency",
            "9",
            "--warmup-iterations",
            "2",
            "--warmup-timeout",
            "90",
        ]
    )

    assert args.warmup_enabled is True
    assert args.warmup_input_lens == "64,512"
    assert args.warmup_output_tokens == 8
    assert args.warmup_concurrency == 9
    assert args.warmup_iterations == 2
    assert args.warmup_timeout == 90


def _warmup_config(**overrides):
    values = {
        "warmup_enabled": True,
        "warmup_input_lens": "128,256",
        "warmup_output_tokens": 8,
        "warmup_concurrency": 12,
        "warmup_iterations": 1,
        "warmup_timeout": 30,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_maybe_run_warmup_runs_when_enabled(monkeypatch):
    calls = []

    async def fake_run_warmup(
        engine_client, shapes, bos_token_id, concurrency, iterations, timeout_s
    ):
        calls.append(len(shapes))
        return WarmupResult(len(shapes), 0, False, 1.0)

    monkeypatch.setattr("dynamo.vllm.warmup.run_warmup", fake_run_warmup)

    result = asyncio.run(maybe_run_warmup(object(), _warmup_config(), bos_token_id=1))

    assert result is not None
    assert calls == [2 + WARMUP_DECODE_FILL]


def test_maybe_run_warmup_skipped_when_disabled(monkeypatch):
    calls = []

    async def fake_run_warmup(*args, **kwargs):
        calls.append(1)
        return None

    monkeypatch.setattr("dynamo.vllm.warmup.run_warmup", fake_run_warmup)

    result = asyncio.run(
        maybe_run_warmup(object(), _warmup_config(warmup_enabled=False), bos_token_id=1)
    )

    assert result is None
    assert calls == []


@pytest.mark.parametrize("output_tokens", [1, 8])
def test_maybe_run_warmup_is_role_agnostic(monkeypatch, output_tokens):
    captured = []

    async def fake_run_warmup(
        engine_client, shapes, bos_token_id, concurrency, iterations, timeout_s
    ):
        captured.extend(shapes)
        return WarmupResult(len(shapes), 0, False, 1.0)

    monkeypatch.setattr("dynamo.vllm.warmup.run_warmup", fake_run_warmup)

    result = asyncio.run(
        maybe_run_warmup(
            object(),
            _warmup_config(warmup_input_lens="128", warmup_output_tokens=output_tokens),
            bos_token_id=1,
        )
    )

    assert result is not None
    assert len(captured) == 1 + WARMUP_DECODE_FILL
    assert all(shape.output_tokens == output_tokens for shape in captured)


def test_maybe_run_warmup_fail_open_on_bad_config():
    result = asyncio.run(
        maybe_run_warmup(
            object(),
            _warmup_config(warmup_input_lens="not-an-int"),
            bos_token_id=1,
        )
    )

    assert result is None
