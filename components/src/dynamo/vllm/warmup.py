# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Warm a worker's JIT kernels before registering it in discovery."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from vllm.inputs import TokensPrompt
from vllm.sampling_params import SamplingParams

logger = logging.getLogger(__name__)

# Real sampling is required to compile the top-k/top-p kernels; the greedy
# one-token health check does not exercise them.
WARMUP_TEMPERATURE = 0.7
WARMUP_TOP_P = 0.9
# vLLM runs its Triton top-k/top-p sampler only at decode batch size >= 8.
WARMUP_DECODE_FILL = 8


@dataclass(frozen=True)
class WarmupShape:
    input_tokens: int
    output_tokens: int
    temperature: float
    top_p: float


@dataclass(frozen=True)
class WarmupResult:
    completed: int
    failed: int
    timed_out: bool
    elapsed_s: float


@dataclass
class _WarmupProgress:
    completed: int = 0
    failed: int = 0


def parse_input_lens(spec: str) -> list[int]:
    """Parse comma-separated prompt lengths."""
    return [int(value.strip()) for value in spec.split(",") if value.strip()]


def build_warmup_shapes(
    input_lens_spec: str,
    output_tokens: int,
    temperature: float,
    top_p: float,
) -> list[WarmupShape]:
    """Build one warmup shape per configured prompt length."""
    return [
        WarmupShape(input_tokens, output_tokens, temperature, top_p)
        for input_tokens in parse_input_lens(input_lens_spec)
    ]


async def _drive_one(
    engine_client: Any,
    shape: WarmupShape,
    bos_token_id: int,
    request_id: str,
) -> None:
    prompt = TokensPrompt(prompt_token_ids=[bos_token_id] * shape.input_tokens)
    sampling_params = SamplingParams(
        temperature=shape.temperature,
        top_p=shape.top_p,
        max_tokens=shape.output_tokens,
        ignore_eos=True,
    )
    async for _ in engine_client.generate(prompt, sampling_params, request_id):
        pass


async def _run_all(
    engine_client: Any,
    shapes: Sequence[WarmupShape],
    bos_token_id: int,
    concurrency: int,
    iterations: int,
    progress: _WarmupProgress,
) -> None:
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def guarded(shape: WarmupShape, request_id: str) -> None:
        async with semaphore:
            try:
                await _drive_one(engine_client, shape, bos_token_id, request_id)
            except Exception as error:
                progress.failed += 1
                logger.warning("Warmup request %s failed: %s", request_id, error)
            else:
                progress.completed += 1

    requests = [
        (shape, f"warmup-{iteration}-{shape_index}")
        for iteration in range(iterations)
        for shape_index, shape in enumerate(shapes)
    ]
    await asyncio.gather(
        *(guarded(shape, request_id) for shape, request_id in requests)
    )


async def run_warmup(
    engine_client: Any,
    shapes: Sequence[WarmupShape],
    bos_token_id: int,
    concurrency: int,
    iterations: int,
    timeout_s: float,
    monotonic: Callable[[], float] = time.monotonic,
) -> WarmupResult:
    """Drive warmup shapes, failing open on request errors or timeout."""
    start = monotonic()
    progress = _WarmupProgress()
    try:
        await asyncio.wait_for(
            _run_all(
                engine_client,
                shapes,
                bos_token_id,
                concurrency,
                iterations,
                progress,
            ),
            timeout=timeout_s,
        )
    except asyncio.TimeoutError:
        logger.warning("Warmup timed out after %.1fs; registering anyway", timeout_s)
        return WarmupResult(
            progress.completed,
            progress.failed,
            True,
            monotonic() - start,
        )
    except Exception as error:
        logger.warning("Warmup aborted (%s); registering anyway", error)

    return WarmupResult(
        progress.completed,
        progress.failed,
        False,
        monotonic() - start,
    )


async def maybe_run_warmup(
    engine_client: Any, config: Any, bos_token_id: int
) -> WarmupResult | None:
    """Run role-agnostic pre-registration warmup when configured."""
    if not config.warmup_enabled:
        return None

    try:
        if config.warmup_concurrency < WARMUP_DECODE_FILL:
            logger.warning(
                "warmup_concurrency=%d < %d: the decode batch may not reach the "
                "threshold for vLLM's Triton top-k/top-p sampler",
                config.warmup_concurrency,
                WARMUP_DECODE_FILL,
            )

        shapes = build_warmup_shapes(
            config.warmup_input_lens,
            config.warmup_output_tokens,
            WARMUP_TEMPERATURE,
            WARMUP_TOP_P,
        )
        shapes.extend(
            WarmupShape(
                128,
                config.warmup_output_tokens,
                WARMUP_TEMPERATURE,
                WARMUP_TOP_P,
            )
            for _ in range(WARMUP_DECODE_FILL)
        )
    except (AttributeError, TypeError, ValueError) as error:
        logger.warning("Worker warmup setup failed (%s); registering anyway", error)
        return None

    logger.info(
        "Starting worker warmup: %d shapes (timeout %ds)",
        len(shapes),
        config.warmup_timeout,
    )
    result = await run_warmup(
        engine_client,
        shapes,
        bos_token_id=bos_token_id,
        concurrency=config.warmup_concurrency,
        iterations=config.warmup_iterations,
        timeout_s=config.warmup_timeout,
    )
    logger.info("Worker warmup done: %s", result)
    return result
