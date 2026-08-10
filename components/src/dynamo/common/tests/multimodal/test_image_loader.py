# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for ImageLoader in-flight dedup, cancellation, and error contract."""

import asyncio
import base64
from io import BytesIO
from unittest.mock import AsyncMock, patch

import pytest
from PIL import Image

from dynamo.common.http import HttpStatusError, HttpTimeoutError
from dynamo.common.http.url_validator import UrlValidationError, UrlValidationPolicy
from dynamo.common.multimodal.image_loader import (
    URL_VARIANT_KEY,
    ImageLoader,
    ImageValidationError,
)

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]

_FETCH_BYTES_PATH = "dynamo.common.multimodal.image_loader.fetch_bytes"


def _make_png_bytes() -> bytes:
    """Create a minimal valid PNG in memory."""
    img = Image.new("RGB", (2, 2), color="red")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


PNG_BYTES = _make_png_bytes()


def _permissive_policy(
    allowed_local_path: str | None = None,
) -> UrlValidationPolicy:
    """Return a policy that permits the schemes used by tests without DNS hits."""
    return UrlValidationPolicy(
        allow_http=True,
        allow_private_ips=True,
        allowed_local_path=allowed_local_path,
    )


def _mock_fetch_bytes(
    content: bytes = PNG_BYTES,
    delay: float = 0.0,
    side_effect: Exception | None = None,
) -> AsyncMock:
    """Return an AsyncMock drop-in for ``fetch_bytes(url, timeout, policy=...)``.

    Args:
        content: Raw bytes returned as the fetch result.
        delay: Seconds to sleep before responding (simulates network latency).
        side_effect: If set, the mock raises this exception instead of returning.
    """

    async def _fetch(url, timeout, *, policy=None, max_bytes=None):
        if delay > 0:
            await asyncio.sleep(delay)
        if side_effect is not None:
            raise side_effect
        return content

    return AsyncMock(side_effect=_fetch)


@pytest.fixture(autouse=True)
def loader() -> ImageLoader:
    return ImageLoader(
        cache_size=4,
        http_timeout=30.0,
        url_policy=_permissive_policy(),
    )


# --- Concurrent same-URL dedup ---


async def test_concurrent_same_url_deduplicates(loader: ImageLoader) -> None:
    """Two concurrent load_image calls for the same URL should issue only one HTTP fetch."""
    mock_fetch = _mock_fetch_bytes(delay=0.05)
    with patch(_FETCH_BYTES_PATH, mock_fetch):
        results = await asyncio.gather(
            loader.load_image("https://example.com/img.png"),
            loader.load_image("https://example.com/img.png"),
        )

    assert len(results) == 2
    assert results[0].size == results[1].size
    assert mock_fetch.call_count == 1


async def test_concurrent_different_urls_fetch_independently(
    loader: ImageLoader,
) -> None:
    """Different URLs should each get their own fetch."""
    mock_fetch = _mock_fetch_bytes()
    with patch(_FETCH_BYTES_PATH, mock_fetch):
        await asyncio.gather(
            loader.load_image("https://example.com/a.png"),
            loader.load_image("https://example.com/b.png"),
        )

    assert mock_fetch.call_count == 2


# --- Waiter cancellation isolation ---


async def test_waiter_cancellation_does_not_cancel_shared_task(
    loader: ImageLoader,
) -> None:
    """Cancelling one waiter should not prevent the other from getting the image."""
    mock_fetch = _mock_fetch_bytes(delay=0.1)
    with patch(_FETCH_BYTES_PATH, mock_fetch):
        task_a = asyncio.create_task(loader.load_image("https://example.com/img.png"))
        task_b = asyncio.create_task(loader.load_image("https://example.com/img.png"))
        await asyncio.sleep(0.01)
        task_a.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task_a

        result_b = await task_b
        assert isinstance(result_b, Image.Image)


# --- Retry after failure ---


async def test_retry_after_failure(loader: ImageLoader) -> None:
    """After a fetch failure, the next caller should start a fresh fetch."""
    fail_fetch = _mock_fetch_bytes(side_effect=HttpTimeoutError("timeout"))
    ok_fetch = _mock_fetch_bytes()

    with patch(_FETCH_BYTES_PATH, fail_fetch):
        with pytest.raises(ValueError, match="Timeout"):
            await loader.load_image("https://example.com/img.png")

    # _inflight should be cleared after failure
    assert "https://example.com/img.png" not in loader._inflight

    with patch(_FETCH_BYTES_PATH, ok_fetch):
        result = await loader.load_image("https://example.com/img.png")
        assert isinstance(result, Image.Image)


# --- Error contract preserved for non-HTTP ---


async def test_http_rejected_by_default() -> None:
    """Wiring smoke: ImageLoader plumbs ``url_policy`` to the validator.

    Validator behavior is covered in ``test_url_validator.py``;
    per-hop SSRF revalidation in ``http/test_http_backends.py``.
    """
    strict_loader = ImageLoader(
        cache_size=4, http_timeout=30.0, url_policy=UrlValidationPolicy()
    )
    with pytest.raises(ValueError, match="scheme|not allowed"):
        await strict_loader.load_image("http://example.com/x.png")


async def test_data_url_invalid_base64_normalized(loader: ImageLoader) -> None:
    """Malformed base64 data URL should raise ImageValidationError."""
    with pytest.raises(ImageValidationError, match="Invalid base64"):
        await loader.load_image("data:image/png;base64,NOT_VALID!!!")


async def test_data_url_non_image_rejected(loader: ImageLoader) -> None:
    """data: URL with non-image media type should raise ImageValidationError."""
    with pytest.raises(ImageValidationError, match="Data URL must be an image type"):
        await loader.load_image("data:text/plain;base64,aGVsbG8=")


async def test_http_corrupt_image_raises_validation_error(
    loader: ImageLoader,
) -> None:
    """HTTP bytes that cannot decode as an image are a client validation error."""
    mock_fetch = _mock_fetch_bytes(content=b"not an image")
    with patch(_FETCH_BYTES_PATH, mock_fetch):
        with pytest.raises(ImageValidationError, match="Invalid image data"):
            await loader.load_image("https://example.com/img.png")


async def test_http_image_over_size_limit_raises_validation_error() -> None:
    """HTTP image payloads over the encoded-byte cap should be rejected."""
    capped_loader = ImageLoader(
        cache_size=4,
        http_timeout=30.0,
        url_policy=_permissive_policy(),
        max_image_bytes=len(PNG_BYTES) - 1,
    )
    mock_fetch = _mock_fetch_bytes(content=PNG_BYTES)
    with patch(_FETCH_BYTES_PATH, mock_fetch):
        with pytest.raises(ImageValidationError, match="exceeds maximum size"):
            await capped_loader.load_image("https://example.com/img.png")


async def test_data_url_image_over_size_limit_raises_validation_error() -> None:
    """Decoded data URL payloads over the encoded-byte cap should be rejected."""
    capped_loader = ImageLoader(
        cache_size=4,
        http_timeout=30.0,
        url_policy=_permissive_policy(),
        max_image_bytes=len(PNG_BYTES) - 1,
    )
    data_url = "data:image/png;base64," + base64.b64encode(PNG_BYTES).decode()
    with pytest.raises(ImageValidationError, match="exceeds maximum size"):
        await capped_loader.load_image(data_url)


async def test_batch_preserves_image_validation_error(loader: ImageLoader) -> None:
    """Batch aggregation must retain the typed client validation error."""
    mock_fetch = _mock_fetch_bytes(content=b"not an image")
    with patch(_FETCH_BYTES_PATH, mock_fetch):
        with pytest.raises(ImageValidationError, match="Invalid image data"):
            await loader.load_image_batch(
                [{URL_VARIANT_KEY: "https://example.com/img.png"}]
            )


async def test_batch_preserves_cancellation(loader: ImageLoader) -> None:
    loader.load_image = AsyncMock(  # type: ignore[method-assign]
        side_effect=asyncio.CancelledError
    )
    with pytest.raises(asyncio.CancelledError):
        await loader.load_image_batch(
            [{URL_VARIANT_KEY: "https://example.com/img.png"}]
        )


# --- HTTP error contract ---


async def test_http_timeout_raises_valueerror(loader: ImageLoader) -> None:
    """HTTP timeout should be normalized to ValueError."""
    mock_fetch = _mock_fetch_bytes(side_effect=HttpTimeoutError("timed out"))
    with patch(_FETCH_BYTES_PATH, mock_fetch):
        with pytest.raises(ValueError, match="Timeout loading image"):
            await loader.load_image("https://example.com/img.png")


async def test_http_status_error_propagated(loader: ImageLoader) -> None:
    """HTTP 4xx/5xx should propagate as HttpStatusError."""
    mock_fetch = _mock_fetch_bytes(
        side_effect=HttpStatusError(404, "Not Found", "https://example.com/img.png")
    )
    with patch(_FETCH_BYTES_PATH, mock_fetch):
        with pytest.raises(HttpStatusError) as exc_info:
            await loader.load_image("https://example.com/img.png")
        assert exc_info.value.status == 404


# --- Cache behavior ---


async def test_cache_hit_skips_fetch(loader: ImageLoader) -> None:
    """A cached image should be returned without making an HTTP request."""
    img = Image.new("RGB", (2, 2))
    loader._image_cache["https://example.com/img.png"] = img

    result = await loader.load_image("https://example.com/img.png")
    assert result is img


def _make_svg_bytes() -> bytes:
    return b"<svg xmlns='http://www.w3.org/2000/svg' width='10' height='10'/>"


def _make_image_bytes(image_format: str) -> bytes:
    image = Image.new("RGB", (2, 2), color="red")
    buffer = BytesIO()
    image.save(buffer, format=image_format)
    return buffer.getvalue()


def _make_heif_sequence_bytes() -> bytes:
    """Create a two-frame HEIF sequence without using Pillow's plugin.

    Encoding needs pillow-heif: the runtime dependency is pi-heif, the
    decode-only build of the same project, which ships no encoder. pillow-heif
    is therefore a test-only dependency (requirements.test.txt) and must not be
    imported at module scope -- it is absent from the runtime image, whose
    license policy denies its bundled GPL-2.0 x265 encoder.
    """
    pillow_heif = pytest.importorskip(
        "pillow_heif", reason="HEIF encoding is test-only; see requirements.test.txt"
    )
    primary = Image.new("RGB", (8, 6), color="red")
    secondary = Image.new("RGB", (8, 6), color="blue")
    heif_file = pillow_heif.from_pillow(primary)
    heif_file.add_from_pillow(secondary)

    buffer = BytesIO()
    heif_file.save(buffer, quality=-1)
    encoded = bytearray(buffer.getvalue())
    assert encoded[4:8] == b"ftyp"
    encoded[8:12] = b"msf1"
    return bytes(encoded)


async def test_open_image_sync_decodes_primary_heif_frame_as_rgb() -> None:
    """HEIF sequences preserve the single-image contract by loading frame zero."""
    image = ImageLoader._open_image_sync(BytesIO(_make_heif_sequence_bytes()))

    assert image.size == (8, 6)
    assert image.mode == "RGB"
    red, green, blue = image.getpixel((0, 0))
    assert red > 250
    assert green < 5
    assert blue < 5


async def test_unsupported_format_url_raises_415(loader: ImageLoader) -> None:
    """Fetching a URL that returns an unsupported image format (e.g. SVG) should raise
    HttpStatusError with status 415, not 500."""
    mock_fetch = _mock_fetch_bytes(content=_make_svg_bytes())
    with patch(_FETCH_BYTES_PATH, mock_fetch):
        with pytest.raises(HttpStatusError) as exc_info:
            await loader.load_image("https://example.com/image.svg")
        assert exc_info.value.status == 415


@pytest.mark.parametrize("image_format", ["BMP", "TIFF"])
async def test_identifiable_blocked_format_raises_415(
    loader: ImageLoader, image_format: str
) -> None:
    """Known but unsupported raster formats are 415, unlike corrupt bytes (400)."""
    mock_fetch = _mock_fetch_bytes(content=_make_image_bytes(image_format))
    with patch(_FETCH_BYTES_PATH, mock_fetch):
        with pytest.raises(HttpStatusError) as exc_info:
            await loader.load_image(f"https://example.com/image.{image_format.lower()}")
    assert exc_info.value.status == 415


async def test_unsupported_format_data_url_raises_415(loader: ImageLoader) -> None:
    """A data: URL carrying an SVG payload should raise HttpStatusError 415."""
    svg_b64 = base64.b64encode(_make_svg_bytes()).decode()
    with pytest.raises(HttpStatusError) as exc_info:
        await loader.load_image(f"data:image/svg+xml;base64,{svg_b64}")
    assert exc_info.value.status == 415


async def test_unsupported_format_batch_url_raises_415(loader: ImageLoader) -> None:
    """The batch path must preserve the 415 status instead of collapsing it into a
    generic Exception. This is the path the frontend actually drives, so the status
    has to survive load_image_batch's exception aggregation."""
    mock_fetch = _mock_fetch_bytes(content=_make_svg_bytes())
    with patch(_FETCH_BYTES_PATH, mock_fetch):
        with pytest.raises(HttpStatusError) as exc_info:
            await loader.load_image_batch(
                [{URL_VARIANT_KEY: "https://example.com/image.svg"}]
            )
        assert exc_info.value.status == 415


async def test_unsupported_format_batch_data_url_raises_415(
    loader: ImageLoader,
) -> None:
    """The batch path must preserve the 415 status for data: URLs as well."""
    svg_b64 = base64.b64encode(_make_svg_bytes()).decode()
    with pytest.raises(HttpStatusError) as exc_info:
        await loader.load_image_batch(
            [{URL_VARIANT_KEY: f"data:image/svg+xml;base64,{svg_b64}"}]
        )
    assert exc_info.value.status == 415


# --- SSRF / URL-validation error contract ---


# Cloud metadata IP (link-local) -- the canonical SSRF target; rejected as a
# blocked IP literal before any DNS/connect, so this test makes no network call.
_METADATA_IP_URL = "https://169.254.169.254/latest/meta-data/"


async def test_ssrf_blocked_batch_url_raises_url_validation_error() -> None:
    """An SSRF-blocked URL must escape load_image_batch as UrlValidationError (a
    ValueError -> 4xx), not the bare Exception the aggregator used to raise (500)."""
    strict_loader = ImageLoader(
        cache_size=4, http_timeout=30.0, url_policy=UrlValidationPolicy()
    )
    with pytest.raises(UrlValidationError, match="blocked range"):
        await strict_loader.load_image_batch([{URL_VARIANT_KEY: _METADATA_IP_URL}])


async def test_url_validation_error_from_fetch_preserved(
    loader: ImageLoader,
) -> None:
    """A UrlValidationError raised mid-fetch (redirect revalidation) must survive
    _fetch_and_process's except branch, not be flattened to a plain ValueError."""
    mock_fetch = _mock_fetch_bytes(
        side_effect=UrlValidationError("Too many redirects (max=3)")
    )
    with patch(_FETCH_BYTES_PATH, mock_fetch):
        with pytest.raises(UrlValidationError, match="Too many redirects"):
            await loader.load_image_batch(
                [{URL_VARIANT_KEY: "https://example.com/img.png"}]
            )


async def test_cache_is_lru_not_fifo(loader: ImageLoader) -> None:
    """Accessing a cached entry should protect it from eviction (LRU, not FIFO)."""
    loader._cache_size = 3
    mock_fetch = _mock_fetch_bytes()

    with patch(_FETCH_BYTES_PATH, mock_fetch):
        await loader.load_image("https://example.com/a.png")
        await loader.load_image("https://example.com/b.png")
        await loader.load_image("https://example.com/c.png")
        assert len(loader._image_cache) == 3

        # Touch "a" so it becomes most-recently-used
        await loader.load_image("https://example.com/a.png")

        # Insert "d" — should evict "b" (least recently used), not "a"
        await loader.load_image("https://example.com/d.png")

    assert "https://example.com/a.png" in loader._image_cache
    assert "https://example.com/b.png" not in loader._image_cache
    assert "https://example.com/c.png" in loader._image_cache
    assert "https://example.com/d.png" in loader._image_cache


async def test_load_image_batch_bounds_concurrency() -> None:
    """load_image_batch must never have more than fetch_concurrency images
    fetching+decoding at once.

    Regression test: the batch previously did
    ``asyncio.gather(*[self.load_image(u) for u in urls])`` with no bound, so a
    request with N images materialised N decoded RGB bitmaps in host memory
    simultaneously. A 1000-image request OOM-killed a 200Gi prefill worker
    (cgroup SIGKILL, exit 137). DYN_MM_IMAGE_MAX_BYTES does not help -- it caps
    the ENCODED payload, while the decoded form is what accumulates.
    """
    limit = 4
    total = 40
    in_flight = 0
    peak = 0

    loader = ImageLoader(
        cache_size=1,  # tiny cache so every url is a real fetch, not a cache hit
        url_policy=_permissive_policy(),
        fetch_concurrency=limit,
    )

    async def _tracking_fetch(url: str, *args, **kwargs) -> bytes:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        try:
            await asyncio.sleep(0.005)  # hold the slot so overlap is observable
            return _make_image_bytes("PNG")
        finally:
            in_flight -= 1

    items = [{URL_VARIANT_KEY: f"http://example.com/{i}.png"} for i in range(total)]
    with patch(_FETCH_BYTES_PATH, side_effect=_tracking_fetch):
        results = await loader.load_image_batch(items)

    assert len(results) == total
    assert peak <= limit, f"peak in-flight {peak} exceeded fetch_concurrency {limit}"


async def test_load_image_batch_preserves_order_under_bound() -> None:
    """Bounding concurrency must not reorder results w.r.t. image_mm_items."""
    loader = ImageLoader(
        cache_size=1,
        url_policy=_permissive_policy(),
        fetch_concurrency=2,
    )
    sizes = [(2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7)]

    async def _sized_fetch(url: str, *args, **kwargs) -> bytes:
        idx = int(url.rsplit("/", 1)[-1].split(".")[0])
        buffer = BytesIO()
        Image.new("RGB", sizes[idx], color="red").save(buffer, format="PNG")
        # later items return faster, so an unordered impl would visibly reorder
        await asyncio.sleep((len(sizes) - idx) * 0.001)
        return buffer.getvalue()

    items = [{URL_VARIANT_KEY: f"http://example.com/{i}.png"} for i in range(len(sizes))]
    with patch(_FETCH_BYTES_PATH, side_effect=_sized_fetch):
        results = await loader.load_image_batch(items)

    assert [im.size for im in results] == sizes
