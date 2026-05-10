"""Tests for the ``remote_write`` push sink."""

from __future__ import annotations

import httpx
import pytest

from promforecast.sink import (
    ForecastCurvePoint,
    RemoteWriteSink,
    SinkError,
    encode_prometheus_text,
)


def _point(value: float = 1.0, timestamp_ms: int = 1_700_000_000_000) -> ForecastCurvePoint:
    return ForecastCurvePoint(
        metric="m_forecast",
        labels={"group": "g", "model": "AutoARIMA"},
        value=value,
        timestamp_ms=timestamp_ms,
    )


def test_encode_emits_prometheus_text_with_millisecond_timestamps() -> None:
    body = encode_prometheus_text([_point(value=1.5, timestamp_ms=1_700_000_000_000)])
    assert body == 'm_forecast{group="g",model="AutoARIMA"} 1.5 1700000000000'


def test_encode_sorts_labels_for_determinism() -> None:
    p = ForecastCurvePoint(
        metric="x_forecast",
        labels={"z": "1", "a": "2", "m": "3"},
        value=42.0,
        timestamp_ms=1,
    )
    assert encode_prometheus_text([p]) == 'x_forecast{a="2",m="3",z="1"} 42.0 1'


def test_encode_escapes_label_values() -> None:
    p = ForecastCurvePoint(
        metric="x_forecast",
        labels={"weird": 'has"quotes\\and\nnewline'},
        value=1.0,
        timestamp_ms=1,
    )
    out = encode_prometheus_text([p])
    assert 'weird="has\\"quotes\\\\and\\nnewline"' in out


def test_encode_empty_input_returns_empty_string() -> None:
    assert encode_prometheus_text([]) == ""


def test_remote_write_sink_requires_url() -> None:
    with pytest.raises(ValueError, match="non-empty url"):
        RemoteWriteSink(url="", timeout_seconds=1.0)


@pytest.mark.asyncio
async def test_remote_write_sink_posts_text_body_to_url() -> None:
    posted: list[tuple[str, bytes, dict[str, str]]] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        posted.append((str(request.url), request.content, dict(request.headers)))
        return httpx.Response(204)

    transport = httpx.MockTransport(_handler)
    client = httpx.AsyncClient(transport=transport)
    sink = RemoteWriteSink(
        url="http://vm:8428/api/v1/import/prometheus",
        timeout_seconds=5.0,
        client=client,
    )
    await sink.write([_point()])
    await sink.aclose()
    await client.aclose()

    assert len(posted) == 1
    url, body, headers = posted[0]
    assert url == "http://vm:8428/api/v1/import/prometheus"
    assert b"m_forecast{" in body
    assert headers["content-type"].startswith("text/plain")


@pytest.mark.asyncio
async def test_remote_write_sink_raises_sink_error_on_http_error() -> None:
    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="overloaded")

    transport = httpx.MockTransport(_handler)
    client = httpx.AsyncClient(transport=transport)
    sink = RemoteWriteSink(url="http://vm:8428/x", timeout_seconds=5.0, client=client)
    try:
        with pytest.raises(SinkError) as excinfo:
            await sink.write([_point()])
        assert excinfo.value.reason == "http_error"
        assert "503" in str(excinfo.value)
    finally:
        await sink.aclose()
        await client.aclose()


@pytest.mark.asyncio
async def test_remote_write_sink_raises_sink_error_on_timeout() -> None:
    def _handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("simulated")

    transport = httpx.MockTransport(_handler)
    client = httpx.AsyncClient(transport=transport)
    sink = RemoteWriteSink(url="http://vm:8428/x", timeout_seconds=5.0, client=client)
    try:
        with pytest.raises(SinkError) as excinfo:
            await sink.write([_point()])
        assert excinfo.value.reason == "timeout"
    finally:
        await sink.aclose()
        await client.aclose()


@pytest.mark.asyncio
async def test_remote_write_sink_skips_post_on_empty_input() -> None:
    """Avoid generating a 411 from a zero-byte POST when nothing to send."""
    posted: list[httpx.Request] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        posted.append(request)
        return httpx.Response(204)

    transport = httpx.MockTransport(_handler)
    client = httpx.AsyncClient(transport=transport)
    sink = RemoteWriteSink(url="http://vm:8428/x", timeout_seconds=5.0, client=client)
    try:
        await sink.write([])
    finally:
        await sink.aclose()
        await client.aclose()
    assert posted == []


@pytest.mark.asyncio
async def test_remote_write_sink_chunks_large_pushes_into_multiple_posts() -> None:
    """Bodies are bounded by ``chunk_size`` so a 10k-series fan-out doesn't
    generate one giant request that the TSDB has to buffer in memory."""
    posted_bodies: list[bytes] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        posted_bodies.append(request.content)
        return httpx.Response(204)

    transport = httpx.MockTransport(_handler)
    client = httpx.AsyncClient(transport=transport)
    sink = RemoteWriteSink(
        url="http://vm:8428/x",
        timeout_seconds=5.0,
        client=client,
        chunk_size=3,
    )
    try:
        await sink.write([_point(value=float(i), timestamp_ms=i + 1) for i in range(10)])
    finally:
        await sink.aclose()
        await client.aclose()
    # 10 points / chunk_size=3 → 4 POSTs (3 + 3 + 3 + 1).
    assert len(posted_bodies) == 4
    line_counts = [body.count(b"\n") + 1 for body in posted_bodies]
    assert line_counts == [3, 3, 3, 1]


@pytest.mark.asyncio
async def test_remote_write_sink_aborts_remaining_chunks_on_failure() -> None:
    """A failure on one chunk raises immediately; the next refresh
    re-sends the full curve so partial-success is fine."""
    posted = 0

    def _handler(request: httpx.Request) -> httpx.Response:
        nonlocal posted
        posted += 1
        if posted == 2:
            return httpx.Response(503, text="overloaded")
        return httpx.Response(204)

    transport = httpx.MockTransport(_handler)
    client = httpx.AsyncClient(transport=transport)
    sink = RemoteWriteSink(
        url="http://vm:8428/x",
        timeout_seconds=5.0,
        client=client,
        chunk_size=2,
    )
    try:
        with pytest.raises(SinkError):
            await sink.write([_point(value=float(i), timestamp_ms=i + 1) for i in range(10)])
    finally:
        await sink.aclose()
        await client.aclose()
    # 1st chunk succeeded, 2nd raised — the remaining 3 chunks aren't sent.
    assert posted == 2


def test_remote_write_sink_rejects_non_positive_chunk_size() -> None:
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        RemoteWriteSink(url="http://x", timeout_seconds=1.0, chunk_size=0)
