"""Tests for the PromQL response parser and the HTTP retry path."""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest

from promforecast.source import PromSource, QueryError, _parse_matrix


def test_parse_matrix_basic() -> None:
    payload = {
        "status": "success",
        "data": {
            "resultType": "matrix",
            "result": [
                {
                    "metric": {"__name__": "up", "job": "node"},
                    "values": [[1700000000, "1"], [1700000060, "0"]],
                }
            ],
        },
    }
    frames = _parse_matrix(payload)
    assert len(frames) == 1
    assert frames[0].labels == {"__name__": "up", "job": "node"}
    assert frames[0].values == [1.0, 0.0]
    assert len(frames[0].timestamps) == 2


def test_parse_matrix_drops_nan() -> None:
    payload = {
        "data": {
            "resultType": "matrix",
            "result": [
                {
                    "metric": {"job": "n"},
                    "values": [[1700000000, "1"], [1700000060, "NaN"], [1700000120, "2"]],
                }
            ],
        },
    }
    frames = _parse_matrix(payload)
    assert frames[0].values == [1.0, 2.0]


def test_parse_matrix_skips_empty_series() -> None:
    payload = {
        "data": {
            "resultType": "matrix",
            "result": [
                {"metric": {"job": "empty"}, "values": []},
                {"metric": {"job": "ok"}, "values": [[1700000000, "5"]]},
            ],
        },
    }
    frames = _parse_matrix(payload)
    assert len(frames) == 1
    assert frames[0].labels == {"job": "ok"}


def test_parse_matrix_rejects_wrong_result_type() -> None:
    payload = {"data": {"resultType": "vector", "result": []}}
    with pytest.raises(QueryError):
        _parse_matrix(payload)


# -- HTTP retry path -------------------------------------------------------


def _success_payload() -> dict[str, object]:
    return {
        "status": "success",
        "data": {
            "resultType": "matrix",
            "result": [
                {
                    "metric": {"job": "n"},
                    "values": [[1700000000, "1"]],
                }
            ],
        },
    }


def _build_source(transport: httpx.MockTransport, on_retry: object | None = None) -> PromSource:
    """Construct a PromSource whose AsyncClient is swapped for the mock.

    The constructor builds its own httpx.AsyncClient against the live
    transport — for these tests we replace it with a MockTransport-backed
    client so retries don't actually hit the network.
    """
    src = PromSource(
        url="http://vm",
        timeout_seconds=5.0,
        on_retry=on_retry,  # type: ignore[arg-type]
    )
    src._client = httpx.AsyncClient(transport=transport, timeout=5.0)
    return src


@pytest.mark.asyncio
async def test_source_retries_5xx_and_records_retry() -> None:
    """A retried 5xx must fire on_retry with reason ``http_5xx``."""
    calls = 0
    retries: list[str] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(503, text="overloaded")
        return httpx.Response(200, json=_success_payload())

    src = _build_source(httpx.MockTransport(_handler), on_retry=retries.append)
    try:
        frames = await src.query_range("up", datetime.now(tz=UTC), datetime.now(tz=UTC), 60)
    finally:
        await src.aclose()
    assert len(frames) == 1
    assert calls == 2
    assert retries == ["http_5xx"]


@pytest.mark.asyncio
async def test_source_retries_transport_error_and_records_retry() -> None:
    """A retried transport error must fire on_retry with reason ``transport``."""
    calls = 0
    retries: list[str] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ConnectError("simulated", request=request)
        return httpx.Response(200, json=_success_payload())

    src = _build_source(httpx.MockTransport(_handler), on_retry=retries.append)
    try:
        await src.query_range("up", datetime.now(tz=UTC), datetime.now(tz=UTC), 60)
    finally:
        await src.aclose()
    assert calls == 2
    assert retries == ["transport"]


@pytest.mark.asyncio
async def test_source_does_not_retry_4xx() -> None:
    """4xx fails fast: tenacity must not retry, on_retry must not fire."""
    calls = 0
    retries: list[str] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(400, text="bad query")

    src = _build_source(httpx.MockTransport(_handler), on_retry=retries.append)
    try:
        with pytest.raises(httpx.HTTPStatusError):
            await src.query_range("up", datetime.now(tz=UTC), datetime.now(tz=UTC), 60)
    finally:
        await src.aclose()
    assert calls == 1
    assert retries == []


def test_source_applies_pool_limits_to_default_client() -> None:
    """Pool config must reach the underlying httpx client.

    Regression check for the constructor wiring — without explicit
    limits, httpx defaults to 100/20 and the operator-facing knob
    would silently do nothing.
    """
    src = PromSource(
        url="http://vm",
        timeout_seconds=1.0,
        max_concurrency=13,
        max_keepalive_connections=5,
    )
    assert src.limits.max_connections == 13
    assert src.limits.max_keepalive_connections == 5


def test_source_pool_keepalive_defaults_to_concurrency() -> None:
    """When no explicit keepalive is given, it tracks max_concurrency.

    This keeps the semaphore as the only bottleneck — without the
    default coupling, the pool's keepalive cap could shadow the
    semaphore intent.
    """
    src = PromSource(url="http://vm", timeout_seconds=1.0, max_concurrency=6)
    assert src.limits.max_keepalive_connections == 6


@pytest.mark.asyncio
async def test_source_no_on_retry_callback_is_safe() -> None:
    """A retried request without an on_retry callback must not error out."""
    calls = 0

    def _handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(503)
        return httpx.Response(200, json=_success_payload())

    src = _build_source(httpx.MockTransport(_handler))
    try:
        await src.query_range("up", datetime.now(tz=UTC), datetime.now(tz=UTC), 60)
    finally:
        await src.aclose()
    assert calls == 2
