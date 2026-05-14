"""Tests for the PromQL query cache.

Covers the in-memory backend behaviour (TTL, LRU eviction, error
handling) plus the high-level :class:`QueryCache` serialisation +
counter wiring. The Redis backend is exercised by integration tests in
docker-compose; here we use a tiny in-memory fake that satisfies the
:class:`QueryCacheBackend` contract.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest

from promforecast.cache import (
    MemoryQueryCache,
    QueryCache,
    _normalise_promql,
)
from promforecast.source import PromSource, SeriesFrame


def _frame(value: float) -> SeriesFrame:
    return SeriesFrame(
        labels={"job": "n"},
        timestamps=[datetime.now(tz=UTC)],
        values=[value],
    )


@pytest.mark.asyncio
async def test_memory_cache_basic_set_get() -> None:
    cache = MemoryQueryCache(max_entries=4)
    await cache.set("k1", b"payload", ttl_seconds=60)
    assert await cache.get("k1") == b"payload"


@pytest.mark.asyncio
async def test_memory_cache_evicts_oldest() -> None:
    """An overflowed LRU drops the oldest insertion, not the most recent."""
    cache = MemoryQueryCache(max_entries=2)
    await cache.set("a", b"1", ttl_seconds=60)
    await cache.set("b", b"2", ttl_seconds=60)
    await cache.set("c", b"3", ttl_seconds=60)
    assert await cache.get("a") is None
    assert await cache.get("b") == b"2"
    assert await cache.get("c") == b"3"


@pytest.mark.asyncio
async def test_memory_cache_expires_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    """A read past the TTL drops the entry and returns ``None``."""
    cache = MemoryQueryCache(max_entries=2)

    fake_clock = {"now": 100.0}

    def _monotonic() -> float:
        return fake_clock["now"]

    monkeypatch.setattr("promforecast.cache.time.monotonic", _monotonic)
    await cache.set("a", b"1", ttl_seconds=5)
    fake_clock["now"] = 102.0
    assert await cache.get("a") == b"1"
    fake_clock["now"] = 110.0
    assert await cache.get("a") is None


@pytest.mark.asyncio
async def test_query_cache_round_trips_frames() -> None:
    """Frames written through QueryCache decode identically on read."""
    cache = QueryCache(MemoryQueryCache(), ttl_seconds=60)
    frames = [_frame(1.0), _frame(2.0)]
    key = cache.cache_key("up", lookback_seconds=300, step_seconds=15)
    await cache.set(key, frames)
    got = await cache.get(key)
    assert got is not None
    assert [f.values for f in got] == [[1.0], [2.0]]


@pytest.mark.asyncio
async def test_query_cache_key_ignores_whitespace_diff() -> None:
    cache = QueryCache(MemoryQueryCache(), ttl_seconds=60)
    k1 = cache.cache_key("up", 60, 15)
    k2 = cache.cache_key("  up  ", 60, 15)
    k3 = cache.cache_key("\tup\n", 60, 15)
    assert k1 == k2 == k3


@pytest.mark.asyncio
async def test_query_cache_emits_counter_events() -> None:
    events: list[str] = []
    cache = QueryCache(MemoryQueryCache(), ttl_seconds=60, on_event=events.append)
    key = cache.cache_key("up", 60, 15)
    assert await cache.get(key) is None  # miss
    await cache.set(key, [_frame(1.0)])
    assert await cache.get(key) is not None  # hit
    assert events == ["miss", "hit"]


@pytest.mark.asyncio
async def test_query_cache_handles_backend_decode_error() -> None:
    """Corrupt payload counts as an error but doesn't surface to the caller."""

    class _BrokenBackend:
        async def get(self, _: str) -> bytes:
            return b"not a real pickle payload"

        async def set(self, *_: Any, **__: Any) -> None:
            return None

        async def aclose(self) -> None:
            return None

    events: list[str] = []
    cache = QueryCache(_BrokenBackend(), ttl_seconds=60, on_event=events.append)
    assert await cache.get("anything") is None
    assert events == ["error"]


def test_normalise_promql_collapses_whitespace() -> None:
    assert _normalise_promql("foo\n  bar  baz") == "foo bar baz"
    assert _normalise_promql('up{job="x"}') == 'up{job="x"}'


@pytest.mark.asyncio
async def test_source_uses_cache_on_second_call() -> None:
    """A repeated query_range with the same window goes through the cache."""
    cache = QueryCache(MemoryQueryCache(), ttl_seconds=60)
    calls = {"n": 0}

    def _handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(
            200,
            json={
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
            },
        )

    src = PromSource(url="http://vm", timeout_seconds=5.0, cache=cache)
    src._client = httpx.AsyncClient(transport=httpx.MockTransport(_handler), timeout=5.0)
    try:
        start = datetime(2026, 1, 1, tzinfo=UTC)
        # Different absolute timestamps, same lookback window -> same key.
        await src.query_range("up", start, datetime(2026, 1, 1, 0, 5, tzinfo=UTC), step_seconds=15)
        await src.query_range(
            "up",
            datetime(2026, 1, 1, 12, tzinfo=UTC),
            datetime(2026, 1, 1, 12, 5, tzinfo=UTC),
            step_seconds=15,
        )
    finally:
        await src.aclose()
    assert calls["n"] == 1
    stats = cache.stats()
    assert stats.hits == 1
    assert stats.misses == 1


def test_memory_cache_size_reports_live_entries() -> None:
    cache = MemoryQueryCache(max_entries=8)
    assert cache.size() == 0
    asyncio.run(cache.set("k", b"v", ttl_seconds=60))
    assert cache.size() == 1
