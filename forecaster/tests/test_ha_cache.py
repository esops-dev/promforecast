"""Tests for the HA snapshot cache + redis-client builder.

The cache and Redis-client wiring are mostly thin wrappers around the
``redis.asyncio`` API. The interesting behaviour — graceful degradation
when Redis is unavailable, bytes-vs-str coercion on read — is covered
here with a small in-memory fake.
"""

from __future__ import annotations

import pytest

from promforecast.ha_cache import SnapshotCache, build_redis_client


class _FakeRedis:
    """Just enough of ``redis.asyncio.Redis`` for SnapshotCache to round-trip."""

    def __init__(self) -> None:
        self.store: dict[str, bytes] = {}
        self.fail_on_get = False
        self.fail_on_set = False

    async def get(self, key: str) -> bytes | None:
        if self.fail_on_get:
            raise RuntimeError("simulated redis failure")
        return self.store.get(key)

    async def set(self, key: str, value: bytes, ex: int | None = None) -> None:
        if self.fail_on_set:
            raise RuntimeError("simulated redis failure")
        self.store[key] = value


@pytest.mark.asyncio
async def test_snapshot_cache_round_trip() -> None:
    fake = _FakeRedis()
    cache = SnapshotCache(fake, key="snap", ttl_seconds=60)
    await cache.set(b"hello")
    assert await cache.get() == b"hello"


@pytest.mark.asyncio
async def test_snapshot_cache_compresses_payload_by_default() -> None:
    """The bytes stored in Redis are smaller than the rendered payload."""
    fake = _FakeRedis()
    cache = SnapshotCache(fake, ttl_seconds=60)
    rendered = b'# HELP foo a metric\n# TYPE foo gauge\nfoo{label="x"} 1\n' * 200
    await cache.set(rendered)
    stored = next(iter(fake.store.values()))
    assert stored.startswith(b"\x1f\x8b"), "expected gzip magic header"
    assert len(stored) < len(rendered) // 3
    assert await cache.get() == rendered


@pytest.mark.asyncio
async def test_snapshot_cache_reads_legacy_uncompressed_payload() -> None:
    """A payload written by an older version (no compression) still decodes."""
    fake = _FakeRedis()
    # Pre-populate the store with raw bytes — simulating an old leader.
    fake.store["promforecast:snapshot"] = b"plain text payload"
    cache = SnapshotCache(fake, ttl_seconds=60)
    assert await cache.get() == b"plain text payload"


@pytest.mark.asyncio
async def test_snapshot_cache_returns_none_on_get_failure() -> None:
    fake = _FakeRedis()
    fake.fail_on_get = True
    cache = SnapshotCache(fake, ttl_seconds=60)
    assert await cache.get() is None  # error is swallowed


@pytest.mark.asyncio
async def test_snapshot_cache_swallows_set_failure() -> None:
    """Set errors are logged but never raised — the leader must not crash."""
    fake = _FakeRedis()
    fake.fail_on_set = True
    cache = SnapshotCache(fake, ttl_seconds=60)
    await cache.set(b"x")  # must not raise


def test_build_redis_client_rejects_bad_address() -> None:
    with pytest.raises(ValueError):
        build_redis_client("not-a-host-port")
