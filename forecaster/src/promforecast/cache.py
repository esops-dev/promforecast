"""Query result cache for PromQL fetches.

Two backends, one API:

* :class:`MemoryQueryCache` — process-local LRU with per-entry TTL. The default,
  active whenever caching is enabled and no Redis address is supplied.
* :class:`RedisQueryCache` — shared cache across replicas, reusing the same
  Redis connection that HA mode uses. Picked automatically when caching is
  enabled in HA mode (or when the operator points the cache at a Redis
  address explicitly).

Both backends speak :class:`QueryCacheBackend`, so the wiring in
:class:`promforecast.source.PromSource` is identical for both. Cache misses
fall through to the TSDB; cache writes happen *after* a successful fetch so a
failed query never poisons the cache.

The key is computed from the *normalised* PromQL plus the lookback duration
and step. The query window's absolute ``start``/``end`` is **not** part of
the key — every refresh tick moves the window forward by seconds and would
otherwise miss the cache on every call. TTL plus the configured refresh
interval together bound how stale a cached result can be.
"""

from __future__ import annotations

import asyncio
import hashlib
import pickle
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import structlog

if TYPE_CHECKING:
    from .source import SeriesFrame

logger = structlog.get_logger(__name__)


# Pickle is used to round-trip a list[SeriesFrame] across the in-memory and
# Redis backends. The trust boundary here is "promforecast wrote the bytes,
# promforecast reads the bytes" — values never come from user input. A
# version byte is prepended so a forecaster upgrade with an incompatible
# SeriesFrame shape can reject stale entries instead of crashing.
_PICKLE_VERSION = b"\x01"


class QueryCacheBackend(Protocol):
    """Minimal backend contract: bytes in, bytes out, async-friendly."""

    async def get(self, key: str) -> bytes | None: ...

    async def set(self, key: str, value: bytes, ttl_seconds: int) -> None: ...

    async def aclose(self) -> None: ...


@dataclass
class _MemoryEntry:
    value: bytes
    expires_at: float


class MemoryQueryCache:
    """In-process LRU + TTL backend.

    Eviction is LRU on insert; a per-entry expiry is checked on read. The map
    is small (default 1 024 entries; tunable) so the wall-clock overhead of
    the get/set path is dominated by pickling, not by the bookkeeping.
    """

    def __init__(self, max_entries: int = 1024) -> None:
        self._max = max(1, max_entries)
        self._entries: dict[str, _MemoryEntry] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> bytes | None:
        async with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            if entry.expires_at <= time.monotonic():
                # Lazy expiry: don't pay for a sweeper task; the cost of a
                # stale read is one dict lookup we throw away.
                self._entries.pop(key, None)
                return None
            # Touch for LRU order (dict preserves insertion order in 3.7+).
            self._entries.pop(key)
            self._entries[key] = entry
            return entry.value

    async def set(self, key: str, value: bytes, ttl_seconds: int) -> None:
        async with self._lock:
            self._entries.pop(key, None)
            self._entries[key] = _MemoryEntry(
                value=value,
                expires_at=time.monotonic() + max(1, ttl_seconds),
            )
            while len(self._entries) > self._max:
                # popitem(last=False) on a regular dict — pop oldest.
                oldest_key = next(iter(self._entries))
                self._entries.pop(oldest_key, None)

    async def aclose(self) -> None:
        async with self._lock:
            self._entries.clear()

    def size(self) -> int:
        """Diagnostic — number of live entries (including expired-but-not-swept)."""
        return len(self._entries)


class RedisQueryCache:
    """Redis-backed shared cache.

    Uses the ``redis.asyncio`` client. Failures degrade gracefully: a Redis
    outage logs the error and returns ``None``/skips the write so the
    forecaster keeps serving from the TSDB.
    """

    def __init__(self, client: object, key_prefix: str = "promforecast:querycache:") -> None:
        self._client = client
        self._prefix = key_prefix

    async def get(self, key: str) -> bytes | None:
        try:
            raw = await self._client.get(self._prefix + key)  # type: ignore[attr-defined]
        except Exception as exc:
            logger.warning("query_cache_get_failed", error=str(exc))
            return None
        if raw is None:
            return None
        if not isinstance(raw, bytes | bytearray):
            return None
        return bytes(raw)

    async def set(self, key: str, value: bytes, ttl_seconds: int) -> None:
        try:
            await self._client.set(  # type: ignore[attr-defined]
                self._prefix + key,
                value,
                ex=max(1, ttl_seconds),
            )
        except Exception as exc:
            logger.warning("query_cache_set_failed", error=str(exc))

    async def aclose(self) -> None:
        # Don't close the underlying client here: it may be shared with the
        # HA snapshot cache. The owner closes it explicitly at shutdown.
        return None


@dataclass(frozen=True)
class CacheStats:
    """Counters surfaced as ``forecast_query_cache_*`` for observability."""

    hits: int = 0
    misses: int = 0
    errors: int = 0


class QueryCache:
    """High-level wrapper: serialisation, key derivation, and stat accounting.

    Owns counters for the exporter so a flaky cache (or a hot key on Redis)
    is visible without reading logs. The ``hit`` and ``miss`` counters are
    bumped *inside* ``get`` so callers don't have to know about them.
    """

    def __init__(
        self,
        backend: QueryCacheBackend,
        *,
        ttl_seconds: int,
        on_event: object | None = None,
    ) -> None:
        self._backend = backend
        self._ttl = max(1, ttl_seconds)
        self._lock = asyncio.Lock()
        self._hits = 0
        self._misses = 0
        self._errors = 0
        self._on_event = on_event  # Callable[[str], None] - bumps counters

    def cache_key(self, promql: str, lookback_seconds: int, step_seconds: int) -> str:
        """Stable hash over the *normalised* PromQL plus the window shape.

        Two queries that differ only in whitespace must hit the same key
        (otherwise the cache loses value across cosmetic edits in YAML).
        """
        normalised = _normalise_promql(promql)
        digest = hashlib.sha256(
            f"{normalised}\x00{lookback_seconds}\x00{step_seconds}".encode()
        ).hexdigest()
        return digest

    async def get(self, key: str) -> list[SeriesFrame] | None:
        try:
            raw = await self._backend.get(key)
        except Exception as exc:
            logger.warning("query_cache_get_unexpected", error=str(exc))
            self._bump("error")
            return None
        if raw is None:
            self._bump("miss")
            return None
        try:
            frames = _decode_frames(raw)
        except Exception as exc:
            # Corrupt or version-mismatched payload — treat as a miss; the
            # caller will refetch and overwrite.
            logger.warning("query_cache_decode_failed", error=str(exc))
            self._bump("error")
            return None
        self._bump("hit")
        return frames

    async def set(self, key: str, frames: list[SeriesFrame]) -> None:
        try:
            payload = _encode_frames(frames)
        except Exception as exc:
            logger.warning("query_cache_encode_failed", error=str(exc))
            self._bump("error")
            return
        try:
            await self._backend.set(key, payload, self._ttl)
        except Exception as exc:
            logger.warning("query_cache_set_unexpected", error=str(exc))
            self._bump("error")

    async def aclose(self) -> None:
        await self._backend.aclose()

    def stats(self) -> CacheStats:
        return CacheStats(hits=self._hits, misses=self._misses, errors=self._errors)

    def _bump(self, kind: str) -> None:
        # Cheap atomic-ish accounting; GIL-protected here because all callers
        # are single-event-loop. We don't take the lock — counters are
        # advisory diagnostics, not correctness-critical.
        if kind == "hit":
            self._hits += 1
        elif kind == "miss":
            self._misses += 1
        else:
            self._errors += 1
        if self._on_event is not None:
            self._on_event(kind)  # type: ignore[operator]


_WHITESPACE = re.compile(r"\s+")


def _normalise_promql(promql: str) -> str:
    """Whitespace-collapse + strip so cosmetically-different PromQL still hits.

    We deliberately don't try to parse PromQL — that's a much bigger
    contract than the cache needs. Collapsing whitespace handles the
    realistic cases (line-folded YAML literals, trailing newlines from
    block scalars) without touching the semantics.
    """
    return _WHITESPACE.sub(" ", promql).strip()


def _encode_frames(frames: list[SeriesFrame]) -> bytes:
    return _PICKLE_VERSION + pickle.dumps(frames, protocol=pickle.HIGHEST_PROTOCOL)


def _decode_frames(raw: bytes) -> list[SeriesFrame]:
    if not raw or raw[:1] != _PICKLE_VERSION:
        raise ValueError("unknown cache payload version")
    obj = pickle.loads(raw[1:])
    if not isinstance(obj, list):
        raise ValueError("cache payload is not a list of SeriesFrame")
    return obj


__all__ = [
    "CacheStats",
    "MemoryQueryCache",
    "QueryCache",
    "QueryCacheBackend",
    "RedisQueryCache",
]
