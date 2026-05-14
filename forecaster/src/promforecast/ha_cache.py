"""Shared snapshot cache for HA followers.

When :class:`promforecast.leader.LeaderElector` is active and the operator
configured a Redis backend, the elected leader writes the *rendered* /metrics
payload to Redis after every successful run. Followers read the same payload
on every scrape and serve it directly.

Rendered text — not a structured snapshot — is what crosses the wire because:

* It survives schema changes between leader and follower (rolling upgrades).
* The serialisation overhead is bounded and predictable (one ``render`` call
  per refresh interval, regardless of follower scrape rate).
* Followers don't need pandas/statsforecast/numba imported to render — they
  can run on a slim image if we ever ship one.

The payload is gzip-compressed before writing. Prometheus exposition text is
~80% repetition (label keys, help/type comments) so compression cuts the
Redis bandwidth and memory footprint by roughly 10x. Readers sniff the gzip
magic bytes and transparently decompress, so a mixed-version rollout
(uncompressed-leader / compressed-follower, or vice versa) keeps working.

The Redis client is shared with the query cache when both are configured, so
HA mode incurs a single connection pool to one backend.
"""

from __future__ import annotations

import gzip

import structlog

logger = structlog.get_logger(__name__)


# Gzip's own magic bytes — first two bytes of any gzip stream. We sniff
# them on read so a payload written by an older version (or by a sidecar
# debugger) without compression is still served correctly.
_GZIP_MAGIC = b"\x1f\x8b"


class SnapshotCache:
    """Redis-backed rendered-/metrics cache shared across replicas.

    Two operations only:

    * :meth:`set` — leader writes after each successful run.
    * :meth:`get` — followers read on every scrape.

    Both degrade gracefully: a Redis outage logs once and returns no payload,
    so followers fall back to whatever they have locally (typically nothing,
    since followers don't fit models).
    """

    def __init__(
        self,
        client: object,
        *,
        key: str = "promforecast:snapshot",
        ttl_seconds: int = 300,
        compress: bool = True,
    ) -> None:
        self._client = client
        self._key = key
        self._ttl = max(1, ttl_seconds)
        self._compress = compress

    async def set(self, rendered: bytes) -> None:
        payload = _encode(rendered, compress=self._compress)
        try:
            await self._client.set(self._key, payload, ex=self._ttl)  # type: ignore[attr-defined]
        except Exception as exc:
            logger.warning("ha_snapshot_set_failed", error=str(exc))

    async def get(self) -> bytes | None:
        try:
            raw = await self._client.get(self._key)  # type: ignore[attr-defined]
        except Exception as exc:
            logger.warning("ha_snapshot_get_failed", error=str(exc))
            return None
        if raw is None:
            return None
        # redis-py may return str for non-binary clients; coerce defensively.
        payload = bytes(raw) if isinstance(raw, bytes | bytearray) else str(raw).encode("utf-8")
        return _decode(payload)


def _encode(rendered: bytes, *, compress: bool) -> bytes:
    """Compress when asked, with a level chosen for low CPU and good ratio.

    Level 1 is intentional. Prometheus exposition compresses well at any
    level; the marginal gain from level 6 (the default) is a couple of
    percent, but the CPU cost is ~3x higher. On a leader pushing several
    MB per refresh the CPU savings matter more than the bandwidth.
    """
    if not compress:
        return rendered
    return gzip.compress(rendered, compresslevel=1)


def _decode(payload: bytes) -> bytes:
    """Auto-detect gzip and decompress, otherwise return verbatim."""
    if payload.startswith(_GZIP_MAGIC):
        try:
            return gzip.decompress(payload)
        except Exception as exc:
            logger.warning("ha_snapshot_decode_failed", error=str(exc))
            return b""
    return payload


def build_redis_client(address: str, password: str | None = None) -> object:
    """Construct a ``redis.asyncio.Redis`` from ``host:port`` and optional auth.

    Kept here (rather than in ``main``) so unit tests can monkeypatch the
    factory without touching the entrypoint. Raises ``RuntimeError`` if the
    optional ``redis`` package is not installed — the chart only mounts a
    Redis address when the operator opts into HA mode, so this is the right
    place to surface a missing dependency.
    """
    try:
        import redis.asyncio as redis_asyncio  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "highAvailability requires the 'redis' package; "
            "install with `pip install 'promforecast[ha]'`"
        ) from exc

    host, _, port = address.rpartition(":")
    if not host or not port:
        raise ValueError(f"redis address must be host:port, got {address!r}")
    return redis_asyncio.Redis(
        host=host,
        port=int(port),
        password=password,
        socket_connect_timeout=5,
        socket_timeout=5,
        decode_responses=False,
    )


async def close_redis_client(client: object) -> None:
    """Close a shared Redis client. Tolerates ``None`` and double-close."""
    if client is None:
        return
    aclose = getattr(client, "aclose", None)
    if aclose is None:
        return
    try:
        await aclose()
    except Exception as exc:  # pragma: no cover - cleanup path
        logger.info("redis_close_failed", error=str(exc))


__all__ = [
    "SnapshotCache",
    "build_redis_client",
    "close_redis_client",
]
