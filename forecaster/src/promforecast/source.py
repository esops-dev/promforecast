"""PromQL query client.

Thin wrapper around the long-term TSDB (VictoriaMetrics by default).
Production Prometheus is never queried directly; all reads go through the
long-term TSDB.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import httpx
import structlog
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

logger = structlog.get_logger(__name__)


class QueryError(RuntimeError):
    """Raised when the TSDB returns an error or unparseable response."""


@dataclass(frozen=True)
class SeriesFrame:
    """A single time series returned by the TSDB."""

    labels: dict[str, str]
    timestamps: list[datetime]
    values: list[float]


class PromSource:
    """PromQL range query client with retry and timeout.

    A per-source semaphore caps in-flight requests so a forecaster with many
    groups can't accidentally hammer the TSDB. ``max_concurrency`` defaults
    to 8, which is comfortable for a single-replica forecaster against a
    single-node VictoriaMetrics; tune if you observe queueing.

    The httpx client's connection pool is sized to match: keepalive
    connections track ``max_concurrency`` so the semaphore is the only
    bottleneck. Without explicit limits, httpx defaults to 100 max /
    20 keepalive, which silently masked the semaphore's intent.

    ``on_retry`` is invoked once per retry attempt with a short reason
    token (``transport`` | ``http_5xx``). The runner wires this to the
    exporter's source-retry counter so a flaky TSDB papered over by
    retries is observable before it tips into outright failure.
    """

    def __init__(
        self,
        url: str,
        timeout_seconds: float,
        max_concurrency: int = 8,
        max_keepalive_connections: int | None = None,
        on_retry: Callable[[str], None] | None = None,
    ) -> None:
        self._url = url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        keepalive = (
            max_keepalive_connections
            if max_keepalive_connections is not None
            else max_concurrency
        )
        # Stash the resolved limits as a public attribute so callers
        # (and tests) can introspect what the client was actually
        # configured with — httpx hides the values inside its transport
        # pool's private attributes, which is fragile to depend on.
        self.limits = httpx.Limits(
            max_connections=max_concurrency,
            max_keepalive_connections=keepalive,
        )
        self._client = httpx.AsyncClient(timeout=timeout_seconds, limits=self.limits)
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._on_retry = on_retry

    async def aclose(self) -> None:
        await self._client.aclose()

    async def query_range(
        self,
        promql: str,
        start: datetime,
        end: datetime,
        step_seconds: int,
    ) -> list[SeriesFrame]:
        """Run a PromQL range query and return one frame per series."""
        params = {
            "query": promql,
            "start": f"{start.timestamp():.3f}",
            "end": f"{end.timestamp():.3f}",
            "step": f"{step_seconds}s",
        }
        endpoint = f"{self._url}/api/v1/query_range"
        async with self._semaphore:
            payload = await self._request(endpoint, params)
        return _parse_matrix(payload)

    async def _request(self, endpoint: str, params: dict[str, str]) -> dict[str, Any]:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=0.5, max=4),
            retry=retry_if_exception(_is_retryable),
            before_sleep=self._record_retry,
            reraise=True,
        ):
            with attempt:
                response = await self._client.get(endpoint, params=params)
                response.raise_for_status()
                payload: dict[str, Any] = response.json()
                if payload.get("status") != "success":
                    raise QueryError(
                        f"query failed: {payload.get('errorType')}: {payload.get('error')}"
                    )
                return payload
        raise QueryError("retries exhausted")  # pragma: no cover

    def _record_retry(self, retry_state: RetryCallState) -> None:
        """tenacity ``before_sleep`` hook: bump the retry counter.

        ``before_sleep`` fires after a retryable failure but before the
        wait — so it counts every attempt that *will* be retried. The
        terminal failure (after attempts exhausted) does not fire
        ``before_sleep`` again, which is the right semantic: the
        terminal one is already counted by ``forecast_failures_total``.
        """
        if self._on_retry is None or retry_state.outcome is None:
            return
        exc = retry_state.outcome.exception()
        if isinstance(exc, httpx.HTTPStatusError):
            self._on_retry("http_5xx")
        elif isinstance(exc, httpx.TransportError):
            self._on_retry("transport")


def _is_retryable(exc: BaseException) -> bool:
    """Retry transport flakes and 5xx; let 4xx (bad query, auth) fail fast.

    Retrying a 400/404 just multiplies the latency operators see when their
    PromQL is wrong, and burns redundant TSDB CPU on a deterministic
    failure. 5xx responses (TSDB overloaded, partial outage) are the cases
    where backoff actually helps.
    """
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500  # noqa: PLR2004
    return False


def _parse_matrix(payload: dict[str, Any]) -> list[SeriesFrame]:
    data = payload.get("data") or {}
    if data.get("resultType") != "matrix":
        raise QueryError(f"unexpected resultType: {data.get('resultType')!r}")
    frames: list[SeriesFrame] = []
    for series in data.get("result", []):
        labels = dict(series.get("metric", {}))
        timestamps: list[datetime] = []
        values: list[float] = []
        for ts, val in series.get("values", []):
            try:
                v = float(val)
            except (TypeError, ValueError):
                continue
            if math.isnan(v):  # VM/Prom can return NaN as "NaN"
                continue
            timestamps.append(datetime.fromtimestamp(float(ts), tz=UTC))
            values.append(v)
        if values:
            frames.append(SeriesFrame(labels=labels, timestamps=timestamps, values=values))
    return frames
