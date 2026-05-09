"""PromQL query client.

Thin wrapper around the long-term TSDB (VictoriaMetrics by default).
Production Prometheus is never queried directly; all reads go through the
long-term TSDB.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import httpx
import structlog
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
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
    """PromQL range query client with retry, timeout, and auth.

    A per-source semaphore caps in-flight requests so a forecaster with many
    groups can't accidentally hammer the TSDB. ``max_concurrency`` defaults
    to 8, which is comfortable for a single-replica forecaster against a
    single-node VictoriaMetrics; tune if you observe queueing.
    """

    def __init__(
        self,
        url: str,
        timeout_seconds: float,
        auth: object | None = None,
        max_concurrency: int = 8,
    ) -> None:
        self._url = url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._auth = auth
        self._client = httpx.AsyncClient(timeout=timeout_seconds)
        self._semaphore = asyncio.Semaphore(max_concurrency)

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
            retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
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
