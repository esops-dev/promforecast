"""PromQL query client.

Thin wrapper around the long-term TSDB (VictoriaMetrics by default).
Production Prometheus is never queried directly; all reads go through the
long-term TSDB.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class SeriesFrame:
    """A single time series returned by the TSDB."""

    labels: dict[str, str]
    timestamps: list[datetime]
    values: list[float]


class PromSource:
    """PromQL range query client with retry, timeout, and auth."""

    def __init__(self, url: str, timeout_seconds: float, auth: object | None = None) -> None:
        self._url = url
        self._timeout_seconds = timeout_seconds
        self._auth = auth

    async def query_range(
        self,
        promql: str,
        start: datetime,
        end: datetime,
        step_seconds: int,
    ) -> list[SeriesFrame]:
        """Run a PromQL range query and return one frame per series."""
        raise NotImplementedError
