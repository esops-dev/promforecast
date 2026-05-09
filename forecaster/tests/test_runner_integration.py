"""Integration tests for Runner orchestration with a fake source.

The fit path (statsforecast) needs heavy deps and is exercised end-to-end via
docker-compose. These tests cover the orchestration layer only: query
dispatch, the cap pipeline, the min_points gate, failure isolation, and the
snapshot update.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from promforecast.config import (
    Config,
    DatasourceConfig,
    DefaultsConfig,
    GroupConfig,
    QueryConfig,
    SafetyConfig,
    ServerConfig,
)
from promforecast.exporter import Exporter, ForecastPoint
from promforecast.runner import Runner
from promforecast.source import PromSource, SeriesFrame

if TYPE_CHECKING:
    pass


class _FakeSource(PromSource):
    """Returns canned frames; never hits the network."""

    def __init__(self, by_query: dict[str, list[SeriesFrame]]) -> None:
        self._by_query = by_query
        self.calls: list[str] = []

    async def query_range(
        self,
        promql: str,
        start: datetime,
        end: datetime,
        step_seconds: int,
    ) -> list[SeriesFrame]:
        self.calls.append(promql)
        return self._by_query.get(promql, [])

    async def aclose(self) -> None:
        return None


def _config(
    queries: list[QueryConfig],
    *,
    min_points: int = 10,
    max_series_per_query: int = 500,
    max_total_series: int = 0,
) -> Config:
    return Config(
        datasource=DatasourceConfig(url="http://x"),
        server=ServerConfig(),
        safety=SafetyConfig(
            max_series_per_query=max_series_per_query,
            max_total_series=max_total_series,
        ),
        defaults=DefaultsConfig(min_points=min_points, models=["SeasonalNaive"]),
        groups=[GroupConfig(name="g", queries=queries)],
    )


def _frame(label_value: str, n_points: int) -> SeriesFrame:
    return SeriesFrame(
        labels={"instance": label_value},
        timestamps=[datetime.now(tz=UTC)] * n_points,
        values=[1.0] * n_points,
    )


@pytest.mark.asyncio
async def test_run_group_records_min_points_drops() -> None:
    cfg = _config(
        [QueryConfig(id="m", promql="up")],
        min_points=100,
    )
    source = _FakeSource({"up": [_frame("a", n_points=5), _frame("b", n_points=5)]})
    exporter = Exporter()
    runner = Runner(config=cfg, source=source, exporter=exporter)

    await runner.run_group(cfg.groups[0])

    body = exporter.render().decode()
    assert 'forecast_series_dropped_total{group="g",reason="min_points"} 2.0' in body
    assert 'forecast_series_count{group="g"} 2.0' in body


@pytest.mark.asyncio
async def test_run_group_isolates_query_errors() -> None:
    class _ErroringSource(_FakeSource):
        async def query_range(
            self, promql: str, start: datetime, end: datetime, step_seconds: int
        ) -> list[SeriesFrame]:
            if promql == "bad":
                raise RuntimeError("simulated TSDB outage")
            return await super().query_range(promql, start, end, step_seconds)

    cfg = _config(
        [
            QueryConfig(id="ok", promql="up"),
            QueryConfig(id="bd", promql="bad"),
        ],
        min_points=100,
    )
    source = _ErroringSource({"up": [_frame("a", n_points=5)]})
    exporter = Exporter()
    runner = Runner(config=cfg, source=source, exporter=exporter)

    await runner.run_group(cfg.groups[0])

    body = exporter.render().decode()
    assert 'forecast_failures_total{group="g",reason="query_error"} 1.0' in body


@pytest.mark.asyncio
async def test_run_group_full_snapshot_with_canned_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _config(
        [QueryConfig(id="m", promql="up")],
        min_points=1,
    )
    source = _FakeSource({"up": [_frame("host-a", n_points=10)]})
    exporter = Exporter()
    runner = Runner(config=cfg, source=source, exporter=exporter)

    async def _canned(self, query, frame, group_name):  # type: ignore[no-untyped-def]
        return [
            ForecastPoint(
                metric="m_forecast",
                labels={**frame.labels, "group": group_name, "model": "Fake"},
                value=42.0,
            )
        ]

    monkeypatch.setattr(Runner, "_fit_series", _canned)
    await runner.run_group(cfg.groups[0])

    body = exporter.render().decode()
    assert "m_forecast{" in body
    assert 'instance="host-a"' in body
    assert 'forecast_series_count{group="g"} 1.0' in body


@pytest.mark.asyncio
async def test_run_group_global_cap_truncates_across_queries() -> None:
    cfg = _config(
        [
            QueryConfig(id="a", promql="qa"),
            QueryConfig(id="b", promql="qb"),
        ],
        min_points=1,
        max_series_per_query=100,
        max_total_series=1,
    )
    source = _FakeSource(
        {
            "qa": [_frame("a", n_points=5)],
            "qb": [_frame("b", n_points=5)],
        }
    )
    exporter = Exporter()
    runner = Runner(config=cfg, source=source, exporter=exporter)

    await runner.run_group(cfg.groups[0])

    body = exporter.render().decode()
    assert 'forecast_series_count{group="g"} 1.0' in body
    assert 'reason="total_cap"' in body
