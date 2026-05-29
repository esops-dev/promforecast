"""Integration test for the canary-vs-baseline deviation emission.

Uses the real SeasonalNaive model on a small baseline so the runner path
(split → baseline fit → per-canary deviation → exporter family) is exercised
end to end without mocking the fit.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pytest

from promforecast.config import (
    CanaryComparisonConfig,
    Config,
    DatasourceConfig,
    DefaultsConfig,
    GroupConfig,
    QueryConfig,
    SafetyConfig,
    ServerConfig,
)
from promforecast.exporter import Exporter
from promforecast.runner import Runner
from promforecast.source import PromSource, SeriesFrame


class _FakeSource(PromSource):
    def __init__(self, by_query: dict[str, list[SeriesFrame]]) -> None:
        self._by_query = by_query

    async def query_range(
        self, promql: str, start: datetime, end: datetime, step_seconds: int
    ) -> list[SeriesFrame]:
        return self._by_query.get(promql, [])

    async def aclose(self) -> None:
        return None


def _series(labels: dict[str, str], values: list[float]) -> SeriesFrame:
    base = datetime(2026, 1, 1, tzinfo=UTC)
    ts = [base + timedelta(hours=i) for i in range(len(values))]
    return SeriesFrame(labels=labels, timestamps=ts, values=values)


def _canary_config() -> Config:
    return Config(
        datasource=DatasourceConfig(url="http://x"),
        server=ServerConfig(),
        safety=SafetyConfig(max_series_per_query=500, max_total_series=0),
        defaults=DefaultsConfig(
            min_points=10,
            models=["SeasonalNaive"],
            step=timedelta(hours=1),
            horizon=timedelta(hours=3),
            season_length=12,
            confidence_levels=[80],
        ),
        groups=[
            GroupConfig(
                name="g",
                queries=[
                    QueryConfig(
                        id="http_requests",
                        promql="http_requests_total",
                        canary=CanaryComparisonConfig(
                            enabled=True,
                            canary_selector='{version="canary"}',
                            baseline_selector='{version!="canary"}',
                        ),
                    )
                ],
            )
        ],
    )


@pytest.mark.asyncio
async def test_run_group_emits_canary_deviation() -> None:
    n = 40
    baseline_a = _series({"version": "stable", "pod": "a"}, [10.0 + (i % 12) for i in range(n)])
    baseline_b = _series({"version": "stable", "pod": "b"}, [12.0 + (i % 12) for i in range(n)])
    # Canary pod running far hotter than the baseline band predicts.
    canary = _series({"version": "canary", "pod": "c"}, [200.0 + (i % 12) for i in range(n)])
    source = _FakeSource({"http_requests_total": [baseline_a, baseline_b, canary]})
    exporter = Exporter()
    runner = Runner(config=_canary_config(), source=source, exporter=exporter)

    await runner.run_group(_canary_config().groups[0])

    body = exporter.render().decode()
    assert "http_requests_canary_forecast_deviation" in body
    assert 'pod="c"' in body
    # The hot canary should land outside the baseline band.
    assert "http_requests_canary_forecast_deviation_outside_band" in body
    outside = [
        line
        for line in body.splitlines()
        if line.startswith("http_requests_canary_forecast_deviation_outside_band")
        and not line.startswith("#")
    ]
    assert outside, "expected an outside_band sample"
    assert any(math.isclose(float(line.rsplit(" ", 1)[1]), 1.0) for line in outside)


@pytest.mark.asyncio
async def test_canary_skipped_when_population_missing() -> None:
    n = 40
    # Only baseline series — no canary population.
    baseline = _series({"version": "stable", "pod": "a"}, [10.0 + (i % 12) for i in range(n)])
    source = _FakeSource({"http_requests_total": [baseline]})
    exporter = Exporter()
    runner = Runner(config=_canary_config(), source=source, exporter=exporter)

    await runner.run_group(_canary_config().groups[0])

    body = exporter.render().decode()
    assert "http_requests_canary_forecast_deviation" not in body
