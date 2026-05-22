"""Runner-side integration tests for cold-start / count-aware / intermittent.

Cold-start, count-aware safeguards, and the intermittent profile all
plumb through ``Runner._do_run_query`` / ``Runner._fit_series``. We
test them with the same canned-source pattern the rest of the runner
suite uses so the heavy ``statsforecast`` deps stay out of the path.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

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
from promforecast.exporter import Exporter
from promforecast.source import PromSource, SeriesFrame


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


def _sibling_frame(label: str, values: list[float]) -> SeriesFrame:
    start = datetime(2026, 5, 21, tzinfo=UTC)
    timestamps = [start + timedelta(minutes=5 * i) for i in range(len(values))]
    return SeriesFrame(labels={"instance": label}, timestamps=timestamps, values=values)


def _config(queries: list[QueryConfig], *, min_points: int = 100) -> Config:
    return Config(
        datasource=DatasourceConfig(url="http://x"),
        server=ServerConfig(),
        safety=SafetyConfig(max_series_per_query=500, max_total_series=0),
        defaults=DefaultsConfig(
            min_points=min_points,
            lookback=timedelta(hours=2),
            horizon=timedelta(minutes=20),
            step=timedelta(minutes=5),
            season_length=4,
            confidence_levels=[80],
            models=["SeasonalNaive"],
        ),
        groups=[GroupConfig(name="g", queries=queries)],
    )


@pytest.mark.asyncio
async def test_cold_start_emits_borrowed_forecast_when_target_short() -> None:
    """A target below ``min_points`` plus a borrow query emits a ColdStart fit.

    Without the borrow query the runner would drop the series with
    ``reason="min_points"``; with the cold-start config it should
    instead emit a forecast carrying ``model="ColdStart"`` and
    ``cold_start="true"`` and bump ``forecast_cold_start_siblings``.
    """
    target_frame = _sibling_frame("new", [10.0, 12.0, 11.0])  # below min_points
    siblings = [
        _sibling_frame("peer-a", [1.0, 2.0, 3.0, 2.0] * 6),
        _sibling_frame("peer-b", [2.0, 4.0, 6.0, 4.0] * 6),
        _sibling_frame("peer-c", [3.0, 6.0, 9.0, 6.0] * 6),
    ]
    cfg = _config(
        [
            QueryConfig(
                id="m",
                promql="target",
                cold_start={
                    "borrow_from": {
                        "promql": "siblings",
                        "method": "median_normalized",
                        "min_siblings": 3,
                    }
                },
            )
        ]
    )
    source = _FakeSource({"target": [target_frame], "siblings": siblings})
    exporter = Exporter()

    from promforecast.runner import Runner  # noqa: PLC0415

    runner = Runner(config=cfg, source=source, exporter=exporter)
    await runner.run_group(cfg.groups[0])

    body = exporter.render().decode()
    assert 'cold_start="true"' in body
    assert 'model="ColdStart"' in body
    assert 'forecast_cold_start_siblings{group="g",query="m"} 3.0' in body


@pytest.mark.asyncio
async def test_cold_start_falls_through_when_siblings_thin() -> None:
    """Below ``min_siblings``, cold-start falls back to the standard drop."""
    target_frame = _sibling_frame("new", [10.0, 12.0])
    siblings = [_sibling_frame("peer-a", [1.0, 2.0, 3.0, 2.0] * 6)]
    cfg = _config(
        [
            QueryConfig(
                id="m",
                promql="target",
                cold_start={
                    "borrow_from": {
                        "promql": "siblings",
                        "method": "median_normalized",
                        "min_siblings": 3,
                    }
                },
            )
        ]
    )
    source = _FakeSource({"target": [target_frame], "siblings": siblings})
    exporter = Exporter()

    from promforecast.runner import Runner  # noqa: PLC0415

    runner = Runner(config=cfg, source=source, exporter=exporter)
    await runner.run_group(cfg.groups[0])

    body = exporter.render().decode()
    assert 'forecast_series_dropped_total{group="g",reason="min_points"} 1.0' in body
    # No cold-start emission with thin siblings.
    assert 'model="ColdStart"' not in body


@pytest.mark.asyncio
async def test_cold_start_caps_quality_score() -> None:
    """Cold-start quality is capped at ``max_quality_score`` (default 0.5)."""
    target_frame = _sibling_frame("new", [10.0, 12.0, 11.0])
    siblings = [
        _sibling_frame("peer-a", [1.0, 2.0, 3.0, 2.0] * 6),
        _sibling_frame("peer-b", [2.0, 4.0, 6.0, 4.0] * 6),
        _sibling_frame("peer-c", [3.0, 6.0, 9.0, 6.0] * 6),
    ]
    cfg = _config(
        [
            QueryConfig(
                id="m",
                promql="target",
                cold_start={
                    "borrow_from": {
                        "promql": "siblings",
                        "min_siblings": 3,
                    },
                    "max_quality_score": 0.3,
                },
            )
        ]
    )
    source = _FakeSource({"target": [target_frame], "siblings": siblings})
    exporter = Exporter()

    from promforecast.runner import Runner  # noqa: PLC0415

    runner = Runner(config=cfg, source=source, exporter=exporter)
    await runner.run_group(cfg.groups[0])

    body = exporter.render().decode()
    # Quality score should be exactly the configured cap.
    quality_lines = [
        line for line in body.splitlines() if line.startswith("forecast_quality_score{")
    ]
    assert any("0.3" in line for line in quality_lines)


@pytest.mark.asyncio
async def test_cold_start_borrow_fetched_once_for_multiple_short_series() -> None:
    """The borrow PromQL must fire exactly once per query refresh.

    Regression guard: an earlier draft of the cold-start path called
    ``fetch_siblings`` inside the per-series loop, so a query with N
    short series issued N redundant borrow queries every refresh.
    """
    short_a = _sibling_frame("a", [10.0, 12.0, 11.0])
    short_b = _sibling_frame("b", [9.0, 11.0, 10.0])
    short_c = _sibling_frame("c", [8.0, 10.0, 9.0])
    siblings = [
        _sibling_frame("peer-a", [1.0, 2.0, 3.0, 2.0] * 6),
        _sibling_frame("peer-b", [2.0, 4.0, 6.0, 4.0] * 6),
        _sibling_frame("peer-c", [3.0, 6.0, 9.0, 6.0] * 6),
    ]
    cfg = _config(
        [
            QueryConfig(
                id="m",
                promql="target",
                cold_start={
                    "borrow_from": {"promql": "siblings", "min_siblings": 3},
                },
            )
        ]
    )
    source = _FakeSource({"target": [short_a, short_b, short_c], "siblings": siblings})
    exporter = Exporter()

    from promforecast.runner import Runner  # noqa: PLC0415

    runner = Runner(config=cfg, source=source, exporter=exporter)
    await runner.run_group(cfg.groups[0])

    # Exactly one borrow fetch, regardless of the three short series.
    assert source.calls.count("siblings") == 1


@pytest.mark.asyncio
async def test_cold_start_borrow_skipped_when_no_series_is_short() -> None:
    """A query whose series all clear ``min_points`` must not issue the borrow.

    Cold-start is opt-in; configuring ``borrow_from`` shouldn't add a
    per-refresh TSDB query cost on fleets where no series is actually
    short. The prefetch is therefore lazy — deferred until the first
    short series is observed.
    """
    long_series = _sibling_frame("alive", [float(i % 10) for i in range(200)])
    cfg = _config(
        [
            QueryConfig(
                id="m",
                promql="target",
                cold_start={
                    "borrow_from": {"promql": "siblings", "min_siblings": 3},
                },
            )
        ],
        min_points=50,
    )
    source = _FakeSource(
        {"target": [long_series], "siblings": []}  # any sibling data — unused
    )
    exporter = Exporter()

    from promforecast.runner import Runner  # noqa: PLC0415

    runner = Runner(config=cfg, source=source, exporter=exporter)
    await runner.run_group(cfg.groups[0])

    assert "siblings" not in source.calls


@pytest.mark.asyncio
async def test_cold_start_skipped_when_required_regressor_configured() -> None:
    """Required regressors are not fed into the cold-start prior.

    Operators who declare a regressor as ``required`` are asking the
    forecaster not to emit without that input. Cold-start can't honor
    that contract, so the runner falls through to the standard
    ``min_points`` drop rather than synthesising a regressor-free
    forecast.
    """
    target_frame = _sibling_frame("new", [10.0, 12.0, 11.0])
    siblings = [
        _sibling_frame("peer-a", [1.0, 2.0, 3.0, 2.0] * 6),
        _sibling_frame("peer-b", [2.0, 4.0, 6.0, 4.0] * 6),
        _sibling_frame("peer-c", [3.0, 6.0, 9.0, 6.0] * 6),
    ]
    regressor_frame = _sibling_frame("d", [1.0] * 24)
    cfg = _config(
        [
            QueryConfig(
                id="m",
                promql="target",
                regressors=[
                    {"id": "deploys", "type": "custom", "promql": "deploys", "required": True}
                ],
                cold_start={
                    "borrow_from": {"promql": "siblings", "min_siblings": 3},
                },
            )
        ]
    )
    source = _FakeSource(
        {"target": [target_frame], "siblings": siblings, "deploys": [regressor_frame]}
    )
    exporter = Exporter()

    from promforecast.runner import Runner  # noqa: PLC0415

    runner = Runner(config=cfg, source=source, exporter=exporter)
    await runner.run_group(cfg.groups[0])

    body = exporter.render().decode()
    # Cold-start did not run — no ColdStart forecast, no sibling fetch.
    assert 'model="ColdStart"' not in body
    assert "siblings" not in source.calls
    # The series falls through to the regular min_points drop.
    assert 'forecast_series_dropped_total{group="g",reason="min_points"} 1.0' in body
