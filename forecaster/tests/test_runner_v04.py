"""Runner integration tests for the v0.4 features.

Covers the discovery / templating path end-to-end through the runner's
``_run_query`` orchestration — without the heavy statsforecast fit. We
use a fake source that returns canned frames per PromQL string and
assert on the per-query failure / dropped / discovery-expansion
counters that surface on the exporter snapshot.
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from promforecast.config import (
    CanaryConfig,
    Config,
    DatasourceConfig,
    DefaultsConfig,
    DiscoveryVariable,
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


def _frame(labels: dict[str, str], n: int = 5) -> SeriesFrame:
    """Frame with ``n`` points — below the runner's min_points so the
    series is dropped before any fit happens, keeping these tests free
    of statsforecast."""
    return SeriesFrame(
        labels=labels,
        timestamps=[datetime.now(tz=UTC)] * n,
        values=[1.0] * n,
    )


def _config_with(query: QueryConfig, *, min_points: int = 1_000_000) -> Config:
    return Config(
        datasource=DatasourceConfig(url="http://x"),
        server=ServerConfig(),
        safety=SafetyConfig(max_series_per_query=50, max_total_series=0),
        defaults=DefaultsConfig(min_points=min_points, models=["SeasonalNaive"]),
        groups=[GroupConfig(name="g", queries=[query])],
    )


@pytest.mark.asyncio
async def test_run_group_expands_discovery_and_records_expansions() -> None:
    """One templated query produces one fetch per discovered value."""
    query = QueryConfig(
        id="cpu",
        promql='rate(x{instance="$instance"}[5m])',
        discover=[DiscoveryVariable(name="instance", label="instance", promql='up{job="node"}')],
    )
    cfg = _config_with(query)
    src = _FakeSource(
        {
            'up{job="node"}': [_frame({"instance": "a"}), _frame({"instance": "b"})],
            'rate(x{instance="a"}[5m])': [_frame({"instance": "a"})],
            'rate(x{instance="b"}[5m])': [_frame({"instance": "b"})],
        }
    )
    exporter = Exporter()
    runner = Runner(config=cfg, source=src, exporter=exporter)

    await runner.run_group(cfg.groups[0])

    rendered = exporter.render().decode()
    assert 'forecast_discovery_expansions{group="g",query="cpu"} 2.0' in rendered
    # Both rendered queries were fanned out to the source.
    assert 'rate(x{instance="a"}[5m])' in src.calls
    assert 'rate(x{instance="b"}[5m])' in src.calls


@pytest.mark.asyncio
async def test_run_group_no_template_records_single_expansion() -> None:
    """Non-templated queries still report 1 expansion for dashboard parity."""
    query = QueryConfig(id="cpu", promql="up")
    cfg = _config_with(query)
    src = _FakeSource({"up": [_frame({"instance": "a"})]})
    exporter = Exporter()
    runner = Runner(config=cfg, source=src, exporter=exporter)

    await runner.run_group(cfg.groups[0])

    rendered = exporter.render().decode()
    assert 'forecast_discovery_expansions{group="g",query="cpu"} 1.0' in rendered


@pytest.mark.asyncio
async def test_run_group_records_discovery_failure() -> None:
    """A discovery query that returns no usable label values fails clearly."""
    query = QueryConfig(
        id="cpu",
        promql='rate(x{instance="$instance"}[5m])',
        discover=[DiscoveryVariable(name="instance", label="instance", promql="probe")],
    )
    cfg = _config_with(query)
    src = _FakeSource({"probe": []})
    exporter = Exporter()
    runner = Runner(config=cfg, source=src, exporter=exporter)

    await runner.run_group(cfg.groups[0])

    rendered = exporter.render().decode()
    assert "forecast_discovery_failures_total" in rendered
    assert 'reason="empty_result"' in rendered


@pytest.mark.asyncio
async def test_leader_metric_starts_at_zero_for_followers() -> None:
    """``set_leader_state`` with ``is_leader=False`` exposes a 0 gauge."""
    exporter = Exporter()
    exporter.set_leader_state(identity="pod-1", is_leader=False)
    rendered = exporter.render().decode()
    assert 'forecast_leader_elected{instance="pod-1"} 0.0' in rendered

    exporter.set_leader_state(identity="pod-1", is_leader=True)
    rendered = exporter.render().decode()
    assert 'forecast_leader_elected{instance="pod-1"} 1.0' in rendered
    # One transition: the initial False matches Exporter's default, so only
    # the flip to True counts.
    assert 'forecast_leader_transitions_total{instance="pod-1"} 1.0' in rendered


@pytest.mark.asyncio
async def test_query_cache_metrics_always_emitted() -> None:
    """Even at zero usage the family must appear so dashboards have data."""
    exporter = Exporter()
    rendered = exporter.render().decode()
    assert 'forecast_query_cache_events_total{kind="hit"} 0.0' in rendered
    assert 'forecast_query_cache_events_total{kind="miss"} 0.0' in rendered
    exporter.increment_query_cache_event("hit")
    exporter.increment_query_cache_event("hit")
    rendered = exporter.render().decode()
    assert 'forecast_query_cache_events_total{kind="hit"} 2.0' in rendered


@pytest.mark.asyncio
async def test_discovery_expansion_fetches_run_concurrently() -> None:
    """Discovery fan-out must dispatch fetches in parallel, not sequentially.

    Regression check for the old serial loop: with N rendered queries each
    taking T seconds, total wall time should be ~T, not N*T. We assert a
    multiplier well below N to leave headroom for scheduler jitter.
    """

    class _SlowSource(PromSource):
        def __init__(self) -> None:
            self._sleep = 0.2

        async def query_range(
            self,
            promql: str,
            start: datetime,
            end: datetime,
            step_seconds: int,
        ) -> list[SeriesFrame]:
            # Discovery probe returns the variable list synchronously;
            # only the expansion fetches sleep so we measure their
            # parallelism specifically.
            if promql == "probe":
                return [_frame({"i": str(n)}) for n in range(5)]
            await asyncio.sleep(self._sleep)
            return []

        async def aclose(self) -> None:
            return None

    query = QueryConfig(
        id="cpu",
        promql='x{i="$i"}',
        discover=[DiscoveryVariable(name="i", label="i", promql="probe")],
    )
    cfg = _config_with(query)
    exporter = Exporter()
    runner = Runner(config=cfg, source=_SlowSource(), exporter=exporter)
    t0 = time.monotonic()
    await runner.run_group(cfg.groups[0])
    elapsed = time.monotonic() - t0
    # Serial would be 5 * 0.2 = 1.0s; concurrent is bounded by max(per-fetch)
    # plus discovery probe + scheduling overhead. Allow generous headroom.
    assert elapsed < 0.7, f"expected concurrent fan-out, took {elapsed:.2f}s"


@pytest.mark.asyncio
async def test_season_length_auto_bumps_detection_counter() -> None:
    """``season_length: auto`` exposes one detection per series on /metrics.

    Uses SeasonalNaive (fast, deterministic) so the fit path completes
    and the resolver runs end-to-end. The test asserts on the counter,
    not on forecast values — actual model output is exercised in the
    statsforecast-integration tests.
    """
    query = QueryConfig(id="cpu", promql="up", season_length="auto")
    cfg = Config(
        datasource=DatasourceConfig(url="http://x"),
        server=ServerConfig(),
        safety=SafetyConfig(max_series_per_query=50, max_total_series=0),
        defaults=DefaultsConfig(
            min_points=10,
            models=["SeasonalNaive"],
            confidence_levels=[80],
            # Set a tiny lookback/step/horizon to keep the synthetic fit
            # cheap. The values are scaffolding; SeasonalNaive doesn't
            # care about the actual numbers.
        ),
        groups=[GroupConfig(name="g", queries=[query])],
    )

    # White-noise frames → detector falls back on every series. 500
    # points clears the detector's `too_short` gate and is comfortably
    # above `min_points=10` so the fit runs (cheaply).
    rng = np.random.default_rng(seed=1)
    base = datetime(2026, 5, 11, tzinfo=UTC)
    frames = [
        SeriesFrame(
            labels={"i": str(n)},
            timestamps=[base + timedelta(minutes=5 * k) for k in range(500)],
            values=list(rng.normal(size=500)),
        )
        for n in range(3)
    ]
    src = _FakeSource({"up": frames})
    exporter = Exporter()
    runner = Runner(config=cfg, source=src, exporter=exporter)

    await runner.run_group(cfg.groups[0])

    rendered = exporter.render().decode()
    # 3 detections, all on the fallback path (noise has no ACF peak).
    assert (
        'forecast_season_length_detections_total{group="g",query="cpu",source="fallback"} 3.0'
        in rendered
    )


@pytest.mark.asyncio
async def test_season_length_explicit_integer_does_not_run_detector() -> None:
    """A query without ``auto`` must not bump the detection counter at all."""
    query = QueryConfig(id="cpu", promql="up")  # default season_length: None → integer
    cfg = _config_with(query)
    src = _FakeSource({"up": [_frame({"i": "0"})]})
    exporter = Exporter()
    runner = Runner(config=cfg, source=src, exporter=exporter)

    await runner.run_group(cfg.groups[0])

    rendered = exporter.render().decode()
    assert "forecast_season_length_detections_total" not in rendered


@pytest.mark.asyncio
async def test_canary_samples_subset_and_records_dropped() -> None:
    """Canary mode samples ``ceil(weight * N)`` series and counts the drop.

    The query returns 10 series; canary weight 0.3 must keep ``ceil(10*0.3)=3``
    and surface ``forecast_series_dropped_total{reason="canary"} 7``.
    """
    query = QueryConfig(id="cpu", promql="up")
    cfg = Config(
        datasource=DatasourceConfig(url="http://x"),
        server=ServerConfig(),
        safety=SafetyConfig(max_series_per_query=50, max_total_series=0),
        defaults=DefaultsConfig(min_points=1_000_000, models=["SeasonalNaive"]),
        groups=[
            GroupConfig(
                name="g",
                queries=[query],
                canary=CanaryConfig(enabled=True, weight=0.3),
            )
        ],
    )
    frames = [_frame({"i": str(n)}) for n in range(10)]
    src = _FakeSource({"up": frames})
    exporter = Exporter()
    runner = Runner(config=cfg, source=src, exporter=exporter)

    await runner.run_group(cfg.groups[0])

    rendered = exporter.render().decode()
    assert 'forecast_series_dropped_total{group="g",reason="canary"} 7.0' in rendered


@pytest.mark.asyncio
async def test_canary_disabled_drops_nothing() -> None:
    """Disabled canary must be a no-op — no ``reason="canary"`` row."""
    query = QueryConfig(id="cpu", promql="up")
    cfg = _config_with(query)
    frames = [_frame({"i": str(n)}) for n in range(10)]
    src = _FakeSource({"up": frames})
    exporter = Exporter()
    runner = Runner(config=cfg, source=src, exporter=exporter)

    await runner.run_group(cfg.groups[0])

    rendered = exporter.render().decode()
    assert 'reason="canary"' not in rendered


@pytest.mark.asyncio
async def test_canary_keeps_at_least_one_series() -> None:
    """A small population must still produce one fit, not zero (``ceil`` rule)."""
    query = QueryConfig(id="cpu", promql="up")
    cfg = Config(
        datasource=DatasourceConfig(url="http://x"),
        server=ServerConfig(),
        safety=SafetyConfig(max_series_per_query=50, max_total_series=0),
        defaults=DefaultsConfig(min_points=1_000_000, models=["SeasonalNaive"]),
        groups=[
            GroupConfig(
                name="g",
                queries=[query],
                canary=CanaryConfig(enabled=True, weight=0.1),
            )
        ],
    )
    frames = [_frame({"i": str(n)}) for n in range(3)]  # 3 series, 10% would round to 0
    src = _FakeSource({"up": frames})
    exporter = Exporter()
    runner = Runner(config=cfg, source=src, exporter=exporter)

    await runner.run_group(cfg.groups[0])

    rendered = exporter.render().decode()
    # ceil(3 * 0.1) = 1 kept; 2 dropped
    assert 'forecast_series_dropped_total{group="g",reason="canary"} 2.0' in rendered


@pytest.mark.asyncio
async def test_runner_set_snapshot_publisher_called_after_run() -> None:
    """The HA leader's snapshot publish hook fires once per successful run."""
    query = QueryConfig(id="cpu", promql="up")
    cfg = _config_with(query)
    src = _FakeSource({"up": [_frame({"a": "x"})]})
    exporter = Exporter()
    runner = Runner(config=cfg, source=src, exporter=exporter)

    calls = {"n": 0}

    async def _publish() -> None:
        calls["n"] += 1

    runner.set_snapshot_publisher(_publish)
    await runner.run_group(cfg.groups[0])
    assert calls["n"] == 1
