"""Runner-level sharding behaviour: ownership filtering, reconciliation, and
the ``shard_id`` label threading into output points.

The pure ownership logic is covered in ``test_sharding``; here we assert the
runner *acts* on a coordinator — only owned groups get scheduled loops, an
ownership change reconciles the running set, and the ``shard_id`` label is
folded into the factory's output.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from promforecast import sharding
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
from promforecast.point_factory import SeriesPointFactory
from promforecast.runner import Runner
from promforecast.source import PromSource, SeriesFrame


class _EmptySource(PromSource):
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def query_range(
        self, promql: str, start: datetime, end: datetime, step_seconds: int
    ) -> list[SeriesFrame]:
        self.calls.append(promql)
        return []

    async def aclose(self) -> None:
        return None


def _multi_group_config(group_names: list[str]) -> Config:
    return Config(
        datasource=DatasourceConfig(url="http://x"),
        server=ServerConfig(),
        safety=SafetyConfig(),
        defaults=DefaultsConfig(min_points=100, models=["SeasonalNaive"]),
        groups=[
            GroupConfig(name=n, queries=[QueryConfig(id=f"m_{n}", promql="up")])
            for n in group_names
        ],
    )


def test_point_factory_folds_shard_id_into_every_family() -> None:
    forecast = pd.DataFrame({"yhat": [1.0, 2.0]})
    factory = SeriesPointFactory(
        metric_id="m",
        base_labels={"instance": "host-a", "__name__": "raw"},
        group="g",
        levels=[80],
        shard_id="shard-0",
    )
    points = list(factory.forecast_points(forecast, "SeasonalNaive", None))
    assert points and all(p.labels["shard_id"] == "shard-0" for p in points)
    # __name__ stripped, source labels preserved.
    assert points[0].labels["instance"] == "host-a"
    assert "__name__" not in points[0].labels


def test_point_factory_omits_shard_id_when_unset() -> None:
    forecast = pd.DataFrame({"yhat": [1.0]})
    factory = SeriesPointFactory(
        metric_id="m", base_labels={"instance": "a"}, group="g", levels=[80]
    )
    points = list(factory.forecast_points(forecast, "SeasonalNaive", None))
    assert points and "shard_id" not in points[0].labels


@pytest.mark.asyncio
async def test_start_schedules_only_owned_groups() -> None:
    names = [f"grp{i}" for i in range(12)]
    cfg = _multi_group_config(names)
    coord = sharding.ConsistentHashCoordinator(shard_index=0, replica_count=3, shard_id="s0")
    exporter = Exporter()
    runner = Runner(config=cfg, source=_EmptySource(), exporter=exporter)
    runner.set_shard_coordinator(coord)
    try:
        await runner.start()
        owned = coord.owned_groups(names)
        assert owned  # shard 0 owns at least one of 12 groups over 3 shards
        assert set(runner._tasks) == owned
        # The ownership gauge mirrors what was scheduled.
        body = exporter.render().decode()
        for name in owned:
            assert f'forecast_shard_owns_group{{group="{name}",shard_id="s0"}} 1.0' in body
        for name in set(names) - owned:
            assert f'group="{name}"' not in body
    finally:
        await runner.stop()


class _MutableCoordinator:
    """Coordinator whose owned set can be flipped to drive reconciliation."""

    def __init__(self, owned: set[str]) -> None:
        self._owned = owned

    @property
    def shard_id(self) -> str:
        return "s0"

    def owns(self, group_name: str) -> bool:
        return group_name in self._owned

    def owned_groups(self, group_names: object) -> set[str]:
        return {n for n in group_names if n in self._owned}  # type: ignore[union-attr]

    def needs_periodic_refresh(self) -> bool:
        return True

    async def refresh(self, group_names: object) -> None:
        return None

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_reconcile_starts_and_drains_on_ownership_change() -> None:
    names = ["a", "b", "c"]
    cfg = _multi_group_config(names)
    coord = _MutableCoordinator(owned={"a", "b"})
    exporter = Exporter()
    runner = Runner(config=cfg, source=_EmptySource(), exporter=exporter)
    runner.set_shard_coordinator(coord)
    try:
        await runner.start()
        assert set(runner._tasks) == {"a", "b"}
        # Seed per-(group, query) tracker state for the soon-to-be-relinquished
        # group "a" and the retained group "b".
        runner._drift_windows[("a", "m_a", "k", "SeasonalNaive")] = object()  # type: ignore[assignment]
        runner._drift_windows[("b", "m_b", "k", "SeasonalNaive")] = object()  # type: ignore[assignment]
        # Ownership shifts: lose "a", gain "c".
        coord._owned = {"b", "c"}
        await runner.reconcile_shard_ownership()
        assert set(runner._tasks) == {"b", "c"}
        body = exporter.render().decode()
        assert 'forecast_shard_owns_group{group="c",shard_id="s0"} 1.0' in body
        assert 'group="a"' not in body
        # Relinquished group's tracker state pruned; retained group's kept.
        drift_groups = {k[0] for k in runner._drift_windows}
        assert "a" not in drift_groups
        assert "b" in drift_groups
    finally:
        await runner.stop()


@pytest.mark.asyncio
async def test_unsharded_runner_owns_every_group() -> None:
    names = ["a", "b", "c"]
    cfg = _multi_group_config(names)
    runner = Runner(config=cfg, source=_EmptySource(), exporter=Exporter())
    try:
        await runner.start()
        assert set(runner._tasks) == set(names)
    finally:
        await runner.stop()
