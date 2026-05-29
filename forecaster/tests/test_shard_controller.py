"""Unit tests for ``main._ShardController`` orchestration.

The controller is the assembled wiring that sharding's runtime depends on:
it refreshes ownership before starting the runner, runs the assigner elector
(``leader_assigned``), and drives the periodic refresh→reconcile loop. These
tests exercise that orchestration with fakes — no Kubernetes, no Redis — so
they run in the normal suite. The real-cluster elector path is covered
separately in ``test_leader_assigned_kind`` (skipped without a cluster).
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from datetime import datetime

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
from promforecast.main import _ShardController
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


def _config(group_names: list[str]) -> Config:
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


class _FakeCoordinator:
    """Coordinator whose owned set can flip; counts refresh() calls."""

    def __init__(self, owned: set[str], *, periodic: bool) -> None:
        self.owned = owned
        self._periodic = periodic
        self.refreshed = 0

    @property
    def shard_id(self) -> str:
        return "s0"

    def owns(self, group_name: str) -> bool:
        return group_name in self.owned

    def owned_groups(self, group_names: Iterable[str]) -> set[str]:
        return {n for n in group_names if n in self.owned}

    def needs_periodic_refresh(self) -> bool:
        return self._periodic

    async def refresh(self, group_names: Iterable[str]) -> None:
        self.refreshed += 1

    async def close(self) -> None:
        return None


class _FakeElector:
    """Stand-in elector: run() blocks until stop() like the real one."""

    def __init__(self, *, leader: bool = True) -> None:
        self._leader = leader
        self._stop = asyncio.Event()
        self.ran = False
        self.stopped = False

    async def run(self) -> None:
        self.ran = True
        await self._stop.wait()

    async def stop(self) -> None:
        self.stopped = True
        self._stop.set()

    def is_leader(self) -> bool:
        return self._leader


async def _wait_for(predicate, timeout: float = 2.0) -> bool:  # type: ignore[no-untyped-def]
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.02)
    return predicate()


async def test_controller_starts_runner_with_owned_slice_and_no_loop_when_static() -> None:
    # consistent_hash-style: no periodic refresh, just refresh-once + start.
    cfg = _config(["a", "b", "c"])
    coord = _FakeCoordinator(owned={"a", "c"}, periodic=False)
    runner = Runner(config=cfg, source=_EmptySource(), exporter=Exporter())
    runner.set_shard_coordinator(coord)  # type: ignore[arg-type]
    controller = _ShardController(
        coordinator=coord,  # type: ignore[arg-type]
        runner=runner,
        elector=None,
        refresh_interval_seconds=0.05,
    )
    try:
        await controller.start()
        assert coord.refreshed == 1  # refreshed once before runner start
        assert set(runner._tasks) == {"a", "c"}  # only owned groups scheduled
        # No periodic loop for a static coordinator.
        assert controller._refresh_task is None
    finally:
        await controller.stop()
        await runner.stop()


async def test_controller_runs_elector_and_periodic_reconcile() -> None:
    cfg = _config(["a", "b", "c"])
    coord = _FakeCoordinator(owned={"a", "b"}, periodic=True)
    elector = _FakeElector(leader=True)
    runner = Runner(config=cfg, source=_EmptySource(), exporter=Exporter())
    runner.set_shard_coordinator(coord)  # type: ignore[arg-type]
    controller = _ShardController(
        coordinator=coord,  # type: ignore[arg-type]
        runner=runner,
        elector=elector,  # type: ignore[arg-type]
        refresh_interval_seconds=0.05,
    )
    try:
        await controller.start()
        assert await _wait_for(lambda: elector.ran)  # the assigner elector task launched
        assert set(runner._tasks) == {"a", "b"}
        assert controller._refresh_task is not None  # periodic loop running

        # Ownership rebalances (a replica joined): drop "a", gain "c".
        coord.owned = {"b", "c"}
        converged = await _wait_for(lambda: set(runner._tasks) == {"b", "c"})
        assert converged, f"reconcile did not converge: {set(runner._tasks)}"
        # The periodic loop kept calling refresh().
        assert coord.refreshed >= 2
    finally:
        await controller.stop()
        await runner.stop()
    # stop() tore down the elector and the refresh loop cleanly.
    assert elector.stopped
    assert controller._refresh_task is not None and controller._refresh_task.done()


async def test_controller_stop_is_idempotent_without_elector() -> None:
    cfg = _config(["a"])
    coord = _FakeCoordinator(owned={"a"}, periodic=True)
    runner = Runner(config=cfg, source=_EmptySource(), exporter=Exporter())
    runner.set_shard_coordinator(coord)  # type: ignore[arg-type]
    controller = _ShardController(
        coordinator=coord,  # type: ignore[arg-type]
        runner=runner,
        elector=None,
        refresh_interval_seconds=0.05,
    )
    await controller.start()
    await controller.stop()
    await controller.stop()  # second stop must not raise
    await runner.stop()
