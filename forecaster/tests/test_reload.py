"""Tests for the config hot-reload coordinator and runner diff logic.

The reload story has three moving parts: parse + validate, runtime diff /
swap, and the trigger surfaces (HTTP / SIGHUP / inotify). These tests cover
the first two end-to-end and rely on the trigger surfaces being thin
wrappers over ``ConfigReloader.reload``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
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
from promforecast.exporter import Exporter, GroupSnapshot
from promforecast.reload import (
    ConfigMapWatcher,
    ConfigReloader,
    ReloadError,
    log_group_queries_loaded,
)
from promforecast.runner import Runner
from promforecast.source import PromSource

if TYPE_CHECKING:
    from datetime import datetime

    from promforecast.source import SeriesFrame


class _NullSource(PromSource):
    """Stand-in source: never queries, never errors. The reload tests only
    exercise the orchestration plane, never the actual fit path."""

    def __init__(self) -> None:
        self.closed = False

    async def query_range(
        self,
        promql: str,
        start: datetime,
        end: datetime,
        step_seconds: int,
    ) -> list[SeriesFrame]:
        return []

    async def aclose(self) -> None:
        self.closed = True


def _config(groups: list[GroupConfig]) -> Config:
    return Config(
        datasource=DatasourceConfig(url="http://x"),
        server=ServerConfig(),
        safety=SafetyConfig(),
        defaults=DefaultsConfig(min_points=10, models=["SeasonalNaive"]),
        groups=groups,
    )


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(body)
    return path


_BASE_CONFIG = """
apiVersion: promforecast.io/v1
datasource:
  url: http://vm:8428/
server:
  refresh_interval: 1h
defaults:
  models: [SeasonalNaive]
groups:
  - name: g1
    queries:
      - id: m
        promql: up
"""


@pytest.mark.asyncio
async def test_apply_config_diffs_groups_and_clears_removed_snapshots() -> None:
    """A removed group's snapshot should disappear from /metrics."""
    cfg_v1 = _config(
        [
            GroupConfig(name="a", queries=[QueryConfig(id="ma", promql="up")]),
            GroupConfig(name="b", queries=[QueryConfig(id="mb", promql="up")]),
        ]
    )
    exporter = Exporter()
    runner = Runner(config=cfg_v1, source=_NullSource(), exporter=exporter)
    # Seed both groups with a fake snapshot so the removal is observable.
    exporter.update(GroupSnapshot(group="a"))
    exporter.update(GroupSnapshot(group="b"))

    cfg_v2 = _config([GroupConfig(name="a", queries=[QueryConfig(id="ma", promql="up")])])
    result = await runner.apply_config(cfg_v2)

    assert result.removed == ["b"]
    assert result.added == []
    body = exporter.render().decode()
    # Group b should no longer appear in operational metrics.
    assert 'group="b"' not in body


@pytest.mark.asyncio
async def test_apply_config_starts_added_group_task() -> None:
    cfg_v1 = _config([GroupConfig(name="a", queries=[QueryConfig(id="ma", promql="up")])])
    exporter = Exporter()
    runner = Runner(config=cfg_v1, source=_NullSource(), exporter=exporter)

    cfg_v2 = _config(
        [
            GroupConfig(name="a", queries=[QueryConfig(id="ma", promql="up")]),
            GroupConfig(name="c", queries=[QueryConfig(id="mc", promql="up")]),
        ]
    )
    result = await runner.apply_config(cfg_v2)

    assert result.added == ["c"]
    assert "c" in runner._tasks
    # Cleanup so pytest doesn't warn about the dangling task.
    await runner.stop()


@pytest.mark.asyncio
async def test_apply_config_detects_query_change_in_existing_group() -> None:
    cfg_v1 = _config([GroupConfig(name="a", queries=[QueryConfig(id="ma", promql="up")])])
    exporter = Exporter()
    runner = Runner(config=cfg_v1, source=_NullSource(), exporter=exporter)

    cfg_v2 = _config([GroupConfig(name="a", queries=[QueryConfig(id="ma", promql="rate(up[5m])")])])
    result = await runner.apply_config(cfg_v2)
    assert result.changed == ["a"]
    await runner.stop()


@pytest.mark.asyncio
async def test_reloader_validate_then_swap_rolls_back_on_bad_config(tmp_path: Path) -> None:
    """A broken config must NOT replace the running config or kill the loop."""
    path = _write(tmp_path, _BASE_CONFIG)
    exporter = Exporter()

    from promforecast import config as config_module  # noqa: PLC0415

    cfg = config_module.load(path)
    runner = Runner(config=cfg, source=_NullSource(), exporter=exporter)
    reloader = ConfigReloader(runner=runner, exporter=exporter, config_path=path)

    # Corrupt the file so the next reload fails validation.
    path.write_text("datasource: { url: http://vm:8428/ }\ngroups: [{name: g, queries: 'oops'}]\n")
    with pytest.raises(ReloadError):
        await reloader.reload()

    # Running config preserved: the runner still holds the v1 group config.
    assert [g.name for g in runner._config.groups] == ["g1"]
    # Telemetry: failure counter incremented, success flag flipped to False.
    body = exporter.render().decode()
    assert "forecast_config_reload_failures_total 1.0" in body
    assert "forecast_config_reload_success 0.0" in body
    await runner.stop()


@pytest.mark.asyncio
async def test_reloader_swap_succeeds_and_records_telemetry(tmp_path: Path) -> None:
    path = _write(tmp_path, _BASE_CONFIG)
    exporter = Exporter()

    from promforecast import config as config_module  # noqa: PLC0415

    cfg = config_module.load(path)
    runner = Runner(config=cfg, source=_NullSource(), exporter=exporter)
    reloader = ConfigReloader(runner=runner, exporter=exporter, config_path=path)

    # Add a new group on disk and reload.
    path.write_text(
        _BASE_CONFIG
        + """  - name: g2
    queries:
      - id: m2
        promql: rate(up[5m])
"""
    )
    outcome = await reloader.reload()
    assert outcome.result.added == ["g2"]
    assert outcome.result.removed == []
    assert outcome.result.changed == []

    body = exporter.render().decode()
    assert "forecast_config_reload_success 1.0" in body
    # timestamp is non-zero now
    assert "forecast_config_last_reload_timestamp_seconds 0.0" not in body
    await runner.stop()


@pytest.mark.asyncio
async def test_reloader_missing_file_raises_reloaderror(tmp_path: Path) -> None:
    path = _write(tmp_path, _BASE_CONFIG)

    from promforecast import config as config_module  # noqa: PLC0415

    cfg = config_module.load(path)
    exporter = Exporter()
    runner = Runner(config=cfg, source=_NullSource(), exporter=exporter)
    reloader = ConfigReloader(runner=runner, exporter=exporter, config_path=path)

    path.unlink()
    with pytest.raises(ReloadError) as info:
        await reloader.reload()
    assert "config file not found" in str(info.value)
    await runner.stop()


@pytest.mark.asyncio
async def test_apply_config_is_no_op_when_config_unchanged() -> None:
    """An identical config must not drain or restart any tasks.

    Kubernetes occasionally re-projects ConfigMaps with the same bytes
    (e.g. on pod-spec churn). If we naively drained on every reload call,
    every group would pause needlessly.
    """
    cfg = _config([GroupConfig(name="a", queries=[QueryConfig(id="ma", promql="up")])])
    runner = Runner(config=cfg, source=_NullSource(), exporter=Exporter())
    await runner.start()
    original_task = runner._tasks["a"]

    # Re-apply the *same* config object; ReloadResult should be empty and
    # the existing task should still be running (not replaced).
    result = await runner.apply_config(cfg)
    assert result.added == []
    assert result.removed == []
    assert result.changed == []
    assert runner._tasks["a"] is original_task
    await runner.stop()


@pytest.mark.asyncio
async def test_apply_config_drains_in_flight_run_before_swapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reload must wait for the current ``run_group`` to finish.

    The user story explicitly requires "in-flight runs are not interrupted".
    Hard cancellation would tear down the fit mid-way and lose the
    snapshot it was about to write; we want graceful drain instead.
    """
    cfg = _config([GroupConfig(name="a", queries=[QueryConfig(id="ma", promql="up")])])
    exporter = Exporter()

    in_flight = asyncio.Event()
    let_finish = asyncio.Event()
    completed = False

    async def _slow_run_group(self, group, *, queries=None):  # type: ignore[no-untyped-def]
        nonlocal completed
        in_flight.set()
        await let_finish.wait()
        completed = True

    monkeypatch.setattr(Runner, "run_group", _slow_run_group)

    runner = Runner(config=cfg, source=_NullSource(), exporter=exporter)
    await runner.start()
    # Wait for the loop to actually enter run_group before reloading.
    await asyncio.wait_for(in_flight.wait(), timeout=2.0)

    # Kick off the reload; it should block on the drain.
    cfg_v2 = _config([GroupConfig(name="a", queries=[QueryConfig(id="ma", promql="rate(up[5m])")])])
    reload_task = asyncio.create_task(runner.apply_config(cfg_v2))

    # Give apply_config a moment to start; it must NOT have completed yet
    # because run_group is still blocked on let_finish.
    await asyncio.sleep(0.05)
    assert not reload_task.done(), "reload completed without waiting for in-flight run"
    assert not completed, "run_group was interrupted instead of being drained"

    # Release the in-flight run; the drain should now complete.
    let_finish.set()
    await asyncio.wait_for(reload_task, timeout=2.0)
    assert completed, "run_group did not finish before the swap"
    await runner.stop()


@pytest.mark.asyncio
async def test_apply_config_hard_cancels_after_drain_grace_expires(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stuck run_group must not block a reload indefinitely."""
    cfg = _config([GroupConfig(name="a", queries=[QueryConfig(id="ma", promql="up")])])
    # Drive the grace floor down to ~0s so the test runs in milliseconds.
    monkeypatch.setattr(Runner, "_drain_grace_seconds", staticmethod(lambda config: 0.05))

    in_flight = asyncio.Event()

    async def _hanging_run_group(self, group, *, queries=None):  # type: ignore[no-untyped-def]
        in_flight.set()
        await asyncio.sleep(60)

    monkeypatch.setattr(Runner, "run_group", _hanging_run_group)

    runner = Runner(config=cfg, source=_NullSource(), exporter=Exporter())
    await runner.start()
    await asyncio.wait_for(in_flight.wait(), timeout=2.0)

    cfg_v2 = _config([GroupConfig(name="a", queries=[QueryConfig(id="ma", promql="rate(up[5m])")])])
    # Should complete within the test timeout — graceful drain falls back
    # to hard cancellation when the grace period expires.
    await asyncio.wait_for(runner.apply_config(cfg_v2), timeout=2.0)
    await runner.stop()


def test_log_group_queries_loaded_emits_one_line_per_group() -> None:
    """Boot/reload emission lists every loaded query ID per group.

    The line set must mirror what's *actually* in the live config — so a
    macOS Docker Desktop bind-mount that ships a truncated YAML on reload
    is visible from the logs (one missing query == one missing line, or a
    short ``queries`` list).
    """
    import structlog  # noqa: PLC0415

    cfg = _config(
        [
            GroupConfig(
                name="full",
                queries=[
                    QueryConfig(id="m1", promql="up"),
                    QueryConfig(id="m2", promql="rate(up[5m])"),
                ],
            ),
            GroupConfig(name="empty", queries=[QueryConfig(id="m3", promql="up")]),
        ]
    )
    with structlog.testing.capture_logs() as captured:
        log_group_queries_loaded(cfg, source="boot")
    events = [e for e in captured if e["event"] == "group_queries_loaded"]
    assert [(e["group"], e["queries"], e["source"]) for e in events] == [
        ("full", ["m1", "m2"], "boot"),
        ("empty", ["m3"], "boot"),
    ]


@pytest.mark.asyncio
async def test_reload_logs_truncated_query_set(tmp_path: Path) -> None:
    """Truncated YAML on reload → emitted log lines match the truncation.

    Re-enacts the macOS Docker Desktop bind-mount footgun: the file on
    disk loses a query and the next reload picks up the shorter set. The
    ``group_queries_loaded`` line emitted after ``config_reload_succeeded``
    is the operator's signal that the swap did not include the missing
    query — which would otherwise be invisible behind the still-green
    reload status.
    """
    import structlog  # noqa: PLC0415

    full_config = (
        _BASE_CONFIG
        + """      - id: m2
        promql: rate(up[5m])
"""
    )
    path = _write(tmp_path, full_config)
    from promforecast import config as config_module  # noqa: PLC0415

    cfg = config_module.load(path)
    exporter = Exporter()
    runner = Runner(config=cfg, source=_NullSource(), exporter=exporter)
    reloader = ConfigReloader(runner=runner, exporter=exporter, config_path=path)

    # Drop ``m2`` on disk — mimics a truncated bind-mount projection.
    path.write_text(_BASE_CONFIG)
    with structlog.testing.capture_logs() as captured:
        await reloader.reload()

    events = [e for e in captured if e["event"] == "group_queries_loaded"]
    assert len(events) == 1
    assert events[0]["group"] == "g1"
    assert events[0]["queries"] == ["m"]
    assert events[0]["source"] == "reload"
    await runner.stop()


def test_boot_load_emits_config_loaded_then_group_queries_in_order(tmp_path: Path) -> None:
    """``main._load_config_for_boot`` must emit ``config_loaded`` first, then
    one ``group_queries_loaded`` per group with ``source="boot"``.

    Guards the ordering contract documented in
    ``docs/operations/reload.md`` so a log filter pinned to
    ``config_loaded -> group_queries_loaded`` keeps working even if the
    boot path is refactored.
    """
    import structlog  # noqa: PLC0415

    from promforecast.main import _load_config_for_boot  # noqa: PLC0415

    full_config = (
        _BASE_CONFIG
        + """      - id: m2
        promql: rate(up[5m])
"""
    )
    path = _write(tmp_path, full_config)
    with structlog.testing.capture_logs() as captured:
        cfg = _load_config_for_boot(path)

    events = [e for e in captured if e["event"] in {"config_loaded", "group_queries_loaded"}]
    assert events[0]["event"] == "config_loaded"
    assert events[0]["groups"] == len(cfg.groups)
    # Every other event must be the group line tagged ``source="boot"``.
    boot_events = [e for e in events[1:] if e["event"] == "group_queries_loaded"]
    assert boot_events, "boot path did not emit any group_queries_loaded line"
    for event in boot_events:
        assert event["source"] == "boot"
    assert [e["group"] for e in boot_events] == [g.name for g in cfg.groups]
    assert [e["queries"] for e in boot_events] == [[q.id for q in g.queries] for g in cfg.groups]


@pytest.mark.asyncio
async def test_configmap_watcher_triggers_reload_on_content_change(tmp_path: Path) -> None:
    """Inotify-style watcher: file change → reload fires."""
    path = _write(tmp_path, _BASE_CONFIG)
    from promforecast import config as config_module  # noqa: PLC0415

    cfg = config_module.load(path)
    exporter = Exporter()
    runner = Runner(config=cfg, source=_NullSource(), exporter=exporter)
    reloader = ConfigReloader(runner=runner, exporter=exporter, config_path=path)

    triggered = asyncio.Event()
    real_reload = reloader.reload

    async def _spy() -> object:
        result = await real_reload()
        triggered.set()
        return result

    reloader.reload = _spy  # type: ignore[method-assign]

    watcher = ConfigMapWatcher(reloader, poll_interval_seconds=0.05)
    await watcher.start()
    try:
        # Force a content change: a benign group rename.
        path.write_text(_BASE_CONFIG.replace("id: m", "id: m_renamed"))
        await asyncio.wait_for(triggered.wait(), timeout=2.0)
    finally:
        await watcher.stop()
        await runner.stop()
