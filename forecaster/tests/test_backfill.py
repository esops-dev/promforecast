"""Tests for the backfill CLI's pure helpers.

The full backfill walk requires the model registry + statsforecast
and is exercised end-to-end via docker-compose. This module covers
the parsing helpers, the window-cap guard, the per-group timestamp
walk, and the curve-rebasing math, plus a synthetic happy-path test
that wires ``_run_query_at`` against a fake source and a fake model
without touching the real statsforecast import surface.
"""

from __future__ import annotations

import io
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from promforecast import backfill as backfill_module
from promforecast import models as model_registry
from promforecast.backfill import (
    BackfillError,
    _check_window,
    _iter_backfill_timestamps,
    _parse_duration_seconds,
    _parse_to_timestamp,
    _rebase_curve_to_as_of,
    _run_query_at,
    _select_groups,
    run_backfill,
)
from promforecast.config import (
    AccuracyConfig,
    Config,
    DatasourceConfig,
    DefaultsConfig,
    EmissionConfig,
    GroupConfig,
    QueryConfig,
    SafetyConfig,
    ServerConfig,
)
from promforecast.sink import ForecastCurvePoint
from promforecast.source import PromSource, SeriesFrame


def _config(
    *,
    refresh_interval: timedelta = timedelta(hours=1),
    backfill_max_window: timedelta = timedelta(days=90),
) -> Config:
    return Config(
        datasource=DatasourceConfig(url="http://x"),
        server=ServerConfig(refresh_interval=refresh_interval),
        safety=SafetyConfig(backfill_max_window=backfill_max_window),
        defaults=DefaultsConfig(),
        groups=[
            GroupConfig(name="g1", queries=[QueryConfig(id="m", promql="up")]),
            GroupConfig(name="g2", queries=[QueryConfig(id="m", promql="up")]),
        ],
    )


def test_parse_duration_seconds_accepts_compact_form() -> None:
    assert _parse_duration_seconds("5m", flag="--from") == 300.0
    assert _parse_duration_seconds("1h", flag="--from") == 3600.0
    assert _parse_duration_seconds("30d", flag="--from") == 30 * 86_400


def test_parse_duration_seconds_rejects_garbage() -> None:
    with pytest.raises(BackfillError):
        _parse_duration_seconds("nope", flag="--from")


def test_parse_to_timestamp_now() -> None:
    before = datetime.now(tz=UTC)
    dt = _parse_to_timestamp(None)
    after = datetime.now(tz=UTC)
    assert before <= dt <= after


def test_parse_to_timestamp_iso8601() -> None:
    dt = _parse_to_timestamp("2026-05-24T12:00:00Z")
    assert dt == datetime(2026, 5, 24, 12, 0, 0, tzinfo=UTC)


def test_parse_to_timestamp_rejects_garbage() -> None:
    with pytest.raises(BackfillError):
        _parse_to_timestamp("yesterday")


def test_check_window_passes_under_cap() -> None:
    cfg = _config(backfill_max_window=timedelta(days=30))
    _check_window(cfg, from_seconds=14 * 86_400)


def test_check_window_rejects_over_cap() -> None:
    cfg = _config(backfill_max_window=timedelta(days=30))
    with pytest.raises(BackfillError) as exc:
        _check_window(cfg, from_seconds=60 * 86_400)
    assert "backfill_max_window" in str(exc.value)


def test_iter_backfill_timestamps_walks_at_refresh_cadence() -> None:
    cfg = _config(refresh_interval=timedelta(hours=1))
    group = cfg.groups[0]
    to_time = datetime(2026, 5, 24, 12, 0, 0, tzinfo=UTC)
    timestamps = list(_iter_backfill_timestamps(cfg, group, 3 * 3600.0, to_time))
    # 3h window at 1h cadence: 4 timestamps (3h ago, 2h ago, 1h ago, now).
    assert len(timestamps) == 4
    assert timestamps[0] == to_time - timedelta(hours=3)
    assert timestamps[-1] == to_time


def test_iter_backfill_timestamps_floors_cadence_at_60s() -> None:
    """A misconfigured sub-minute refresh interval doesn't explode timestamp count."""
    cfg = _config(refresh_interval=timedelta(seconds=10))
    group = cfg.groups[0]
    to_time = datetime(2026, 5, 24, 12, 0, 0, tzinfo=UTC)
    timestamps = list(_iter_backfill_timestamps(cfg, group, 600.0, to_time))
    # 10 minutes / 60s cadence = 11 timestamps.
    assert len(timestamps) == 11


def test_select_groups_returns_all_when_no_selector() -> None:
    cfg = _config()
    assert [g.name for g in _select_groups(cfg, None)] == ["g1", "g2"]


def test_select_groups_filters_to_selector() -> None:
    cfg = _config()
    assert [g.name for g in _select_groups(cfg, ["g2"])] == ["g2"]
    assert _select_groups(cfg, ["missing"]) == []


def test_rebase_curve_shifts_to_target_anchor() -> None:
    """Curve points keep their relative spacing but anchor at as_of."""
    as_of = datetime(2026, 1, 1, tzinfo=UTC)
    raw_ts_ms = int(datetime(2026, 5, 24, tzinfo=UTC).timestamp() * 1000)
    step_ms = 60_000  # 60s
    points = [
        ForecastCurvePoint(
            metric="up_forecast",
            labels={"id": "up", "horizon": "next"},
            value=1.0,
            timestamp_ms=raw_ts_ms + i * step_ms,
        )
        for i in range(3)
    ]
    rebased = list(_rebase_curve_to_as_of(points, as_of=as_of, step_seconds=60))
    target_first_ms = int(as_of.timestamp() * 1000)
    assert rebased[0].timestamp_ms == target_first_ms
    assert rebased[1].timestamp_ms == target_first_ms + step_ms
    assert rebased[2].timestamp_ms == target_first_ms + 2 * step_ms


def test_rebase_curve_handles_empty_list() -> None:
    assert list(_rebase_curve_to_as_of([], as_of=datetime.now(tz=UTC), step_seconds=60)) == []


def test_run_backfill_returns_2_on_missing_config() -> None:
    rc = run_backfill(
        config_path=Path("/nonexistent/foo.yaml"),
        from_duration="1h",
        to_timestamp=None,
        group_selector=None,
        sink_url_override=None,
        out=io.StringIO(),
    )
    assert rc == 2


def test_run_backfill_returns_1_when_no_sink_configured(tmp_path: Path) -> None:
    cfg_body = """
apiVersion: promforecast.io/v1
datasource:
  url: http://127.0.0.1:1
groups:
  - name: g
    queries:
      - id: existing
        promql: up
"""
    path = tmp_path / "cfg.yaml"
    path.write_text(cfg_body)
    rc = run_backfill(
        config_path=path,
        from_duration="1h",
        to_timestamp=None,
        group_selector=None,
        sink_url_override=None,
        out=io.StringIO(),
    )
    assert rc == 1


class _FakeSource(PromSource):
    """Synthetic PromSource: returns canned frames, never opens sockets."""

    def __init__(self, frames: Sequence[SeriesFrame]) -> None:
        self._frames = list(frames)
        self.calls: list[str] = []

    async def query_range(
        self,
        promql: str,
        start: datetime,
        end: datetime,
        step_seconds: int,
    ) -> list[SeriesFrame]:
        self.calls.append(promql)
        return list(self._frames)

    async def aclose(self) -> None:
        return None


class _FakeModel:
    """Drop-in for a real statsforecast model used by the backfill path."""

    def __init__(self, name: str, value: float = 1.5) -> None:
        self.name = name
        self._value = value

    def fit_predict(
        self,
        df: pd.DataFrame,
        horizon_steps: int,
        freq: str,
        levels: list[int],
        regressors: pd.DataFrame | None,
    ) -> pd.DataFrame:
        start_ts = df["ds"].iloc[-1] + pd.Timedelta(freq)
        ts = pd.date_range(start=start_ts, periods=horizon_steps, freq=freq)
        out: dict[str, Any] = {"ds": ts, "yhat": [self._value] * horizon_steps}
        for level in levels:
            out[f"yhat_lower_{level}"] = [self._value - 1.0] * horizon_steps
            out[f"yhat_upper_{level}"] = [self._value + 1.0] * horizon_steps
        return pd.DataFrame(out)


@pytest.mark.asyncio
async def test_run_query_at_happy_path_emits_anchored_curve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end ``_run_query_at`` covers fit + safeguard + rebase.

    Wires a fake model registry entry so the real statsforecast import
    stays out of the path; covers the dominant code path (preprocess
    accepts → frame fits → curve points emit anchored at ``as_of``).
    Without this the per-fit pipeline — which is the only functionally
    novel code in the module — has no unit coverage.
    """
    monkeypatch.setattr(
        model_registry,
        "build",
        lambda name, season_length, params=None: _FakeModel(name, value=2.0),
    )
    as_of = datetime(2026, 1, 15, 12, 0, 0, tzinfo=UTC)
    step = timedelta(minutes=1)
    timestamps = [as_of - step * (n + 1) for n in reversed(range(60))]
    frame = SeriesFrame(
        labels={"instance": "host-a", "__name__": "up"},
        timestamps=timestamps,
        values=[float(i) for i in range(60)],
    )
    cfg = Config(
        datasource=DatasourceConfig(url="http://x"),
        server=ServerConfig(),
        safety=SafetyConfig(),
        defaults=DefaultsConfig(
            min_points=10,
            models=["AutoARIMA"],
            confidence_levels=[80],
            step=step,
            lookback=timedelta(hours=1),
            horizon=timedelta(minutes=10),
            accuracy=AccuracyConfig(
                evaluate=False, auto_select=False, horizons=[timedelta(minutes=5)]
            ),
            emission=EmissionConfig(deviation=False, quality=False, contribution=False),
        ),
        groups=[GroupConfig(name="g", queries=[QueryConfig(id="m", promql="up")])],
    )
    source = _FakeSource([frame])
    produced = await _run_query_at(
        config=cfg,
        group=cfg.groups[0],
        query=cfg.groups[0].queries[0],
        source=source,
        as_of=as_of,
    )
    assert produced.fit_count == ["AutoARIMA"], "exactly one fit per configured model"
    assert produced.curve, "fit must produce a non-empty curve"

    # Pure forecast curve points are anchored at as_of (in ms).
    target_first_ms = int(as_of.timestamp() * 1000)
    forecast_points = [p for p in produced.curve if p.metric.endswith("_forecast")]
    assert forecast_points, "expected at least one ``..._forecast`` curve point"
    assert min(p.timestamp_ms for p in forecast_points) == target_first_ms


class _RecordingSink:
    """Records every batch passed to ``write`` so the audit metric is observable."""

    def __init__(self) -> None:
        self.batches: list[list[ForecastCurvePoint]] = []

    async def write(
        self,
        points: list[ForecastCurvePoint],
        on_retry: object | None = None,
    ) -> None:
        del on_retry
        self.batches.append(list(points))

    async def aclose(self) -> None:
        return None


@pytest.mark.asyncio
async def test_walk_backfill_emits_audit_row_with_gauge_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The audit row uses ``_last`` (gauge) not ``_total`` (counter).

    Backfill runs are independent one-shot operations with no
    monotonic-counter semantics; a second run with fewer fits than
    the first would false-fire ``rate()`` / ``increase()`` if the
    metric carried the ``_total`` suffix. This test pins the gauge
    naming so a refactor can't silently regress it.
    """
    monkeypatch.setattr(
        model_registry,
        "build",
        lambda name, season_length, params=None: _FakeModel(name, value=1.0),
    )
    as_of = datetime(2026, 1, 15, 12, 0, 0, tzinfo=UTC)
    step = timedelta(minutes=1)
    timestamps = [as_of - step * (n + 1) for n in reversed(range(60))]
    frame = SeriesFrame(
        labels={"instance": "host-a"},
        timestamps=timestamps,
        values=[float(i) for i in range(60)],
    )
    cfg = Config(
        datasource=DatasourceConfig(url="http://x"),
        server=ServerConfig(refresh_interval=timedelta(hours=1)),
        safety=SafetyConfig(),
        defaults=DefaultsConfig(
            min_points=10,
            models=["AutoARIMA"],
            confidence_levels=[80],
            step=step,
            lookback=timedelta(hours=1),
            horizon=timedelta(minutes=10),
            accuracy=AccuracyConfig(
                evaluate=False, auto_select=False, horizons=[timedelta(minutes=5)]
            ),
            emission=EmissionConfig(deviation=False, quality=False, contribution=False),
        ),
        groups=[GroupConfig(name="g", queries=[QueryConfig(id="m", promql="up")])],
    )
    source = _FakeSource([frame])
    sink = _RecordingSink()
    await backfill_module._walk_backfill(  # type: ignore[attr-defined]
        config=cfg,
        source=source,
        sink=sink,  # type: ignore[arg-type]
        from_seconds=3 * 3600.0,
        to_time=as_of,
        groups=cfg.groups,
        out=io.StringIO(),
    )
    # Find the audit batch — exactly one ForecastCurvePoint with
    # metric name "forecast_backfill_fits_last".
    audit_points = [
        p for batch in sink.batches for p in batch if p.metric == "forecast_backfill_fits_last"
    ]
    assert len(audit_points) == 1
    audit = audit_points[0]
    assert audit.labels == {"group": "g"}
    assert audit.value > 0
    # And the old (incorrect) name must NOT appear anywhere.
    assert not any(
        p.metric == "forecast_backfill_fits_total" for batch in sink.batches for p in batch
    ), "old counter-style name should be gone"


@pytest.mark.asyncio
async def test_run_query_at_isolates_per_series_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bad series doesn't abort the rest of the query's walk.

    Mirrors the live runner's per-series failure isolation contract.
    Force the preprocess helper to raise on one frame; the other
    must still produce a non-empty curve.
    """
    monkeypatch.setattr(
        model_registry,
        "build",
        lambda name, season_length, params=None: _FakeModel(name, value=1.0),
    )

    real_preprocess = backfill_module.preprocess_mod.preprocess_series
    call_count = {"n": 0}

    def explosive_preprocess(*args: Any, **kwargs: Any) -> Any:
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("synthetic preprocess failure")
        return real_preprocess(*args, **kwargs)

    monkeypatch.setattr(backfill_module.preprocess_mod, "preprocess_series", explosive_preprocess)

    as_of = datetime(2026, 1, 15, 12, 0, 0, tzinfo=UTC)
    step = timedelta(minutes=1)
    timestamps = [as_of - step * (n + 1) for n in reversed(range(60))]
    bad_frame = SeriesFrame(labels={"instance": "host-a"}, timestamps=timestamps, values=[1.0] * 60)
    good_frame = SeriesFrame(
        labels={"instance": "host-b"},
        timestamps=timestamps,
        values=[float(i) for i in range(60)],
    )
    cfg = Config(
        datasource=DatasourceConfig(url="http://x"),
        server=ServerConfig(),
        safety=SafetyConfig(),
        defaults=DefaultsConfig(
            min_points=10,
            models=["AutoARIMA"],
            confidence_levels=[80],
            step=step,
            lookback=timedelta(hours=1),
            horizon=timedelta(minutes=10),
            accuracy=AccuracyConfig(
                evaluate=False, auto_select=False, horizons=[timedelta(minutes=5)]
            ),
            emission=EmissionConfig(deviation=False, quality=False, contribution=False),
        ),
        groups=[GroupConfig(name="g", queries=[QueryConfig(id="m", promql="up")])],
    )
    source = _FakeSource([bad_frame, good_frame])
    produced = await _run_query_at(
        config=cfg,
        group=cfg.groups[0],
        query=cfg.groups[0].queries[0],
        source=source,
        as_of=as_of,
    )
    # The good frame still fit; the bad one didn't take the query down.
    assert produced.fit_count == ["AutoARIMA"]


def test_run_backfill_returns_1_when_window_over_cap(tmp_path: Path) -> None:
    """An over-cap window fails before any TSDB I/O happens."""
    cfg_body = """
apiVersion: promforecast.io/v1
datasource:
  url: http://127.0.0.1:1
safety:
  backfill_max_window: 7d
sink:
  remote_write:
    enabled: true
    url: http://127.0.0.1:1/api/v1/import/prometheus
groups:
  - name: g
    queries:
      - id: existing
        promql: up
"""
    path = tmp_path / "cfg.yaml"
    path.write_text(cfg_body)
    rc = run_backfill(
        config_path=path,
        from_duration="30d",
        to_timestamp=None,
        group_selector=None,
        sink_url_override=None,
        out=io.StringIO(),
    )
    assert rc == 1
