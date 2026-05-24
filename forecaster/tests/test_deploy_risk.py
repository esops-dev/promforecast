"""Tests for the ``deploy_risk`` module and the runner's deploy-anchor wiring."""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import pytest

from promforecast import deploy_risk
from promforecast.config import (
    AccuracyConfig,
    Config,
    DatasourceConfig,
    DefaultsConfig,
    DeployAnchorConfig,
    DeploySignalConfig,
    DeploySignalsConfig,
    GroupConfig,
    QueryConfig,
    RegressorConfig,
    SafetyConfig,
    ServerConfig,
)
from promforecast.exporter import DeployRiskPoint, DeviationPoint, Exporter, GroupSnapshot
from promforecast.source import PromSource, SeriesFrame

# ---- Pure-function tests ----------------------------------------------------


def test_deploy_tracker_records_event() -> None:
    tracker = deploy_risk.DeployTracker()
    event = deploy_risk.DeployEvent(
        deployment_id="svc-1", service="svc", observed_at_seconds=100.0, active_window_seconds=300.0
    )
    tracker.record(event)
    assert tracker.active(now_monotonic=200.0) == [event]


def test_deploy_tracker_expires_after_window() -> None:
    tracker = deploy_risk.DeployTracker()
    tracker.record(
        deploy_risk.DeployEvent(
            deployment_id="svc-1",
            service="svc",
            observed_at_seconds=100.0,
            active_window_seconds=10.0,
        )
    )
    # Just inside the window
    assert len(tracker.active(now_monotonic=109.0)) == 1
    # Just outside the window
    assert tracker.active(now_monotonic=111.0) == []


def test_deploy_tracker_deduplicates_same_id() -> None:
    tracker = deploy_risk.DeployTracker()
    first = deploy_risk.DeployEvent(
        deployment_id="svc-1", service="svc", observed_at_seconds=100.0, active_window_seconds=60.0
    )
    duplicate = deploy_risk.DeployEvent(
        deployment_id="svc-1", service="svc", observed_at_seconds=130.0, active_window_seconds=60.0
    )
    tracker.record(first)
    tracker.record(duplicate)
    active = tracker.active(now_monotonic=120.0)
    assert len(active) == 1
    # Original event preserved — the active window starts at FIRST observation
    assert active[0].observed_at_seconds == 100.0


def test_deploy_tracker_evicts_when_over_cap() -> None:
    tracker = deploy_risk.DeployTracker(max_per_service=2)
    for i in range(3):
        tracker.record(
            deploy_risk.DeployEvent(
                deployment_id=f"svc-{i}",
                service="svc",
                observed_at_seconds=100.0 + i,
                active_window_seconds=600.0,
            )
        )
    active = tracker.active(now_monotonic=150.0)
    # Oldest one (svc-0) was evicted.
    assert len(active) == 2
    assert {e.deployment_id for e in active} == {"svc-1", "svc-2"}


def test_deploy_tracker_rejects_zero_cap() -> None:
    with pytest.raises(ValueError):
        deploy_risk.DeployTracker(max_per_service=0)


def test_deploy_tracker_per_service_caps_independently() -> None:
    """Per-service caps from the config map enforce independently per service."""
    tracker = deploy_risk.DeployTracker(
        max_per_service={"a": 1, "b": 3},
        default_max=16,
    )
    for i in range(2):
        tracker.record(
            deploy_risk.DeployEvent(
                deployment_id=f"a-{i}",
                service="a",
                observed_at_seconds=100.0 + i,
                active_window_seconds=600.0,
            )
        )
    for i in range(4):
        tracker.record(
            deploy_risk.DeployEvent(
                deployment_id=f"b-{i}",
                service="b",
                observed_at_seconds=100.0 + i,
                active_window_seconds=600.0,
            )
        )
    # service ``a`` cap is 1; only the most recent survives eviction.
    a_active = tracker.active_for_service("a", now_monotonic=150.0)
    assert {e.deployment_id for e in a_active} == {"a-1"}
    # service ``b`` cap is 3; oldest evicted.
    b_active = tracker.active_for_service("b", now_monotonic=150.0)
    assert {e.deployment_id for e in b_active} == {"b-1", "b-2", "b-3"}


def test_deploy_tracker_unknown_service_uses_default_cap() -> None:
    """Recording for a service not in the per-service map falls back to default_max."""
    tracker = deploy_risk.DeployTracker(
        max_per_service={"known": 5},
        default_max=2,
    )
    for i in range(3):
        tracker.record(
            deploy_risk.DeployEvent(
                deployment_id=f"u-{i}",
                service="unknown",
                observed_at_seconds=100.0 + i,
                active_window_seconds=600.0,
            )
        )
    assert len(tracker.active_for_service("unknown", now_monotonic=150.0)) == 2


def test_deploy_tracker_rejects_negative_cap_in_map() -> None:
    with pytest.raises(ValueError, match=r"my-svc"):
        deploy_risk.DeployTracker(max_per_service={"my-svc": 0}, default_max=5)


def test_compute_risk_score_returns_weighted_mean() -> None:
    deploys = [
        deploy_risk.DeployEvent(
            deployment_id="svc-1",
            service="svc",
            observed_at_seconds=100.0,
            active_window_seconds=300.0,
        )
    ]
    signals = {
        "svc": [
            deploy_risk.SignalSpec(query_id="latency", weight=1.0),
            deploy_risk.SignalSpec(query_id="errors", weight=2.0),
        ]
    }
    deviations = {("svc", "latency"): 0.5, ("svc", "errors"): 1.0}
    out = deploy_risk.compute_deploy_risk_scores(
        deploys=deploys,
        signals_by_service=signals,
        deviation_lookup=lambda service, qid: deviations[(service, qid)],
    )
    # Weighted mean: (1.0*0.5 + 2.0*1.0) / (1.0 + 2.0) = 2.5/3 ≈ 0.833
    assert out[0].score == pytest.approx(2.5 / 3.0)
    assert out[0].contributing_signals == 2


def test_compute_risk_score_uses_absolute_deviation() -> None:
    """A negative deviation contributes the same risk as a positive one."""
    deploys = [
        deploy_risk.DeployEvent(
            deployment_id="svc-1",
            service="svc",
            observed_at_seconds=100.0,
            active_window_seconds=300.0,
        )
    ]
    signals = {"svc": [deploy_risk.SignalSpec(query_id="latency", weight=1.0)]}
    out = deploy_risk.compute_deploy_risk_scores(
        deploys=deploys,
        signals_by_service=signals,
        deviation_lookup=lambda service, qid: -2.0,
    )
    assert out[0].score == 2.0


def test_compute_risk_score_drops_nan_signals() -> None:
    """A signal returning NaN deviation is excluded; contributing_signals reflects it."""
    deploys = [
        deploy_risk.DeployEvent(
            deployment_id="svc-1",
            service="svc",
            observed_at_seconds=100.0,
            active_window_seconds=300.0,
        )
    ]
    signals = {
        "svc": [
            deploy_risk.SignalSpec(query_id="latency", weight=1.0),
            deploy_risk.SignalSpec(query_id="errors", weight=1.0),
            deploy_risk.SignalSpec(query_id="saturation", weight=1.0),
        ]
    }
    deviations = {
        ("svc", "latency"): 0.3,
        ("svc", "errors"): float("nan"),
        ("svc", "saturation"): 0.6,
    }
    out = deploy_risk.compute_deploy_risk_scores(
        deploys=deploys,
        signals_by_service=signals,
        deviation_lookup=lambda service, qid: deviations[(service, qid)],
    )
    assert out[0].score == pytest.approx((0.3 + 0.6) / 2)
    assert out[0].contributing_signals == 2


def test_compute_risk_score_handles_no_configured_signals() -> None:
    """A deploy whose service has no signals emits a 0-score sentinel row."""
    deploys = [
        deploy_risk.DeployEvent(
            deployment_id="svc-1",
            service="unknown_service",
            observed_at_seconds=100.0,
            active_window_seconds=300.0,
        )
    ]
    out = deploy_risk.compute_deploy_risk_scores(
        deploys=deploys,
        signals_by_service={},
        deviation_lookup=lambda s, q: 1.0,
    )
    assert out[0].score == 0.0
    assert out[0].contributing_signals == 0


def test_detect_deploy_from_regressor_fires_on_transition() -> None:
    observed, ts = deploy_risk.detect_deploy_from_regressor(
        values=[0.0, 1.0],
        timestamps_unix=[100.0, 200.0],
    )
    assert observed is True
    assert ts == 200.0


def test_detect_deploy_from_regressor_no_transition() -> None:
    """Two consecutive positives don't re-trigger — already a deploy in progress."""
    observed, ts = deploy_risk.detect_deploy_from_regressor(
        values=[1.0, 1.0],
        timestamps_unix=[100.0, 200.0],
    )
    assert observed is False
    assert ts is None


def test_detect_deploy_from_regressor_below_threshold() -> None:
    observed, _ = deploy_risk.detect_deploy_from_regressor(
        values=[0.0, 0.5],
        timestamps_unix=[100.0, 200.0],
        threshold=1.0,
    )
    assert observed is False


def test_detect_deploy_from_regressor_handles_nan() -> None:
    observed, _ = deploy_risk.detect_deploy_from_regressor(
        values=[float("nan"), 1.0],
        timestamps_unix=[100.0, 200.0],
    )
    assert observed is False


def test_detect_deploy_from_regressor_handles_short_input() -> None:
    observed, _ = deploy_risk.detect_deploy_from_regressor(values=[1.0], timestamps_unix=[100.0])
    assert observed is False


def test_derive_deployment_id_includes_service_and_timestamp() -> None:
    out = deploy_risk.derive_deployment_id("payments", 1715000000.5)
    assert out == "payments-1715000000"


# ---- Config validation -----------------------------------------------------


def _base_config(deploy_signals: list[DeploySignalsConfig]) -> Config:
    return Config(
        datasource=DatasourceConfig(url="http://x"),
        server=ServerConfig(),
        safety=SafetyConfig(),
        defaults=DefaultsConfig(
            min_points=5,
            models=["SeasonalNaive"],
            confidence_levels=[80],
            accuracy=AccuracyConfig(evaluate=False),
        ),
        groups=[
            GroupConfig(
                name="g",
                queries=[
                    QueryConfig(
                        id="latency",
                        promql="up{type='latency'}",
                        regressors=[RegressorConfig(id="deploys", promql="up")],
                    ),
                    QueryConfig(id="errors", promql="up{type='errors'}"),
                ],
            ),
        ],
        deploy_signals=deploy_signals,
    )


def test_config_validates_known_anchor_and_signal_query() -> None:
    """Reference checking ensures anchor + signal queries exist."""
    from promforecast.config import _validate_deploy_signals  # noqa: PLC0415

    cfg = _base_config(
        [
            DeploySignalsConfig(
                service="svc",
                anchor=DeployAnchorConfig(query="latency", regressor_id="deploys"),
                signals=[
                    DeploySignalConfig(query="latency", weight=1.0),
                    DeploySignalConfig(query="errors", weight=2.0),
                ],
            )
        ]
    )
    # Should not raise.
    _validate_deploy_signals(cfg)


def test_config_rejects_unknown_anchor_query() -> None:
    from promforecast.config import _validate_deploy_signals  # noqa: PLC0415

    cfg = _base_config(
        [
            DeploySignalsConfig(
                service="svc",
                anchor=DeployAnchorConfig(query="missing", regressor_id="deploys"),
                signals=[DeploySignalConfig(query="latency")],
            )
        ]
    )
    with pytest.raises(ValueError, match=r"anchor\.query"):
        _validate_deploy_signals(cfg)


def test_config_rejects_unknown_signal_query() -> None:
    from promforecast.config import _validate_deploy_signals  # noqa: PLC0415

    cfg = _base_config(
        [
            DeploySignalsConfig(
                service="svc",
                anchor=DeployAnchorConfig(query="latency", regressor_id="deploys"),
                signals=[DeploySignalConfig(query="nope")],
            )
        ]
    )
    with pytest.raises(ValueError, match="nope"):
        _validate_deploy_signals(cfg)


def test_config_rejects_missing_anchor_regressor() -> None:
    from promforecast.config import _validate_deploy_signals  # noqa: PLC0415

    cfg = _base_config(
        [
            DeploySignalsConfig(
                service="svc",
                anchor=DeployAnchorConfig(query="errors", regressor_id="deploys"),
                signals=[DeploySignalConfig(query="latency")],
            )
        ]
    )
    with pytest.raises(ValueError, match="regressor_id"):
        _validate_deploy_signals(cfg)


def test_config_rejects_change_point_anchor() -> None:
    """The schema only accepts ``regressor`` today; ``change_point`` is rejected
    at pydantic-parse time so a config can't pass ``helm install`` and then
    crash the forecaster on boot.
    """
    with pytest.raises(ValueError, match="type"):
        DeployAnchorConfig(type="change_point", query="latency", regressor_id="deploys")  # type: ignore[arg-type]


def test_config_rejects_duplicate_service() -> None:
    from promforecast.config import _validate_deploy_signals  # noqa: PLC0415

    cfg = _base_config(
        [
            DeploySignalsConfig(
                service="svc",
                anchor=DeployAnchorConfig(query="latency", regressor_id="deploys"),
                signals=[DeploySignalConfig(query="latency")],
            ),
            DeploySignalsConfig(
                service="svc",
                anchor=DeployAnchorConfig(query="latency", regressor_id="deploys"),
                signals=[DeploySignalConfig(query="errors")],
            ),
        ]
    )
    with pytest.raises(ValueError, match="appears twice"):
        _validate_deploy_signals(cfg)


# ---- Exporter integration --------------------------------------------------


def test_exporter_renders_deploy_risk_score() -> None:
    exporter = Exporter()
    exporter.replace_deploy_risk_scores(
        [
            DeployRiskPoint(
                labels={
                    "service": "payments",
                    "deployment_id": "payments-100",
                    "contributing_signals": "2",
                },
                score=0.75,
            )
        ]
    )
    body = exporter.render().decode()
    assert "forecast_deploy_risk_score" in body
    assert 'service="payments"' in body
    assert 'deployment_id="payments-100"' in body


def test_exporter_replace_clears_stale_scores() -> None:
    exporter = Exporter()
    exporter.replace_deploy_risk_scores(
        [
            DeployRiskPoint(
                labels={"service": "a", "deployment_id": "a-1", "contributing_signals": "1"},
                score=0.5,
            )
        ]
    )
    # Replace with empty list → gauge family disappears.
    exporter.replace_deploy_risk_scores([])
    body = exporter.render().decode()
    assert "forecast_deploy_risk_score" not in body


def test_replace_deploy_risk_scores_noop_skips_lock_when_idle() -> None:
    """At steady state (no active deploys) the runner calls
    ``replace_deploy_risk_scores([])`` on every group fit. The exporter
    fast-paths the empty→empty replace so the scrape-path lock stays
    uncontended."""
    exporter = Exporter()
    # First empty call on a fresh exporter must be cheap — verify by
    # short-circuiting before touching state (no row appears either way,
    # so we exercise the fast path by asserting the gauge stays absent
    # after many calls and the state dict is the same object identity).
    state_before = exporter._deploy_risk_scores
    for _ in range(100):
        exporter.replace_deploy_risk_scores([])
    assert exporter._deploy_risk_scores is state_before
    body = exporter.render().decode()
    assert "forecast_deploy_risk_score" not in body


# ---- Runner wiring ---------------------------------------------------------


class _StubSource(PromSource):
    """Returns a single SeriesFrame whose values trigger a deploy crossing."""

    def __init__(self, values: list[float] | None = None) -> None:
        self._values = values if values is not None else [0.0, 1.0]

    async def query_range(
        self, promql: str, start: datetime, end: datetime, step_seconds: int
    ) -> list[SeriesFrame]:
        timestamps = [start, end]
        return [SeriesFrame(labels={}, timestamps=timestamps, values=self._values)]

    async def aclose(self) -> None:
        return None


@pytest.mark.asyncio
async def test_runner_refresh_deploy_risk_detects_and_publishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: anchor regressor transitions → tracker records → exporter receives a score."""
    from promforecast.runner import Runner  # noqa: PLC0415

    cfg = _base_config(
        [
            DeploySignalsConfig(
                service="svc",
                anchor=DeployAnchorConfig(query="latency", regressor_id="deploys"),
                signals=[
                    DeploySignalConfig(query="latency", weight=1.0),
                    DeploySignalConfig(query="errors", weight=1.0),
                ],
                active_window=timedelta(minutes=5),
            )
        ]
    )
    exporter = Exporter()
    # Pre-seed the snapshot with deviation rows so the score is non-zero.
    snapshot = GroupSnapshot(group="g")
    snapshot.deviation_points.extend(
        [
            DeviationPoint(
                labels={"id": "latency", "model": "M", "group": "g", "level": "95"},
                ratio=0.6,
                outside_band=True,
            ),
            DeviationPoint(
                labels={"id": "errors", "model": "M", "group": "g", "level": "95"},
                ratio=-0.4,
                outside_band=False,
            ),
        ]
    )
    exporter.update(snapshot)

    runner = Runner(config=cfg, source=_StubSource(), exporter=exporter)
    group = cfg.groups[0]
    await runner._refresh_deploy_risk(group, run_query_ids={"latency"})
    body = exporter.render().decode()
    assert "forecast_deploy_risk_score" in body
    # Weighted mean of |0.6| and |-0.4| with equal weights = 0.5
    assert "0.5" in body


@pytest.mark.asyncio
async def test_runner_refresh_deploy_risk_no_op_when_unconfigured() -> None:
    """No deploy_signals configured → nothing emitted, no crash."""
    from promforecast.runner import Runner  # noqa: PLC0415

    cfg = _base_config([])
    exporter = Exporter()
    runner = Runner(config=cfg, source=_StubSource(), exporter=exporter)
    group = cfg.groups[0]
    await runner._refresh_deploy_risk(group, run_query_ids={"latency"})
    body = exporter.render().decode()
    assert "forecast_deploy_risk_score" not in body


@pytest.mark.asyncio
async def test_runner_refresh_deploy_risk_swallows_anchor_query_failure() -> None:
    """A raising source must not crash the deploy detection path."""
    from promforecast.runner import Runner  # noqa: PLC0415

    class _RaisingSource(PromSource):
        def __init__(self) -> None:
            pass

        async def query_range(self, *a: object, **k: object) -> list[SeriesFrame]:
            raise RuntimeError("source down")

        async def aclose(self) -> None:
            return None

    cfg = _base_config(
        [
            DeploySignalsConfig(
                service="svc",
                anchor=DeployAnchorConfig(query="latency", regressor_id="deploys"),
                signals=[DeploySignalConfig(query="latency", weight=1.0)],
            )
        ]
    )
    exporter = Exporter()
    runner = Runner(config=cfg, source=_RaisingSource(), exporter=exporter)
    group = cfg.groups[0]
    # Should not raise — the failure logs and the run continues.
    await runner._refresh_deploy_risk(group, run_query_ids={"latency"})
    body = exporter.render().decode()
    # No deploy was detected, so no score row should appear.
    assert "forecast_deploy_risk_score" not in body


@pytest.mark.asyncio
async def test_runner_refresh_deploy_risk_active_deploy_persists_across_refreshes() -> None:
    """A deploy detected once stays active until its window elapses."""
    from promforecast.runner import Runner  # noqa: PLC0415

    cfg = _base_config(
        [
            DeploySignalsConfig(
                service="svc",
                anchor=DeployAnchorConfig(query="latency", regressor_id="deploys"),
                signals=[DeploySignalConfig(query="latency", weight=1.0)],
                active_window=timedelta(seconds=600),
            )
        ]
    )
    exporter = Exporter()
    snapshot = GroupSnapshot(group="g")
    snapshot.deviation_points.append(
        DeviationPoint(
            labels={"id": "latency", "model": "M", "group": "g", "level": "95"},
            ratio=0.2,
            outside_band=False,
        )
    )
    exporter.update(snapshot)
    # First call: source returns a fresh transition; deploy detected.
    runner = Runner(config=cfg, source=_StubSource([0.0, 1.0]), exporter=exporter)
    group = cfg.groups[0]
    await runner._refresh_deploy_risk(group, run_query_ids={"latency"})
    assert len(runner._deploy_tracker.active()) == 1
    # Second call: source no longer shows a fresh transition. The
    # tracker should still report the previously-detected deploy.
    runner._source = _StubSource([1.0, 1.0])  # type: ignore[attr-defined]
    await runner._refresh_deploy_risk(group, run_query_ids={"latency"})
    assert len(runner._deploy_tracker.active()) == 1


@pytest.mark.asyncio
async def test_reload_clears_deploy_risk_scores_on_partial_drop() -> None:
    """A reload that removes one service from a multi-service ``deploy_signals``
    list must clear the gauge state — otherwise the dropped service's last
    emitted scores linger on /metrics until the next group refresh (which
    may be minutes away and is never if no group still touches an anchor).
    """
    from promforecast.runner import Runner  # noqa: PLC0415

    # Pre-seed the exporter with a score for "payments" before the reload.
    cfg = _base_config(
        [
            DeploySignalsConfig(
                service="payments",
                anchor=DeployAnchorConfig(query="latency", regressor_id="deploys"),
                signals=[DeploySignalConfig(query="latency", weight=1.0)],
            ),
            DeploySignalsConfig(
                service="search",
                anchor=DeployAnchorConfig(query="latency", regressor_id="deploys"),
                signals=[DeploySignalConfig(query="errors", weight=1.0)],
            ),
        ]
    )
    exporter = Exporter()
    exporter.replace_deploy_risk_scores(
        [
            DeployRiskPoint(
                labels={
                    "service": "payments",
                    "deployment_id": "payments-1",
                    "contributing_signals": "1",
                },
                score=0.5,
            ),
            DeployRiskPoint(
                labels={
                    "service": "search",
                    "deployment_id": "search-1",
                    "contributing_signals": "1",
                },
                score=0.3,
            ),
        ]
    )
    runner = Runner(config=cfg, source=_StubSource(), exporter=exporter)
    body = exporter.render().decode()
    assert 'service="payments"' in body
    assert 'service="search"' in body

    # Reload with "payments" dropped; "search" retained.
    new_cfg = _base_config(
        [
            DeploySignalsConfig(
                service="search",
                anchor=DeployAnchorConfig(query="latency", regressor_id="deploys"),
                signals=[DeploySignalConfig(query="errors", weight=1.0)],
            ),
        ]
    )
    await runner.apply_config(new_cfg)
    body = exporter.render().decode()
    # Both previously-emitted rows must be cleared. "search" will be
    # re-published on the next group refresh; until then the gauge is
    # silent (which is the correct fail-safe — better blank than stale).
    assert 'service="payments"' not in body
    assert 'service="search"' not in body
    await runner.stop()


# ---- Module surface ---------------------------------------------------------


def test_module_exports_documented_names() -> None:
    assert deploy_risk.DeployEvent is not None
    assert deploy_risk.DeployTracker is not None
    assert deploy_risk.compute_deploy_risk_scores is not None
    assert deploy_risk.detect_deploy_from_regressor is not None
    assert deploy_risk.derive_deployment_id is not None


def test_active_window_seconds_floors_at_one() -> None:
    assert deploy_risk.active_window_seconds(timedelta(seconds=0)) == 1.0
    assert deploy_risk.active_window_seconds(timedelta(seconds=60)) == 60.0


def test_aggregate_regressor_values_single_frame_fast_path() -> None:
    """The common-case single-frame response skips the per-row mean
    allocation — verify the values pass through unchanged and the
    timestamps are unix-seconds floats."""
    from datetime import UTC, datetime  # noqa: PLC0415

    from promforecast.runner._scheduler import _aggregate_regressor_values  # noqa: PLC0415

    base = datetime(2026, 5, 21, 12, 0, tzinfo=UTC)
    frame = SeriesFrame(
        labels={},
        timestamps=[base, base + timedelta(minutes=5)],
        values=[0.0, 1.0],
    )
    values, timestamps = _aggregate_regressor_values([frame])
    assert values == [0.0, 1.0]
    assert timestamps == [base.timestamp(), (base + timedelta(minutes=5)).timestamp()]


def test_aggregate_regressor_values_multi_frame_takes_mean() -> None:
    """Multi-frame responses average values per step; misaligned-length
    frames are dropped from the average rather than corrupting it."""
    from datetime import UTC, datetime  # noqa: PLC0415

    from promforecast.runner._scheduler import _aggregate_regressor_values  # noqa: PLC0415

    base = datetime(2026, 5, 21, 12, 0, tzinfo=UTC)
    timestamps = [base, base + timedelta(minutes=5)]
    aligned1 = SeriesFrame(labels={"a": "1"}, timestamps=timestamps, values=[0.0, 2.0])
    aligned2 = SeriesFrame(labels={"a": "2"}, timestamps=timestamps, values=[1.0, 3.0])
    misaligned = SeriesFrame(labels={"a": "3"}, timestamps=timestamps, values=[5.0])
    values, _ = _aggregate_regressor_values([aligned1, aligned2, misaligned])
    # Means of [0,2] and [1,3] → [0.5, 2.5]; the misaligned frame is dropped.
    assert values == [0.5, 2.5]


def test_aggregate_regressor_values_empty_input() -> None:
    """Empty or empty-leading-frame inputs return empty tuples — defensive
    for a degenerate source response."""
    from promforecast.runner._scheduler import _aggregate_regressor_values  # noqa: PLC0415

    assert _aggregate_regressor_values([]) == ([], [])
    empty_frame = SeriesFrame(labels={}, timestamps=[], values=[])
    assert _aggregate_regressor_values([empty_frame]) == ([], [])


def test_isfinite_guard_in_compute() -> None:
    """Inf in deviation lookup still excludes the signal (math.isfinite catches inf)."""
    deploys = [
        deploy_risk.DeployEvent(
            deployment_id="x",
            service="svc",
            observed_at_seconds=0.0,
            active_window_seconds=60.0,
        )
    ]
    signals = {"svc": [deploy_risk.SignalSpec(query_id="q", weight=1.0)]}
    out = deploy_risk.compute_deploy_risk_scores(
        deploys=deploys,
        signals_by_service=signals,
        deviation_lookup=lambda s, q: math.inf,
    )
    assert out[0].contributing_signals == 0
    assert out[0].score == 0.0
