"""Unit tests for the self-observability trackers and rendering."""

from __future__ import annotations

import math

import pytest

from promforecast.exporter import Exporter
from promforecast.self_observability import (
    FitOutcomeTracker,
    GroupResourceTracker,
    RegressorStabilityTracker,
    compute_stability_score,
)


def test_fit_outcome_tracker_records_successes_and_failures() -> None:
    tracker = FitOutcomeTracker(windows_seconds=(60, 3600))
    now = 1_000_000.0
    for _ in range(8):
        tracker.record(group="g", model="AutoARIMA", success=True, now=now)
    for _ in range(2):
        tracker.record(group="g", model="AutoARIMA", success=False, now=now)
    points = tracker.render(now=now)
    by_window = {p.window_label: p.ratio for p in points}
    assert by_window["1m"] == pytest.approx(0.8)
    assert by_window["1h"] == pytest.approx(0.8)


def test_fit_outcome_tracker_prunes_outside_window() -> None:
    tracker = FitOutcomeTracker(windows_seconds=(60,))
    now = 1_000_000.0
    # Old samples
    tracker.record(group="g", model="m", success=False, now=now - 1000)
    tracker.record(group="g", model="m", success=False, now=now - 800)
    # Recent samples
    tracker.record(group="g", model="m", success=True, now=now - 30)
    points = tracker.render(now=now)
    assert len(points) == 1
    assert points[0].ratio == pytest.approx(1.0)


def test_fit_outcome_tracker_empty_window_emits_no_point() -> None:
    tracker = FitOutcomeTracker(windows_seconds=(60,))
    now = 1_000_000.0
    tracker.record(group="g", model="m", success=True, now=now - 1000)
    # The sample is outside the 60s window; render should be empty.
    assert tracker.render(now=now) == []


def test_fit_outcome_tracker_retain_groups_drops_others() -> None:
    tracker = FitOutcomeTracker()
    tracker.record(group="keep", model="m", success=True)
    tracker.record(group="drop", model="m", success=True)
    tracker.retain_groups({"keep"})
    points = tracker.render()
    assert {p.group for p in points} == {"keep"}


def test_compute_stability_score_no_drift() -> None:
    # Identical lookback and recent -> 1.0
    values = [10.0] * 50
    score = compute_stability_score(values)
    assert score == pytest.approx(1.0)


def test_compute_stability_score_moderate_drift() -> None:
    # Baseline mean 100, recent mean 110 -> 10% shift -> score 0.9
    values = [100.0] * 40 + [110.0] * 10
    score = compute_stability_score(values, recent_fraction=0.2)
    assert score == pytest.approx(0.9, abs=0.01)


def test_compute_stability_score_saturates_at_zero() -> None:
    # Baseline 100, recent 250 -> 150% shift -> saturates at 0
    values = [100.0] * 40 + [250.0] * 10
    score = compute_stability_score(values, recent_fraction=0.2, saturation=1.0)
    assert score == 0.0


def test_compute_stability_score_nan_on_thin_input() -> None:
    assert math.isnan(compute_stability_score([1.0, 2.0]))


def test_compute_stability_score_constant_zero_is_stable() -> None:
    # A constant zero series is by definition stable — recent and
    # baseline both have mean 0, and the implementation's denom-floor
    # keeps the relative metric well-defined.
    assert compute_stability_score([0.0] * 50) == pytest.approx(1.0)


def test_regressor_stability_tracker_records_and_renders() -> None:
    tracker = RegressorStabilityTracker()
    tracker.record(
        group="g",
        query="q",
        regressor_id="deploys",
        values=[100.0] * 50,
    )
    points = tracker.render()
    assert len(points) == 1
    assert points[0].group == "g"
    assert points[0].score == pytest.approx(1.0)


def test_group_resource_tracker_accumulates_cpu() -> None:
    tracker = GroupResourceTracker()
    tracker.add_cpu("g", 1.5)
    tracker.add_cpu("g", 2.5)
    tracker.set_memory("g", 1024.0)
    points = tracker.render()
    assert len(points) == 1
    assert points[0].cpu_seconds_total == pytest.approx(4.0)
    assert points[0].memory_bytes == pytest.approx(1024.0)


def test_exporter_records_self_observability_state() -> None:
    exporter = Exporter()
    exporter.record_fit_outcome(group="g", model="AutoARIMA", success=True)
    exporter.record_fit_outcome(group="g", model="AutoARIMA", success=False)
    exporter.record_regressor_stability(
        group="g",
        query="cpu",
        regressor_id="deploys",
        values=[1.0] * 50,
    )
    exporter.add_group_cpu_seconds("g", 3.0)
    exporter.set_group_memory_bytes("g", 4096.0)
    body = exporter.render().decode()
    assert "forecast_model_fit_success_ratio" in body
    assert "forecast_regressor_stability_score" in body
    assert "forecast_group_memory_bytes" in body
    assert "forecast_group_cpu_seconds_total" in body
    # 1/2 success -> ratio 0.5
    assert 'forecast_model_fit_success_ratio{group="g",model="AutoARIMA"' in body
    # Cleanup hook drops the group's rows.
    exporter.retain_self_observability_groups(set())
    body2 = exporter.render().decode()
    assert "forecast_model_fit_success_ratio" not in body2
    assert "forecast_group_memory_bytes" not in body2
