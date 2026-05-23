"""Tests for the capacity-planning primitives (ETA + growth rate)."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pandas as pd

from promforecast.capacity import (
    compute_growth_rates,
    compute_threshold_etas,
    compute_will_breach,
)
from promforecast.config import ThresholdConfig


def _forecast_frame(values: list[float], *, step_seconds: int = 60) -> pd.DataFrame:
    base = datetime(2026, 5, 21, tzinfo=UTC)
    timestamps = [base + timedelta(seconds=i * step_seconds) for i in range(len(values))]
    return pd.DataFrame({"unique_id": ["s"] * len(values), "ds": timestamps, "yhat": values})


def test_eta_zero_when_already_past_threshold() -> None:
    """When the forecast starts above the threshold, the ETA must be 0.

    Operators read ``time_to_threshold == 0`` as "already in violation"
    — silently emitting NaN would mask an in-progress incident.
    """
    df = _forecast_frame([100.0, 105.0, 110.0])
    out = compute_threshold_etas(
        forecast=df,
        thresholds=[ThresholdConfig(name="warn", value=80.0, comparator="gt")],
        levels=[],
    )
    assert len(out) == 1
    assert out[0].seconds == 0.0


def test_eta_linear_interpolates_crossing() -> None:
    """Crossing time should land between adjacent steps via linear interp."""
    # values: 0, 50, 100 at t=0,60,120 — threshold at 75 crosses between
    # step 1 (50) and step 2 (100), at exactly t=90s.
    df = _forecast_frame([0.0, 50.0, 100.0])
    out = compute_threshold_etas(
        forecast=df,
        thresholds=[ThresholdConfig(name="warn", value=75.0, comparator="gt")],
        levels=[],
    )
    assert math.isclose(out[0].seconds, 90.0)


def test_eta_returns_nan_when_no_crossing_in_horizon() -> None:
    """A horizon that never crosses should report NaN, not 0 or last step."""
    df = _forecast_frame([10.0, 20.0, 30.0])
    out = compute_threshold_etas(
        forecast=df,
        thresholds=[ThresholdConfig(name="warn", value=100.0, comparator="gt")],
        levels=[],
    )
    assert math.isnan(out[0].seconds)


def test_eta_lt_comparator_uses_lower_direction() -> None:
    """``lt`` thresholds fire on the first dip below the value."""
    df = _forecast_frame([100.0, 50.0, 20.0])
    # Threshold value=40: crosses between t=60 (50) and t=120 (20). Linear:
    # 50 -> 40 is a 10/30 fraction of the segment → 60 + (10/30)*60 = 80.
    out = compute_threshold_etas(
        forecast=df,
        thresholds=[ThresholdConfig(name="critical", value=40.0, comparator="lt")],
        levels=[],
    )
    assert math.isclose(out[0].seconds, 80.0)


def test_eta_per_level_uses_conservative_band_side() -> None:
    """For ``gt``, ETAs against ``yhat_upper`` should be earlier than against yhat."""
    df = _forecast_frame([10.0, 50.0, 60.0])
    df["yhat_upper_80"] = [20.0, 70.0, 100.0]
    df["yhat_lower_80"] = [0.0, 30.0, 20.0]
    out = compute_threshold_etas(
        forecast=df,
        thresholds=[ThresholdConfig(name="warn", value=55.0, comparator="gt")],
        levels=[80],
    )
    # Point ETA: crosses between t=60 (50) and t=120 (60) → ~90s.
    # Band ETA (yhat_upper): crosses between t=0 (20) and t=60 (70) → 42s.
    point = next(e for e in out if e.level is None)
    band = next(e for e in out if e.level == 80)
    assert band.seconds < point.seconds


def test_eta_lt_uses_lower_band_for_conservatism() -> None:
    """For ``lt``, ETAs against ``yhat_lower`` should be the conservative answer."""
    df = _forecast_frame([100.0, 50.0, 20.0])
    df["yhat_upper_95"] = [120.0, 70.0, 40.0]
    df["yhat_lower_95"] = [80.0, 30.0, 10.0]
    out = compute_threshold_etas(
        forecast=df,
        thresholds=[ThresholdConfig(name="critical", value=40.0, comparator="lt")],
        levels=[95],
    )
    point = next(e for e in out if e.level is None)
    band = next(e for e in out if e.level == 95)
    assert band.seconds < point.seconds


def test_eta_no_thresholds_emits_nothing() -> None:
    df = _forecast_frame([1.0, 2.0, 3.0])
    out = compute_threshold_etas(forecast=df, thresholds=[], levels=[80, 95])
    assert out == []


def test_growth_rate_units_per_second_linear() -> None:
    """Linear ramp of +1/step at 60s step → slope 1/60 units/second."""
    df = _forecast_frame([0.0, 1.0, 2.0, 3.0, 4.0], step_seconds=60)
    rates = compute_growth_rates(
        forecast=df,
        windows_seconds=[240.0],
        units="units_per_second",
    )
    assert len(rates) == 1
    assert math.isclose(rates[0].rate, 1 / 60, rel_tol=1e-6)


def test_growth_rate_percent_relative_change() -> None:
    """``percent`` mode returns ``(end - start) / |start|`` over the window."""
    df = _forecast_frame([10.0, 12.0, 14.0, 16.0, 20.0], step_seconds=60)
    rates = compute_growth_rates(
        forecast=df,
        windows_seconds=[240.0],
        units="percent",
    )
    # Window ends at 240s; rows in window: 0,60,120,180,240 — start 10, end 20.
    assert math.isclose(rates[0].rate, 1.0)


def test_growth_rate_window_wider_than_horizon_reports_nan() -> None:
    """Windows beyond the horizon emit NaN, not a degraded shorter window."""
    df = _forecast_frame([0.0, 1.0], step_seconds=60)
    rates = compute_growth_rates(
        forecast=df,
        windows_seconds=[3600.0],
        units="units_per_second",
    )
    assert len(rates) == 1
    assert math.isnan(rates[0].rate)


def test_growth_rate_zero_baseline_percent_is_nan() -> None:
    """``percent`` mode with a 0 start avoids divide-by-zero by reporting NaN."""
    df = _forecast_frame([0.0, 1.0, 2.0], step_seconds=60)
    rates = compute_growth_rates(
        forecast=df,
        windows_seconds=[120.0],
        units="percent",
    )
    assert math.isnan(rates[0].rate)


def test_growth_rate_skips_no_windows() -> None:
    df = _forecast_frame([1.0, 2.0])
    rates = compute_growth_rates(forecast=df, windows_seconds=[], units="units_per_second")
    assert rates == []


def test_will_breach_fires_when_eta_below_window() -> None:
    """A finite ETA below ``alert_within`` flips the gauge to 1."""
    # values 0, 50, 100 at t=0,60,120 → threshold 75 crosses at 90s.
    df = _forecast_frame([0.0, 50.0, 100.0])
    threshold = ThresholdConfig(
        name="warn",
        value=75.0,
        comparator="gt",
        alert_within=timedelta(seconds=120),
        severity="critical",
    )
    etas = compute_threshold_etas(forecast=df, thresholds=[threshold], levels=[])
    breaches = compute_will_breach(etas=etas, thresholds=[threshold])
    assert len(breaches) == 1
    assert breaches[0].value == 1.0
    assert breaches[0].severity == "critical"
    assert breaches[0].level is None


def test_will_breach_quiet_when_eta_above_window() -> None:
    """A finite ETA above ``alert_within`` emits a 0 (zero-baseline series)."""
    df = _forecast_frame([0.0, 50.0, 100.0])
    threshold = ThresholdConfig(
        name="warn",
        value=75.0,
        comparator="gt",
        alert_within=timedelta(seconds=30),  # crossing at 90s is well above 30s
        severity="warning",
    )
    etas = compute_threshold_etas(forecast=df, thresholds=[threshold], levels=[])
    breaches = compute_will_breach(etas=etas, thresholds=[threshold])
    assert len(breaches) == 1
    assert breaches[0].value == 0.0


def test_will_breach_emits_zero_when_no_crossing() -> None:
    """NaN ETA = no crossing in horizon — emit 0 so the series exists."""
    df = _forecast_frame([10.0, 20.0, 30.0])
    threshold = ThresholdConfig(
        name="warn",
        value=100.0,
        comparator="gt",
        alert_within=timedelta(seconds=120),
        severity="warning",
    )
    etas = compute_threshold_etas(forecast=df, thresholds=[threshold], levels=[])
    breaches = compute_will_breach(etas=etas, thresholds=[threshold])
    assert len(breaches) == 1
    assert breaches[0].value == 0.0


def test_will_breach_skips_thresholds_without_alert_within() -> None:
    """Thresholds without ``alert_within`` produce no will-breach samples."""
    df = _forecast_frame([0.0, 50.0, 100.0])
    threshold = ThresholdConfig(name="warn", value=75.0, comparator="gt")
    etas = compute_threshold_etas(forecast=df, thresholds=[threshold], levels=[])
    breaches = compute_will_breach(etas=etas, thresholds=[threshold])
    assert breaches == []


def test_drain_eta_returns_inf_on_no_crossing() -> None:
    """``le_drain`` returns +Inf — operators alert on ``... == +Inf for 30m``.

    The distinction from ``lt`` matters: ``lt`` returns NaN (dropped at
    render time) so an alert like ``time_to_threshold < N`` doesn't
    silently match every series; ``le_drain``'s NaN-equivalent has to
    round-trip through Prometheus and stay queryable as ``+Inf``.
    """
    # Forecast stays well above 0 the whole horizon → no drain crossing.
    df = _forecast_frame([10.0, 12.0, 15.0])
    out = compute_threshold_etas(
        forecast=df,
        thresholds=[ThresholdConfig(name="drained", value=0.0, comparator="le_drain")],
        levels=[],
    )
    assert len(out) == 1
    assert math.isinf(out[0].seconds)


def test_drain_eta_emits_finite_when_curve_crosses_from_above() -> None:
    """A draining queue → finite ETA on the first crossing-from-above."""
    # 100 → 50 → 0 → -10 at 60s step. Threshold value=0 crossed between
    # step 1 (50) and step 2 (0). Linear interp: at exactly the boundary.
    df = _forecast_frame([100.0, 50.0, 0.0, -10.0])
    out = compute_threshold_etas(
        forecast=df,
        thresholds=[ThresholdConfig(name="drained", value=0.0, comparator="le_drain")],
        levels=[],
    )
    assert math.isfinite(out[0].seconds)
    assert out[0].seconds > 0


def test_drain_band_uses_yhat_upper() -> None:
    """The conservative drain band is ``yhat_upper`` — the *latest* drain.

    Alerts ask "queue still not draining" — the conservative answer is
    the latest predicted drain. ``yhat_upper`` is a larger queue depth
    that crosses zero later than ``yhat`` does.
    """
    # Central drains by step 2; yhat_upper still positive at step 2.
    df = _forecast_frame([100.0, 50.0, 0.0, -10.0])
    df["yhat_upper_95"] = [120.0, 80.0, 20.0, -5.0]
    df["yhat_lower_95"] = [80.0, 20.0, -10.0, -15.0]
    out = compute_threshold_etas(
        forecast=df,
        thresholds=[ThresholdConfig(name="drained", value=0.0, comparator="le_drain")],
        levels=[95],
    )
    point = next(e for e in out if e.level is None)
    band = next(e for e in out if e.level == 95)
    # yhat_upper drains later than yhat → band ETA strictly greater.
    assert band.seconds > point.seconds


def test_drain_eta_zero_when_already_drained() -> None:
    """A forecast that starts below the threshold reads as already-drained (ETA=0)."""
    df = _forecast_frame([-5.0, -10.0, -15.0])
    out = compute_threshold_etas(
        forecast=df,
        thresholds=[ThresholdConfig(name="drained", value=0.0, comparator="le_drain")],
        levels=[],
    )
    assert out[0].seconds == 0.0


def test_drain_eta_zero_when_forecast_exactly_at_threshold() -> None:
    """A forecast that *equals* the drain value is drained (predicate is ``<=``).

    Regression guard: a queue forecast that touches zero exactly used to
    report ``+Inf`` "not draining" because the predicate was strict
    ``<``. The fix is to use ``<=`` for ``le_drain`` so the alert
    doesn't fire on a healthy zero-queue.
    """
    df = _forecast_frame([0.0, 0.0, 0.0])
    out = compute_threshold_etas(
        forecast=df,
        thresholds=[ThresholdConfig(name="drained", value=0.0, comparator="le_drain")],
        levels=[],
    )
    assert out[0].seconds == 0.0
    assert not math.isinf(out[0].seconds)


def test_drain_eta_still_inf_for_lt_predicate_at_threshold() -> None:
    """``lt`` keeps strict ``<`` semantics — at-threshold is *not* a crossing.

    Pairs with ``test_drain_eta_zero_when_forecast_exactly_at_threshold``
    to assert the predicate split: ``le_drain`` is ``<=``, ``lt`` stays
    ``<``. Without this guard a refactor that unifies the two predicates
    would silently change ``lt`` semantics.
    """
    df = _forecast_frame([0.0, 0.0, 0.0])
    out = compute_threshold_etas(
        forecast=df,
        thresholds=[ThresholdConfig(name="floor", value=0.0, comparator="lt")],
        levels=[],
    )
    assert math.isnan(out[0].seconds)


def test_will_breach_emits_per_level_row() -> None:
    """Per-level ETAs produce per-level will-breach rows distinguished by level."""
    df = _forecast_frame([10.0, 50.0, 60.0])
    df["yhat_upper_80"] = [20.0, 70.0, 100.0]
    df["yhat_lower_80"] = [0.0, 30.0, 20.0]
    threshold = ThresholdConfig(
        name="warn",
        value=55.0,
        comparator="gt",
        # Point ETA ~90s; band ETA (yhat_upper) ~42s — pick a window that
        # excludes the point but includes the band, so the two rows
        # disagree and we can prove the level distinction.
        alert_within=timedelta(seconds=60),
        severity="critical",
    )
    etas = compute_threshold_etas(forecast=df, thresholds=[threshold], levels=[80])
    breaches = compute_will_breach(etas=etas, thresholds=[threshold])
    assert len(breaches) == 2
    point = next(b for b in breaches if b.level is None)
    band = next(b for b in breaches if b.level == 80)
    assert point.value == 0.0
    assert band.value == 1.0
