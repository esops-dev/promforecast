"""Tests for the SLO error-budget primitives."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pandas as pd

from promforecast.config import SLOConfig
from promforecast.slo import compute_budget


def _forecast_frame(values: list[float], *, step_seconds: int = 60) -> pd.DataFrame:
    base = datetime(2026, 5, 21, tzinfo=UTC)
    timestamps = [base + timedelta(seconds=i * step_seconds) for i in range(len(values))]
    return pd.DataFrame({"unique_id": ["s"] * len(values), "ds": timestamps, "yhat": values})


def test_remaining_seconds_matches_objective_times_window() -> None:
    """``remaining_seconds`` is the static budget — ``(1 - SLO/100) * window``."""
    # 99.9% SLO over 30d → allowed_rate = 0.001 → budget = 0.001 * 30d = 2592s
    df = _forecast_frame([0.001, 0.001, 0.001])
    slo = SLOConfig(objective=99.9, windows=[timedelta(days=30)])
    out = compute_budget(forecast=df, slo=slo, levels=[])
    assert len(out) == 1
    assert math.isclose(out[0].remaining_seconds, 0.001 * 30 * 86400, rel_tol=1e-6)


def test_burn_rate_one_when_consuming_at_allowed_rate() -> None:
    """Burn rate of ``1`` means consuming the budget at the rate that exhausts it."""
    # SLO 99.9% → allowed_rate 0.001. yhat 0.001 → burn_rate 1.0.
    df = _forecast_frame([0.001, 0.001, 0.001])
    slo = SLOConfig(objective=99.9, windows=[timedelta(days=30)])
    out = compute_budget(forecast=df, slo=slo, levels=[])
    assert math.isclose(out[0].burn_rate, 1.0)


def test_burn_rate_scales_with_predicted_rate() -> None:
    """Burn rate of 30 means consuming the budget 30x faster than sustainable."""
    df = _forecast_frame([0.03, 0.03, 0.03])
    slo = SLOConfig(objective=99.9, windows=[timedelta(days=30)])
    out = compute_budget(forecast=df, slo=slo, levels=[])
    assert math.isclose(out[0].burn_rate, 30.0)


def test_exhausted_in_inverse_of_burn_rate_times_window() -> None:
    """``exhausted_in = window / burn_rate`` for the central forecast."""
    # burn_rate=30 → exhaust at 30x the rate → window / 30 seconds.
    df = _forecast_frame([0.03, 0.03, 0.03])
    slo = SLOConfig(objective=99.9, windows=[timedelta(days=30)])
    out = compute_budget(forecast=df, slo=slo, levels=[])
    expected = 30 * 86400 / 30  # window / burn_rate
    assert math.isclose(out[0].exhausted_in_point, expected, rel_tol=1e-6)


def test_exhausted_in_is_infinite_on_zero_forecast() -> None:
    """A predicted error rate of zero means the budget never exhausts → +Inf."""
    df = _forecast_frame([0.0, 0.0, 0.0])
    slo = SLOConfig(objective=99.9, windows=[timedelta(days=30)])
    out = compute_budget(forecast=df, slo=slo, levels=[])
    assert math.isinf(out[0].exhausted_in_point)


def test_exhausted_in_is_infinite_on_negative_forecast() -> None:
    """A negative forecast (budget recovering) also gives +Inf exhausted_in."""
    df = _forecast_frame([-0.001, -0.001, -0.001])
    slo = SLOConfig(objective=99.9, windows=[timedelta(days=30)])
    out = compute_budget(forecast=df, slo=slo, levels=[])
    assert math.isinf(out[0].exhausted_in_point)


def test_per_level_exhausted_in_uses_upper_band() -> None:
    """Per-level row uses ``yhat_upper`` — the more pessimistic (faster-burn) side."""
    df = _forecast_frame([0.01, 0.01, 0.01])
    df["yhat_upper_95"] = [0.03, 0.03, 0.03]
    df["yhat_lower_95"] = [0.005, 0.005, 0.005]
    slo = SLOConfig(objective=99.9, windows=[timedelta(days=30)])
    out = compute_budget(forecast=df, slo=slo, levels=[95])
    sample = out[0]
    # Point uses yhat=0.01 → burn_rate=10 → exhaust at 30d/10
    # Band uses yhat_upper=0.03 → burn_rate=30 → exhaust at 30d/30 (sooner)
    assert sample.exhausted_in_per_level[95] < sample.exhausted_in_point


def test_no_emission_for_empty_forecast() -> None:
    """An empty frame yields no samples (caller can ``extend`` unconditionally)."""
    df = pd.DataFrame({"ds": [], "yhat": []})
    slo = SLOConfig(objective=99.9, windows=[timedelta(days=30)])
    assert compute_budget(forecast=df, slo=slo, levels=[]) == []


def test_multiple_windows_emit_one_sample_each() -> None:
    """Each configured window produces its own row."""
    df = _forecast_frame([0.001, 0.001])
    slo = SLOConfig(
        objective=99.9,
        windows=[timedelta(minutes=5), timedelta(hours=1), timedelta(days=30)],
    )
    out = compute_budget(forecast=df, slo=slo, levels=[])
    assert len(out) == 3
    window_seconds = sorted(s.window_seconds for s in out)
    assert window_seconds == [300.0, 3600.0, 30 * 86400.0]


def test_no_windows_emits_nothing() -> None:
    slo = SLOConfig(objective=99.9, windows=[])
    df = _forecast_frame([0.001, 0.001])
    assert compute_budget(forecast=df, slo=slo, levels=[]) == []


def test_exhausted_in_responds_to_calibrated_band() -> None:
    """A wider ``yhat_upper`` (after conformal calibration) shortens exhausted_in.

    Operators alert on the per-level row; the band shown in /metrics
    must agree with the budget projection at the same labelset. This
    test exercises the contract on the pure helper — the runner-level
    fix that threads calibration into ``compute_budget`` is verified
    by the runner test suite.
    """
    df = _forecast_frame([0.005, 0.005, 0.005])
    # Raw band: yhat_upper exhausted_in at level=95 corresponds to a
    # particular budget consumption rate; widening it should make the
    # budget exhaust sooner.
    df["yhat_upper_95"] = [0.005, 0.005, 0.005]
    df["yhat_lower_95"] = [0.005, 0.005, 0.005]
    slo = SLOConfig(objective=99.9, windows=[timedelta(days=30)])
    raw = compute_budget(forecast=df, slo=slo, levels=[95])
    raw_level = raw[0].exhausted_in_per_level[95]

    df["yhat_upper_95"] = [0.05, 0.05, 0.05]  # calibrated wider
    cal = compute_budget(forecast=df, slo=slo, levels=[95])
    cal_level = cal[0].exhausted_in_per_level[95]
    assert cal_level < raw_level
