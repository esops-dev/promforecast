"""Tests for the deviation primitive."""

from __future__ import annotations

import math

from promforecast.deviation import compute


def test_compute_actual_at_centre_yields_zero_ratio_inside_band() -> None:
    out = compute(actual=10.0, yhat=10.0, yhat_lower=8.0, yhat_upper=12.0)
    assert out.ratio == 0.0
    assert out.outside_band is False


def test_compute_actual_at_upper_edge_is_inside() -> None:
    out = compute(actual=12.0, yhat=10.0, yhat_lower=8.0, yhat_upper=12.0)
    assert math.isclose(out.ratio, 0.5)
    assert out.outside_band is False


def test_compute_actual_above_upper_edge_is_outside() -> None:
    out = compute(actual=15.0, yhat=10.0, yhat_lower=8.0, yhat_upper=12.0)
    assert out.ratio > 0.5
    assert out.outside_band is True


def test_compute_actual_below_lower_edge_is_outside() -> None:
    out = compute(actual=0.0, yhat=10.0, yhat_lower=8.0, yhat_upper=12.0)
    assert out.ratio < 0
    assert out.outside_band is True


def test_compute_swapped_band_is_normalised() -> None:
    # Some model implementations could plausibly return upper < lower for
    # very narrow bands; we accept and normalise.
    out = compute(actual=10.0, yhat=10.0, yhat_lower=12.0, yhat_upper=8.0)
    assert out.outside_band is False
    assert out.ratio == 0.0


def test_compute_degenerate_band_with_drift_marks_outside() -> None:
    out = compute(actual=11.0, yhat=10.0, yhat_lower=10.0, yhat_upper=10.0)
    assert math.isnan(out.ratio)
    assert out.outside_band is True


def test_compute_degenerate_band_at_centre_is_inside() -> None:
    out = compute(actual=10.0, yhat=10.0, yhat_lower=10.0, yhat_upper=10.0)
    assert math.isnan(out.ratio)
    assert out.outside_band is False


def test_compute_non_finite_inputs_yield_nan_inside() -> None:
    out = compute(actual=float("nan"), yhat=10.0, yhat_lower=8.0, yhat_upper=12.0)
    assert math.isnan(out.ratio)
    assert out.outside_band is False
