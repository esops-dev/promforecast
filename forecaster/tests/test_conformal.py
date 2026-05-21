"""Tests for split-conformal band calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd

from promforecast.conformal import apply_offsets, compute_offsets, holdout_split
from promforecast.exporter import ForecastPoint
from promforecast.runner import (
    _apply_offsets_to_curve_points,
    _apply_offsets_to_snapshot_points,
)
from promforecast.sink import ForecastCurvePoint


def _calib_predictions(
    *,
    yhat: list[float],
    lower_80: list[float],
    upper_80: list[float],
    lower_95: list[float] | None = None,
    upper_95: list[float] | None = None,
) -> pd.DataFrame:
    data: dict[str, list[float]] = {
        "yhat": yhat,
        "yhat_lower_80": lower_80,
        "yhat_upper_80": upper_80,
    }
    if lower_95 is not None and upper_95 is not None:
        data["yhat_lower_95"] = lower_95
        data["yhat_upper_95"] = upper_95
    return pd.DataFrame(data)


def test_tight_band_widens_to_match_residuals() -> None:
    # Bands are tighter than the empirical residuals -> positive offset.
    actuals = np.array([0.0, 5.0, 10.0, 15.0, 20.0])
    preds = _calib_predictions(
        yhat=[0.0, 0.0, 0.0, 0.0, 0.0],
        lower_80=[-1.0, -1.0, -1.0, -1.0, -1.0],
        upper_80=[1.0, 1.0, 1.0, 1.0, 1.0],
    )
    result = compute_offsets(
        calib_actuals=actuals,
        calib_predictions=preds,
        levels=[80],
    )
    assert 80 in result.offsets
    # 80th percentile of residuals [0, 5, 10, 15, 20] is ~16; minus the
    # current half-width 1.0 -> ~15.
    assert result.offsets[80] > 10


def test_wide_band_offset_zero() -> None:
    # Bands are already huge -> no calibration correction needed.
    actuals = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    preds = _calib_predictions(
        yhat=[0.0, 0.0, 0.0, 0.0, 0.0],
        lower_80=[-100.0, -100.0, -100.0, -100.0, -100.0],
        upper_80=[100.0, 100.0, 100.0, 100.0, 100.0],
    )
    result = compute_offsets(
        calib_actuals=actuals,
        calib_predictions=preds,
        levels=[80],
    )
    assert result.offsets.get(80, 0.0) == 0.0


def test_missing_level_columns_are_skipped() -> None:
    actuals = np.array([0.0, 1.0, 2.0])
    preds = _calib_predictions(
        yhat=[0.0, 0.0, 0.0],
        lower_80=[-1.0, -1.0, -1.0],
        upper_80=[1.0, 1.0, 1.0],
    )
    # 95 isn't present in preds -> should be absent from offsets.
    result = compute_offsets(
        calib_actuals=actuals,
        calib_predictions=preds,
        levels=[80, 95],
    )
    assert 80 in result.offsets
    assert 95 not in result.offsets


def test_apply_offsets_widens_bands_symmetrically() -> None:
    forecast = pd.DataFrame(
        {
            "yhat": [10.0, 11.0, 12.0],
            "yhat_lower_80": [9.0, 10.0, 11.0],
            "yhat_upper_80": [11.0, 12.0, 13.0],
        }
    )
    result = apply_offsets(forecast.copy(), {80: 2.0}, levels=[80])
    assert list(result["yhat"]) == [10.0, 11.0, 12.0]  # yhat untouched
    assert list(result["yhat_lower_80"]) == [7.0, 8.0, 9.0]
    assert list(result["yhat_upper_80"]) == [13.0, 14.0, 15.0]


def test_apply_offsets_zero_is_no_op() -> None:
    forecast = pd.DataFrame(
        {
            "yhat": [10.0],
            "yhat_lower_80": [9.0],
            "yhat_upper_80": [11.0],
        }
    )
    result = apply_offsets(forecast.copy(), {80: 0.0}, levels=[80])
    assert list(result["yhat_lower_80"]) == [9.0]
    assert list(result["yhat_upper_80"]) == [11.0]


def test_holdout_split_returns_tail() -> None:
    df = pd.DataFrame({"y": list(range(100))})
    split = holdout_split(df, holdout_fraction=0.2)
    assert split is not None
    train, calib = split
    assert len(train) == 80
    assert len(calib) == 20
    # Calib should be the last 20 values.
    assert list(calib["y"]) == list(range(80, 100))


def test_holdout_split_short_series_returns_none() -> None:
    df = pd.DataFrame({"y": list(range(8))})
    assert holdout_split(df, holdout_fraction=0.2) is None


def test_empty_inputs_return_zero_samples() -> None:
    result = compute_offsets(
        calib_actuals=np.array([]),
        calib_predictions=pd.DataFrame(),
        levels=[80],
    )
    assert result.offsets == {}
    assert result.samples == 0


def test_offset_is_non_decreasing_in_level() -> None:
    # 95th percentile of residuals >= 80th, so 95% band offset >= 80%.
    rng = np.random.default_rng(seed=99)
    actuals = rng.normal(0.0, 5.0, 200)
    preds = _calib_predictions(
        yhat=[0.0] * 200,
        lower_80=[-1.0] * 200,
        upper_80=[1.0] * 200,
        lower_95=[-2.0] * 200,
        upper_95=[2.0] * 200,
    )
    result = compute_offsets(
        calib_actuals=actuals,
        calib_predictions=preds,
        levels=[80, 95],
    )
    assert result.offsets[95] >= result.offsets[80]


# ---------- runner helpers (snapshot + curve symmetry) --------------------


def test_snapshot_and_curve_calibration_symmetric() -> None:
    """/metrics and remote_write must see the same widened bands.

    Regression test for the bug where conformal offsets were applied to
    the snapshot ``ForecastPoint`` list only, leaving the remote_write
    sink to push uncalibrated bands.
    """
    base_labels = {"model": "AutoARIMA", "level": "80"}
    offsets = {"AutoARIMA": {80: 2.5}}

    snap = [
        ForecastPoint(metric="m_forecast_lower", labels=base_labels, value=10.0),
        ForecastPoint(metric="m_forecast_upper", labels=base_labels, value=20.0),
        ForecastPoint(metric="m_forecast", labels={"model": "AutoARIMA"}, value=15.0),
    ]
    curve = [
        ForecastCurvePoint(
            metric="m_forecast_lower", labels=base_labels, value=10.0, timestamp_ms=1_000
        ),
        ForecastCurvePoint(
            metric="m_forecast_upper", labels=base_labels, value=20.0, timestamp_ms=1_000
        ),
        ForecastCurvePoint(
            metric="m_forecast", labels={"model": "AutoARIMA"}, value=15.0, timestamp_ms=1_000
        ),
    ]

    out_snap = _apply_offsets_to_snapshot_points(snap, offsets)
    out_curve = _apply_offsets_to_curve_points(curve, offsets)

    # Snapshot: lower widened down, upper widened up, calibrated label
    # added on bands, yhat untouched.
    assert out_snap[0].value == 7.5
    assert out_snap[0].labels["calibrated"] == "true"
    assert out_snap[1].value == 22.5
    assert out_snap[1].labels["calibrated"] == "true"
    assert out_snap[2].value == 15.0
    assert "calibrated" not in out_snap[2].labels

    # Curve: same band adjustments, same labelling, timestamp_ms preserved.
    assert out_curve[0].value == 7.5
    assert out_curve[0].labels["calibrated"] == "true"
    assert out_curve[0].timestamp_ms == 1_000
    assert out_curve[1].value == 22.5
    assert out_curve[1].labels["calibrated"] == "true"
    assert out_curve[1].timestamp_ms == 1_000
    assert out_curve[2].value == 15.0
    assert "calibrated" not in out_curve[2].labels


def test_empty_offsets_returns_inputs_unchanged() -> None:
    snap = [
        ForecastPoint(metric="m_forecast_lower", labels={"model": "M", "level": "80"}, value=1.0)
    ]
    curve = [
        ForecastCurvePoint(
            metric="m_forecast_lower",
            labels={"model": "M", "level": "80"},
            value=1.0,
            timestamp_ms=0,
        )
    ]
    assert _apply_offsets_to_snapshot_points(snap, {}) is snap
    assert _apply_offsets_to_curve_points(curve, {}) is curve


def test_zero_offset_skips_calibrated_label() -> None:
    """An offset of 0 means the parametric band already covers the level.

    Don't mark a row ``calibrated="true"`` just because the calibrator
    ran — the label should reflect an actual adjustment, otherwise
    dashboards over-count the calibrated population.
    """
    snap = [
        ForecastPoint(metric="m_forecast_lower", labels={"model": "M", "level": "80"}, value=1.0)
    ]
    out = _apply_offsets_to_snapshot_points(snap, {"M": {80: 0.0}})
    assert out[0].value == 1.0
    assert "calibrated" not in out[0].labels
