"""Tests for the backtest + accuracy module.

The ``backtest`` step is exercised via a fake ``ForecastModel`` so the test
matrix doesn't depend on statsforecast.
"""

from __future__ import annotations

import math
from datetime import timedelta

import pandas as pd

from promforecast.accuracy import (
    HorizonSpec,
    backtest,
    evaluate,
    horizons_from_durations,
    overall_mape,
)


class _ConstantModel:
    """Predict a single constant — useful for hand-checked MAPE/MASE."""

    name = "Const"

    def __init__(self, value: float) -> None:
        self._value = value

    def fit_predict(
        self,
        df: pd.DataFrame,
        horizon: int,
        freq: str,
        levels: list[int],
        regressors: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        idx = pd.date_range(start=df["ds"].iloc[-1], periods=horizon + 1, freq=freq)[1:]
        return pd.DataFrame(
            {
                "unique_id": ["s"] * horizon,
                "ds": idx,
                "yhat": [self._value] * horizon,
            }
        )


def _series(values: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unique_id": ["s"] * len(values),
            "ds": pd.date_range("2024-01-01", periods=len(values), freq="5min"),
            "y": values,
        }
    )


def test_horizons_from_durations_drops_subsecond_horizons() -> None:
    out = horizons_from_durations(
        [timedelta(hours=1), timedelta(hours=6), timedelta(seconds=10)],
        step=timedelta(minutes=5),
    )
    assert [h.label for h in out] == ["1h", "6h"]
    assert [h.steps for h in out] == [12, 72]


def test_backtest_returns_none_when_series_too_short() -> None:
    df = _series([1.0, 2.0, 3.0])
    out = backtest(_ConstantModel(0.0), df, horizon_steps=10, freq="5min", levels=[80])
    assert out is None


def test_backtest_holds_out_last_n() -> None:
    df = _series([float(i) for i in range(30)])
    result = backtest(_ConstantModel(99.0), df, horizon_steps=5, freq="5min", levels=[80])
    assert result is not None
    test, predictions = result
    assert list(test["y"]) == [25.0, 26.0, 27.0, 28.0, 29.0]
    assert list(predictions["yhat"]) == [99.0] * 5


def test_evaluate_computes_mape_and_mase() -> None:
    train = _series([10.0] * 100)
    # Actuals = 10, prediction = 11 -> MAPE = 10%, MAE = 1
    test = pd.DataFrame({"y": [10.0] * 12})
    predictions = pd.DataFrame({"yhat": [11.0] * 12})
    horizons = [HorizonSpec(label="1h", steps=12)]
    out = evaluate(
        test=test,
        predictions=predictions,
        train=train,
        season_length=12,
        horizons=horizons,
        model_name="Const",
    )
    assert len(out) == 1
    assert out[0].model == "Const"
    assert out[0].horizon == "1h"
    assert math.isclose(out[0].mape, 10.0, rel_tol=1e-6)
    # Constant series → seasonal-naive scale is 0 → MASE is NaN by definition.
    assert math.isnan(out[0].mase)


def test_evaluate_per_horizon_slicing() -> None:
    train = _series([float(i % 10) for i in range(100)])
    # Actuals are linearly increasing; prediction is off by 1 always.
    test = pd.DataFrame({"y": [10.0 + i for i in range(72)]})
    predictions = pd.DataFrame({"yhat": [11.0 + i for i in range(72)]})
    horizons = [HorizonSpec(label="1h", steps=12), HorizonSpec(label="6h", steps=72)]
    out = evaluate(
        test=test,
        predictions=predictions,
        train=train,
        season_length=10,
        horizons=horizons,
        model_name="Const",
    )
    by_horizon = {r.horizon: r for r in out}
    # 1h MAPE averages |1/10|..|1/21| ≈ ~6.5% — closer-in MAPE is larger.
    assert by_horizon["1h"].mape > by_horizon["6h"].mape


def test_evaluate_returns_empty_for_empty_predictions() -> None:
    out = evaluate(
        test=pd.DataFrame({"y": []}),
        predictions=pd.DataFrame({"yhat": []}),
        train=_series([1.0, 2.0, 3.0]),
        season_length=2,
        horizons=[HorizonSpec(label="1h", steps=12)],
        model_name="X",
    )
    assert out == []


def test_overall_mape_averages_finite_only() -> None:
    from promforecast.accuracy import AccuracyResult  # noqa: PLC0415

    results = [
        AccuracyResult(model="X", horizon="1h", mape=10.0, mase=1.0),
        AccuracyResult(model="X", horizon="6h", mape=20.0, mase=1.0),
        AccuracyResult(model="X", horizon="24h", mape=math.nan, mase=math.nan),
    ]
    assert overall_mape(results) == 15.0


def test_overall_mape_returns_none_when_all_nan() -> None:
    from promforecast.accuracy import AccuracyResult  # noqa: PLC0415

    results = [AccuracyResult(model="X", horizon="1h", mape=math.nan, mase=math.nan)]
    assert overall_mape(results) is None
