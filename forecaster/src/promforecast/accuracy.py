"""Backtesting and multi-horizon accuracy metrics.

A single hold-out window is used: the most recent ``max_horizon_steps`` of
the lookback window are held out as the evaluation set; the model fits on
the prefix and predicts those held-out points. We compute MAPE and MASE per
configured horizon by slicing the prediction to the first N steps for each
horizon (``"1h"`` -> first 12 steps at 5-min cadence, etc.).

A single window keeps the cost predictable (one extra fit per series per
configured model) and avoids the trap where multiple windows make a slow
``AutoARIMA`` fit dominate the per-group budget. statsforecast's own
``cross_validation`` is more rigorous but ~N_windows times more expensive; if
benchmarks later show that's worthwhile we can swap implementations behind
this module's API without touching the runner.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from .models import ForecastModel


@dataclass(frozen=True)
class AccuracyResult:
    """Per-(model, horizon) backtest output for a single series."""

    model: str
    horizon: str
    mape: float
    mase: float


@dataclass(frozen=True)
class HorizonSpec:
    """A configured accuracy horizon plus its derived step count."""

    label: str
    steps: int


def horizons_from_durations(durations: list[timedelta], step: timedelta) -> list[HorizonSpec]:
    """Convert ``accuracy_horizons`` durations to (label, n_steps) pairs.

    Horizons that round to 0 steps (because they're shorter than ``step``)
    are dropped — emitting a 0-step accuracy metric is meaningless and would
    just produce NaNs.
    """
    step_seconds = max(1, int(step.total_seconds()))
    out: list[HorizonSpec] = []
    for dur in durations:
        n = int(dur.total_seconds() // step_seconds)
        if n < 1:
            continue
        out.append(HorizonSpec(label=_format_horizon(dur), steps=n))
    return out


def _format_horizon(d: timedelta) -> str:
    """Render a timedelta as a Prometheus-style duration label.

    Hours are preferred over days up to a week — ``24h`` is more idiomatic
    in monitoring contexts than ``1d``, and matches the canonical example
    horizon labels (``"1h"``, ``"6h"``, ``"24h"``).
    """
    seconds = int(d.total_seconds())
    week_seconds = 7 * 86400
    if seconds >= week_seconds and seconds % 86400 == 0:
        return f"{seconds // 86400}d"
    if seconds % 3600 == 0:
        return f"{seconds // 3600}h"
    if seconds % 60 == 0:
        return f"{seconds // 60}m"
    return f"{seconds}s"


def backtest(
    model: ForecastModel,
    df: pd.DataFrame,
    horizon_steps: int,
    freq: str,
    levels: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Run a single hold-out backtest.

    Returns ``(test_actuals, predictions)`` aligned 1:1 in row order, or
    ``None`` if the series is too short for a meaningful split.
    """
    if horizon_steps < 1 or len(df) <= horizon_steps + 1:
        return None
    train = df.iloc[:-horizon_steps].reset_index(drop=True)
    test = df.iloc[-horizon_steps:].reset_index(drop=True)
    predictions = model.fit_predict(df=train, horizon=horizon_steps, freq=freq, levels=levels)
    return test, predictions


def evaluate(
    test: pd.DataFrame,
    predictions: pd.DataFrame,
    train: pd.DataFrame,
    season_length: int,
    horizons: list[HorizonSpec],
    model_name: str,
) -> list[AccuracyResult]:
    """Compute per-horizon MAPE/MASE from a backtest result."""
    if predictions.empty or test.empty:
        return []
    y_true = test["y"].to_numpy(dtype=float)
    y_pred = predictions["yhat"].to_numpy(dtype=float)
    train_y = train["y"].to_numpy(dtype=float)
    n = min(len(y_true), len(y_pred))
    if n == 0:
        return []
    y_true = y_true[:n]
    y_pred = y_pred[:n]
    scale = _mase_scale(train_y, season_length)
    out: list[AccuracyResult] = []
    for h in horizons:
        k = min(h.steps, n)
        if k < 1:
            continue
        out.append(
            AccuracyResult(
                model=model_name,
                horizon=h.label,
                mape=_mape(y_true[:k], y_pred[:k]),
                mase=_mase(y_true[:k], y_pred[:k], scale),
            )
        )
    return out


def overall_mape(results: list[AccuracyResult]) -> float | None:
    """Mean MAPE across horizons, ignoring NaNs.

    Used by the auto-selector as a single-number summary per model. We treat
    horizons equally; weighting by horizon length would bias toward the
    longest horizon (where every model tends to do worst).
    """
    finite = [r.mape for r in results if math.isfinite(r.mape)]
    if not finite:
        return None
    return float(sum(finite) / len(finite))


def _mape(y_true: object, y_pred: object) -> float:
    import numpy as np  # noqa: PLC0415

    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = yt != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])) * 100.0)


def _mase(y_true: object, y_pred: object, scale: float) -> float:
    import numpy as np  # noqa: PLC0415

    if not math.isfinite(scale) or scale == 0:
        return float("nan")
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(yt - yp)) / scale)


def _mase_scale(train_y: object, season_length: int) -> float:
    """Mean absolute seasonal-naive in-sample error.

    Falls back to the non-seasonal naive (lag-1) when the training window is
    shorter than the seasonality, which is what most reference
    implementations do.
    """
    import numpy as np  # noqa: PLC0415

    arr = np.asarray(train_y, dtype=float)
    min_points_for_diff = 2
    if len(arr) < min_points_for_diff:
        return float("nan")
    m = season_length if 0 < season_length < len(arr) else 1
    diffs = np.abs(arr[m:] - arr[:-m])
    if diffs.size == 0:
        return float("nan")
    return float(np.mean(diffs))
