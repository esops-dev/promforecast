"""Split-conformal band calibration.

Standard parametric confidence bands (AutoARIMA, AutoETS, ...) routinely
under-cover on real metric data: a nominal 95% band often empirically
covers only 60-80% of future actuals. Split-conformal prediction wraps
*any* underlying point/interval forecaster and adjusts the bands so the
nominal level matches the empirical level on a held-out tail of the
lookback window:

1. Split the lookback into ``train`` (first ``1 - holdout_fraction``) and
   ``calib`` (last ``holdout_fraction``).
2. Fit the underlying model on ``train`` and forecast ``len(calib)``
   steps ahead.
3. Compute the per-level quantile of absolute residuals (``|actual -
   yhat|``) on the held-out window.
4. Widen the production forecast's bands by the additive correction
   ``offset = max(0, q_level - current_half_width)`` at each level.

The offset itself is exposed as ``forecast_band_calibration_offset`` so an
operator can see exactly how much the bands moved per series/model/level.
Conformal-adjusted forecast points carry a ``calibrated="true"`` label so
dashboards can distinguish corrected vs. uncorrected emissions.

This module only computes; the runner is responsible for wiring the
``forecast_calibration_offset`` gauge and applying ``apply_offsets``
post-fit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd
    from numpy.typing import NDArray


@dataclass(frozen=True)
class CalibrationResult:
    """Per-level additive correction applied to a model's bands."""

    offsets: dict[int, float]
    samples: int


def compute_offsets(
    *,
    calib_actuals: NDArray[np.float64],
    calib_predictions: pd.DataFrame,
    levels: list[int],
) -> CalibrationResult:
    """Compute the additive correction needed at each confidence level.

    ``calib_predictions`` is the model's forecast over the calibration
    window — at minimum it must carry ``yhat`` plus a
    ``yhat_lower_{level}`` / ``yhat_upper_{level}`` pair for each
    requested ``level``. ``calib_actuals`` are the matching observed
    values (same length, same order).

    The returned offset for each level is the smallest non-negative
    quantity that, when added to the current half-width, lifts the
    empirical coverage of the band to at least the nominal level. A
    level whose band is already at or above its nominal empirical
    coverage gets ``offset = 0``.

    Levels with no usable data (degenerate frames, no matching band
    columns) are omitted from the result rather than returned as
    ``NaN``. The caller treats "missing level" as "no calibration this
    run", which is the right behaviour for dashboards.
    """
    if calib_actuals.size == 0 or calib_predictions.empty:
        return CalibrationResult(offsets={}, samples=0)
    yhat_col = "yhat"
    if yhat_col not in calib_predictions.columns:
        return CalibrationResult(offsets={}, samples=0)
    n = min(calib_actuals.size, len(calib_predictions))
    if n < 2:  # noqa: PLR2004 — two points is the bare minimum for a quantile
        return CalibrationResult(offsets={}, samples=n)

    yhat = calib_predictions[yhat_col].to_numpy(dtype=float)[:n]
    actuals = calib_actuals[:n]
    residuals = np.abs(actuals - yhat)
    # Drop non-finite residuals (NaN in either actual or prediction) so
    # the quantile is computed on the well-defined subset.
    finite = np.isfinite(residuals)
    if not finite.any():
        return CalibrationResult(offsets={}, samples=0)
    residuals = residuals[finite]

    offsets: dict[int, float] = {}
    for level in levels:
        lo_col = f"yhat_lower_{level}"
        hi_col = f"yhat_upper_{level}"
        if lo_col not in calib_predictions.columns or hi_col not in calib_predictions.columns:
            continue
        lo = calib_predictions[lo_col].to_numpy(dtype=float)[:n][finite]
        hi = calib_predictions[hi_col].to_numpy(dtype=float)[:n][finite]
        half_widths = (hi - lo) / 2.0
        half_widths = half_widths[np.isfinite(half_widths)]
        # Use the (level / 100) quantile of |residual| — the additive
        # half-band needed for empirical coverage at the nominal level.
        target = max(0.0, min(1.0, level / 100.0))
        q_resid = float(np.quantile(residuals, target))
        avg_half = float(np.mean(half_widths)) if half_widths.size else 0.0
        offset = max(0.0, q_resid - avg_half)
        if math.isfinite(offset):
            offsets[level] = offset
    return CalibrationResult(offsets=offsets, samples=int(residuals.size))


@dataclass(frozen=True)
class QuantileCalibrationResult:
    """Per-quantile *signed* additive correction for conformalised quantiles."""

    offsets: dict[float, float]
    samples: int


def compute_quantile_offsets(
    *,
    calib_actuals: NDArray[np.float64],
    calib_predictions: pd.DataFrame,
    quantiles: list[float],
) -> QuantileCalibrationResult:
    """Conformalise each quantile to its empirical coverage on the holdout.

    For a one-sided quantile ``q`` the calibrated prediction must satisfy
    ``P(actual <= predicted_q + offset) = q`` on the calibration window.
    The empirical solution is ``offset = quantile_q(actual - predicted_q)``
    — the ``q``-quantile of the signed residuals — which is the standard
    additive conformalised-quantile-regression correction. Unlike the
    two-sided band offset this is *signed*: a quantile the model placed too
    high gets a negative correction.

    ``calib_predictions`` is the model's forecast over the calibration
    window carrying the per-level band columns from which each quantile is
    derived. Quantiles whose source band column is absent are omitted from
    the result (the caller treats "missing quantile" as "no calibration
    this run").
    """
    from . import quantiles as quantiles_mod  # noqa: PLC0415

    if calib_actuals.size == 0 or calib_predictions.empty:
        return QuantileCalibrationResult(offsets={}, samples=0)
    n = min(calib_actuals.size, len(calib_predictions))
    if n < 2:  # noqa: PLR2004 — two points is the bare minimum for a quantile
        return QuantileCalibrationResult(offsets={}, samples=n)
    actuals = calib_actuals[:n]
    offsets: dict[float, float] = {}
    samples = 0
    for q in quantiles:
        series = quantiles_mod.predicted_quantile_series(calib_predictions, q)
        if series is None:
            continue
        pred = series.to_numpy(dtype=float)[:n]
        signed = actuals - pred
        finite = np.isfinite(signed)
        if not finite.any():
            continue
        finite_signed = signed[finite]
        offset = float(np.quantile(finite_signed, max(0.0, min(1.0, q))))
        if math.isfinite(offset):
            offsets[q] = offset
            samples = max(samples, int(finite_signed.size))
    return QuantileCalibrationResult(offsets=offsets, samples=samples)


def apply_offsets(
    forecast: pd.DataFrame,
    offsets: dict[int, float],
    *,
    levels: list[int],
) -> pd.DataFrame:
    """Widen the forecast bands in-place by the per-level additive offset.

    Returns the same frame for call-site convenience; the operation is
    in-place. A level with offset == 0 is a no-op. A level missing from
    ``offsets`` is left untouched.
    """
    for level in levels:
        offset = offsets.get(level)
        if offset is None or offset <= 0:
            continue
        lo_col = f"yhat_lower_{level}"
        hi_col = f"yhat_upper_{level}"
        if lo_col in forecast.columns:
            forecast[lo_col] = forecast[lo_col] - offset
        if hi_col in forecast.columns:
            forecast[hi_col] = forecast[hi_col] + offset
    return forecast


def holdout_split(
    df: pd.DataFrame,
    *,
    holdout_fraction: float,
    min_holdout: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Tail-split ``df`` into ``(train, calib)`` for split-conformal.

    Returns ``None`` when the holdout would be smaller than
    ``min_holdout`` — calibrating against a one- or two-point quantile is
    worse than not calibrating at all. The runner treats a ``None`` split
    as "skip calibration for this series this run".
    """
    n = len(df)
    h = int(n * holdout_fraction)
    if h < min_holdout or n - h < min_holdout:
        return None
    train = df.iloc[:-h].reset_index(drop=True)
    calib = df.iloc[-h:].reset_index(drop=True)
    return train, calib
