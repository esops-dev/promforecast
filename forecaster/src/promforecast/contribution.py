"""Per-regressor forecast contribution / explainability metric.

We expose a single number per (series, regressor): a normalised share of
the absolute Pearson correlation between the regressor's in-sample values
and the primary series' actuals. The shares for a given series sum to 1
when at least one regressor produced a finite correlation.

Why correlation rather than a model-derived coefficient? The built-in
``statsforecast`` models don't surface per-regressor coefficients, and we
deliberately keep the registry pluggable: any meaningful "this regressor
contributed X% of the forecast" needs the regressor signal anyway, so a
data-only proxy gives operators a useful first-order view today and
generalises to any model. Plug-in models that *do* expose coefficients
can override this in a future revision; the metric name is stable.

Failure modes are pessimistic: NaN correlation (constant regressor or
constant target) becomes a 0 contribution rather than dragging the rest
of the share down. If every regressor returns NaN we emit 0 for all of
them so the metric is still present on /metrics — operators alerting on
"contribution dropped to 0" want a sample, not a missing series.

The math runs vectorised through numpy. With high-cardinality groups
(thousands of series, thousands of points, multiple regressors) a pure
Python implementation dominated the CPU profile of a refresh during
initial benchmarking; numpy keeps the contribution computation well
under the per-fit cost of any model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


_MIN_PEARSON_SAMPLES = 2


@dataclass(frozen=True)
class ContributionResult:
    """Normalised share + raw absolute correlation for one regressor."""

    regressor_id: str
    ratio: float
    correlation: float


def compute_contributions(
    y: list[float],
    regressors: pd.DataFrame | None,
) -> list[ContributionResult]:
    """Return one :class:`ContributionResult` per regressor column.

    ``y`` is the primary series' historical actuals; ``regressors`` is the
    aligned per-series exogenous frame (already produced by
    :func:`promforecast.regressors.align_for_series`). The ``ds`` column is
    skipped — every other column is treated as a regressor.

    The result list preserves the column order of ``regressors`` so callers
    can correlate it against the schema's regressor order if needed; metric
    rendering only relies on the ``regressor_id`` label.
    """
    if regressors is None or regressors.empty or not y:
        return []
    columns = [c for c in regressors.columns if c != "ds"]
    if not columns:
        return []
    y_arr = np.asarray(y, dtype=float)
    raw: list[tuple[str, float]] = []
    for col in columns:
        x_arr = regressors[col].to_numpy(dtype=float, copy=False)
        raw.append((col, _abs_pearson(x_arr, y_arr)))
    finite_total = float(sum(c for _, c in raw if np.isfinite(c)))
    if finite_total <= 0:
        # No regressor produced a usable correlation — emit zeros so the
        # metric is still present (alerts on "contribution = 0" continue
        # to fire) but the ratios don't sum to 1.
        return [ContributionResult(regressor_id=col, ratio=0.0, correlation=0.0) for col, _ in raw]
    return [
        ContributionResult(
            regressor_id=col,
            ratio=float(c / finite_total) if np.isfinite(c) else 0.0,
            correlation=float(c) if np.isfinite(c) else 0.0,
        )
        for col, c in raw
    ]


def _abs_pearson(x: np.ndarray[Any, Any], y: np.ndarray[Any, Any]) -> float:
    """Absolute Pearson correlation, NaN-safe and vectorised.

    Returns ``nan`` when either side is constant after masking, when the
    inputs disagree on length, or when fewer than two finite pairs remain
    — all signals that the contribution is undefined rather than zero, and
    the caller's branch on ``finite_total`` keeps that distinction visible.
    """
    n = int(min(x.shape[0], y.shape[0]))
    if n < _MIN_PEARSON_SAMPLES:
        return float("nan")
    x = x[:n]
    y = y[:n]
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < _MIN_PEARSON_SAMPLES:
        return float("nan")
    x = x[mask]
    y = y[mask]
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    den = float(np.sqrt(float((x_centered * x_centered).sum()))) * float(
        np.sqrt(float((y_centered * y_centered).sum()))
    )
    if den == 0:
        return float("nan")
    return float(abs(float((x_centered * y_centered).sum()) / den))
