"""Composite forecast quality / reliability score.

The score is a single number in ``[0, 1]`` that combines three sub-scores
multiplicatively, computed once per fit per series:

1. ``accuracy_score``: depends on the chosen scoring unit.

   * ``score_kind="mape"`` (the default for continuous series):
     ``max(0, 1 - mape/100)`` clamped to ``[0, 1]``. A perfect backtest
     (MAPE = 0) scores 1; a 100%+ MAPE scores 0.
   * ``score_kind="mase"`` (used by ``data_profile: intermittent``,
     where MAPE is undefined wherever actuals are zero):
     ``1 / (1 + max(0, mase))``. MASE = 0 (perfect) maps to 1.0; MASE
     = 1 (matches the seasonal-naive baseline) maps to 0.5; larger
     MASE values decay smoothly toward 0 without ever crossing
     it — a noisy intermittent series still contributes a non-zero
     quality.

2. ``band_score``: ``1 / (1 + band_width_relative)``. Penalises wide bands —
   a band whose width equals the typical actual value scores 0.5; an
   infinitely wide band approaches 0.
3. ``coverage_score``: the data-coverage fraction itself, clamped to
   ``[0, 1]``, scaled by ``(1 - imputed_fraction)`` so heavily-imputed
   series carry a proportional penalty. A fully-covered lookback with no
   imputation scores 1.0; a fully-covered lookback whose values are all
   forward-fills scores 0.0.

The product (``accuracy_score * band_score * coverage_score``) is used so
that any one weak dimension drags the overall score down: wide-but-
accurate, or narrow-band-but-poor-MAPE, both deserve scepticism.

Inputs are expected from the runner. Non-finite or missing values short to
0 — a series with no MAPE / no data has no quality, by definition.
"""

from __future__ import annotations

import math
from typing import Literal

ScoreKind = Literal["mape", "mase"]


def compute(
    backtest_mape: float | None,
    band_width_relative: float | None,
    coverage: float | None,
    imputed_fraction: float | None = None,
    score_kind: ScoreKind = "mape",
) -> float:
    """Return a quality score in ``[0.0, 1.0]``.

    Args:
        backtest_mape: Cross-validation accuracy score (lower is better).
            Interpretation depends on ``score_kind``: a percentage when
            ``"mape"``, a raw scaled-error number when ``"mase"``.
            ``None`` or non-finite -> accuracy sub-score 0.
        band_width_relative: Confidence-band width relative to the magnitude
            of the actual values; ``(yhat_upper - yhat_lower) / mean(|y|)``.
            ``None`` or non-finite -> band sub-score 0.
        coverage: Fraction of expected points present during the lookback
            window, in ``[0, 1]``. ``None`` -> coverage sub-score 0.
        imputed_fraction: Fraction of the coverage that was synthesised by
            gap imputation (forward-fill / linear interpolation). The
            coverage sub-score is multiplied by ``(1 - imputed_fraction)``
            so a series whose "coverage" is entirely fabricated does not
            quietly inherit a high quality score. ``None`` is treated as
            ``0`` for backward compatibility with callers that don't run
            preprocessing.
        score_kind: Which accuracy unit ``backtest_mape`` carries. Use
            ``"mase"`` for intermittent / count series where MAPE is
            undefined on zero actuals; the default ``"mape"`` is the
            continuous-series path.

    Returns:
        ``accuracy_score * band_score * coverage_score`` clamped to ``[0, 1]``.
    """
    accuracy = _mase_score(backtest_mape) if score_kind == "mase" else _mape_score(backtest_mape)
    return accuracy * _band_score(band_width_relative) * _coverage_score(coverage, imputed_fraction)


def _mape_score(mape: float | None) -> float:
    if mape is None or not math.isfinite(mape):
        return 0.0
    return max(0.0, min(1.0, 1.0 - mape / 100.0))


def _mase_score(mase: float | None) -> float:
    """Map MASE in ``[0, +inf)`` to a sub-score in ``(0, 1]``.

    Uses ``1 / (1 + mase)`` so MASE = 1 (the seasonal-naive baseline) lands
    on exactly 0.5 — a memorable midpoint that operators can read as
    "this model is no better than seasonal naive". Smaller MASE rewards
    smoothly toward 1; larger MASE decays toward 0 without ever quite
    crossing it. The function is monotonic so the comparator agrees with
    the auto-select ranking. Non-finite / negative values short to 0.
    """
    if mase is None or not math.isfinite(mase) or mase < 0:
        return 0.0
    return 1.0 / (1.0 + mase)


def _band_score(band_width_relative: float | None) -> float:
    if band_width_relative is None or not math.isfinite(band_width_relative):
        return 0.0
    if band_width_relative < 0:
        return 0.0
    return 1.0 / (1.0 + band_width_relative)


def _coverage_score(coverage: float | None, imputed_fraction: float | None) -> float:
    if coverage is None or not math.isfinite(coverage):
        return 0.0
    base = max(0.0, min(1.0, coverage))
    if imputed_fraction is None or not math.isfinite(imputed_fraction):
        return base
    penalty = max(0.0, min(1.0, imputed_fraction))
    return base * (1.0 - penalty)
