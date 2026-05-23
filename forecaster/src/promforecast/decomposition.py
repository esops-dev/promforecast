"""Forecast component decomposition.

Implements a model-agnostic decomposition of the forecast curve into
``trend`` / ``seasonal`` / ``level`` / ``residual`` components. Used by
the runner when ``emission.decompose: true`` is set on a query.

The decomposition is **derived from the forecast itself**, not from the
underlying model's internal state, so it works uniformly across the
built-in model registry and any plug-in model whose forecast frame
exposes the standard ``yhat`` column. The trade-off is that the split
between components is a heuristic projection rather than the model's
own attribution — operators get a useful "what is shaping this curve"
visual, not a parameter-level introspection.

Two pieces of structure are recovered:

* **Level** = the in-sample mean (a single number broadcast over the
  horizon). Captures the baseline the forecast oscillates around.
* **Trend** = the linear best-fit slope of the forecast curve over the
  horizon, anchored at zero at the first step so the level isn't
  double-counted.
* **Seasonal** = the deviation from (level + trend) averaged at each
  position within the configured ``season_length`` cycle. Recovers the
  recurring "Mondays are higher" shape.
* **Residual** = what's left after subtracting the other three —
  enforces the contract that ``trend + seasonal + level + residual ==
  yhat`` row-by-row up to floating-point noise.

The contract is enforced by :func:`decompose`; tests assert it on every
fit.

A registry of *non-decomposable* model names is maintained alongside so
the runner can short-circuit the decomposition for, e.g., intermittent
wrappers or neural extensions that don't have a meaningful trend/season
split. Names listed in :data:`NON_DECOMPOSABLE_MODELS` are reported as
``forecast_decomposition_skipped_total{reason="unsupported_model"}``
without an attempt being made.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


COMPONENT_NAMES = ("trend", "seasonal", "level", "residual")


# Models whose forecast shape doesn't admit a meaningful trend/season
# decomposition. Intermittent-demand methods spread sparse mass across
# the horizon — separating that into trend + seasonal would just label
# noise. Operators see the skip via
# ``forecast_decomposition_skipped_total{reason="unsupported_model"}``.
NON_DECOMPOSABLE_MODELS: frozenset[str] = frozenset(
    {
        "Croston",
        "CrostonSBA",
        "ADIDA",
        "IMAPA",
        "ColdStart",
    }
)


@dataclass(frozen=True)
class DecompositionResult:
    """Per-component value lists, all aligned to the forecast frame's rows.

    ``components`` keys are exactly :data:`COMPONENT_NAMES`. Each list
    has length equal to the input forecast's row count.
    """

    components: dict[str, list[float]]


class DecompositionError(RuntimeError):
    """Raised when the forecast frame can't be decomposed.

    Carries a short ``reason`` token that becomes a label value on
    ``forecast_decomposition_skipped_total``.
    """

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


def is_decomposable(model_name: str) -> bool:
    """Return whether ``model_name`` is in the decomposable registry."""
    return model_name not in NON_DECOMPOSABLE_MODELS


def decompose(
    *,
    forecast: pd.DataFrame,
    in_sample_values: list[float],
    season_length: int,
) -> DecompositionResult:
    """Project a forecast frame onto trend/seasonal/level/residual components.

    Args:
        forecast: The forecast frame produced by ``model.fit_predict``.
            Must contain a ``yhat`` column with one value per horizon
            step.
        in_sample_values: The training-side ``y`` values used to
            establish the baseline level. The mean of this list becomes
            the ``level`` component (broadcast over the horizon).
        season_length: Number of forecast steps per cycle. Used to fold
            the (yhat - level - trend) residual into per-position
            seasonal averages.

    Returns a :class:`DecompositionResult` whose component lists each
    match the forecast frame's row count.

    Raises :class:`DecompositionError` with a bounded ``reason`` token
    when the inputs can't be decomposed. The runner translates the
    token into a label value on
    ``forecast_decomposition_skipped_total``.
    """
    if forecast.empty or "yhat" not in forecast.columns:
        raise DecompositionError(
            "components_unavailable",
            "forecast frame is empty or missing the 'yhat' column",
        )
    yhat = [float(v) for v in forecast["yhat"].tolist()]
    n = len(yhat)
    if n == 0:
        raise DecompositionError(
            "components_unavailable",
            "forecast frame has zero rows",
        )

    # Level: in-sample mean. Falls back to the forecast mean when no
    # in-sample data is supplied (the runner always supplies it; the
    # branch is defensive for direct callers).
    level_value = _finite_mean(in_sample_values) if in_sample_values else _finite_mean(yhat)
    level = [level_value] * n

    # Trend: linear best-fit slope across the horizon, anchored at zero
    # on step 0 so the average isn't double-counted with ``level``.
    slope = _least_squares_slope(yhat)
    trend = [slope * i for i in range(n)]

    # Seasonal: average residual at each position within the cycle of
    # ``season_length``. Steps that share a position contribute to the
    # same bucket; the bucket mean becomes the seasonal value for every
    # step at that position. A non-positive season collapses to zero so
    # the residual component picks up everything (correct contract).
    detrended = [yhat[i] - level[i] - trend[i] for i in range(n)]
    if season_length and season_length > 0:
        buckets: list[list[float]] = [[] for _ in range(season_length)]
        for i, v in enumerate(detrended):
            if math.isfinite(v):
                buckets[i % season_length].append(v)
        bucket_means = [sum(b) / len(b) if b else 0.0 for b in buckets]
        seasonal = [bucket_means[i % season_length] for i in range(n)]
    else:
        seasonal = [0.0] * n

    residual = [yhat[i] - level[i] - trend[i] - seasonal[i] for i in range(n)]

    components = {
        "trend": trend,
        "seasonal": seasonal,
        "level": level,
        "residual": residual,
    }
    _enforce_sum_invariant(yhat=yhat, components=components)
    return DecompositionResult(components=components)


# Tolerances for the sum-to-yhat invariant. The closed-form math is exact
# under IEEE-754 once you account for the trailing residual subtraction;
# anything beyond a few ULPs scaled by the row magnitude is a logic bug,
# not float noise. The check is the *larger* of an absolute floor
# (handles rows near zero where relative tolerance would be too strict)
# and a relative band (handles large-magnitude metrics like disk bytes
# where the absolute floor would falsely trip even though the math is
# correct to nine significant digits). A pure absolute tolerance was the
# original implementation but produced false ``invariant_violation``
# skips on legitimate large-magnitude forecasts.
_INVARIANT_ABS_TOLERANCE = 1e-6
_INVARIANT_REL_TOLERANCE = 1e-9


def _enforce_sum_invariant(*, yhat: list[float], components: dict[str, list[float]]) -> None:
    """Validate that ``sum(components, row=i) ≈ yhat[i]`` for every row.

    The runner relies on this to keep the stacked-area panel honest. A
    component list that drifts off the invariant silently would be worse
    than emitting nothing — operators would attribute the missing mass to
    the wrong cause. We raise instead so the caller emits the skip
    counter (``reason="invariant_violation"``) and the family stays off
    /metrics for the offending fit.

    The tolerance is ``max(abs_tol, rel_tol * |yhat|)`` so the check
    holds both on near-zero rows (where relative would be too strict)
    and on large-magnitude metrics like disk bytes (where a fixed
    absolute floor would falsely trip on harmless float noise).
    """
    summed = [0.0] * len(yhat)
    for values in components.values():
        for i, v in enumerate(values):
            summed[i] += v
    for i, (s, y) in enumerate(zip(summed, yhat, strict=True)):
        if not (math.isfinite(s) and math.isfinite(y)):
            continue
        tolerance = max(_INVARIANT_ABS_TOLERANCE, _INVARIANT_REL_TOLERANCE * abs(y))
        if abs(s - y) > tolerance:
            raise DecompositionError(
                "invariant_violation",
                f"component sum diverges from yhat at row {i}: "
                f"|{s} - {y}| = {abs(s - y)} > {tolerance}",
            )


def reconstruct(components: dict[str, list[float]]) -> list[float]:
    """Sum the components row-by-row.

    Helper used by tests to assert
    ``sum(components, row=i) ≈ yhat[i]`` for every ``i``.
    """
    if not components:
        return []
    rows = max((len(vs) for vs in components.values()), default=0)
    out = [0.0] * rows
    for values in components.values():
        for i in range(rows):
            if i < len(values):
                out[i] += values[i]
    return out


def _finite_mean(values: list[float]) -> float:
    """Mean over the finite entries; 0.0 when none are finite."""
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return 0.0
    return sum(finite) / len(finite)


def _least_squares_slope(values: list[float]) -> float:
    """Closed-form OLS slope of ``values`` regressed on row index.

    Returns 0 when the input is too short or every row is non-finite,
    so the trend component degrades silently to "no slope" rather than
    raising.
    """
    pairs = [(float(i), float(v)) for i, v in enumerate(values) if math.isfinite(v)]
    if len(pairs) < 2:  # noqa: PLR2004
        return 0.0
    n = float(len(pairs))
    sum_x = sum(x for x, _ in pairs)
    sum_y = sum(y for _, y in pairs)
    sum_xy = sum(x * y for x, y in pairs)
    sum_xx = sum(x * x for x, _ in pairs)
    denom = n * sum_xx - sum_x * sum_x
    if denom == 0:
        return 0.0
    return (n * sum_xy - sum_x * sum_y) / denom
