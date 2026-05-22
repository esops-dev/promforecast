"""Model selection: explicit, auto-select, ensemble, fallback.

Auto-select picks the lowest-MAPE model from the configured set per series.
If the winner's backtest MAPE exceeds ``fallback_mape_threshold`` the
selector switches to ``fallback_model`` and flags ``fallback_applied``;
this prevents a model that ran a bad horse race from being treated as
authoritative just because it "won".

Ensemble mode (``auto_select: ensemble``) is a separate code path that
combines forecasts from every configured model, weighted by inverse-MAPE.
The two modes share backtest data but produce different sets of output
points; see :mod:`promforecast.runner` for the dispatch.

For ``data_profile: intermittent`` series MAPE is undefined wherever the
actual is zero — which is precisely the regime that profile targets.
The runner swaps the score map for MASE in that case and calls
:func:`select` exactly the same way; the comparator below is
unit-agnostic, so the change is a single-line plumbing decision on the
caller's side.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SelectionResult:
    """Outcome of selecting a model for one series.

    ``mape`` carries the winning model's score in the comparator unit
    the caller chose — MAPE in the default path, MASE for intermittent
    profiles. The field is named after MAPE for backwards-compatibility
    of consumers (label values on emitted points), but the magnitude is
    only meaningful relative to the unit the caller passed.
    """

    model: str
    best_model: str
    fallback_applied: bool
    mape: float | None


def select(
    fallback_model: str,
    fallback_mape_threshold: float | None,
    backtest_mape: dict[str, float],
    *,
    candidate_preference: frozenset[str] | None = None,
) -> SelectionResult:
    """Pick a model for one series given backtest results.

    ``backtest_mape`` may carry MAPE or MASE depending on the caller's
    chosen scoring unit; the comparator is unit-agnostic — the lowest
    finite score wins.

    With no usable backtest scores (every fit failed, or every model
    returned NaN), the selector falls back unconditionally — emitting
    nothing would silently drop the series, which is harder to debug
    than a fallback-tagged forecast.

    ``candidate_preference`` is an optional whitelist of model names the
    caller would prefer the winner to come from. When at least one
    finite-scored candidate matches the whitelist, the selector
    restricts the search to that subset; otherwise it falls back to
    ranking across every model (so a misconfigured intermittent profile
    that lists no intermittent models still produces a forecast). Used
    by ``data_profile: intermittent`` to prefer the dedicated
    intermittent-demand methods when the operator has registered any.
    """
    finite = {m: v for m, v in backtest_mape.items() if math.isfinite(v)}
    if not finite:
        return SelectionResult(
            model=fallback_model, best_model=fallback_model, fallback_applied=True, mape=None
        )

    if candidate_preference:
        preferred = {m: v for m, v in finite.items() if m in candidate_preference}
        if preferred:
            finite = preferred

    winner = min(finite, key=lambda m: finite[m])
    winner_mape = finite[winner]

    if fallback_mape_threshold is not None and winner_mape > fallback_mape_threshold:
        return SelectionResult(
            model=fallback_model,
            best_model=winner,
            fallback_applied=True,
            mape=winner_mape,
        )

    return SelectionResult(
        model=winner, best_model=winner, fallback_applied=False, mape=winner_mape
    )


def ensemble_weights(backtest_mape: dict[str, float]) -> dict[str, float]:
    """Inverse-MAPE weights for the ensemble combiner.

    A model with MAPE ``m`` gets raw weight ``1 / max(m, eps)`` (the floor
    keeps a perfect-fit MAPE of 0 from blowing up to infinity and dominating
    the ensemble), then we normalise so the weights sum to 1. Models with
    NaN/non-finite MAPE are dropped — they would inject NaNs into the
    weighted average. If *every* model is non-finite we fall back to equal
    weights so the ensemble still has something to combine; that case
    typically coincides with degenerate series and the resulting forecast
    is wrong either way, but at least the pipeline doesn't silently lose
    the series.

    Returns an empty dict when ``backtest_mape`` is empty.
    """
    if not backtest_mape:
        return {}
    eps = 1e-9
    finite = {m: max(v, eps) for m, v in backtest_mape.items() if math.isfinite(v)}
    if not finite:
        # Every model returned NaN. Equal weights across the original set
        # keeps the ensemble well-defined; the caller flags this via the
        # standard accuracy metrics so it's still observable.
        n = len(backtest_mape)
        return {m: 1.0 / n for m in backtest_mape}
    raw = {m: 1.0 / v for m, v in finite.items()}
    total = sum(raw.values())
    return {m: w / total for m, w in raw.items()}
