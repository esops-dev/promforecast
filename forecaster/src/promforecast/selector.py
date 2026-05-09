"""Model selection: explicit, auto-select, ensemble, fallback.

Auto-select picks the lowest-MAPE model from the configured set per series.
If the winner's backtest MAPE exceeds ``fallback_mape_threshold`` the
selector switches to ``fallback_model`` and flags ``fallback_applied``;
this prevents a model that ran a bad horse race from being treated as
authoritative just because it "won".
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SelectionResult:
    """Outcome of selecting a model for one series."""

    model: str
    best_model: str
    fallback_applied: bool
    mape: float | None


def select(
    fallback_model: str,
    fallback_mape_threshold: float | None,
    backtest_mape: dict[str, float],
) -> SelectionResult:
    """Pick a model for one series given backtest results.

    With no usable backtest scores (every fit failed, or every model returned
    NaN MAPE), the selector falls back unconditionally — emitting nothing
    would silently drop the series, which is harder to debug than a
    fallback-tagged forecast.
    """
    finite = {m: v for m, v in backtest_mape.items() if math.isfinite(v)}
    if not finite:
        return SelectionResult(
            model=fallback_model, best_model=fallback_model, fallback_applied=True, mape=None
        )

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
