"""Model selection: explicit, auto-select, ensemble, fallback."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SelectionResult:
    """Outcome of selecting a model (or ensemble) for one series."""

    model: str
    fallback_applied: bool
    mape: float | None


def select(
    models: list[str],
    auto_select: bool | str,
    fallback_model: str,
    fallback_mape_threshold: float | None,
    backtest_mape: dict[str, float],
) -> SelectionResult:
    """Pick a model (or ensemble) for one series given backtest results."""
    raise NotImplementedError
