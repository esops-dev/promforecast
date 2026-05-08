"""Backtesting and multi-horizon accuracy metrics.

Emits ``forecast_accuracy_mape`` and ``forecast_accuracy_mase`` per series
per model per horizon (1h, 6h, 24h by default).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AccuracyResult:
    horizon: str
    mape: float
    mase: float


def evaluate(
    actuals: list[float],
    predictions: list[float],
    seasonality: int,
) -> list[AccuracyResult]:
    """Cross-validate forecasts and return per-horizon accuracy results."""
    raise NotImplementedError
