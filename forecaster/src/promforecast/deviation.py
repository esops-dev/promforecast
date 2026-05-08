"""Actual-vs-forecast deviation metrics.

Emits ``forecast_deviation_ratio`` and ``forecast_deviation_outside_band``
when ``emit_deviation: true``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Deviation:
    ratio: float
    outside_band: bool


def compute(actual: float, yhat_lower: float, yhat_upper: float) -> Deviation:
    """Compute deviation metrics for a single (actual, forecast band) pair."""
    raise NotImplementedError
