"""Composite forecast quality / reliability score.

Combines backtest MAPE, confidence-band width, and data coverage into a
single 0..1 score.
"""

from __future__ import annotations


def compute(
    backtest_mape: float,
    band_width_relative: float,
    coverage: float,
) -> float:
    """Return a quality score in [0.0, 1.0].

    Args:
        backtest_mape: Cross-validation MAPE percentage (lower is better).
        band_width_relative: Confidence-band width relative to actual values.
        coverage: Fraction of expected points present during the lookback
            window.
    """
    raise NotImplementedError
