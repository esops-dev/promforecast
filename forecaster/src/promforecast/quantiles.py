"""Arbitrary-quantile derivation from a model's confidence-level bands.

The built-in models emit a central forecast plus a symmetric
``yhat_lower_<level>`` / ``yhat_upper_<level>`` band per requested
confidence level. A two-sided level ``L`` brackets the one-sided
quantiles ``(1 - L/100) / 2`` (the lower edge) and ``1 - (1 - L/100) / 2``
(the upper edge) — so any one-sided quantile ``q`` can be read off the
band of the level that places ``q`` on one of its edges, *without
re-fitting the model at a new level*:

* ``q == 0.5`` -> the central forecast (``yhat``).
* ``q  < 0.5`` -> the lower edge of level ``round((1 - 2q) * 100)``.
* ``q  > 0.5`` -> the upper edge of level ``round((2q - 1) * 100)``.

For example ``q = 0.95`` reads ``yhat_upper_90``; ``q = 0.1`` reads
``yhat_lower_80``; ``q = 0.99`` reads ``yhat_upper_98``. The implied
levels are added to the single forward fit's requested level set so the
columns exist to read from — the fit itself is shared.

This mapping is exact for symmetric-interval models and gives genuinely
asymmetric quantiles for the count-distribution models (whose per-level
bands are already asymmetric). Extreme quantiles clamp to the level-99
band edge, which the docs call out.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

# Quantiles within this tolerance of 0.5 are treated as the median (the
# central forecast). A direct float equality would miss a value like
# ``0.5`` that round-tripped through YAML as ``0.49999999999``.
_MEDIAN_TOL = 1e-9
# Levels are clamped to ``[1, 99]``: level 0 collapses the band to the
# point forecast and level 100 is an infinite edge, so quantiles closer to
# the tails than the 99% band can resolve are approximated by that edge.
_MIN_LEVEL = 1
_MAX_LEVEL = 99
# The median splits the lower / upper band-edge mapping.
_MEDIAN = 0.5


def format_quantile(q: float) -> str:
    """Render a quantile as the stable ``q`` label value (e.g. ``0.95``)."""
    return f"{q:g}"


def quantile_column(q: float) -> str:
    """Column name carrying the derived quantile values on a forecast frame."""
    return f"yhat_quantile_{format_quantile(q)}"


def level_and_side_for_quantile(q: float) -> tuple[int, str] | None:
    """Map a one-sided quantile to the ``(level, side)`` band edge it reads.

    Returns ``None`` for the median (which reads ``yhat`` directly).
    ``side`` is ``"lower"`` for ``q < 0.5`` and ``"upper"`` for ``q > 0.5``.
    """
    if abs(q - _MEDIAN) <= _MEDIAN_TOL:
        return None
    if q < _MEDIAN:
        raw = round((1.0 - 2.0 * q) * 100.0)
        side = "lower"
    else:
        raw = round((2.0 * q - 1.0) * 100.0)
        side = "upper"
    level = max(_MIN_LEVEL, min(_MAX_LEVEL, int(raw)))
    return level, side


def quantile_levels(quantiles: list[float]) -> list[int]:
    """Sorted unique confidence levels needed to derive ``quantiles``.

    The median contributes no level (it reads ``yhat``); every other
    quantile contributes the level of the band edge it maps to. The runner
    folds these into the single forward fit's level set so the band
    columns exist to read from.
    """
    levels: set[int] = set()
    for q in quantiles:
        mapped = level_and_side_for_quantile(q)
        if mapped is not None:
            levels.add(mapped[0])
    return sorted(levels)


def predicted_quantile_series(forecast: pd.DataFrame, q: float) -> pd.Series | None:
    """Return the (uncalibrated) per-step values for quantile ``q``.

    Reads ``yhat`` for the median, otherwise the mapped band-edge column.
    Returns ``None`` when the required column is absent (the level wasn't
    in the fit) so the caller can skip the quantile rather than emit
    garbage.
    """
    mapped = level_and_side_for_quantile(q)
    if mapped is None:
        return forecast["yhat"] if "yhat" in forecast.columns else None
    level, side = mapped
    col = f"yhat_{side}_{level}"
    if col not in forecast.columns:
        return None
    return forecast[col]


def add_quantile_columns(
    forecast: pd.DataFrame,
    quantiles: list[float],
    offsets: dict[float, float] | None = None,
) -> list[float]:
    """Add ``yhat_quantile_<q>`` columns to ``forecast`` in place.

    Each column is the band-edge series for that quantile plus its
    (signed) conformal ``offsets`` correction when supplied. Returns the
    list of quantiles that resolved to a column — quantiles whose source
    band column is missing are skipped and excluded from the return so the
    caller emits only the rows it can back with data.
    """
    resolved: list[float] = []
    for q in quantiles:
        base = predicted_quantile_series(forecast, q)
        if base is None:
            continue
        offset = offsets.get(q, 0.0) if offsets else 0.0
        forecast[quantile_column(q)] = base + offset
        resolved.append(q)
    return resolved
