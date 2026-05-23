"""Shared time-axis helper for forecast-frame post-processing.

Both :mod:`promforecast.capacity` (threshold ETAs, drain ETAs, growth
rates) and :mod:`promforecast.maintenance` (recommended windows) need to
project a forecast's ``ds`` column onto seconds-from-the-leading-row.
The conversion is identical, defensively handles malformed timestamps
the same way, and lives here so the two callers can stay in sync.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd


def seconds_from_start(ds: pd.Series[Any]) -> list[float] | None:
    """Return per-row seconds since the leading timestamp, or ``None``.

    Anchors at row 0 because the operator's mental model for everything
    derived from a forecast curve is "time from now", and "now" is the
    moment the forecast was issued. ``None`` is returned if the input is
    empty or any row carries a value that doesn't quack like a datetime
    (no ``.timestamp()``), so the caller can treat the failure mode the
    same way an empty forecast is treated.
    """
    if ds.empty:
        return None
    try:
        start_ts = float(ds.iloc[0].timestamp())
    except (AttributeError, ValueError, TypeError):
        return None
    out: list[float] = []
    for item in ds.tolist():
        try:
            out.append(float(item.timestamp()) - start_ts)
        except (AttributeError, ValueError, TypeError):
            return None
    return out
