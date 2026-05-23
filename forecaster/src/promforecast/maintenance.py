"""Recommended-maintenance-window picker.

Given a forecast curve and a ``(duration, lookahead, count)`` request,
return the top-N predicted-lowest-load windows of ``duration`` within
the next ``lookahead`` of the forecast horizon. Each candidate window
is scored by the *mean* of the forecast values inside it; the lowest
mean wins.

Pure-function implementation — the runner reaches for this once per
emitted (model, series) when ``recommended_windows`` is configured on
the query, and the math is unit-testable without a full Runner fixture.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from ._time_axis import seconds_from_start as _seconds_from_start

if TYPE_CHECKING:
    import pandas as pd

    from .config import RecommendedWindowsConfig


@dataclass(frozen=True)
class RecommendedWindow:
    """One ranked window suggestion derived from a forecast.

    ``rank`` is 1-indexed (1 = lowest predicted load). ``predicted_load``
    is the mean ``yhat`` across the rows that fall inside the window —
    the same metric an operator reading the Grafana panel would use to
    eyeball the window themselves.
    """

    rank: int
    start_timestamp: float
    duration_seconds: float
    predicted_load: float


def recommend_windows(
    *,
    forecast: pd.DataFrame,
    config: RecommendedWindowsConfig,
) -> list[RecommendedWindow]:
    """Score every aligned candidate window in the lookahead and return top-N.

    Candidate windows are anchored at each forecast step (so the
    resolution is the forecast step itself) and span ``duration``. The
    last fully-fitting window's start is the latest valid anchor; later
    anchors would overflow either the lookahead or the available
    forecast horizon and are skipped silently.

    Returns at most ``config.count`` windows, ranked ascending by
    predicted load. Ties are broken by start timestamp (earlier first)
    so successive refreshes produce a stable ranking when the curve is
    flat — without this an operator's "next window" recommendation
    would jitter across equally-good slots.
    """
    if forecast.empty or "yhat" not in forecast.columns or "ds" not in forecast.columns:
        return []
    if config.count <= 0 or config.duration.total_seconds() <= 0:
        return []
    duration_seconds = config.duration.total_seconds()
    lookahead_seconds = config.lookahead.total_seconds()
    seconds = _seconds_from_start(forecast["ds"])
    if seconds is None or not seconds:
        return []
    start_unix = _unix_seconds(forecast["ds"].iloc[0])
    if start_unix is None:
        return []
    yhat = [float(v) for v in forecast["yhat"].tolist()]
    candidates: list[RecommendedWindow] = []
    n = len(seconds)
    for i in range(n):
        window_start = seconds[i]
        if window_start > lookahead_seconds:
            # All subsequent anchors are also out of the lookahead — the
            # forecast timestamps are monotone, so we can stop early.
            break
        window_end = window_start + duration_seconds
        # j: half-open upper bound — the first row strictly past the
        # window's end. If we never find one, the window overflows the
        # forecast horizon and the candidate is skipped.
        j = i
        while j < n and seconds[j] < window_end:
            j += 1
        if j == n and (seconds[-1] - window_start) < duration_seconds:
            # The forecast ends before the window does; skip rather than
            # using a partial mean (we'd recommend windows we can't see
            # the full load of).
            continue
        values_in_window = [v for v in yhat[i:j] if math.isfinite(v)]
        if not values_in_window:
            continue
        predicted_load = sum(values_in_window) / len(values_in_window)
        candidates.append(
            RecommendedWindow(
                rank=0,  # filled in after sorting
                start_timestamp=start_unix + window_start,
                duration_seconds=duration_seconds,
                predicted_load=predicted_load,
            )
        )
    # Stable sort: ascending load, then ascending start_timestamp for
    # deterministic ranking when several windows tie on load.
    candidates.sort(key=lambda w: (w.predicted_load, w.start_timestamp))
    top = candidates[: config.count]
    return [
        RecommendedWindow(
            rank=i + 1,
            start_timestamp=w.start_timestamp,
            duration_seconds=w.duration_seconds,
            predicted_load=w.predicted_load,
        )
        for i, w in enumerate(top)
    ]


def _unix_seconds(ts: object) -> float | None:
    """Return ``ts.timestamp()`` (seconds since the epoch) or ``None``.

    Accepts ``datetime`` and pandas ``Timestamp`` (which is duck-typed as
    a datetime but doesn't subclass it in older pandas releases); the
    fallback path uses ``getattr`` to avoid asserting an attribute that
    ``object`` doesn't carry.
    """
    if isinstance(ts, datetime):
        return ts.timestamp()
    timestamp_fn = getattr(ts, "timestamp", None)
    if timestamp_fn is None:
        return None
    try:
        return float(timestamp_fn())
    except (AttributeError, ValueError, TypeError):
        return None
