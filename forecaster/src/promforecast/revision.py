"""Forecast-to-forecast revision magnitude.

Captures how much the *forecast itself* changes between fits. The
predicted value for "tomorrow 3pm" moving 30% since the last refit is
a signal in its own right, distinct from the input-data drift the
``forecast_drift_score`` family captures.

The metric is normalised by the prior forecast's typical scale —
``|new - old| / max(|old|, scale_floor)`` — so a 5%-of-baseline shift
scores 0.05 regardless of whether the underlying values are tens or
gigabytes. ``scale_floor`` is a small absolute floor (default 1e-9)
to prevent division-by-zero on values near zero; it's not configurable
on a per-query basis because the floor is a numerical guard rather
than a policy knob.

Per-(series, model, horizon) cache lives on the runner (one entry per
emitted snapshot row, keyed by the same labelset that lands on
/metrics). A reload that drops a (group, query) prunes the matching
cache entries via :meth:`RevisionCache.prune_to`; the runner already
calls this on reload so the cache stays bounded by the live config.

To survive series-key churn on workloads with high label cardinality
(rolling pod IPs, ephemeral container IDs) without leaking forever
between reloads, the cache enforces an LRU cap. New writes evict the
least-recently-touched entry when the cap is hit.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass

# Numerical floor for the relative-shift denominator. Values smaller
# than this are clamped up so a near-zero baseline doesn't blow the
# ratio to infinity. Picked well below the smallest realistic
# operational value but above ordinary float64 noise.
_SCALE_FLOOR = 1e-9

# LRU cap on the per-(series, model, horizon) cache. Sized so a typical
# deployment (low thousands of series x a handful of models x a handful
# of horizons) sits well below the cap, while an ephemeral-labelset
# deployment that would otherwise leak indefinitely caps at a bounded
# memory footprint (~few MB worst case at one float plus a 5-tuple key
# per entry). Chosen as a round number rather than tuned — the cache is
# only deciding whether the next fit reports a magnitude or a NaN.
_DEFAULT_MAX_ENTRIES = 100_000


@dataclass(frozen=True)
class RevisionPoint:
    """One ``forecast_revision_magnitude{...}`` sample.

    ``magnitude`` is the relative shift between the new and previous
    forecast for the same horizon. ``NaN`` when no prior fit was
    cached (first fit after boot or after a reload that pruned the
    entry) so the exporter can drop it without firing an alert on
    "no prior data".
    """

    labels: dict[str, str]
    magnitude: float


def compute_revision_magnitude(*, previous: float, current: float) -> float:
    """Relative absolute shift between two scalar forecasts.

    Returns ``NaN`` when either input is non-finite. Otherwise the
    return value is ``|current - previous| / max(|previous|, _SCALE_FLOOR)``
    so a 30% shift maps to 0.3 regardless of magnitude. Negative
    returns are not possible — the metric is unsigned by design
    (operators alert on "the forecast moved a lot", not "the forecast
    moved up").
    """
    if not (math.isfinite(previous) and math.isfinite(current)):
        return float("nan")
    denom = max(abs(previous), _SCALE_FLOOR)
    return abs(current - previous) / denom


class RevisionCache:
    """LRU-bounded cache of the previous central-forecast value per emitted row.

    Keys are ``(group, query, series_key, model, horizon)`` so the
    cache rotates correctly when the underlying labelset changes
    (e.g. a series rotated out of the source's response). One float
    per key — the cache holds only the *previous* central value, not
    the full curve, so the per-(emitted-row) footprint stays
    minimal. Reload-time pruning via :meth:`prune_to` clears
    (group, query) pairs that have left the live config; in-fit churn
    on series labels (high-cardinality ephemeral labels like rolling
    pod IPs) is bounded by the LRU cap so the cache never grows
    unboundedly between reloads.
    """

    def __init__(self, *, max_entries: int = _DEFAULT_MAX_ENTRIES) -> None:
        # OrderedDict gives us O(1) move-to-end on read + O(1) popitem
        # on the LRU eviction path, both of which the dict primitive
        # doesn't expose.
        self._previous: OrderedDict[tuple[str, str, str, str, str], float] = OrderedDict()
        self._max_entries = max_entries

    def get(
        self,
        *,
        group: str,
        query: str,
        series_key: str,
        model: str,
        horizon: str,
    ) -> float | None:
        key = (group, query, series_key, model, horizon)
        value = self._previous.get(key)
        if value is not None:
            # Touch on read so a series that's still being scored each
            # tick stays at the most-recent end of the LRU even if its
            # value happens to match the prior fit exactly (a put
            # touches separately).
            self._previous.move_to_end(key)
        return value

    def put(
        self,
        *,
        group: str,
        query: str,
        series_key: str,
        model: str,
        horizon: str,
        value: float,
    ) -> None:
        if not math.isfinite(value):
            return
        key = (group, query, series_key, model, horizon)
        if key in self._previous:
            self._previous.move_to_end(key)
        self._previous[key] = value
        # Evict from the oldest end until we're back under the cap.
        # popitem(last=False) is O(1).
        while len(self._previous) > self._max_entries:
            self._previous.popitem(last=False)

    def prune_to(self, live_pairs: set[tuple[str, str]]) -> None:
        """Drop entries whose (group, query) no longer exists in the live config."""
        survivors = [
            (key, value) for key, value in self._previous.items() if (key[0], key[1]) in live_pairs
        ]
        self._previous = OrderedDict(survivors)

    def __len__(self) -> int:
        return len(self._previous)


__all__ = [
    "RevisionCache",
    "RevisionPoint",
    "compute_revision_magnitude",
]
