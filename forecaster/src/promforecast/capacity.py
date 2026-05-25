"""Capacity-planning derivations from the forecast curve.

This module turns the raw forecast vector into the two operator-facing
shapes "when will this hit threshold X?" (ETA) and "how fast is the
forecast growing?" (growth rate). Both are configured per query and emit
purpose-built metric families:

* ``<id>_forecast_time_to_threshold_seconds{name, comparator, ...}`` —
  seconds from "now" (the first forecast step) until the forecast first
  crosses ``threshold.value`` in the direction specified by
  ``threshold.comparator``. ``NaN`` when the crossing does not happen
  within the configured horizon. When confidence levels are emitted, the
  ETA is also computed against ``yhat_upper`` (for ``gt``) /
  ``yhat_lower`` (for ``lt``) so the answer is conservative at the band.
* ``<id>_forecast_growth_rate{window, units, ...}`` — slope of the
  forecast over the trailing ``window`` of horizon, expressed either as
  ``units_per_second`` (the default — directly interpretable when the
  underlying metric is a rate) or as a fractional ``percent`` change.

Pure-function implementations so the runner stays orchestration-shaped and
the tests can exercise the math directly.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from ._time_axis import seconds_from_start as _seconds_from_start

if TYPE_CHECKING:
    import pandas as pd

    from .config import ThresholdConfig

Comparator = Literal["gt", "lt", "le_drain"]
GrowthUnits = Literal["units_per_second", "percent"]

# Comparators that emit the dedicated "queue is draining" ETA family
# (``<id>_forecast_time_to_drain_seconds``) instead of the standard
# threshold ETA. The drain family reports ``+inf`` on no-crossing —
# operators alert with ``... == +Inf for 30m`` to catch "queue is not
# actually draining". The lt/gt families report NaN on no-crossing
# (dropped at render time) because the typical alert against them is
# ``... < N`` and a missing sample is the right "no concern" signal.
DRAIN_COMPARATORS: frozenset[str] = frozenset({"le_drain"})

# Forecast frames smaller than this can't define a meaningful slope or
# crossing — one row is a point, not a curve.
_MIN_ROWS_FOR_DURATION = 2


@dataclass(frozen=True)
class ThresholdEta:
    """Computed ETA for a single (threshold, source-series) combination.

    ``level`` is ``None`` for the point-forecast ETA, and the numeric
    level for band ETAs. ``seconds`` is the time from the forecast's
    first step to the crossing, or NaN when no crossing happens in the
    horizon.
    """

    name: str
    comparator: Comparator
    value: float
    seconds: float
    level: int | None = None


@dataclass(frozen=True)
class GrowthRate:
    """Computed growth rate for one (forecast-series, window) combination."""

    window_seconds: float
    units: GrowthUnits
    rate: float


@dataclass(frozen=True)
class WillBreach:
    """0/1 predictive sample derived from an ETA + an ``alert_within`` window.

    ``value`` is ``1.0`` when the threshold ETA is finite and below the
    operator's ``alert_within`` lookahead, ``0.0`` otherwise. ``level`` is
    ``None`` for the central forecast, a numeric level for band-edge
    samples — same convention as :class:`ThresholdEta`.
    """

    name: str
    severity: str
    level: int | None
    value: float


@dataclass(frozen=True)
class ThresholdActionCost:
    """Config-derived cost-of-action annotation for one threshold.

    Surfaces as the per-(group, query, threshold) sample of
    ``forecast_threshold_action_cost`` so dashboards can rank capacity
    ETAs by cost-to-ignore. ``cost_type`` corresponds to the
    operator-configured ``action_cost.type`` and renders on the
    ``type`` label (renamed locally to avoid shadowing the builtin).
    """

    name: str
    cost_type: str
    cost_value: float


def build_threshold_action_costs(
    thresholds: list[ThresholdConfig],
) -> list[ThresholdActionCost]:
    """Project a query's configured thresholds onto action-cost annotations.

    Skips thresholds without an ``action_cost`` block so the family
    only carries the opted-in subset. The output ordering matches the
    config ordering so successive runs publish a stable row sequence.
    """
    out: list[ThresholdActionCost] = []
    for threshold in thresholds:
        cost = threshold.action_cost
        if cost is None:
            continue
        out.append(
            ThresholdActionCost(
                name=threshold.name,
                cost_type=cost.type,
                cost_value=float(cost.value),
            )
        )
    return out


def compute_will_breach(
    *,
    etas: list[ThresholdEta],
    thresholds: list[ThresholdConfig],
) -> list[WillBreach]:
    """Project ``etas`` onto 0/1 will-breach samples for the configured thresholds.

    Walks the (threshold, level) ETAs already produced by
    :func:`compute_threshold_etas` and emits a :class:`WillBreach` for
    each (threshold, level) whose threshold opted into the predictive
    family (``alert_within`` + ``severity`` both set). ETAs whose
    threshold did not opt in are skipped; non-finite ETAs (no crossing
    in the horizon) emit ``0`` rather than being suppressed so an alert
    on ``forecast_will_breach == 1`` has a stable zero-baseline series
    to chart against.
    """
    if not etas or not thresholds:
        return []
    by_name = {
        t.name: t for t in thresholds if t.alert_within is not None and t.severity is not None
    }
    if not by_name:
        return []
    out: list[WillBreach] = []
    for eta in etas:
        threshold = by_name.get(eta.name)
        if threshold is None:
            continue
        # mypy: narrowed by ``by_name`` filter above.
        assert threshold.alert_within is not None
        assert threshold.severity is not None
        window_seconds = threshold.alert_within.total_seconds()
        value = 1.0 if math.isfinite(eta.seconds) and eta.seconds < window_seconds else 0.0
        out.append(
            WillBreach(
                name=eta.name,
                severity=threshold.severity,
                level=eta.level,
                value=value,
            )
        )
    return out


def compute_threshold_etas(
    *,
    forecast: pd.DataFrame,
    thresholds: list[ThresholdConfig],
    levels: list[int],
) -> list[ThresholdEta]:
    """Compute per-threshold ETAs against the point forecast and bands.

    Returns one :class:`ThresholdEta` per (threshold, source) where source
    is the central forecast (``level=None``) plus one per emitted
    confidence level (when present in the forecast frame). The order is:
    point ETA first, then per-level ETAs in the order ``levels`` are
    requested.

    No samples are emitted when the forecast frame is empty or has fewer
    than two rows (a single step can't define a duration). Crossings are
    detected by linear interpolation between adjacent forecast steps,
    which matches the smooth-curve mental model operators have when they
    read a Grafana panel of the forecast.
    """
    if forecast.empty or len(forecast) < _MIN_ROWS_FOR_DURATION or not thresholds:
        return []
    if "ds" not in forecast.columns or "yhat" not in forecast.columns:
        return []
    seconds_axis = _seconds_from_start(forecast["ds"])
    if seconds_axis is None:
        return []
    yhat = [float(v) for v in forecast["yhat"].tolist()]
    out: list[ThresholdEta] = []
    for threshold in thresholds:
        comparator: Comparator = threshold.comparator
        out.append(
            ThresholdEta(
                name=threshold.name,
                comparator=comparator,
                value=threshold.value,
                level=None,
                seconds=_first_crossing(
                    seconds=seconds_axis,
                    values=yhat,
                    threshold=threshold.value,
                    comparator=comparator,
                ),
            )
        )
        # The conservative band side depends on the direction:
        # * ``gt`` (e.g. "disk fills"): ``yhat_upper`` reports the
        #   earliest possible crossing.
        # * ``lt`` (e.g. "free space drops below floor"):
        #   ``yhat_lower`` reports the earliest possible crossing.
        # * ``le_drain`` (e.g. "queue drains to zero from above"):
        #   ``yhat_upper`` is the *latest* drain time — the conservative
        #   answer when alerting on "queue is not draining". A drain
        #   that uses the lower band would tell us the optimistic
        #   answer, which is the opposite of what the alert needs.
        band_side = "yhat_upper" if comparator in ("gt", "le_drain") else "yhat_lower"
        for level in levels:
            col = f"{band_side}_{level}"
            if col not in forecast.columns:
                continue
            band_values = [float(v) for v in forecast[col].tolist()]
            out.append(
                ThresholdEta(
                    name=threshold.name,
                    comparator=comparator,
                    value=threshold.value,
                    level=level,
                    seconds=_first_crossing(
                        seconds=seconds_axis,
                        values=band_values,
                        threshold=threshold.value,
                        comparator=comparator,
                    ),
                )
            )
    return out


def compute_growth_rates(
    *,
    forecast: pd.DataFrame,
    windows_seconds: list[float],
    units: GrowthUnits,
) -> list[GrowthRate]:
    """Compute the forecast growth rate over each window.

    Windows wider than the available horizon are reported as NaN (rather
    than silently truncated) so an operator's ``5d`` rule still shows the
    sample with ``rate=NaN`` instead of being masquerading-as-a-shorter
    window. The slope is fitted over the indices of forecast rows whose
    timestamp falls within ``[start, start + window]``.
    """
    if forecast.empty or len(forecast) < _MIN_ROWS_FOR_DURATION or not windows_seconds:
        return []
    if "ds" not in forecast.columns or "yhat" not in forecast.columns:
        return []
    seconds_axis = _seconds_from_start(forecast["ds"])
    if seconds_axis is None:
        return []
    yhat = [float(v) for v in forecast["yhat"].tolist()]
    out: list[GrowthRate] = []
    horizon_seconds = seconds_axis[-1] - seconds_axis[0]
    for window in windows_seconds:
        if window <= 0:
            continue
        if window > horizon_seconds:
            out.append(GrowthRate(window_seconds=window, units=units, rate=float("nan")))
            continue
        rate = _slope_over_window(
            seconds=seconds_axis,
            values=yhat,
            window_seconds=window,
            units=units,
        )
        out.append(GrowthRate(window_seconds=window, units=units, rate=rate))
    return out


def _first_crossing(
    *,
    seconds: list[float],
    values: list[float],
    threshold: float,
    comparator: Comparator,
) -> float:
    """Return the seconds-from-start of the first crossing, or NaN / +Inf.

    A "crossing" is the first adjacent pair where the predicate (``> v``
    or ``< v``) flips from False to True. Linear interpolation between
    the two endpoints estimates the precise crossing time, which is the
    behaviour operators expect from reading a continuous curve in
    Grafana. If the series starts already on the wrong side of the
    threshold, the ETA is 0.

    The no-crossing sentinel depends on the comparator:

    * ``gt`` / ``lt``: NaN (dropped at render time so an alert on
      ``... < N`` doesn't silently match).
    * ``le_drain``: ``+inf``. Operators alert on
      ``... == +Inf for 30m`` to catch "queue is not draining" — the
      sentinel must round-trip through Prometheus and stay queryable
      rather than being filtered out at render time.
    """
    if not values or not seconds or len(values) != len(seconds):
        return _no_crossing_sentinel(comparator)
    # ``gt`` and ``lt`` use strict inequality (a value exactly at the
    # threshold is "approaching", not "crossed"). ``le_drain`` uses
    # non-strict ``<=`` so a forecast that touches the drain value
    # (e.g. a queue forecast that hits exactly zero) registers as
    # drained — otherwise an already-empty queue would report
    # ``+Inf`` "not draining" and the QueueNotDraining alert would
    # fire on a healthy queue.
    triggered = _triggered_for(comparator, threshold)
    if triggered(values[0]) and math.isfinite(values[0]):
        return 0.0
    for i in range(1, len(values)):
        prev_v, curr_v = values[i - 1], values[i]
        if not (math.isfinite(prev_v) and math.isfinite(curr_v)):
            continue
        if not triggered(prev_v) and triggered(curr_v):
            # Linear interpolation between the two anchor points: solve
            # for the time at which the line crosses ``threshold``.
            span = curr_v - prev_v
            if span == 0:
                return seconds[i]
            ratio = (threshold - prev_v) / span
            ratio = max(0.0, min(1.0, ratio))
            return seconds[i - 1] + ratio * (seconds[i] - seconds[i - 1])
    return _no_crossing_sentinel(comparator)


def _triggered_for(comparator: Comparator, threshold: float) -> Callable[[float], bool]:
    """Return the per-step predicate that decides "has the curve crossed?"

    Strict inequality for ``gt`` / ``lt`` (a value exactly at the
    threshold is approaching, not crossed). Non-strict ``<=`` for
    ``le_drain`` so a queue forecast that lands exactly on the drain
    value is recognised as drained.
    """
    if comparator == "gt":

        def gt_predicate(v: float) -> bool:
            return v > threshold

        return gt_predicate
    if comparator == "le_drain":

        def le_predicate(v: float) -> bool:
            return v <= threshold

        return le_predicate

    def lt_predicate(v: float) -> bool:
        return v < threshold

    return lt_predicate


def _no_crossing_sentinel(comparator: Comparator) -> float:
    """Sentinel value for a curve that never crosses ``threshold``.

    Centralised so the two callers (the leading-row triggered? guard
    and the loop exit) agree on which sentinel each comparator uses.
    """
    return float("inf") if comparator in DRAIN_COMPARATORS else float("nan")


def _slope_over_window(
    *,
    seconds: list[float],
    values: list[float],
    window_seconds: float,
    units: GrowthUnits,
) -> float:
    """Return slope of ``values`` over the leading ``window_seconds``.

    For ``units_per_second`` this is a least-squares slope (numerically
    stable when the forecast is noisy). For ``percent`` it's the
    relative change ``(end / start) - 1`` taken from the values at the
    window's endpoints — operators read percent change as a discrete
    ratio, not a regression coefficient.

    Returns NaN when the window contains fewer than two finite samples
    or when ``percent`` mode hits a zero baseline.
    """
    end_t = seconds[0] + window_seconds
    idx_in_window: list[int] = []
    for i, t in enumerate(seconds):
        if t <= end_t + 1e-9:
            idx_in_window.append(i)
        else:
            break
    if len(idx_in_window) < _MIN_ROWS_FOR_DURATION:
        return float("nan")
    xs = [seconds[i] for i in idx_in_window if math.isfinite(values[i])]
    ys = [values[i] for i in idx_in_window if math.isfinite(values[i])]
    if len(xs) < _MIN_ROWS_FOR_DURATION:
        return float("nan")
    if units == "percent":
        start_v, end_v = ys[0], ys[-1]
        if start_v == 0:
            return float("nan")
        return (end_v - start_v) / abs(start_v)
    # Closed-form least squares slope; cheaper and avoids a numpy import.
    n = float(len(xs))
    sum_x = sum(xs)
    sum_y = sum(ys)
    sum_xy = sum(x * y for x, y in zip(xs, ys, strict=True))
    sum_x2 = sum(x * x for x in xs)
    denom = n * sum_x2 - sum_x * sum_x
    if denom == 0:
        return float("nan")
    return (n * sum_xy - sum_x * sum_y) / denom
