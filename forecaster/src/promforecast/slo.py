"""Per-query SLO error-budget forecasting.

A forecast that represents an error rate (or any rate of bad events) is
the input to three derivations that map the SRE workbook's burn-rate
vocabulary onto first-class Prometheus metrics:

* ``<id>_forecast_error_budget_remaining_seconds{window}`` — the
  total allowed error budget for the configured window. With an SLO
  objective of 99.9% over a 30d window this is ``0.001 * 30d = 2592s``.
* ``<id>_forecast_error_budget_burn_rate{window}`` — the predicted
  consumption rate divided by the allowed rate. A value of ``1`` means
  "burning the budget at exactly the rate that exhausts it in one
  window"; values above ``1`` mean the budget is being consumed faster
  than sustainable.
* ``<id>_forecast_error_budget_exhausted_in_seconds{window[, level]}`` —
  projected time until the cumulative consumption at the forecast's
  current rate exceeds the budget. The point row carries no ``level``
  label; band rows use ``yhat_upper`` (the conservative side — a
  higher predicted rate exhausts the budget sooner).

Pure-function implementation so the runner stays orchestration-shaped
and the tests exercise the math directly. Mirrors the structure of
:mod:`promforecast.capacity`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from .config import SLOConfig

# Objective is configured as a percentage in (0, 100); divide by this
# to get the fractional allowed-error rate (``0.001`` for a 99.9% SLO).
_PERCENT_FULL = 100.0


@dataclass(frozen=True)
class BudgetSample:
    """Per-(window) SLO budget samples derived from one forecast.

    ``exhausted_in_per_level`` is a ``{level: seconds}`` map; ``level``
    is ``None`` for the point sample tracked separately on
    ``exhausted_in_point`` (mirrors the ``level=None`` convention on
    :class:`promforecast.capacity.ThresholdEta`). All time values are in
    seconds; non-positive forecast values produce ``+inf`` (= "budget
    never exhausts at this rate") rather than NaN so an alert on
    ``... < N`` is well-defined and won't silently match.
    """

    window_seconds: float
    remaining_seconds: float
    burn_rate: float
    exhausted_in_point: float
    exhausted_in_per_level: dict[int, float] = field(default_factory=dict)


def compute_budget(
    *,
    forecast: pd.DataFrame,
    slo: SLOConfig,
    levels: list[int],
) -> list[BudgetSample]:
    """Project ``forecast`` onto per-window SLO budget samples.

    The forecast is interpreted as the instantaneous bad-event rate
    (e.g. fraction of failing requests, fraction of time in error). The
    leading row is "now" — the burn rate and exhausted-in projections
    are computed from that snapshot to match how alerts read the
    metrics.

    Returns one :class:`BudgetSample` per configured window. Empty input,
    or a forecast missing the ``yhat`` column, yields the empty list so
    the caller can ``extend`` unconditionally.
    """
    if forecast.empty or "yhat" not in forecast.columns:
        return []
    if not slo.windows:
        return []
    # Pydantic already enforces 0 < objective < 100; guard at runtime too
    # so a hand-built config (tests, preview) can't smuggle a degenerate
    # value into the math.
    if not 0.0 < slo.objective < _PERCENT_FULL:
        return []
    allowed_rate = (_PERCENT_FULL - slo.objective) / _PERCENT_FULL
    try:
        current_yhat = float(forecast["yhat"].iloc[0])
    except (KeyError, ValueError, IndexError):
        return []
    out: list[BudgetSample] = []
    for window in slo.windows:
        window_seconds = window.total_seconds()
        if window_seconds <= 0:
            continue
        budget_seconds = allowed_rate * window_seconds
        # Point burn rate from yhat. ``yhat < 0`` is unusual for an
        # error rate but well-defined here: a negative rate means the
        # error budget is *recovering*, which the burn rate should
        # report as a negative number (operators read it directly).
        if math.isfinite(current_yhat) and allowed_rate > 0:
            burn_rate = current_yhat / allowed_rate
        else:
            burn_rate = float("nan")
        exhausted_in_point = _exhausted_in(budget_seconds, current_yhat)
        exhausted_in_per_level: dict[int, float] = {}
        for level in levels:
            col = f"yhat_upper_{level}"
            if col not in forecast.columns:
                continue
            try:
                upper = float(forecast[col].iloc[0])
            except (KeyError, ValueError, IndexError):
                continue
            exhausted_in_per_level[level] = _exhausted_in(budget_seconds, upper)
        out.append(
            BudgetSample(
                window_seconds=window_seconds,
                remaining_seconds=budget_seconds,
                burn_rate=burn_rate,
                exhausted_in_point=exhausted_in_point,
                exhausted_in_per_level=exhausted_in_per_level,
            )
        )
    return out


def _exhausted_in(budget_seconds: float, rate: float) -> float:
    """Return seconds until ``budget_seconds`` is consumed at ``rate``.

    ``rate <= 0`` (no errors, or an error budget that's recovering)
    means the budget never exhausts at this trajectory; we report that
    as ``+inf`` rather than NaN so alerts on
    ``... > some_threshold`` work the way operators expect — ``inf``
    propagates the "no concern" answer instead of silently dropping
    the sample.

    Non-finite ``rate`` (degenerate forecasts) yields ``NaN`` so the
    exporter drops the row at render time. Mixing the two would mask
    real model breakage behind a comforting "no exhaustion" answer.
    """
    if not math.isfinite(rate):
        return float("nan")
    if rate <= 0:
        return float("inf")
    return budget_seconds / rate
