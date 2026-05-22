"""Actual-vs-forecast deviation metrics.

When ``emission.deviation: true`` the forecaster compares the most recent actual
value of a series to the forecast band emitted on the previous run and emits:

- ``forecast_deviation_ratio``: signed normalised distance from the centre of
  the band, ``(actual - yhat) / (yhat_upper - yhat_lower)``. ``0`` means the
  actual sits exactly on the point forecast; ``+/-0.5`` means it is at the
  edge of the band; ``|ratio| > 0.5`` means it is outside.
- ``forecast_deviation_outside_band``: ``1`` if the actual is outside the
  band, else ``0``. Suitable for ``for: 15m`` style alerts.

The band width is taken to be ``yhat_upper - yhat_lower``. When the band is
degenerate (zero or non-finite) the ratio is reported as ``NaN`` and
``outside_band`` is ``True`` if the actual is not equal to the point
forecast, else ``False`` — i.e. an exact match is never counted as outside.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Deviation:
    """Pair of deviation values for a single (actual, band) tuple."""

    ratio: float
    outside_band: bool


def compute(
    actual: float,
    yhat: float,
    yhat_lower: float,
    yhat_upper: float,
) -> Deviation:
    """Compute deviation of ``actual`` from the forecast band.

    Args:
        actual: Observed value.
        yhat: Point forecast.
        yhat_lower: Lower edge of the confidence band.
        yhat_upper: Upper edge of the confidence band.

    Returns:
        A :class:`Deviation`. ``ratio`` is NaN when the band is degenerate or
        the inputs are non-finite; ``outside_band`` is still well-defined.
    """
    if not all(math.isfinite(v) for v in (actual, yhat, yhat_lower, yhat_upper)):
        return Deviation(ratio=float("nan"), outside_band=False)

    lo, hi = (yhat_lower, yhat_upper) if yhat_lower <= yhat_upper else (yhat_upper, yhat_lower)
    width = hi - lo

    if width <= 0:
        # Degenerate band: ratio is undefined. Outside-band still meaningful:
        # any drift from the point forecast counts as outside.
        return Deviation(ratio=float("nan"), outside_band=actual != yhat)

    ratio = (actual - yhat) / width
    return Deviation(ratio=ratio, outside_band=not inside_band(actual, lo, hi))


def inside_band(actual: float, lower: float, upper: float) -> bool:
    """Return ``True`` when ``actual`` lies inside ``[lower, upper]``.

    Single source of truth for the inside/outside test — shared with the
    rolling :mod:`promforecast.band_coverage` path so the two diagnostics
    can never drift in how they classify an observation.

    The caller is expected to have already filtered out non-finite
    inputs; this helper assumes finite arguments. Bands are ordered if
    ``lower > upper`` (a band whose lower bound is above the upper
    bound is normalised so the comparison still makes sense).
    """
    lo, hi = (lower, upper) if lower <= upper else (upper, lower)
    return lo <= actual <= hi
