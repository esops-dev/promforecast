"""Count-aware emission safeguards (clip-from-zero, round-to-integer).

When a query's ``emission.clip_non_negative`` is True every emitted
forecast / band value is clamped at zero from below; when its
``emission.round_to_integer`` is True every value is rounded to the
nearest integer. Both are applied symmetrically to the snapshot
``ForecastPoint`` list (consumed by /metrics) and to the
``ForecastCurvePoint`` list (consumed by the remote_write sink) so
scrape and push see the same numbers.

Counter semantics live in :func:`_classify_emission_correction` —
``kind="negative"`` ticks only when the raw value was actually below
zero (a real model-output anomaly worth surfacing), and
``kind="rounded"`` ticks only when the fractional component was at
least :data:`_ROUND_COUNTER_TRIGGER` (so the counter doesn't blow up
on every continuous emission that happens to be slightly non-integer).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from ...exporter import ForecastPoint
from ...sink import ForecastCurvePoint

if TYPE_CHECKING:
    import pandas as pd


_ROUND_COUNTER_TRIGGER = 0.05


def _classify_emission_correction(
    raw: float,
    *,
    clip: bool,
    rounder: bool,
) -> tuple[float, list[str]]:
    """Return ``(adjusted_value, kinds)`` after applying the safeguards.

    ``kinds`` is a list of bounded tokens ``["negative", "rounded"]``
    that the caller folds into the per-query emission-clipped counter.
    Each entry is recorded *once per emission* (a value that is both
    negative and rounded triggers both counters); a value that didn't
    need correction produces an empty list and contributes no counter
    bumps.

    Non-finite values pass through unmodified — clipping or rounding a
    NaN would surface the silent transformation as a confusing
    /metrics line later.
    """
    if not math.isfinite(raw):
        return raw, []
    value = raw
    kinds: list[str] = []
    if clip and value < 0.0:
        value = 0.0
        kinds.append("negative")
    if rounder:
        rounded = float(round(value))
        if abs(value - rounded) >= _ROUND_COUNTER_TRIGGER:
            kinds.append("rounded")
        value = rounded
    return value, kinds


def _apply_safeguards_to_snapshot_points(
    points: list[ForecastPoint],
    *,
    clip: bool,
    rounder: bool,
    counter: dict[str, int],
) -> list[ForecastPoint]:
    """Apply count-aware safeguards to ``ForecastPoint``s and bump ``counter``.

    Returns a freshly constructed list; the input list is not mutated.
    ``counter`` is updated in-place with ``kind -> count`` so the
    runner can fold it into the per-(query, kind) total without
    threading a second return value.
    """
    if not (clip or rounder):
        return points
    out: list[ForecastPoint] = []
    for point in points:
        new_value, kinds = _classify_emission_correction(point.value, clip=clip, rounder=rounder)
        for kind in kinds:
            counter[kind] = counter.get(kind, 0) + 1
        if kinds or new_value != point.value:
            out.append(ForecastPoint(metric=point.metric, labels=point.labels, value=new_value))
        else:
            out.append(point)
    return out


def _apply_safeguards_to_curve_points(
    points: list[ForecastCurvePoint],
    *,
    clip: bool,
    rounder: bool,
) -> list[ForecastCurvePoint]:
    """Apply count-aware safeguards to ``ForecastCurvePoint``s.

    Counter bumps live with the snapshot pass — the runner emits both
    families from the same fit and counting twice would double-bump
    the operator-facing total. The curve pass therefore only mutates
    values; it does not touch the counter.
    """
    if not (clip or rounder):
        return points
    out: list[ForecastCurvePoint] = []
    for point in points:
        new_value, _ = _classify_emission_correction(point.value, clip=clip, rounder=rounder)
        if new_value != point.value:
            out.append(
                ForecastCurvePoint(
                    metric=point.metric,
                    labels=point.labels,
                    value=new_value,
                    timestamp_ms=point.timestamp_ms,
                )
            )
        else:
            out.append(point)
    return out


def _apply_safeguards_to_forecast_frame(
    forecast: pd.DataFrame,
    *,
    levels: list[int],
    clip: bool,
    rounder: bool,
) -> pd.DataFrame:
    """Return a copy of ``forecast`` with count-aware safeguards applied.

    Used by the conditional emission path so the snapshot points, the
    sink curve, and the deviation gauges built from the same frame all
    see the same clipped/rounded values. Counter bumps live with the
    unconditional pass — the conditional emission re-uses the same fit
    semantics and counting twice would double-charge the per-(query,
    kind) total.

    Uses numpy vectorisation so a typical horizon (288 rows by 5 columns
    at default 2 levels) is one pass per column instead of 1440
    Python-level float conversions.
    """
    if forecast.empty or not (clip or rounder):
        return forecast
    import numpy as np  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415

    out = forecast.copy()
    cols = ["yhat"] + [
        col for level in levels for col in (f"yhat_lower_{level}", f"yhat_upper_{level}")
    ]
    for col in cols:
        if col not in out.columns:
            continue
        # Coerce to float so clip / round have well-defined semantics
        # against pandas extension-array columns. ``errors="coerce"``
        # turns non-numeric entries into NaN, which both numpy ops
        # tolerate and propagate — matching the Python-loop version's
        # behaviour of passing such entries through unchanged.
        arr = pd.to_numeric(out[col], errors="coerce").to_numpy(dtype=float)
        # Pre-record the NaN mask so non-finite inputs round-trip
        # unchanged: ``np.round`` of NaN is NaN already, but clipping
        # would turn NaN into 0.0 under naive ``np.maximum`` semantics —
        # we restore NaN at the end so the contract matches the scalar
        # :func:`_classify_emission_correction` path.
        nan_mask = ~np.isfinite(arr)
        if clip:
            # ``np.maximum`` preserves the input shape and dtype (unlike
            # ``np.where`` whose typed return is multidim) — the column
            # is 1-D so this is the right primitive here.
            arr = np.maximum(arr, 0.0)
        if rounder:
            arr = np.round(arr)
        if nan_mask.any():
            arr[nan_mask] = np.nan
        out[col] = arr
    return out
