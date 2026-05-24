"""Pure helpers for conformal calibration application.

Conformal calibration is applied symmetrically to both the snapshot
``ForecastPoint`` list (consumed by /metrics) and the
``ForecastCurvePoint`` list (consumed by the remote_write sink). The two
carry the same (metric, labels, value) shape but ``ForecastCurvePoint``
also has a ``timestamp_ms`` field. The decision logic — "if this is a
lower/upper row for a calibrated (model, level), widen and label it" —
is shared via :func:`_calibrated_band_value`; the per-type list rebuild
lives in the two thin wrappers so each wrapper is straight-line code
without a constructor callback.

The fit-time application (``_compute_conformal_offsets`` and
``_compute_conditional_conformal_offsets``) lives on the ``Runner``
class because it needs the per-instance fit executor / model registry.
This module collects only the post-fit pure-function pieces.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...exporter import ForecastPoint
from ...sink import ForecastCurvePoint

if TYPE_CHECKING:
    import pandas as pd


def _calibrated_band_value(
    metric: str,
    labels: dict[str, str],
    value: float,
    offsets_by_model: dict[str, dict[int, float]],
) -> tuple[float, bool] | None:
    """Return ``(new_value, calibrated)`` if the point is a calibratable band, else ``None``.

    ``None`` means "leave the point alone". ``(value, False)`` means the
    offset is zero/missing for this (model, level), so the band is left as
    the model emitted it; ``(value, True)`` means we widened and the
    caller should add a ``calibrated="true"`` label. The yhat row (metric
    exactly ending in ``_forecast``) is always returned as ``None`` —
    calibration is a band adjustment, not a point-forecast adjustment.
    """
    model_name = labels.get("model", "")
    offsets = offsets_by_model.get(model_name)
    if offsets is None or metric.endswith("_forecast"):
        return None
    try:
        level = int(labels.get("level", ""))
    except ValueError:
        return None
    offset = offsets.get(level)
    if offset is None or offset <= 0:
        return None
    if metric.endswith("_forecast_lower"):
        return value - offset, True
    if metric.endswith("_forecast_upper"):
        return value + offset, True
    return None


def _apply_offsets_to_threshold_eta_inputs(
    forecast: pd.DataFrame,
    offsets: dict[int, float] | None,
    levels: list[int],
) -> pd.DataFrame:
    """Return a copy of ``forecast`` with band columns shifted by the per-level offset.

    The conformal calibration offset widens both sides of each band
    symmetrically: ``yhat_upper_<level> += offset`` and
    ``yhat_lower_<level> -= offset``. Re-applying that shift here so the
    capacity ETA computation sees the same bands /metrics exposes keeps
    ``time_to_threshold{level="95"}`` and ``yhat_upper{level="95"}``
    self-consistent.

    ``offsets=None`` or an empty dict is the common case (no calibration
    ran for this model) and the frame is returned unmodified. Only
    non-zero positive offsets actually mutate columns; a zero offset is
    a no-op so we skip the copy.
    """
    if not offsets or forecast.empty:
        return forecast
    if not any(o > 0 for o in offsets.values()):
        return forecast
    out = forecast.copy()
    for level, offset in offsets.items():
        if offset <= 0 or level not in levels:
            continue
        lo_col = f"yhat_lower_{level}"
        hi_col = f"yhat_upper_{level}"
        if lo_col in out.columns:
            out[lo_col] = out[lo_col].to_numpy(dtype=float) - offset
        if hi_col in out.columns:
            out[hi_col] = out[hi_col].to_numpy(dtype=float) + offset
    return out


def _apply_offsets_to_snapshot_points(
    points: list[ForecastPoint],
    offsets_by_model: dict[str, dict[int, float]],
) -> list[ForecastPoint]:
    """Widen ``ForecastPoint`` bands by the per-(model, level) offset.

    Returns a freshly constructed list; the input list is not mutated.
    """
    if not offsets_by_model:
        return points
    out: list[ForecastPoint] = []
    for point in points:
        adjusted = _calibrated_band_value(point.metric, point.labels, point.value, offsets_by_model)
        if adjusted is None:
            out.append(point)
            continue
        new_value, calibrated = adjusted
        new_labels = {**point.labels, "calibrated": "true"} if calibrated else point.labels
        out.append(ForecastPoint(metric=point.metric, labels=new_labels, value=new_value))
    return out


def _apply_offsets_to_curve_points(
    points: list[ForecastCurvePoint],
    offsets_by_model: dict[str, dict[int, float]],
) -> list[ForecastCurvePoint]:
    """Widen ``ForecastCurvePoint`` bands by the per-(model, level) offset.

    Returns a freshly constructed list; the input list is not mutated.
    Carries the original ``timestamp_ms`` through unchanged — calibration
    shifts the band level, not when the forecast is for.
    """
    if not offsets_by_model:
        return points
    out: list[ForecastCurvePoint] = []
    for point in points:
        adjusted = _calibrated_band_value(point.metric, point.labels, point.value, offsets_by_model)
        if adjusted is None:
            out.append(point)
            continue
        new_value, calibrated = adjusted
        new_labels = {**point.labels, "calibrated": "true"} if calibrated else point.labels
        out.append(
            ForecastCurvePoint(
                metric=point.metric,
                labels=new_labels,
                value=new_value,
                timestamp_ms=point.timestamp_ms,
            )
        )
    return out
