"""Per-workload right-sizing recommendations from forecast peaks.

The forecaster already has the model machinery, the per-series quality
signal, and the future trajectory. This module turns those into a
per-workload "you can downsize from X to Y" gauge — one row per
operator-configured (workload, kind) entry, gated on
``forecast_quality_score`` so untrustworthy forecasts never produce a
recommendation.

The math is intentionally simple:

* **Predicted peak** — ``max(yhat_upper_<highest_level>)`` across the
  forecast horizon (falling back to ``max(yhat)`` when no bands were
  emitted). The upper band is the conservative answer; right-sizing on
  the central forecast would systematically under-provision for the
  expected variability.
* **Recommended size** — ``peak * (1 + headroom_pct / 100)``. Headroom
  defaults to 30% which is the SRE-standard "leave some rope for
  unexpected load" cushion.
* **Confidence label** — derived from the forecast's quality score so
  operators can see at a glance which recommendations are worth acting
  on (``high`` ≥ 0.8, ``medium`` ≥ 0.6, ``low`` below).

Pure functions — the runner threads the per-series snapshots through
and gets back a flat list of recommendation rows ready for the
exporter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import ResourceSizingConfig, ResourceSizingWorkloadConfig
    from .exporter import ForecastPoint, QualityPoint

# Confidence buckets — match the gates operators typically wire on
# ``forecast_quality_score`` so the categorical signal aligns with the
# alerting story.
_CONFIDENCE_HIGH = 0.8
_CONFIDENCE_MEDIUM = 0.6


@dataclass(frozen=True)
class ResourceSizingRecommendation:
    """One ``forecast_recommended_resource_size`` row destined for /metrics.

    ``recommended`` is the sample value of the primary gauge; the
    optional ``current`` / ``savings_pct`` populate the sibling
    gauges when the operator wired ``current_promql`` on the workload.
    ``confidence`` rides as a categorical label
    (``high|medium|low``) so dashboards can filter quickly.
    """

    workload: str
    kind: str
    confidence: str
    recommended: float
    current: float | None = None
    savings_pct: float | None = None


def confidence_for(quality_score: float) -> str:
    """Map a quality score in ``[0, 1]`` to a categorical confidence band."""
    if not math.isfinite(quality_score):
        return "low"
    if quality_score >= _CONFIDENCE_HIGH:
        return "high"
    if quality_score >= _CONFIDENCE_MEDIUM:
        return "medium"
    return "low"


def compute_predicted_peak(
    forecast_points: list[ForecastPoint],
    query_id: str,
) -> float | None:
    """Return the conservative peak from the forecast's upper band, or None.

    Walks the snapshot's forecast points filtered to ``query_id`` and
    picks the maximum of the highest-confidence ``yhat_upper`` row
    (looking for ``<id>_forecast_upper`` metric names). Falls back to
    the central ``<id>_forecast`` peak when no bands are present. NaN
    samples are skipped so a single degenerate row doesn't poison the
    peak; an entirely-non-finite forecast returns ``None`` so the
    caller can skip the workload silently rather than emit a NaN
    recommendation.

    Two-pass implementation: bucket the upper-band rows by level
    first, then pick the max value at the highest level. Clearer
    than interleaving "which level wins" with "which value wins" in
    a single branch.
    """
    upper_metric = f"{query_id}_forecast_upper"
    central_metric = f"{query_id}_forecast"
    upper_by_level: dict[int, float] = {}
    central_max: float = float("-inf")
    for point in forecast_points:
        if not math.isfinite(point.value):
            continue
        if point.metric == upper_metric:
            level_str = point.labels.get("level", "")
            try:
                level = int(level_str)
            except ValueError:
                continue
            prev = upper_by_level.get(level, float("-inf"))
            upper_by_level[level] = max(prev, point.value)
        elif point.metric == central_metric:
            central_max = max(central_max, point.value)
    if upper_by_level:
        # Highest configured level wins — the conservative-est band
        # the operator opted into.
        best_level = max(upper_by_level)
        return float(upper_by_level[best_level])
    if math.isfinite(central_max):
        return float(central_max)
    return None


def compute_recommendation(
    *,
    workload: ResourceSizingWorkloadConfig,
    config: ResourceSizingConfig,
    forecast_points: list[ForecastPoint],
    quality_points: list[QualityPoint],
    current: float | None = None,
) -> ResourceSizingRecommendation | None:
    """Build a single recommendation row, or ``None`` when ungated.

    Returns ``None`` (the workload is skipped) when:

    * the forecast snapshot has no finite peak for ``forecast_query``,
    * every quality_point for the query is non-finite, or
    * the (max) quality score is below ``config.quality_floor``.

    When ``current`` is provided we compute the savings percentage as
    ``(current - recommended) / current * 100`` (negative means the
    forecast predicts a need *above* the current limit — operators
    read that as "upsize, don't downsize"). A zero or negative
    ``current`` is treated as missing because the percentage is
    undefined / misleading there.
    """
    peak = compute_predicted_peak(forecast_points, workload.forecast_query)
    if peak is None:
        return None
    # Quality gate: take the *mean* of finite quality scores for the
    # query so the gate reflects "this forecast is broadly trustworthy",
    # not "at least one emission was good". A naive max-quality gate
    # would let a single high-quality model pull a workload's
    # recommendation through even when every other emission for the
    # same query is poor — and the peak we picked above might have
    # come from one of those poor emissions (max-quality on one model
    # doesn't constrain max-peak across models). The mean is the
    # safer default; operators who want a per-model gate can configure
    # `auto_select: true` so only the winner contributes a
    # quality_point.
    quality = _mean_quality_for_query(quality_points, workload.forecast_query)
    if quality is None or quality < config.quality_floor:
        return None
    recommended = peak * (1.0 + config.headroom_pct / 100.0)
    # Drop ``current`` (and therefore ``savings_pct``) when the value
    # is missing, non-finite, or non-positive. A zero / negative
    # current makes the percentage undefined and surfacing a
    # ``current=0`` row would mislead operators into reading the
    # workload as "unprovisioned" instead of "we couldn't measure it".
    has_current = current is not None and math.isfinite(current) and current > 0
    savings: float | None = None
    if has_current:
        # mypy: ``has_current`` already narrowed current to a non-None
        # finite positive float.
        assert current is not None
        savings = (current - recommended) / current * 100.0
    return ResourceSizingRecommendation(
        workload=workload.workload,
        kind=workload.kind,
        confidence=confidence_for(quality),
        recommended=recommended,
        current=current if has_current else None,
        savings_pct=savings,
    )


def _mean_quality_for_query(
    quality_points: list[QualityPoint],
    query_id: str,
) -> float | None:
    """Return the mean of finite quality scores for ``query_id``, or None.

    Empty / all-non-finite input returns ``None`` so the caller can
    skip the workload silently rather than treating a missing
    measurement as a passing gate.
    """
    finite_scores: list[float] = []
    for point in quality_points:
        if point.labels.get("id") != query_id:
            continue
        if not math.isfinite(point.score):
            continue
        finite_scores.append(float(point.score))
    if not finite_scores:
        return None
    return sum(finite_scores) / len(finite_scores)
