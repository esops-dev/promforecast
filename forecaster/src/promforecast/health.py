"""Service-level forecast health composite.

Aggregates ``forecast_quality_score`` rows across the whole fleet into a
single number per ``(service, environment, ...)`` bucket. The result lands
as ``forecast_service_health_score`` so executives and platform teams can
report on the reliability of the *prediction infrastructure* without
staring at 30 individual series.

The aggregation is dimension-driven by the operator's
``governance.health_score.aggregate_by`` list — any subset of the
source-side labels carried on every quality_point qualifies. Series whose
label set is missing any of the requested keys are dropped, so a single
unlabelled forecast can't silently inflate or deflate every bucket.

Pure functions: the runner feeds in a flat list of quality points (read
from the exporter's per-group snapshots) plus the operator's
``HealthScoreConfig`` and gets back a flat list of aggregated rows. The
exporter then renders them under the global metric name.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from .config import HealthScoreConfig
    from .exporter import QualityPoint


@dataclass(frozen=True)
class ServiceHealthScore:
    """One aggregated row destined for ``forecast_service_health_score``.

    ``labels`` carries the ``aggregate_by`` dimension values (e.g.
    ``service="checkout"``, ``environment="prod"``) plus the
    ``aggregator`` and ``weight_by`` documentation labels so a Grafana
    panel can record which interpretation was in effect. ``score`` is
    the aggregated value in ``[0, 1]``.
    """

    labels: dict[str, str]
    score: float
    series_count: int


def compute_service_health_scores(
    quality_points: Iterable[QualityPoint],
    config: HealthScoreConfig,
) -> list[ServiceHealthScore]:
    """Aggregate per-series quality scores into per-dim composites.

    * ``config.aggregate_by`` enumerates the discriminator labels. A
      quality_point missing *any* of these labels is silently dropped
      — without the drop one un-labelled forecast would land in
      every bucket and skew the report.
    * ``config.quality_floor`` (when set) drops scores below the floor
      before aggregation, so cold-start / low-quality emissions
      don't pull a service's composite down below the alert gate the
      operator already set elsewhere.
    * ``config.aggregator`` picks the summary statistic; only
      ``weighted_mean`` ships today. With per-series quality_points
      each row contributes weight ``1``, so the result is the
      arithmetic mean of the bucket. ``config.weight_by`` is
      forwarded as a documentation label only — the math is the same
      across both options until a future per-group aggregated emission
      shape exists.
    * Non-finite quality_points are dropped before aggregation so a
      single NaN-shaped backtest can't poison the bucket.

    Output ordering is deterministic — buckets sort by their tuple of
    label values so successive runs publish a stable row sequence and
    Grafana series colours stay pinned across refreshes.
    """
    buckets: dict[tuple[str, ...], list[float]] = {}
    for point in quality_points:
        if not math.isfinite(point.score):
            continue
        if config.quality_floor is not None and point.score < config.quality_floor:
            continue
        key = _bucket_key(point.labels, config.aggregate_by)
        if key is None:
            continue
        buckets.setdefault(key, []).append(float(point.score))
    out: list[ServiceHealthScore] = []
    for key in sorted(buckets):
        scores = buckets[key]
        if not scores:
            continue
        labels = {dim: value for dim, value in zip(config.aggregate_by, key, strict=True)}
        labels["aggregator"] = config.aggregator
        labels["weight_by"] = config.weight_by
        # ``series_count`` rides as a label so a dashboard can read the
        # bucket size alongside the aggregated value. A separate count
        # gauge would force callers to join on the same labels and
        # double the storage footprint for no extra signal.
        labels["series_count"] = str(len(scores))
        out.append(
            ServiceHealthScore(
                labels=labels,
                score=sum(scores) / len(scores),
                series_count=len(scores),
            )
        )
    return out


def _bucket_key(labels: dict[str, str], aggregate_by: list[str]) -> tuple[str, ...] | None:
    """Project ``labels`` onto the ``aggregate_by`` tuple, or None if any are missing.

    Empty-string label values (which happen when a label is technically
    present but rendered as ``label=""``) are treated as missing — a
    bucket like ``service=""`` is almost certainly a labelling bug, and
    surfacing it as a real row would dilute every meaningful bucket.
    """
    out: list[str] = []
    for dim in aggregate_by:
        value = labels.get(dim, "")
        if not value:
            return None
        out.append(value)
    return tuple(out)
