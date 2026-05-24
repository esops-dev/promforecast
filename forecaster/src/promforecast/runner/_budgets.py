"""Series-budget allocation and per-query overflow / canary sampling.

Pulled out of the runner monolith because none of these helpers touch
runner instance state; they're pure data shaping. The runner reaches
for them once per group run to decide which series survive the
per-group / total / canary caps.
"""

from __future__ import annotations

import math
import random
from typing import Literal

from ..config import CanaryConfig, GroupConfig
from ..source import SeriesFrame

OverflowStrategy = Literal["drop_lowest_priority", "reject", "sample"]


def _compute_global_budgets(groups: list[GroupConfig], max_total_series: int) -> dict[str, int]:
    """Static per-group share of ``max_total_series``, weighted by priority.

    A static split avoids cross-group coordination at runtime: groups run on
    independent timers and may overlap. With weights ``priority/sum(priorities)``
    the lowest-priority groups give up budget first, which matches the intent
    of ``series_overflow: drop_lowest_priority``.
    """
    if max_total_series <= 0 or not groups:
        return {g.name: 0 for g in groups}
    weights = {g.name: max(1, g.priority) for g in groups}
    total_weight = sum(weights.values())
    return {
        name: max(0, (max_total_series * weight) // total_weight)
        for name, weight in weights.items()
    }


def _apply_overflow(
    frames: list[SeriesFrame], cap: int, strategy: OverflowStrategy
) -> tuple[list[SeriesFrame], int]:
    """Reduce ``frames`` to fit ``cap`` according to ``strategy``.

    Returns ``(allowed, dropped_count)``. With ``cap <= 0`` everything is
    dropped; ``reject`` drops the whole query rather than half-emitting it.
    """
    if cap <= 0:
        return [], len(frames)
    if len(frames) <= cap:
        return frames, 0
    if strategy == "reject":
        return [], len(frames)
    if strategy == "sample":
        return random.sample(frames, cap), len(frames) - cap
    return frames[:cap], len(frames) - cap


def _apply_canary(
    frames: list[SeriesFrame],
    canary: CanaryConfig,
) -> tuple[list[SeriesFrame], int]:
    """Random-sample ``ceil(weight * N)`` frames when canary mode is on.

    Returns ``(kept, dropped)``. ``ceil`` (not ``floor``) so a tiny
    population doesn't round down to zero — ``weight=0.1`` on a 5-series
    query still fits one series, which is the useful canary case. The
    caller emits the drop count as
    ``forecast_series_dropped_total{reason="canary"}`` so the sampling
    is observable from /metrics.

    Sampling uses :func:`random.sample` (no fixed seed) so successive runs
    rotate through the population. A fixed seed would bias the canary to
    the same N series forever and defeat coverage over time.
    """
    if not canary.enabled or not frames:
        return frames, 0
    if canary.weight >= 1.0:
        return frames, 0
    keep = max(1, math.ceil(len(frames) * canary.weight))
    if keep >= len(frames):
        return frames, 0
    sampled = random.sample(frames, keep)
    return sampled, len(frames) - keep
