"""Self-observability metrics for the forecaster itself.

Three operator-facing families:

* ``forecast_model_fit_success_ratio{group, model, window}`` — the fraction
  of attempted fits that completed within ``safety.fit_timeout`` without
  raising, computed over a rolling window. Two windows are emitted (``1h``
  and ``24h`` by default). A degrading model surfaces here before its bad
  forecasts pollute downstream alerts.
* ``forecast_regressor_stability_score{group, query, regressor_id}`` —
  a 0..1 score reflecting how much the regressor's recent value
  distribution has drifted from its lookback distribution. Reuses the
  same MAPE-style scoring as the forecast-drift module so the
  number scales the same way operators already understand.
* ``forecast_group_memory_bytes{group}`` and
  ``forecast_group_cpu_seconds_total{group}`` — per-group resource
  attribution. CPU is the cumulative sum of per-model fit durations
  (every fit that ran on the shared executor counts). Memory is a
  best-effort snapshot of the in-flight per-group dataframes plus the
  exporter's per-group snapshot, taken at the end of each group run
  via ``sys.getsizeof`` summation — no psutil dependency to avoid
  pulling in another wheel for one gauge.

None of these touch the per-series cardinality budget; they're
group / model / query labelled and bounded by configuration size.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable


# Default rolling windows for the success-ratio gauge. The chart's HPA
# threshold language ("ratio < 0.9 for 5m") works against either window;
# the 1h flavour is for spotting a model that just started failing, the
# 24h flavour is the slower trend.
DEFAULT_SUCCESS_WINDOWS_SECONDS: tuple[int, ...] = (3600, 86400)

# Minimum series length required for a meaningful "recent vs. lookback"
# split. Below this the relative-shift metric is more noise than signal,
# so the stability score returns NaN and the exporter drops the row.
_MIN_STABILITY_SAMPLES = 5


@dataclass(frozen=True)
class FitSuccessSample:
    """One (success?, timestamp) observation in the rolling window."""

    success: bool
    timestamp: float


@dataclass(frozen=True)
class SuccessRatioPoint:
    """Rendered point for ``forecast_model_fit_success_ratio``."""

    group: str
    model: str
    window_label: str
    ratio: float


@dataclass(frozen=True)
class RegressorStabilityPoint:
    """Rendered point for ``forecast_regressor_stability_score``."""

    group: str
    query: str
    regressor_id: str
    score: float


@dataclass(frozen=True)
class GroupResourcePoint:
    """One ``(group, memory_bytes, cpu_seconds_total)`` triple at render time."""

    group: str
    memory_bytes: float
    cpu_seconds_total: float


class FitOutcomeTracker:
    """Per-(group, model) rolling success/failure ledger.

    A ``deque`` per (group, model) keeps every observation in the
    longest configured window; ``ratio_at`` returns the fraction of
    successes that fall inside the asked-for window. Observations
    older than the longest window are pruned at record time so the
    deque stays bounded at the per-(group, model) operational
    cardinality the runner already imposes.
    """

    def __init__(self, *, windows_seconds: Iterable[int] = DEFAULT_SUCCESS_WINDOWS_SECONDS) -> None:
        self._windows_seconds: tuple[int, ...] = tuple(sorted(set(int(w) for w in windows_seconds)))
        if not self._windows_seconds:
            raise ValueError("FitOutcomeTracker requires at least one window")
        self._max_window = max(self._windows_seconds)
        self._lock = threading.Lock()
        self._samples: dict[tuple[str, str], deque[FitSuccessSample]] = {}

    @property
    def windows_seconds(self) -> tuple[int, ...]:
        return self._windows_seconds

    def record(self, *, group: str, model: str, success: bool, now: float | None = None) -> None:
        """Record one fit outcome.

        ``now`` defaults to ``time.time()`` and is overridable in tests
        so the rolling-window expiry is deterministic without
        ``freezegun`` or sleeps.
        """
        ts = time.time() if now is None else now
        with self._lock:
            bucket = self._samples.setdefault((group, model), deque())
            bucket.append(FitSuccessSample(success=success, timestamp=ts))
            self._prune_locked(bucket, ts)

    def render(self, *, now: float | None = None) -> list[SuccessRatioPoint]:
        """Snapshot the success ratio at every configured window.

        Returns one point per (group, model, window) that has at least
        one observation inside the window. Empty windows are skipped —
        a 0/0 ratio is not a "100% success" signal, but emitting NaN
        would just confuse the alert path.
        """
        ts = time.time() if now is None else now
        out: list[SuccessRatioPoint] = []
        with self._lock:
            for (group, model), bucket in self._samples.items():
                self._prune_locked(bucket, ts)
                for window in self._windows_seconds:
                    ratio = _ratio_in_window(bucket, ts, window)
                    if ratio is None:
                        continue
                    out.append(
                        SuccessRatioPoint(
                            group=group,
                            model=model,
                            window_label=_format_window(window),
                            ratio=ratio,
                        )
                    )
        return out

    def retain_groups(self, names: set[str]) -> None:
        """Drop per-(group, model) buckets for groups removed by a reload."""
        with self._lock:
            self._samples = {
                key: bucket for key, bucket in self._samples.items() if key[0] in names
            }

    def retain_models(self, live_models: set[str]) -> None:
        """Drop per-(group, model) buckets for models no longer in the live config.

        Mirrors the staleness sweep's "model dropped from defaults.models"
        cleanup: without this, a model removed from the config keeps its
        success-ratio gauge on /metrics forever (no new observations
        arrive to age out the existing samples, and the alert keeps
        binding to the stale rows).
        """
        with self._lock:
            self._samples = {
                key: bucket for key, bucket in self._samples.items() if key[1] in live_models
            }

    def _prune_locked(self, bucket: deque[FitSuccessSample], now: float) -> None:
        cutoff = now - self._max_window
        while bucket and bucket[0].timestamp < cutoff:
            bucket.popleft()


def _ratio_in_window(
    bucket: deque[FitSuccessSample], now: float, window_seconds: int
) -> float | None:
    cutoff = now - window_seconds
    total = 0
    success = 0
    for sample in bucket:
        if sample.timestamp < cutoff:
            continue
        total += 1
        if sample.success:
            success += 1
    if total == 0:
        return None
    return success / total


class RegressorStabilityTracker:
    """Per-(group, query, regressor_id) drift gauge of recent vs. lookback values.

    The runner pushes the last-seen regressor value series into
    :meth:`record` after each fit; :meth:`render` computes a 0..1
    score where 1.0 means the recent tail matches the lookback head
    perfectly and 0.0 means it has drifted past the saturating
    threshold. The implementation uses the same MAPE-on-quantiles
    flavour the forecast-drift module uses for its baseline
    comparison so the operator-facing semantics agree across the two
    metrics.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._scores: dict[tuple[str, str, str], float] = {}

    def record(
        self,
        *,
        group: str,
        query: str,
        regressor_id: str,
        values: list[float],
        recent_fraction: float = 0.2,
    ) -> None:
        """Record a stability score derived from ``values``.

        The trailing ``recent_fraction`` of the series is compared
        against the leading prefix; a small drift (≤ 10% relative)
        yields a near-1.0 score, a 100%+ drift yields 0. Series too
        short for a meaningful split short-circuit to ``NaN`` (which
        the exporter drops) so a thin regressor doesn't spuriously
        score as "unstable".
        """
        score = compute_stability_score(values, recent_fraction=recent_fraction)
        with self._lock:
            self._scores[(group, query, regressor_id)] = score

    def render(self) -> list[RegressorStabilityPoint]:
        with self._lock:
            return [
                RegressorStabilityPoint(group=g, query=q, regressor_id=r, score=score)
                for (g, q, r), score in self._scores.items()
            ]

    def retain_groups(self, names: set[str]) -> None:
        with self._lock:
            self._scores = {key: score for key, score in self._scores.items() if key[0] in names}

    def retain_queries(self, live_pairs: set[tuple[str, str]]) -> None:
        with self._lock:
            self._scores = {
                key: score for key, score in self._scores.items() if (key[0], key[1]) in live_pairs
            }


def compute_stability_score(
    values: list[float], *, recent_fraction: float = 0.2, saturation: float = 1.0
) -> float:
    """Map "recent vs. lookback" relative shift to a 0..1 score.

    Returns ``NaN`` when the input is too thin to split, when the
    lookback baseline is degenerate (all-zero or non-finite), or
    when every value is non-finite. Otherwise the relative shift is
    ``|mean(recent) - mean(baseline)| / max(|mean(baseline)|, eps)``
    and the score is ``max(0, 1 - shift / saturation)``: a 10%
    relative shift scores 0.9 at ``saturation=1.0``, a 100% shift
    scores 0, and any larger shift saturates at 0.
    """
    finite = [v for v in values if math.isfinite(v)]
    if len(finite) < _MIN_STABILITY_SAMPLES:
        return float("nan")
    cut = max(1, int(len(finite) * recent_fraction))
    recent = finite[-cut:]
    baseline = finite[:-cut]
    if not baseline:
        return float("nan")
    baseline_mean = sum(baseline) / len(baseline)
    if not math.isfinite(baseline_mean):
        return float("nan")
    recent_mean = sum(recent) / len(recent)
    denom = max(abs(baseline_mean), 1e-9)
    shift = abs(recent_mean - baseline_mean) / denom
    if not math.isfinite(shift):
        return float("nan")
    saturation = max(saturation, 1e-9)
    return max(0.0, 1.0 - shift / saturation)


class GroupResourceTracker:
    """Cumulative CPU + most-recent memory snapshot per group.

    CPU is the sum of per-fit wall-clock seconds reported by the
    runner's ``fit_durations`` accumulator. Memory is a cheap
    ``sys.getsizeof``-based estimate of the per-group snapshot the
    exporter is currently holding — best-effort, intended to surface
    "which group's snapshot is dominant" rather than be an exact
    process-level breakdown.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cpu_total: dict[str, float] = {}
        self._memory: dict[str, float] = {}

    def add_cpu(self, group: str, seconds: float) -> None:
        if seconds <= 0 or not math.isfinite(seconds):
            return
        with self._lock:
            self._cpu_total[group] = self._cpu_total.get(group, 0.0) + float(seconds)

    def set_memory(self, group: str, memory_bytes: float) -> None:
        # Drop non-finite, negative, *and* zero values at write time so a
        # missing-snapshot probe (which returns 0.0) doesn't populate the
        # dict only to be filtered at render. Without this the dict
        # accumulates phantom zero entries for every group that hasn't
        # produced a snapshot yet.
        if not math.isfinite(memory_bytes) or memory_bytes <= 0:
            return
        with self._lock:
            self._memory[group] = float(memory_bytes)

    def render(self) -> list[GroupResourcePoint]:
        with self._lock:
            groups = set(self._cpu_total) | set(self._memory)
            return [
                GroupResourcePoint(
                    group=group,
                    memory_bytes=self._memory.get(group, 0.0),
                    cpu_seconds_total=self._cpu_total.get(group, 0.0),
                )
                for group in groups
            ]

    def retain_groups(self, names: set[str]) -> None:
        with self._lock:
            self._cpu_total = {k: v for k, v in self._cpu_total.items() if k in names}
            self._memory = {k: v for k, v in self._memory.items() if k in names}


def _format_window(seconds: int) -> str:
    """Render a duration as the Prometheus-style ``<n><unit>`` label."""
    if seconds <= 0:
        return "0s"
    if seconds % 86_400 == 0:
        return f"{seconds // 86_400}d"
    if seconds % 3_600 == 0:
        return f"{seconds // 3_600}h"
    if seconds % 60 == 0:
        return f"{seconds // 60}m"
    return f"{seconds}s"


__all__ = [
    "DEFAULT_SUCCESS_WINDOWS_SECONDS",
    "FitOutcomeTracker",
    "FitSuccessSample",
    "GroupResourcePoint",
    "GroupResourceTracker",
    "RegressorStabilityPoint",
    "RegressorStabilityTracker",
    "SuccessRatioPoint",
    "compute_stability_score",
]
