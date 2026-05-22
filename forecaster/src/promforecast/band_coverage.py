"""Rolling empirical band-coverage diagnostics.

When operators set bands at ``95%`` they expect roughly 95% of actuals to
land inside. In practice the nominal level can diverge a lot from the
empirical level — especially before conformal calibration is on, and as
a confirmation signal that calibration is working when it is on.

This module keeps a small ring buffer per (series, level, window) of
recent ``actual ∈ band`` booleans and reports the fraction inside the
band as a single gauge. The exporter holds the state; the runner feeds
new observations after every fit.

Memory cost is bounded by ``window / observation_period`` booleans per
(series, level, window) — the runner records one observation per fit,
and the caller passes the inter-fit refresh interval as the
``step`` argument to :meth:`BandCoverageTracker.observe`. With the
default windows ``1d`` + ``7d``, a 1h refresh, and 5000 series at two
levels, that's ``(24 + 168) * 5000 * 2 ≈ 1.9M bits = ~230 KB`` — a
fraction of a megabyte.

Sibling-project label alignment: the ``level`` and ``window`` labels
on ``forecast_band_empirical_coverage`` are diagnostic-only and do not
have a counterpart in the sibling ``promanomaly`` exposition. Because
the metric describes the calibration of *the forecaster's own band* it
is not expected to align with any anomaly-side metric; users wanting to
combine the two should join on ``id`` and ``group`` and ignore
``level``/``window`` on the forecaster side.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import timedelta


@dataclass(frozen=True)
class CoverageSample:
    """A rendered empirical-coverage sample for one (series, model, level, window)."""

    labels: dict[str, str]
    fraction: float
    samples: int


class BandCoverageTracker:
    """Maintains per-key rolling boolean buffers and reports coverage.

    Keys are ``(group, query, series_key, model, level, window_label)``.
    Each bucket is a ``deque`` with a maxlen sized to ``window / step``.
    A new observation is recorded as ``True`` (inside) or ``False``
    (outside); the gauge reports ``inside / total``.

    The tracker is per-replica and in-memory; HA followers do not see
    the rolling state but they read the leader's snapshot of the
    rendered gauge from the shared cache, so /metrics agrees across
    replicas without sharing the underlying buffers.

    ``forget`` is invoked from the runner's config-reload path to drop
    state for groups or queries that no longer exist in the live YAML.
    """

    def __init__(self) -> None:
        # Bucket of recent (inside) booleans keyed by the label tuple.
        # The shape ``(window_seconds, step_seconds)`` is stored alongside
        # so a bucket gets reset when *either* changes, not just when the
        # derived ``maxlen`` happens to differ — two (window, step) pairs
        # can yield the same ``maxlen`` (e.g. 1d/1h and 24h/1h) which the
        # ring would otherwise treat as unchanged.
        self._buckets: dict[
            tuple[str, str, str, str, int, str], tuple[int, int, deque[bool]]
        ] = {}

    def observe(
        self,
        *,
        group: str,
        query: str,
        series_key: str,
        model: str,
        level: int,
        window_label: str,
        window: timedelta,
        step: timedelta,
        inside: bool,
    ) -> None:
        """Record one (inside / outside) observation for the rolling buffer.

        ``window / step`` controls the buffer size. We clamp to ``1`` so a
        misconfigured step doesn't crash with a zero-length deque.
        """
        step_seconds = max(1, int(step.total_seconds()))
        window_seconds = max(step_seconds, int(window.total_seconds()))
        maxlen = max(1, window_seconds // step_seconds)
        key = (group, query, series_key, model, level, window_label)
        existing = self._buckets.get(key)
        if existing is None or (existing[0], existing[1]) != (window_seconds, step_seconds):
            # Either first observation or the operator changed the
            # window/step; reset rather than truncate so a resized
            # window doesn't carry stale data forward.
            bucket: deque[bool] = deque(maxlen=maxlen)
        else:
            bucket = existing[2]
        bucket.append(inside)
        self._buckets[key] = (window_seconds, step_seconds, bucket)

    def samples(self) -> list[CoverageSample]:
        """Render the current snapshot of per-key coverage gauges."""
        out: list[CoverageSample] = []
        for key, (_window_seconds, _step_seconds, bucket) in self._buckets.items():
            if not bucket:
                continue
            group, query, series_key, model, level, window_label = key
            del series_key  # Currently dropped from emitted labels; see docstring.
            inside = sum(1 for v in bucket if v)
            fraction = inside / len(bucket)
            labels: dict[str, str] = {
                "id": query,
                "group": group,
                "model": model,
                "level": str(level),
                "window": window_label,
            }
            out.append(
                CoverageSample(
                    labels=labels,
                    fraction=fraction,
                    samples=len(bucket),
                )
            )
        return out

    def forget(self, live_pairs: set[tuple[str, str]]) -> None:
        """Drop buffers whose (group, query) no longer exists in the config."""
        self._buckets = {k: v for k, v in self._buckets.items() if (k[0], k[1]) in live_pairs}
