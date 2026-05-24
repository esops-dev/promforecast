"""Recurring incident pattern detection.

When a query opts into ``data_profile: events`` the runner routes it
to this module instead of the usual forecast pipeline. The input is
expected to be a counter that ticks on each discrete event
(``pagerduty_incidents_total``,
``alertmanager_notifications_total{severity="critical"}``, a custom
synthetic counter from a webhook). The output is a recurrence
intensity score per ``(day_of_week, hour_of_day)`` bucket, emitted
as ``forecast_incident_recurrence_score`` and rendered as a 7 x 24
heatmap by the bundled dashboard.

Two derivation steps run sequentially:

1. **Counter to event deltas.** The cumulative counter is converted to
   per-step deltas. Negative deltas (counter resets) are dropped —
   they're spurious, not real events. Non-finite samples are dropped.
2. **Bucket and normalise.** Each non-zero delta is attributed to the
   bucket defined by the sample's UTC timestamp:
   ``day_of_week`` in ``[0, 6]`` (Monday = 0, ISO convention) and
   ``hour_of_day`` in ``[0, 23]``. Each bucket's score is the bucket's
   total event count divided by the mean event count across the
   non-empty buckets — values above 1.0 mean "this bucket sees more
   incidents than the average occupied bucket".

The output is always a full 7x24 grid, including zeros, so dashboards
render a consistent heatmap from day 0 — values < 1.0 indicate
"below-average concentration" and values exactly 0.0 indicate "no
events in this bucket during the lookback".

**FFT-based seasonality validation** is an optional follow-up —
confirming that the per-bucket pattern repeats across the lookback
rather than reflecting a single cluster. The shipped detector emits
the heatmap unconditionally so operators can spot the recurrence
visually; a future pass can gate emission on a detected seasonality.

The compute path is pure-function; the runner is responsible for
fetching the counter time series and pushing the resulting points
into the exporter via
:meth:`promforecast.exporter.Exporter.replace_incident_recurrence_for_query`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

import structlog

logger = structlog.get_logger(__name__)

# 7 days x 24 hours = 168 buckets. Fixed cardinality per opted-in
# query — operators can budget this exactly in advance.
_DAYS_PER_WEEK = 7
_HOURS_PER_DAY = 24


@dataclass(frozen=True)
class RecurrenceBucket:
    """One (day_of_week, hour_of_day) cell with its normalised score.

    ``count`` is the absolute event count attributed to this bucket
    over the lookback (informational; not emitted as a label). ``score``
    is the bucket's count divided by the mean count across non-empty
    buckets — > 1 means above-average concentration, 0 means no
    events observed.
    """

    day_of_week: int
    hour_of_day: int
    count: int
    score: float


def counter_to_event_counts(
    *,
    timestamps: list[datetime],
    values: list[float],
) -> list[tuple[datetime, int]]:
    """Convert a cumulative counter into per-step (timestamp, event_count) pairs.

    Counter deltas are computed between consecutive samples. Negative
    deltas (counter resets — counter restarted from zero after a
    scrape target restart) are treated as zero, not as a "negative
    incident". Non-finite samples short-circuit the pair containing
    them, so a single dropped scrape doesn't corrupt the whole
    bucketing pass.

    Floats are rounded to the nearest integer at emission time so a
    counter that ticks fractionally (e.g. a rate-derived input) is
    bucketed sensibly. Each pair carries the *trailing* timestamp,
    so an event detected between t-1 and t is attributed to t — the
    moment the operator's alerting pipeline observed it.

    Returns an empty list when there's fewer than two samples or
    when no positive deltas exist.
    """
    if len(timestamps) < 2 or len(values) != len(timestamps):  # noqa: PLR2004
        return []
    out: list[tuple[datetime, int]] = []
    for i in range(1, len(values)):
        prev_v = values[i - 1]
        curr_v = values[i]
        if not (math.isfinite(prev_v) and math.isfinite(curr_v)):
            continue
        delta = curr_v - prev_v
        if delta <= 0.0:
            continue
        count = round(delta)
        if count <= 0:
            continue
        out.append((timestamps[i], count))
    return out


def bucket_events_by_week_hour(
    events: list[tuple[datetime, int]],
) -> list[RecurrenceBucket]:
    """Bucket events by (day_of_week, hour_of_day) and normalise scores.

    Always returns a full 7x24 grid (168 buckets); empty buckets get
    ``count=0`` and ``score=0.0`` so dashboards render a complete
    heatmap. Bucket order is row-major (day 0 hour 0, day 0 hour 1,
    ..., day 6 hour 23).

    Normalisation: each non-empty bucket's count divided by the mean
    count across all non-empty buckets. With no events the grid is
    all zeros. With exactly one event, that bucket scores 1.0 and
    the rest score 0.0 — a single-event "pattern" reads as "100% of
    activity in this bucket" which is the right interpretation for
    sparse incident streams.

    Time-of-day is read in UTC. Operators with a strong timezone
    preference can rotate the heatmap manually; explicit per-query
    timezone configuration is a documented follow-up.
    """
    counts: dict[tuple[int, int], int] = {}
    for timestamp, count in events:
        day = timestamp.weekday()  # Monday=0..Sunday=6 (Python convention)
        hour = timestamp.hour
        key = (day, hour)
        counts[key] = counts.get(key, 0) + count
    populated = [count for count in counts.values() if count > 0]
    mean = (sum(populated) / len(populated)) if populated else 0.0
    out: list[RecurrenceBucket] = []
    for day in range(_DAYS_PER_WEEK):
        for hour in range(_HOURS_PER_DAY):
            count = counts.get((day, hour), 0)
            score = (count / mean) if (mean > 0 and count > 0) else 0.0
            out.append(
                RecurrenceBucket(
                    day_of_week=day,
                    hour_of_day=hour,
                    count=count,
                    score=score,
                )
            )
    return out


def compute_recurrence(
    *,
    timestamps: list[datetime],
    values: list[float],
) -> list[RecurrenceBucket]:
    """End-to-end: counter samples → normalised 7x24 bucket grid.

    Convenience wrapper combining :func:`counter_to_event_counts` +
    :func:`bucket_events_by_week_hour`. Returns the full grid even
    when no events were observed (every bucket scores 0.0); the
    runner pushes the whole grid so the heatmap stays correctly
    sized over time.
    """
    events = counter_to_event_counts(timestamps=timestamps, values=values)
    return bucket_events_by_week_hour(events)


__all__ = [
    "RecurrenceBucket",
    "bucket_events_by_week_hour",
    "compute_recurrence",
    "counter_to_event_counts",
]
