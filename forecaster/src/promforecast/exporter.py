"""Prometheus exposition: snapshot cache + metric-name validator.

The validator enforces Prometheus naming rules strictly. Colons are reserved
for recording rules and rejected here. There is no escape hatch.

Rendering goes through ``prometheus_client.CollectorRegistry`` via a custom
collector. The collector reads the snapshot under a lock once per scrape and
emits ``GaugeMetricFamily`` / ``CounterMetricFamily`` instances; this keeps
the snapshot model (one replace-per-run per group) but delegates exposition
formatting, label-value escaping, and the ``# TYPE`` / ``# HELP`` ordering
invariants to the well-tested upstream library.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from prometheus_client import CollectorRegistry, generate_latest
from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily

if TYPE_CHECKING:
    from collections.abc import Iterator

_METRIC_NAME_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


class InvalidMetricNameError(ValueError):
    """Raised when an output metric name violates Prometheus conventions."""


def validate_metric_name(name: str) -> None:
    """Reject metric names that are not safe for Prometheus exposition.

    Rules:
        - Must match ``[a-zA-Z_][a-zA-Z0-9_]*``.
        - Must not contain a colon (reserved for recording rules).

    Raises:
        InvalidMetricNameError: If ``name`` does not satisfy the rules.
    """
    if ":" in name:
        raise InvalidMetricNameError(
            f"metric name {name!r} contains a colon; colons are reserved for recording rules"
        )
    if not _METRIC_NAME_RE.match(name):
        raise InvalidMetricNameError(f"metric name {name!r} does not match Prometheus naming regex")


@dataclass(frozen=True)
class ForecastPoint:
    """A single forecast row destined for /metrics."""

    metric: str
    labels: dict[str, str]
    value: float


@dataclass(frozen=True)
class AccuracyPoint:
    """One backtest result row destined for /metrics.

    Rendered as ``forecast_accuracy_mape`` and ``forecast_accuracy_mase``
    gauges; carrying both numbers in one struct avoids two parallel lists
    that would have to be kept in lockstep.
    """

    labels: dict[str, str]
    mape: float
    mase: float


@dataclass(frozen=True)
class DeviationPoint:
    """A single deviation observation destined for /metrics.

    Rendered as ``forecast_deviation_ratio`` and
    ``forecast_deviation_outside_band`` gauges.
    """

    labels: dict[str, str]
    ratio: float
    outside_band: bool


@dataclass(frozen=True)
class QualityPoint:
    """A composite forecast-quality observation destined for /metrics.

    Rendered as ``forecast_quality_score`` (gauge, 0..1).
    """

    labels: dict[str, str]
    score: float


@dataclass
class GroupSnapshot:
    """The most recent successful forecast output for one group."""

    group: str
    points: list[ForecastPoint] = field(default_factory=list)
    accuracy_points: list[AccuracyPoint] = field(default_factory=list)
    deviation_points: list[DeviationPoint] = field(default_factory=list)
    quality_points: list[QualityPoint] = field(default_factory=list)
    last_run_timestamp: float = 0.0
    run_duration_seconds: float = 0.0
    series_count: int = 0
    # series_dropped reasons (bounded set):
    #   per_query_cap, total_cap, min_points
    series_dropped: dict[str, int] = field(default_factory=dict)
    # failures reasons (bounded set):
    #   query_error, fit_error, invalid_metric_name, group_error,
    #   fit_timeout
    failures: dict[str, int] = field(default_factory=dict)
    # Per-(query, model) fit duration in seconds (most recent run). Used to
    # render ``forecast_fit_duration_seconds`` so operators can spot a slow
    # model on a specific query without scraping logs.
    fit_durations: dict[tuple[str, str], float] = field(default_factory=dict)
    # Per-(query, model) slow-fit counter (cumulative). A fit is "slow" when
    # it takes longer than ``safety.fit_timeout * slow_fit_ratio`` (see
    # ``runner.SLOW_FIT_RATIO``). Surfaced as a counter so operators can
    # alert on persistent slowness without losing transient blips.
    slow_fits: dict[tuple[str, str], int] = field(default_factory=dict)


class Exporter:
    """Holds the most recent forecast snapshot and renders /metrics."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._groups: dict[str, GroupSnapshot] = {}
        # Reload telemetry: timestamp (0.0 = never) + last-reload success flag
        # + cumulative failure count (counter; never decreases).
        self._reload_timestamp: float = 0.0
        self._reload_success: bool = True
        self._reload_failures: int = 0
        self._registry = CollectorRegistry()
        self._registry.register(_SnapshotCollector(self))

    def update(self, snapshot: GroupSnapshot) -> None:
        """Replace the snapshot for one group atomically."""
        with self._lock:
            self._groups[snapshot.group] = snapshot

    def increment_failure(self, group: str, reason: str) -> None:
        """Bump the failure counter for ``group``/``reason`` without touching points."""
        with self._lock:
            snap = self._groups.setdefault(group, GroupSnapshot(group=group))
            snap.failures[reason] = snap.failures.get(reason, 0) + 1

    def retain_groups(self, names: set[str]) -> set[str]:
        """Drop snapshots for any group not in ``names``.

        Returns the set of group names that were removed. Used after a config
        reload so /metrics no longer exposes series for groups that have been
        deleted from the YAML.
        """
        with self._lock:
            current = set(self._groups)
            removed = current - names
            for name in removed:
                del self._groups[name]
        return removed

    def set_reload_state(self, *, success: bool, timestamp: float) -> None:
        """Record the most recent reload outcome.

        Surfaces as ``forecast_config_last_reload_timestamp_seconds`` and
        ``forecast_config_reload_success`` so dashboards/alerts can detect
        a config rollback.
        """
        with self._lock:
            self._reload_timestamp = timestamp
            self._reload_success = success

    def increment_reload_failure(self) -> None:
        """Bump the reload failure counter."""
        with self._lock:
            self._reload_failures += 1

    def render(self) -> bytes:
        """Render the current snapshot in Prometheus exposition format."""
        return generate_latest(self._registry)

    def snapshot_for(self, group: str) -> GroupSnapshot | None:
        """Return the most recent snapshot for ``group`` (or ``None``).

        The runner uses this to merge per-(query, model) counter values
        across runs without reaching into ``_groups`` directly. Returns
        the live object — callers must treat it as read-only.
        """
        with self._lock:
            return self._groups.get(group)

    def _snapshot(self) -> list[GroupSnapshot]:
        with self._lock:
            return list(self._groups.values())

    def _reload_state(self) -> tuple[float, bool, int]:
        with self._lock:
            return self._reload_timestamp, self._reload_success, self._reload_failures


class _SnapshotCollector:
    """prometheus_client Collector that streams metrics from an Exporter snapshot."""

    def __init__(self, exporter: Exporter) -> None:
        self._exporter = exporter

    def collect(self) -> Iterator[GaugeMetricFamily | CounterMetricFamily]:
        groups = self._exporter._snapshot()
        reload_state = self._exporter._reload_state()
        yield from _forecast_families(groups)
        yield from _accuracy_families(groups)
        yield from _deviation_families(groups)
        yield from _quality_families(groups)
        yield from _operational_families(groups)
        yield from _reload_families(reload_state)


def _forecast_families(
    groups: list[GroupSnapshot],
) -> Iterator[GaugeMetricFamily]:
    """One GaugeMetricFamily per distinct ``<id>_forecast*`` metric name.

    Within a single forecast metric name (e.g. ``node_cpu_busy_pct_forecast``)
    every sample comes from the same TSDB query, so the label-key set is
    consistent. We compute the union defensively in case future code adds
    optional labels.
    """
    by_metric: dict[str, list[ForecastPoint]] = {}
    for snap in groups:
        for point in snap.points:
            by_metric.setdefault(point.metric, []).append(point)

    for metric_name, points in sorted(by_metric.items()):
        label_keys = sorted({k for p in points for k in p.labels})
        family = GaugeMetricFamily(metric_name, "promforecast forecast value.", labels=label_keys)
        for point in points:
            family.add_metric([point.labels.get(k, "") for k in label_keys], point.value)
        yield family


def _accuracy_families(
    groups: list[GroupSnapshot],
) -> Iterator[GaugeMetricFamily]:
    """Render ``forecast_accuracy_mape`` and ``forecast_accuracy_mase`` gauges.

    NaN MAPE/MASE values (degenerate series, all-zero actuals) are dropped
    rather than emitted: ``NaN`` round-trips through Prometheus exposition
    fine but pollutes alerts and dashboards.
    """
    import math  # noqa: PLC0415

    accuracy_points = [(snap, p) for snap in groups for p in snap.accuracy_points]
    if not accuracy_points:
        return

    label_keys = sorted({k for _, p in accuracy_points for k in p.labels})

    mape_family = GaugeMetricFamily(
        "forecast_accuracy_mape",
        "Mean absolute percentage error from the most recent backtest, in percent.",
        labels=label_keys,
    )
    mase_family = GaugeMetricFamily(
        "forecast_accuracy_mase",
        "Mean absolute scaled error from the most recent backtest.",
        labels=label_keys,
    )

    for _, point in accuracy_points:
        values = [point.labels.get(k, "") for k in label_keys]
        if math.isfinite(point.mape):
            mape_family.add_metric(values, point.mape)
        if math.isfinite(point.mase):
            mase_family.add_metric(values, point.mase)
    yield mape_family
    yield mase_family


def _deviation_families(
    groups: list[GroupSnapshot],
) -> Iterator[GaugeMetricFamily]:
    """Render ``forecast_deviation_ratio`` and ``forecast_deviation_outside_band``.

    NaN ratios (degenerate confidence band) are dropped from the ratio metric
    but still produce a well-defined ``outside_band`` sample, so an alert on
    ``forecast_deviation_outside_band == 1`` works even when the band is
    pathological.
    """
    import math  # noqa: PLC0415

    deviation_points = [(snap, p) for snap in groups for p in snap.deviation_points]
    if not deviation_points:
        return

    label_keys = sorted({k for _, p in deviation_points for k in p.labels})

    ratio_family = GaugeMetricFamily(
        "forecast_deviation_ratio",
        "Signed deviation of the most recent actual from the forecast band, "
        "as (actual - yhat) / (yhat_upper - yhat_lower).",
        labels=label_keys,
    )
    outside_family = GaugeMetricFamily(
        "forecast_deviation_outside_band",
        "1 if the most recent actual is outside the forecast band, else 0.",
        labels=label_keys,
    )

    for _, point in deviation_points:
        values = [point.labels.get(k, "") for k in label_keys]
        if math.isfinite(point.ratio):
            ratio_family.add_metric(values, point.ratio)
        outside_family.add_metric(values, 1.0 if point.outside_band else 0.0)
    yield ratio_family
    yield outside_family


def _quality_families(
    groups: list[GroupSnapshot],
) -> Iterator[GaugeMetricFamily]:
    """Render ``forecast_quality_score`` gauges (0..1 per series/model)."""
    import math  # noqa: PLC0415

    quality_points = [(snap, p) for snap in groups for p in snap.quality_points]
    if not quality_points:
        return

    label_keys = sorted({k for _, p in quality_points for k in p.labels})

    family = GaugeMetricFamily(
        "forecast_quality_score",
        "Composite forecast quality score in [0, 1] combining backtest MAPE, "
        "confidence-band width, and lookback data coverage.",
        labels=label_keys,
    )
    for _, point in quality_points:
        if not math.isfinite(point.score):
            continue
        values = [point.labels.get(k, "") for k in label_keys]
        family.add_metric(values, point.score)
    yield family


def _operational_families(
    groups: list[GroupSnapshot],
) -> Iterator[GaugeMetricFamily | CounterMetricFamily]:
    if not groups:
        return

    last_run = GaugeMetricFamily(
        "forecast_last_run_timestamp_seconds",
        "Unix timestamp of the most recent successful run for the group.",
        labels=["group"],
    )
    duration = GaugeMetricFamily(
        "forecast_run_duration_seconds",
        "Wall-clock duration of the most recent run for the group.",
        labels=["group"],
    )
    series_count = GaugeMetricFamily(
        "forecast_series_count",
        "Number of series accepted by the most recent run for the group.",
        labels=["group"],
    )
    for snap in groups:
        last_run.add_metric([snap.group], snap.last_run_timestamp)
        duration.add_metric([snap.group], snap.run_duration_seconds)
        series_count.add_metric([snap.group], float(snap.series_count))
    yield last_run
    yield duration
    yield series_count

    # CounterMetricFamily appends _total to the name, so we pass the bare stem.
    dropped = CounterMetricFamily(
        "forecast_series_dropped",
        "Series dropped before forecasting, by reason (per_query_cap | total_cap | min_points).",
        labels=["group", "reason"],
    )
    failures = CounterMetricFamily(
        "forecast_failures",
        "Forecast run failures, by reason "
        "(query_error | query_timeout | fit_error | fit_timeout | "
        "invalid_metric_name | group_error).",
        labels=["group", "reason"],
    )
    for snap in groups:
        for reason, count in snap.series_dropped.items():
            dropped.add_metric([snap.group, reason], float(count))
        for reason, count in snap.failures.items():
            failures.add_metric([snap.group, reason], float(count))
    yield dropped
    yield failures

    fit_duration = GaugeMetricFamily(
        "forecast_fit_duration_seconds",
        "Wall-clock duration of the most recent fit for the (group, query, model) tuple.",
        labels=["group", "query", "model"],
    )
    slow_fits = CounterMetricFamily(
        "forecast_slow_fits",
        "Number of fits that exceeded the slow-fit threshold "
        "(safety.fit_timeout * runner.SLOW_FIT_RATIO).",
        labels=["group", "query", "model"],
    )
    have_fit_data = False
    for snap in groups:
        for (query_id, model_name), duration_s in snap.fit_durations.items():
            fit_duration.add_metric([snap.group, query_id, model_name], duration_s)
            have_fit_data = True
        for (query_id, model_name), count in snap.slow_fits.items():
            slow_fits.add_metric([snap.group, query_id, model_name], float(count))
            have_fit_data = True
    if have_fit_data:
        yield fit_duration
        yield slow_fits


def _reload_families(
    state: tuple[float, bool, int],
) -> Iterator[GaugeMetricFamily | CounterMetricFamily]:
    """Render config-reload telemetry.

    ``forecast_config_last_reload_timestamp_seconds`` is 0 when the config has
    never been reloaded since boot. ``forecast_config_reload_success`` carries
    the last-reload outcome (1 = applied, 0 = rolled back). The
    ``forecast_config_reload_failures_total`` counter never decreases, so
    alerts can detect transient reload failures even if a later reload
    succeeds and flips ``..._reload_success`` back to 1.
    """
    timestamp, success, failures = state
    last_reload = GaugeMetricFamily(
        "forecast_config_last_reload_timestamp_seconds",
        "Unix timestamp of the most recent config reload attempt; 0 if never reloaded.",
    )
    last_reload.add_metric([], timestamp)
    yield last_reload

    success_family = GaugeMetricFamily(
        "forecast_config_reload_success",
        "1 if the most recent config reload was applied, 0 if it was rolled back.",
    )
    success_family.add_metric([], 1.0 if success else 0.0)
    yield success_family

    failure_counter = CounterMetricFamily(
        "forecast_config_reload_failures",
        "Total number of config reload attempts that failed validation and were rolled back.",
    )
    failure_counter.add_metric([], float(failures))
    yield failure_counter
