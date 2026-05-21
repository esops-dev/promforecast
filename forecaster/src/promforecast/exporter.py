"""Prometheus exposition: snapshot cache.

Rendering goes through ``prometheus_client.CollectorRegistry`` via a custom
collector. The collector reads the snapshot under a lock once per scrape and
emits ``GaugeMetricFamily`` / ``CounterMetricFamily`` instances; this keeps
the snapshot model (one replace-per-run per group) but delegates exposition
formatting, label-value escaping, and the ``# TYPE`` / ``# HELP`` ordering
invariants to the well-tested upstream library.

The metric-name validator lives in :mod:`promforecast._naming` so that
``config.py`` (and CI tooling that only needs schema validation) can pull
it in without the ``prometheus_client`` dependency. ``InvalidMetricNameError``
and ``validate_metric_name`` are re-exported from here as a backwards-
compatibility shim.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from prometheus_client import CollectorRegistry, generate_latest
from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily

from ._naming import InvalidMetricNameError, validate_metric_name

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

__all__ = [
    "AccuracyPoint",
    "CalibrationOffsetPoint",
    "ContributionPoint",
    "DeviationPoint",
    "EnsembleWeightPoint",
    "Exporter",
    "ForecastPoint",
    "GroupSnapshot",
    "InvalidMetricNameError",
    "PreprocessStats",
    "QualityPoint",
    "validate_metric_name",
]


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


@dataclass(frozen=True)
class ContributionPoint:
    """Per-regressor explainability sample.

    Rendered as ``<id>_forecast_contribution_ratio`` (gauge, 0..1) so the
    metric name carries the primary metric's identity, and the
    ``regressor`` label distinguishes contributors. Ratios across all
    regressors of a series sum to 1 when at least one regressor produced a
    finite correlation.
    """

    metric: str
    labels: dict[str, str]
    value: float


@dataclass(frozen=True)
class EnsembleWeightPoint:
    """Per-model contribution weight inside an ensemble forecast.

    Rendered as ``forecast_ensemble_weight`` (gauge, 0..1). One sample per
    (series, model) when ``accuracy.auto_select == "ensemble"``; absent
    otherwise.
    """

    labels: dict[str, str]
    weight: float


@dataclass(frozen=True)
class CalibrationOffsetPoint:
    """Additive band correction applied by split-conformal calibration.

    Rendered as ``forecast_band_calibration_offset{id, group, model, level}``
    (gauge, units: same as the underlying metric). An offset of 0 means the
    parametric band already covered its nominal level on the calibration
    hold-out; a positive offset is how much each side of the band was
    widened. Emitted only when ``accuracy.conformal.enabled`` is true.
    """

    labels: dict[str, str]
    offset: float


@dataclass
class PreprocessStats:
    """Per-(group, query) preprocessing telemetry.

    The runner accumulates one of these per query per run; the exporter
    folds them into a small set of counters and gauges:

    * ``forecast_input_gaps_filled_total{strategy}`` — per imputation
      strategy that actually ran.
    * ``forecast_input_max_gap_seconds`` — largest gap observed in the
      lookback this run (0 when none).
    * ``forecast_input_outliers_replaced_total{method}`` — points replaced
      by the outlier scrubber.
    * ``forecast_input_change_point_total`` — detected structural breaks.
    * ``forecast_input_effective_lookback_seconds`` — post-truncation
      lookback span; equals the full lookback when no change-point was
      detected or the action was ``reweight_recent``.
    """

    gaps_filled: dict[str, int] = field(default_factory=dict)
    max_gap_seconds: float = 0.0
    outliers_replaced: dict[str, int] = field(default_factory=dict)
    change_points: int = 0
    effective_lookback_seconds: float = 0.0


@dataclass
class GroupSnapshot:
    """The most recent successful forecast output for one group."""

    group: str
    points: list[ForecastPoint] = field(default_factory=list)
    accuracy_points: list[AccuracyPoint] = field(default_factory=list)
    deviation_points: list[DeviationPoint] = field(default_factory=list)
    quality_points: list[QualityPoint] = field(default_factory=list)
    ensemble_weights: list[EnsembleWeightPoint] = field(default_factory=list)
    contribution_points: list[ContributionPoint] = field(default_factory=list)
    last_run_timestamp: float = 0.0
    run_duration_seconds: float = 0.0
    series_count: int = 0
    # series_dropped reasons (bounded set):
    #   per_query_cap, total_cap, min_points, discovery_cap, canary,
    #   preprocess_drop
    series_dropped: dict[str, int] = field(default_factory=dict)
    # failures reasons (bounded set):
    #   query_error, fit_error, invalid_metric_name, group_error,
    #   fit_timeout
    failures: dict[str, int] = field(default_factory=dict)
    # Per-(query, regressor_id, reason) regressor-resolution failure
    # counter. Surfaces as ``forecast_regressor_failures_total`` so a flaky
    # exogenous regressor can be alerted on without polluting the generic
    # ``forecast_failures_total`` series.
    regressor_failures: dict[tuple[str, str, str], int] = field(default_factory=dict)
    # Per-(query, model) fit duration in seconds (most recent run). Used to
    # render ``forecast_fit_duration_seconds`` so operators can spot a slow
    # model on a specific query without scraping logs.
    fit_durations: dict[tuple[str, str], float] = field(default_factory=dict)
    # Per-(query, model) slow-fit counter (cumulative). A fit is "slow" when
    # it takes longer than ``safety.fit_timeout * slow_fit_ratio`` (see
    # ``runner.SLOW_FIT_RATIO``). Surfaced as a counter so operators can
    # alert on persistent slowness without losing transient blips.
    slow_fits: dict[tuple[str, str], int] = field(default_factory=dict)
    # Per-(series, model, level) conformal calibration offsets from the
    # most recent run. Gauge — only the latest run's samples belong on
    # /metrics; series that disappear from the config disappear from the
    # exposition until they reappear.
    calibration_offsets: list[CalibrationOffsetPoint] = field(default_factory=list)
    # Per-(query) preprocessing statistics for the most recent run.
    # Counters (gaps_filled, outliers_replaced, change_points) are merged
    # with the previous snapshot so the exposed counter never decreases;
    # gauges (max_gap_seconds, effective_lookback_seconds) reflect only
    # the most recent observation.
    preprocess_stats: dict[str, PreprocessStats] = field(default_factory=dict)
    # Cumulative per-(query, strategy) gap-fill counter. Same monotonic
    # treatment as slow_fits: merged with previous snapshot at run end.
    gaps_filled_cumulative: dict[tuple[str, str], int] = field(default_factory=dict)
    # Cumulative per-(query, method) outlier-replacement counter.
    outliers_replaced_cumulative: dict[tuple[str, str], int] = field(default_factory=dict)
    # Cumulative per-(query) change-point detection counter.
    change_points_cumulative: dict[str, int] = field(default_factory=dict)


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
        # SHA-256 of the currently-loaded config file, hex-encoded. Empty
        # until the first successful load. Surfaces as the
        # ``forecast_config_hash{sha256="..."}`` info-style gauge so dashboards
        # can confirm a deployment is running the expected revision.
        self._config_hash: str = ""
        # Per-(group, reason) push-sink failure counter. The reason set is
        # bounded by the sink (``timeout``, ``http_error``) so cardinality
        # is small and predictable.
        self._remote_write_failures: dict[tuple[str, str], int] = {}
        # Per-(group, reason) push-sink retry counter. Bumped *during* a
        # push (before the SinkError surfaces), so an operator can spot a
        # flaky TSDB that retries succeed against — without the counter
        # only firing on terminal failures, intermittent latency would
        # be invisible until it tipped over into outright failure.
        self._remote_write_retries: dict[tuple[str, str], int] = {}
        # Per-reason source (PromQL) retry counter. The source is shared
        # across groups so this isn't labelled by group; per-group query
        # latency is already observable via ``forecast_run_duration_seconds``.
        self._source_retries: dict[str, int] = {}
        # Leader election state. ``_leader_identity`` is the local replica
        # name (the value the ``instance`` label takes); ``_is_leader`` is
        # the gauge value (1 = elected, 0 = follower / disabled).
        self._leader_identity: str = ""
        self._is_leader: bool = False
        self._leader_transitions: int = 0
        # Query cache counters (hits / misses / errors). Always emitted so
        # dashboards can chart a zero series before the first query runs.
        self._query_cache_events: dict[str, int] = {}
        # Per-(group, query, variable, reason) discovery failure counter.
        self._discovery_failures: dict[tuple[str, str, str, str], int] = {}
        # Per-(group, query) expansion count (gauge): how many concrete
        # PromQL fetches the most recent run derived from this template.
        self._discovery_expansions: dict[tuple[str, str], int] = {}
        # Per-(group, query, source) season_length auto-detection counter.
        # ``source`` is bounded to {detected, snapped, fallback, too_short}
        # so cardinality is small and predictable. Useful for alerting on
        # "the detector is always falling back on this query" without
        # scraping per-series logs.
        self._season_length_detections: dict[tuple[str, str, str], int] = {}
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

    def set_config_hash(self, sha256_hex: str) -> None:
        """Replace the recorded config hash.

        ``sha256_hex`` is rendered as the ``sha256`` label on
        ``forecast_config_hash``; an empty string disables emission of the
        family entirely. The exporter doesn't validate that the value really
        is a hex digest — callers compute it from raw config bytes and pass
        it through untouched.
        """
        with self._lock:
            self._config_hash = sha256_hex

    def increment_remote_write_failure(self, group: str, reason: str) -> None:
        """Bump the remote_write push-sink failure counter for ``group``."""
        with self._lock:
            key = (group, reason)
            self._remote_write_failures[key] = self._remote_write_failures.get(key, 0) + 1

    def increment_remote_write_retry(self, group: str, reason: str) -> None:
        """Bump the remote_write push-sink retry counter for ``group``.

        Distinct from the failure counter: a retry is bumped on *every*
        retry attempt, including ones that eventually succeed. The
        failure counter only ticks on terminal failure, so without this
        a flaky TSDB papered-over by retries would be invisible.
        """
        with self._lock:
            key = (group, reason)
            self._remote_write_retries[key] = self._remote_write_retries.get(key, 0) + 1

    def increment_source_retry(self, reason: str) -> None:
        """Bump the PromQL-source retry counter for ``reason``.

        Reason is one of ``transport`` (connection-level error) or
        ``http_5xx`` (5xx response). 4xx responses fail fast and are not
        retried; they surface as ``forecast_failures_total{reason="query_error"}``.
        """
        with self._lock:
            self._source_retries[reason] = self._source_retries.get(reason, 0) + 1

    def set_leader_state(self, *, identity: str, is_leader: bool) -> None:
        """Record this replica's election state.

        Surfaces as ``forecast_leader_elected{instance="<identity>"}``
        (1 = leader, 0 = follower). The identity is set even when no
        election is configured (single-replica deployments) so a
        join-on-instance dashboard works uniformly.
        """
        with self._lock:
            if identity != self._leader_identity:
                self._leader_identity = identity
            if is_leader != self._is_leader:
                self._leader_transitions += 1
                self._is_leader = is_leader

    def increment_query_cache_event(self, kind: str) -> None:
        """Bump a query-cache counter for ``kind`` (hit | miss | error)."""
        with self._lock:
            self._query_cache_events[kind] = self._query_cache_events.get(kind, 0) + 1

    def record_discovery(
        self,
        *,
        group: str,
        query: str,
        expansions: int,
        failures: dict[tuple[str, str], int] | None = None,
    ) -> None:
        """Record the result of a discovery / template expansion pass.

        ``expansions`` is a gauge (the count from the most recent run).
        ``failures`` increments the per-(variable, reason) counter so a
        flaky discovery query is visible without log scraping.
        """
        with self._lock:
            self._discovery_expansions[(group, query)] = expansions
            if failures:
                for (variable, reason), count in failures.items():
                    key = (group, query, variable, reason)
                    self._discovery_failures[key] = self._discovery_failures.get(key, 0) + count

    def retain_discovery_groups(self, names: set[str]) -> None:
        """Drop discovery records for groups not in ``names``.

        Called after a config reload removes groups; without this their
        expansion gauges would linger on /metrics forever.
        """
        with self._lock:
            self._discovery_expansions = {
                k: v for k, v in self._discovery_expansions.items() if k[0] in names
            }
            self._discovery_failures = {
                k: v for k, v in self._discovery_failures.items() if k[0] in names
            }
            self._season_length_detections = {
                k: v for k, v in self._season_length_detections.items() if k[0] in names
            }

    def increment_season_length_detection(self, group: str, query: str, source: str) -> None:
        """Bump the auto-seasonality detection counter for one outcome.

        ``source`` is one of ``detected``, ``snapped``, ``fallback``,
        ``too_short`` — the bounded set produced by
        :func:`promforecast.seasonality.detect_season_length`. Any other
        token would inflate cardinality; callers must use the bounded set.
        """
        with self._lock:
            key = (group, query, source)
            self._season_length_detections[key] = self._season_length_detections.get(key, 0) + 1

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

    def _config_hash_state(self) -> str:
        with self._lock:
            return self._config_hash

    def _remote_write_state(self) -> dict[tuple[str, str], int]:
        with self._lock:
            return dict(self._remote_write_failures)

    def _remote_write_retries_state(self) -> dict[tuple[str, str], int]:
        with self._lock:
            return dict(self._remote_write_retries)

    def _source_retries_state(self) -> dict[str, int]:
        with self._lock:
            return dict(self._source_retries)

    def _leader_state(self) -> tuple[str, bool, int]:
        with self._lock:
            return self._leader_identity, self._is_leader, self._leader_transitions

    def _query_cache_state(self) -> dict[str, int]:
        with self._lock:
            return dict(self._query_cache_events)

    def _discovery_state(
        self,
    ) -> tuple[dict[tuple[str, str], int], dict[tuple[str, str, str, str], int]]:
        with self._lock:
            return dict(self._discovery_expansions), dict(self._discovery_failures)

    def _season_length_state(self) -> dict[tuple[str, str, str], int]:
        with self._lock:
            return dict(self._season_length_detections)


class _SnapshotCollector:
    """prometheus_client Collector that streams metrics from an Exporter snapshot."""

    def __init__(self, exporter: Exporter) -> None:
        self._exporter = exporter

    def collect(self) -> Iterator[GaugeMetricFamily | CounterMetricFamily]:
        groups = self._exporter._snapshot()
        reload_state = self._exporter._reload_state()
        remote_write_state = self._exporter._remote_write_state()
        remote_write_retries = self._exporter._remote_write_retries_state()
        source_retries = self._exporter._source_retries_state()
        config_hash = self._exporter._config_hash_state()
        leader_state = self._exporter._leader_state()
        cache_state = self._exporter._query_cache_state()
        discovery_state = self._exporter._discovery_state()
        season_length_state = self._exporter._season_length_state()
        yield from _forecast_families(groups)
        yield from _accuracy_families(groups)
        yield from _deviation_families(groups)
        yield from _quality_families(groups)
        yield from _ensemble_families(groups)
        yield from _contribution_families(groups)
        yield from _calibration_families(groups)
        yield from _preprocess_families(groups)
        yield from _operational_families(groups)
        yield from _reload_families(reload_state)
        yield from _config_hash_families(config_hash)
        yield from _remote_write_families(remote_write_state)
        yield from _remote_write_retries_families(remote_write_retries)
        yield from _source_retries_families(source_retries)
        yield from _leader_families(leader_state)
        yield from _query_cache_families(cache_state)
        yield from _discovery_families(discovery_state)
        yield from _season_length_families(season_length_state)


def _emit_gauge_family(
    points: list[Any],
    metric_name: str,
    description: str,
    value_fn: Callable[[Any], float],
    *,
    skip_fn: Callable[[Any], bool] | None = None,
) -> Iterator[GaugeMetricFamily]:
    """Build one GaugeMetricFamily from a flat list of points.

    Each metric family in the snapshot follows the same recipe — derive the
    union of label keys (defensively, in case future code adds optional
    labels), build the family, then iterate. ``value_fn`` extracts the
    sample value; ``skip_fn`` (optional) drops points that should not be
    emitted at all (e.g. NaN ratios).

    Yields nothing when the input is empty so callers can ``yield from``
    unconditionally.

    Single-pass: collect label keys and surviving points together so
    skipped rows aren't iterated twice and the label-key union runs while
    we're already touching each point's dict.
    """
    if not points:
        return
    label_keys_set: set[str] = set()
    selected: list[Any] = []
    for point in points:
        if skip_fn is not None and skip_fn(point):
            continue
        label_keys_set.update(point.labels)
        selected.append(point)
    if not selected:
        return
    label_keys = sorted(label_keys_set)
    family = GaugeMetricFamily(metric_name, description, labels=label_keys)
    for point in selected:
        family.add_metric([point.labels.get(k, "") for k in label_keys], value_fn(point))
    yield family


def _forecast_families(
    groups: list[GroupSnapshot],
) -> Iterator[GaugeMetricFamily]:
    """One GaugeMetricFamily per distinct ``<id>_forecast*`` metric name.

    Within a single forecast metric name (e.g. ``node_cpu_busy_pct_forecast``)
    every sample comes from the same TSDB query, so the label-key set is
    consistent. We bucket by metric name first, then defer to
    :func:`_emit_gauge_family` for the per-bucket rendering.
    """
    by_metric: dict[str, list[ForecastPoint]] = {}
    for snap in groups:
        for point in snap.points:
            by_metric.setdefault(point.metric, []).append(point)

    for metric_name, points in sorted(by_metric.items()):
        yield from _emit_gauge_family(
            points,
            metric_name,
            "promforecast forecast value.",
            value_fn=lambda p: p.value,
        )


def _accuracy_families(
    groups: list[GroupSnapshot],
) -> Iterator[GaugeMetricFamily]:
    """Render ``forecast_accuracy_mape`` and ``forecast_accuracy_mase`` gauges.

    NaN MAPE/MASE values (degenerate series, all-zero actuals) are dropped
    rather than emitted: ``NaN`` round-trips through Prometheus exposition
    fine but pollutes alerts and dashboards.
    """
    points = [p for snap in groups for p in snap.accuracy_points]
    yield from _emit_gauge_family(
        points,
        "forecast_accuracy_mape",
        "Mean absolute percentage error from the most recent backtest, in percent.",
        value_fn=lambda p: p.mape,
        skip_fn=lambda p: not math.isfinite(p.mape),
    )
    yield from _emit_gauge_family(
        points,
        "forecast_accuracy_mase",
        "Mean absolute scaled error from the most recent backtest.",
        value_fn=lambda p: p.mase,
        skip_fn=lambda p: not math.isfinite(p.mase),
    )


def _deviation_families(
    groups: list[GroupSnapshot],
) -> Iterator[GaugeMetricFamily]:
    """Render ``forecast_deviation_ratio`` and ``forecast_deviation_outside_band``.

    NaN ratios (degenerate confidence band) are dropped from the ratio metric
    but still produce a well-defined ``outside_band`` sample, so an alert on
    ``forecast_deviation_outside_band == 1`` works even when the band is
    pathological.
    """
    points = [p for snap in groups for p in snap.deviation_points]
    yield from _emit_gauge_family(
        points,
        "forecast_deviation_ratio",
        "Signed deviation of the most recent actual from the forecast band, "
        "as (actual - yhat) / (yhat_upper - yhat_lower).",
        value_fn=lambda p: p.ratio,
        skip_fn=lambda p: not math.isfinite(p.ratio),
    )
    yield from _emit_gauge_family(
        points,
        "forecast_deviation_outside_band",
        "1 if the most recent actual is outside the forecast band, else 0.",
        value_fn=lambda p: 1.0 if p.outside_band else 0.0,
    )


def _quality_families(
    groups: list[GroupSnapshot],
) -> Iterator[GaugeMetricFamily]:
    """Render ``forecast_quality_score`` gauges (0..1 per series/model)."""
    points = [p for snap in groups for p in snap.quality_points]
    yield from _emit_gauge_family(
        points,
        "forecast_quality_score",
        "Composite forecast quality score in [0, 1] combining backtest MAPE, "
        "confidence-band width, and lookback data coverage.",
        value_fn=lambda p: p.score,
        skip_fn=lambda p: not math.isfinite(p.score),
    )


def _ensemble_families(
    groups: list[GroupSnapshot],
) -> Iterator[GaugeMetricFamily]:
    """Render ``forecast_ensemble_weight`` gauges (one per component model).

    Only populated when ``accuracy.auto_select == "ensemble"``. Weights are
    inverse-MAPE normalised to sum to 1; non-finite samples are dropped.
    """
    points = [p for snap in groups for p in snap.ensemble_weights]
    yield from _emit_gauge_family(
        points,
        "forecast_ensemble_weight",
        "Inverse-MAPE component weight inside an ensemble forecast (sum = 1 per series).",
        value_fn=lambda p: p.weight,
        skip_fn=lambda p: not math.isfinite(p.weight),
    )


def _contribution_families(
    groups: list[GroupSnapshot],
) -> Iterator[GaugeMetricFamily]:
    """Render ``<id>_forecast_contribution_ratio`` gauges, one family per id.

    The metric name is derived from the primary metric's id (the same shape
    forecast points use), and the ``regressor`` label discriminates the
    contributors. Bucketing by metric name first matches the forecast-family
    rendering and keeps label sets uniform within each family.
    """
    by_metric: dict[str, list[ContributionPoint]] = {}
    for snap in groups:
        for point in snap.contribution_points:
            by_metric.setdefault(point.metric, []).append(point)
    for metric_name, points in sorted(by_metric.items()):
        yield from _emit_gauge_family(
            points,
            metric_name,
            "Per-regressor explainability share for the forecast in [0, 1]; "
            "ratios across regressors of a series sum to 1 when finite.",
            value_fn=lambda p: p.value,
            skip_fn=lambda p: not math.isfinite(p.value),
        )


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
        "Series dropped before forecasting, by reason "
        "(per_query_cap | total_cap | min_points | discovery_cap | canary | preprocess_drop).",
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

    regressor_failures = CounterMetricFamily(
        "forecast_regressor_failures",
        "Exogenous regressor resolution failures, by reason "
        "(query_error | query_timeout | empty_result | invalid_type | no_source).",
        labels=["group", "query", "regressor", "reason"],
    )
    have_regressor_failures = False
    for snap in groups:
        for (query_id, regressor_id, reason), count in snap.regressor_failures.items():
            regressor_failures.add_metric(
                [snap.group, query_id, regressor_id, reason], float(count)
            )
            have_regressor_failures = True
    if have_regressor_failures:
        yield regressor_failures


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


def _config_hash_families(
    sha256_hex: str,
) -> Iterator[GaugeMetricFamily]:
    """Render ``forecast_config_hash{sha256="..."} 1.0``.

    Skipped entirely when no config has been loaded yet (the empty-hash
    case). Info-style gauges are rendered as a constant ``1`` so dashboards
    can join on the label without doing arithmetic on the value, which is
    the established Prometheus convention for "info" metrics.
    """
    if not sha256_hex:
        return
    family = GaugeMetricFamily(
        "forecast_config_hash",
        "SHA-256 hash of the currently-loaded config file (info-style; "
        "value is always 1, the hash lives on the sha256 label).",
        labels=["sha256"],
    )
    family.add_metric([sha256_hex], 1.0)
    yield family


def _remote_write_families(
    state: dict[tuple[str, str], int],
) -> Iterator[CounterMetricFamily]:
    """Render the ``forecast_remote_write_failures_total`` counter.

    Always emitted (even when empty) so dashboards can display "0 errors"
    rather than 404'ing the panel before the first failure occurs.
    """
    counter = CounterMetricFamily(
        "forecast_remote_write_failures",
        "Total number of remote_write push attempts that failed, by reason (timeout | http_error).",
        labels=["group", "reason"],
    )
    for (group, reason), count in state.items():
        counter.add_metric([group, reason], float(count))
    yield counter


def _remote_write_retries_families(
    state: dict[tuple[str, str], int],
) -> Iterator[CounterMetricFamily]:
    """Render the ``forecast_remote_write_retries_total`` counter.

    Bumped on every retry attempt — including ones that eventually
    succeed — so a flaky TSDB papered over by retries is visible before
    it tips into outright failure. Always emitted so dashboards have a
    series to chart from boot.
    """
    counter = CounterMetricFamily(
        "forecast_remote_write_retries",
        "Total number of remote_write push attempts that were retried, by reason "
        "(timeout | http_error). Includes retries that eventually succeeded.",
        labels=["group", "reason"],
    )
    for (group, reason), count in state.items():
        counter.add_metric([group, reason], float(count))
    yield counter


def _source_retries_families(
    state: dict[str, int],
) -> Iterator[CounterMetricFamily]:
    """Render the ``forecast_source_retries_total`` counter.

    Tracks PromQL-side retries (transport errors, 5xx). Not labelled by
    group: the source client is shared, and per-group query latency is
    already observable via ``forecast_run_duration_seconds``.
    """
    counter = CounterMetricFamily(
        "forecast_source_retries",
        "Total number of PromQL fetches that were retried, by reason "
        "(transport | http_5xx). Includes retries that eventually succeeded.",
        labels=["reason"],
    )
    for reason, count in state.items():
        counter.add_metric([reason], float(count))
    yield counter


def _leader_families(
    state: tuple[str, bool, int],
) -> Iterator[GaugeMetricFamily | CounterMetricFamily]:
    """Render leader-election state.

    ``forecast_leader_elected{instance="..."}`` is 1 for the elected
    leader, 0 otherwise. ``forecast_leader_transitions_total`` counts how
    many times this replica's local state flipped — alerts on rapid
    transitions catch lease thrash.

    Both are emitted only after :meth:`Exporter.set_leader_state` has been
    called at least once. Without the gate, single-replica deployments
    (where leader election is disabled) would emit a meaningless
    ``instance=""`` series.
    """
    identity, is_leader, transitions = state
    if not identity:
        return
    gauge = GaugeMetricFamily(
        "forecast_leader_elected",
        "1 if this replica is the elected leader, 0 if it is a follower or election is disabled.",
        labels=["instance"],
    )
    gauge.add_metric([identity], 1.0 if is_leader else 0.0)
    yield gauge
    counter = CounterMetricFamily(
        "forecast_leader_transitions",
        "Number of times this replica's leadership state changed since boot.",
        labels=["instance"],
    )
    counter.add_metric([identity], float(transitions))
    yield counter


def _query_cache_families(
    state: dict[str, int],
) -> Iterator[CounterMetricFamily]:
    """Render the ``forecast_query_cache_events_total`` counter.

    Tracks hits, misses, and errors. Always emitted (with zero values when
    nothing has happened) so dashboards can chart a flat-zero baseline
    before the cache sees its first query.
    """
    counter = CounterMetricFamily(
        "forecast_query_cache_events",
        "PromQL query-cache events, by kind (hit | miss | error).",
        labels=["kind"],
    )
    for kind in ("hit", "miss", "error"):
        counter.add_metric([kind], float(state.get(kind, 0)))
    yield counter


def _discovery_families(
    state: tuple[dict[tuple[str, str], int], dict[tuple[str, str, str, str], int]],
) -> Iterator[GaugeMetricFamily | CounterMetricFamily]:
    """Render dynamic-discovery expansion telemetry.

    ``forecast_discovery_expansions{group, query}`` is a gauge with the
    most recent expansion count; ``forecast_discovery_failures_total``
    counts per-variable resolution failures. Emitted only when at least
    one templated query has been seen this run.
    """
    expansions, failures = state
    if expansions:
        gauge = GaugeMetricFamily(
            "forecast_discovery_expansions",
            "Number of concrete queries the discovery template fanned out to "
            "in the most recent run.",
            labels=["group", "query"],
        )
        for (group, query), count in expansions.items():
            gauge.add_metric([group, query], float(count))
        yield gauge
    if failures:
        counter = CounterMetricFamily(
            "forecast_discovery_failures",
            "Discovery variable resolution failures, by reason "
            "(query_error | query_timeout | empty_result | unknown_label | no_source).",
            labels=["group", "query", "variable", "reason"],
        )
        for (group, query, variable, reason), count in failures.items():
            counter.add_metric([group, query, variable, reason], float(count))
        yield counter


def _season_length_families(
    state: dict[tuple[str, str, str], int],
) -> Iterator[CounterMetricFamily]:
    """Render ``forecast_season_length_detections_total{group, query, source}``.

    Bumped once per series-detection. The ``source`` label is bounded to
    ``{detected, snapped, fallback, too_short}`` so cardinality is small
    even with hundreds of queries — far smaller than per-series logging
    at INFO. Emitted only when at least one detection has happened this
    process lifetime so /metrics stays quiet for deployments that don't
    use ``season_length: auto``.
    """
    if not state:
        return
    counter = CounterMetricFamily(
        "forecast_season_length_detections",
        "Outcomes of the auto-seasonality detector, by source "
        "(detected | snapped | fallback | too_short).",
        labels=["group", "query", "source"],
    )
    for (group, query, source), count in state.items():
        counter.add_metric([group, query, source], float(count))
    yield counter


def _calibration_families(
    groups: list[GroupSnapshot],
) -> Iterator[GaugeMetricFamily]:
    """Render ``forecast_band_calibration_offset`` gauges.

    Emitted only when at least one series has a non-empty offset list — a
    deployment with ``accuracy.conformal.enabled: false`` everywhere
    produces no samples and the family is omitted entirely so /metrics
    stays quiet.
    """
    points = [p for snap in groups for p in snap.calibration_offsets]
    yield from _emit_gauge_family(
        points,
        "forecast_band_calibration_offset",
        "Additive correction applied to the forecast band by split-conformal "
        "calibration. Positive values widen each side of the parametric band; "
        "0 means the band already empirically covered its nominal level.",
        value_fn=lambda p: p.offset,
        skip_fn=lambda p: not math.isfinite(p.offset),
    )


def _preprocess_families(
    groups: list[GroupSnapshot],
) -> Iterator[GaugeMetricFamily | CounterMetricFamily]:
    """Render the v1.1 preprocessing telemetry families.

    Emits five families:

    * ``forecast_input_gaps_filled_total{group, query, strategy}``
    * ``forecast_input_max_gap_seconds{group, query}``
    * ``forecast_input_outliers_replaced_total{group, query, method}``
    * ``forecast_input_change_point_total{group, query}``
    * ``forecast_input_effective_lookback_seconds{group, query}``

    Each is emitted only when at least one group has produced data for it
    so /metrics doesn't carry empty families when preprocessing is off
    everywhere.
    """
    yield from _gaps_filled_family(groups)
    yield from _max_gap_family(groups)
    yield from _outliers_family(groups)
    yield from _change_points_family(groups)
    yield from _effective_lookback_family(groups)


def _gaps_filled_family(groups: list[GroupSnapshot]) -> Iterator[CounterMetricFamily]:
    family = CounterMetricFamily(
        "forecast_input_gaps_filled",
        "Lookback samples imputed before fitting, by strategy "
        "(forward_fill | linear_interp | nan).",
        labels=["group", "query", "strategy"],
    )
    have = False
    for snap in groups:
        for (query_id, strategy), count in snap.gaps_filled_cumulative.items():
            family.add_metric([snap.group, query_id, strategy], float(count))
            have = True
    if have:
        yield family


def _max_gap_family(groups: list[GroupSnapshot]) -> Iterator[GaugeMetricFamily]:
    family = GaugeMetricFamily(
        "forecast_input_max_gap_seconds",
        "Largest gap observed in the lookback for the (group, query) tuple "
        "in the most recent run. 0 when no gaps were detected.",
        labels=["group", "query"],
    )
    have = False
    for snap in groups:
        for query_id, stats in snap.preprocess_stats.items():
            family.add_metric([snap.group, query_id], float(stats.max_gap_seconds))
            have = True
    if have:
        yield family


def _outliers_family(groups: list[GroupSnapshot]) -> Iterator[CounterMetricFamily]:
    family = CounterMetricFamily(
        "forecast_input_outliers_replaced",
        "Lookback samples replaced by the outlier scrubber, by detector method "
        "(hampel | iqr | stl_residual).",
        labels=["group", "query", "method"],
    )
    have = False
    for snap in groups:
        for (query_id, method), count in snap.outliers_replaced_cumulative.items():
            family.add_metric([snap.group, query_id, method], float(count))
            have = True
    if have:
        yield family


def _change_points_family(groups: list[GroupSnapshot]) -> Iterator[CounterMetricFamily]:
    family = CounterMetricFamily(
        "forecast_input_change_point",
        "Number of structural breaks detected in the lookback for the "
        "(group, query) tuple. Driven by preprocess.change_point.",
        labels=["group", "query"],
    )
    have = False
    for snap in groups:
        for query_id, count in snap.change_points_cumulative.items():
            family.add_metric([snap.group, query_id], float(count))
            have = True
    if have:
        yield family


def _effective_lookback_family(groups: list[GroupSnapshot]) -> Iterator[GaugeMetricFamily]:
    family = GaugeMetricFamily(
        "forecast_input_effective_lookback_seconds",
        "Effective post-truncation lookback window used for the most recent "
        "fit of the (group, query) tuple. Equal to the configured lookback "
        "when no truncation happened.",
        labels=["group", "query"],
    )
    have = False
    for snap in groups:
        for query_id, stats in snap.preprocess_stats.items():
            if stats.effective_lookback_seconds > 0:
                family.add_metric([snap.group, query_id], float(stats.effective_lookback_seconds))
                have = True
    if have:
        yield family
