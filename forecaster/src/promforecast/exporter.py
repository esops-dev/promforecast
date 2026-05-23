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
    "BandCoveragePoint",
    "CalibrationOffsetPoint",
    "ComponentDecompositionPoint",
    "ConditionalDeviationPoint",
    "ConditionalForecastPoint",
    "ContributionPoint",
    "DeviationPoint",
    "DriftPoint",
    "EnsembleWeightPoint",
    "ErrorBudgetPoint",
    "Exporter",
    "ForecastPoint",
    "GroupSnapshot",
    "GrowthRatePoint",
    "InvalidMetricNameError",
    "PreprocessStats",
    "QualityPoint",
    "RecommendedMaintenanceWindowPoint",
    "ThresholdEtaPoint",
    "TimeToDrainPoint",
    "WillBreachPoint",
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
class ThresholdEtaPoint:
    """One ``<id>_forecast_time_to_threshold_seconds`` sample.

    The metric name is per-id (mirrors the forecast point family) so it
    can be discovered/aggregated alongside the underlying forecast.
    ``seconds`` carries the ETA value; NaN is preserved through the
    exporter and dropped at render time so dashboards/alerts don't see
    sentinel rows.
    """

    metric: str
    labels: dict[str, str]
    seconds: float


@dataclass(frozen=True)
class ConditionalForecastPoint:
    """A single conditional forecast row destined for /metrics.

    Mirrors :class:`ForecastPoint` but its metric name carries the
    ``_forecast_conditional`` / ``_forecast_conditional_lower`` /
    ``_forecast_conditional_upper`` suffix family. Emitted by the
    conditional re-fit path when ``emission.conditional`` is on for the
    query.
    """

    metric: str
    labels: dict[str, str]
    value: float


@dataclass(frozen=True)
class ConditionalDeviationPoint:
    """A single conditional deviation observation.

    Rendered as ``forecast_conditional_deviation_ratio`` and
    ``forecast_conditional_deviation_outside_band`` gauges. Mirrors the
    existing :class:`DeviationPoint` shape — the band reference is the
    conditional fit's leading row rather than the backtest trailing
    band.
    """

    labels: dict[str, str]
    ratio: float
    outside_band: bool


@dataclass(frozen=True)
class ComponentDecompositionPoint:
    """A single ``<id>_forecast_component`` sample.

    The metric name is per-id (mirrors the forecast family). ``labels``
    carry ``component`` (one of ``trend``/``seasonal``/``level``/
    ``residual``) plus the standard ``model`` / ``horizon`` discriminators
    and the source series labels. One sample is emitted per horizon step
    per component when ``emission.decompose`` is on for a decomposable
    model.
    """

    metric: str
    labels: dict[str, str]
    value: float


@dataclass(frozen=True)
class TimeToDrainPoint:
    """One ``<id>_forecast_time_to_drain_seconds`` sample.

    Sibling of :class:`ThresholdEtaPoint` for the ``le_drain`` comparator.
    The metric name is per-id (mirrors the ETA family) and the
    no-crossing sentinel is ``+Inf`` rather than NaN — operators alert
    on ``forecast_time_to_drain_seconds == +Inf`` to catch "queue is
    not actually draining".
    """

    metric: str
    labels: dict[str, str]
    seconds: float


@dataclass(frozen=True)
class ErrorBudgetPoint:
    """One row in the SLO error-budget family destined for /metrics.

    Rendered through three per-id metric names:

    * ``<id>_forecast_error_budget_remaining_seconds{window}``
    * ``<id>_forecast_error_budget_burn_rate{window}``
    * ``<id>_forecast_error_budget_exhausted_in_seconds{window[, level]}``

    The metric field carries the full name including the ``<id>_`` prefix
    so the exporter can bucket and render one ``GaugeMetricFamily`` per
    name (matching the forecast / ETA / will-breach family shape).
    """

    metric: str
    labels: dict[str, str]
    value: float


@dataclass(frozen=True)
class RecommendedMaintenanceWindowPoint:
    """One ``forecast_recommended_maintenance_window`` sample.

    Global metric name (no per-id prefix) — the ``id`` and ``group`` are
    discriminator labels so a single Grafana panel can render the
    ranked windows across every query that opts in. The sample value is
    the predicted load at the window (mean ``yhat`` across the window's
    forecast rows); labels carry ``rank``, ``duration_seconds`` and
    ``start_timestamp_seconds`` so dashboards can plot the window's
    position on the timeline.
    """

    labels: dict[str, str]
    predicted_load: float


@dataclass(frozen=True)
class WillBreachPoint:
    """One ``<id>_forecast_will_breach`` 0/1 sample.

    The metric name is per-id (mirrors the ETA family). ``labels`` carry
    ``name``, ``severity``, and ``model``; ``level`` is added for
    per-band rows and absent for the central forecast.
    """

    metric: str
    labels: dict[str, str]
    value: float


@dataclass(frozen=True)
class GrowthRatePoint:
    """One ``<id>_forecast_growth_rate`` sample.

    The metric name is per-id (mirrors the forecast point family).
    ``value`` is the slope in the configured units; ``labels`` carry
    ``window`` (e.g. ``"1d"``) and ``units``
    (``units_per_second``/``percent``).
    """

    metric: str
    labels: dict[str, str]
    value: float


@dataclass(frozen=True)
class DriftPoint:
    """One ``forecast_drift_score`` sample.

    Rendered as a gauge labelled with ``id``, ``group``, ``model``,
    plus the rest of the series base labels so the value can be joined
    against the underlying forecast in Grafana. ``method`` carries the
    scoring method (``mape``/``wasserstein``) so dashboards can switch
    between the two without re-keying.
    """

    labels: dict[str, str]
    score: float


@dataclass(frozen=True)
class BandCoveragePoint:
    """One ``forecast_band_empirical_coverage`` sample.

    Rendered as a gauge in ``[0, 1]``. ``samples`` carries the number of
    rolling observations the fraction was computed over so dashboards
    can dim panels whose window hasn't filled yet (``< maxlen``).
    """

    labels: dict[str, str]
    fraction: float
    samples: int


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
    # Capacity-planning families: per-(series, model, threshold[, level])
    # ETAs and per-(series, model, window) growth-rate gauges. Both are
    # snapshot-style — only the most recent run's samples belong on
    # /metrics; queries that disappear from the config disappear from the
    # exposition until they reappear.
    threshold_etas: list[ThresholdEtaPoint] = field(default_factory=list)
    growth_rates: list[GrowthRatePoint] = field(default_factory=list)
    # Predictive will-breach gauge: per-(series, model, threshold[, level])
    # 0/1 sample derived from the ETA + threshold ``alert_within`` window.
    # Snapshot-style — only the most recent run's samples belong on
    # /metrics; thresholds that drop out of config disappear from the
    # exposition until they reappear.
    will_breach_points: list[WillBreachPoint] = field(default_factory=list)
    # Queue burn-down / drain ETA: parallel ``<id>_forecast_time_to_drain_seconds``
    # family for thresholds with comparator=``le_drain``. ``+Inf`` samples
    # are kept so the "queue is not draining" alert can match them.
    time_to_drain_points: list[TimeToDrainPoint] = field(default_factory=list)
    # SLO error-budget snapshot rows (remaining, burn rate, exhausted_in).
    # Snapshot-style — only the most recent run's samples belong on
    # /metrics; queries that disappear from config disappear from the
    # exposition until they reappear.
    error_budget_points: list[ErrorBudgetPoint] = field(default_factory=list)
    # Recommended maintenance windows (top-N lowest-load windows in the
    # forecast horizon). Snapshot-style; the next refresh rotates the
    # set as time moves forward and earlier windows leave the lookahead.
    recommended_maintenance_window_points: list[RecommendedMaintenanceWindowPoint] = field(
        default_factory=list
    )
    # Parallel conditional forecast family. Same shape as ``points`` —
    # rendered through one ``GaugeMetricFamily`` per distinct metric name —
    # but the metric names carry the ``_forecast_conditional`` suffix
    # family.
    conditional_points: list[ConditionalForecastPoint] = field(default_factory=list)
    # Conditional deviation gauges (one per emitted (series, model, level)).
    conditional_deviation_points: list[ConditionalDeviationPoint] = field(default_factory=list)
    # Per-horizon-step component decomposition (per emitted series, model).
    # Rendered as ``<id>_forecast_component{component="trend|...|residual"}``.
    component_points: list[ComponentDecompositionPoint] = field(default_factory=list)
    last_run_timestamp: float = 0.0
    run_duration_seconds: float = 0.0
    # Per-query run duration in seconds for the most recent fit of each
    # query. Drives ``forecast_run_duration_seconds{query=...}`` so an
    # operator can see which queries dominate fit cost when per-query
    # refresh intervals diverge.
    run_duration_per_query: dict[str, float] = field(default_factory=dict)
    series_count: int = 0
    # Per-query series count: same shape as ``run_duration_per_query``.
    # Surfaces ``forecast_series_count{query=...}`` so queries running on
    # different refresh cadences each report their own cardinality.
    series_count_per_query: dict[str, int] = field(default_factory=dict)
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
    # Cumulative per-(query, kind) emission clipping counter.
    # ``kind`` is bounded to ``negative`` (a value clipped from below zero
    # by ``emission.clip_non_negative``) and ``rounded`` (a value whose
    # fractional part was non-trivial when ``emission.round_to_integer``
    # was on). Merged with the previous snapshot so the counter is
    # monotonic across runs.
    emission_clipped_cumulative: dict[tuple[str, str], int] = field(default_factory=dict)
    # Per-(query) cold-start sibling counts from the most recent run.
    # Gauge — only the latest run's samples belong on /metrics. Surfaces
    # as ``forecast_cold_start_siblings`` so operators can see when a
    # cold-start fit is borrowing from a thinner sibling set than the
    # operator configured (e.g. peer renames thinning the borrow query).
    cold_start_siblings: dict[str, int] = field(default_factory=dict)
    # Drift telemetry — per-(series, model) gauge of how much the
    # forecast curve moved between successive fits, plus a cumulative
    # per-(query) alert counter that bumps when the gauge crosses the
    # configured threshold.
    drift_points: list[DriftPoint] = field(default_factory=list)
    drift_alerts_cumulative: dict[str, int] = field(default_factory=dict)
    # Empirical band coverage — per-(series, model, level, window)
    # rolling gauge in [0, 1]. The exporter holds the rolling state in a
    # separate ``BandCoverageTracker`` and snapshots the current
    # fractions here at render time.
    band_coverage_points: list[BandCoveragePoint] = field(default_factory=list)


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
        # Per-(group, query, model, reason) decomposition-skipped counter.
        # Bumped when ``emission.decompose: true`` is set on a query but
        # the chosen model does not decompose cleanly. The ``reason`` set
        # is bounded by the runner (``unsupported_model``, ``fit_failed``,
        # ``components_unavailable``, ``invariant_violation``); the
        # ``group``/``query`` labels let operators alert "which query
        # keeps tripping the skip path" without log scraping. The
        # cardinality multiplier stays small (typical decompose-opted-in
        # query has 1-2 emitted models, and skips only fire when
        # something is wrong).
        self._decomposition_skipped: dict[tuple[str, str, str, str], int] = {}
        # Per-(group, query, model) conditional-aliased counter. Bumped
        # whenever the conditional emission path short-circuits the
        # refit (because the model advertises ``uses_regressors=False``
        # or no regressors are configured). Lets operators confirm the
        # CPU saving from the short-circuit is actually firing in
        # their deployment; a flat-zero counter means every conditional
        # emission is paying the refit cost.
        self._conditional_aliased: dict[tuple[str, str, str], int] = {}
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

    def increment_conditional_aliased(self, group: str, query: str, model: str) -> None:
        """Bump the per-(group, query, model) conditional-aliased counter.

        Called by the runner whenever the conditional emission path
        aliases the unconditional forecast instead of refitting. The
        operator can chart ``rate(forecast_conditional_aliased_total[5m])``
        to confirm the short-circuit is firing.
        """
        with self._lock:
            key = (group, query, model)
            self._conditional_aliased[key] = self._conditional_aliased.get(key, 0) + 1

    def increment_decomposition_skipped(
        self,
        group: str,
        query: str,
        model: str,
        reason: str,
    ) -> None:
        """Bump the per-(group, query, model, reason) decomposition-skipped counter.

        Bounded ``reason`` set lives on the runner (``unsupported_model``,
        ``fit_failed``, ``components_unavailable``, ``invariant_violation``)
        so cardinality stays small even with the four-label fan-out.
        Operators can alert per (group, query) on persistent skips
        instead of relying on a global "something is being skipped"
        signal.
        """
        with self._lock:
            key = (group, query, model, reason)
            self._decomposition_skipped[key] = self._decomposition_skipped.get(key, 0) + 1

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

    def _decomposition_skipped_state(self) -> dict[tuple[str, str, str, str], int]:
        with self._lock:
            return dict(self._decomposition_skipped)

    def _conditional_aliased_state(self) -> dict[tuple[str, str, str], int]:
        with self._lock:
            return dict(self._conditional_aliased)

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
        decomposition_skipped_state = self._exporter._decomposition_skipped_state()
        conditional_aliased_state = self._exporter._conditional_aliased_state()
        yield from _forecast_families(groups)
        yield from _conditional_forecast_families(groups)
        yield from _conditional_deviation_families(groups)
        yield from _component_families(groups)
        yield from _accuracy_families(groups)
        yield from _deviation_families(groups)
        yield from _quality_families(groups)
        yield from _ensemble_families(groups)
        yield from _contribution_families(groups)
        yield from _threshold_eta_families(groups)
        yield from _will_breach_families(groups)
        yield from _time_to_drain_families(groups)
        yield from _error_budget_families(groups)
        yield from _recommended_window_families(groups)
        yield from _growth_rate_families(groups)
        yield from _calibration_families(groups)
        yield from _preprocess_families(groups)
        yield from _emission_clipped_families(groups)
        yield from _cold_start_siblings_families(groups)
        yield from _drift_families(groups)
        yield from _band_coverage_families(groups)
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
        yield from _decomposition_skipped_families(decomposition_skipped_state)
        yield from _conditional_aliased_families(conditional_aliased_state)


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


def _threshold_eta_families(
    groups: list[GroupSnapshot],
) -> Iterator[GaugeMetricFamily]:
    """Render ``<id>_forecast_time_to_threshold_seconds`` gauges, one family per id.

    The metric name is per-id (mirrors the forecast family) so the ETA can
    be discovered alongside the underlying forecast in Grafana. NaN ETAs
    (no crossing within the horizon) are dropped at render time — alerts
    on ``time_to_threshold_seconds < 3600`` are the typical use case and
    NaN samples would silently break them.
    """
    by_metric: dict[str, list[ThresholdEtaPoint]] = {}
    for snap in groups:
        for point in snap.threshold_etas:
            by_metric.setdefault(point.metric, []).append(point)
    for metric_name, points in sorted(by_metric.items()):
        yield from _emit_gauge_family(
            points,
            metric_name,
            "Seconds from the forecast's start until the first crossing of the "
            "configured threshold; NaN crossings are not emitted.",
            value_fn=lambda p: p.seconds,
            skip_fn=lambda p: not math.isfinite(p.seconds),
        )


def _conditional_forecast_families(
    groups: list[GroupSnapshot],
) -> Iterator[GaugeMetricFamily]:
    """Render the ``<id>_forecast_conditional*`` family, one per distinct name.

    Same bucketing as the unconditional forecast family. The metric
    names that show up here are ``<id>_forecast_conditional``,
    ``<id>_forecast_conditional_lower``, ``<id>_forecast_conditional_upper``.
    """
    by_metric: dict[str, list[ConditionalForecastPoint]] = {}
    for snap in groups:
        for point in snap.conditional_points:
            by_metric.setdefault(point.metric, []).append(point)
    for metric_name, points in sorted(by_metric.items()):
        yield from _emit_gauge_family(
            points,
            metric_name,
            "promforecast conditional forecast value — model re-fit with "
            "regressors held at their most-recent value.",
            value_fn=lambda p: p.value,
        )


def _conditional_deviation_families(
    groups: list[GroupSnapshot],
) -> Iterator[GaugeMetricFamily]:
    """Render the conditional-deviation ratio + outside-band gauges.

    Mirrors the unconditional ``_deviation_families`` shape; the
    conditional ratio drops NaN samples while the boolean outside-band
    still emits a 0 / 1 reading so alerts can detect "outside the
    conditional band" cleanly.
    """
    points = [p for snap in groups for p in snap.conditional_deviation_points]
    yield from _emit_gauge_family(
        points,
        "forecast_conditional_deviation_ratio",
        "Signed deviation of the most recent actual from the conditional "
        "forecast band (regressors held at their current value).",
        value_fn=lambda p: p.ratio,
        skip_fn=lambda p: not math.isfinite(p.ratio),
    )
    yield from _emit_gauge_family(
        points,
        "forecast_conditional_deviation_outside_band",
        "1 if the most recent actual is outside the conditional forecast band, else 0.",
        value_fn=lambda p: 1.0 if p.outside_band else 0.0,
    )


def _component_families(
    groups: list[GroupSnapshot],
) -> Iterator[GaugeMetricFamily]:
    """Render the ``<id>_forecast_component`` decomposition family per id."""
    by_metric: dict[str, list[ComponentDecompositionPoint]] = {}
    for snap in groups:
        for point in snap.component_points:
            by_metric.setdefault(point.metric, []).append(point)
    for metric_name, points in sorted(by_metric.items()):
        yield from _emit_gauge_family(
            points,
            metric_name,
            "Per-component decomposition of the forecast curve "
            "(trend / seasonal / level / residual). Components sum to "
            "the central forecast within numerical tolerance.",
            value_fn=lambda p: p.value,
            skip_fn=lambda p: not math.isfinite(p.value),
        )


def _conditional_aliased_families(
    state: dict[tuple[str, str, str], int],
) -> Iterator[CounterMetricFamily]:
    """Render ``forecast_conditional_aliased_total{group, query, model}``.

    Bumped every time the conditional emission path skips the refit
    and aliases the unconditional forecast frame instead. Lets
    operators chart how often the short-circuit is firing
    (and therefore how much CPU it is saving them); a deployment
    where the counter stays at 0 is paying the doubled-fit cost for
    every conditional emission.

    Emitted only after at least one alias has fired so /metrics stays
    quiet for deployments that don't enable ``emission.conditional``.
    """
    if not state:
        return
    counter = CounterMetricFamily(
        "forecast_conditional_aliased",
        "Number of times the conditional emission path aliased the unconditional "
        "forecast frame instead of refitting with held-constant regressors. "
        "Bumped per (series, model) when no regressors are configured or the model "
        "advertises uses_regressors=False.",
        labels=["group", "query", "model"],
    )
    for (group, query, model), count in state.items():
        counter.add_metric([group, query, model], float(count))
    yield counter


def _decomposition_skipped_families(
    state: dict[tuple[str, str, str, str], int],
) -> Iterator[CounterMetricFamily]:
    """Render ``forecast_decomposition_skipped_total{group, query, model, reason}``.

    Emitted only when at least one skip has occurred so /metrics stays
    quiet for deployments that don't enable decomposition (or only
    enable it for decomposable models).
    """
    if not state:
        return
    counter = CounterMetricFamily(
        "forecast_decomposition_skipped",
        "Number of times the runner skipped component-decomposition emission "
        "for the (group, query, model, reason) tuple. Reason is one of "
        "(unsupported_model | fit_failed | components_unavailable | "
        "invariant_violation).",
        labels=["group", "query", "model", "reason"],
    )
    for (group, query, model, reason), count in state.items():
        counter.add_metric([group, query, model, reason], float(count))
    yield counter


def _will_breach_families(
    groups: list[GroupSnapshot],
) -> Iterator[GaugeMetricFamily]:
    """Render ``<id>_forecast_will_breach`` gauges, one family per id.

    The metric name is per-id (mirrors the ETA family). Values are
    always 0 or 1 — operators alert with
    ``forecast_will_breach{severity="critical"} == 1 for 10m`` and the
    zero-baseline samples keep the panel/chart well-defined even before
    the first crossing.
    """
    by_metric: dict[str, list[WillBreachPoint]] = {}
    for snap in groups:
        for point in snap.will_breach_points:
            by_metric.setdefault(point.metric, []).append(point)
    for metric_name, points in sorted(by_metric.items()):
        yield from _emit_gauge_family(
            points,
            metric_name,
            "1 if the forecast predicts a threshold crossing within the "
            "configured `alert_within` window for the (name, severity[, level]) "
            "label set; 0 otherwise.",
            value_fn=lambda p: p.value,
        )


def _time_to_drain_families(
    groups: list[GroupSnapshot],
) -> Iterator[GaugeMetricFamily]:
    """Render ``<id>_forecast_time_to_drain_seconds`` gauges, one family per id.

    Sibling of :func:`_threshold_eta_families` for the ``le_drain``
    comparator. Unlike the ETA family this *preserves* the
    no-crossing sentinel (``+Inf``) because the operator-facing alert
    is ``forecast_time_to_drain_seconds == +Inf for 30m`` — dropping
    non-finite samples at render time would make that rule silently
    never match. NaN values (degenerate inputs) are still dropped
    because the alert can't act on them.
    """
    by_metric: dict[str, list[TimeToDrainPoint]] = {}
    for snap in groups:
        for point in snap.time_to_drain_points:
            by_metric.setdefault(point.metric, []).append(point)
    for metric_name, points in sorted(by_metric.items()):
        yield from _emit_gauge_family(
            points,
            metric_name,
            "Seconds from the forecast's start until the projected curve first "
            "crosses ``value`` from above (queue drains). ``+Inf`` means no "
            "crossing within the horizon — alert on ``... == +Inf for 30m`` to "
            "catch 'queue is not draining'.",
            value_fn=lambda p: p.seconds,
            skip_fn=lambda p: math.isnan(p.seconds),
        )


def _error_budget_families(
    groups: list[GroupSnapshot],
) -> Iterator[GaugeMetricFamily]:
    """Render the SLO error-budget family per-id.

    Three sibling metric names are bucketed by name then emitted as
    individual gauge families. The exhausted-in family carries an
    explicit ``+Inf`` sample when the forecast says no errors — that
    propagates the "no concern" answer through to alerts on
    ``... > N`` instead of dropping the sample. NaN is still skipped
    because alerts can't act on it.
    """
    by_metric: dict[str, list[ErrorBudgetPoint]] = {}
    for snap in groups:
        for point in snap.error_budget_points:
            by_metric.setdefault(point.metric, []).append(point)
    for metric_name, points in sorted(by_metric.items()):
        yield from _emit_gauge_family(
            points,
            metric_name,
            "SLO error-budget projection derived from the forecast. The "
            "remaining-seconds family is the total allowed budget for the "
            "window; the burn-rate family is the predicted rate divided by "
            "the allowed rate; the exhausted-in family projects when the "
            "budget would be consumed at the current trajectory.",
            value_fn=lambda p: p.value,
            skip_fn=lambda p: math.isnan(p.value),
        )


def _recommended_window_families(
    groups: list[GroupSnapshot],
) -> Iterator[GaugeMetricFamily]:
    """Render ``forecast_recommended_maintenance_window`` gauges.

    Single global family (no per-id prefix) — ``id`` is a label so a
    Grafana panel can render the ranked recommendations across every
    opted-in query. The sample value is the predicted load at the
    window; lower is better. Labels carry ``rank``,
    ``duration_seconds``, ``start_timestamp_seconds``.
    """
    points = [p for snap in groups for p in snap.recommended_maintenance_window_points]
    yield from _emit_gauge_family(
        points,
        "forecast_recommended_maintenance_window",
        "Predicted load (mean forecast value) for a recommended maintenance "
        "window. Lower is better; ``rank=1`` is the suggested window. The "
        "``start_timestamp_seconds`` and ``duration_seconds`` labels carry "
        "the window's position on the forecast horizon.",
        value_fn=lambda p: p.predicted_load,
        skip_fn=lambda p: not math.isfinite(p.predicted_load),
    )


def _growth_rate_families(
    groups: list[GroupSnapshot],
) -> Iterator[GaugeMetricFamily]:
    """Render ``<id>_forecast_growth_rate`` gauges, one family per id.

    The metric name is per-id (mirrors the forecast family); the ``units``
    label distinguishes ``units_per_second`` (the default) from
    ``percent`` (fractional relative change). NaN growth rates (windows
    that overflow the horizon, degenerate zero baselines for percent
    units) are dropped at render time.
    """
    by_metric: dict[str, list[GrowthRatePoint]] = {}
    for snap in groups:
        for point in snap.growth_rates:
            by_metric.setdefault(point.metric, []).append(point)
    for metric_name, points in sorted(by_metric.items()):
        yield from _emit_gauge_family(
            points,
            metric_name,
            "Forecast growth rate measured as the slope of the forecast curve "
            "over the window; units (units_per_second | percent) live on the "
            "label of the same name.",
            value_fn=lambda p: p.value,
            skip_fn=lambda p: not math.isfinite(p.value),
        )


def _operational_families(  # noqa: PLR0912
    groups: list[GroupSnapshot],
) -> Iterator[GaugeMetricFamily | CounterMetricFamily]:
    if not groups:
        return

    last_run = GaugeMetricFamily(
        "forecast_last_run_timestamp_seconds",
        "Unix timestamp of the most recent successful run for the group.",
        labels=["group"],
    )
    # Group-aggregate gauges. Stable schema (single ``group`` label) —
    # dashboards relying on these built before per-query refresh intervals
    # existed keep working unchanged. Per-query breakdowns are emitted as
    # separate families (``forecast_query_run_duration_seconds`` and
    # ``forecast_query_series_count``) so dashboards can opt in without
    # being forced to change their label matchers.
    duration = GaugeMetricFamily(
        "forecast_run_duration_seconds",
        "Wall-clock duration of the most recent run for the group "
        "(sum across all queries when per-query refresh intervals diverge).",
        labels=["group"],
    )
    series_count = GaugeMetricFamily(
        "forecast_series_count",
        "Number of series accepted by the most recent run for the group.",
        labels=["group"],
    )
    per_query_duration = GaugeMetricFamily(
        "forecast_query_run_duration_seconds",
        "Wall-clock duration of the most recent run for the "
        "(group, query) tuple. Emitted when per-query refresh intervals "
        "cause query-level fit costs to diverge from the group total.",
        labels=["group", "query"],
    )
    per_query_series_count = GaugeMetricFamily(
        "forecast_query_series_count",
        "Number of series accepted by the most recent run for the "
        "(group, query) tuple. Emitted alongside "
        "``forecast_series_count`` to enable per-query cardinality views.",
        labels=["group", "query"],
    )
    have_per_query_duration = False
    have_per_query_series_count = False
    for snap in groups:
        last_run.add_metric([snap.group], snap.last_run_timestamp)
        duration.add_metric([snap.group], snap.run_duration_seconds)
        series_count.add_metric([snap.group], float(snap.series_count))
        for query_id, dur in snap.run_duration_per_query.items():
            per_query_duration.add_metric([snap.group, query_id], float(dur))
            have_per_query_duration = True
        for query_id, count in snap.series_count_per_query.items():
            per_query_series_count.add_metric([snap.group, query_id], float(count))
            have_per_query_series_count = True
    yield last_run
    yield duration
    yield series_count
    if have_per_query_duration:
        yield per_query_duration
    if have_per_query_series_count:
        yield per_query_series_count

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
        "invalid_metric_name | group_error | conditional_fit_error | "
        "conditional_fit_timeout).",
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
    """Render the preprocessing telemetry families.

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


def _emission_clipped_families(
    groups: list[GroupSnapshot],
) -> Iterator[CounterMetricFamily]:
    """Render ``forecast_emission_clipped_total{group, query, kind}``.

    Bumped only when an emission safeguard (``clip_non_negative`` or
    ``round_to_integer``) actually corrected a value:

    * ``kind="negative"`` increments per emission whose model output
      was clipped from below zero — a real model-output anomaly worth
      surfacing.
    * ``kind="rounded"`` increments only when the rounding distance
      was non-trivial (``|fractional part| >= 0.05``) so the counter
      does not explode on every emission of an inherently non-integer
      model. This gives operators a useful signal for "rounding is
      masking systematic fractional output" without flooding the
      counter for benign integer-ish forecasts.

    Emitted only when at least one group has produced data so /metrics
    stays quiet for deployments that don't enable count-aware safeguards.
    """
    family = CounterMetricFamily(
        "forecast_emission_clipped",
        "Lookback-free emission corrections applied by count-aware safeguards, "
        "by kind (negative = clipped from below zero | rounded = non-trivial "
        "fractional component rounded to integer).",
        labels=["group", "query", "kind"],
    )
    have = False
    for snap in groups:
        for (query_id, kind), count in snap.emission_clipped_cumulative.items():
            family.add_metric([snap.group, query_id, kind], float(count))
            have = True
    if have:
        yield family


def _drift_families(
    groups: list[GroupSnapshot],
) -> Iterator[GaugeMetricFamily | CounterMetricFamily]:
    """Render the forecast-drift gauge and alert counter.

    ``forecast_drift_score{id, group, model, ...}`` reports how much
    the forecast curve moved between successive fits (default: MAPE
    between the two yhat curves, configurable per query). Non-finite
    samples are dropped at render time — they signal "no prior fit
    available" and emitting NaN would just confuse downstream alerts.

    ``forecast_drift_alerts_total{group, id}`` ticks once every time
    the drift score crossed the configured threshold; counter so
    ``increase(...)`` alerts survive transient flips back below the
    threshold.
    """
    drift_points = [p for snap in groups for p in snap.drift_points]
    yield from _emit_gauge_family(
        drift_points,
        "forecast_drift_score",
        "Per-fit drift score: how much the forecast curve moved between "
        "successive fits (label `method` carries the scoring algorithm).",
        value_fn=lambda p: p.score,
        skip_fn=lambda p: not math.isfinite(p.score),
    )
    have_alerts = any(snap.drift_alerts_cumulative for snap in groups)
    if not have_alerts:
        return
    counter = CounterMetricFamily(
        "forecast_drift_alerts",
        "Number of times the forecast drift score crossed the configured "
        "alert threshold for the (group, query) tuple.",
        labels=["group", "id"],
    )
    for snap in groups:
        for query_id, count in snap.drift_alerts_cumulative.items():
            counter.add_metric([snap.group, query_id], float(count))
    yield counter


def _band_coverage_families(
    groups: list[GroupSnapshot],
) -> Iterator[GaugeMetricFamily]:
    """Render ``forecast_band_empirical_coverage`` gauges.

    One gauge per (series, model, level, window). The value is the
    fraction of recent actuals that landed inside the nominal band — a
    reading of 0.4 on a 95% band tells the operator the band is
    drastically under-covering. Emitted only when at least one series
    has produced a rolling sample so /metrics stays quiet for
    deployments that don't opt into the diagnostic.
    """
    points = [p for snap in groups for p in snap.band_coverage_points]
    yield from _emit_gauge_family(
        points,
        "forecast_band_empirical_coverage",
        "Rolling fraction of actuals that fell inside the nominal forecast "
        "band over the configured window. 0.0..1.0; expect ~level/100 when "
        "the bands are well calibrated.",
        value_fn=lambda p: p.fraction,
        skip_fn=lambda p: not math.isfinite(p.fraction),
    )


def _cold_start_siblings_families(
    groups: list[GroupSnapshot],
) -> Iterator[GaugeMetricFamily]:
    """Render ``forecast_cold_start_siblings{group, query}``.

    Gauge — the number of usable sibling series the cold-start borrow
    query produced for the most recent run of the (group, query). A
    sudden drop (e.g. peer renames thinning the borrow query) is the
    early-warning signal that the configured ``min_siblings`` threshold
    may stop being met. Emitted only when at least one cold-start fit
    has run so /metrics stays empty for deployments that don't use the
    feature.
    """
    family = GaugeMetricFamily(
        "forecast_cold_start_siblings",
        "Number of usable sibling series the cold-start borrow query produced "
        "for the most recent run of the (group, query) tuple. Compared against "
        "the configured cold_start.borrow_from.min_siblings to see how much "
        "headroom the borrow has.",
        labels=["group", "query"],
    )
    have = False
    for snap in groups:
        for query_id, count in snap.cold_start_siblings.items():
            family.add_metric([snap.group, query_id], float(count))
            have = True
    if have:
        yield family
