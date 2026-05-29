"""Per-group execution and scheduling.

Each group runs as a long-lived asyncio task that fits, exports, and sleeps
for ``server.refresh_interval``. A per-group lock prevents a slow run from
being triggered twice concurrently. Failure isolation is layered: per query,
per series, per regressor.

This module owns the :class:`Runner` orchestration class. The dataclasses,
pure helpers, per-family emission post-processing, and budget allocation
all live in sibling modules under ``runner/``; this file is intentionally
narrow to the scheduler + per-fit pipeline.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import time
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import structlog

from .. import annotations as annotations_mod
from .. import band_coverage as band_coverage_mod
from .. import canary as canary_mod
from .. import capacity as capacity_mod
from .. import cold_start as cold_start_mod
from .. import conformal as conformal_mod
from .. import contribution as contribution_mod
from .. import decomposition as decomposition_mod
from .. import deploy_risk as deploy_risk_mod
from .. import deviation as deviation_mod
from .. import discovery as discovery_mod
from .. import drift as drift_mod
from .. import drift_explainer as drift_explainer_mod
from .. import health as health_mod
from .. import incident_pattern as incident_pattern_mod
from .. import maintenance as maintenance_mod
from .. import models as model_registry
from .. import preprocess as preprocess_mod
from .. import quality as quality_mod
from .. import regressors as regressors_mod
from .. import revision as revision_mod
from .. import right_sizing as right_sizing_mod
from .. import seasonality as seasonality_mod
from .. import slo as slo_mod
from .. import telemetry as telemetry_mod
from .. import what_if as what_if_mod
from ..accuracy import (
    AccuracyResult,
    HorizonSpec,
    evaluate,
    horizons_from_durations,
    overall_mape,
    overall_mase,
)
from ..config import (
    DeploySignalsConfig,
    GroupConfig,
    ModelSpec,
    QueryConfig,
    ResourceSizingWorkloadConfig,
    effective_emission_safeguards,
    effective_refresh_interval,
    effective_season_length,
    effective_step,
)
from ..exporter import (
    BandCoveragePoint,
    CanaryDeviationPoint,
    DeployRiskPoint,
    DriftPoint,
    Exporter,
    ForecastPoint,
    GroupSnapshot,
    IncidentRecurrencePoint,
    InvalidMetricNameError,
    ModelDisagreementPoint,
    PreprocessStats,
    QualityPoint,
    ResourceSizingPoint,
    RevisionPoint,
    ServiceHealthScorePoint,
    ThresholdActionCostPoint,
    validate_metric_name,
)
from ..point_factory import BacktestOutcome, SeriesPointFactory
from ..selector import ensemble_weights, select
from ..sink import ForecastCurvePoint, ForecastSink, SinkError
from ..source import PromSource, SeriesFrame
from ._budgets import _apply_canary, _apply_overflow, _compute_global_budgets
from ._helpers import (
    _coefficient_of_variation,
    _combine_ensemble,
    _estimate_group_memory_bytes,
    _extract_per_horizon_yhat,
    _format_band_window,
    _pandas_freq,
    _query_id_from_metric,
    _resolve_spec,
    _run_backtest,
    _run_fit,
    _series_key_from_labels,
)
from ._state import (
    ReloadResult,
    _BacktestNeeds,
    _CapacityInput,
    _FitResult,
    _inherit_prior_for_skipped_queries,
    _merge_fit_into_ctx,
    _RunContext,
)
from .emission.calibration import (
    _apply_offsets_to_curve_points,
    _apply_offsets_to_snapshot_points,
    _apply_offsets_to_threshold_eta_inputs,
)
from .emission.conditional import _flatten_regressors_to_last_value
from .emission.safeguards import (
    _apply_safeguards_to_curve_points,
    _apply_safeguards_to_forecast_frame,
    _apply_safeguards_to_snapshot_points,
)

if TYPE_CHECKING:
    import pandas as pd

    from ..annotations import GrafanaAnnotationSink
    from ..config import Config
    from ..selector import SelectionResult
    from ..triggers import TriggerEngine
    from ..webhook import WebhookRegressorClient

logger = structlog.get_logger(__name__)

# Reference to the parent package; used inside the Runner class to read
# ``SLOW_FIT_RATIO`` and ``_summarise_backtest`` dynamically so the
# pre-split test pattern
# ``monkeypatch.setattr(promforecast.runner, "SLOW_FIT_RATIO", ...)``
# (and the same for ``_summarise_backtest``) continues to work after the
# split — the patched value is picked up on the next attribute lookup.
# The import is intentionally written as ``import promforecast.runner``
# (not ``from . import ...``) so the binding is the package object
# itself, whose attributes are dynamically looked up at call time. The
# package is partial at import time but fully populated by the time any
# Runner method runs.
import promforecast.runner as _pkg  # noqa: E402

__all__ = [
    "Runner",
]


# Number of rows the incident-recurrence heatmap emits per opted-in
# (data_profile: events) query. Sourced from
# :mod:`promforecast.incident_pattern`'s fixed 7-days x 24-hours grid.
# Held as a module-level constant rather than recomputed on the hot
# path because event-query emission charges it against the global
# series budget every refresh.
_INCIDENT_RECURRENCE_ROWS = 7 * 24

# Minimum number of distinct emitted models required to compute the
# per-(series, horizon) model-disagreement coefficient of variation.
# Below two there is no spread to measure; the gauge is suppressed.
_MIN_MODELS_FOR_DISAGREEMENT = 2


def _build_deploy_tracker(config: Config) -> deploy_risk_mod.DeployTracker:
    """Construct a :class:`DeployTracker` with per-service caps from config.

    Each ``deploy_signals[].max_active_deploys`` becomes the per-service
    cap; services not in the config fall back to a default of 16 (the
    pre-config-aware behaviour, kept so an event recorded for a
    just-removed service still gets a sensible bound while the reload
    settles). Returning a fresh tracker on every call lets the reload
    path drop in-flight active deploys cleanly — see
    ``apply_config`` for the trade-off.
    """
    if not config.deploy_signals:
        return deploy_risk_mod.DeployTracker(max_per_service=16)
    caps = {entry.service: entry.max_active_deploys for entry in config.deploy_signals}
    return deploy_risk_mod.DeployTracker(max_per_service=caps, default_max=16)


def _sum_counter_series(
    frames: list[SeriesFrame],
) -> tuple[list[datetime], list[float]]:
    """Sum a multi-series counter response into one aggregate series.

    Used by the event-pattern path to fold per-(severity, team) counter
    fan-outs into a single per-step aggregate before bucketing. The
    timestamps come from the first frame; any frame whose timestamps
    don't match (different scrape cadence per series, for example) is
    dropped — the resulting aggregate is only meaningful when every
    contributor shares the same time index.

    Returns empty lists when ``frames`` is empty or no two frames
    agree on timestamps.
    """
    if not frames:
        return [], []
    primary = frames[0]
    timestamps = list(primary.timestamps)
    if not timestamps:
        return [], []
    n = len(timestamps)
    aggregate = [0.0] * n
    for frame in frames:
        if frame.timestamps != primary.timestamps:
            continue
        if len(frame.values) != n:
            continue
        for i, value in enumerate(frame.values):
            if math.isfinite(value):
                aggregate[i] += float(value)
    return timestamps, aggregate


def _aggregate_regressor_values(
    frames: list[SeriesFrame],
) -> tuple[list[float], list[float]]:
    """Collapse a multi-series regressor response to (values, timestamps).

    Used by the deploy-anchored risk path to inspect an anchor
    regressor's recent values. The reduction is a per-step mean
    across the returned series, matching the regressors module's
    own prefetch reduction so the runner sees the same number
    sequence the actual fit would have used.

    Returns empty lists when ``frames`` is empty or when the leading
    frame has no samples. The leading frame's timestamps anchor the
    output; other frames whose lengths don't match are silently
    dropped from the mean (their misaligned cadence would otherwise
    poison every row). The all-NaN case is *not* filtered here —
    the detector rejects NaN rows row-by-row instead.
    """
    if not frames:
        return [], []
    primary = frames[0]
    if not primary.values or not primary.timestamps:
        return [], []
    timestamps = [ts.timestamp() for ts in primary.timestamps]
    # Fast path: a single returned series — by far the common case
    # for an anchor regressor — avoids the per-row mean allocation.
    if len(frames) == 1:
        return [float(v) for v in primary.values], timestamps
    expected_len = len(primary.values)
    aligned = [
        [float(v) for v in frame.values] for frame in frames if len(frame.values) == expected_len
    ]
    # ``aligned`` always contains at least the primary frame (it
    # matches its own length), so no empty-aligned guard is needed.
    n = expected_len
    means: list[float] = []
    for i in range(n):
        samples = [series[i] for series in aligned if math.isfinite(series[i])]
        means.append(sum(samples) / len(samples) if samples else math.nan)
    return means, timestamps


def _trailing_regressor_values(regressors: Any | None) -> dict[str, float]:
    """Extract the most recent finite value of each regressor column.

    The drift explainer reads "what was loud at the drift moment" from
    the latest row of the aligned regressor DataFrame. NaN-trailing
    columns walk backwards to find the last finite reading so a
    one-step gap at the end of the lookback (common after a deploy)
    doesn't blank out the regressor's signal.

    Returns an empty dict when no regressors are configured (the
    common case) so the explainer falls back to ``"unknown"``
    cleanly.
    """
    if regressors is None:
        return {}
    out: dict[str, float] = {}
    # The frame is expected to be a pandas DataFrame whose columns are
    # ``unique_id``, ``ds``, and one column per regressor id. Avoid
    # importing pandas at module load — this helper is called only
    # when the runner has already imported it via the fit pipeline.
    for column in regressors.columns:
        if column in ("unique_id", "ds"):
            continue
        series = regressors[column]
        if series.empty:
            continue
        for value in reversed(list(series)):
            try:
                fv = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(fv):
                out[str(column)] = fv
                break
    return out


def _build_threshold_action_cost_points(group: GroupConfig) -> list[ThresholdActionCostPoint]:
    """Project a group's configured thresholds onto action-cost points.

    Pure projection from config — the points carry the operator's
    declared cost as the sample value and one row per
    (group, query, threshold) with ``action_cost`` configured.
    Independent of fit results: a query whose fit failed (or was
    skipped this tick) still publishes its action_cost rows so the
    Grafana "rank by cost" panel doesn't go blank.
    """
    out: list[ThresholdActionCostPoint] = []
    for query in group.queries:
        annotations = capacity_mod.build_threshold_action_costs(query.thresholds)
        for ann in annotations:
            out.append(
                ThresholdActionCostPoint(
                    labels={
                        "id": query.id,
                        "group": group.name,
                        "name": ann.name,
                        "type": ann.cost_type,
                    },
                    value=ann.cost_value,
                )
            )
    return out


def _staleness_max_age_seconds(config: Config) -> float:
    """Threshold for the staleness sweep on ``apply_config``.

    Returns the larger of one day and 24x the *shortest* configured
    refresh interval (so the sweep keeps pace with sub-hour cadences
    without dropping rows that are merely in their second alert
    cycle). Picked well past the canonical
    ``ForecastSeriesStale`` alert horizon (4x refresh) so an operator
    gets multiple alert cycles before a row silently disappears, while
    still bounding the memory cost of a noisy fleet.
    """
    shortest = config.server.refresh_interval.total_seconds()
    for group in config.groups:
        for query in group.queries:
            if query.refresh_interval is not None:
                shortest = min(shortest, query.refresh_interval.total_seconds())
    return max(86_400.0, shortest * 24)


class Runner:
    """Schedules and executes per-group forecast runs.

    ``safety.fit_timeout`` is enforced via ``asyncio.wait_for`` around a
    ``ThreadPoolExecutor`` task. This is a **soft** timeout: when it fires
    the asyncio task gives up and the run continues, but the underlying
    fit thread keeps running until statsforecast returns (or until the
    process exits). For pathological cases — degenerate series that hang
    AutoARIMA — this means a worker thread is consumed for longer than
    ``fit_timeout``. With ``max_workers`` sized comfortably above the
    expected concurrent fit count this is rarely visible. A hard timeout
    would require ``ProcessPoolExecutor`` with pool restart on timeout
    (~hundreds of MB of fork overhead per timeout); deferred until there's
    evidence it's needed.
    """

    def __init__(
        self,
        config: Config,
        source: PromSource,
        exporter: Exporter,
        executor: ThreadPoolExecutor | None = None,
        sink: ForecastSink | None = None,
        tracer: telemetry_mod.Tracer | None = None,
        webhook_client: WebhookRegressorClient | None = None,
        trigger_engine: TriggerEngine | None = None,
        annotation_sink: GrafanaAnnotationSink | None = None,
    ) -> None:
        self._config = config
        self._source = source
        self._exporter = exporter
        self._executor = executor or ThreadPoolExecutor(max_workers=4)
        self._sink = sink
        # Optional client for ``type: webhook`` regressors. ``None`` when no
        # webhook regressor is configured (the common case) — the prefetch
        # path then degrades any webhook regressor gracefully.
        self._webhook_client = webhook_client
        # Optional forecast-driven webhook triggers and Grafana-annotations
        # sink. Both ``None`` unless the operator configured them; both are
        # evaluated after each group run (post-snapshot, post-sink) so a
        # delivery hiccup never delays the standard exposition.
        self._trigger_engine = trigger_engine
        self._annotation_sink = annotation_sink
        # Drift cache: persists across runs in the same process so
        # successive fits can be compared. HA followers don't see the
        # cache; they read the leader's rendered drift gauges via the
        # shared snapshot.
        self._drift_cache = drift_mod.DriftCache()
        # Forecast-to-forecast revision cache: stores the previous
        # central yhat value per (series, model, horizon) so the next
        # fit can score how much that horizon's prediction moved. Same
        # lifecycle as ``_drift_cache`` — pruned on every reload to
        # stay bounded by the live config.
        self._revision_cache = revision_mod.RevisionCache()
        # Drift-explainer rolling windows. Keyed per
        # ``(group, query, series_key, model)`` so the Pearson
        # correlation between drift_score and each regressor value
        # captures the genuine per-series behaviour rather than
        # averaging across the query's whole fan-out. Pruned on
        # reload alongside the drift cache.
        self._drift_windows: dict[tuple[str, str, str, str], drift_explainer_mod.DriftWindow] = {}
        # Rolling band-coverage tracker: keeps a bounded ring of
        # (actual ∈ band) booleans per (series, level, window). Lives on
        # the Runner so the rolling state survives across runs without
        # leaking into the exporter's lock surface.
        self._band_coverage = band_coverage_mod.BandCoverageTracker()
        # Deploy-anchored risk score tracker. Keyed by service; bounded
        # per service so a runaway pipeline can't blow cardinality.
        # The cap is sourced from ``deploy_signals[].max_active_deploys``
        # so an operator can tune per-service overlap (a chatty internal
        # service may legitimately overlap deploys more than a stable
        # infrastructure tier). When ``deploy_signals`` is empty the
        # tracker stays harmlessly idle.
        self._deploy_tracker = _build_deploy_tracker(config)
        # Fleet what-if throttle. Lives here (rather than in the FastAPI
        # closure where it was originally) so ``apply_config`` can
        # update ``min_interval`` on reload without the operator
        # needing a process restart. ``main._build_app`` reads it via
        # ``self._what_if_throttle`` so the endpoint and the runner
        # share one instance.
        self._what_if_throttle = what_if_mod.WhatIfThrottle(
            min_interval_seconds=config.safety.fleet_what_if_min_interval.total_seconds(),
        )
        # Concurrent-fits semaphore. Per-query refresh schedules can
        # synchronise multiple groups onto the same tick; without a global
        # cap a burst of cheap queries could saturate the executor pool.
        # The semaphore is recreated whenever the config reload changes
        # ``safety.max_concurrent_fits`` so the cap honours hot reloads.
        self._fits_semaphore = asyncio.Semaphore(max(1, config.safety.max_concurrent_fits))
        # ``tracer`` is the optional OpenTelemetry facade. We accept None
        # here (rather than mandating one) so call sites that don't care
        # about tracing — tests, ad-hoc tools — don't have to construct a
        # disabled tracer themselves. The disabled tracer is a free
        # singleton; instantiating one in every test is harmless but
        # noisy.
        self._tracer = tracer or telemetry_mod.disabled()
        # Optional async hook called after every successful group run.
        # Used by HA mode to publish the rendered /metrics payload to the
        # shared snapshot cache; ``None`` in single-replica deployments.
        self._snapshot_publisher: Callable[[], Awaitable[None]] | None = None
        self._locks: dict[str, asyncio.Lock] = {g.name: asyncio.Lock() for g in config.groups}
        # Per-group task: keyed by group.name, holds the long-lived loop. We
        # key per-group (rather than a flat list) so apply_config can diff and
        # restart only the groups whose definition changed.
        self._tasks: dict[str, asyncio.Task[None]] = {}
        # Per-group cooperative-stop signals. apply_config sets these before
        # waiting for the loop to drain, instead of cancelling the task —
        # cancellation would tear down a mid-flight ``run_group`` and lose
        # the snapshot it was about to write.
        self._stop_events: dict[str, asyncio.Event] = {}
        self._stopped = asyncio.Event()
        self._global_budgets = _compute_global_budgets(
            list(config.groups), config.safety.max_total_series
        )
        # Reload coordination: a single lock so concurrent reload requests
        # (e.g. /-/reload + SIGHUP arriving together) serialise cleanly.
        self._reload_lock = asyncio.Lock()

    def set_snapshot_publisher(self, publisher: Callable[[], Awaitable[None]] | None) -> None:
        """Install (or clear) the post-run snapshot publishing hook.

        HA leaders use this to push the rendered /metrics payload into the
        shared cache after each successful group run. Idempotent; called
        once at boot from :func:`main.run`.
        """
        self._snapshot_publisher = publisher

    async def start(self) -> None:
        """Run each group once immediately, then on the configured interval."""
        # Guard against double-start: HA mode's leader-acquired callback
        # invokes ``start`` every time the lease flips back to us. Without
        # this we'd spawn duplicate group tasks each acquisition.
        if self._tasks:
            return
        self._stopped.clear()
        for group in self._config.groups:
            self._spawn_group_loop(group)

    def _spawn_group_loop(self, group: GroupConfig) -> None:
        """Create the stop event + task for one group. Idempotent per name."""
        ev = asyncio.Event()
        self._stop_events[group.name] = ev
        self._tasks[group.name] = asyncio.create_task(self._loop(group, ev))

    async def apply_config(self, new_config: Config) -> ReloadResult:
        """Swap in a new validated config, draining in-flight runs first.

        Validate-before-swap is the caller's contract: ``new_config`` must
        already be a fully-validated :class:`Config`. This method only
        performs the runtime swap so a parse failure can't kill in-flight
        runs.

        The drain is *cooperative*: the per-group stop events are set, then
        we wait for each loop to finish its current ``run_group`` and exit
        on its own. This is deliberate — interrupting a fit mid-flight
        wastes the executor work that has already happened and loses the
        snapshot the run was about to write. A bounded grace period
        (``2 * safety.fit_timeout``, with a 60s floor) caps the worst case;
        any task that hasn't drained by then is hard-cancelled.

        Identical configs are a fast no-op (no drain, no restart).

        Returns a :class:`ReloadResult` summarising what changed. Removed
        groups have their snapshots cleared from the exporter so /metrics no
        longer shows their series.
        """
        async with self._reload_lock:
            old = self._config
            old_by_name = {g.name: g for g in old.groups}
            new_by_name = {g.name: g for g in new_config.groups}

            added = sorted(set(new_by_name) - set(old_by_name))
            removed = sorted(set(old_by_name) - set(new_by_name))
            changed = sorted(
                name
                for name in set(new_by_name) & set(old_by_name)
                if old_by_name[name] != new_by_name[name]
            )

            log = logger.bind(added=added, removed=removed, changed=changed)

            # Unconditionally sweep the staleness state by age before
            # the fast no-op short-circuit. The sweep is idempotent and
            # cheap (dict comprehension over a bounded set), and running
            # it on every reload gives operators a deterministic
            # "trigger a cleanup" knob even when the config itself
            # hasn't changed (the per-group run path also sweeps, but
            # an explicit reload is a clearer mental model for
            # debugging stale rows).
            self._exporter.prune_stale_staleness(_staleness_max_age_seconds(new_config))

            # Fast no-op path: pydantic models compare structurally, so an
            # equal config means nothing changed and we shouldn't drain.
            # This matters when a ConfigMap is re-projected with identical
            # bytes (kubelet behaviour) — no group should pause.
            if new_config == old:
                log.info("config_reload_no_op")
                return ReloadResult()

            log.info("config_reload_apply")

            await self._drain_all_groups(grace_seconds=self._drain_grace_seconds(old))

            # All loops are quiescent — safe to swap config + budgets without
            # racing readers. Loops that read self._config later will all see
            # the new value consistently.
            self._config = new_config
            self._global_budgets = _compute_global_budgets(
                list(new_config.groups), new_config.safety.max_total_series
            )
            self._locks = {g.name: asyncio.Lock() for g in new_config.groups}
            self._exporter.retain_groups(set(new_by_name))
            self._exporter.retain_discovery_groups(set(new_by_name))
            self._exporter.retain_incident_recurrence_groups(set(new_by_name))
            self._exporter.retain_staleness_groups(set(new_by_name))
            self._exporter.retain_self_observability_groups(set(new_by_name))
            # Drop fit-success-ratio buckets for models removed from
            # ``defaults.models``; without this a dropped model's
            # frozen-in-time ratio rows would linger on /metrics until
            # process restart, and the ``ForecastModelFitFailing`` alert
            # would either misfire on the stale numbers or have to be
            # silenced manually.
            self._exporter.retain_fit_outcome_models(
                {spec.name for spec in new_config.defaults.models}
            )
            # Per-query staleness cleanup: a reload that drops a single
            # query from a still-present group needs the per-(group, query)
            # entries pruned too, mirroring the incident-recurrence
            # treatment below. Without this the dropped query's series
            # would linger on /metrics with a forever-growing staleness
            # value because no fit will ever reset their clock.
            live_query_pairs = {
                (group.name, query.id) for group in new_config.groups for query in group.queries
            }
            self._exporter.retain_staleness_queries(live_query_pairs)
            self._exporter.retain_regressor_stability_queries(live_query_pairs)
            # Per-query cleanup for the incident-recurrence heatmap:
            # the group-level retain above handles whole-group removal,
            # but a reload that drops a single ``data_profile: events``
            # query from a still-present group needs a per-query
            # sweep too, otherwise its 168 rows linger on /metrics.
            self._exporter.retain_incident_recurrence_queries(
                {
                    (group.name, query.id)
                    for group in new_config.groups
                    for query in group.queries
                    if query.data_profile == "events"
                }
            )

            # Reload may have changed ``safety.max_concurrent_fits``; rebuild
            # the semaphore so the new cap takes effect for upcoming fits.
            self._fits_semaphore = asyncio.Semaphore(max(1, new_config.safety.max_concurrent_fits))
            # Drift/band-coverage state lives keyed by ``(group, query)`` —
            # prune entries for queries that no longer exist so a hot
            # reload that drops a query doesn't leak its state forever.
            live_pairs = {(g.name, q.id) for g in new_config.groups for q in g.queries}
            self._drift_cache.prune_to(live_pairs)
            self._revision_cache.prune_to(live_pairs)
            self._band_coverage.forget(live_pairs)
            # Drift-explainer windows are keyed per
            # ``(group, query, series_key, model)`` — the live_pairs
            # set is ``(group, query)``; we project the windows to
            # those that survive the reload.
            self._drift_windows = {
                key: window
                for key, window in self._drift_windows.items()
                if (key[0], key[1]) in live_pairs
            }
            self._exporter.retain_drift_explained_queries(live_pairs)
            # Rebuild the deploy tracker against the new spec set. The
            # rebuild deliberately drops every in-flight active deploy —
            # a reload mid-rollout starts the next risk-score emission
            # from a clean slate, which is the documented trade-off
            # (the alternative would be reconciling per-service caps
            # against possibly-shrunken specs and trying to preserve
            # deployment ids whose ``service`` is still in the new
            # config, code we don't yet need).
            #
            # Always clear the gauge state too, not only when the new
            # ``deploy_signals`` list is empty. A partial reload that
            # drops one service from a multi-service list also needs
            # the scrape view cleared — otherwise the dropped
            # service's last-known scores linger on /metrics until
            # the next group refresh re-publishes (which may be
            # minutes away, and never if no groups remain that
            # touch any anchor). The next ``_publish_deploy_risk_scores``
            # call repopulates from the rebuilt tracker.
            self._deploy_tracker = _build_deploy_tracker(new_config)
            self._exporter.replace_deploy_risk_scores([])
            # A reload that turns off ``governance.health_score`` must
            # drop the prior aggregated rows immediately rather than
            # waiting for the next group run to clear them. The
            # opposite case (newly enabled) just lets the next group
            # run repopulate via :meth:`_refresh_service_health_scores`.
            if new_config.governance.health_score is None:
                self._exporter.replace_service_health_scores([])
            # Same treatment for right-sizing — a reload that disables
            # the feature (or empties the workload list) must drop the
            # prior recommendations.
            new_rs = new_config.recommendations.resource_sizing
            if not new_rs.enabled or not new_rs.workloads:
                self._exporter.replace_resource_sizing_recommendations([])
            # Hot-reload the what-if throttle interval. The next-allowed
            # timestamp inside the throttle is preserved so a reload
            # can't be used to bypass an in-flight throttle window;
            # the new interval applies to subsequent calls only.
            self._what_if_throttle.update_min_interval(
                new_config.safety.fleet_what_if_min_interval.total_seconds(),
            )
            # Rebuild the v1.9 outbound/inbound integrations (webhook
            # regressor client, forecast-driven trigger engine, Grafana
            # annotations sink) so a hot reload that adds, changes, or
            # removes ``triggers:`` / ``annotations:`` / webhook regressors
            # takes effect without a restart — the loops are quiesced here,
            # so swapping them is race-free.
            await self._rebuild_integrations(old_config=old)
            for group in new_config.groups:
                self._spawn_group_loop(group)

            return ReloadResult(added=added, removed=removed, changed=changed)

    async def _rebuild_integrations(self, *, old_config: Config) -> None:
        """Reconcile webhook client / trigger engine / annotation sink to config.

        Called from :meth:`apply_config` after the config swap (loops already
        drained). Each integration is only rebuilt when its slice of the
        config actually changed, so an unrelated reload preserves the trigger
        rising-edge state and the annotation rising-edge dedup set rather than
        re-firing everything that is currently active.
        """
        new = self._config
        # Webhook client: needed iff some query still configures a webhook
        # regressor. The client is config-agnostic (each ``fetch`` reads the
        # live ``reg``), so keep an existing one rather than churning its
        # connection pool + response cache; only create / tear down on the
        # needed-ness boundary.
        needs_webhook = any(
            reg.type == "webhook"
            for group in new.groups
            for query in group.queries
            for reg in query.regressors
        )
        if needs_webhook and self._webhook_client is None:
            from ..webhook import WebhookRegressorClient  # noqa: PLC0415

            self._webhook_client = WebhookRegressorClient()
            logger.info("webhook_regressor_client_enabled_on_reload")
        elif not needs_webhook and self._webhook_client is not None:
            stale_client = self._webhook_client
            self._webhook_client = None
            with contextlib.suppress(Exception):
                await stale_client.aclose()
            logger.info("webhook_regressor_client_disabled_on_reload")

        # Trigger engine: rebuild only when the trigger list changed.
        if new.triggers != old_config.triggers:
            stale_engine = self._trigger_engine
            if new.triggers:
                from ..triggers import TriggerEngine  # noqa: PLC0415

                self._trigger_engine = TriggerEngine(
                    triggers=new.triggers, source=self._source, exporter=self._exporter
                )
                logger.info("triggers_reloaded", count=len(new.triggers))
            else:
                self._trigger_engine = None
                logger.info("triggers_disabled_on_reload")
            if stale_engine is not None:
                with contextlib.suppress(Exception):
                    await stale_engine.aclose()

        # Annotation sink: rebuild only when the annotations block changed.
        if new.annotations != old_config.annotations:
            stale_sink = self._annotation_sink
            if new.annotations.enabled:
                from ..annotations import GrafanaAnnotationSink  # noqa: PLC0415

                self._annotation_sink = GrafanaAnnotationSink(new.annotations, self._exporter)
                logger.info("grafana_annotations_reloaded", events=list(new.annotations.events))
            else:
                self._annotation_sink = None
                logger.info("grafana_annotations_disabled_on_reload")
            if stale_sink is not None:
                with contextlib.suppress(Exception):
                    await stale_sink.aclose()

    @staticmethod
    def _drain_grace_seconds(config: Config) -> float:
        """How long to wait for an in-flight run to finish before hard-cancelling.

        We pick ``max(60s, 2 * fit_timeout)`` so a single backtest+forecast
        round (typically capped at fit_timeout) has comfortable headroom,
        and we always wait at least a minute even if fit_timeout is set
        very low. Tests can override fit_timeout to drive this down.
        """
        return max(60.0, 2.0 * config.safety.fit_timeout.total_seconds())

    async def _drain_all_groups(self, *, grace_seconds: float) -> None:
        """Signal every group to stop after its current run, then await drain.

        Loops that are currently sleeping will exit immediately when their
        stop event fires. Loops mid-``run_group`` will finish the run, write
        their snapshot, and exit on the next iteration check. Anything still
        running after ``grace_seconds`` is hard-cancelled to keep reload
        latency bounded.
        """
        for ev in self._stop_events.values():
            ev.set()
        tasks = list(self._tasks.values())
        self._tasks.clear()
        self._stop_events.clear()
        if not tasks:
            return
        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=grace_seconds,
            )
        except TimeoutError:
            logger.warning(
                "config_reload_drain_timeout",
                grace_s=grace_seconds,
                still_running=sum(1 for t in tasks if not t.done()),
            )
            for task in tasks:
                task.cancel()
            for task in tasks:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task

    async def _loop(
        self,
        group: GroupConfig,
        stop_event: asyncio.Event,
    ) -> None:
        """Run each query of ``group`` on its own effective refresh interval.

        Per-query ``refresh_interval`` overrides let a single group mix
        metrics with different cadences. The loop tracks the
        per-query next-run-time and on each tick runs only the queries
        that are due. Queries with no override inherit
        ``server.refresh_interval``; behaviour is unchanged from the
        single-cadence world when no overrides are configured.
        """
        # ``-inf`` so every query is due on the first tick (the runner
        # always fires an immediate first run, matching the old contract).
        next_run_at: dict[str, float] = {q.id: float("-inf") for q in group.queries}
        while not stop_event.is_set() and not self._stopped.is_set():
            now = time.monotonic()
            due = [q for q in group.queries if next_run_at[q.id] <= now]
            try:
                if due:
                    await self.run_group(group, queries=due)
                    for q in due:
                        interval = effective_refresh_interval(
                            q, self._config.server
                        ).total_seconds()
                        next_run_at[q.id] = now + max(1.0, interval)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("group_run_failed", group=group.name)
                self._exporter.increment_failure(group.name, reason="group_error")
            sleep_seconds = self._next_query_due_in(group, next_run_at)
            await self._sleep_until_stopped_or(stop_event, sleep_seconds)

    @staticmethod
    def _next_query_due_in(group: GroupConfig, next_run_at: dict[str, float]) -> float:
        """Return seconds until the soonest-due query, clamped to >= 1s.

        We never sleep sub-second so a misconfigured tiny refresh
        interval can't busy-loop the event loop; the tick granularity
        of one second is comfortably finer than any realistic
        refit cadence.
        """
        now = time.monotonic()
        # Floor at 1s so a barely-overdue query that just ran doesn't
        # spin on an immediate wake. Ceil chosen pragmatically: the
        # group refresh is typically minutes-to-hours.
        wait = float("inf")
        for query in group.queries:
            wait = min(wait, next_run_at[query.id] - now)
        if not group.queries or not math.isfinite(wait):
            wait = 60.0
        return max(1.0, wait)

    async def _sleep_until_stopped_or(
        self,
        stop_event: asyncio.Event,
        timeout_seconds: float,
    ) -> None:
        """Sleep up to ``timeout_seconds``; return early on either stop signal.

        Implementing this with ``asyncio.wait`` (rather than two nested
        ``wait_for`` calls) keeps the cancellation semantics simple: when
        ``timeout`` expires we cancel the unsignalled waiters explicitly so
        no orphan tasks linger across loop iterations.
        """
        waiters = [
            asyncio.create_task(stop_event.wait()),
            asyncio.create_task(self._stopped.wait()),
        ]
        try:
            await asyncio.wait(
                waiters,
                timeout=timeout_seconds,
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            for waiter in waiters:
                if not waiter.done():
                    waiter.cancel()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await waiter

    async def run_group(  # noqa: PLR0912, PLR0915
        self,
        group: GroupConfig,
        *,
        queries: list[QueryConfig] | None = None,
    ) -> None:
        """Execute (a subset of) a group's queries: query, fit, select, export.

        ``queries=None`` runs every query in the group — the legacy
        single-cadence behaviour. When the per-query refresh scheduler
        passes a subset, only those queries are fitted; data for the
        other queries is preserved from the prior snapshot so a
        partial-run tick doesn't strip /metrics of stable rows.
        """
        queries_to_run = queries if queries is not None else list(group.queries)
        run_query_ids = {q.id for q in queries_to_run}
        async with self._locks[group.name]:
            started = time.monotonic()
            log = logger.bind(
                group=group.name,
                query_count=len(queries_to_run),
                total_queries=len(group.queries),
            )
            log.info("group_run_start")

            ctx = _RunContext(global_remaining=self._global_budgets.get(group.name, 0))
            previous_snapshot = self._exporter.snapshot_for(group.name)
            if previous_snapshot is not None and len(run_query_ids) != len(group.queries):
                _inherit_prior_for_skipped_queries(
                    ctx, previous_snapshot, run_query_ids=run_query_ids
                )
            with self._tracer.span(
                "promforecast.group_run",
                group=group.name,
                query_count=len(queries_to_run),
            ):
                for query in queries_to_run:
                    q_started = time.monotonic()
                    await self._run_query(query, group, ctx, log)
                    ctx.run_duration_per_query[query.id] = time.monotonic() - q_started

            duration = time.monotonic() - started
            # Counter semantics: merge this run's slow_fits with the previous
            # snapshot's so the exposed counter never decreases. Without the
            # merge each run would reset to 0 — Prometheus would treat that
            # as a counter-reset and rate() would lie. We *prune* the merge
            # to (query, model) tuples whose query still exists in the
            # current group config; otherwise a hot-reload that drops a
            # query would leak its old slow_fits label set forever.
            current_query_ids = {q.id for q in group.queries}
            merged_slow: dict[tuple[str, str], int] = (
                {
                    key: count
                    for key, count in previous_snapshot.slow_fits.items()
                    if key[0] in current_query_ids
                }
                if previous_snapshot is not None
                else {}
            )
            for key, count in ctx.slow_fits.items():
                merged_slow[key] = merged_slow.get(key, 0) + count
            # Same monotonicity treatment for regressor failures, pruned to
            # (query, regressor) tuples that still exist in the live config.
            current_regressor_pairs = {(q.id, r.id) for q in group.queries for r in q.regressors}
            merged_regressor_failures: dict[tuple[str, str, str], int] = (
                {
                    key: count
                    for key, count in previous_snapshot.regressor_failures.items()
                    if (key[0], key[1]) in current_regressor_pairs
                }
                if previous_snapshot is not None
                else {}
            )
            for rkey, rcount in ctx.regressor_failures.items():
                merged_regressor_failures[rkey] = merged_regressor_failures.get(rkey, 0) + rcount
            # Same monotonic merge for the preprocessing counters, pruned
            # to query IDs that still exist in the live group config.
            merged_gaps: dict[tuple[str, str], int] = (
                {
                    key: count
                    for key, count in previous_snapshot.gaps_filled_cumulative.items()
                    if key[0] in current_query_ids
                }
                if previous_snapshot is not None
                else {}
            )
            for gkey, gcount in ctx.gaps_filled.items():
                merged_gaps[gkey] = merged_gaps.get(gkey, 0) + gcount
            merged_outliers: dict[tuple[str, str], int] = (
                {
                    key: count
                    for key, count in previous_snapshot.outliers_replaced_cumulative.items()
                    if key[0] in current_query_ids
                }
                if previous_snapshot is not None
                else {}
            )
            for okey, ocount in ctx.outliers_replaced.items():
                merged_outliers[okey] = merged_outliers.get(okey, 0) + ocount
            merged_changepoints: dict[str, int] = (
                {
                    key: count
                    for key, count in previous_snapshot.change_points_cumulative.items()
                    if key in current_query_ids
                }
                if previous_snapshot is not None
                else {}
            )
            for cpkey, cpcount in ctx.change_points.items():
                merged_changepoints[cpkey] = merged_changepoints.get(cpkey, 0) + cpcount
            # Blacklist-window sample counter: monotonic merge keyed on
            # (query, reason), pruned to live query IDs so a reload that
            # drops a query also drops its row.
            merged_blacklisted: dict[tuple[str, str], int] = (
                {
                    key: count
                    for key, count in previous_snapshot.blacklisted_samples_cumulative.items()
                    if key[0] in current_query_ids
                }
                if previous_snapshot is not None
                else {}
            )
            for bkey, bcount in ctx.blacklisted_samples.items():
                merged_blacklisted[bkey] = merged_blacklisted.get(bkey, 0) + bcount
            # Monotonic merge for the emission-clipped counter, pruned
            # to query IDs that still exist in the live group config so a
            # reload that drops a query doesn't leak its old label set
            # forever.
            merged_emission_clipped: dict[tuple[str, str], int] = (
                {
                    key: count
                    for key, count in previous_snapshot.emission_clipped_cumulative.items()
                    if key[0] in current_query_ids
                }
                if previous_snapshot is not None
                else {}
            )
            for ekey, ecount in ctx.emission_clipped.items():
                merged_emission_clipped[ekey] = merged_emission_clipped.get(ekey, 0) + ecount
            # Drift-alert counter: same monotonic-merge pattern as the
            # other per-(query) counters above. Skipped queries inherit
            # their prior cumulative value via this merge (rather than
            # through ``_inherit_prior_for_skipped_queries``), so the
            # counter is preserved without a separate inherit path.
            merged_drift_alerts: dict[str, int] = (
                {
                    key: count
                    for key, count in previous_snapshot.drift_alerts_cumulative.items()
                    if key in current_query_ids
                }
                if previous_snapshot is not None
                else {}
            )
            for dkey, dcount in ctx.drift_alerts.items():
                merged_drift_alerts[dkey] = merged_drift_alerts.get(dkey, 0) + dcount
            # Band-coverage gauges are rendered from the persistent
            # tracker on the runner — pull a snapshot now so the next
            # /metrics scrape sees the most recent rolling fractions.
            band_coverage_points = self._build_band_coverage_points(group, run_query_ids)
            # Threshold action-cost is config-derived (not fit-derived),
            # so rebuild the list directly from the current group config
            # every run. A threshold removed from YAML disappears from
            # /metrics on the next refresh; partial-tick refreshes still
            # publish the full set because every configured query is
            # walked here regardless of which subset actually ran.
            threshold_action_cost_points = _build_threshold_action_cost_points(group)
            self._exporter.update(
                GroupSnapshot(
                    group=group.name,
                    points=ctx.points,
                    accuracy_points=ctx.accuracy_points,
                    deviation_points=ctx.deviation_points,
                    canary_deviation_points=ctx.canary_deviation_points,
                    quality_points=ctx.quality_points,
                    ensemble_weights=ctx.ensemble_weight_points,
                    contribution_points=ctx.contribution_points,
                    threshold_etas=ctx.threshold_etas,
                    growth_rates=ctx.growth_rates,
                    will_breach_points=ctx.will_breach_points,
                    time_to_drain_points=ctx.time_to_drain_points,
                    error_budget_points=ctx.error_budget_points,
                    recommended_maintenance_window_points=(
                        ctx.recommended_maintenance_window_points
                    ),
                    conditional_points=ctx.conditional_points,
                    conditional_deviation_points=ctx.conditional_deviation_points,
                    component_points=ctx.component_points,
                    last_run_timestamp=time.time(),
                    run_duration_seconds=duration,
                    run_duration_per_query=dict(ctx.run_duration_per_query),
                    series_count=ctx.series_count,
                    series_count_per_query=dict(ctx.series_count_per_query),
                    series_dropped=ctx.dropped,
                    failures=ctx.failures,
                    regressor_failures=merged_regressor_failures,
                    # ``fit_durations`` is a gauge — only this run's samples
                    # belong on /metrics. Removed queries simply disappear
                    # from the exposition until they reappear in config.
                    fit_durations=ctx.fit_durations,
                    slow_fits=merged_slow,
                    calibration_offsets=ctx.calibration_offsets,
                    conditional_calibration_offsets=ctx.conditional_calibration_offsets,
                    preprocess_stats=ctx.preprocess_stats,
                    gaps_filled_cumulative=merged_gaps,
                    outliers_replaced_cumulative=merged_outliers,
                    change_points_cumulative=merged_changepoints,
                    blacklisted_samples_cumulative=merged_blacklisted,
                    emission_clipped_cumulative=merged_emission_clipped,
                    cold_start_siblings=dict(ctx.cold_start_siblings),
                    drift_points=list(ctx.drift_points),
                    drift_alerts_cumulative=merged_drift_alerts,
                    band_coverage_points=band_coverage_points,
                    revision_points=list(ctx.revision_points),
                    model_disagreement_points=list(ctx.model_disagreement_points),
                    threshold_action_cost_points=threshold_action_cost_points,
                )
            )
            log.info(
                "group_run_complete",
                duration_s=round(duration, 3),
                series=ctx.series_count,
                points=len(ctx.points),
                accuracy_points=len(ctx.accuracy_points),
                deviation_points=len(ctx.deviation_points),
                quality_points=len(ctx.quality_points),
                ensemble_weight_points=len(ctx.ensemble_weight_points),
                contribution_points=len(ctx.contribution_points),
                curve_points=len(ctx.curve_points),
                dropped=dict(ctx.dropped),
                failures=dict(ctx.failures),
                regressor_failures=len(ctx.regressor_failures),
            )

            # Self-observability: per-group resource attribution.
            # CPU is the sum of per-(query, model) wall-clock seconds this
            # run consumed on the shared executor — the runner's only
            # honest CPU number without process-level instrumentation.
            # **Crucially** we filter to queries that *ran this tick* — the
            # inheritance pass populates ``ctx.fit_durations`` with prior
            # values for skipped queries, and those durations already
            # contributed to the counter in an earlier cycle. Without the
            # filter, partial-tick refreshes would silently re-add the
            # same seconds on every refresh. Memory is a best-effort
            # snapshot footprint captured here — emitting it before the
            # snapshot is updated would miss the current run's contribution.
            cpu_this_run = sum(
                seconds for (q_id, _), seconds in ctx.fit_durations.items() if q_id in run_query_ids
            )
            if cpu_this_run > 0:
                self._exporter.add_group_cpu_seconds(group.name, cpu_this_run)
            self._exporter.set_group_memory_bytes(
                group.name, _estimate_group_memory_bytes(group.name, self._exporter)
            )

            # Publish the rendered snapshot to the shared HA cache if a
            # publisher is wired. We do this *after* updating the local
            # exporter so the rendered payload reflects the run we just
            # finished. Errors are swallowed by the publisher (the
            # snapshot cache logs internally) — a Redis blip mustn't
            # crash the run.
            if self._snapshot_publisher is not None:
                try:
                    await self._snapshot_publisher()
                except Exception:
                    log.exception("snapshot_publish_failed")

            # Push to optional remote_write sink *after* the snapshot is
            # exposed via /metrics so a sink failure never delays the
            # standard exposition. Sink errors are bounded into the
            # ``forecast_remote_write_failures_total`` counter so dashboards
            # can alert on chronic push failures without crashing the run.
            # ``on_retry`` is bound to *this* group so transient retries
            # show up under the right label without the sink needing to
            # know about group identity.
            sink_payload = (
                ctx.curve_points + ctx.conditional_curve_points + ctx.component_curve_points
            )
            if self._sink is not None and sink_payload:

                def _record_sink_retry(reason: str) -> None:
                    self._exporter.increment_remote_write_retry(group.name, reason)

                try:
                    await self._sink.write(sink_payload, on_retry=_record_sink_retry)
                except SinkError as exc:
                    log.warning(
                        "remote_write_failed",
                        reason=exc.reason,
                        error=str(exc),
                        points=len(sink_payload),
                    )
                    self._exporter.increment_remote_write_failure(group.name, exc.reason)
                except Exception as exc:
                    log.exception("remote_write_unexpected_error")
                    self._exporter.increment_remote_write_failure(group.name, "unexpected")
                    del exc  # logged via exception(); avoid unused-name lint

            # Update deploy-anchored risk scores when this group's
            # queries include any anchor referenced by ``deploy_signals``.
            # Best-effort: a fetch failure is logged but never crashes
            # the run.
            try:
                await self._refresh_deploy_risk(group, run_query_ids)
            except Exception:
                log.exception("deploy_risk_refresh_failed")

            # Grafana annotations + forecast-driven webhook triggers run
            # last, after the snapshot, sink, and deploy-risk refresh, so
            # (a) the standard exposition is never delayed by an external
            # POST and (b) trigger conditions can read the freshest metrics
            # (sink) and annotation events the freshest deploy-risk scores.
            # Both swallow their own delivery errors; this wrapper guards
            # against an unexpected defect crashing the group run.
            if self._annotation_sink is not None:
                try:
                    await self._annotation_sink.post_events(
                        self._build_annotation_events(group, ctx), scope=group.name
                    )
                except Exception:
                    log.exception("annotation_post_failed")
            if self._trigger_engine is not None:
                try:
                    await self._trigger_engine.evaluate()
                except Exception:
                    log.exception("trigger_evaluate_failed")

            # Recompute the service-level health composite across all
            # current group snapshots. Cheap (a single pass over the
            # already-on-exporter quality_points) and idempotent — every
            # group run rebuilds the full list rather than diff-merging.
            # Best-effort: a defect in the aggregator logs but never
            # crashes the run, matching the
            # ``_refresh_resource_sizing_recommendations`` /
            # ``_refresh_deploy_risk`` failure-isolation pattern.
            try:
                self._refresh_service_health_scores()
            except Exception:
                log.exception("service_health_refresh_failed")

            # Refresh per-workload right-sizing recommendations.
            # Best-effort: any current_promql fetch failure logs but
            # never crashes the run. Idempotent — whole-state
            # replacement on the exporter drops workloads whose
            # forecast fell below the quality floor this refresh.
            try:
                await self._refresh_resource_sizing_recommendations()
            except Exception:
                log.exception("resource_sizing_refresh_failed")

            # Sweep aged staleness rows after each successful group run.
            # This is the only path that catches ephemeral-series leaks
            # (per-pod metrics, deployment-id labels) without requiring
            # an explicit ``/-/reload`` — the apply_config sweep handles
            # model removal, but a series that simply stops appearing
            # in the source data would otherwise linger forever. The
            # operation is a single dict comprehension over a bounded
            # dict, so amortising it across every group run adds
            # negligible latency to the fit path.
            self._exporter.prune_stale_staleness(_staleness_max_age_seconds(self._config))

    async def _refresh_deploy_risk(
        self,
        group: GroupConfig,
        run_query_ids: set[str],
    ) -> None:
        """Detect deploys from anchor regressors and emit risk scores.

        Cheap per-call: only the anchor queries whose query just ran
        this tick are polled. Re-firing on every subsequent fit is
        prevented by two layers — the detector
        (:func:`deploy_risk.detect_deploy_from_regressor`) returns
        ``observed=False`` while the regressor stays above the
        threshold (it requires a fresh ``previous <= threshold ->
        leading > threshold`` transition on the leading pair), and
        :meth:`DeployTracker.record` dedups on
        ``(service, deployment_id)`` so a transient blip that
        produces the same wall-clock ``deployment_id`` twice
        collapses to one event. The function does **not** keep a
        per-(group, query, regressor) memo across refreshes — the
        edge check is computed locally from the query_range window
        each tick.

        Refresh-cadence caveat: detection only catches deploys whose
        transition lands inside the *last two samples* of the
        anchor's lookback window. For the common case where
        ``refresh_interval ≈ step`` this is fine (each tick brings
        in one new sample and the leading pair compares old vs.
        new). Configurations where ``refresh_interval`` is much
        larger than the regressor's step can under-detect because
        the transition gets stranded mid-window — operators with
        that shape should add a per-(group, query, regressor) memo
        as a follow-up.

        The function is deliberately defensive — every failure mode
        (regressor fetch failure, missing regressor row, signal
        deviation lookup miss) degrades gracefully. The worst case is
        "no deploy detected this cycle"; the runner's main loop is
        unaffected.
        """
        deploy_specs = self._config.deploy_signals
        if not deploy_specs:
            return
        relevant_specs = [
            spec
            for spec in deploy_specs
            if spec.anchor.type == "regressor"
            and any(q.id == spec.anchor.query and q.id in run_query_ids for q in group.queries)
        ]
        if relevant_specs:
            await self._detect_deploys_for_specs(group, relevant_specs)
        # Score recomputation always runs (even when no new deploy was
        # observed this tick) so an active deploy whose signals are now
        # updated by a different group's run sees its score refresh.
        self._publish_deploy_risk_scores()

    async def _detect_deploys_for_specs(
        self,
        group: GroupConfig,
        specs: list[DeploySignalsConfig],
    ) -> None:
        """Poll each relevant spec's anchor regressor and register fresh deploys."""
        # Fetch each anchor regressor's recent values via a small
        # query_range. The lookback is intentionally narrow (one hour
        # at the configured step) — we only need the leading-pair to
        # decide "transitioned across threshold".
        end = datetime.now(tz=UTC)
        for spec in specs:
            query = next((q for q in group.queries if q.id == spec.anchor.query), None)
            if query is None:
                continue
            regressor = next(
                (r for r in query.regressors if r.id == spec.anchor.regressor_id),
                None,
            )
            if regressor is None or not regressor.promql:
                # Calendar / holidays regressors don't have a PromQL —
                # using them as anchors is not meaningful and was
                # rejected at config load, but guard here too.
                continue
            eff_step = effective_step(query, self._config.defaults)
            step_seconds = max(1, int(eff_step.total_seconds()))
            # Two hours of lookback gives us a transition window
            # regardless of refresh cadence; the detector only needs
            # the leading pair.
            start = end - timedelta(hours=2)
            try:
                frames = await self._source.query_range(regressor.promql, start, end, step_seconds)
            except Exception:
                logger.exception(
                    "deploy_anchor_query_failed",
                    group=group.name,
                    service=spec.service,
                    regressor=regressor.id,
                )
                continue
            if not frames:
                continue
            # Aggregate across returned series via mean — same shape
            # as the regressor module's own prefetch reduction.
            values, timestamps = _aggregate_regressor_values(frames)
            observed, ts = deploy_risk_mod.detect_deploy_from_regressor(
                values=values,
                timestamps_unix=timestamps,
                threshold=spec.anchor.threshold,
            )
            if not observed or ts is None:
                continue
            event = deploy_risk_mod.DeployEvent(
                deployment_id=deploy_risk_mod.derive_deployment_id(spec.service, ts),
                service=spec.service,
                observed_at_seconds=time.monotonic(),
                active_window_seconds=deploy_risk_mod.active_window_seconds(spec.active_window),
            )
            self._deploy_tracker.record(event)
            logger.info(
                "deploy_detected",
                service=spec.service,
                deployment_id=event.deployment_id,
                anchor_query=spec.anchor.query,
                anchor_regressor=regressor.id,
            )

    def _publish_deploy_risk_scores(self) -> None:
        """Recompute scores from the live snapshot and push to the exporter."""
        deploy_specs = self._config.deploy_signals
        if not deploy_specs:
            return
        signals_by_service: dict[str, list[deploy_risk_mod.SignalSpec]] = {
            spec.service: [
                deploy_risk_mod.SignalSpec(query_id=s.query, weight=s.weight) for s in spec.signals
            ]
            for spec in deploy_specs
        }
        active = self._deploy_tracker.active()
        if not active:
            # No active deploys — clear the gauge so an expired deploy
            # disappears from /metrics immediately.
            self._exporter.replace_deploy_risk_scores([])
            return
        # Build a per-service deviation lookup against the most recent
        # exporter snapshot. The lookup returns the absolute deviation
        # ratio averaged across every (series, model, level) row for
        # the named query — operators read the composite as a
        # service-level score, not a per-(series, model) one.
        lookup = self._build_deviation_lookup()
        scores = deploy_risk_mod.compute_deploy_risk_scores(
            deploys=active,
            signals_by_service=signals_by_service,
            deviation_lookup=lookup,
        )
        points = [
            DeployRiskPoint(
                labels={
                    "service": score.service,
                    "deployment_id": score.deployment_id,
                    "contributing_signals": str(score.contributing_signals),
                },
                score=score.score,
            )
            for score in scores
        ]
        self._exporter.replace_deploy_risk_scores(points)

    def _refresh_service_health_scores(self) -> None:
        """Rebuild ``forecast_service_health_score`` from current snapshots.

        Cheap: walks the per-group quality_points already on the
        exporter and feeds them through
        :func:`health.compute_service_health_scores`. The runner calls
        this after every group run; whole-state replacement on the
        exporter means buckets that disappear (e.g. the only series
        whose label set matched the dim tuple dropped to NaN this
        refresh) leave /metrics on the next scrape.

        No-op when ``governance.health_score`` is unset, so deployments
        that don't opt in pay zero cost.
        """
        cfg = self._config.governance.health_score
        if cfg is None:
            # Clear any prior state — a reload that turns the feature
            # off must drop the old gauge rows.
            self._exporter.replace_service_health_scores([])
            return
        quality_points: list[QualityPoint] = []
        for group_name in (g.name for g in self._config.groups):
            snap = self._exporter.snapshot_for(group_name)
            if snap is None:
                continue
            quality_points.extend(snap.quality_points)
        scores = health_mod.compute_service_health_scores(quality_points, cfg)
        self._exporter.replace_service_health_scores(
            [ServiceHealthScorePoint(labels=dict(s.labels), score=s.score) for s in scores]
        )

    async def _refresh_resource_sizing_recommendations(self) -> None:
        """Recompute right-sizing recommendations from current snapshots.

        Cheap per call: bounded by the operator-configured workload
        count (typically a small number). For each workload:

        1. Pull the forecast snapshot whose query id matches
           ``forecast_query`` (walking every group's snapshot — the
           workload doesn't bind to a specific group).
        2. Optionally fetch ``current_promql`` via the source.
        3. Delegate the math to
           :func:`right_sizing.compute_recommendation` which handles
           the quality gate.
        4. Replace the exporter's full recommendation map.

        Empty / disabled config short-circuits at the top — the
        ``replace_resource_sizing_recommendations([])`` call there
        clears any stale rows from a prior config that had this on.
        """
        cfg = self._config.recommendations.resource_sizing
        if not cfg.enabled or not cfg.workloads:
            self._exporter.replace_resource_sizing_recommendations([])
            return
        # Build a flat lookup of forecast / quality points keyed by
        # query id so we don't re-walk every group per workload.
        forecast_by_id: dict[str, list[ForecastPoint]] = {}
        quality_by_id: dict[str, list[QualityPoint]] = {}
        for group in self._config.groups:
            snap = self._exporter.snapshot_for(group.name)
            if snap is None:
                continue
            for point in snap.points:
                qid = _query_id_from_metric(point.metric)
                if qid:
                    forecast_by_id.setdefault(qid, []).append(point)
            for qp in snap.quality_points:
                qid = qp.labels.get("id", "")
                if qid:
                    quality_by_id.setdefault(qid, []).append(qp)
        out_points: list[ResourceSizingPoint] = []
        for workload in cfg.workloads:
            forecast_points = forecast_by_id.get(workload.forecast_query, [])
            quality_points = quality_by_id.get(workload.forecast_query, [])
            current_value = await self._fetch_current_for_workload(workload)
            rec = right_sizing_mod.compute_recommendation(
                workload=workload,
                config=cfg,
                forecast_points=forecast_points,
                quality_points=quality_points,
                current=current_value,
            )
            if rec is None:
                continue
            out_points.append(
                ResourceSizingPoint(
                    labels={
                        "workload": rec.workload,
                        "kind": rec.kind,
                        "confidence": rec.confidence,
                    },
                    recommended=rec.recommended,
                    current=rec.current,
                    savings_pct=rec.savings_pct,
                )
            )
        self._exporter.replace_resource_sizing_recommendations(out_points)

    async def _fetch_current_for_workload(
        self,
        workload: ResourceSizingWorkloadConfig,
    ) -> float | None:
        """Resolve ``current_promql`` to a single float, or ``None`` on failure.

        A workload without ``current_promql`` skips the fetch entirely.
        On error (transport failure, malformed PromQL, no data) we
        return ``None`` so the recommendation still emits without the
        sibling ``current`` / ``savings_pct`` rows — better than
        crashing the whole refresh on one workload's flaky query.

        The lookback window is intentionally narrow: 10 minutes at
        ``defaults.step`` gives us enough samples to pick the most
        recent finite reading without dragging in stale data.
        """
        if workload.current_promql is None:
            return None
        end = datetime.now(tz=UTC)
        step = self._config.defaults.step
        start = end - timedelta(minutes=10)
        step_seconds = max(1, int(step.total_seconds()))
        try:
            frames = await self._source.query_range(
                workload.current_promql, start, end, step_seconds
            )
        except Exception:
            logger.exception(
                "resource_sizing_current_query_failed",
                workload=workload.workload,
                kind=workload.kind,
            )
            return None
        for frame in frames:
            for value in reversed(frame.values):
                if math.isfinite(value):
                    return float(value)
        return None

    def _build_deviation_lookup(self) -> deploy_risk_mod.DeviationLookup:
        """Return a callable that maps (service, query_id) to mean |ratio|.

        The exporter's per-group snapshots carry the deviation points;
        we collapse them across (model, level, series) per query id by
        taking the mean of the absolute ratios. Non-finite ratios are
        ignored (so a degenerate band doesn't poison the composite).
        """
        # Build the index eagerly so multiple lookups per call don't
        # re-iterate the same snapshot. Keyed by query id only —
        # ``service`` is ignored because the deviation labelset has no
        # service label; routing happens at the spec level (via
        # ``signals_by_service``) instead.
        by_query: dict[str, list[float]] = {}
        for group_name in (g.name for g in self._config.groups):
            snap = self._exporter.snapshot_for(group_name)
            if snap is None:
                continue
            for dp in snap.deviation_points:
                qid = dp.labels.get("id", "")
                if not qid or not math.isfinite(dp.ratio):
                    continue
                by_query.setdefault(qid, []).append(abs(dp.ratio))

        def lookup(service: str, query_id: str) -> float:
            samples = by_query.get(query_id)
            if not samples:
                return float("nan")
            return sum(samples) / len(samples)

        return lookup

    async def _run_query(
        self,
        query: QueryConfig,
        group: GroupConfig,
        ctx: _RunContext,
        log: structlog.stdlib.BoundLogger,
    ) -> None:
        qlog = log.bind(query=query.id)
        with self._tracer.span("promforecast.query", group=group.name, query=query.id):
            await self._do_run_query(query, group, ctx, qlog)

    async def _do_run_query(  # noqa: PLR0912, PLR0915
        self,
        query: QueryConfig,
        group: GroupConfig,
        ctx: _RunContext,
        qlog: structlog.stdlib.BoundLogger,
    ) -> None:
        # Seed the per-query series count so a query that fails before the
        # accept-loop still reports ``forecast_query_series_count=0`` on
        # this tick. Without this seed the gauge would be silently missing
        # for a failing query and dashboards would see a stale value
        # inherited from a prior successful run.
        ctx.series_count_per_query.setdefault(query.id, 0)
        # Event-profile queries take a separate code path: no forecast
        # curve, no per-step model fit. Instead the runner reads the
        # configured counter, derives per-(day_of_week, hour_of_day)
        # arrival intensities, and emits a heatmap-shaped
        # ``forecast_incident_recurrence_score`` family. The query never
        # touches statsforecast.
        if query.data_profile == "events":
            await self._run_event_pattern_query(query, group, ctx, qlog)
            return
        frames = await self._discover_and_fetch(query, group, ctx, qlog)
        if frames is None:
            return

        # Canary-vs-baseline deviation reads the *full* fetched population
        # (it needs both the canary and the baseline series), so it runs
        # before the group-level canary *sampling* and the overflow caps
        # shrink the set. It emits its own ``<id>_canary_forecast_deviation``
        # family and does not affect the normal forecast path below.
        if query.canary.enabled:
            await self._emit_canary_deviation(query, group, frames, ctx, qlog)

        # Canary sampling runs *before* the overflow caps so the operator's
        # opt-in population shrink happens first, and the per-query/global
        # caps then bound the sampled population. The inverse order — caps
        # first, then sample — would defeat the point of canary mode for a
        # group whose total cardinality is already below caps but whose
        # fit cost is the actual concern.
        frames, canary_dropped = _apply_canary(frames, group.canary)
        if canary_dropped:
            ctx.dropped["canary"] = ctx.dropped.get("canary", 0) + canary_dropped
            qlog.info(
                "canary_sampled",
                weight=group.canary.weight,
                sampled=len(frames),
                dropped=canary_dropped,
            )

        safety = self._config.safety
        global_cap_enabled = safety.max_total_series > 0
        effective_cap = safety.max_series_per_query
        cap_reason = "per_query_cap"
        if global_cap_enabled and ctx.global_remaining < effective_cap:
            effective_cap = ctx.global_remaining
            cap_reason = "total_cap"

        allowed, dropped_n = _apply_overflow(frames, effective_cap, safety.series_overflow)
        if dropped_n:
            ctx.dropped[cap_reason] = ctx.dropped.get(cap_reason, 0) + dropped_n
            qlog.info(
                "series_dropped",
                reason=cap_reason,
                dropped=dropped_n,
                returned=len(frames),
                allowed=len(allowed),
            )
        if global_cap_enabled:
            ctx.global_remaining = max(0, ctx.global_remaining - len(allowed))
        ctx.series_count += len(allowed)
        ctx.series_count_per_query[query.id] = ctx.series_count_per_query.get(query.id, 0) + len(
            allowed
        )

        # Prefetch every PromQL regressor *once* per primary query — not
        # per series. Without this split a query that returned 100 series
        # with two PromQL regressors would issue 200 redundant TSDB
        # requests every refresh interval; the prefetch caps it at 2.
        try:
            prefetch = await self._prefetch_query_regressors(query)
        except regressors_mod.RegressorError as exc:
            # A required regressor's PromQL failed at fetch time. Every
            # series of this query depended on it, so skip the whole
            # query rather than emitting N copies of the same failure.
            qlog.warning(
                "required_regressor_failed_query",
                reason=exc.reason,
                error=str(exc),
            )
            ctx.failures["required_regressor"] = ctx.failures.get("required_regressor", 0) + 1
            return
        # Optional PromQL fetch failures: counted *once* per query (not
        # per series) so flaky regressors don't multiply the counter by
        # the cardinality of the primary query.
        for failure in prefetch.failures:
            rkey = (query.id, failure.regressor_id, failure.reason)
            ctx.regressor_failures[rkey] = ctx.regressor_failures.get(rkey, 0) + 1

        # Self-observability: record the per-(group, query, regressor_id)
        # stability score derived from the aggregated PromQL values. One
        # pass per query (not per series) keeps the bookkeeping cheap;
        # calendar regressors are deterministic functions of timestamps so
        # they're skipped — a "stability" measurement on a deterministic
        # series isn't meaningful.
        if prefetch.aggregated:
            for regressor_id, agg_df in prefetch.aggregated.items():
                try:
                    values = agg_df["v"].tolist()
                except (KeyError, AttributeError):
                    continue
                self._exporter.record_regressor_stability(
                    group=group.name,
                    query=query.id,
                    regressor_id=regressor_id,
                    values=[float(v) for v in values],
                )

        min_points = self._config.defaults.min_points
        eff_step = effective_step(query, self._config.defaults)
        # Cold-start sibling cache: fetched lazily on the first short
        # series and reused for every subsequent short series of the
        # same query. The lazy strategy preserves the "one fetch per
        # query" guarantee from the per-series-fetch bug fix while
        # also avoiding a wasted borrow PromQL on the common case
        # where no series of the query falls below ``min_points``
        # (e.g. a query with cold-start configured "just in case"
        # against a fully matured fleet). ``cold_start_siblings`` is
        # ``None`` either before resolution OR after resolution when
        # cold-start is ineligible / the fetch failed; the
        # ``cold_start_resolved`` flag distinguishes the two.
        cold_start_siblings: list[SeriesFrame] | None = None
        cold_start_resolved = False
        for frame in allowed:
            cleaned = preprocess_mod.preprocess_series(
                timestamps=list(frame.timestamps),
                values=list(frame.values),
                step=eff_step,
                config=group.preprocess,
            )
            if cleaned.dropped:
                ctx.dropped["preprocess_drop"] = ctx.dropped.get("preprocess_drop", 0) + 1
                # Still record the largest gap so operators see what got
                # dropped without scraping logs.
                query_stats = ctx.preprocess_stats.setdefault(query.id, PreprocessStats())
                query_stats.max_gap_seconds = max(
                    query_stats.max_gap_seconds, cleaned.max_gap_seconds
                )
                # Blacklist samples may have driven the drop (the empty-
                # series short-circuit when the entire window is
                # blacklisted); preserve the per-reason counter so the
                # operator can see *why* the series dropped.
                for reason, count in cleaned.blacklisted_samples.items():
                    bkey = (query.id, reason)
                    ctx.blacklisted_samples[bkey] = ctx.blacklisted_samples.get(bkey, 0) + count
                continue
            self._merge_preprocess_stats(
                ctx=ctx,
                query_id=query.id,
                cleaned=cleaned,
                outlier_method=group.preprocess.outliers.method,
            )
            preprocessed_frame = SeriesFrame(
                labels=frame.labels,
                timestamps=cleaned.timestamps,
                values=cleaned.values,
            )
            if len(cleaned.values) < min_points:
                if not cold_start_resolved:
                    cold_start_siblings = await self._prefetch_cold_start_siblings(query, qlog)
                    cold_start_resolved = True
                if cold_start_siblings is not None:
                    handled = await self._try_cold_start(
                        query=query,
                        group=group,
                        frame=preprocessed_frame,
                        ctx=ctx,
                        qlog=qlog,
                        sibling_frames=cold_start_siblings,
                    )
                    if handled:
                        continue
                ctx.dropped["min_points"] = ctx.dropped.get("min_points", 0) + 1
                continue
            stop_query = await self._run_one_series(
                query,
                group,
                preprocessed_frame,
                ctx,
                qlog,
                prefetch,
                imputed_fraction=cleaned.imputed_fraction,
            )
            if stop_query:
                return

    async def _prefetch_query_regressors(self, query: QueryConfig) -> regressors_mod.PromQLPrefetch:
        """Run every custom regressor's PromQL once per primary query.

        Regressors share the primary query's step so their values align
        cleanly with the series time index — a regressor sampled at a
        different cadence would either alias under merge_asof or force a
        second alignment pass for no real benefit.
        """
        d = self._config.defaults
        if not query.regressors:
            return regressors_mod.PromQLPrefetch(fetched={}, failures=[])
        end = datetime.now(tz=UTC)
        start = end - d.lookback
        step_seconds = max(1, int(effective_step(query, d).total_seconds()))
        return await regressors_mod.prefetch_promql(
            regressors=query.regressors,
            source=self._source,
            query_start=start,
            query_end=end,
            step_seconds=step_seconds,
            query_timeout_seconds=self._config.safety.query_timeout.total_seconds(),
            query_id=query.id,
            webhook_client=self._webhook_client,
        )

    def _merge_preprocess_stats(
        self,
        *,
        ctx: _RunContext,
        query_id: str,
        cleaned: preprocess_mod.PreprocessResult,
        outlier_method: str,
    ) -> None:
        """Fold one series' preprocessing stats into the run context.

        Counters (gaps, outliers, change-points) accumulate across the
        series of one query; gauges (max gap, effective lookback) keep
        the largest / most recent value seen this run so the exposed
        per-(group, query) sample matches what at least one series
        actually went through. ``outlier_method`` is read off the group's
        config so the per-method counter label matches the YAML the
        operator wrote.
        """
        stats = ctx.preprocess_stats.setdefault(query_id, PreprocessStats())
        for strategy, count in cleaned.gaps_filled.items():
            stats.gaps_filled[strategy] = stats.gaps_filled.get(strategy, 0) + count
            key = (query_id, strategy)
            ctx.gaps_filled[key] = ctx.gaps_filled.get(key, 0) + count
        stats.max_gap_seconds = max(stats.max_gap_seconds, cleaned.max_gap_seconds)
        if cleaned.outliers_replaced:
            stats.outliers_replaced[outlier_method] = (
                stats.outliers_replaced.get(outlier_method, 0) + cleaned.outliers_replaced
            )
            okey = (query_id, outlier_method)
            ctx.outliers_replaced[okey] = (
                ctx.outliers_replaced.get(okey, 0) + cleaned.outliers_replaced
            )
        if cleaned.change_points:
            stats.change_points += cleaned.change_points
            ctx.change_points[query_id] = ctx.change_points.get(query_id, 0) + cleaned.change_points
        for reason, count in cleaned.blacklisted_samples.items():
            bkey = (query_id, reason)
            ctx.blacklisted_samples[bkey] = ctx.blacklisted_samples.get(bkey, 0) + count
        if cleaned.effective_lookback_seconds > 0:
            stats.effective_lookback_seconds = cleaned.effective_lookback_seconds

    async def _run_one_series(
        self,
        query: QueryConfig,
        group: GroupConfig,
        frame: SeriesFrame,
        ctx: _RunContext,
        qlog: structlog.stdlib.BoundLogger,
        prefetch: regressors_mod.PromQLPrefetch,
        imputed_fraction: float = 0.0,
    ) -> bool:
        """Fit one series and merge its results into ``ctx``.

        Returns ``True`` when the caller must abandon the rest of the query
        (currently only the ``invalid_metric_name`` case, which is a
        config-level bug — once one series trips it, the rest will too).
        Returns ``False`` for per-series failures that should not stop the
        loop.
        """
        try:
            fit_result = await self._fit_series(
                query, frame, group, prefetch, imputed_fraction=imputed_fraction
            )
        except InvalidMetricNameError:
            qlog.exception("invalid_metric_name")
            ctx.failures["invalid_metric_name"] = ctx.failures.get("invalid_metric_name", 0) + 1
            return True
        except regressors_mod.RegressorError as exc:
            # A required *calendar* regressor failed during alignment
            # (PromQL ones were caught at prefetch time). Skip this series
            # only — calendar failures are deterministic per timestamp set
            # but we don't assume the rest of the query has the same one.
            qlog.warning(
                "required_regressor_failed",
                labels=frame.labels,
                reason=exc.reason,
                error=str(exc),
            )
            ctx.failures["required_regressor"] = ctx.failures.get("required_regressor", 0) + 1
            return False
        except TimeoutError:
            # ``safety.fit_timeout`` exceeded — distinct from a generic
            # fit error so dashboards can split "model crashed" from
            # "model is too slow" without log scraping.
            qlog.error(
                "fit_timed_out",
                labels=frame.labels,
                timeout_s=self._config.safety.fit_timeout.total_seconds(),
            )
            ctx.failures["fit_timeout"] = ctx.failures.get("fit_timeout", 0) + 1
            return False
        except Exception:
            qlog.exception("fit_failed", labels=frame.labels)
            ctx.failures["fit_error"] = ctx.failures.get("fit_error", 0) + 1
            return False
        _merge_fit_into_ctx(ctx, fit_result, query.id)
        self._record_series_fit_successes(
            fit_result=fit_result,
            group_name=group.name,
            query_id=query.id,
            frame=frame,
        )
        return False

    def _record_series_fit_successes(
        self,
        *,
        fit_result: _FitResult,
        group_name: str,
        query_id: str,
        frame: SeriesFrame,
    ) -> None:
        """Stamp ``forecast_data_staleness_seconds`` for every emitted model.

        Iterates ``fit_result.points`` to extract the unique set of
        models that produced a snapshot row for this series — that's
        the precise success signal we want the staleness clock to
        track. Models present in ``fit_result.fit_durations`` but
        absent from ``fit_result.points`` (e.g. a backtest-only fit)
        intentionally don't reset the clock; the operator-visible
        signal is "is this series being forecast", not "did any
        fit run".

        The conditional refit family (model names suffixed
        ``/conditional``) is filtered out — staleness tracks the
        underlying model's emission cadence, and the conditional
        emission rides on the same fit success.

        Called from both :meth:`_run_one_series` (the normal path)
        and :meth:`_try_cold_start` (the cold-start path) so
        synthesised cold-start forecasts reset the clock too.
        """
        if not fit_result.points:
            return
        models: set[str] = set()
        for point in fit_result.points:
            model = point.labels.get("model", "")
            if not model or model.endswith("/conditional"):
                continue
            models.add(model)
        if not models:
            return
        series_key = _series_key_from_labels(frame.labels)
        source_labels = {k: v for k, v in frame.labels.items() if k != "__name__"}
        for model_name in models:
            labels = {
                **source_labels,
                "id": query_id,
                "group": group_name,
                "model": model_name,
            }
            self._exporter.record_series_fit_success(
                group=group_name,
                query_id=query_id,
                model_name=model_name,
                series_key=series_key,
                labels=labels,
            )

    def _eligible_for_cold_start(self, query: QueryConfig) -> tuple[bool, str | None]:
        """Decide whether cold-start can run for ``query`` at all.

        Returns ``(eligible, reason)`` where ``reason`` is a short token
        describing why cold-start is disabled (``None`` when eligible).
        The check intentionally short-circuits two contracts the
        SeasonalNaive-style cold-start path cannot honor:

        * **Required regressors**: ``regressors[].required: true`` is the
          operator saying "this metric must not be forecast without this
          exogenous input"; synthesising a forecast from peer shapes
          ignores that input entirely. Better to fall through to the
          standard ``min_points`` drop than to silently produce a
          regressor-free forecast for a series whose contract demanded
          one.
        * **Intermittent profile**: median-normalising a sparse, zero-
          heavy series produces a degenerate prior (the per-step median
          collapses to zero where most siblings sit on the zero
          baseline). Skipping for ``data_profile: intermittent`` keeps
          a useless forecast off /metrics; a future Croston-aware
          cold-start variant can lift this skip when it ships.
        """
        if query.cold_start.borrow_from is None:
            return False, None
        if any(r.required for r in query.regressors):
            return False, "required_regressor"
        if query.data_profile == "intermittent":
            return False, "intermittent_profile"
        return True, None

    async def _prefetch_cold_start_siblings(
        self,
        query: QueryConfig,
        qlog: structlog.stdlib.BoundLogger,
    ) -> list[SeriesFrame] | None:
        """Fetch the borrow query once per primary query.

        Returns ``None`` when cold-start would refuse to run at all
        (no ``borrow_from`` configured, intermittent profile, required
        regressor declared) or when the borrow PromQL fails — in either
        case the per-frame loop falls through to the standard
        ``min_points`` drop.

        The sibling list is bounded by ``safety.max_series_per_query``
        / ``safety.series_overflow`` so a misconfigured borrow PromQL
        (``up`` instead of ``up{job="peer"}``) cannot load the entire
        TSDB cardinality into memory.
        """
        eligible, reason = self._eligible_for_cold_start(query)
        if not eligible:
            if reason is not None:
                qlog.info("cold_start_disabled", reason=reason)
            return None
        cold = query.cold_start
        borrow = cold.borrow_from
        assert borrow is not None  # narrowed by _eligible_for_cold_start
        d = self._config.defaults
        eff_step = effective_step(query, d)
        try:
            frames = await cold_start_mod.fetch_siblings(
                source=self._source,
                promql=borrow.promql,
                lookback=d.lookback,
                step=eff_step,
                query_timeout_seconds=self._config.safety.query_timeout.total_seconds(),
            )
        except TimeoutError:
            qlog.warning("cold_start_borrow_timeout")
            return None
        except Exception:
            qlog.warning("cold_start_borrow_failed", exc_info=True)
            return None

        safety = self._config.safety
        capped, dropped = _apply_overflow(
            frames, safety.max_series_per_query, safety.series_overflow
        )
        if dropped:
            qlog.info(
                "cold_start_siblings_capped",
                returned=len(frames),
                allowed=len(capped),
                dropped=dropped,
            )
        return capped

    async def _try_cold_start(
        self,
        *,
        query: QueryConfig,
        group: GroupConfig,
        frame: SeriesFrame,
        ctx: _RunContext,
        qlog: structlog.stdlib.BoundLogger,
        sibling_frames: list[SeriesFrame],
    ) -> bool:
        """Attempt a cold-start fit when a series is below ``min_points``.

        ``sibling_frames`` was prefetched once per query — see
        :meth:`_prefetch_cold_start_siblings`. We do not re-issue the
        borrow PromQL here, so cost is bounded to one TSDB round trip
        per (group, query) refresh regardless of how many of the
        query's series fall below ``min_points``.

        Emits a SeasonalNaive-style forecast carrying the
        ``cold_start="true"`` and ``model="ColdStart"`` labels. Returns
        ``True`` when a cold-start forecast was emitted, ``False`` when
        the caller should fall through to the normal ``min_points``
        drop (insufficient siblings, degenerate shape, etc.).

        ``forecast_quality_score`` for cold-start fits is capped at the
        configured ``max_quality_score`` so they don't pass strict alert
        gates like ``forecast_quality_score > 0.6`` on fabricated data.

        Cold-start fits skip backtest / deviation / calibration / capacity
        ETAs / growth rates by design: the underlying borrowed shape is
        a prior, not a fit; running backtests against a borrowed series
        would just amplify noise.
        """
        cold = query.cold_start
        borrow = cold.borrow_from
        if borrow is None:
            return False

        d = self._config.defaults
        eff_step = effective_step(query, d)
        eff_season_length = effective_season_length(query, d, step=eff_step)
        horizon = max(1, int(d.horizon.total_seconds() // eff_step.total_seconds()))

        # The forecast anchors at the target series's last observed
        # timestamp so the cold-start curve continues where the (short)
        # actuals left off — same convention the model-fit path follows.
        # For a newly-onboarded service this is "moments ago"; for a
        # series that stopped reporting hours ago the curve starts in the
        # past, which is consistent with how the regular path treats stale
        # input.
        last_ts = frame.timestamps[-1] if frame.timestamps else None
        outcome = cold_start_mod.synthesise_cold_start_forecast(
            target_values=frame.values,
            sibling_frames=sibling_frames,
            season_length=eff_season_length,
            horizon=horizon,
            step=eff_step,
            min_siblings=borrow.min_siblings,
            last_ts=last_ts,
        )
        if outcome is None:
            return False

        # Synthesise a minimal ``_FitResult`` directly so the caller's
        # standard ``_merge_fit_into_ctx`` plumbing stays unchanged. The
        # cold-start forecast carries no backtest data, so we skip
        # accuracy / deviation / capacity / contribution emission and
        # only produce forecast points + (optionally) a capped quality
        # sample.
        factory = SeriesPointFactory(
            metric_id=query.id,
            base_labels=frame.labels,
            group=group.name,
            levels=list(d.confidence_levels),
        )
        fit_result = _FitResult()
        forecast_points = [
            ForecastPoint(
                metric=point.metric,
                labels={**point.labels, "cold_start": "true"},
                value=point.value,
            )
            for point in factory.forecast_points(
                outcome.forecast, cold_start_mod.COLD_START_MODEL_NAME, selection=None
            )
        ]
        # Mirror snapshot points into the remote_write curve so a TSDB-
        # backed deployment sees the same cold-start curve as the
        # ``/metrics`` scrape. Without this, an operator who enabled the
        # ``sink.remote_write`` push would see a snapshot row but no
        # historical curve for cold-start series — a silent gap.
        forecast_curve = [
            ForecastCurvePoint(
                metric=point.metric,
                labels={**point.labels, "cold_start": "true"},
                value=point.value,
                timestamp_ms=point.timestamp_ms,
            )
            for point in factory.forecast_curve_points(
                outcome.forecast, cold_start_mod.COLD_START_MODEL_NAME, selection=None
            )
        ]

        # Apply count-aware emission safeguards to cold-start forecasts
        # as well so a count-shaped series doesn't suddenly
        # emit negative or fractional values just because its first
        # forecast came from a borrowed shape. Snapshot pass owns the
        # counter — the curve pass mutates values only — so we don't
        # double-count corrections across the two emission paths.
        clip, rounder = effective_emission_safeguards(query)
        if clip or rounder:
            counter: dict[str, int] = {}
            forecast_points = _apply_safeguards_to_snapshot_points(
                forecast_points, clip=clip, rounder=rounder, counter=counter
            )
            forecast_curve = _apply_safeguards_to_curve_points(
                forecast_curve, clip=clip, rounder=rounder
            )
            for kind, count in counter.items():
                fit_result.emission_clipped[kind] = fit_result.emission_clipped.get(kind, 0) + count

        fit_result.points = forecast_points
        fit_result.curve = forecast_curve
        fit_result.cold_start_siblings = outcome.sibling_count

        if d.emission.quality:
            # Quality is normally derived from backtest data. Cold-start
            # forecasts have no backtest, so we synthesise the score from
            # two cold-start-specific signals — how many siblings
            # contributed and how tightly they agreed on the shape —
            # then clamp at ``max_quality_score`` so the score still
            # cannot pass strict alert gates regardless of how
            # confident the sibling set looked.
            quality_label = {
                **{k: v for k, v in frame.labels.items() if k != "__name__"},
                "id": query.id,
                "group": group.name,
                "model": cold_start_mod.COLD_START_MODEL_NAME,
                "cold_start": "true",
                # Cold-start has no backtest, so we can't split the
                # quality score by horizon — emit the aggregated row
                # only (``horizon=""``) for label-shape consistency
                # with the steady-state quality emission family.
                "horizon": "",
            }
            quality_score = cold_start_mod.cold_start_quality_score(
                sibling_count=outcome.sibling_count,
                shape_dispersion=outcome.shape_dispersion,
                min_siblings=borrow.min_siblings,
                cap=cold.max_quality_score,
            )
            fit_result.quality.append(QualityPoint(labels=quality_label, score=quality_score))

        _merge_fit_into_ctx(ctx, fit_result, query.id)
        self._record_series_fit_successes(
            fit_result=fit_result,
            group_name=group.name,
            query_id=query.id,
            frame=frame,
        )
        qlog.info(
            "cold_start_fit_emitted",
            siblings=outcome.sibling_count,
            min_siblings=borrow.min_siblings,
            horizon=horizon,
        )
        return True

    async def _run_event_pattern_query(
        self,
        query: QueryConfig,
        group: GroupConfig,
        ctx: _RunContext,
        qlog: structlog.stdlib.BoundLogger,
    ) -> None:
        """Run the recurring-incident-pattern path for a ``data_profile: events`` query.

        Fetches the configured counter over the standard lookback,
        sums the per-step deltas across any returned series (when the
        query fans out across labels), buckets the events by
        (day_of_week, hour_of_day), and publishes the resulting 7x24
        grid to the exporter as ``forecast_incident_recurrence_score``.

        Failures isolate per-query — a query that fails here doesn't
        affect the rest of the group; ``forecast_failures_total{reason="query_error"}``
        is bumped instead.

        No backtest, no model fit, no per-step forecast curve — the
        event path is entirely separate from the continuous-forecast
        pipeline. The query never lands rows in the ``points``,
        ``deviation_points``, etc. accumulators on the context.
        """
        eff_step = effective_step(query, self._config.defaults)
        step_seconds = max(1, int(eff_step.total_seconds()))
        end = datetime.now(tz=UTC)
        start = end - self._config.defaults.lookback
        try:
            frames = await self._source.query_range(query.promql, start, end, step_seconds)
        except Exception:
            qlog.exception("event_query_failed")
            ctx.failures["query_error"] = ctx.failures.get("query_error", 0) + 1
            return
        if not frames:
            qlog.info("event_query_returned_no_series")
            self._exporter.replace_incident_recurrence_for_query(
                group=group.name, query=query.id, points=[]
            )
            return
        # Sum across every returned series: incident counters are
        # often fanned out per (severity, team, service) and the
        # operator wants the aggregate heatmap, not per-series ones.
        # Per-series heatmaps via a label passthrough are a
        # documented follow-up; today the shipped shape is
        # aggregate-only.
        timestamps, values = _sum_counter_series(frames)
        buckets = incident_pattern_mod.compute_recurrence(timestamps=timestamps, values=values)
        # Label set mirrors the other forecast families: ``id`` is the
        # query id (matching every other ``<id>_forecast*`` family in
        # this repo and the sibling promanomaly project so dashboards
        # can ``and on (id, group)`` across both), ``group`` the group
        # name. Bucket coordinates ride on (day_of_week, hour_of_day).
        points = [
            IncidentRecurrencePoint(
                labels={
                    "group": group.name,
                    "id": query.id,
                    "day_of_week": str(bucket.day_of_week),
                    "hour_of_day": str(bucket.hour_of_day),
                },
                score=bucket.score,
            )
            for bucket in buckets
        ]
        self._exporter.replace_incident_recurrence_for_query(
            group=group.name, query=query.id, points=points
        )
        # Charge the 168 emitted heatmap rows against the global
        # series budget so an operator with many event queries can't
        # silently blow ``safety.max_total_series``. The series-count
        # gauges report the source-side count (1 — the aggregate),
        # so we keep the gauge value at 1 and apply the multiplier
        # only to the budget. See validate.py for the matching
        # cardinality-estimator multiplier.
        budget_charge = _INCIDENT_RECURRENCE_ROWS
        ctx.series_count += 1
        ctx.series_count_per_query[query.id] = 1
        if ctx.global_remaining > 0:
            ctx.global_remaining = max(0, ctx.global_remaining - budget_charge)
        qlog.info(
            "event_pattern_emitted",
            buckets=len(points),
            non_zero=sum(1 for p in points if p.score > 0),
            frames=len(frames),
            budget_charged=budget_charge,
        )

    async def _discover_and_fetch(
        self,
        query: QueryConfig,
        group: GroupConfig,
        ctx: _RunContext,
        qlog: structlog.stdlib.BoundLogger,
    ) -> list[SeriesFrame] | None:
        """Expand any template variables and fetch every rendered query.

        Returns the combined ``SeriesFrame`` list across all expansions on
        success, or ``None`` when the caller should abandon the query (no
        variables resolved, discovery itself blew up). All discovery and
        fetch failures are recorded on ``ctx`` / the exporter so callers
        don't need to know the details.
        """
        try:
            expansion = await self._expand_query(query)
        except Exception:
            qlog.exception("discovery_failed")
            ctx.failures["discovery_error"] = ctx.failures.get("discovery_error", 0) + 1
            self._exporter.record_discovery(group=group.name, query=query.id, expansions=0)
            return None

        discovery_failure_counts: dict[tuple[str, str], int] = {}
        for failure in expansion.failures:
            key = (failure.variable, failure.reason)
            discovery_failure_counts[key] = discovery_failure_counts.get(key, 0) + 1
        self._exporter.record_discovery(
            group=group.name,
            query=query.id,
            expansions=len(expansion.rendered),
            failures=discovery_failure_counts,
        )
        if expansion.dropped:
            ctx.dropped["discovery_cap"] = ctx.dropped.get("discovery_cap", 0) + expansion.dropped
        if not expansion.rendered:
            if expansion.failures:
                qlog.warning(
                    "discovery_no_values",
                    failures=[
                        {"variable": f.variable, "reason": f.reason} for f in expansion.failures
                    ],
                )
                ctx.failures["discovery_empty"] = ctx.failures.get("discovery_empty", 0) + 1
            else:
                qlog.info("discovery_no_values")
            return None

        # Fan out the expanded fetches concurrently. Wall time becomes
        # ``max(per-fetch)`` instead of ``sum(per-fetch)`` — at the typical
        # 100ms per-fetch latency this is the difference between a 10s and
        # a 200ms refresh for a 100-series template. ``PromSource`` already
        # caps total concurrency via its internal semaphore, so this fan-out
        # cannot accidentally hammer the TSDB.
        outcomes = await asyncio.gather(
            *(
                self._fetch_promql(query, rendered_promql)
                for rendered_promql, _ in expansion.rendered
            ),
            return_exceptions=True,
        )
        frames: list[SeriesFrame] = []
        for (rendered_promql, _values), outcome in zip(expansion.rendered, outcomes, strict=True):
            if isinstance(outcome, TimeoutError):
                qlog.error(
                    "query_timed_out",
                    promql=rendered_promql,
                    timeout_s=self._config.safety.query_timeout.total_seconds(),
                )
                ctx.failures["query_timeout"] = ctx.failures.get("query_timeout", 0) + 1
                continue
            if isinstance(outcome, BaseException):
                qlog.warning(
                    "query_failed",
                    promql=rendered_promql,
                    error=str(outcome),
                )
                ctx.failures["query_error"] = ctx.failures.get("query_error", 0) + 1
                continue
            frames.extend(outcome)
        return frames

    async def _fetch_promql(self, query: QueryConfig, promql: str) -> list[SeriesFrame]:
        """Run a single PromQL string against the configured datasource.

        Split out from the original ``_fetch(query)`` so the discovery
        expansion path can issue one fetch per rendered query while still
        going through the same retry / timeout / cache wiring. The original
        :meth:`_fetch` is kept as a thin compatibility wrapper for callers
        that don't care about the rendered string (and for tests).
        """
        d = self._config.defaults
        s = self._config.safety
        end = datetime.now(tz=UTC)
        start = end - d.lookback
        step_seconds = max(1, int(effective_step(query, d).total_seconds()))
        return await asyncio.wait_for(
            self._source.query_range(promql, start, end, step_seconds),
            timeout=s.query_timeout.total_seconds(),
        )

    async def _fetch(self, query: QueryConfig) -> list[SeriesFrame]:
        return await self._fetch_promql(query, query.promql)

    async def _expand_query(self, query: QueryConfig) -> discovery_mod.ExpansionResult:
        """Resolve discovery variables (or pass through when no template)."""
        if not query.discover:
            return discovery_mod.ExpansionResult(rendered=[(query.promql, {})], failures=[])
        d = self._config.defaults
        s = self._config.safety
        step_seconds = max(1, int(effective_step(query, d).total_seconds()))
        return await discovery_mod.expand_query(
            query=query,
            source=self._source,
            lookback=d.lookback,
            step_seconds=step_seconds,
            query_timeout_seconds=s.query_timeout.total_seconds(),
            max_series_per_query=s.max_series_per_query,
            overflow_strategy=s.series_overflow,
        )

    async def _fit_series(  # noqa: PLR0912, PLR0915
        self,
        query: QueryConfig,
        frame: SeriesFrame,
        group: GroupConfig,
        prefetch: regressors_mod.PromQLPrefetch,
        imputed_fraction: float = 0.0,
    ) -> _FitResult:
        import pandas as pd  # noqa: PLC0415

        d = self._config.defaults
        group_name = group.name
        validate_metric_name(query.id)

        df = pd.DataFrame(
            {
                "unique_id": ["s"] * len(frame.values),
                "ds": pd.to_datetime(frame.timestamps),
                "y": frame.values,
            }
        )
        # Per-query overrides cascade onto the cached defaults: step controls
        # both the pandas freq string and how many forecast steps live inside
        # the configured horizon, season_length controls model seasonality.
        eff_step = effective_step(query, d)
        eff_season_length = self._resolve_season_length(query, group_name, frame, eff_step)
        freq = _pandas_freq(eff_step)
        horizon = max(1, int(d.horizon.total_seconds() // eff_step.total_seconds()))
        levels = list(d.confidence_levels)
        factory = SeriesPointFactory(
            metric_id=query.id,
            base_labels=frame.labels,
            group=group_name,
            levels=levels,
        )

        result = _FitResult()

        # Align pre-fetched PromQL + apply calendar features for *this*
        # series. ``prefetch`` is shared across every series of the
        # primary query, so this is a cheap dataframe operation; the
        # expensive PromQL fan-out already happened once in ``_run_query``.
        regressors_frame = self._align_regressors_for_series(
            query=query, frame=frame, result=result, prefetch=prefetch
        )

        accuracy_horizons = horizons_from_durations(list(d.accuracy.horizons), eff_step)
        auto_select_enabled = bool(d.accuracy.auto_select)
        needs = _BacktestNeeds(
            accuracy=d.accuracy.evaluate,
            selection=auto_select_enabled,
            deviation=d.emission.deviation,
            quality=d.emission.quality,
        )

        model_specs = list(d.models)
        backtests: dict[str, BacktestOutcome] = {}
        if needs.any and accuracy_horizons:
            backtests = await self._run_backtests(
                df=df,
                models=model_specs,
                horizons=accuracy_horizons,
                freq=freq,
                levels=levels,
                season_length=eff_season_length,
                group_name=group_name,
                query_id=query.id,
                needs=needs,
                regressors=regressors_frame,
            )

        backtest_results: dict[str, list[AccuracyResult]] = {
            name: outcome.results for name, outcome in backtests.items()
        }

        is_ensemble = d.accuracy.auto_select == "ensemble"
        if auto_select_enabled and not is_ensemble:
            forecast_points = await self._auto_select_and_forecast(
                df=df,
                query=query,
                factory=factory,
                group_name=group_name,
                horizon=horizon,
                freq=freq,
                levels=levels,
                season_length=eff_season_length,
                backtest_results=backtest_results,
                result=result,
                regressors=regressors_frame,
            )
        elif is_ensemble:
            forecast_points = await self._ensemble_forecast(
                df=df,
                models=model_specs,
                query=query,
                factory=factory,
                group_name=group_name,
                horizon=horizon,
                freq=freq,
                levels=levels,
                season_length=eff_season_length,
                backtest_results=backtest_results,
                result=result,
                regressors=regressors_frame,
            )
        else:
            forecast_points = await self._explicit_forecast(
                df=df,
                models=model_specs,
                query=query,
                factory=factory,
                group_name=group_name,
                horizon=horizon,
                freq=freq,
                levels=levels,
                season_length=eff_season_length,
                result=result,
                regressors=regressors_frame,
            )

        result.imputed_fraction = imputed_fraction
        offsets_by_model: dict[str, dict[int, float]] = {}
        if d.accuracy.conformal.enabled and forecast_points:
            forecast_points, offsets_by_model = await self._apply_conformal_calibration(
                df=df,
                forecast_points=forecast_points,
                model_specs=model_specs,
                freq=freq,
                levels=levels,
                season_length=eff_season_length,
                regressors=regressors_frame,
                holdout_fraction=d.accuracy.conformal.holdout_fraction,
                result=result,
                factory=factory,
            )

        # Capacity-planning emission runs *after* calibration so band
        # ETAs reflect the calibrated bands the scrape will expose.
        # Without this ordering, an alert on ``time_to_threshold{level=N}``
        # would disagree with the corresponding ``yhat_upper`` /
        # ``yhat_lower`` at the same level.
        for ci in result.capacity_inputs:
            offsets = offsets_by_model.get(ci.model_name) if ci.apply_calibration else None
            self._emit_capacity_metrics(
                query=query,
                forecast=ci.forecast,
                model_name=ci.model_name,
                levels=levels,
                factory=factory,
                selection=ci.selection,
                result=result,
                offsets=offsets,
            )
        # Component decomposition emission. Pure post-processing on the
        # already-computed forecast frame, so it costs no extra model
        # fits. Runs on every emitted (model, series) when
        # ``emission.decompose`` is on; unsupported models log a skip
        # counter rather than emit garbage.
        if query.emission.decompose:
            for ci in result.capacity_inputs:
                self._emit_decomposition(
                    group_name=group_name,
                    query_id=query.id,
                    forecast=ci.forecast,
                    model_name=ci.model_name,
                    selection=ci.selection,
                    factory=factory,
                    in_sample_values=list(frame.values),
                    season_length=eff_season_length,
                    result=result,
                )

        # Conditional forecast emission. Re-fits each emitted model with
        # a regressors frame whose values are flat-held at the most
        # recent observation — operators alert on the divergence from the
        # *conditional* baseline ("is the actual unusual given current
        # regressors") rather than the unconditional band. For built-in
        # models that ignore the regressors parameter the conditional
        # output equals the unconditional output; the family still emits
        # so downstream alerts have a stable schema. When conformal
        # calibration is enabled, a fresh per-fit calibration on a
        # held-constant-regressor holdout produces dedicated conditional
        # offsets so the band carries its own empirical-coverage
        # guarantee (rather than inheriting the unconditional pass's
        # offsets, which scored residuals on a holdout where regressors
        # *did* vary). Per-(model) failures fall back to inherited
        # offsets and bump ``forecast_conditional_calibration_failures_total``.
        if query.emission.conditional:
            cond_clip, cond_rounder = effective_emission_safeguards(query)
            conditional_offsets_by_model: dict[str, dict[int, float]] = {}
            if d.accuracy.conformal.enabled and result.capacity_inputs:
                conditional_offsets_by_model = await self._compute_conditional_conformal_offsets(
                    df=df,
                    group_name=group_name,
                    query_id=query.id,
                    emitted_inputs=list(result.capacity_inputs),
                    model_specs=model_specs,
                    freq=freq,
                    levels=levels,
                    season_length=eff_season_length,
                    regressors=regressors_frame,
                    holdout_fraction=d.accuracy.conformal.holdout_fraction,
                )
            await self._emit_conditional_forecasts(
                df=df,
                query=query,
                group_name=group_name,
                model_specs=model_specs,
                emitted_inputs=list(result.capacity_inputs),
                actual=float(frame.values[-1]) if frame.values else float("nan"),
                horizon=horizon,
                freq=freq,
                levels=levels,
                season_length=eff_season_length,
                regressors=regressors_frame,
                factory=factory,
                result=result,
                offsets_by_model=offsets_by_model,
                conditional_offsets_by_model=conditional_offsets_by_model,
                clip=cond_clip,
                rounder=cond_rounder,
            )

        # Forecast-to-forecast revision and model-disagreement gauges.
        # Both need the per-emitted-model forecast frames + the accuracy
        # horizons, so they ride next to ``capacity_inputs`` and must
        # land before the next line clears that buffer.
        self._record_revision_and_disagreement(
            query=query,
            group_name=group_name,
            frame=frame,
            inputs=list(result.capacity_inputs),
            horizon_specs=accuracy_horizons,
            auto_select=auto_select_enabled,
            result=result,
        )

        # Free the retained per-model dataframes — they can be MB-scale,
        # and ``_RunContext`` retains them across the per-series loop.
        result.capacity_inputs.clear()

        # Count-aware emission safeguards. Applied *after* the
        # calibration widening so a band that widened into negative
        # territory still gets clipped. Run on both the snapshot and
        # the remote_write curve so scrape + push see the same numbers;
        # the counter only ticks on the snapshot pass (see
        # ``_apply_safeguards_to_curve_points``).
        clip, rounder = effective_emission_safeguards(query)
        if clip or rounder:
            counter: dict[str, int] = {}
            forecast_points = _apply_safeguards_to_snapshot_points(
                forecast_points, clip=clip, rounder=rounder, counter=counter
            )
            if result.curve:
                result.curve = _apply_safeguards_to_curve_points(
                    result.curve, clip=clip, rounder=rounder
                )
            for kind, count in counter.items():
                result.emission_clipped[kind] = result.emission_clipped.get(kind, 0) + count

        result.points = forecast_points

        if needs.accuracy and backtest_results:
            result.accuracy = list(factory.accuracy_points(backtest_results))

        # Restrict deviation/quality to the model(s) actually emitted as
        # forecasts. In auto-select mode that's the winner; in explicit mode
        # it's all configured models. ``ensemble`` is treated as another
        # emitted model — it doesn't have backtest data of its own, so it's
        # filtered out by the deviation/quality helpers (they look up
        # ``backtests[model_name]``).
        emitted_models = sorted({p.labels.get("model", "") for p in forecast_points})
        emitted_models = [m for m in emitted_models if m]

        if needs.deviation:
            result.deviation = list(factory.deviation_points(backtests, emitted_models))

        if needs.quality:
            expected_points = max(
                1,
                int(d.lookback.total_seconds() // max(1, int(eff_step.total_seconds()))),
            )
            coverage = min(1.0, len(frame.values) / float(expected_points))
            # ``data_profile: intermittent`` swaps MAPE for MASE in the
            # quality calculation as well — without this the quality
            # sub-score collapses to 0 for a series whose actuals are
            # mostly zero (NaN MAPE) and silently trips the
            # ``ForecastQualityZero`` alert even when the intermittent
            # model is fitting fine.
            quality_kind: quality_mod.ScoreKind = (
                "mase" if query.data_profile == "intermittent" else "mape"
            )
            result.quality = list(
                factory.quality_points(
                    backtests,
                    emitted_models,
                    frame.values,
                    coverage,
                    imputed_fraction=imputed_fraction,
                    score_kind=quality_kind,
                )
            )

        if d.emission.contribution and regressors_frame is not None:
            contributions = contribution_mod.compute_contributions(
                y=list(frame.values),
                regressors=regressors_frame,
            )
            if contributions:
                result.contribution = list(factory.contribution_points(contributions))

        # Drift telemetry. We compare the current full yhat curve to the
        # cached previous yhat curve per (group, query, series, model).
        # ``result.curve`` has one row per horizon step per emitted band
        # row; ``_record_drift`` filters to the ``_forecast`` (yhat) rows.
        if query.emission.drift.enabled and result.curve:
            self._record_drift(
                query=query,
                group_name=group_name,
                series_key=_series_key_from_labels(frame.labels),
                curve_points=result.curve,
                factory=factory,
                result=result,
                regressors=regressors_frame,
            )

        # Band coverage: record one (inside / outside) observation per
        # (model, level, window) against the trailing backtest row.
        # The buffer maxlen is sized to ``window / refresh_interval``
        # rather than ``window / step`` because the runner records one
        # observation per fit, not per data step — keeping the two in
        # lockstep so a "full" buffer means "this window of refresh
        # cadences is observed" instead of a 12x-oversized buffer that
        # never fills.
        if query.emission.band_coverage.enabled and backtests:
            self._record_band_coverage(
                query=query,
                group_name=group_name,
                series_key=_series_key_from_labels(frame.labels),
                backtests=backtests,
                emitted_models=emitted_models,
                observation_period=effective_refresh_interval(query, self._config.server),
            )

        return result

    def _record_drift(
        self,
        *,
        query: QueryConfig,
        group_name: str,
        series_key: str,
        curve_points: list[ForecastCurvePoint],
        factory: SeriesPointFactory,
        result: _FitResult,
        regressors: Any | None = None,
    ) -> None:
        """Compute drift gauges for every emitted (series, model) pair.

        We isolate the per-model **full** yhat curve from
        ``curve_points`` (each carries ``model`` on its labels, a
        ``_forecast`` / ``_forecast_lower`` / ``_forecast_upper``
        metric name, and a per-step ``timestamp_ms``). Only the
        yhat-row entries (metric exactly ``<id>_forecast``) feed the
        drift score so the bands don't pollute the comparison.

        Curves are compared against the previous curve cached on
        :attr:`_drift_cache`; the new curve overwrites the cache so
        the next refit gets the same treatment. ``NaN`` scores
        (no prior fit, or both curves all-NaN) skip the gauge — the
        exporter's render path filters non-finite samples and the
        alert counter only ticks on finite >= threshold values.

        ``regressors`` (when provided) is the per-series regressor
        DataFrame aligned to the fit window; the change-correlation
        explainer reads each column's trailing finite value to
        identify "what was loud at the drift moment" and to compute
        the rolling Pearson correlation between drift_score and
        regressor history.
        """
        drift_cfg = query.emission.drift
        yhat_metric = f"{query.id}_forecast"
        curves_by_model: dict[str, list[float]] = {}
        for point in curve_points:
            # The curve carries one row per horizon step per emitted
            # band; restrict to the exact yhat metric for this query so
            # the bands don't pollute the comparison and a query whose
            # id legitimately contains ``forecast`` (e.g.
            # ``api_forecast_count``) isn't mis-matched by a substring
            # check.
            if point.metric != yhat_metric:
                continue
            model_name = point.labels.get("model", "")
            if not model_name:
                continue
            curves_by_model.setdefault(model_name, []).append(point.value)
        current_regressor_values = _trailing_regressor_values(regressors)
        for model_name, curve in curves_by_model.items():
            previous = self._drift_cache.get(
                group=group_name,
                query=query.id,
                series_key=series_key,
                model=model_name,
            )
            score = (
                drift_mod.compute_drift(previous, curve, method=drift_cfg.method)
                if previous is not None
                else float("nan")
            )
            self._drift_cache.put(
                group=group_name,
                query=query.id,
                series_key=series_key,
                model=model_name,
                curve=curve,
            )
            if math.isfinite(score):
                labels = {**factory.drift_labels(model_name), "method": drift_cfg.method}
                result.drift_points.append(DriftPoint(labels=labels, score=score))
                # Push the (score, regressor_values) observation into
                # the per-(series, model) rolling window every fit —
                # not only on threshold crossings — so the
                # correlation has enough history to be meaningful by
                # the time drift fires.
                window = self._drift_window_for(
                    group_name=group_name,
                    query_id=query.id,
                    series_key=series_key,
                    model_name=model_name,
                    maxlen=drift_cfg.correlation_window,
                )
                window.record(score, current_regressor_values)
                if drift_cfg.alert_threshold is not None and score >= drift_cfg.alert_threshold:
                    result.drift_alerts += 1
                    self._emit_drift_explanation(
                        group_name=group_name,
                        query=query,
                        series_key=series_key,
                        model_name=model_name,
                        score=score,
                        threshold=drift_cfg.alert_threshold,
                        window=window,
                        current_regressor_values=current_regressor_values,
                    )

    def _drift_window_for(
        self,
        *,
        group_name: str,
        query_id: str,
        series_key: str,
        model_name: str,
        maxlen: int,
    ) -> drift_explainer_mod.DriftWindow:
        """Lazily create the per-(series, model) rolling window."""
        key = (group_name, query_id, series_key, model_name)
        window = self._drift_windows.get(key)
        if window is None or window.maxlen != maxlen:
            # Resize on reload: a new ``correlation_window`` config
            # value rebuilds the window so the bound takes effect on
            # the next fit. Prior samples drop because the historical
            # ones were collected under the old (possibly larger)
            # window.
            window = drift_explainer_mod.DriftWindow(maxlen=maxlen)
            self._drift_windows[key] = window
        return window

    def _emit_drift_explanation(
        self,
        *,
        group_name: str,
        query: QueryConfig,
        series_key: str,
        model_name: str,
        score: float,
        threshold: float,
        window: drift_explainer_mod.DriftWindow,
        current_regressor_values: dict[str, float],
    ) -> None:
        """Log a ``forecast_drift_explained`` event and bump the counter.

        Always emits the structured log line — even when no suspects
        emerge (no regressors configured, or none correlating /
        loud at the moment) — so "drift fired with no candidate
        explanation" is itself a visible operator signal rather than
        a silent gap.

        Two contextual fields ride alongside the suspects so the
        investigation hypothesis isn't limited to regressor changes:

        * ``last_config_reload_timestamp`` — surfaces "did someone
          just reload?" alongside the drift event. A recent reload
          is often itself the explanation. ``None`` when no reload
          has ever been recorded (cold-boot before the first apply).
        * ``change_point_count`` — cumulative input-side change-points
          detected for the (group, query). Lets an operator
          correlate "input-side regime shift detected" with
          "output-side drift fired" without scraping a second
          dashboard.
        """
        explanation = drift_explainer_mod.explain(window, current_regressor_values)
        reload_ts, _success, _failures = self._exporter._reload_state()
        snap = self._exporter.snapshot_for(group_name)
        change_point_count = (
            snap.change_points_cumulative.get(query.id, 0) if snap is not None else 0
        )
        logger.info(
            "forecast_drift_explained",
            group=group_name,
            query=query.id,
            model=model_name,
            series_key=series_key,
            drift_score=round(score, 6),
            drift_threshold=threshold,
            primary_suspect=explanation.primary_suspect,
            top_value_suspect=explanation.top_value_suspect,
            top_correlation_suspect=explanation.top_correlation_suspect,
            top_regressor_correlation=(
                round(explanation.top_regressor_correlation, 6)
                if math.isfinite(explanation.top_regressor_correlation)
                else None
            ),
            top_regressor_correlation_confidence=(explanation.top_regressor_correlation_confidence),
            correlation_window_size=explanation.correlation_window_size,
            regressor_values_at_drift={
                rid: round(v, 6) for rid, v in current_regressor_values.items()
            },
            last_config_reload_timestamp=reload_ts if reload_ts > 0 else None,
            change_point_count=change_point_count,
        )
        self._exporter.increment_drift_explained(
            group=group_name,
            query=query.id,
            primary_suspect=explanation.primary_suspect,
        )

    def _record_revision_and_disagreement(
        self,
        *,
        query: QueryConfig,
        group_name: str,
        frame: SeriesFrame,
        inputs: list[_CapacityInput],
        horizon_specs: list[HorizonSpec],
        auto_select: bool,
        result: _FitResult,
    ) -> None:
        """Emit ``forecast_revision_magnitude`` + ``forecast_model_disagreement``.

        Revision: per-(series, model, horizon) — relative shift from the
        previous fit's central forecast at the same horizon. Always
        emitted when at least one model produced a forecast for an
        accuracy horizon; the first fit per (series, model, horizon)
        records ``NaN`` (filtered at render time) and seeds the cache.

        Disagreement: per-(series, horizon) — coefficient of variation
        across emitted models' central forecasts at the same horizon.
        Only emitted in explicit mode (``auto_select=False``) with 2+
        models contributing a finite yhat at that horizon; auto-select
        modes only ever emit one model's forecast per series so the
        gauge would be a constant zero.
        """
        if not inputs or not horizon_specs:
            return
        series_key = _series_key_from_labels(frame.labels)
        common = {
            **{k: v for k, v in frame.labels.items() if k != "__name__"},
            "id": query.id,
            "group": group_name,
        }
        # Pull each emitted model's per-horizon yhat once and reuse for
        # both metrics. The frames retained on ``inputs`` are the raw
        # (uncalibrated) per-model forecasts — exactly what the next
        # fit will produce, so the revision number reflects the actual
        # prediction change rather than a calibration-side shift.
        per_horizon: dict[str, dict[str, float]] = {}
        for capacity_input in inputs:
            yhats = _extract_per_horizon_yhat(capacity_input.forecast, horizon_specs)
            if yhats:
                per_horizon[capacity_input.model_name] = yhats
        if not per_horizon:
            return
        # Revision: compare current to previous cache; update cache.
        for model_name, yhats in per_horizon.items():
            for horizon_label, current in yhats.items():
                previous = self._revision_cache.get(
                    group=group_name,
                    query=query.id,
                    series_key=series_key,
                    model=model_name,
                    horizon=horizon_label,
                )
                self._revision_cache.put(
                    group=group_name,
                    query=query.id,
                    series_key=series_key,
                    model=model_name,
                    horizon=horizon_label,
                    value=current,
                )
                magnitude = (
                    revision_mod.compute_revision_magnitude(previous=previous, current=current)
                    if previous is not None
                    else float("nan")
                )
                if not math.isfinite(magnitude):
                    continue
                result.revision_points.append(
                    RevisionPoint(
                        labels={**common, "model": model_name, "horizon": horizon_label},
                        magnitude=magnitude,
                    )
                )
        # Disagreement: per-(series, horizon) coefficient of variation
        # across the emitted models. Suppressed in auto-select / ensemble
        # modes because only one model's yhat is emitted per series, so
        # the cross-model spread is undefined.
        if auto_select or len(per_horizon) < _MIN_MODELS_FOR_DISAGREEMENT:
            return
        horizons_seen: set[str] = set()
        for yhats in per_horizon.values():
            horizons_seen.update(yhats.keys())
        for horizon_label in horizons_seen:
            values = [
                yhats[horizon_label] for yhats in per_horizon.values() if horizon_label in yhats
            ]
            if len(values) < _MIN_MODELS_FOR_DISAGREEMENT:
                continue
            score = _coefficient_of_variation(values)
            if not math.isfinite(score):
                continue
            result.model_disagreement_points.append(
                ModelDisagreementPoint(
                    labels={**common, "horizon": horizon_label},
                    score=score,
                )
            )

    def _record_band_coverage(
        self,
        *,
        query: QueryConfig,
        group_name: str,
        series_key: str,
        backtests: dict[str, BacktestOutcome],
        emitted_models: list[str],
        observation_period: timedelta,
    ) -> None:
        """Push one (inside / outside) observation per (model, level, window).

        Uses the trailing backtest row's (actual, lower, upper) tuple as
        the most recent observation — same source the deviation metrics
        consume. ``observation_period`` is the time between successive
        observations the runner produces; the tracker sizes each
        bucket's ring to ``window / observation_period`` so a "full"
        bucket really means "this window's worth of refreshes is
        recorded".
        """
        windows = query.emission.band_coverage.windows
        for model_name in emitted_models:
            outcome = backtests.get(model_name)
            if outcome is None:
                continue
            actual = outcome.last_actual
            if not math.isfinite(actual):
                continue
            for level, band in outcome.last_bands.items():
                if not (math.isfinite(band.lower) and math.isfinite(band.upper)):
                    continue
                inside = deviation_mod.inside_band(actual, band.lower, band.upper)
                for window in windows:
                    self._band_coverage.observe(
                        group=group_name,
                        query=query.id,
                        series_key=series_key,
                        model=model_name,
                        level=level,
                        window_label=_format_band_window(window),
                        window=window,
                        step=observation_period,
                        inside=inside,
                    )

    def _align_regressors_for_series(
        self,
        *,
        query: QueryConfig,
        frame: SeriesFrame,
        result: _FitResult,
        prefetch: regressors_mod.PromQLPrefetch,
    ) -> pd.DataFrame | None:
        """Align pre-fetched PromQL + materialise calendar features per series.

        The expensive PromQL fan-out already happened in
        :meth:`_prefetch_query_regressors`; this method is a pure pandas
        operation. Required-calendar failures (a config that escaped the
        boot validator) raise as :class:`RegressorError` and the caller
        skips just this series.

        Returning ``None`` is the signal to downstream model fits "no
        exogenous frame this run".
        """
        if not query.regressors:
            return None
        outcome = regressors_mod.align_for_series(
            regressors=query.regressors,
            prefetch=prefetch,
            timestamps=frame.timestamps,
        )
        for failure in outcome.failures:
            logger.info(
                "regressor_skipped",
                query=query.id,
                regressor=failure.regressor_id,
                reason=failure.reason,
                error=failure.message,
            )
            result.regressor_failures.append((failure.regressor_id, failure.reason))
        return outcome.frame

    @staticmethod
    def _emit_capacity_metrics(
        *,
        query: QueryConfig,
        forecast: pd.DataFrame,
        model_name: str,
        levels: list[int],
        factory: SeriesPointFactory,
        selection: SelectionResult | None,
        result: _FitResult,
        offsets: dict[int, float] | None = None,
    ) -> None:
        """Compute ETA + growth-rate samples from one model's forecast.

        Called once per emitted (model, series) after conformal calibration
        has decided how much to widen each band. ``offsets`` is the
        per-level additive widening to apply to ``yhat_upper`` / shrink
        from ``yhat_lower`` before computing band ETAs; ``None`` (or an
        empty dict) means no calibration ran for this model and the bands
        emit as the model produced them. Without this shift, an alert on
        ``time_to_threshold{level="95"}`` would disagree with the band a
        scrape can see at the same level — see
        ``_apply_offsets_to_threshold_eta_inputs`` below.

        Growth rate is computed from the point forecast (``yhat``) only,
        which calibration does not touch, so ``offsets`` does not affect
        it.
        """
        if forecast.empty:
            return
        # Apply the conformal offsets once and reuse the calibrated frame
        # for every consumer that reads ``yhat_upper_<level>`` /
        # ``yhat_lower_<level>`` (threshold ETAs and the SLO budget
        # projection). Without the cache a query with both ``thresholds:``
        # and ``slo:`` pays the DataFrame ``.copy()`` twice per fit.
        # ``_apply_offsets_to_threshold_eta_inputs`` short-circuits to the
        # original frame when no calibration ran, so the common
        # zero-offsets path stays zero-copy.
        needs_calibrated_bands = bool(query.thresholds) or query.slo is not None
        calibrated_df = (
            _apply_offsets_to_threshold_eta_inputs(forecast, offsets, levels)
            if needs_calibrated_bands
            else forecast
        )
        if query.thresholds:
            etas = capacity_mod.compute_threshold_etas(
                forecast=calibrated_df,
                thresholds=query.thresholds,
                levels=levels,
            )
            if etas:
                result.threshold_etas.extend(
                    factory.threshold_eta_points(model_name, etas, selection)
                )
                # The drain ETA family is the parallel emission for
                # ``le_drain`` thresholds — ``threshold_eta_points``
                # already filters them out so the two metric names
                # don't collide. Routing happens in the factory so
                # adding a future ``ge_fill`` (or similar) lands as
                # one new dispatch line, not a fresh runner branch.
                result.time_to_drain.extend(
                    factory.time_to_drain_points(model_name, etas, selection)
                )
                # Derive the predictive will_breach gauge from the same
                # ETA set so an alert on ``forecast_will_breach{...} == 1``
                # is always self-consistent with the
                # ``forecast_time_to_threshold_seconds`` numbers
                # displayed alongside it.
                breaches = capacity_mod.compute_will_breach(etas=etas, thresholds=query.thresholds)
                if breaches:
                    result.will_breach.extend(
                        factory.will_breach_points(model_name, breaches, selection)
                    )
        growth_cfg = query.emission.growth_rate
        if growth_cfg.enabled and growth_cfg.windows:
            windows_seconds = [w.total_seconds() for w in growth_cfg.windows]
            rates = capacity_mod.compute_growth_rates(
                forecast=forecast,
                windows_seconds=windows_seconds,
                units=growth_cfg.units,
            )
            if rates:
                result.growth_rates.extend(factory.growth_rate_points(model_name, rates, selection))
        # SLO error budget projection. Off unless ``slo:`` is configured
        # on the query; uses the leading forecast row as "now" so the
        # burn rate / exhausted-in match what an operator reading the
        # current /metrics scrape would compute by hand. Reads
        # ``yhat_upper_N`` from the *calibrated* frame so the per-level
        # ``exhausted_in`` agrees with the band an operator queries via
        # ``yhat_upper`` at the same labelset.
        if query.slo is not None:
            budget_samples = slo_mod.compute_budget(
                forecast=calibrated_df,
                slo=query.slo,
                levels=levels,
            )
            if budget_samples:
                result.error_budget.extend(
                    factory.error_budget_points(model_name, budget_samples, selection)
                )
        # Recommended maintenance windows. Sweeps the lookahead for the
        # top-N lowest-mean-load windows of the configured duration.
        rw_cfg = query.recommended_windows
        if rw_cfg.enabled:
            windows = maintenance_mod.recommend_windows(forecast=forecast, config=rw_cfg)
            if windows:
                result.recommended_windows.extend(
                    factory.recommended_maintenance_window_points(model_name, windows, selection)
                )

    def _emit_decomposition(
        self,
        *,
        group_name: str,
        query_id: str,
        forecast: pd.DataFrame,
        model_name: str,
        selection: SelectionResult | None,
        factory: SeriesPointFactory,
        in_sample_values: list[float],
        season_length: int,
        result: _FitResult,
    ) -> None:
        """Project ``forecast`` into trend/seasonal/level/residual snapshot rows.

        Pure post-processing on the already-computed forecast frame;
        cost is bounded to a few hundred float operations per (series,
        model). Models listed in
        :data:`decomposition.NON_DECOMPOSABLE_MODELS` skip with
        ``reason="unsupported_model"`` so operators see the no-op via
        ``forecast_decomposition_skipped_total``. ``group_name`` /
        ``query_id`` flow onto the skip counter so persistent skips
        can be alerted per-query without log scraping.
        """
        if not decomposition_mod.is_decomposable(model_name):
            result.decomposition_skipped.append((model_name, "unsupported_model"))
            self._exporter.increment_decomposition_skipped(
                group_name, query_id, model_name, "unsupported_model"
            )
            return
        try:
            decomposition = decomposition_mod.decompose(
                forecast=forecast,
                in_sample_values=in_sample_values,
                season_length=season_length,
            )
        except decomposition_mod.DecompositionError as exc:
            result.decomposition_skipped.append((model_name, exc.reason))
            self._exporter.increment_decomposition_skipped(
                group_name, query_id, model_name, exc.reason
            )
            return
        if not decomposition.components:
            result.decomposition_skipped.append((model_name, "components_unavailable"))
            self._exporter.increment_decomposition_skipped(
                group_name, query_id, model_name, "components_unavailable"
            )
            return
        result.component_points.extend(
            factory.component_points(
                forecast=forecast,
                model_name=model_name,
                components=decomposition.components,
                selection=selection,
            )
        )
        result.component_curve.extend(
            factory.component_curve_points(
                forecast=forecast,
                model_name=model_name,
                components=decomposition.components,
                selection=selection,
            )
        )

    async def _emit_conditional_forecasts(
        self,
        *,
        df: pd.DataFrame,
        query: QueryConfig,
        group_name: str,
        model_specs: list[ModelSpec],
        emitted_inputs: list[_CapacityInput],
        actual: float,
        horizon: int,
        freq: str,
        levels: list[int],
        season_length: int,
        regressors: pd.DataFrame | None,
        factory: SeriesPointFactory,
        result: _FitResult,
        offsets_by_model: dict[str, dict[int, float]],
        conditional_offsets_by_model: dict[str, dict[int, float]],
        clip: bool,
        rounder: bool,
    ) -> None:
        """Re-fit each emitted model with held-constant regressors.

        The pass adds one fit per emitted (model, series) on top of the
        regular fit, so the cost roughly doubles the forecast budget
        for queries that opt in. See the "Conditional forecasts" section
        of ``docs/inputs/regressors.md``.

        When ``regressors`` is None (no regressors configured), the
        held-constant frame is also None and the re-fit produces the
        same numbers as the unconditional pass — the family still emits
        so dashboards and alerts can target it unconditionally.

        ``conditional_offsets_by_model`` carries per-(model, level)
        offsets from a dedicated calibration pass that held regressors
        flat at their final pre-holdout value — matching the
        production-time conditional inference contract. When present
        for a model, those offsets widen the conditional band (and
        emit a ``forecast_conditional_band_calibration_offset`` gauge
        through ``result.conditional_calibration_offsets``). When
        absent (calibration off, fit failed, holdout too small), the
        runner falls back to ``offsets_by_model`` — the inherited
        unconditional offsets used before fresh per-fit calibration
        was added. The fallback path is no worse than that behaviour but
        ``forecast_conditional_calibration_failures_total`` ticks so an
        operator can see when fresh calibration silently isn't running.

        ``clip`` / ``rounder`` mirror the count-aware safeguards applied
        to the unconditional emission, so a count-shaped query sees
        consistent rounding/clipping on both families.

        Per-fit failures bump ``forecast_failures_total`` with a
        distinct ``conditional_fit_error`` / ``conditional_fit_timeout``
        reason so the operator can tell which fit family is misbehaving
        without log scraping.
        """
        conditional_regressors = _flatten_regressors_to_last_value(regressors)
        spec_by_name = {spec.name: spec for spec in model_specs}
        # Two short-circuits below skip the refit when it would be a
        # waste of CPU:
        #   1. No regressors configured at all -> the held-constant frame
        #      is identical to the original input, so the conditional fit
        #      would produce the same numbers.
        #   2. The model advertises ``uses_regressors = False`` -> the
        #      model's ``fit_predict`` ignores the regressors parameter,
        #      so a refit with held-constant regressors produces the same
        #      numbers as the unconditional fit.
        # In both cases we alias the already-computed unconditional frame
        # rather than burning a fresh fit. The family still emits so
        # downstream alerts have a stable schema (the docs explicitly
        # promise this for queries with no regressors).
        no_regressors_configured = regressors is None
        for ci in emitted_inputs:
            model_name = ci.model_name
            # The synthetic ``ensemble`` row in ensemble mode doesn't
            # correspond to a registered model, so we can't re-fit it.
            # Skip cleanly rather than papering over with a bad fit.
            if model_name == "ensemble":
                continue
            spec = spec_by_name.get(model_name) or ModelSpec(name=model_name)
            short_circuit = no_regressors_configured or not model_registry.uses_regressors(
                model_name
            )
            if short_circuit:
                # Alias the unconditional forecast — same numeric result
                # the refit would produce, at zero extra CPU. No
                # ``fit_duration`` is recorded for an aliased emission
                # because no fit ran; the slow-fit counter likewise
                # isn't touched. We *do* bump the operational
                # ``forecast_conditional_aliased_total`` counter so
                # operators can confirm the short-circuit is firing —
                # a flat-zero counter means every conditional emission
                # is paying the refit cost.
                self._exporter.increment_conditional_aliased(group_name, query.id, model_name)
                cond_forecast = ci.forecast
            else:
                t0 = time.monotonic()
                try:
                    cond_forecast = await self._fit_one(
                        spec=spec,
                        df=df,
                        horizon=horizon,
                        freq=freq,
                        levels=levels,
                        season_length=season_length,
                        regressors=conditional_regressors,
                    )
                except TimeoutError:
                    logger.warning(
                        "conditional_fit_timeout",
                        group=group_name,
                        query=query.id,
                        model=model_name,
                    )
                    result.conditional_failures["conditional_fit_timeout"] = (
                        result.conditional_failures.get("conditional_fit_timeout", 0) + 1
                    )
                    continue
                except Exception as exc:
                    logger.warning(
                        "conditional_fit_failed",
                        group=group_name,
                        query=query.id,
                        model=model_name,
                        error=str(exc),
                    )
                    result.conditional_failures["conditional_fit_error"] = (
                        result.conditional_failures.get("conditional_fit_error", 0) + 1
                    )
                    continue
                fit_seconds = time.monotonic() - t0
                # Track the conditional fit's duration alongside the
                # unconditional one so ``forecast_fit_duration_seconds``
                # reports both. Without this the slow-fit counter would
                # tick under ``model="X/conditional"`` but the gauge
                # would be silently absent.
                result.fit_durations[f"{model_name}/conditional"] = fit_seconds
                self._record_slow_fit_if_needed(
                    result=result,
                    group=group_name,
                    query_id=query.id,
                    model_name=f"{model_name}/conditional",
                    fit_seconds=fit_seconds,
                )

            # Prefer the fresh per-fit conditional offsets when the
            # dedicated calibration pass produced them; otherwise fall
            # back to the inherited unconditional offsets so the band
            # is no worse off than the pre-fresh-calibration behaviour.
            # When fresh offsets were computed, emit a
            # ``forecast_conditional_band_calibration_offset`` row per
            # (model, level) so the per-(model, level) widening
            # applied to the conditional band specifically is visible
            # from /metrics.
            fresh_offsets = conditional_offsets_by_model.get(model_name)
            if fresh_offsets:
                model_offsets: dict[int, float] | None = fresh_offsets
                result.conditional_calibration_offsets.extend(
                    factory.conditional_calibration_offset_points(model_name, fresh_offsets)
                )
            else:
                model_offsets = offsets_by_model.get(model_name)
            calibrated_df = _apply_offsets_to_threshold_eta_inputs(
                cond_forecast, model_offsets, levels
            )
            # Apply count-aware safeguards to the dataframe columns so
            # the snapshot rows, the sink curve, and the deviation
            # gauges all see the same clipped/rounded values.
            adjusted_df = _apply_safeguards_to_forecast_frame(
                calibrated_df, levels=levels, clip=clip, rounder=rounder
            )

            # Mark the conditional band rows with ``calibrated="true"``
            # when fresh per-fit offsets were applied — matches the
            # unconditional family's labelling so dashboards filtering
            # on ``calibrated`` work consistently across the two
            # families. We pass the flag instead of post-processing the
            # rows because the conditional path applies offsets to the
            # DataFrame *before* invoking the factory, so there's no
            # second pass available; the unconditional path goes the
            # other way (points first, then ``_apply_offsets_to_*``
            # rewrites them with the label) because its offsets are
            # computed after the per-series fits already produced
            # snapshot rows.
            band_calibrated = bool(fresh_offsets)
            result.conditional_points.extend(
                factory.conditional_forecast_points(
                    forecast=adjusted_df,
                    model_name=model_name,
                    selection=ci.selection,
                    calibrated=band_calibrated,
                )
            )
            # Stream the per-step conditional curve into the sink list so
            # remote_write users see the same smooth curve they get for
            # the unconditional family. The curve uses a parallel metric
            # name family (``_forecast_conditional*``) so the two paths
            # never collide on the TSDB side.
            result.conditional_curve.extend(
                factory.conditional_forecast_curve_points(
                    adjusted_df,
                    model_name,
                    selection=ci.selection,
                    calibrated=band_calibrated,
                )
            )
            if math.isfinite(actual):
                result.conditional_deviation_points.extend(
                    factory.conditional_deviation_points(
                        forecast=adjusted_df,
                        model_name=model_name,
                        actual=actual,
                    )
                )

    async def _explicit_forecast(
        self,
        *,
        df: pd.DataFrame,
        models: list[ModelSpec],
        query: QueryConfig,
        factory: SeriesPointFactory,
        group_name: str,
        horizon: int,
        freq: str,
        levels: list[int],
        season_length: int,
        result: _FitResult,
        regressors: pd.DataFrame | None = None,
    ) -> list[ForecastPoint]:
        # Fits run concurrently against the shared executor: per-series wall
        # time becomes ``max(fit_time)`` instead of ``sum(fit_time)`` whenever
        # the pool has spare workers. The pool itself caps total concurrency,
        # so this doesn't risk overloading anything beyond what a single
        # serial pass already touched.

        async def _timed(spec: ModelSpec) -> tuple[str, pd.DataFrame, float]:
            t0 = time.monotonic()
            forecast = await self._fit_one(
                spec=spec,
                df=df,
                horizon=horizon,
                freq=freq,
                levels=levels,
                season_length=season_length,
                regressors=regressors,
            )
            return spec.name, forecast, time.monotonic() - t0

        tasks = [_timed(spec) for spec in models]
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)
        out: list[ForecastPoint] = []
        for spec, outcome in zip(models, outcomes, strict=False):
            if isinstance(outcome, BaseException):
                logger.exception(
                    "fit_failed_in_explicit_mode",
                    group=group_name,
                    query=query.id,
                    exc_info=outcome,
                )
                # Per-model fit failure: record against the rolling success
                # ratio so the model-health alert fires regardless of which
                # mode (explicit/auto/ensemble) drove the fit.
                self._record_fit_outcome(group=group_name, model=spec.name, success=False)
                continue
            model_name, forecast, fit_seconds = outcome
            result.fit_durations[model_name] = fit_seconds
            self._record_slow_fit_if_needed(
                result=result,
                group=group_name,
                query_id=query.id,
                model_name=model_name,
                fit_seconds=fit_seconds,
            )
            self._record_fit_outcome(group=group_name, model=model_name, success=True)
            out.extend(factory.forecast_points(forecast, model_name, selection=None))
            result.curve.extend(factory.forecast_curve_points(forecast, model_name, selection=None))
            result.capacity_inputs.append(
                _CapacityInput(model_name=model_name, forecast=forecast, selection=None)
            )
        return out

    async def _auto_select_and_forecast(
        self,
        *,
        df: pd.DataFrame,
        query: QueryConfig,
        factory: SeriesPointFactory,
        group_name: str,
        horizon: int,
        freq: str,
        levels: list[int],
        season_length: int,
        backtest_results: dict[str, list[AccuracyResult]],
        result: _FitResult,
        regressors: pd.DataFrame | None = None,
    ) -> list[ForecastPoint]:
        d = self._config.defaults
        # ``data_profile: intermittent`` swaps the comparator from MAPE to
        # MASE — MAPE is undefined wherever the actual is zero, which is
        # precisely the regime intermittent models target. The selector
        # contract is unit-agnostic; we hand it MASE the same way we'd
        # hand it MAPE and the lowest finite score still wins. The
        # ``candidate_preference`` whitelist nudges the search toward the
        # registered intermittent wrappers when any are configured, so an
        # operator who lists ``[AutoARIMA, Croston, CrostonSBA]`` gets the
        # Croston family preferred without removing AutoARIMA as a safety
        # net.
        is_intermittent = query.data_profile == "intermittent"
        if is_intermittent:
            score_map = {
                model_name: score
                for model_name, results in backtest_results.items()
                if (score := overall_mase(results)) is not None
            }
        else:
            score_map = {
                model_name: mape
                for model_name, results in backtest_results.items()
                if (mape := overall_mape(results)) is not None
            }
        preference = model_registry.INTERMITTENT_MODEL_NAMES if is_intermittent else None
        selection = select(
            fallback_model=d.accuracy.fallback_model,
            fallback_mape_threshold=d.accuracy.fallback_mape_threshold,
            backtest_mape=score_map,
            candidate_preference=preference,
        )
        # Look up params for the chosen model. The fallback model may live
        # outside ``defaults.models`` (e.g. SeasonalNaive used as the safety
        # net for AutoARIMA-only setups); ``_resolve_spec`` returns an empty
        # params dict in that case so the fallback runs with stock defaults.
        spec = _resolve_spec(d.models, selection.model)
        t0 = time.monotonic()
        try:
            forecast = await self._fit_one(
                spec=spec,
                df=df,
                horizon=horizon,
                freq=freq,
                levels=levels,
                season_length=season_length,
                regressors=regressors,
            )
        except Exception:
            # Narrow on Exception (not BaseException) so a runner cancel
            # — asyncio.CancelledError, KeyboardInterrupt — propagates
            # without being recorded as a model-fit failure. TimeoutError
            # inherits from Exception in 3.11+, so the success-ratio
            # gauge still drops on real fit timeouts.
            self._record_fit_outcome(group=group_name, model=selection.model, success=False)
            raise
        fit_seconds = time.monotonic() - t0
        result.fit_durations[selection.model] = fit_seconds
        self._record_slow_fit_if_needed(
            result=result,
            group=group_name,
            query_id=query.id,
            model_name=selection.model,
            fit_seconds=fit_seconds,
        )
        self._record_fit_outcome(group=group_name, model=selection.model, success=True)
        result.curve.extend(
            factory.forecast_curve_points(forecast, selection.model, selection=selection)
        )
        result.capacity_inputs.append(
            _CapacityInput(model_name=selection.model, forecast=forecast, selection=selection)
        )
        return list(factory.forecast_points(forecast, selection.model, selection=selection))

    async def _ensemble_forecast(
        self,
        *,
        df: pd.DataFrame,
        models: list[ModelSpec],
        query: QueryConfig,
        factory: SeriesPointFactory,
        group_name: str,
        horizon: int,
        freq: str,
        levels: list[int],
        season_length: int,
        backtest_results: dict[str, list[AccuracyResult]],
        result: _FitResult,
        regressors: pd.DataFrame | None = None,
    ) -> list[ForecastPoint]:
        """Combine every model's forecast into an inverse-MAPE weighted average.

        Emits both the component models (so users can compare and inspect)
        *and* a synthetic ``ensemble`` model whose ``yhat``/bands are the
        weighted average across components, weighted by ``1 / MAPE``.

        Component models that fail to fit are simply excluded from the
        ensemble; if every model fails the result has no ensemble row but
        keeps whatever component forecasts succeeded. Bands are averaged
        with the same weights — this is conservative (a wide individual
        band drags the ensemble's wider) but stays well-defined when only
        some models report a given confidence level.
        """

        async def _timed(spec: ModelSpec) -> tuple[str, pd.DataFrame, float]:
            t0 = time.monotonic()
            forecast = await self._fit_one(
                spec=spec,
                df=df,
                horizon=horizon,
                freq=freq,
                levels=levels,
                season_length=season_length,
                regressors=regressors,
            )
            return spec.name, forecast, time.monotonic() - t0

        outcomes = await asyncio.gather(*(_timed(spec) for spec in models), return_exceptions=True)
        component_forecasts: dict[str, pd.DataFrame] = {}
        out: list[ForecastPoint] = []
        for spec, outcome in zip(models, outcomes, strict=False):
            if isinstance(outcome, BaseException):
                logger.exception(
                    "fit_failed_in_ensemble_mode",
                    group=group_name,
                    query=query.id,
                    exc_info=outcome,
                )
                self._record_fit_outcome(group=group_name, model=spec.name, success=False)
                continue
            model_name, forecast, fit_seconds = outcome
            result.fit_durations[model_name] = fit_seconds
            self._record_slow_fit_if_needed(
                result=result,
                group=group_name,
                query_id=query.id,
                model_name=model_name,
                fit_seconds=fit_seconds,
            )
            self._record_fit_outcome(group=group_name, model=model_name, success=True)
            component_forecasts[model_name] = forecast
            out.extend(factory.forecast_points(forecast, model_name, selection=None))
            result.curve.extend(factory.forecast_curve_points(forecast, model_name, selection=None))
            result.capacity_inputs.append(
                _CapacityInput(model_name=model_name, forecast=forecast, selection=None)
            )

        if not component_forecasts:
            return out

        # Compute inverse-score weights restricted to models that produced
        # a forecast this run; otherwise a backtested-but-failed-to-fit
        # model would leak into the ensemble's denominator.
        # ``data_profile: intermittent`` swaps MAPE for MASE so a series
        # whose actuals are mostly zero (and therefore has NaN MAPE
        # everywhere) still produces meaningful weights instead of
        # silently degrading to equal weights for every component.
        score_summary = overall_mase if query.data_profile == "intermittent" else overall_mape
        scoreable_scores = {
            name: score
            for name, results in backtest_results.items()
            if name in component_forecasts and (score := score_summary(results)) is not None
        }
        weights = ensemble_weights(scoreable_scores) or {
            name: 1.0 / len(component_forecasts) for name in component_forecasts
        }
        weights = {name: w for name, w in weights.items() if name in component_forecasts}
        if not weights:
            return out
        # Renormalise after the filter so the sum is back to 1.0; equal
        # weights from the empty-MAPE branch are already normalised.
        total = sum(weights.values())
        if total > 0:
            weights = {n: w / total for n, w in weights.items()}

        ensemble_df = _combine_ensemble(component_forecasts, weights, levels)
        out.extend(factory.forecast_points(ensemble_df, "ensemble", selection=None))
        result.curve.extend(factory.forecast_curve_points(ensemble_df, "ensemble", selection=None))
        # Calibration is intentionally skipped for the synthetic ensemble
        # row (``_compute_conformal_offsets`` filters it out), so we mark
        # this input as not eligible for offset application — its bands
        # are emitted as-is in both /metrics and the ETA family.
        result.capacity_inputs.append(
            _CapacityInput(
                model_name="ensemble",
                forecast=ensemble_df,
                selection=None,
                apply_calibration=False,
            )
        )
        result.ensemble = list(factory.ensemble_weight_points(weights))
        return out

    def _resolve_season_length(
        self,
        query: QueryConfig,
        group_name: str,
        frame: SeriesFrame,
        eff_step: timedelta,
    ) -> int:
        """Pick a season_length for this series, running auto-detection if asked.

        For non-``auto`` queries the static config-side resolver is the only
        source of truth. For ``"auto"`` we run :func:`seasonality.detect_season_length`
        on the series values; the static step-derived heuristic is used as
        the fallback when ACF can't find a confident peak.

        Observability:

        * The per-(group, query, source) counter
          ``forecast_season_length_detections_total`` is bumped on every
          detection — bounded cardinality, suitable for dashboards/alerts.
        * Anomalous outcomes (``fallback``, ``too_short``) are logged at
          INFO; healthy outcomes (``detected``, ``snapped``) at DEBUG so
          a high-cardinality group doesn't flood the log pipeline at
          INFO every refresh.
        """
        if query.season_length != "auto":
            return effective_season_length(query, self._config.defaults, step=eff_step)
        fallback = effective_season_length(query, self._config.defaults, step=eff_step)
        outcome = seasonality_mod.detect_season_length(
            values=frame.values,
            step=eff_step,
            fallback=fallback,
        )
        self._exporter.increment_season_length_detection(
            group=group_name, query=query.id, source=outcome.source
        )
        log_event = "season_length_auto_resolved"
        log_payload = {
            "query": query.id,
            "period": outcome.period,
            "source": outcome.source,
            "correlation": (
                None if not math.isfinite(outcome.correlation) else round(outcome.correlation, 3)
            ),
            "fallback": fallback,
        }
        if outcome.source in {"fallback", "too_short"}:
            logger.info(log_event, **log_payload)
        else:
            logger.debug(log_event, **log_payload)
        return outcome.period

    def _record_slow_fit_if_needed(
        self,
        *,
        result: _FitResult,
        group: str,
        query_id: str,
        model_name: str,
        fit_seconds: float,
    ) -> None:
        """Log + tag a fit that crossed the slow-fit threshold.

        See ``SLOW_FIT_RATIO`` for the threshold rationale. We log at WARN
        and add the model name to ``result.slow_fits`` so the run-context
        accumulator can bump the per-(query, model) counter; rendering as
        a counter (not a gauge) is the right shape for ``rate(...)`` alerts.
        """
        timeout_seconds = self._config.safety.fit_timeout.total_seconds()
        if timeout_seconds <= 0:
            return
        threshold = timeout_seconds * _pkg.SLOW_FIT_RATIO
        if fit_seconds < threshold:
            return
        result.slow_fits.add(model_name)
        logger.warning(
            "fit_slow",
            group=group,
            query=query_id,
            model=model_name,
            fit_seconds=round(fit_seconds, 3),
            threshold_seconds=round(threshold, 3),
            timeout_seconds=round(timeout_seconds, 3),
        )

    def _record_fit_outcome(self, *, group: str, model: str, success: bool) -> None:
        """Record one per-(group, model) main-forecast fit attempt.

        Wraps the exporter's tracker so the runner doesn't reach into
        private attributes. Counts both successes and failures (any
        per-model exception or timeout) on the main forecast emission
        path — explicit, auto-select, and ensemble modes are all
        instrumented. The optional conditional-band refit is **not**
        counted here on purpose: a model that succeeds on the
        unconditional forecast but happens to fail the conditional
        refit (e.g. regressor flat-hold degenerates the search) isn't
        "the model is failing"; downstream ``ForecastModelFitFailing``
        alerts would misfire. Conditional failures land separately on
        ``forecast_failures_total{reason="conditional_fit_*"}``.
        """
        self._exporter.record_fit_outcome(group=group, model=model, success=success)

    def _build_annotation_events(
        self, group: GroupConfig, ctx: _RunContext
    ) -> list[annotations_mod.AnnotationEvent]:
        """Translate this run's significant events into annotation events.

        Reads the per-group run context (drift alerts, change points,
        predicted breaches, outside-band deviations) plus the global
        deploy-risk scores. The sink itself filters by configured event
        family and dedups repeats, so this builder emits liberally for
        every *currently firing* event.
        """
        events: list[annotations_mod.AnnotationEvent] = []
        g = group.name
        for query_id, count in ctx.drift_alerts.items():
            if count > 0:
                events.append(
                    annotations_mod.AnnotationEvent(
                        event_type="drift",
                        dedup_key=f"{g}:{query_id}",
                        tags=["promforecast", "drift", f"group:{g}", f"id:{query_id}"],
                        text=f"Forecast drift threshold crossed for {query_id} in group {g}",
                    )
                )
        for query_id, count in ctx.change_points.items():
            if count > 0:
                events.append(
                    annotations_mod.AnnotationEvent(
                        event_type="change_point",
                        dedup_key=f"{g}:{query_id}",
                        tags=["promforecast", "change_point", f"group:{g}", f"id:{query_id}"],
                        text=f"Change-point detected in {query_id} (group {g})",
                    )
                )
        for point in ctx.will_breach_points:
            if point.value >= 1.0:
                qid = _query_id_from_metric(point.metric)
                name = point.labels.get("name", "")
                events.append(
                    annotations_mod.AnnotationEvent(
                        event_type="threshold_breach_predicted",
                        dedup_key=f"{g}:{qid}:{name}:{_series_key_from_labels(point.labels)}",
                        tags=[
                            "promforecast",
                            "threshold_breach_predicted",
                            f"group:{g}",
                            f"id:{qid}",
                        ],
                        text=(
                            f"Predicted threshold breach for {qid} (threshold {name!r}, group {g})"
                        ),
                    )
                )
        for dev in ctx.deviation_points:
            if dev.outside_band:
                qid = dev.labels.get("id", "")
                events.append(
                    annotations_mod.AnnotationEvent(
                        event_type="deviation_outside_band",
                        dedup_key=f"{g}:{qid}:{_series_key_from_labels(dev.labels)}",
                        tags=["promforecast", "deviation_outside_band", f"group:{g}", f"id:{qid}"],
                        text=f"Actual outside forecast band for {qid} (group {g})",
                    )
                )
        min_score = self._config.annotations.deploy_risk_min_score
        for risk in self._exporter._deploy_risk_state():
            # Only an *elevated* deploy posts an annotation — a rollout sitting
            # at a low risk score is steady-state, not a moment worth marking.
            if risk.score < min_score:
                continue
            service = risk.labels.get("service", "")
            deployment_id = risk.labels.get("deployment_id", "")
            events.append(
                annotations_mod.AnnotationEvent(
                    event_type="deploy_risk",
                    dedup_key=f"{service}:{deployment_id}",
                    tags=["promforecast", "deploy_risk", f"service:{service}"],
                    text=(
                        f"Deploy risk score {risk.score:.2f} for {service} "
                        f"(deployment {deployment_id})"
                    ),
                )
            )
        return events

    async def _emit_canary_deviation(
        self,
        query: QueryConfig,
        group: GroupConfig,
        frames: list[SeriesFrame],
        ctx: _RunContext,
        qlog: structlog.stdlib.BoundLogger,
    ) -> None:
        """Fit the baseline population and score each canary series against it.

        Splits ``frames`` into a canary and a baseline population by the
        configured selectors, fits the first configured model on the baseline
        aggregate, backtests one horizon to obtain a trailing forecast band,
        and emits ``<id>_canary_forecast_deviation`` for each canary series'
        most recent actual versus that band. Failures degrade gracefully —
        an unfittable baseline or a missing population simply emits nothing.
        """
        import pandas as pd  # noqa: PLC0415

        d = self._config.defaults
        try:
            canary_frames, baseline_frames = canary_mod.split_frames(
                frames,
                canary_selector=query.canary.canary_selector,
                baseline_selector=query.canary.baseline_selector,
            )
        except canary_mod.CanarySelectorError as exc:
            qlog.warning("canary_selector_invalid", error=str(exc))
            ctx.failures["canary_selector"] = ctx.failures.get("canary_selector", 0) + 1
            return
        if not canary_frames or not baseline_frames:
            qlog.info(
                "canary_population_empty",
                canary=len(canary_frames),
                baseline=len(baseline_frames),
            )
            return

        timestamps, values = canary_mod.aggregate_baseline(baseline_frames)
        eff_step = effective_step(query, d)
        horizon_steps = max(1, int(d.horizon.total_seconds() // eff_step.total_seconds()))
        eval_steps = min(horizon_steps, len(values) - 1)
        if len(values) < max(d.min_points, eval_steps + 2):
            qlog.info("canary_baseline_too_short", points=len(values))
            return

        df = pd.DataFrame(
            {
                "unique_id": ["baseline"] * len(values),
                "ds": pd.to_datetime(timestamps),
                "y": values,
            }
        )
        freq = _pandas_freq(eff_step)
        season_length = effective_season_length(query, d, step=eff_step)
        levels = list(d.confidence_levels)
        spec = d.models[0]
        summary = await self._backtest_baseline(
            df=df,
            spec=spec,
            eval_steps=eval_steps,
            freq=freq,
            levels=levels,
            season_length=season_length,
            group_name=group.name,
            query_id=query.id,
        )
        if summary is None:
            qlog.info("canary_baseline_unfittable")
            return
        _last_actual, last_yhat, last_bands = summary

        points: list[CanaryDeviationPoint] = []
        for frame in canary_frames:
            actual = canary_mod.latest_actual(frame)
            if actual is None:
                continue
            for level in levels:
                band = last_bands.get(level)
                if band is None:
                    continue
                dev = deviation_mod.compute(actual, last_yhat, band.lower, band.upper)
                # ``id`` mirrors the v0.2 deviation shape and lets the
                # partial-tick inheritance registry scope these rows by query.
                labels = {
                    **frame.labels,
                    "id": query.id,
                    "model": spec.name,
                    "level": str(level),
                }
                points.append(
                    CanaryDeviationPoint(
                        metric=query.id,
                        labels=labels,
                        ratio=dev.ratio,
                        outside_band=dev.outside_band,
                    )
                )
        ctx.canary_deviation_points.extend(points)
        qlog.info(
            "canary_deviation_emitted",
            canary_series=len(canary_frames),
            baseline_series=len(baseline_frames),
            points=len(points),
        )

    async def _backtest_baseline(
        self,
        *,
        df: pd.DataFrame,
        spec: ModelSpec,
        eval_steps: int,
        freq: str,
        levels: list[int],
        season_length: int,
        group_name: str,
        query_id: str,
    ) -> tuple[float, float, dict[int, Any]] | None:
        """Run one baseline backtest in the executor; return its trailing band.

        Returns ``(last_actual, last_yhat, bands)`` exactly as
        ``_summarise_backtest`` does, or ``None`` when the fit fails or the
        frames are degenerate.
        """
        loop = asyncio.get_running_loop()
        timeout = self._config.safety.fit_timeout.total_seconds()
        model = model_registry.build(spec.name, season_length=season_length, params=spec.params)
        try:
            with self._tracer.span(
                "promforecast.fit",
                model=spec.name,
                kind="canary_baseline",
                horizon=eval_steps,
                freq=freq,
                season_length=season_length,
            ):
                async with self._fits_semaphore:
                    outcome = await asyncio.wait_for(
                        loop.run_in_executor(
                            self._executor,
                            _run_backtest,
                            model,
                            df,
                            eval_steps,
                            freq,
                            levels,
                            None,
                        ),
                        timeout=timeout,
                    )
        except Exception:
            logger.exception("canary_baseline_backtest_failed", group=group_name, query=query_id)
            return None
        if outcome is None:
            return None
        test, predictions = outcome
        return _pkg._summarise_backtest(predictions=predictions, test=test, levels=levels)

    async def _run_backtests(
        self,
        *,
        df: pd.DataFrame,
        models: list[ModelSpec],
        horizons: list[HorizonSpec],
        freq: str,
        levels: list[int],
        season_length: int,
        group_name: str,
        query_id: str,
        needs: _BacktestNeeds,
        regressors: pd.DataFrame | None = None,
    ) -> dict[str, BacktestOutcome]:
        """Backtest each configured model. Per-model failures are isolated.

        The hold-out window is ``max(horizon.steps)`` — large enough to score
        every horizon from a single fit, which keeps the cost to one extra
        per-model fit per series. With ``auto_select: true`` the per-series
        cost is therefore N + 1 fits (N backtests + 1 final forecast on full
        data); with ``auto_select: false`` it's N + N (N backtests + N final
        forecasts). Set ``accuracy.evaluate: false`` *and* disable deviation/
        quality emission to skip backtesting when the fit budget is tight.

        Backtest fits dispatch concurrently against the executor pool, so
        per-series wall time tracks the slowest model rather than the sum.

        ``needs`` controls which post-fit steps run: per-horizon MAPE/MASE
        (``needs_results``) and the trailing (actual, yhat, bands) summary
        (``needs_summary``) are skipped when no consumer wants them.
        """
        if not horizons:
            return {}
        eval_steps = max(h.steps for h in horizons)
        if len(df) <= eval_steps + 1:
            return {}
        train = df.iloc[:-eval_steps].reset_index(drop=True)
        loop = asyncio.get_running_loop()
        timeout = self._config.safety.fit_timeout.total_seconds()

        async def _one(spec: ModelSpec) -> tuple[pd.DataFrame, pd.DataFrame] | None:
            model = model_registry.build(spec.name, season_length=season_length, params=spec.params)
            # Mirror the production-fit span shape so an OTel trace shows
            # backtest fits alongside forecast fits — same `model` /
            # `season_length` / `freq` attributes, plus a `kind` tag so a
            # query can be filtered to "backtest" or "forecast" cleanly.
            with self._tracer.span(
                "promforecast.fit",
                model=spec.name,
                kind="backtest",
                horizon=eval_steps,
                freq=freq,
                season_length=season_length,
            ):
                async with self._fits_semaphore:
                    return await asyncio.wait_for(
                        loop.run_in_executor(
                            self._executor,
                            _run_backtest,
                            model,
                            df,
                            eval_steps,
                            freq,
                            levels,
                            regressors,
                        ),
                        timeout=timeout,
                    )

        outcomes = await asyncio.gather(*(_one(spec) for spec in models), return_exceptions=True)
        results: dict[str, BacktestOutcome] = {}
        model_names = [spec.name for spec in models]
        for model_name, outcome in zip(model_names, outcomes, strict=True):
            if isinstance(outcome, BaseException):
                logger.exception(
                    "backtest_failed",
                    group=group_name,
                    query=query_id,
                    model=model_name,
                    exc_info=outcome,
                )
                continue
            if outcome is None:
                continue
            test, predictions = outcome
            if needs.needs_summary:
                summary = _pkg._summarise_backtest(
                    predictions=predictions, test=test, levels=levels
                )
                # Deviation and quality both rely on the trailing tuple; if it
                # can't be derived (degenerate frames) drop the model rather
                # than emit half-populated rows downstream.
                if summary is None:
                    continue
                last_actual, last_yhat, last_bands = summary
            else:
                last_actual, last_yhat, last_bands = float("nan"), float("nan"), {}
            scores = (
                evaluate(
                    test=test,
                    predictions=predictions,
                    train=train,
                    season_length=season_length,
                    horizons=horizons,
                    model_name=model_name,
                )
                if needs.needs_results
                else []
            )
            results[model_name] = BacktestOutcome(
                results=scores,
                last_actual=last_actual,
                last_yhat=last_yhat,
                last_bands=last_bands,
            )
        return results

    async def _apply_conformal_calibration(
        self,
        *,
        df: pd.DataFrame,
        forecast_points: list[ForecastPoint],
        model_specs: list[ModelSpec],
        freq: str,
        levels: list[int],
        season_length: int,
        regressors: pd.DataFrame | None,
        holdout_fraction: float,
        result: _FitResult,
        factory: SeriesPointFactory,
    ) -> tuple[list[ForecastPoint], dict[str, dict[int, float]]]:
        """Compute and apply conformal offsets to /metrics and curve points.

        Runs an extra fit on the configured holdout tail, computes per-level
        offsets from the absolute residuals, and applies them to both the
        snapshot ``forecast_points`` (consumed by /metrics) and to
        ``result.curve`` (consumed by the remote_write sink) so a deployment
        that pushes the full forecast curve to a TSDB sees the same
        calibrated bands the scrape exposition serves.

        The offsets per (series, model, level) are emitted as
        ``forecast_band_calibration_offset`` samples on ``result.calibration_offsets``.
        Failures inside :meth:`_compute_conformal_offsets` are isolated per
        model — the forecast still emits, just without the band correction.

        The offsets dict is also returned so the caller can feed it into
        the capacity-planning emission and keep ``time_to_threshold`` band
        ETAs aligned with the bands /metrics actually exposes.
        """
        offsets_by_model = await self._compute_conformal_offsets(
            df=df,
            forecast_points=forecast_points,
            model_specs=model_specs,
            freq=freq,
            levels=levels,
            season_length=season_length,
            regressors=regressors,
            holdout_fraction=holdout_fraction,
        )
        if not offsets_by_model:
            return forecast_points, {}
        adjusted_points = _apply_offsets_to_snapshot_points(forecast_points, offsets_by_model)
        if result.curve:
            result.curve = _apply_offsets_to_curve_points(result.curve, offsets_by_model)
        for model_name, offsets in offsets_by_model.items():
            result.calibration_offsets.extend(
                factory.calibration_offset_points(model_name, offsets)
            )
        return adjusted_points, offsets_by_model

    async def _compute_conformal_offsets(
        self,
        *,
        df: pd.DataFrame,
        forecast_points: list[ForecastPoint],
        model_specs: list[ModelSpec],
        freq: str,
        levels: list[int],
        season_length: int,
        regressors: pd.DataFrame | None,
        holdout_fraction: float,
    ) -> dict[str, dict[int, float]]:
        """For every emitted model, compute per-level conformal offsets.

        The calibration fit is a single extra model fit per series per
        emitted model, on the pre-holdout prefix; cost-wise that's the
        same order as one extra backtest. Failures (degenerate holdout,
        fit timeout, model error) are isolated to the affected model: the
        forecast emission still happens, just without the calibration
        correction.
        """
        # Which models actually have emitted forecast rows this run? In
        # auto-select mode that's the winner only; in explicit mode it's
        # every configured model; in ensemble mode it's every component
        # plus a synthetic "ensemble" row. We skip the synthetic ensemble
        # row because it has no underlying fit_predict to recalibrate
        # against on the holdout.
        emitted_models: set[str] = {
            p.labels.get("model", "") for p in forecast_points if p.metric.endswith("_forecast")
        }
        emitted_models.discard("")
        emitted_models.discard("ensemble")
        if not emitted_models:
            return {}
        split = conformal_mod.holdout_split(df, holdout_fraction=holdout_fraction)
        if split is None:
            return {}
        train_df, calib_df = split
        train_regressors: pd.DataFrame | None
        if regressors is not None and len(regressors) >= len(train_df):
            train_regressors = regressors.iloc[: len(train_df)].reset_index(drop=True)
        else:
            train_regressors = None
        calib_actuals = calib_df["y"].to_numpy(dtype=float)
        calib_steps = len(calib_df)
        # Two points is the bare statistical floor for a quantile estimate
        # — anything smaller is just noise. The split itself already caps
        # us below at 5 via ``holdout_split``'s ``min_holdout``, but this
        # is the second-line guard for callers that bypass the split helper.
        min_calib_for_quantile = 2
        if calib_steps < min_calib_for_quantile:
            return {}

        spec_by_name = {spec.name: spec for spec in model_specs}
        results: dict[str, dict[int, float]] = {}
        for model_name in sorted(emitted_models):
            spec = spec_by_name.get(model_name) or ModelSpec(name=model_name)
            try:
                calib_predictions = await self._fit_one(
                    spec=spec,
                    df=train_df,
                    horizon=calib_steps,
                    freq=freq,
                    levels=levels,
                    season_length=season_length,
                    regressors=train_regressors,
                )
            except TimeoutError:
                # Hit ``safety.fit_timeout`` — distinct from a generic
                # fit error so logs split "calibration too slow" from
                # "calibration model crashed". The forecast still emits
                # without calibration; the missing
                # ``forecast_band_calibration_offset`` sample is the
                # operator-facing signal that calibration didn't run.
                logger.warning(
                    "conformal_calibration_timeout",
                    model=model_name,
                    timeout_s=self._config.safety.fit_timeout.total_seconds(),
                )
                continue
            except Exception as exc:
                # Asyncio CancelledError is intentionally *not* caught
                # (Exception does not include it in Python 3.8+) so a
                # shutdown / drain still tears down cleanly.
                logger.warning(
                    "conformal_calibration_fit_failed",
                    model=model_name,
                    error=str(exc),
                )
                continue
            calib = conformal_mod.compute_offsets(
                calib_actuals=calib_actuals,
                calib_predictions=calib_predictions,
                levels=levels,
            )
            if calib.offsets:
                results[model_name] = calib.offsets
        return results

    async def _compute_conditional_conformal_offsets(  # noqa: PLR0912
        self,
        *,
        df: pd.DataFrame,
        group_name: str,
        query_id: str,
        emitted_inputs: list[_CapacityInput],
        model_specs: list[ModelSpec],
        freq: str,
        levels: list[int],
        season_length: int,
        regressors: pd.DataFrame | None,
        holdout_fraction: float,
    ) -> dict[str, dict[int, float]]:
        """Mirror of :meth:`_compute_conformal_offsets` for the conditional band.

        The only difference from the unconditional pass is the regressor
        frame: the calibration fit's training regressors are flat-held at
        the **final pre-holdout** value, matching the production-time
        conditional inference contract. Computing fresh per-fit offsets
        here means the conditional band's empirical-coverage guarantee
        is grounded in residuals scored against the same regressor
        regime production sees, rather than the unconditional pass's
        residuals where regressors *did* vary across the holdout.

        Per-(model) failures isolate cleanly and bump
        ``forecast_conditional_calibration_failures_total`` with a
        bounded ``reason``. The caller falls back to
        ``unconditional_offsets_by_model`` for any model missing from
        the returned dict, so the fallback path is no worse than the
        pre-fresh-calibration behaviour.

        Short-circuits to the empty dict (and bumps no failure counter)
        when none of the emitted models actually consumes regressors —
        the fresh fit would produce the same numbers as the unconditional
        fit, so emitting two parallel offset gauges with identical
        values would just be noise.
        """
        emitted_model_names: set[str] = set()
        for ci in emitted_inputs:
            if ci.model_name and ci.model_name != "ensemble":
                emitted_model_names.add(ci.model_name)
        if not emitted_model_names:
            return {}
        # If every emitted model ignores regressors, the conditional fit
        # produces the same numbers as the unconditional fit — no point
        # spending CPU to compute identical offsets.
        if all(not model_registry.uses_regressors(name) for name in emitted_model_names):
            return {}
        split = conformal_mod.holdout_split(df, holdout_fraction=holdout_fraction)
        if split is None:
            self._exporter.increment_conditional_calibration_failure(
                group_name, query_id, "degenerate_holdout"
            )
            return {}
        train_df, calib_df = split
        calib_actuals = calib_df["y"].to_numpy(dtype=float)
        calib_steps = len(calib_df)
        min_calib_for_quantile = 2
        if calib_steps < min_calib_for_quantile:
            self._exporter.increment_conditional_calibration_failure(
                group_name, query_id, "degenerate_holdout"
            )
            return {}
        # Build the held-constant regressor frame matching the
        # production-time conditional contract: every row equals the
        # final value of the pre-holdout window. ``None`` propagates
        # through to a no-regressors fit for the (unusual) case where
        # the configured regressors all resolved to nothing during the
        # main fit.
        train_regressors: pd.DataFrame | None
        if regressors is not None and len(regressors) >= len(train_df):
            train_regressors = _flatten_regressors_to_last_value(
                regressors.iloc[: len(train_df)].reset_index(drop=True)
            )
        else:
            train_regressors = None

        spec_by_name = {spec.name: spec for spec in model_specs}
        results: dict[str, dict[int, float]] = {}
        for model_name in sorted(emitted_model_names):
            # Models that don't consume regressors produce identical
            # numbers either way; skip the extra fit (and skip the
            # gauge emission so the operator isn't shown a duplicate
            # of the unconditional widening).
            if not model_registry.uses_regressors(model_name):
                continue
            spec = spec_by_name.get(model_name) or ModelSpec(name=model_name)
            try:
                calib_predictions = await self._fit_one(
                    spec=spec,
                    df=train_df,
                    horizon=calib_steps,
                    freq=freq,
                    levels=levels,
                    season_length=season_length,
                    regressors=train_regressors,
                )
            except TimeoutError:
                logger.warning(
                    "conditional_conformal_calibration_timeout",
                    group=group_name,
                    query=query_id,
                    model=model_name,
                    timeout_s=self._config.safety.fit_timeout.total_seconds(),
                )
                self._exporter.increment_conditional_calibration_failure(
                    group_name, query_id, "fit_timeout"
                )
                continue
            except Exception as exc:
                logger.warning(
                    "conditional_conformal_calibration_fit_failed",
                    group=group_name,
                    query=query_id,
                    model=model_name,
                    error=str(exc),
                )
                self._exporter.increment_conditional_calibration_failure(
                    group_name, query_id, "fit_error"
                )
                continue
            calib = conformal_mod.compute_offsets(
                calib_actuals=calib_actuals,
                calib_predictions=calib_predictions,
                levels=levels,
            )
            if calib.offsets:
                results[model_name] = calib.offsets
        return results

    async def _fit_one(
        self,
        *,
        spec: ModelSpec,
        df: pd.DataFrame,
        horizon: int,
        freq: str,
        levels: list[int],
        season_length: int,
        regressors: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        loop = asyncio.get_running_loop()
        model = model_registry.build(spec.name, season_length=season_length, params=spec.params)
        with self._tracer.span(
            "promforecast.fit",
            model=spec.name,
            kind="forecast",
            horizon=horizon,
            freq=freq,
            season_length=season_length,
        ):
            # ``_fits_semaphore`` caps the number of concurrent fits across
            # every group/query — important when per-query refresh
            # schedules synchronise multiple groups onto the same tick.
            async with self._fits_semaphore:
                return await asyncio.wait_for(
                    loop.run_in_executor(
                        self._executor,
                        _run_fit,
                        model,
                        df,
                        horizon,
                        freq,
                        levels,
                        regressors,
                    ),
                    timeout=self._config.safety.fit_timeout.total_seconds(),
                )

    def _build_band_coverage_points(
        self,
        group: GroupConfig,
        run_query_ids: set[str],
    ) -> list[BandCoveragePoint]:
        """Snapshot the tracker into ``BandCoveragePoint``s for the exporter.

        We render *every* tracked sample, regardless of which queries
        ran this tick, so /metrics always shows the latest rolling
        fraction for every (group, query) tuple with data. The
        ``run_query_ids`` parameter is reserved for future filtering
        but unused today — the tracker is bounded by the live config
        already via :meth:`band_coverage.BandCoverageTracker.forget`.
        """
        del run_query_ids
        live_query_ids = {q.id for q in group.queries}
        out: list[BandCoveragePoint] = []
        for sample in self._band_coverage.samples():
            if sample.labels.get("group") != group.name:
                continue
            if sample.labels.get("id") not in live_query_ids:
                continue
            out.append(
                BandCoveragePoint(
                    labels=sample.labels,
                    fraction=sample.fraction,
                    samples=sample.samples,
                )
            )
        return out

    async def pause(self) -> None:
        """Stop scheduling, leave the executor / source / sink alive.

        HA mode uses this when leadership is lost: we want to halt new fits
        immediately but keep the process available to take leadership back
        without rebuilding the heavy state.
        """
        self._stopped.set()
        tasks = list(self._tasks.values())
        self._tasks.clear()
        self._stop_events.clear()
        for task in tasks:
            task.cancel()
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    async def stop(self) -> None:
        """Shut down the scheduler and cancel in-flight runs.

        Unlike a config reload, process shutdown does not give in-flight
        runs a grace period: SIGTERM gets ten seconds in Kubernetes by
        default, and we'd rather drop a half-finished snapshot than
        delay the termination.
        """
        await self.pause()
        self._executor.shutdown(wait=False, cancel_futures=True)
        await self._source.aclose()
        if self._sink is not None:
            with contextlib.suppress(Exception):
                await self._sink.aclose()
        if self._webhook_client is not None:
            with contextlib.suppress(Exception):
                await self._webhook_client.aclose()
        if self._trigger_engine is not None:
            with contextlib.suppress(Exception):
                await self._trigger_engine.aclose()
        if self._annotation_sink is not None:
            with contextlib.suppress(Exception):
                await self._annotation_sink.aclose()
