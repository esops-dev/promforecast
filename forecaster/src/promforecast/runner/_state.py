"""Per-fit and per-group state accumulators threaded through the runner.

``_FitResult`` holds the per-series outputs returned by
``Runner._fit_series``; ``_RunContext`` is the per-group mutable
accumulator that the runner extends as each series in a group is
processed. ``_merge_fit_into_ctx`` splices the two together, and
``_inherit_prior_for_skipped_queries`` preloads the context with
prior-snapshot data for any query that's not running on the current
tick (the per-query refresh-interval case).

The inheritance loop is driven by a small ``_InheritanceSpec`` registry
so adding an emission family means appending one tuple to
:data:`_INHERITANCE_REGISTRY` rather than hand-coding a copy block.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from ..exporter import (
    AccuracyPoint,
    CalibrationOffsetPoint,
    ComponentDecompositionPoint,
    ConditionalCalibrationOffsetPoint,
    ConditionalDeviationPoint,
    ConditionalForecastPoint,
    ContributionPoint,
    DeviationPoint,
    DriftPoint,
    EnsembleWeightPoint,
    ErrorBudgetPoint,
    ForecastPoint,
    GroupSnapshot,
    GrowthRatePoint,
    ModelDisagreementPoint,
    PreprocessStats,
    QualityPoint,
    RecommendedMaintenanceWindowPoint,
    RevisionPoint,
    ThresholdEtaPoint,
    TimeToDrainPoint,
    WillBreachPoint,
)
from ..sink import ForecastCurvePoint
from ._helpers import _query_id_from_metric


@dataclass
class _CapacityInput:
    """One emitted (model, forecast frame) tuple, deferred to post-calibration.

    The forecast frame is kept by reference (not copied) so capturing it
    is essentially free. ``selection`` is the auto-select decision that
    produced this emission, ``None`` for explicit / ensemble component
    rows. ``apply_calibration`` is False for the synthetic ensemble row
    because conformal calibration is not run against ensemble outputs.
    """

    model_name: str
    forecast: Any  # pandas.DataFrame (avoid the import in non-TYPE_CHECKING land)
    selection: Any  # SelectionResult | None
    apply_calibration: bool = True


@dataclass
class _FitResult:
    """Per-series outputs returned by ``Runner._fit_series``."""

    points: list[ForecastPoint] = field(default_factory=list)
    accuracy: list[AccuracyPoint] = field(default_factory=list)
    deviation: list[DeviationPoint] = field(default_factory=list)
    quality: list[QualityPoint] = field(default_factory=list)
    ensemble: list[EnsembleWeightPoint] = field(default_factory=list)
    contribution: list[ContributionPoint] = field(default_factory=list)
    # Full forecast curve (every horizon step) per emitted model. Consumed
    # only by the ``remote_write`` sink — the snapshot exporter ignores it.
    curve: list[ForecastCurvePoint] = field(default_factory=list)
    # Per-model wall-clock fit time (seconds), aggregated to a single
    # number per (query, model) by ``_RunContext`` below. Surfaces as
    # ``forecast_fit_duration_seconds`` for fit-time observability.
    fit_durations: dict[str, float] = field(default_factory=dict)
    # Set of model names whose fit crossed the slow-fit threshold during
    # this series; bumped into the per-(query, model) counter on the run
    # context.
    slow_fits: set[str] = field(default_factory=set)
    # Per-(regressor_id, reason) regressor-resolution failures recorded for
    # this series. Aggregated to (query_id, regressor_id, reason) on the
    # run context.
    regressor_failures: list[tuple[str, str]] = field(default_factory=list)
    # Per-(model, level) conformal calibration offset for this series.
    # Aggregated into ``calibration_offsets`` on the run context.
    calibration_offsets: list[CalibrationOffsetPoint] = field(default_factory=list)
    # Per-(model, level) conditional-band conformal calibration offset
    # for this series. Same shape as ``calibration_offsets`` but derived
    # from a calibration fit whose regressors are held flat at the final
    # pre-holdout value (matching the production-time conditional
    # inference contract). Aggregated into
    # ``conditional_calibration_offsets`` on the run context.
    conditional_calibration_offsets: list[ConditionalCalibrationOffsetPoint] = field(
        default_factory=list
    )
    # Capacity-planning samples. ETAs are per-(model, threshold[, level]);
    # growth rates are per-(model, window). Both are gauges (snapshot-style),
    # aggregated into the run context without monotonic merging.
    threshold_etas: list[ThresholdEtaPoint] = field(default_factory=list)
    growth_rates: list[GrowthRatePoint] = field(default_factory=list)
    # Per-(model, threshold[, level]) 0/1 predictive will-breach samples
    # derived from the ETAs above. Snapshot-style; folded straight into
    # the run context without monotonic merging.
    will_breach: list[WillBreachPoint] = field(default_factory=list)
    # Drain-ETA family for ``le_drain`` thresholds. Sibling of
    # ``threshold_etas`` — separate field rather than a re-used one so
    # the exporter's per-family rendering can preserve ``+Inf`` samples
    # (the "queue is not draining" alert needs them to round-trip).
    time_to_drain: list[TimeToDrainPoint] = field(default_factory=list)
    # SLO error-budget snapshot rows (remaining, burn rate, exhausted_in).
    # Snapshot-style; folded straight into the run context.
    error_budget: list[ErrorBudgetPoint] = field(default_factory=list)
    # Recommended maintenance windows — top-N lowest-load windows in the
    # forecast horizon for this (series, model). Snapshot-style.
    recommended_windows: list[RecommendedMaintenanceWindowPoint] = field(default_factory=list)
    # Conditional forecast emission: snapshot + deviation + curve. All
    # three are produced together when ``emission.conditional: true`` is
    # on for the query. The curve flows into the optional remote_write
    # sink alongside the unconditional curve.
    conditional_points: list[ConditionalForecastPoint] = field(default_factory=list)
    conditional_deviation_points: list[ConditionalDeviationPoint] = field(default_factory=list)
    conditional_curve: list[ForecastCurvePoint] = field(default_factory=list)
    # Component decomposition emission. Per-(series, model, component)
    # end-of-horizon snapshot rows for /metrics + per-step curve points
    # for the optional remote_write sink (consumed by the stacked-area
    # explainability dashboard).
    component_points: list[ComponentDecompositionPoint] = field(default_factory=list)
    component_curve: list[ForecastCurvePoint] = field(default_factory=list)
    # Per-reason conditional-fit failure counter. Bumped on every fit
    # that times out or raises; merged into ``ctx.failures`` so the
    # operator sees them under ``forecast_failures_total`` with a
    # bounded ``conditional_fit_*`` reason set.
    conditional_failures: dict[str, int] = field(default_factory=dict)
    # ``(model_name, reason)`` skip records the runner folds into the
    # exporter's per-(model, reason) decomposition-skipped counter.
    decomposition_skipped: list[tuple[str, str]] = field(default_factory=list)
    # Per-model forecast dataframes retained until after conformal
    # calibration so the capacity helpers can compute ETAs against the
    # *calibrated* bands (a calibration that widens a band should pull
    # the conservative ETA earlier or later by the same offset). Cleared
    # at the end of ``_fit_series`` so we don't keep multi-MB dataframes
    # alive in ``_RunContext``.
    capacity_inputs: list[_CapacityInput] = field(default_factory=list)
    # Imputed-fraction reported by preprocessing; threaded into the quality
    # score so heavily-imputed series score lower on the coverage dimension.
    imputed_fraction: float = 0.0
    # Count-aware safeguard counters: per-``kind`` increments folded
    # into the run context's per-(query, kind) total. ``kind`` is
    # ``negative`` (a value clipped from below zero) or ``rounded`` (a
    # non-trivial fractional component that was rounded to integer).
    emission_clipped: dict[str, int] = field(default_factory=dict)
    # Cold-start: number of usable siblings the borrow query produced for
    # this series; ``0`` when cold-start did not run.
    cold_start_siblings: int = 0
    # Forecast-drift gauges: one per (series, model). NaN when no prior
    # fit was cached for the (group, query, series, model) tuple.
    drift_points: list[DriftPoint] = field(default_factory=list)
    # Drift alert: how many threshold crossings happened for this query
    # in this run. The runner folds it into the per-(query) total.
    drift_alerts: int = 0
    # Forecast-to-forecast revision gauges: per (series, model, horizon)
    # magnitude of change between this fit's prediction and the prior
    # fit's prediction at the same horizon. NaN on the first fit (no
    # prior cached); the exporter drops NaN samples.
    revision_points: list[RevisionPoint] = field(default_factory=list)
    # Per-(series, horizon) model-disagreement gauge — coefficient of
    # variation across configured models' central forecasts at the same
    # horizon. Only populated in explicit mode with 2+ distinct emitted
    # models; auto-select / cold-start leave this empty.
    model_disagreement_points: list[ModelDisagreementPoint] = field(default_factory=list)


@dataclass(frozen=True)
class _BacktestNeeds:
    """Which downstream consumers want backtest data this run.

    Threaded through ``_run_backtests`` so it can skip post-processing the
    consumers don't want. The fit itself is unconditional once *any* consumer
    is interested — the per-model fit is the dominant cost; trimming the
    cheap MAPE/MASE evaluation or the trailing-band summary saves only a
    fraction of that. The bitmap mostly buys clarity (callers see exactly
    which fields they will read) and a small win when ``accuracy.evaluate: false``.
    """

    accuracy: bool  # forecast_accuracy_mape / mase samples emitted
    selection: bool  # auto-select needs overall_mape per model
    deviation: bool  # forecast_deviation_ratio / outside_band emitted
    quality: bool  # forecast_quality_score (uses results AND summary)

    @property
    def any(self) -> bool:
        return self.accuracy or self.selection or self.deviation or self.quality

    @property
    def needs_results(self) -> bool:
        """Per-horizon AccuracyResult list is needed by accuracy / selection / quality."""
        return self.accuracy or self.selection or self.quality

    @property
    def needs_summary(self) -> bool:
        """Trailing (actual, yhat, bands) summary is needed by deviation / quality."""
        return self.deviation or self.quality


@dataclass(frozen=True)
class ReloadResult:
    """Summary of what changed during ``Runner.apply_config``.

    Returned to the operator (e.g. via the ``/-/reload`` endpoint response)
    so they can confirm the diff matches their intent without diffing
    /metrics output by hand.
    """

    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)


@dataclass
class _RunContext:
    """Mutable accumulator threaded through ``_run_query`` calls.

    Group-level totals (points, drops, failures) used to live as locals in
    ``run_group``; pulling them into a struct keeps the per-query helper a
    method instead of a closure and lets tests inspect per-query state.
    """

    points: list[ForecastPoint] = field(default_factory=list)
    accuracy_points: list[AccuracyPoint] = field(default_factory=list)
    deviation_points: list[DeviationPoint] = field(default_factory=list)
    quality_points: list[QualityPoint] = field(default_factory=list)
    ensemble_weight_points: list[EnsembleWeightPoint] = field(default_factory=list)
    contribution_points: list[ContributionPoint] = field(default_factory=list)
    curve_points: list[ForecastCurvePoint] = field(default_factory=list)
    series_count: int = 0
    dropped: dict[str, int] = field(default_factory=dict)
    failures: dict[str, int] = field(default_factory=dict)
    global_remaining: int = 0
    # Most-recent fit duration per (query_id, model_name). Multiple series
    # under the same query keep overwriting each other intentionally — at
    # snapshot time we want a representative most-recent number, not a sum.
    fit_durations: dict[tuple[str, str], float] = field(default_factory=dict)
    # Cumulative slow-fit counter per (query_id, model_name) across the
    # series in this group run. Reset every run; the exporter merges with
    # the previous snapshot's counter to keep a monotonic counter view.
    slow_fits: dict[tuple[str, str], int] = field(default_factory=dict)
    # Cumulative regressor-resolution failure counter per
    # (query_id, regressor_id, reason). Merged with the previous snapshot
    # in ``run_group`` so the counter stays monotonic across runs.
    regressor_failures: dict[tuple[str, str, str], int] = field(default_factory=dict)
    # Per-series conformal calibration offsets accumulated this run.
    calibration_offsets: list[CalibrationOffsetPoint] = field(default_factory=list)
    # Per-series conditional-band conformal offsets accumulated this run.
    # Sibling of ``calibration_offsets`` — rendered as a parallel gauge
    # family so operators can read the per-(model, level) widening on the
    # conditional band specifically.
    conditional_calibration_offsets: list[ConditionalCalibrationOffsetPoint] = field(
        default_factory=list
    )
    # Capacity-planning families (ETA + growth rate). Snapshot-style;
    # only the current run's samples land on /metrics.
    threshold_etas: list[ThresholdEtaPoint] = field(default_factory=list)
    growth_rates: list[GrowthRatePoint] = field(default_factory=list)
    # Will-breach predictive boolean gauges (per series/model/threshold/level).
    will_breach_points: list[WillBreachPoint] = field(default_factory=list)
    # Drain-ETA points (queue burn-down). Snapshot-style; preserves
    # ``+Inf`` samples for the "queue is not draining" alert.
    time_to_drain_points: list[TimeToDrainPoint] = field(default_factory=list)
    # SLO error-budget snapshot rows accumulated this run.
    error_budget_points: list[ErrorBudgetPoint] = field(default_factory=list)
    # Recommended maintenance windows accumulated this run.
    recommended_maintenance_window_points: list[RecommendedMaintenanceWindowPoint] = field(
        default_factory=list
    )
    # Conditional forecast emission accumulators (snapshot rows +
    # per-(model, level) deviation gauges + per-step curve for the sink).
    conditional_points: list[ConditionalForecastPoint] = field(default_factory=list)
    conditional_deviation_points: list[ConditionalDeviationPoint] = field(default_factory=list)
    conditional_curve_points: list[ForecastCurvePoint] = field(default_factory=list)
    # Per-(series, model, component) end-of-horizon decomposition rows
    # + per-step curve for the sink.
    component_points: list[ComponentDecompositionPoint] = field(default_factory=list)
    component_curve_points: list[ForecastCurvePoint] = field(default_factory=list)
    # Per-query preprocessing statistics for the most recent run.
    preprocess_stats: dict[str, PreprocessStats] = field(default_factory=dict)
    # Per-(query, strategy) gap-fill count for this run; merged with the
    # previous snapshot so the counter stays monotonic.
    gaps_filled: dict[tuple[str, str], int] = field(default_factory=dict)
    # Per-(query, method) outlier replacement count for this run.
    outliers_replaced: dict[tuple[str, str], int] = field(default_factory=dict)
    # Per-query change-point detection count for this run.
    change_points: dict[str, int] = field(default_factory=dict)
    # Per-(query, reason) blacklist-window sample count for this run.
    blacklisted_samples: dict[tuple[str, str], int] = field(default_factory=dict)
    # Per-(query, kind) emission clipping counter for this run. Merged
    # with the previous snapshot in ``run_group`` so the exposed counter
    # stays monotonic across runs.
    emission_clipped: dict[tuple[str, str], int] = field(default_factory=dict)
    # Per-query usable-sibling count for the most recent cold-start fit.
    # The runner prefetches the borrow query once per primary query (see
    # ``_prefetch_cold_start_siblings``), so every cold-start fit within
    # one run of a given query observes the same usable-sibling count —
    # the gauge is deterministic per (group, query) regardless of which
    # short series of the query was processed last.
    cold_start_siblings: dict[str, int] = field(default_factory=dict)
    # Per-(series, model) drift gauges from this run. Same shape as
    # ``deviation_points`` — the renderer reads them directly.
    drift_points: list[DriftPoint] = field(default_factory=list)
    # Per-(query) drift threshold-crossing counter for this run.
    drift_alerts: dict[str, int] = field(default_factory=dict)
    # Per-(series, model, horizon) revision-magnitude gauges for this run.
    revision_points: list[RevisionPoint] = field(default_factory=list)
    # Per-(series, horizon) model-disagreement gauges for this run.
    model_disagreement_points: list[ModelDisagreementPoint] = field(default_factory=list)
    # Per-query series count and run duration for the most recent run.
    # Used to populate the ``query`` label on
    # ``forecast_series_count`` and ``forecast_run_duration_seconds``.
    series_count_per_query: dict[str, int] = field(default_factory=dict)
    run_duration_per_query: dict[str, float] = field(default_factory=dict)


def _merge_fit_into_ctx(ctx: _RunContext, fit_result: _FitResult, query_id: str) -> None:
    """Splice a ``_FitResult`` into the per-group ``_RunContext``.

    Pulled out of ``_run_query`` so that hot path stays readable; keeps the
    point-list extends and per-(query, model) counter bumps in one place.
    """
    ctx.points.extend(fit_result.points)
    ctx.accuracy_points.extend(fit_result.accuracy)
    ctx.deviation_points.extend(fit_result.deviation)
    ctx.quality_points.extend(fit_result.quality)
    ctx.ensemble_weight_points.extend(fit_result.ensemble)
    ctx.contribution_points.extend(fit_result.contribution)
    ctx.curve_points.extend(fit_result.curve)
    ctx.calibration_offsets.extend(fit_result.calibration_offsets)
    ctx.conditional_calibration_offsets.extend(fit_result.conditional_calibration_offsets)
    ctx.threshold_etas.extend(fit_result.threshold_etas)
    ctx.growth_rates.extend(fit_result.growth_rates)
    ctx.will_breach_points.extend(fit_result.will_breach)
    ctx.time_to_drain_points.extend(fit_result.time_to_drain)
    ctx.error_budget_points.extend(fit_result.error_budget)
    ctx.recommended_maintenance_window_points.extend(fit_result.recommended_windows)
    ctx.conditional_points.extend(fit_result.conditional_points)
    ctx.conditional_deviation_points.extend(fit_result.conditional_deviation_points)
    ctx.conditional_curve_points.extend(fit_result.conditional_curve)
    ctx.component_points.extend(fit_result.component_points)
    ctx.component_curve_points.extend(fit_result.component_curve)
    for reason, count in fit_result.conditional_failures.items():
        ctx.failures[reason] = ctx.failures.get(reason, 0) + count
    for model_name, fit_seconds in fit_result.fit_durations.items():
        ctx.fit_durations[(query_id, model_name)] = fit_seconds
    for model_name in fit_result.slow_fits:
        key = (query_id, model_name)
        ctx.slow_fits[key] = ctx.slow_fits.get(key, 0) + 1
    for regressor_id, reason in fit_result.regressor_failures:
        rkey = (query_id, regressor_id, reason)
        ctx.regressor_failures[rkey] = ctx.regressor_failures.get(rkey, 0) + 1
    for kind, count in fit_result.emission_clipped.items():
        ekey = (query_id, kind)
        ctx.emission_clipped[ekey] = ctx.emission_clipped.get(ekey, 0) + count
    if fit_result.cold_start_siblings > 0:
        ctx.cold_start_siblings[query_id] = fit_result.cold_start_siblings
    ctx.drift_points.extend(fit_result.drift_points)
    if fit_result.drift_alerts:
        ctx.drift_alerts[query_id] = ctx.drift_alerts.get(query_id, 0) + fit_result.drift_alerts
    ctx.revision_points.extend(fit_result.revision_points)
    ctx.model_disagreement_points.extend(fit_result.model_disagreement_points)


# Per-family inheritance registry. Each entry tells
# ``_inherit_prior_for_skipped_queries`` which prior-snapshot list to
# copy into the new context, keyed off either the point's ``id`` label
# or the query id encoded in its metric-name suffix. Adding a new
# emission family means appending one tuple here rather than editing a
# hand-coded copy block.
_KeyBy = Literal["label", "metric"]


@dataclass(frozen=True)
class _InheritanceSpec:
    """One per-family rule for ``_inherit_prior_for_skipped_queries``."""

    prior_attr: str
    ctx_attr: str
    by: _KeyBy


def _query_id_of(point: Any, by: _KeyBy) -> str:
    """Extract the query id from ``point`` per the spec's ``by`` mode."""
    if by == "label":
        labels = getattr(point, "labels", None)
        if not labels:
            return ""
        return str(labels.get("id", ""))
    return _query_id_from_metric(point.metric)


_INHERITANCE_REGISTRY: tuple[_InheritanceSpec, ...] = (
    # Lists keyed by the point's ``id`` label.
    _InheritanceSpec("accuracy_points", "accuracy_points", "label"),
    _InheritanceSpec("deviation_points", "deviation_points", "label"),
    _InheritanceSpec("quality_points", "quality_points", "label"),
    _InheritanceSpec("ensemble_weights", "ensemble_weight_points", "label"),
    _InheritanceSpec("calibration_offsets", "calibration_offsets", "label"),
    _InheritanceSpec("conditional_calibration_offsets", "conditional_calibration_offsets", "label"),
    _InheritanceSpec("drift_points", "drift_points", "label"),
    _InheritanceSpec("revision_points", "revision_points", "label"),
    _InheritanceSpec("model_disagreement_points", "model_disagreement_points", "label"),
    # ``forecast_recommended_maintenance_window`` is a global metric
    # name (no per-id prefix); use the ``id`` label to scope inheritance.
    _InheritanceSpec(
        "recommended_maintenance_window_points",
        "recommended_maintenance_window_points",
        "label",
    ),
    # Conditional deviation point exposes ``id`` as a label, so the
    # label-based extractor handles it correctly.
    _InheritanceSpec("conditional_deviation_points", "conditional_deviation_points", "label"),
    # Lists keyed by the metric-name suffix (which encodes the query id).
    _InheritanceSpec("points", "points", "metric"),
    _InheritanceSpec("contribution_points", "contribution_points", "metric"),
    _InheritanceSpec("threshold_etas", "threshold_etas", "metric"),
    _InheritanceSpec("growth_rates", "growth_rates", "metric"),
    _InheritanceSpec("will_breach_points", "will_breach_points", "metric"),
    _InheritanceSpec("time_to_drain_points", "time_to_drain_points", "metric"),
    _InheritanceSpec("error_budget_points", "error_budget_points", "metric"),
    _InheritanceSpec("conditional_points", "conditional_points", "metric"),
    _InheritanceSpec("component_points", "component_points", "metric"),
)


def _inherit_prior_for_skipped_queries(
    ctx: _RunContext,
    prior: GroupSnapshot,
    *,
    run_query_ids: set[str],
) -> None:
    """Preload ``ctx`` with prior-snapshot data for queries skipped this tick.

    With per-query refresh intervals, ``run_group`` may run only a
    subset of the group's queries on any given tick. Without this
    preload the snapshot would lose every other query's last-known
    forecast / accuracy / deviation rows the next time the partial
    snapshot is published. Counter-style fields (slow_fits, gaps_filled,
    ...) are already merged with the prior via the explicit
    counter-merge logic in :meth:`Runner.run_group`; here we only need
    to copy snapshot-style (gauge / list) data.

    ``run_query_ids`` is the set of query IDs that *will* run this
    tick; everything else gets its prior data inherited. ``curve_points``
    are intentionally *not* inherited because they drive the remote_write
    sink — pushing previously-fit curves every refresh would duplicate
    data in the TSDB.

    Per-family list inheritance is registry-driven via
    :data:`_INHERITANCE_REGISTRY`; the structured dict-state
    inheritance (per-query preprocessing stats, fit durations,
    cold-start siblings, series counts, run durations) is split into
    :func:`_inherit_prior_dict_state` because it's structurally
    different — keys are tuples or strings, values include mutable
    dataclasses that need defensive copying.
    """
    for spec in _INHERITANCE_REGISTRY:
        prior_list = getattr(prior, spec.prior_attr)
        ctx_list = getattr(ctx, spec.ctx_attr)
        for point in prior_list:
            if _query_id_of(point, spec.by) not in run_query_ids:
                ctx_list.append(point)
    _inherit_prior_dict_state(ctx, prior, run_query_ids=run_query_ids)


def _inherit_prior_dict_state(
    ctx: _RunContext,
    prior: GroupSnapshot,
    *,
    run_query_ids: set[str],
) -> None:
    """Carry forward per-(query, model) and per-query dict state.

    Unlike the list-based inheritance handled by the registry, these
    are dict-shaped gauge-style accumulators whose keys are tuples or
    strings rather than point objects. ``preprocess_stats`` carries a
    mutable :class:`PreprocessStats` so it's defensively copied — a
    follower serving ``/metrics`` would otherwise see torn state if a
    future writer mutated the prior snapshot mid-render.
    """
    for (q_id, model_name), fit_seconds in prior.fit_durations.items():
        if q_id not in run_query_ids:
            ctx.fit_durations[(q_id, model_name)] = fit_seconds
    for q_id, stats in prior.preprocess_stats.items():
        if q_id not in run_query_ids:
            ctx.preprocess_stats[q_id] = PreprocessStats(
                gaps_filled=dict(stats.gaps_filled),
                max_gap_seconds=stats.max_gap_seconds,
                outliers_replaced=dict(stats.outliers_replaced),
                change_points=stats.change_points,
                effective_lookback_seconds=stats.effective_lookback_seconds,
            )
    for q_id, count in prior.cold_start_siblings.items():
        if q_id not in run_query_ids:
            ctx.cold_start_siblings[q_id] = count
    for q_id, dur in prior.run_duration_per_query.items():
        if q_id not in run_query_ids:
            ctx.run_duration_per_query[q_id] = dur
    for q_id, scount in prior.series_count_per_query.items():
        if q_id not in run_query_ids:
            ctx.series_count_per_query[q_id] = scount
