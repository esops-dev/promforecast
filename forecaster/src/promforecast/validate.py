"""``promforecast validate`` — schema check + optional dry-run sample fit.

``validate`` is the first thing CI runs against a config PR. It must:

* Fail fast and clearly on bad YAML or schema errors (so the GitHub Actions
  log points at the offending config instead of a stack trace).
* Estimate output cardinality so a reviewer can spot a runaway ``models +
  confidence_levels`` combination *before* it ships to prod.
* Optionally run a sample query + fit (``--dry-run``) so the reviewer sees
  the first forecast points and per-model MAPE preview the change will
  produce. The sample is intentionally tiny — it's a smoke test, not a
  benchmark.

The function returns a process exit code so it composes cleanly with
``argparse`` in ``main.cli`` and with shell pipelines in CI.
"""

from __future__ import annotations

import asyncio
import math
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, TYPE_CHECKING

from . import config as config_module
from .accuracy import horizons_from_durations
from .config import effective_season_length, effective_step

if TYPE_CHECKING:
    from .config import Config
    from .source import SeriesFrame

Writer = Callable[[str], None]

# Minimum number of configured models required for the disagreement
# metric to project rows in the cardinality estimate; matches the
# runtime gate in :mod:`promforecast.runner`.
_MIN_MODELS_FOR_DISAGREEMENT_ESTIMATE = 2


@dataclass(frozen=True)
class CardinalityEstimate:
    """A coarse "what will this config emit?" number for review.

    The estimate is an upper bound assuming every query returns
    ``max_series_per_query`` series; users with low-cardinality queries will
    emit fewer, but the worst case is what trips the global cap.
    """

    series_per_query: int
    queries: int
    models: int
    confidence_levels: int
    forecast_lines: int
    accuracy_lines: int
    deviation_lines: int
    quality_lines: int
    contribution_lines: int
    calibration_lines: int
    preprocess_lines: int
    threshold_eta_lines: int
    will_breach_lines: int
    growth_rate_lines: int
    # Conditional emission: parallel forecast snapshot + deviation rows
    # for queries that opt into ``emission.conditional: true``.
    conditional_lines: int
    # Component decomposition: per-(series, model, component) rows for
    # queries that opt into ``emission.decompose: true``.
    decomposition_lines: int
    # Count-aware emission counter — per (query, kind=negative|rounded)
    # when a query opts in via ``data_profile: count`` or sets the
    # explicit emission toggles.
    emission_clipped_lines: int
    # Cold-start sibling gauge — per (query) when cold-start is
    # configured. Bounded; gauge counts as one line per query.
    cold_start_lines: int
    # Drift telemetry: per-(series, model) gauge plus per-(query)
    # alert counter for every query that opts into ``emission.drift``.
    drift_lines: int
    # Empirical band coverage: per-(series, model, level, window) rolling
    # gauge for every query that opts into ``emission.band_coverage``.
    band_coverage_lines: int
    # SLO error-budget snapshot rows (remaining + burn_rate + exhausted_in)
    # per (query, window). Only counted for queries that opt into ``slo:``.
    slo_lines: int
    # Recommended maintenance window rows — bounded by ``count`` per
    # opted-in query.
    recommended_window_lines: int
    # Deploy-anchored risk score rows. Per service, bounded by the
    # configured ``max_active_deploys`` cap — a worst-case upper
    # bound on simultaneously active deploys.
    deploy_risk_lines: int
    # Incident recurrence heatmap — fixed 7 (days) * 24 (hours) per
    # opted-in query with ``data_profile: events``.
    incident_recurrence_lines: int
    # Forecast-to-forecast revision: per-(series, model, horizon)
    # whenever ``accuracy.horizons`` is configured. Independent of
    # ``accuracy.evaluate`` because the metric reads the central
    # forecast at each horizon directly, no backtest needed.
    revision_lines: int
    # Model-disagreement: per-(series, horizon) in explicit mode only.
    # Same gate-shape as ``revision_lines``.
    disagreement_lines: int
    # Threshold action-cost: one config-derived gauge row per
    # threshold with ``action_cost`` configured. Independent of model
    # / series fan-out — the count is the total of opted-in
    # thresholds across the live config.
    action_cost_lines: int
    # Service-level forecast health composite: bounded by the unique
    # combinations of ``aggregate_by`` label values across the live
    # quality_points. We can't know that combinatoric without running,
    # so we estimate the upper bound at ``series_cap * queries`` (one
    # bucket per series in the worst case, when every series has a
    # distinct ``aggregate_by`` tuple) and let the operator narrow
    # via ``quality_floor`` if it overshoots.
    service_health_lines: int
    # Per-workload right-sizing: up to three rows per configured
    # workload (recommended + optional current + optional savings_pct).
    # Bounded by the operator's configured workload count; the actual
    # row count is smaller when ``current_promql`` is unset.
    resource_sizing_lines: int
    operational_lines: int

    @property
    def total(self) -> int:
        return (
            self.forecast_lines
            + self.accuracy_lines
            + self.deviation_lines
            + self.quality_lines
            + self.contribution_lines
            + self.calibration_lines
            + self.preprocess_lines
            + self.threshold_eta_lines
            + self.will_breach_lines
            + self.growth_rate_lines
            + self.conditional_lines
            + self.decomposition_lines
            + self.emission_clipped_lines
            + self.cold_start_lines
            + self.drift_lines
            + self.band_coverage_lines
            + self.slo_lines
            + self.recommended_window_lines
            + self.deploy_risk_lines
            + self.incident_recurrence_lines
            + self.revision_lines
            + self.disagreement_lines
            + self.action_cost_lines
            + self.service_health_lines
            + self.resource_sizing_lines
            + self.operational_lines
        )


def run_validate(  # noqa: PLR0911, PLR0915
    config_path: Path,
    *,
    dry_run: bool = False,
    datasource_url_override: str | None = None,
    strict_cardinality: bool = False,
    estimate_cost: bool = False,
    out: IO[str] | None = None,
) -> int:
    """Validate ``config_path`` and optionally run a sample fit.

    Returns 0 on success, 1 on validation failure (including
    over-budget cardinality in strict mode), 2 on dry-run failure.
    Output is plain text by default; ``out`` lets tests capture it.

    ``strict_cardinality`` flips the projected-total overshoot from an
    advisory ``WARN:`` line into a hard fail (exit code 1) so a CI job
    on a config PR can reject configs whose projected emission exceeds
    ``safety.max_total_series`` before the change ships. The knob is
    opt-in so existing callers keep their advisory behaviour.

    ``estimate_cost`` runs a cheap synthetic-data fit per (group,
    configured model) pair and prints the expected per-refresh CPU /
    fit-duration totals. The estimator is deliberately approximate
    (synthetic data doesn't capture every real-world fit-time
    distribution); the goal is order-of-magnitude headroom checks at
    review time rather than a benchmark.
    """
    stream: IO[str] = out if out is not None else sys.stdout

    def write(line: str) -> None:
        stream.write(line + "\n")

    def err(line: str) -> None:
        sys.stderr.write(line + "\n")

    try:
        cfg = config_module.load(config_path)
    except FileNotFoundError:
        err(f"error: config not found: {config_path}")
        return 1
    except Exception as exc:
        err(f"error: config validation failed: {exc}")
        return 1

    write(f"OK  config valid: {config_path}")
    write(f"    groups:  {len(cfg.groups)}")
    queries = sum(len(g.queries) for g in cfg.groups)
    write(f"    queries: {queries}")
    write(f"    models:  {', '.join(spec.name for spec in cfg.defaults.models)}")

    estimate = _estimate_cardinality(cfg)
    write("")
    write("Estimated cardinality (worst case at max_series_per_query):")
    write(
        f"  forecast lines     {estimate.forecast_lines:>8}  "
        f"= series({estimate.series_per_query}) * models({estimate.models}) "
        f"* (1 + 2*levels({estimate.confidence_levels})) * queries({estimate.queries})"
    )
    write(f"  accuracy lines     {estimate.accuracy_lines:>8}")
    write(f"  deviation lines    {estimate.deviation_lines:>8}")
    write(f"  quality lines      {estimate.quality_lines:>8}")
    write(f"  contribution lines {estimate.contribution_lines:>8}")
    write(f"  calibration lines  {estimate.calibration_lines:>8}")
    write(f"  preprocess lines   {estimate.preprocess_lines:>8}")
    write(f"  threshold ETA      {estimate.threshold_eta_lines:>8}")
    write(f"  will-breach lines  {estimate.will_breach_lines:>8}")
    write(f"  growth rate lines  {estimate.growth_rate_lines:>8}")
    write(f"  conditional lines  {estimate.conditional_lines:>8}")
    write(f"  decomposition lines {estimate.decomposition_lines:>8}")
    write(f"  emission clipped   {estimate.emission_clipped_lines:>8}")
    write(f"  cold start lines   {estimate.cold_start_lines:>8}")
    write(f"  drift lines        {estimate.drift_lines:>8}")
    write(f"  band coverage      {estimate.band_coverage_lines:>8}")
    write(f"  slo lines          {estimate.slo_lines:>8}")
    write(f"  recommended wins   {estimate.recommended_window_lines:>8}")
    write(f"  deploy risk lines  {estimate.deploy_risk_lines:>8}")
    write(f"  incident recur     {estimate.incident_recurrence_lines:>8}")
    write(f"  revision lines     {estimate.revision_lines:>8}")
    write(f"  disagreement lines {estimate.disagreement_lines:>8}")
    write(f"  action cost lines  {estimate.action_cost_lines:>8}")
    write(f"  service health     {estimate.service_health_lines:>8}")
    write(f"  resource sizing    {estimate.resource_sizing_lines:>8}")
    write(f"  operational lines  {estimate.operational_lines:>8}")
    write(f"  total              {estimate.total:>8}")
    rc = _handle_cardinality_overshoot(
        cfg=cfg,
        estimate=estimate,
        strict_cardinality=strict_cardinality,
        write=write,
        err=err,
    )
    if rc != 0:
        return rc

    if estimate_cost:
        write("")
        try:
            _run_cost_estimate(cfg, write)
        except Exception as exc:
            err(f"error: cost estimate failed: {exc}")
            return 2

    if not dry_run:
        return 0

    write("")
    write("Dry-run: running sample query + fit on first query of first group...")
    try:
        rc = asyncio.run(_dry_run(cfg, datasource_url_override, write))
    except Exception as exc:
        err(f"error: dry-run failed: {exc}")
        return 2
    return rc


def _handle_cardinality_overshoot(
    *,
    cfg: Config,
    estimate: CardinalityEstimate,
    strict_cardinality: bool,
    write: Writer,
    err: Writer,
) -> int:
    """Apply the per-mode policy when the estimate overshoots the cap.

    Returns ``0`` when the estimate fits inside ``safety.max_total_series``
    *or* when the legacy advisory-only mode is in effect (the caller emits
    the WARN line and continues). Returns ``1`` only in
    ``strict_cardinality`` mode when the projection exceeds the cap.

    Pulled out of :func:`run_validate` so the runtime statement count
    stays under ruff's PLR0915 threshold without hand-waving with a
    blanket noqa.
    """
    over_budget = bool(cfg.safety.max_total_series and estimate.total > cfg.safety.max_total_series)
    if not over_budget:
        return 0
    if strict_cardinality:
        err(
            f"error: projected total ({estimate.total}) exceeds "
            f"safety.max_total_series ({cfg.safety.max_total_series}); "
            "strict-cardinality mode rejects over-budget configs. "
            "Raise the cap, narrow your queries, or disable opt-in emission "
            "families. See docs/operations/cardinality.md."
        )
        return 1
    write(
        f"  WARN: total exceeds safety.max_total_series ({cfg.safety.max_total_series}); "
        "series_overflow will drop the excess"
    )
    return 0


def _estimate_cardinality(cfg: Config) -> CardinalityEstimate:  # noqa: PLR0912, PLR0915
    """Worst-case forecast/accuracy/deviation/quality/operational line counts.

    Lower-bounding (e.g. accounting for ``min_points`` drops) would require
    real data — the point of this estimate is to catch *configurations*
    that will trip the global cap, so we deliberately overstate.
    """
    queries = sum(len(g.queries) for g in cfg.groups)
    series_cap = cfg.safety.max_series_per_query
    n_models = max(1, len(cfg.defaults.models))
    n_levels = len(cfg.defaults.confidence_levels)
    n_horizons = max(1, len(cfg.defaults.accuracy.horizons))

    # Each forecast: 1 yhat row + 2 (lower/upper) per confidence level.
    per_series_forecast = n_models * (1 + 2 * n_levels)
    # Mode-specific output cardinality:
    #   * single auto-select: only the winner is emitted
    #   * ensemble: every component PLUS a synthetic ``ensemble`` model row
    #   * explicit: every configured model
    auto_select = cfg.defaults.accuracy.auto_select
    if auto_select and auto_select != "ensemble":
        per_series_forecast = 1 + 2 * n_levels
    elif auto_select == "ensemble":
        per_series_forecast = (n_models + 1) * (1 + 2 * n_levels)

    forecast_lines = queries * series_cap * per_series_forecast
    accuracy_lines = (
        queries * series_cap * n_models * n_horizons * 2  # MAPE + MASE
        if cfg.defaults.accuracy.evaluate
        else 0
    )
    # Deviation: one per (series, model, level) on the *emitted* models —
    # in auto-select mode that's 1 model; ensemble emits every component
    # (plus the synthetic "ensemble" row, which has no backtest data and
    # is dropped by the deviation helper); explicit emits every model.
    if auto_select and auto_select != "ensemble":
        emitted_models = 1
    elif auto_select == "ensemble":
        emitted_models = n_models  # synthetic "ensemble" row has no devation data
    else:
        emitted_models = n_models
    deviation_lines = (
        queries * series_cap * emitted_models * n_levels * 2  # ratio + outside_band
        if cfg.defaults.emission.deviation
        else 0
    )
    # Per-horizon segmentation: each (series, model) emits the aggregated
    # ``horizon=""`` row plus one row per configured accuracy horizon.
    # The aggregate row is always emitted (preserves the historical
    # ``horizon``-less shape so existing alerts continue to bind); the
    # per-horizon rows only show up when ``accuracy.evaluate: true``
    # produces backtest results.
    per_horizon_quality_factor = 1 + (n_horizons if cfg.defaults.accuracy.evaluate else 0)
    quality_lines = (
        queries * series_cap * emitted_models * per_horizon_quality_factor
        if cfg.defaults.emission.quality
        else 0
    )
    # Contribution: only emitted for queries that actually configure
    # regressors; the count is the *worst case* assuming every series of
    # those queries fans out one row per regressor. Calendar regressors are
    # included in the count because they're always present after alignment.
    contribution_lines = 0
    if cfg.defaults.emission.contribution:
        for group in cfg.groups:
            for query in group.queries:
                if query.regressors:
                    contribution_lines += series_cap * len(query.regressors)
    # Conformal calibration emits ``forecast_band_calibration_offset`` per
    # (series, model, level) on the *emitted* model set only — same shape
    # as the deviation gauge. Off when ``accuracy.conformal.enabled`` is
    # false.
    calibration_lines = 0
    if cfg.defaults.accuracy.conformal.enabled:
        calibration_lines = queries * series_cap * emitted_models * n_levels

    # Time-to-threshold ETAs: per (series, model, threshold) plus a per-level
    # band ETA for each threshold. Counted only against queries that
    # actually configure thresholds — most don't.
    threshold_eta_lines = 0
    # Predictive will-breach gauge: same shape as the ETA family above
    # but only for thresholds that opt into ``alert_within`` + ``severity``.
    # ``N_thresholds_with_alert * (N_levels + 1)`` per (series, model).
    will_breach_lines = 0
    for group in cfg.groups:
        for q in group.queries:
            if not q.thresholds:
                continue
            per_threshold_emissions = 1 + n_levels  # point + per-level band
            threshold_eta_lines += (
                series_cap * emitted_models * len(q.thresholds) * per_threshold_emissions
            )
            opted_in = sum(
                1 for t in q.thresholds if t.alert_within is not None and t.severity is not None
            )
            if opted_in:
                will_breach_lines += (
                    series_cap * emitted_models * opted_in * per_threshold_emissions
                )

    # Growth-rate: per (series, model, window) gauge. Counted only against
    # queries that opt into ``emission.growth_rate.enabled: true``.
    growth_rate_lines = 0
    # Conditional forecast: parallel emission family + per-(series, model,
    # level) deviation gauges. Charged only against queries that opt
    # into ``emission.conditional: true``.
    conditional_lines = 0
    # Decomposition: per-(series, model, component=trend|seasonal|level|residual)
    # — bounded to 4 lines per (series, model) per opted-in query.
    decomposition_lines = 0
    n_components = 4
    for group in cfg.groups:
        for q in group.queries:
            if q.emission.growth_rate.enabled:
                n_windows = max(1, len(q.emission.growth_rate.windows))
                growth_rate_lines += series_cap * emitted_models * n_windows
            if q.emission.conditional:
                # Per-series snapshot: central + 2 * levels rows per model.
                # Conditional deviation: 2 lines per (series, model, level)
                # (ratio + outside_band).
                conditional_lines += (
                    series_cap * emitted_models * (1 + 2 * n_levels)
                    + series_cap * emitted_models * n_levels * 2
                )
                # Conditional-band conformal calibration emits one
                # ``forecast_conditional_band_calibration_offset`` per
                # (series, model, level) on the *emitted* model set —
                # same shape as the unconditional ``calibration_lines``
                # term, just on the conditional half. Only charged when
                # ``accuracy.conformal.enabled`` is on; otherwise the
                # conditional emission still runs but no calibration
                # gauges land.
                if cfg.defaults.accuracy.conformal.enabled:
                    conditional_lines += series_cap * emitted_models * n_levels
            if q.emission.decompose:
                decomposition_lines += series_cap * emitted_models * n_components
    # Preprocessing emits per-(group, query) telemetry. The label sets
    # are all bounded:
    #   * max_gap_seconds         : 1 line per query
    #   * effective_lookback_secs : 1 line per query
    #   * change_point_total      : 1 line per query (when change_point.enabled)
    #   * gaps_filled_total       : at most |strategies| = 3 per query
    #   * outliers_replaced_total : at most |methods|    = 3 per query (when outliers.enabled)
    # The gauge pair (max gap + effective lookback) is always charged
    # because the runner emits them on every fit; the counters are only
    # charged when the feature is enabled in *any* group, so a config
    # that doesn't touch outliers/change-point doesn't inflate the
    # estimate.
    preprocess_lines = 2 * queries
    any_outliers_on = any(g.preprocess.outliers.enabled for g in cfg.groups)
    any_changepoint_on = any(g.preprocess.change_point.enabled for g in cfg.groups)
    n_gap_strategies = 3
    n_outlier_methods = 3
    preprocess_lines += queries * n_gap_strategies
    if any_outliers_on:
        preprocess_lines += queries * n_outlier_methods
    if any_changepoint_on:
        preprocess_lines += queries
    # Blacklist windows emit ``forecast_input_blacklisted_samples_total``
    # per (group, query, reason) where ``reason`` is bounded by the
    # operator's config (one distinct reason per configured window, plus
    # the empty-string reason for unreasoned windows). The per-query
    # multiplier is the count of distinct reasons across the group's
    # blacklist windows — accounted only on groups that actually
    # configured one.
    for group in cfg.groups:
        if not group.preprocess.blacklist_windows:
            continue
        distinct_reasons = {w.reason for w in group.preprocess.blacklist_windows}
        if distinct_reasons:
            preprocess_lines += len(group.queries) * len(distinct_reasons)

    # Count-aware emission clipping: a per-(query, kind=negative|rounded)
    # counter, bounded to 2 lines per query. Charged only against queries
    # that explicitly opt in (``emission.clip_non_negative`` /
    # ``round_to_integer`` set, or ``data_profile: count``) because most
    # configs leave the toggles off and produce zero samples.
    emission_clipped_lines = 0
    for group in cfg.groups:
        for q in group.queries:
            clip = q.emission.clip_non_negative
            rounder = q.emission.round_to_integer
            profile_enabled = q.data_profile == "count"
            on = (
                (clip is True)
                or (rounder is True)
                or (profile_enabled and (clip is not False) and (rounder is not False))
            )
            if on:
                # Two ``kind`` values (negative, rounded) per query.
                emission_clipped_lines += 2

    # Cold-start: ``forecast_cold_start_siblings{group, query}`` is a
    # single gauge per query that has cold-start configured.
    cold_start_lines = 0
    for group in cfg.groups:
        for q in group.queries:
            if q.cold_start.borrow_from is not None:
                cold_start_lines += 1

    # Drift telemetry. ``forecast_drift_score`` is per (series, model);
    # we count it against ``series_cap`` per opted-in query. The
    # ``forecast_drift_alerts_total`` counter is one line per opted-in
    # query.
    drift_lines = 0
    for group in cfg.groups:
        for q in group.queries:
            if not q.emission.drift.enabled:
                continue
            drift_lines += series_cap * emitted_models
            drift_lines += 1  # the per-(group, query) alert counter row

    # Revision magnitude: per-(series, model, horizon). The runtime
    # only requires ``accuracy.horizons`` to be configured — backtest
    # ``evaluate`` is *not* a gate, because the metric reads the
    # central forecast at each horizon directly, without needing
    # backtest results. First-fit rows are NaN and dropped at render,
    # so the worst case is the steady-state row count.
    revision_lines = queries * series_cap * emitted_models * n_horizons if n_horizons else 0
    # Model disagreement: per-(series, horizon). Only emitted in
    # explicit mode (auto_select == False) with 2+ configured models.
    # Same gate-shape as revision: needs configured horizons, not
    # ``accuracy.evaluate``. ``auto_select == "ensemble"`` counts as
    # active (only the ensemble row is emitted, so there's no
    # cross-model spread to score).
    auto_select_active = cfg.defaults.accuracy.auto_select is not False
    disagreement_lines = 0
    if n_horizons and not auto_select_active and n_models >= _MIN_MODELS_FOR_DISAGREEMENT_ESTIMATE:
        disagreement_lines = queries * series_cap * n_horizons

    # Empirical band coverage. ``forecast_band_empirical_coverage`` is
    # per (series, model, level, window) for opted-in queries.
    band_coverage_lines = 0
    for group in cfg.groups:
        for q in group.queries:
            if not q.emission.band_coverage.enabled:
                continue
            n_windows = max(1, len(q.emission.band_coverage.windows))
            band_coverage_lines += series_cap * emitted_models * n_levels * n_windows

    # SLO error-budget projections. Three sibling metrics — remaining,
    # burn_rate, exhausted_in — per (series, model, window). The
    # exhausted_in row has an additional fan-out by level (central row +
    # one per level), so we charge ``2 + 1 + N_levels = 3 + N_levels``
    # rows per (series, model, window). Bounded; only counted for
    # queries that opt into ``slo:``.
    slo_lines = 0
    for group in cfg.groups:
        for q in group.queries:
            if q.slo is None:
                continue
            n_windows = max(1, len(q.slo.windows))
            slo_lines += series_cap * emitted_models * n_windows * (3 + n_levels)

    # Recommended maintenance windows — ``count`` rows per (series,
    # model) for queries that opt in. Bounded by construction.
    recommended_window_lines = 0
    for group in cfg.groups:
        for q in group.queries:
            if not q.recommended_windows.enabled:
                continue
            recommended_window_lines += series_cap * emitted_models * q.recommended_windows.count

    # Deploy-anchored risk: one row per (service, active deploy)
    # bounded by ``max_active_deploys``. The label set adds
    # ``contributing_signals`` so each row is distinct even when two
    # deploys collide on the same service.
    deploy_risk_lines = sum(entry.max_active_deploys for entry in cfg.deploy_signals)

    # Incident recurrence heatmap: 7 days * 24 hours per opted-in
    # query (queries with ``data_profile: events``).
    incident_recurrence_lines = 0
    event_buckets = 7 * 24
    for group in cfg.groups:
        for q in group.queries:
            if q.data_profile == "events":
                incident_recurrence_lines += event_buckets

    # Threshold action-cost: one config-derived line per threshold
    # with ``action_cost`` configured. Bounded by the operator's
    # configured threshold count; no model / series fan-out.
    action_cost_lines = 0
    for group in cfg.groups:
        for q in group.queries:
            action_cost_lines += sum(1 for t in q.thresholds if t.action_cost is not None)

    # Service-level forecast health composite. Worst-case upper bound:
    # one bucket per quality_point (which is per (series, model,
    # horizon)). The realistic count is far smaller (operators
    # aggregate by 1-2 labels with low cardinality) but we can't
    # introspect the live label values from config alone.
    service_health_lines = 0
    if cfg.governance.health_score is not None:
        # ``per_horizon_quality_factor`` was computed earlier — one
        # quality row per emitted (model, horizon) per series. Upper
        # bound is the quality_lines total itself.
        service_health_lines = quality_lines

    # Right-sizing: up to three rows per configured workload entry —
    # the primary recommended gauge plus the optional current /
    # savings_pct siblings (only emitted when ``current_promql`` is set
    # on the entry). Bounded by the operator's configured workload count.
    resource_sizing_lines = 0
    rs_cfg = cfg.recommendations.resource_sizing
    if rs_cfg.enabled:
        for workload in rs_cfg.workloads:
            # One row for the recommended gauge plus up to two more
            # for the optional sibling pair.
            resource_sizing_lines += 1
            if workload.current_promql is not None:
                resource_sizing_lines += 2  # current + savings_pct

    # Operational: 7 gauges + N reasons per group; we ballpark 7 per group.
    # The runner also emits per-(group, query) rows of
    # ``forecast_query_run_duration_seconds`` and
    # ``forecast_query_series_count`` for every query (not just queries
    # that override ``refresh_interval``), so we add two extra lines per
    # query unconditionally.
    operational_lines = 7 * len(cfg.groups) + 2 * queries

    return CardinalityEstimate(
        series_per_query=series_cap,
        queries=queries,
        models=n_models,
        confidence_levels=n_levels,
        forecast_lines=forecast_lines,
        accuracy_lines=accuracy_lines,
        deviation_lines=deviation_lines,
        quality_lines=quality_lines,
        contribution_lines=contribution_lines,
        calibration_lines=calibration_lines,
        preprocess_lines=preprocess_lines,
        threshold_eta_lines=threshold_eta_lines,
        will_breach_lines=will_breach_lines,
        growth_rate_lines=growth_rate_lines,
        conditional_lines=conditional_lines,
        decomposition_lines=decomposition_lines,
        emission_clipped_lines=emission_clipped_lines,
        cold_start_lines=cold_start_lines,
        drift_lines=drift_lines,
        band_coverage_lines=band_coverage_lines,
        slo_lines=slo_lines,
        recommended_window_lines=recommended_window_lines,
        deploy_risk_lines=deploy_risk_lines,
        incident_recurrence_lines=incident_recurrence_lines,
        revision_lines=revision_lines,
        disagreement_lines=disagreement_lines,
        action_cost_lines=action_cost_lines,
        service_health_lines=service_health_lines,
        resource_sizing_lines=resource_sizing_lines,
        operational_lines=operational_lines,
    )


async def _dry_run(
    cfg: Config,
    datasource_url_override: str | None,
    write: Writer,
) -> int:
    """Sample-query the configured datasource and run a single fit.

    Picks the first query of the first group, queries the configured
    lookback window, runs the first configured model on the first returned
    series, and prints the leading 10 forecast points + a tiny backtest
    preview. Failures are surfaced as a non-zero return code so CI can fail
    on a broken sample without forcing the operator to read logs.
    """
    from .source import PromSource  # noqa: PLC0415

    if not cfg.groups or not cfg.groups[0].queries:
        write("    (no queries to dry-run)")
        return 0

    group = cfg.groups[0]
    query = group.queries[0]
    url = datasource_url_override or cfg.datasource.url

    write(f"  group:  {group.name}")
    write(f"  query:  {query.id}  ({query.promql.strip()[:60]}...)")
    write(f"  source: {url}")

    source = PromSource(
        url=url,
        timeout_seconds=cfg.datasource.timeout.total_seconds(),
        max_concurrency=cfg.datasource.max_concurrent_requests,
        max_keepalive_connections=cfg.datasource.max_keepalive_connections,
    )
    try:
        end = datetime.now(tz=UTC)
        start = end - cfg.defaults.lookback
        sample_step = effective_step(query, cfg.defaults)
        step_seconds = max(1, int(sample_step.total_seconds()))
        t0 = time.monotonic()
        try:
            frames = await source.query_range(query.promql, start, end, step_seconds)
        except Exception as exc:
            write(f"    FAIL  query failed: {exc}")
            return 2
        query_seconds = time.monotonic() - t0

        write(f"    query returned {len(frames)} series in {query_seconds:.2f}s")
        if not frames:
            write("    (empty result — nothing to fit)")
            return 0
        frame = frames[0]
        if len(frame.values) < cfg.defaults.min_points:
            write(
                f"    WARN: first series has {len(frame.values)} points; "
                f"min_points is {cfg.defaults.min_points} — would be dropped at runtime"
            )

        fit_rc = _fit_sample(cfg, frame, write)
        return fit_rc
    finally:
        await source.aclose()


def _fit_sample(cfg: Config, frame: SeriesFrame, write: Writer) -> int:
    """Fit the first configured model and print a 10-line forecast preview.

    Imports of heavy modelling deps are lazy so ``promforecast validate``
    on a config-only PR doesn't pay the import cost when ``--dry-run`` is
    not requested.

    The first query of the first group's per-query ``step`` /
    ``season_length`` overrides apply here too — otherwise dry-running a
    multi-frequency config would silently fit at the default cadence and
    hand the operator misleading numbers.
    """
    import pandas as pd  # noqa: PLC0415

    from . import models as model_registry  # noqa: PLC0415

    spec = cfg.defaults.models[0]
    model_name = spec.name
    write(f"    fitting first model: {model_name}")

    query = cfg.groups[0].queries[0]
    sample_step = effective_step(query, cfg.defaults)
    sample_season_length = effective_season_length(query, cfg.defaults, step=sample_step)

    df = pd.DataFrame(
        {
            "unique_id": ["s"] * len(frame.values),
            "ds": pd.to_datetime(frame.timestamps),
            "y": frame.values,
        }
    )
    freq = _pandas_freq_seconds(int(sample_step.total_seconds()))
    horizon = max(
        1,
        int(cfg.defaults.horizon.total_seconds() // sample_step.total_seconds()),
    )
    levels = list(cfg.defaults.confidence_levels)
    try:
        model = model_registry.build(
            model_name, season_length=sample_season_length, params=spec.params
        )
        t0 = time.monotonic()
        forecast = model.fit_predict(df=df, horizon=horizon, freq=freq, levels=levels)
        fit_seconds = time.monotonic() - t0
    except Exception as exc:
        write(f"    FAIL  fit failed: {exc}")
        return 2

    write(f"    fit completed in {fit_seconds:.2f}s, {len(forecast)} forecast rows")
    write("    first 10 forecast lines:")
    head = forecast.head(10)
    for _, row in head.iterrows():
        ts = row.get("ds")
        yhat = row.get("yhat")
        yhat_str = f"{yhat:.4f}" if yhat is not None and math.isfinite(float(yhat)) else "NaN"
        write(f"      {ts}  yhat={yhat_str}")

    horizons = horizons_from_durations(list(cfg.defaults.accuracy.horizons), sample_step)
    if horizons and len(df) > horizons[0].steps + 1:
        rc = _backtest_preview(model, df, horizons[0].steps, freq, levels, write)
        return rc
    return 0


def _backtest_preview(
    model: object,
    df: object,
    horizon_steps: int,
    freq: str,
    levels: list[int],
    write: Writer,
) -> int:
    """Print a single-horizon MAPE preview from the smallest accuracy horizon.

    A full multi-horizon backtest would multiply the ``--dry-run`` runtime;
    a single horizon is enough to tell "the fit produced sensible numbers"
    from "the fit produced garbage".
    """
    from .accuracy import _mape, backtest  # noqa: PLC0415

    try:
        result = backtest(
            model=model,  # type: ignore[arg-type]
            df=df,  # type: ignore[arg-type]
            horizon_steps=horizon_steps,
            freq=freq,
            levels=levels,
        )
    except Exception as exc:
        write(f"    backtest skipped: {exc}")
        return 0
    if result is None:
        write("    backtest skipped: series too short for a hold-out split")
        return 0
    test, predictions = result
    n = min(len(test), len(predictions))
    if n < 1:
        write("    backtest skipped: empty hold-out")
        return 0
    y_true = test["y"].to_numpy(dtype=float)[:n]
    y_pred = predictions["yhat"].to_numpy(dtype=float)[:n]
    mape = _mape(y_true, y_pred)
    write(f"    backtest preview: MAPE={mape:.2f}% over {n} hold-out points")
    return 0


def _pandas_freq_seconds(seconds: int) -> str:
    if seconds % 3600 == 0:
        return f"{seconds // 3600}h"
    if seconds % 60 == 0:
        return f"{seconds // 60}min"
    return f"{seconds}s"


@dataclass(frozen=True)
class CostEstimate:
    """One group's projected per-refresh resource budget.

    All numbers are best-effort: the per-(group, model) fit time is
    measured against a synthetic sinusoid + trend series, which is
    smoother than typical production input but captures the AutoARIMA
    /AutoETS / SeasonalNaive cost order-of-magnitude. The estimator's
    purpose is to catch *config-level* surprises (e.g. AutoARIMA on
    14 days of 5-minute data over 500 series) at review time, not to
    benchmark a deployment.
    """

    group: str
    refresh_interval_seconds: float
    estimated_series: int
    per_fit_seconds_by_model: dict[str, float]
    total_fit_seconds_per_refresh: float
    peak_memory_bytes: int


def _run_cost_estimate(cfg: Config, write: Writer) -> None:
    """Print a per-group fit-time + memory budget."""
    write("Estimated runtime cost (synthetic-data fit per model):")
    write(
        "  Note: estimator is order-of-magnitude only — synthetic input doesn't "
        "capture every real fit-time shape. Treat as a sanity check, not a benchmark."
    )
    write("")
    estimates = list(_estimate_runtime_cost(cfg))
    if not estimates:
        write("  (no groups to estimate)")
        return
    total_seconds = 0.0
    for est in estimates:
        write(f"  group {est.group}:")
        write(f"    refresh_interval         {est.refresh_interval_seconds:>10.1f}s")
        write(f"    estimated series         {est.estimated_series:>10d}")
        for model_name, seconds in sorted(est.per_fit_seconds_by_model.items()):
            write(f"    per-fit {model_name:<20}{seconds:>10.3f}s")
        write(f"    total fit time / refresh {est.total_fit_seconds_per_refresh:>10.2f}s")
        write(f"    peak memory (tracemalloc){est.peak_memory_bytes:>10d} bytes")
        if (
            est.refresh_interval_seconds > 0
            and est.total_fit_seconds_per_refresh > est.refresh_interval_seconds
        ):
            write(
                "    WARN: projected fit time exceeds refresh_interval — "
                "the group cannot keep up at the configured cadence."
            )
        total_seconds += est.total_fit_seconds_per_refresh
    write("")
    write(f"  total fit time across groups (per refresh): {total_seconds:.2f}s")


def _estimate_runtime_cost(cfg: Config) -> list[CostEstimate]:
    """Time each configured model against synthetic data; project per-refresh totals.

    ``defaults.models`` is config-global, so each model is fit exactly
    once against the synthetic data and the resulting timings are reused
    across every group. Without this, an N-group x M-model config would
    run NxM identical fits where M would suffice — turning a 30-second
    validate into multiple minutes on realistic configs.

    Per-fit peak memory is captured via :mod:`tracemalloc` (Python's
    allocator-level peak), which gives an honest per-block number
    instead of the process-wide RSS that monotonically grows across
    fits and makes the second group's "peak" always meaningless.
    """
    import tracemalloc  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415

    from . import models as model_registry  # noqa: PLC0415

    series_cap = cfg.safety.max_series_per_query
    # Synthetic dataset: 720 hourly points (~30d) — enough for AutoARIMA
    # to do real work but small enough to keep the estimator cheap. The
    # signal is a daily sinusoid + linear trend + light noise so the
    # statsforecast fit isn't trivially degenerate.
    n_points = 720
    rng = np.random.default_rng(seed=42)
    timestamps = pd.date_range("2024-01-01", periods=n_points, freq="1h")
    base = (
        np.sin(np.arange(n_points) * 2 * math.pi / 24) * 10.0
        + np.arange(n_points) * 0.01
        + rng.normal(0, 0.5, n_points)
        + 100.0
    )
    df = pd.DataFrame({"unique_id": ["s"] * n_points, "ds": timestamps, "y": base})
    levels = list(cfg.defaults.confidence_levels)
    horizon = 24
    freq = "1h"

    # Fit each model once and capture the per-fit wall-clock and the
    # tracemalloc peak attributable to that fit alone.
    per_fit_times: dict[str, float] = {}
    per_fit_memory: dict[str, int] = {}
    tracemalloc.start()
    try:
        for spec in cfg.defaults.models:
            try:
                model = model_registry.build(
                    spec.name,
                    season_length=cfg.defaults.season_length,
                    params=spec.params,
                )
            except Exception:
                # Model wasn't registered (typo, missing plug-in). Skip
                # without crashing — validate also reports config-level
                # errors elsewhere.
                continue
            tracemalloc.reset_peak()
            t0 = time.monotonic()
            try:
                _ = model.fit_predict(df=df, horizon=horizon, freq=freq, levels=levels)
            except Exception:
                # Real fit failures are noisy — store NaN so the
                # operator sees "couldn't measure" rather than a
                # spurious zero.
                per_fit_times[spec.name] = float("nan")
                per_fit_memory[spec.name] = 0
                continue
            per_fit_times[spec.name] = time.monotonic() - t0
            _, peak = tracemalloc.get_traced_memory()
            per_fit_memory[spec.name] = int(peak)
    finally:
        tracemalloc.stop()

    out: list[CostEstimate] = []
    for group in cfg.groups:
        # Refresh interval can come from the per-query override; if no
        # queries exist we fall back to the server default.
        if group.queries:
            refresh_interval = config_module.effective_refresh_interval(
                group.queries[0], cfg.server
            ).total_seconds()
        else:
            refresh_interval = cfg.server.refresh_interval.total_seconds()
        # Each query fits every configured model once per series; in
        # auto-select mode only the winner emits but every candidate is
        # still backtested. The estimate sums per-fit cost across
        # configured models and projects to the per-query series cap.
        per_query_fit_time = sum(v for v in per_fit_times.values() if math.isfinite(v))
        queries = len(group.queries)
        total = per_query_fit_time * queries * series_cap
        # Peak memory attributable to a fit in this group: the max
        # per-fit allocation peak across the configured models. Since
        # ``defaults.models`` is global, the per-model peak is the same
        # in every group, but the field shape preserves the per-group
        # report so a downstream UI can render it per row.
        peak_memory = max(per_fit_memory.values(), default=0)
        out.append(
            CostEstimate(
                group=group.name,
                refresh_interval_seconds=refresh_interval,
                estimated_series=series_cap * queries,
                per_fit_seconds_by_model=dict(per_fit_times),
                total_fit_seconds_per_refresh=total,
                peak_memory_bytes=peak_memory,
            )
        )
    return out
