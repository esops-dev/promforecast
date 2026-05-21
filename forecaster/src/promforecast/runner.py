"""Per-group execution and scheduling.

Each group runs as a long-lived asyncio task that fits, exports, and sleeps
for ``server.refresh_interval``. A per-group lock prevents a slow run from
being triggered twice concurrently. Failure isolation is layered: per query,
per series, per regressor.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import random
import time
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Literal

import structlog

from . import conformal as conformal_mod
from . import contribution as contribution_mod
from . import discovery as discovery_mod
from . import models as model_registry
from . import preprocess as preprocess_mod
from . import regressors as regressors_mod
from . import seasonality as seasonality_mod
from . import telemetry as telemetry_mod
from .accuracy import (
    AccuracyResult,
    HorizonSpec,
    backtest,
    evaluate,
    horizons_from_durations,
    overall_mape,
)
from .config import (
    CanaryConfig,
    GroupConfig,
    ModelSpec,
    effective_season_length,
    effective_step,
)
from .exporter import (
    AccuracyPoint,
    CalibrationOffsetPoint,
    ContributionPoint,
    DeviationPoint,
    EnsembleWeightPoint,
    Exporter,
    ForecastPoint,
    GroupSnapshot,
    InvalidMetricNameError,
    PreprocessStats,
    QualityPoint,
    validate_metric_name,
)
from .point_factory import BacktestOutcome, BandSummary, SeriesPointFactory
from .selector import ensemble_weights, select
from .sink import ForecastCurvePoint, ForecastSink, SinkError
from .source import PromSource, SeriesFrame

if TYPE_CHECKING:
    import pandas as pd

    from .config import Config, QueryConfig
    from .models import ForecastModel

logger = structlog.get_logger(__name__)


OverflowStrategy = Literal["drop_lowest_priority", "reject", "sample"]

# A fit is considered "slow" once its wall-clock crosses this fraction of
# ``safety.fit_timeout``. We pick 0.5 (half) so the warning fires comfortably
# before the timeout actually kills the fit - long enough that operators
# can observe a degraded model before alerts go red, short enough that
# noise-induced 5-10% wobble doesn't trip it.
SLOW_FIT_RATIO = 0.5


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
    # Imputed-fraction reported by preprocessing; threaded into the quality
    # score so heavily-imputed series score lower on the coverage dimension.
    imputed_fraction: float = 0.0


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
    # Per-query preprocessing statistics for the most recent run.
    preprocess_stats: dict[str, PreprocessStats] = field(default_factory=dict)
    # Per-(query, strategy) gap-fill count for this run; merged with the
    # previous snapshot so the counter stays monotonic.
    gaps_filled: dict[tuple[str, str], int] = field(default_factory=dict)
    # Per-(query, method) outlier replacement count for this run.
    outliers_replaced: dict[tuple[str, str], int] = field(default_factory=dict)
    # Per-query change-point detection count for this run.
    change_points: dict[str, int] = field(default_factory=dict)


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
    ) -> None:
        self._config = config
        self._source = source
        self._exporter = exporter
        self._executor = executor or ThreadPoolExecutor(max_workers=4)
        self._sink = sink
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
        interval = self._config.server.refresh_interval.total_seconds()
        for group in self._config.groups:
            self._spawn_group_loop(group, interval)

    def _spawn_group_loop(self, group: GroupConfig, interval_seconds: float) -> None:
        """Create the stop event + task for one group. Idempotent per name."""
        ev = asyncio.Event()
        self._stop_events[group.name] = ev
        self._tasks[group.name] = asyncio.create_task(self._loop(group, interval_seconds, ev))

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

            interval = new_config.server.refresh_interval.total_seconds()
            for group in new_config.groups:
                self._spawn_group_loop(group, interval)

            return ReloadResult(added=added, removed=removed, changed=changed)

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
        interval_seconds: float,
        stop_event: asyncio.Event,
    ) -> None:
        while not stop_event.is_set() and not self._stopped.is_set():
            try:
                await self.run_group(group)
            except asyncio.CancelledError:
                # Hard cancellation during shutdown / drain timeout: let it
                # propagate so ``_drain_all_groups`` and ``stop`` can finish.
                raise
            except Exception:
                logger.exception("group_run_failed", group=group.name)
                self._exporter.increment_failure(group.name, reason="group_error")
            await self._sleep_until_stopped_or(stop_event, interval_seconds)

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

    async def run_group(self, group: GroupConfig) -> None:
        """Execute a single group: query, fit, select, export."""
        async with self._locks[group.name]:
            started = time.monotonic()
            log = logger.bind(group=group.name, query_count=len(group.queries))
            log.info("group_run_start")

            ctx = _RunContext(global_remaining=self._global_budgets.get(group.name, 0))
            with self._tracer.span(
                "promforecast.group_run",
                group=group.name,
                query_count=len(group.queries),
            ):
                for query in group.queries:
                    await self._run_query(query, group, ctx, log)

            duration = time.monotonic() - started
            # Counter semantics: merge this run's slow_fits with the previous
            # snapshot's so the exposed counter never decreases. Without the
            # merge each run would reset to 0 — Prometheus would treat that
            # as a counter-reset and rate() would lie. We *prune* the merge
            # to (query, model) tuples whose query still exists in the
            # current group config; otherwise a hot-reload that drops a
            # query would leak its old slow_fits label set forever.
            previous_snapshot = self._exporter.snapshot_for(group.name)
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
            self._exporter.update(
                GroupSnapshot(
                    group=group.name,
                    points=ctx.points,
                    accuracy_points=ctx.accuracy_points,
                    deviation_points=ctx.deviation_points,
                    quality_points=ctx.quality_points,
                    ensemble_weights=ctx.ensemble_weight_points,
                    contribution_points=ctx.contribution_points,
                    last_run_timestamp=time.time(),
                    run_duration_seconds=duration,
                    series_count=ctx.series_count,
                    series_dropped=ctx.dropped,
                    failures=ctx.failures,
                    regressor_failures=merged_regressor_failures,
                    # ``fit_durations`` is a gauge — only this run's samples
                    # belong on /metrics. Removed queries simply disappear
                    # from the exposition until they reappear in config.
                    fit_durations=ctx.fit_durations,
                    slow_fits=merged_slow,
                    calibration_offsets=ctx.calibration_offsets,
                    preprocess_stats=ctx.preprocess_stats,
                    gaps_filled_cumulative=merged_gaps,
                    outliers_replaced_cumulative=merged_outliers,
                    change_points_cumulative=merged_changepoints,
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
            if self._sink is not None and ctx.curve_points:

                def _record_sink_retry(reason: str) -> None:
                    self._exporter.increment_remote_write_retry(group.name, reason)

                try:
                    await self._sink.write(ctx.curve_points, on_retry=_record_sink_retry)
                except SinkError as exc:
                    log.warning(
                        "remote_write_failed",
                        reason=exc.reason,
                        error=str(exc),
                        points=len(ctx.curve_points),
                    )
                    self._exporter.increment_remote_write_failure(group.name, exc.reason)
                except Exception as exc:
                    log.exception("remote_write_unexpected_error")
                    self._exporter.increment_remote_write_failure(group.name, "unexpected")
                    del exc  # logged via exception(); avoid unused-name lint

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

    async def _do_run_query(
        self,
        query: QueryConfig,
        group: GroupConfig,
        ctx: _RunContext,
        qlog: structlog.stdlib.BoundLogger,
    ) -> None:
        frames = await self._discover_and_fetch(query, group, ctx, qlog)
        if frames is None:
            return

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

        min_points = self._config.defaults.min_points
        eff_step = effective_step(query, self._config.defaults)
        query_stats = ctx.preprocess_stats.setdefault(query.id, PreprocessStats())
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
                query_stats.max_gap_seconds = max(
                    query_stats.max_gap_seconds, cleaned.max_gap_seconds
                )
                continue
            self._merge_preprocess_stats(
                ctx=ctx,
                query_id=query.id,
                cleaned=cleaned,
                outlier_method=group.preprocess.outliers.method,
            )
            if len(cleaned.values) < min_points:
                ctx.dropped["min_points"] = ctx.dropped.get("min_points", 0) + 1
                continue
            preprocessed_frame = SeriesFrame(
                labels=frame.labels,
                timestamps=cleaned.timestamps,
                values=cleaned.values,
            )
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
        return False

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

    async def _fit_series(
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
        if d.accuracy.conformal.enabled and forecast_points:
            forecast_points = await self._apply_conformal_calibration(
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
            result.quality = list(
                factory.quality_points(
                    backtests,
                    emitted_models,
                    frame.values,
                    coverage,
                    imputed_fraction=imputed_fraction,
                )
            )

        if d.emission.contribution and regressors_frame is not None:
            contributions = contribution_mod.compute_contributions(
                y=list(frame.values),
                regressors=regressors_frame,
            )
            if contributions:
                result.contribution = list(factory.contribution_points(contributions))

        return result

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
        for outcome in outcomes:
            if isinstance(outcome, BaseException):
                logger.exception(
                    "fit_failed_in_explicit_mode",
                    group=group_name,
                    query=query.id,
                    exc_info=outcome,
                )
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
            out.extend(factory.forecast_points(forecast, model_name, selection=None))
            result.curve.extend(factory.forecast_curve_points(forecast, model_name, selection=None))
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
        backtest_mape = {
            model_name: mape
            for model_name, results in backtest_results.items()
            if (mape := overall_mape(results)) is not None
        }
        selection = select(
            fallback_model=d.accuracy.fallback_model,
            fallback_mape_threshold=d.accuracy.fallback_mape_threshold,
            backtest_mape=backtest_mape,
        )
        # Look up params for the chosen model. The fallback model may live
        # outside ``defaults.models`` (e.g. SeasonalNaive used as the safety
        # net for AutoARIMA-only setups); ``_resolve_spec`` returns an empty
        # params dict in that case so the fallback runs with stock defaults.
        spec = _resolve_spec(d.models, selection.model)
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
        fit_seconds = time.monotonic() - t0
        result.fit_durations[selection.model] = fit_seconds
        self._record_slow_fit_if_needed(
            result=result,
            group=group_name,
            query_id=query.id,
            model_name=selection.model,
            fit_seconds=fit_seconds,
        )
        result.curve.extend(
            factory.forecast_curve_points(forecast, selection.model, selection=selection)
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
        for outcome in outcomes:
            if isinstance(outcome, BaseException):
                logger.exception(
                    "fit_failed_in_ensemble_mode",
                    group=group_name,
                    query=query.id,
                    exc_info=outcome,
                )
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
            component_forecasts[model_name] = forecast
            out.extend(factory.forecast_points(forecast, model_name, selection=None))
            result.curve.extend(factory.forecast_curve_points(forecast, model_name, selection=None))

        if not component_forecasts:
            return out

        # Compute inverse-MAPE weights restricted to models that produced a
        # forecast this run; otherwise a backtested-but-failed-to-fit model
        # would leak into the ensemble's denominator.
        scoreable_mape = {
            name: mape
            for name, results in backtest_results.items()
            if name in component_forecasts and (mape := overall_mape(results)) is not None
        }
        weights = ensemble_weights(scoreable_mape) or {
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
        threshold = timeout_seconds * SLOW_FIT_RATIO
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
                summary = _summarise_backtest(predictions=predictions, test=test, levels=levels)
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
    ) -> list[ForecastPoint]:
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
            return forecast_points
        adjusted_points = _apply_offsets_to_snapshot_points(forecast_points, offsets_by_model)
        if result.curve:
            result.curve = _apply_offsets_to_curve_points(result.curve, offsets_by_model)
        for model_name, offsets in offsets_by_model.items():
            result.calibration_offsets.extend(
                factory.calibration_offset_points(model_name, offsets)
            )
        return adjusted_points

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


def _run_fit(
    model: ForecastModel,
    df: pd.DataFrame,
    horizon: int,
    freq: str,
    levels: list[int],
    regressors: pd.DataFrame | None = None,
) -> pd.DataFrame:
    return model.fit_predict(
        df=df, horizon=horizon, freq=freq, levels=levels, regressors=regressors
    )


# Conformal calibration is applied symmetrically to both the snapshot
# ``ForecastPoint`` list (consumed by /metrics) and the
# ``ForecastCurvePoint`` list (consumed by the remote_write sink). The two
# carry the same (metric, labels, value) shape but ``ForecastCurvePoint``
# also has a ``timestamp_ms`` field. The decision logic — "if this is a
# lower/upper row for a calibrated (model, level), widen and label it" —
# is shared via :func:`_calibrated_band_value`; the per-type list rebuild
# lives in the two thin wrappers so each wrapper is straight-line code
# without a constructor callback.


def _calibrated_band_value(
    metric: str,
    labels: dict[str, str],
    value: float,
    offsets_by_model: dict[str, dict[int, float]],
) -> tuple[float, bool] | None:
    """Return ``(new_value, calibrated)`` if the point is a calibratable band, else ``None``.

    ``None`` means "leave the point alone". ``(value, False)`` means the
    offset is zero/missing for this (model, level), so the band is left as
    the model emitted it; ``(value, True)`` means we widened and the
    caller should add a ``calibrated="true"`` label. The yhat row (metric
    exactly ending in ``_forecast``) is always returned as ``None`` —
    calibration is a band adjustment, not a point-forecast adjustment.
    """
    model_name = labels.get("model", "")
    offsets = offsets_by_model.get(model_name)
    if offsets is None or metric.endswith("_forecast"):
        return None
    try:
        level = int(labels.get("level", ""))
    except ValueError:
        return None
    offset = offsets.get(level)
    if offset is None or offset <= 0:
        return None
    if metric.endswith("_forecast_lower"):
        return value - offset, True
    if metric.endswith("_forecast_upper"):
        return value + offset, True
    return None


def _apply_offsets_to_snapshot_points(
    points: list[ForecastPoint],
    offsets_by_model: dict[str, dict[int, float]],
) -> list[ForecastPoint]:
    """Widen ``ForecastPoint`` bands by the per-(model, level) offset.

    Returns a freshly constructed list; the input list is not mutated.
    """
    if not offsets_by_model:
        return points
    out: list[ForecastPoint] = []
    for point in points:
        adjusted = _calibrated_band_value(point.metric, point.labels, point.value, offsets_by_model)
        if adjusted is None:
            out.append(point)
            continue
        new_value, calibrated = adjusted
        new_labels = {**point.labels, "calibrated": "true"} if calibrated else point.labels
        out.append(ForecastPoint(metric=point.metric, labels=new_labels, value=new_value))
    return out


def _apply_offsets_to_curve_points(
    points: list[ForecastCurvePoint],
    offsets_by_model: dict[str, dict[int, float]],
) -> list[ForecastCurvePoint]:
    """Widen ``ForecastCurvePoint`` bands by the per-(model, level) offset.

    Returns a freshly constructed list; the input list is not mutated.
    Carries the original ``timestamp_ms`` through unchanged — calibration
    shifts the band level, not when the forecast is for.
    """
    if not offsets_by_model:
        return points
    out: list[ForecastCurvePoint] = []
    for point in points:
        adjusted = _calibrated_band_value(point.metric, point.labels, point.value, offsets_by_model)
        if adjusted is None:
            out.append(point)
            continue
        new_value, calibrated = adjusted
        new_labels = {**point.labels, "calibrated": "true"} if calibrated else point.labels
        out.append(
            ForecastCurvePoint(
                metric=point.metric,
                labels=new_labels,
                value=new_value,
                timestamp_ms=point.timestamp_ms,
            )
        )
    return out


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
    for model_name, fit_seconds in fit_result.fit_durations.items():
        ctx.fit_durations[(query_id, model_name)] = fit_seconds
    for model_name in fit_result.slow_fits:
        key = (query_id, model_name)
        ctx.slow_fits[key] = ctx.slow_fits.get(key, 0) + 1
    for regressor_id, reason in fit_result.regressor_failures:
        rkey = (query_id, regressor_id, reason)
        ctx.regressor_failures[rkey] = ctx.regressor_failures.get(rkey, 0) + 1


def _resolve_spec(specs: list[ModelSpec], name: str) -> ModelSpec:
    """Look up a configured ``ModelSpec`` by name; defaults to params=``{}``.

    The fallback model is allowed to live outside ``defaults.models`` (a
    SeasonalNaive safety net for an AutoARIMA-only setup is the canonical
    case). We synthesise an empty-params spec so the fallback runs with
    stock model defaults rather than failing the lookup.
    """
    for spec in specs:
        if spec.name == name:
            return spec
    return ModelSpec(name=name)


def _combine_ensemble(
    component_forecasts: dict[str, pd.DataFrame],
    weights: dict[str, float],
    levels: list[int],
) -> pd.DataFrame:
    """Build a single forecast frame from inverse-MAPE-weighted components.

    Aligns on ``unique_id`` + ``ds``. Each numeric column is the weighted
    sum across components present at that row; missing columns (a model
    that didn't emit a level we requested) contribute weight 0 so the
    ensemble doesn't quietly inherit one component's missing band.
    """
    import pandas as pd  # noqa: PLC0415

    base_cols = ["unique_id", "ds"]
    value_cols = ["yhat"] + [
        f"yhat_{kind}_{level}" for kind in ("lower", "upper") for level in levels
    ]

    import numpy as np  # noqa: PLC0415

    accumulator: pd.DataFrame | None = None
    weight_accumulator: dict[str, np.ndarray[Any, Any]] | None = None
    for name, weight in weights.items():
        df = component_forecasts.get(name)
        if df is None or df.empty:
            continue
        if accumulator is None or weight_accumulator is None:
            accumulator = df[base_cols].reset_index(drop=True).copy()
            weight_accumulator = {col: np.zeros(len(df), dtype=float) for col in value_cols}
            for col in value_cols:
                accumulator[col] = 0.0
        for col in value_cols:
            if col not in df.columns:
                continue
            # Switch to numpy arrays for the accumulation: pandas would
            # try to align on the (matching) index every iteration, which
            # both wastes work and confuses pandas-stubs about the
            # resulting dtype.
            current = accumulator[col].to_numpy(dtype=float)
            accumulator[col] = current + df[col].to_numpy(dtype=float) * weight
            weight_accumulator[col] = weight_accumulator[col] + weight
    if accumulator is None or weight_accumulator is None:
        # No usable components — return an empty frame the caller can detect.
        return pd.DataFrame(columns=[*base_cols, *value_cols])
    # Convert weight sums back to plain Series for the divide-by-weight step;
    # rows where total weight is 0 retain NaN rather than dividing by zero.
    for col in value_cols:
        denom = pd.Series(weight_accumulator[col])
        accumulator[col] = accumulator[col] / denom.where(denom > 0, other=float("nan"))
    return accumulator


def _run_backtest(
    model: ForecastModel,
    df: pd.DataFrame,
    horizon_steps: int,
    freq: str,
    levels: list[int],
    regressors: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    return backtest(
        model=model,
        df=df,
        horizon_steps=horizon_steps,
        freq=freq,
        levels=levels,
        regressors=regressors,
    )


def _pandas_freq(step: timedelta) -> str:
    seconds = int(step.total_seconds())
    if seconds % 3600 == 0:
        return f"{seconds // 3600}h"
    if seconds % 60 == 0:
        return f"{seconds // 60}min"
    return f"{seconds}s"


def _summarise_backtest(
    predictions: pd.DataFrame,
    test: pd.DataFrame,
    levels: list[int],
) -> tuple[float, float, dict[int, BandSummary]] | None:
    """Extract the trailing (actual, yhat, per-level band) tuple, or None.

    Returns ``None`` if either frame is empty or missing the ``y``/``yhat``
    columns; non-finite cells are tolerated and surface downstream as NaN.
    """
    if predictions.empty or test.empty:
        return None
    try:
        last_actual = float(test["y"].iloc[-1])
        last_yhat = float(predictions["yhat"].iloc[-1])
    except (KeyError, IndexError, ValueError):
        return None
    bands: dict[int, BandSummary] = {}
    for level in levels:
        lo_col = f"yhat_lower_{level}"
        hi_col = f"yhat_upper_{level}"
        if lo_col not in predictions.columns or hi_col not in predictions.columns:
            continue
        try:
            lower = float(predictions[lo_col].iloc[-1])
            upper = float(predictions[hi_col].iloc[-1])
        except (ValueError, IndexError):
            continue
        bands[level] = BandSummary(lower=lower, upper=upper)
    return last_actual, last_yhat, bands
