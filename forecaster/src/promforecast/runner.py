"""Per-group execution and scheduling.

Each group runs as a long-lived asyncio task that fits, exports, and sleeps
for ``server.refresh_interval``. A per-group lock prevents a slow run from
being triggered twice concurrently. Failure isolation is layered: per query,
per series, per regressor.
"""

from __future__ import annotations

import asyncio
import contextlib
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Literal

import structlog

from . import deviation as deviation_mod
from . import models as model_registry
from . import quality as quality_mod
from .accuracy import (
    AccuracyResult,
    HorizonSpec,
    backtest,
    evaluate,
    horizons_from_durations,
    overall_mape,
)
from .config import GroupConfig
from .exporter import (
    AccuracyPoint,
    DeviationPoint,
    Exporter,
    ForecastPoint,
    GroupSnapshot,
    InvalidMetricNameError,
    QualityPoint,
    validate_metric_name,
)
from .selector import SelectionResult, select
from .source import PromSource, SeriesFrame

if TYPE_CHECKING:
    from collections.abc import Iterable

    import pandas as pd

    from .config import Config, QueryConfig
    from .models import ForecastModel

logger = structlog.get_logger(__name__)


OverflowStrategy = Literal["drop_lowest_priority", "reject", "sample"]


@dataclass
class _FitResult:
    """Per-series outputs returned by ``Runner._fit_series``."""

    points: list[ForecastPoint] = field(default_factory=list)
    accuracy: list[AccuracyPoint] = field(default_factory=list)
    deviation: list[DeviationPoint] = field(default_factory=list)
    quality: list[QualityPoint] = field(default_factory=list)


@dataclass(frozen=True)
class _BandSummary:
    """Per-level (lower, upper) band from the trailing backtest row."""

    lower: float
    upper: float

    @property
    def width(self) -> float:
        return abs(self.upper - self.lower)


@dataclass
class _BacktestOutcome:
    """Per-(model, horizon) scores plus a compact summary of the trailing row.

    Storing only the last (actual, predicted, band) tuple — not the full
    ``test``/``predictions`` frames — keeps memory bounded for high-cardinality
    groups: at 500 series x 4 models x ~288-row frames the previous design
    spiked ~22 MiB per series; this collapses it to a few hundred bytes.
    """

    results: list[AccuracyResult]
    last_actual: float
    last_yhat: float
    last_bands: dict[int, _BandSummary]


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
    series_count: int = 0
    dropped: dict[str, int] = field(default_factory=dict)
    failures: dict[str, int] = field(default_factory=dict)
    global_remaining: int = 0


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
    ) -> None:
        self._config = config
        self._source = source
        self._exporter = exporter
        self._executor = executor or ThreadPoolExecutor(max_workers=4)
        self._locks: dict[str, asyncio.Lock] = {g.name: asyncio.Lock() for g in config.groups}
        self._tasks: list[asyncio.Task[None]] = []
        self._stopped = asyncio.Event()
        self._global_budgets = _compute_global_budgets(
            list(config.groups), config.safety.max_total_series
        )
        if config.defaults.auto_select == "ensemble":
            # Surfaced once at boot so an operator who picked ``ensemble``
            # doesn't quietly get explicit-mode behaviour. Keeping the value
            # accepted (rather than rejecting it in the schema) means configs
            # written today won't break when ensemble support lands later.
            logger.warning(
                "ensemble_auto_select_not_yet_implemented",
                effective_behaviour="explicit",
            )

    async def start(self) -> None:
        """Run each group once immediately, then on the configured interval."""
        interval = self._config.server.refresh_interval.total_seconds()
        for group in self._config.groups:
            self._tasks.append(asyncio.create_task(self._loop(group, interval)))

    async def _loop(self, group: GroupConfig, interval_seconds: float) -> None:
        while not self._stopped.is_set():
            try:
                await self.run_group(group)
            except Exception:
                logger.exception("group_run_failed", group=group.name)
                self._exporter.increment_failure(group.name, reason="group_error")
            try:
                await asyncio.wait_for(self._stopped.wait(), timeout=interval_seconds)
            except TimeoutError:
                continue

    async def run_group(self, group: GroupConfig) -> None:
        """Execute a single group: query, fit, select, export."""
        async with self._locks[group.name]:
            started = time.monotonic()
            log = logger.bind(group=group.name)
            log.info("group_run_start")

            ctx = _RunContext(global_remaining=self._global_budgets.get(group.name, 0))
            for query in group.queries:
                await self._run_query(query, group, ctx, log)

            duration = time.monotonic() - started
            self._exporter.update(
                GroupSnapshot(
                    group=group.name,
                    points=ctx.points,
                    accuracy_points=ctx.accuracy_points,
                    deviation_points=ctx.deviation_points,
                    quality_points=ctx.quality_points,
                    last_run_timestamp=time.time(),
                    run_duration_seconds=duration,
                    series_count=ctx.series_count,
                    series_dropped=ctx.dropped,
                    failures=ctx.failures,
                )
            )
            log.info(
                "group_run_complete",
                duration_s=round(duration, 3),
                series=ctx.series_count,
                points=len(ctx.points),
            )

    async def _run_query(
        self,
        query: QueryConfig,
        group: GroupConfig,
        ctx: _RunContext,
        log: structlog.stdlib.BoundLogger,
    ) -> None:
        try:
            frames = await self._fetch(query)
        except Exception:
            log.exception("query_failed", id=query.id)
            ctx.failures["query_error"] = ctx.failures.get("query_error", 0) + 1
            return

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
        if global_cap_enabled:
            ctx.global_remaining = max(0, ctx.global_remaining - len(allowed))
        ctx.series_count += len(allowed)

        min_points = self._config.defaults.min_points
        for frame in allowed:
            if len(frame.values) < min_points:
                ctx.dropped["min_points"] = ctx.dropped.get("min_points", 0) + 1
                continue
            try:
                fit_result = await self._fit_series(query, frame, group.name)
            except InvalidMetricNameError:
                log.exception("invalid_metric_name", id=query.id)
                ctx.failures["invalid_metric_name"] = ctx.failures.get("invalid_metric_name", 0) + 1
                return
            except Exception:
                log.exception("fit_failed", id=query.id, labels=frame.labels)
                ctx.failures["fit_error"] = ctx.failures.get("fit_error", 0) + 1
                continue
            ctx.points.extend(fit_result.points)
            ctx.accuracy_points.extend(fit_result.accuracy)
            ctx.deviation_points.extend(fit_result.deviation)
            ctx.quality_points.extend(fit_result.quality)

    async def _fetch(self, query: QueryConfig) -> list[SeriesFrame]:
        d = self._config.defaults
        s = self._config.safety
        end = datetime.now(tz=UTC)
        start = end - d.lookback
        step_seconds = max(1, int(d.step.total_seconds()))
        return await asyncio.wait_for(
            self._source.query_range(query.promql, start, end, step_seconds),
            timeout=s.query_timeout.total_seconds(),
        )

    async def _fit_series(
        self,
        query: QueryConfig,
        frame: SeriesFrame,
        group_name: str,
    ) -> _FitResult:
        import pandas as pd  # noqa: PLC0415

        d = self._config.defaults
        validate_metric_name(query.id)

        df = pd.DataFrame(
            {
                "unique_id": ["s"] * len(frame.values),
                "ds": pd.to_datetime(frame.timestamps),
                "y": frame.values,
            }
        )
        freq = _pandas_freq(d.step)
        horizon = max(1, int(d.horizon.total_seconds() // d.step.total_seconds()))
        levels = list(d.confidence_levels)

        accuracy_horizons = horizons_from_durations(list(d.accuracy_horizons), d.step)
        auto_select_enabled = bool(d.auto_select)
        # Backtests serve three consumers: accuracy metrics, auto-select, and
        # the deviation/quality computations. Run if any of them is on.
        run_backtest = (
            d.evaluate or auto_select_enabled or d.emit_deviation or d.emit_quality
        ) and len(accuracy_horizons) > 0

        backtests: dict[str, _BacktestOutcome] = {}
        if run_backtest:
            backtests = await self._run_backtests(
                df=df,
                models=list(d.models),
                horizons=accuracy_horizons,
                freq=freq,
                levels=levels,
                season_length=d.season_length,
            )

        backtest_results: dict[str, list[AccuracyResult]] = {
            name: outcome.results for name, outcome in backtests.items()
        }

        if auto_select_enabled and d.auto_select != "ensemble":
            forecast_points = await self._auto_select_and_forecast(
                df=df,
                query=query,
                frame=frame,
                group_name=group_name,
                horizon=horizon,
                freq=freq,
                levels=levels,
                backtest_results=backtest_results,
            )
        else:
            forecast_points = await self._explicit_forecast(
                df=df,
                models=list(d.models),
                query=query,
                frame=frame,
                group_name=group_name,
                horizon=horizon,
                freq=freq,
                levels=levels,
            )

        result = _FitResult(points=forecast_points)

        if d.evaluate and backtest_results:
            result.accuracy = list(
                _accuracy_points_for_series(
                    backtest_results=backtest_results,
                    metric_id=query.id,
                    base_labels=frame.labels,
                    group=group_name,
                )
            )

        # Restrict deviation/quality to the model(s) actually emitted as
        # forecasts. In auto-select mode that's the winner; in explicit mode
        # it's all configured models.
        emitted_models = sorted({p.labels.get("model", "") for p in forecast_points})
        emitted_models = [m for m in emitted_models if m]

        if d.emit_deviation:
            result.deviation = list(
                _deviation_points_for_series(
                    backtests=backtests,
                    emitted_models=emitted_models,
                    metric_id=query.id,
                    base_labels=frame.labels,
                    group=group_name,
                    levels=levels,
                )
            )

        if d.emit_quality:
            expected_points = max(
                1,
                int(d.lookback.total_seconds() // max(1, int(d.step.total_seconds()))),
            )
            result.quality = list(
                _quality_points_for_series(
                    backtests=backtests,
                    emitted_models=emitted_models,
                    metric_id=query.id,
                    base_labels=frame.labels,
                    group=group_name,
                    train_y=frame.values,
                    coverage=min(1.0, len(frame.values) / float(expected_points)),
                )
            )

        return result

    async def _explicit_forecast(
        self,
        *,
        df: pd.DataFrame,
        models: list[str],
        query: QueryConfig,
        frame: SeriesFrame,
        group_name: str,
        horizon: int,
        freq: str,
        levels: list[int],
    ) -> list[ForecastPoint]:
        # Fits run concurrently against the shared executor: per-series wall
        # time becomes ``max(fit_time)`` instead of ``sum(fit_time)`` whenever
        # the pool has spare workers. The pool itself caps total concurrency,
        # so this doesn't risk overloading anything beyond what a single
        # serial pass already touched.
        d = self._config.defaults
        tasks = [
            self._fit_one(
                model_name=model_name,
                df=df,
                horizon=horizon,
                freq=freq,
                levels=levels,
                season_length=d.season_length,
            )
            for model_name in models
        ]
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)
        out: list[ForecastPoint] = []
        for model_name, forecast in zip(models, outcomes, strict=True):
            if isinstance(forecast, BaseException):
                logger.exception(
                    "fit_failed_in_explicit_mode",
                    model=model_name,
                    exc_info=forecast,
                )
                continue
            out.extend(
                _forecast_to_points(
                    forecast=forecast,
                    metric_id=query.id,
                    base_labels=frame.labels,
                    group=group_name,
                    model_name=model_name,
                    levels=levels,
                    selection=None,
                )
            )
        return out

    async def _auto_select_and_forecast(
        self,
        *,
        df: pd.DataFrame,
        query: QueryConfig,
        frame: SeriesFrame,
        group_name: str,
        horizon: int,
        freq: str,
        levels: list[int],
        backtest_results: dict[str, list[AccuracyResult]],
    ) -> list[ForecastPoint]:
        d = self._config.defaults
        backtest_mape = {
            model_name: mape
            for model_name, results in backtest_results.items()
            if (mape := overall_mape(results)) is not None
        }
        selection = select(
            fallback_model=d.fallback_model,
            fallback_mape_threshold=d.fallback_mape_threshold,
            backtest_mape=backtest_mape,
        )
        forecast = await self._fit_one(
            model_name=selection.model,
            df=df,
            horizon=horizon,
            freq=freq,
            levels=levels,
            season_length=d.season_length,
        )
        return list(
            _forecast_to_points(
                forecast=forecast,
                metric_id=query.id,
                base_labels=frame.labels,
                group=group_name,
                model_name=selection.model,
                levels=levels,
                selection=selection,
            )
        )

    async def _run_backtests(
        self,
        *,
        df: pd.DataFrame,
        models: list[str],
        horizons: list[HorizonSpec],
        freq: str,
        levels: list[int],
        season_length: int,
    ) -> dict[str, _BacktestOutcome]:
        """Backtest each configured model. Per-model failures are isolated.

        The hold-out window is ``max(horizon.steps)`` — large enough to score
        every horizon from a single fit, which keeps the cost to one extra
        per-model fit per series. With ``auto_select: true`` the per-series
        cost is therefore N + 1 fits (N backtests + 1 final forecast on full
        data); with ``auto_select: false`` it's N + N (N backtests + N final
        forecasts). Set ``evaluate: false`` *and* disable deviation/quality
        emission to skip backtesting when the fit budget is tight.

        Backtest fits dispatch concurrently against the executor pool, so
        per-series wall time tracks the slowest model rather than the sum.

        The returned mapping keeps the raw ``test``/``predictions`` frames so
        the deviation step can use the most recent (actual, predicted band)
        pair without refitting.
        """
        if not horizons:
            return {}
        eval_steps = max(h.steps for h in horizons)
        if len(df) <= eval_steps + 1:
            return {}
        train = df.iloc[:-eval_steps].reset_index(drop=True)
        loop = asyncio.get_running_loop()
        timeout = self._config.safety.fit_timeout.total_seconds()

        async def _one(model_name: str) -> tuple[pd.DataFrame, pd.DataFrame] | None:
            model = model_registry.build(model_name, season_length=season_length)
            return await asyncio.wait_for(
                loop.run_in_executor(
                    self._executor, _run_backtest, model, df, eval_steps, freq, levels
                ),
                timeout=timeout,
            )

        outcomes = await asyncio.gather(*(_one(name) for name in models), return_exceptions=True)
        results: dict[str, _BacktestOutcome] = {}
        for model_name, outcome in zip(models, outcomes, strict=True):
            if isinstance(outcome, BaseException):
                logger.exception("backtest_failed", model=model_name, exc_info=outcome)
                continue
            if outcome is None:
                continue
            test, predictions = outcome
            summary = _summarise_backtest(predictions=predictions, test=test, levels=levels)
            if summary is None:
                continue
            last_actual, last_yhat, last_bands = summary
            results[model_name] = _BacktestOutcome(
                results=evaluate(
                    test=test,
                    predictions=predictions,
                    train=train,
                    season_length=season_length,
                    horizons=horizons,
                    model_name=model_name,
                ),
                last_actual=last_actual,
                last_yhat=last_yhat,
                last_bands=last_bands,
            )
        return results

    async def _fit_one(
        self,
        *,
        model_name: str,
        df: pd.DataFrame,
        horizon: int,
        freq: str,
        levels: list[int],
        season_length: int,
    ) -> pd.DataFrame:
        loop = asyncio.get_running_loop()
        model = model_registry.build(model_name, season_length=season_length)
        return await asyncio.wait_for(
            loop.run_in_executor(
                self._executor,
                _run_fit,
                model,
                df,
                horizon,
                freq,
                levels,
            ),
            timeout=self._config.safety.fit_timeout.total_seconds(),
        )

    async def stop(self) -> None:
        """Shut down the scheduler and cancel in-flight runs."""
        self._stopped.set()
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._executor.shutdown(wait=False, cancel_futures=True)
        await self._source.aclose()


def _run_fit(
    model: ForecastModel,
    df: pd.DataFrame,
    horizon: int,
    freq: str,
    levels: list[int],
) -> pd.DataFrame:
    return model.fit_predict(df=df, horizon=horizon, freq=freq, levels=levels)


def _run_backtest(
    model: ForecastModel,
    df: pd.DataFrame,
    horizon_steps: int,
    freq: str,
    levels: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    return backtest(model=model, df=df, horizon_steps=horizon_steps, freq=freq, levels=levels)


def _pandas_freq(step: timedelta) -> str:
    seconds = int(step.total_seconds())
    if seconds % 3600 == 0:
        return f"{seconds // 3600}h"
    if seconds % 60 == 0:
        return f"{seconds // 60}min"
    return f"{seconds}s"


def _forecast_to_points(
    forecast: pd.DataFrame,
    metric_id: str,
    base_labels: dict[str, str],
    group: str,
    model_name: str,
    levels: list[int],
    selection: SelectionResult | None,
) -> Iterable[ForecastPoint]:
    """Convert the last forecast row into snapshot points.

    /metrics is a snapshot endpoint, so we expose only the *next* forecast
    step. Drawing the full curve in Grafana is the job of the planned
    remote_write sink.

    When ``selection`` is set (auto-select mode), additional labels
    ``best_model`` and ``model_fallback`` are attached so dashboards can
    distinguish "this model won the horse race" from "this is the fallback
    after the winner blew its MAPE budget".
    """
    if forecast.empty:
        return
    last = forecast.iloc[-1]
    horizon_step = "next"
    yhat_value = float(last["yhat"])
    common_labels = {k: v for k, v in base_labels.items() if k != "__name__"}
    common_labels = {
        **common_labels,
        "group": group,
        "model": model_name,
        "horizon": horizon_step,
    }
    if selection is not None:
        common_labels["best_model"] = selection.best_model
        common_labels["model_fallback"] = "true" if selection.fallback_applied else "false"
    yield ForecastPoint(
        metric=f"{metric_id}_forecast",
        labels=common_labels,
        value=yhat_value,
    )
    for level in levels:
        lo_col = f"yhat_lower_{level}"
        hi_col = f"yhat_upper_{level}"
        if lo_col in forecast.columns:
            yield ForecastPoint(
                metric=f"{metric_id}_forecast_lower",
                labels={**common_labels, "level": str(level)},
                value=float(last[lo_col]),
            )
        if hi_col in forecast.columns:
            yield ForecastPoint(
                metric=f"{metric_id}_forecast_upper",
                labels={**common_labels, "level": str(level)},
                value=float(last[hi_col]),
            )


def _accuracy_points_for_series(
    backtest_results: dict[str, list[AccuracyResult]],
    metric_id: str,
    base_labels: dict[str, str],
    group: str,
) -> Iterable[AccuracyPoint]:
    """Flatten per-series backtest results into ``AccuracyPoint``s.

    ``id`` is the same string used for the forecast metric stem so that an
    accuracy dashboard can join cleanly on ``id``.
    """
    common_labels = {k: v for k, v in base_labels.items() if k != "__name__"}
    common_labels = {**common_labels, "id": metric_id, "group": group}
    for model_name, results in backtest_results.items():
        for res in results:
            yield AccuracyPoint(
                labels={
                    **common_labels,
                    "model": model_name,
                    "horizon": res.horizon,
                },
                mape=res.mape,
                mase=res.mase,
            )


def _summarise_backtest(
    predictions: pd.DataFrame,
    test: pd.DataFrame,
    levels: list[int],
) -> tuple[float, float, dict[int, _BandSummary]] | None:
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
    bands: dict[int, _BandSummary] = {}
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
        bands[level] = _BandSummary(lower=lower, upper=upper)
    return last_actual, last_yhat, bands


def _deviation_points_for_series(
    backtests: dict[str, _BacktestOutcome],
    emitted_models: list[str],
    metric_id: str,
    base_labels: dict[str, str],
    group: str,
    levels: list[int],
) -> Iterable[DeviationPoint]:
    """Compute deviation from the most recent backtest (actual, band) pair.

    The forecaster is stateless, so we use the trailing point of the
    backtest hold-out window — the most recent in-sample point where we
    have both an actual and a predicted band — as the canonical "is the
    world still behaving like the model expects?" signal. One sample per
    (series, model, level) is emitted.
    """
    common_labels = {k: v for k, v in base_labels.items() if k != "__name__"}
    common_labels = {**common_labels, "id": metric_id, "group": group}
    for model_name in emitted_models:
        outcome = backtests.get(model_name)
        if outcome is None:
            continue
        for level in levels:
            band = outcome.last_bands.get(level)
            if band is None:
                continue
            dev = deviation_mod.compute(
                actual=outcome.last_actual,
                yhat=outcome.last_yhat,
                yhat_lower=band.lower,
                yhat_upper=band.upper,
            )
            yield DeviationPoint(
                labels={
                    **common_labels,
                    "model": model_name,
                    "level": str(level),
                },
                ratio=dev.ratio,
                outside_band=dev.outside_band,
            )


def _quality_points_for_series(
    backtests: dict[str, _BacktestOutcome],
    emitted_models: list[str],
    metric_id: str,
    base_labels: dict[str, str],
    group: str,
    train_y: list[float],
    coverage: float,
) -> Iterable[QualityPoint]:
    """Compute a composite quality score per emitted (series, model).

    Combines:
      - overall MAPE across the configured accuracy horizons,
      - the mean backtest band width relative to ``mean(|train_y|)`` (paired
        with actuals during backtest, which is more meaningful than scoring
        the unvalidated forward bands), and
      - the lookback data coverage.

    Models with no backtest data — typically the auto-select fallback model
    when it isn't in the configured ``models`` list — are skipped rather
    than scored as zero. Emitting zero quality for fallback'd series would
    fire the ``ForecastQualityZero`` alert without signal; the
    ``ForecastFallbackPersistent`` alert covers that case better.

    See :mod:`promforecast.quality` for the exact formula.
    """
    import math  # noqa: PLC0415

    abs_train = [abs(v) for v in train_y]
    scale = (sum(abs_train) / len(abs_train)) if abs_train else 0.0

    common_labels = {k: v for k, v in base_labels.items() if k != "__name__"}
    common_labels = {**common_labels, "id": metric_id, "group": group}

    for model_name in emitted_models:
        outcome = backtests.get(model_name)
        if outcome is None:
            continue
        mape = overall_mape(outcome.results)
        widths = [b.width for b in outcome.last_bands.values() if math.isfinite(b.width)]
        avg_width = sum(widths) / len(widths) if widths else float("nan")
        band_width_relative = (
            (avg_width / scale) if scale > 0 and math.isfinite(avg_width) else None
        )
        score = quality_mod.compute(
            backtest_mape=mape,
            band_width_relative=band_width_relative,
            coverage=coverage,
        )
        yield QualityPoint(
            labels={**common_labels, "model": model_name},
            score=score,
        )
