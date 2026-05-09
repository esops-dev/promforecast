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
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Literal

import structlog

from . import models as model_registry
from .config import GroupConfig
from .exporter import (
    Exporter,
    ForecastPoint,
    GroupSnapshot,
    InvalidMetricNameError,
    validate_metric_name,
)
from .source import PromSource, SeriesFrame

if TYPE_CHECKING:
    from collections.abc import Iterable

    import pandas as pd

    from .config import Config, QueryConfig
    from .models import ForecastModel

logger = structlog.get_logger(__name__)


OverflowStrategy = Literal["drop_lowest_priority", "reject", "sample"]


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
    (~hundreds of MB of fork overhead per timeout); deferred to v0.2 when
    we have evidence it's needed.
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

            points: list[ForecastPoint] = []
            series_count = 0
            dropped: dict[str, int] = {}
            failures: dict[str, int] = {}

            safety = self._config.safety
            strategy: OverflowStrategy = safety.series_overflow
            global_remaining = self._global_budgets.get(group.name, 0)
            global_cap_enabled = safety.max_total_series > 0

            for query in group.queries:
                try:
                    frames = await self._fetch(query)
                except Exception:
                    log.exception("query_failed", id=query.id)
                    failures["query_error"] = failures.get("query_error", 0) + 1
                    continue

                effective_cap = safety.max_series_per_query
                cap_reason = "per_query_cap"
                if global_cap_enabled and global_remaining < effective_cap:
                    effective_cap = global_remaining
                    cap_reason = "total_cap"

                allowed, dropped_n = _apply_overflow(frames, effective_cap, strategy)
                if dropped_n:
                    dropped[cap_reason] = dropped.get(cap_reason, 0) + dropped_n
                if global_cap_enabled:
                    global_remaining = max(0, global_remaining - len(allowed))
                series_count += len(allowed)

                min_points = self._config.defaults.min_points
                for frame in allowed:
                    if len(frame.values) < min_points:
                        dropped["min_points"] = dropped.get("min_points", 0) + 1
                        continue
                    try:
                        new_points = await self._fit_series(query, frame, group.name)
                    except InvalidMetricNameError:
                        log.exception("invalid_metric_name", id=query.id)
                        failures["invalid_metric_name"] = failures.get("invalid_metric_name", 0) + 1
                        break
                    except Exception:
                        log.exception("fit_failed", id=query.id, labels=frame.labels)
                        failures["fit_error"] = failures.get("fit_error", 0) + 1
                        continue
                    points.extend(new_points)

            duration = time.monotonic() - started
            self._exporter.update(
                GroupSnapshot(
                    group=group.name,
                    points=points,
                    last_run_timestamp=time.time(),
                    run_duration_seconds=duration,
                    series_count=series_count,
                    series_dropped=dropped,
                    failures=failures,
                )
            )
            log.info(
                "group_run_complete",
                duration_s=round(duration, 3),
                series=series_count,
                points=len(points),
            )

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
    ) -> list[ForecastPoint]:
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

        all_points: list[ForecastPoint] = []
        loop = asyncio.get_running_loop()
        for model_name in d.models:
            model = model_registry.build(model_name, season_length=d.season_length)
            forecast = await asyncio.wait_for(
                loop.run_in_executor(
                    self._executor,
                    _run_fit,
                    model,
                    df,
                    horizon,
                    freq,
                    list(d.confidence_levels),
                ),
                timeout=self._config.safety.fit_timeout.total_seconds(),
            )
            all_points.extend(
                _forecast_to_points(
                    forecast=forecast,
                    metric_id=query.id,
                    base_labels=frame.labels,
                    group=group_name,
                    model_name=model_name,
                    levels=list(d.confidence_levels),
                )
            )
        return all_points

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
) -> Iterable[ForecastPoint]:
    """Convert the last forecast row into snapshot points.

    /metrics is a snapshot endpoint, so we expose only the *next* forecast
    step. Drawing the full curve in Grafana is the job of the remote_write
    sink (v0.3) — see CLAUDE.md.
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
