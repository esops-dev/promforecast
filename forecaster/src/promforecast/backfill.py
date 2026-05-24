"""Historical SLO-window backfill.

The runner only emits metrics from "now" forward — a fresh install
gets an empty :metric:`forecast_quality_score`, empty
:metric:`forecast_accuracy_mape`, empty band-coverage gauges, and
so on, until enough refresh cycles have accumulated for an SLO
window to be meaningful. This module replays the fit pipeline at
simulated past timestamps and pushes the resulting forecast curves
(plus per-fit accuracy / quality summaries) to the configured
``sink.remote_write`` URL with explicit historical
``timestamp_ms`` values so VM / Mimir / Thanos store them as if
they had been emitted at that time.

Runs as a one-shot pod separate from the live forecaster (see
``examples/k8s/backfill-job.yaml``) so a long backfill does not
contend with the scheduled fit budget. The live forecaster's
snapshot is never touched — backfill is a pure write-side
operation against the long-term TSDB.

**Scope.** This module backfills the operator-visible families
that have a natural "as-of-time" interpretation:

* the forecast curve (yhat + per-level bands), one per
  (series, model) per emitted horizon step;
* the per-(series, model) backtest accuracy summary
  (:metric:`forecast_accuracy_mape` /
  :metric:`forecast_accuracy_mase`); and
* the composite :metric:`forecast_quality_score` for each
  (series, model) emission.

Snapshot-only operational metrics
(:metric:`forecast_failures_total`,
:metric:`forecast_run_duration_seconds`, the deploy-risk score)
are deliberately *not* backfilled: they describe live execution
state, not the underlying signal. An operator who wants them as
"historical zero" can rely on Prometheus's normal absent-series
treatment. The conditional / decomposition / drift / band-coverage
/ deviation families are also skipped — they need cross-fit state
(drift cache, conformal holdout, rolling band-coverage windows)
that the one-shot walk doesn't maintain; backfilling them would
either be wrong (rolling windows distorted by single synthetic
observations) or require running the live runner's full cycle
state machinery from a one-shot script. See
``docs/operations/backfill.md`` for the full omission list.

Regressors are **not** fetched during backfill — the per-fit
pipeline always passes ``regressors=None`` so the historical
curves reflect the model's base seasonality without the deploy /
calendar exogenous shifts the live runner uses. The trade-off is
documented in ``docs/operations/backfill.md``.
"""

from __future__ import annotations

import asyncio
import math
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any

import structlog

from . import accuracy as accuracy_mod
from . import config as config_module
from . import models as model_registry
from . import preprocess as preprocess_mod
from . import quality as quality_mod
from .accuracy import overall_mape, overall_mase
from .config import (
    Config,
    GroupConfig,
    QueryConfig,
    effective_emission_safeguards,
    effective_refresh_interval,
    effective_season_length,
    effective_step,
)
from .point_factory import SeriesPointFactory
from .preview import _pandas_freq
from .runner import _apply_overflow, _apply_safeguards_to_curve_points
from .sink import ForecastCurvePoint, ForecastSink, RemoteWriteSink, SinkError
from .source import PromSource

if TYPE_CHECKING:
    import pandas as pd

logger = structlog.get_logger(__name__)


class BackfillError(RuntimeError):
    """Raised for any operator-visible backfill failure.

    Carries an exit code so ``main.cli`` can hand it straight to
    ``sys.exit``.
    """

    def __init__(self, message: str, *, exit_code: int = 1) -> None:
        super().__init__(message)
        self.exit_code = exit_code


@dataclass(frozen=True)
class BackfillStats:
    """Per-run summary returned for logs + the CLI footer."""

    timestamps_walked: int
    fits_issued: int
    points_pushed: int
    failed_fits: int
    failed_pushes: int
    elapsed_seconds: float


def run_backfill(  # noqa: PLR0911
    *,
    config_path: Path,
    from_duration: str,
    to_timestamp: str | None,
    group_selector: list[str] | None,
    sink_url_override: str | None,
    out: IO[str] | None = None,
) -> int:
    """Top-level entry point used by ``main.cli``.

    Returns ``0`` on success, ``1`` for invalid inputs (config issues,
    over-budget window, no configured sink), ``2`` for TSDB / network
    failures encountered during the walk.
    """
    stream: IO[str] = out if out is not None else sys.stdout

    try:
        cfg = config_module.load(config_path)
    except FileNotFoundError:
        sys.stderr.write(f"error: config not found: {config_path}\n")
        return 2
    except Exception as exc:
        sys.stderr.write(f"error: config validation failed: {exc}\n")
        return 1

    try:
        from_seconds = _parse_duration_seconds(from_duration, flag="--from")
        to_time = _parse_to_timestamp(to_timestamp)
        _check_window(cfg, from_seconds)
    except BackfillError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return exc.exit_code

    sink_url = sink_url_override or cfg.sink.remote_write.url
    if not sink_url:
        sys.stderr.write(
            "error: backfill requires a remote_write sink — either configure "
            "sink.remote_write.{enabled: true, url: ...} or pass --sink-url.\n"
        )
        return 1

    selected_groups = _select_groups(cfg, group_selector)
    if not selected_groups:
        sys.stderr.write("error: --groups selector matched no configured groups\n")
        return 1

    sink: ForecastSink = RemoteWriteSink(
        url=sink_url,
        timeout_seconds=cfg.sink.remote_write.timeout.total_seconds(),
        headers=dict(cfg.sink.remote_write.headers),
        chunk_size=cfg.sink.remote_write.chunk_size,
        max_connections=cfg.sink.remote_write.max_connections,
        max_keepalive_connections=cfg.sink.remote_write.max_keepalive_connections,
    )
    source = PromSource(
        url=cfg.datasource.url,
        timeout_seconds=cfg.datasource.timeout.total_seconds(),
        max_concurrency=cfg.datasource.max_concurrent_requests,
        max_keepalive_connections=cfg.datasource.max_keepalive_connections,
    )

    try:
        stats = asyncio.run(
            _walk_backfill(
                config=cfg,
                source=source,
                sink=sink,
                from_seconds=from_seconds,
                to_time=to_time,
                groups=selected_groups,
                out=stream,
            )
        )
    except BackfillError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return exc.exit_code
    except Exception as exc:
        sys.stderr.write(f"error: unexpected backfill failure: {exc}\n")
        return 2

    stream.write(
        f"backfill complete: timestamps={stats.timestamps_walked} "
        f"fits={stats.fits_issued} points={stats.points_pushed} "
        f"failed_fits={stats.failed_fits} failed_pushes={stats.failed_pushes} "
        f"elapsed_s={stats.elapsed_seconds:.1f}\n"
    )
    return 0


async def _walk_backfill(
    *,
    config: Config,
    source: PromSource,
    sink: ForecastSink,
    from_seconds: float,
    to_time: datetime,
    groups: list[GroupConfig],
    out: IO[str],
) -> BackfillStats:
    """Walk back-in-time, run fits, push to sink, return stats."""
    started = time.monotonic()
    timestamps_walked = 0
    fits_issued = 0
    failed_fits = 0
    failed_pushes = 0
    points_pushed = 0

    try:
        for group in groups:
            timestamps = list(_iter_backfill_timestamps(config, group, from_seconds, to_time))
            group_fits = 0
            for ts in timestamps:
                timestamps_walked += 1
                buffer: list[ForecastCurvePoint] = []
                for query in group.queries:
                    try:
                        produced = await _run_query_at(
                            config=config,
                            group=group,
                            query=query,
                            source=source,
                            as_of=ts,
                        )
                    except BackfillError:
                        # An I/O-shaped failure (e.g. query timeout) carries
                        # its own exit code; surface it to the top-level
                        # walker so the caller distinguishes "TSDB
                        # unreachable" (exit 2) from "fit walked, some
                        # series failed" (exit 1). Catching it under the
                        # generic ``Exception`` arm below would silently
                        # downgrade timeouts into per-query failures.
                        raise
                    except Exception as exc:
                        logger.warning(
                            "backfill_query_failed",
                            group=group.name,
                            query=query.id,
                            ts=ts.isoformat(),
                            error=str(exc),
                        )
                        failed_fits += 1
                        continue
                    fits_issued += len(produced.fit_count)
                    group_fits += len(produced.fit_count)
                    buffer.extend(produced.curve)
                if buffer:
                    try:
                        await sink.write(buffer)
                        points_pushed += len(buffer)
                    except SinkError as exc:
                        logger.warning(
                            "backfill_push_failed",
                            group=group.name,
                            ts=ts.isoformat(),
                            reason=exc.reason,
                            error=str(exc),
                        )
                        failed_pushes += 1
            # Per-group ``forecast_backfill_fits_last`` audit row pushed
            # at the *final* as-of timestamp for the group. Named with
            # the ``_last`` (gauge) suffix rather than ``_total`` so
            # PromQL ``rate()`` / ``increase()`` don't false-fire on
            # a second backfill run that issues fewer fits than the
            # first: backfill runs are independent one-shot operations
            # with no monotonic-counter semantics, and the ``_total``
            # suffix would lie about that contract. The row's timestamp
            # still lets dashboards anchor "backfill happened then".
            if timestamps:
                try:
                    await sink.write(
                        [
                            ForecastCurvePoint(
                                metric="forecast_backfill_fits_last",
                                labels={"group": group.name},
                                value=float(group_fits),
                                timestamp_ms=int(timestamps[-1].timestamp() * 1000),
                            )
                        ]
                    )
                    points_pushed += 1
                except SinkError as exc:
                    logger.warning(
                        "backfill_audit_push_failed",
                        group=group.name,
                        reason=exc.reason,
                        error=str(exc),
                    )
                    failed_pushes += 1
            out.write(f"group {group.name}: timestamps={len(timestamps)} fits={group_fits}\n")
    finally:
        await source.aclose()
        await sink.aclose()

    elapsed = time.monotonic() - started
    return BackfillStats(
        timestamps_walked=timestamps_walked,
        fits_issued=fits_issued,
        points_pushed=points_pushed,
        failed_fits=failed_fits,
        failed_pushes=failed_pushes,
        elapsed_seconds=elapsed,
    )


@dataclass
class _QueryProduced:
    """Per-query backfill output for one historical timestamp."""

    curve: list[ForecastCurvePoint]
    fit_count: list[str]  # one entry per emitted (series, model) tuple


async def _run_query_at(
    *,
    config: Config,
    group: GroupConfig,
    query: QueryConfig,
    source: PromSource,
    as_of: datetime,
) -> _QueryProduced:
    """Fit ``query`` as-of ``as_of`` and translate to sink curve points.

    Built around the same primitives the runner uses — preprocess +
    statsforecast wrappers — but with a stripped-down per-fit
    pipeline that skips the live exporter state, the conditional
    refit, the cold-start fallback, and any feature that needs
    cross-run state. The point of backfill is to populate the
    long-term TSDB with curves an operator can query *now*; nuanced
    features can be replayed in a follow-up after the v1 shape is
    accepted.

    Per-(series, model) failures are isolated: a bad fit for one
    model doesn't abort the rest of the query.
    """
    import pandas as pd  # noqa: PLC0415

    d = config.defaults
    eff_step = effective_step(query, d)
    eff_season_length = effective_season_length(query, d, step=eff_step)
    step_seconds = max(1, int(eff_step.total_seconds()))
    horizon_steps = max(1, int(d.horizon.total_seconds() // step_seconds))
    levels = list(d.confidence_levels)
    freq = _pandas_freq(step_seconds)

    start = as_of - d.lookback
    try:
        frames = await asyncio.wait_for(
            source.query_range(query.promql, start, as_of, step_seconds),
            timeout=config.safety.query_timeout.total_seconds(),
        )
    except TimeoutError as exc:
        raise BackfillError(f"query timed out: {query.id}", exit_code=2) from exc

    # Apply the per-query series cap up front so a high-cardinality
    # PromQL during backfill doesn't blow the one-shot Job's memory.
    # The runner does the same cap inside ``_run_query`` for live
    # fits; without it here a query that legitimately returns 5000
    # series at some point in the past would push 5000 * N_models
    # fits into a single backfill timestamp.
    safety = config.safety
    allowed, _dropped = _apply_overflow(frames, safety.max_series_per_query, safety.series_overflow)

    clip, rounder = effective_emission_safeguards(query)
    produced = _QueryProduced(curve=[], fit_count=[])
    loop = asyncio.get_running_loop()
    for frame in allowed:
        try:
            cleaned = preprocess_mod.preprocess_series(
                timestamps=list(frame.timestamps),
                values=list(frame.values),
                step=eff_step,
                config=group.preprocess,
            )
        except Exception as exc:
            # Mirror the live runner's per-series failure isolation
            # (see ``_scheduler.py::_run_one_series``): a single bad
            # series — e.g. one whose timestamps trip a preprocess
            # invariant — must not abort the rest of the query's
            # backfill walk.
            logger.warning(
                "backfill_series_preprocess_failed",
                group=group.name,
                query=query.id,
                series_labels=frame.labels,
                error=str(exc),
            )
            continue
        if cleaned.dropped or len(cleaned.values) < d.min_points:
            continue
        df = pd.DataFrame(
            {
                "unique_id": ["s"] * len(cleaned.values),
                "ds": pd.to_datetime(cleaned.timestamps),
                "y": cleaned.values,
            }
        )
        factory = SeriesPointFactory(
            metric_id=query.id,
            base_labels=frame.labels,
            group=group.name,
            levels=levels,
        )
        for spec in d.models:
            try:
                model = model_registry.build(
                    spec.name, season_length=eff_season_length, params=spec.params
                )
                forecast = await asyncio.wait_for(
                    loop.run_in_executor(
                        None, model.fit_predict, df, horizon_steps, freq, levels, None
                    ),
                    timeout=config.safety.fit_timeout.total_seconds(),
                )
            except Exception as exc:
                logger.warning(
                    "backfill_fit_failed",
                    group=group.name,
                    query=query.id,
                    model=spec.name,
                    error=str(exc),
                )
                continue
            produced.fit_count.append(spec.name)
            curve_points = list(factory.forecast_curve_points(forecast, spec.name, selection=None))
            # Mirror the live runner's count-aware emission policy so a
            # count-shaped query (``data_profile: count`` or explicit
            # ``clip_non_negative`` / ``round_to_integer``) doesn't
            # silently push negative or fractional values into the
            # historical TSDB. The curve helper is non-mutating so the
            # raw output is preserved upstream.
            if clip or rounder:
                curve_points = _apply_safeguards_to_curve_points(
                    curve_points, clip=clip, rounder=rounder
                )
            produced.curve.extend(
                _rebase_curve_to_as_of(curve_points, as_of=as_of, step_seconds=step_seconds)
            )
            produced.curve.extend(
                _accuracy_curve_points(
                    cleaned_df=df,
                    model=model,
                    model_name=spec.name,
                    freq=freq,
                    levels=levels,
                    horizons=accuracy_mod.horizons_from_durations(
                        list(d.accuracy.horizons), eff_step
                    ),
                    season_length=eff_season_length,
                    factory=factory,
                    as_of=as_of,
                )
            )
    return produced


def _rebase_curve_to_as_of(
    points: list[ForecastCurvePoint],
    *,
    as_of: datetime,
    step_seconds: int,
) -> Iterable[ForecastCurvePoint]:
    """Rewrite curve point timestamps so they anchor at ``as_of``.

    :func:`SeriesPointFactory.forecast_curve_points` stamps each
    point with ``now + i * step``; for backfill we want
    ``as_of + i * step`` so the historical curves line up with what
    a scrape at ``as_of`` would have observed. Counts grow with
    ``step_seconds`` so the time delta is computed once per curve
    rather than per point.
    """
    if not points:
        return ()
    first_ts_ms = min(p.timestamp_ms for p in points)
    target_first_ms = int(as_of.timestamp() * 1000)
    shift_ms = target_first_ms - first_ts_ms
    return [
        ForecastCurvePoint(
            metric=p.metric,
            labels=p.labels,
            value=p.value,
            timestamp_ms=p.timestamp_ms + shift_ms,
        )
        for p in points
    ]


def _accuracy_curve_points(
    *,
    cleaned_df: pd.DataFrame,
    model: Any,
    model_name: str,
    freq: str,
    levels: list[int],
    horizons: list[accuracy_mod.HorizonSpec],
    season_length: int,
    factory: SeriesPointFactory,
    as_of: datetime,
) -> Iterable[ForecastCurvePoint]:
    """Backtest the model at ``as_of`` and emit accuracy + quality samples.

    Pushes per-horizon ``forecast_accuracy_mape`` /
    ``forecast_accuracy_mase`` samples and a composite
    ``forecast_quality_score`` sample, all stamped with ``as_of``.
    Failures are swallowed — the curve is the load-bearing emission;
    the operational summaries are a bonus.
    """
    if not horizons:
        return ()
    try:
        outcome = accuracy_mod.backtest(
            model=model,
            df=cleaned_df,
            horizon_steps=horizons[0].steps,
            freq=freq,
            levels=levels,
        )
    except Exception:
        return ()
    if outcome is None:
        return ()
    test, predictions = outcome
    results = accuracy_mod.evaluate(
        test=test,
        predictions=predictions,
        train=cleaned_df.iloc[: -horizons[0].steps].reset_index(drop=True),
        season_length=season_length,
        horizons=horizons,
        model_name=model_name,
    )
    if not results:
        return ()
    as_of_ms = int(as_of.timestamp() * 1000)
    out: list[ForecastCurvePoint] = []
    for r in results:
        common = factory.drift_labels(model_name)
        if math.isfinite(r.mape):
            out.append(
                ForecastCurvePoint(
                    metric="forecast_accuracy_mape",
                    labels={**common, "horizon": r.horizon},
                    value=r.mape,
                    timestamp_ms=as_of_ms,
                )
            )
        if math.isfinite(r.mase):
            out.append(
                ForecastCurvePoint(
                    metric="forecast_accuracy_mase",
                    labels={**common, "horizon": r.horizon},
                    value=r.mase,
                    timestamp_ms=as_of_ms,
                )
            )

    # Quality score: same machinery the live runner uses
    # (:func:`promforecast.quality.compute`). We don't have a
    # post-fit band-width measurement in the backfill path so the
    # helper falls back to the standard sub-score for missing input;
    # ``coverage`` is assumed 1.0 because the cleaned dataframe is
    # the lookback the runner would have used.
    mape_value = overall_mape(results)
    mase_value = overall_mase(results)
    score_kind: quality_mod.ScoreKind = "mape" if mape_value is not None else "mase"
    score_value = mape_value if score_kind == "mape" else mase_value
    quality = quality_mod.compute(
        backtest_mape=score_value,
        band_width_relative=None,
        coverage=1.0,
        imputed_fraction=0.0,
        score_kind=score_kind,
    )
    if math.isfinite(quality):
        out.append(
            ForecastCurvePoint(
                metric="forecast_quality_score",
                labels=factory.drift_labels(model_name),
                value=quality,
                timestamp_ms=as_of_ms,
            )
        )
    return out


def _iter_backfill_timestamps(
    config: Config,
    group: GroupConfig,
    from_seconds: float,
    to_time: datetime,
) -> Iterable[datetime]:
    """Yield each "as-of" timestamp the backfill should issue fits at.

    The cadence is the group's effective refresh interval — the
    shortest interval across the group's queries — so the backfill
    samples match what the live runner would have produced for any
    query in that group.
    """
    cadence = min(
        (effective_refresh_interval(q, config.server) for q in group.queries),
        default=config.server.refresh_interval,
    )
    cadence_s = max(60.0, cadence.total_seconds())
    earliest = to_time - timedelta(seconds=from_seconds)
    current = earliest
    while current <= to_time:
        yield current
        current = current + timedelta(seconds=cadence_s)


def _select_groups(config: Config, selector: list[str] | None) -> list[GroupConfig]:
    """Filter ``config.groups`` against the operator's ``--groups`` flag."""
    if not selector:
        return list(config.groups)
    wanted = set(selector)
    return [g for g in config.groups if g.name in wanted]


def _check_window(config: Config, from_seconds: float) -> None:
    """Reject backfill windows larger than ``safety.backfill_max_window``."""
    cap = config.safety.backfill_max_window.total_seconds()
    if cap > 0 and from_seconds > cap:
        raise BackfillError(
            f"--from {from_seconds:.0f}s exceeds safety.backfill_max_window "
            f"({cap:.0f}s); raise the cap explicitly when the cluster is sized "
            "for the work. See docs/operations/backfill.md."
        )


def _parse_duration_seconds(value: str, *, flag: str) -> float:
    """Resolve a Prometheus-style duration to seconds.

    Reuses :func:`config.parse_duration` so the CLI accepts the same
    ``5m`` / ``1h`` / ``30d`` syntax the YAML schema uses.
    """
    parsed = config_module.parse_duration(value)
    if not isinstance(parsed, timedelta):
        raise BackfillError(f"{flag} must be a Prometheus duration; got {value!r}")
    seconds = parsed.total_seconds()
    if seconds <= 0:
        raise BackfillError(f"{flag} must be a positive duration; got {value!r}")
    return seconds


def _parse_to_timestamp(value: str | None) -> datetime:
    """Resolve ``--to`` to an aware UTC datetime; ``None`` / ``now`` -> now."""
    if value is None or value.lower() == "now":
        return datetime.now(tz=UTC)
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise BackfillError(f"--to {value!r} must be 'now' or an ISO-8601 timestamp") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt
