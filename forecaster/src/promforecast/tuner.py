"""Hyperparameter recommendation runner.

The live forecaster pins ``season_length``, ``defaults.models``, and any
per-query overrides at config time and rarely revisits them. As the
underlying signal drifts the pinned values can quietly become wrong;
the only visible symptom is a slowly-degrading
``forecast_quality_score`` or a creeping ``forecast_accuracy_mape``,
both of which read as "the model is noisier" rather than "your config
is stale".

This module evaluates a small grid of alternative hyperparameter
choices per (group, query, model) against the same backtest harness
the live runner uses (:mod:`promforecast.accuracy`) and emits one
:class:`RecommendedParam` row per configured + candidate pair whose
backtest MAPE beats the configured baseline by a non-trivial margin.

Operators read the recommendations from a dedicated Grafana panel
(``promforecast_recommended_param`` is a snapshot-style gauge so a
stale recommendation drops as soon as the operator re-runs the
tuner after applying the change), pick the rows with the largest
``expected_mape_improvement_pct``, and apply them by editing the
config. The tuner never auto-applies — the operator stays in
control.

Runs as a one-shot pod separate from the live forecaster (typical
shape: a weekly ``CronJob``, shipped via the
``promforecast-tuner`` sub-chart) so a long grid walk does not
contend with the scheduled fit budget. The live forecaster's
snapshot is never touched — the tuner is a pure read-side operation
against the long-term TSDB plus an opt-in push of the resulting
gauges via the configured ``sink.remote_write`` target.

**Scope.** The shipped grid is intentionally small: one alternative
``season_length`` per query (auto-detected from the recent
periodogram), and the full configured ``defaults.models`` list as
the model-choice axis. A larger grid (damping parameters, per-model
hyperparams) is a future expansion; the metric shape and the
operator-visible workflow stay identical.
"""

from __future__ import annotations

import asyncio
import math
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import IO, TYPE_CHECKING

import structlog

from . import accuracy as accuracy_mod
from . import config as config_module
from . import models as model_registry
from . import preprocess as preprocess_mod
from .accuracy import overall_mape
from .config import Config, GroupConfig, QueryConfig, effective_season_length, effective_step
from .preview import _pandas_freq
from .sink import ForecastCurvePoint, ForecastSink, RemoteWriteSink, SinkError
from .source import PromSource

if TYPE_CHECKING:
    import pandas as pd

logger = structlog.get_logger(__name__)


class TunerError(RuntimeError):
    """Raised for any operator-visible tuner failure.

    Carries an exit code so ``main.cli`` can hand it straight to
    ``sys.exit``.
    """

    def __init__(self, message: str, *, exit_code: int = 1) -> None:
        super().__init__(message)
        self.exit_code = exit_code


@dataclass(frozen=True)
class RecommendedParam:
    """One recommendation row produced by a tuner run.

    Emitted as a sample of ``forecast_recommended_param`` — the value
    carries the ``expected_mape_improvement_pct`` so dashboards can
    sort by impact without parsing the label.
    """

    group: str
    query: str
    model: str
    name: str
    current_value: str
    recommended_value: str
    expected_mape_improvement_pct: float


@dataclass(frozen=True)
class TunerStats:
    """Per-run summary returned for logs + the CLI footer."""

    queries_evaluated: int
    candidates_evaluated: int
    recommendations_emitted: int
    failed_queries: int
    elapsed_seconds: float
    outcome: str  # ``ok | partial | error``


def run_tuner(  # noqa: PLR0911
    *,
    config_path: Path,
    group_selector: list[str] | None,
    sink_url_override: str | None,
    lookback_override: str | None = None,
    min_improvement_pct: float = 5.0,
    out: IO[str] | None = None,
) -> int:
    """Top-level entry point used by ``main.cli``.

    Returns ``0`` when at least one recommendation was emitted or when
    the run completed cleanly with no recommendations (current config
    is already optimal); ``1`` for invalid inputs (config issues,
    missing sink); ``2`` for TSDB / network failure during the data
    fetch.
    """
    stream = out if out is not None else sys.stdout
    write = stream.write
    try:
        cfg = config_module.load(config_path)
    except FileNotFoundError:
        write(f"error: config not found: {config_path}\n")
        return 1
    except Exception as exc:
        write(f"error: config validation failed: {exc}\n")
        return 1

    sink_url = sink_url_override or cfg.sink.remote_write.url
    if not sink_url:
        write(
            "error: no sink configured; either set sink.remote_write.url in the config "
            "or pass --sink-url. The tuner pushes recommendations to the long-term TSDB "
            "via the same path the live forecaster uses.\n"
        )
        return 1

    lookback_seconds: int | None = None
    if lookback_override is not None:
        parsed = config_module.parse_duration(lookback_override)
        if not isinstance(parsed, timedelta):
            write(f"error: invalid --lookback: {lookback_override!r}\n")
            return 1
        lookback_seconds = int(parsed.total_seconds())

    selected_groups = _select_groups(cfg.groups, group_selector)
    if not selected_groups:
        write(
            "error: no groups selected; check --groups against the configured "
            f"group names ({[g.name for g in cfg.groups]!r})\n"
        )
        return 1

    sink = RemoteWriteSink(
        url=sink_url,
        timeout_seconds=cfg.sink.remote_write.timeout.total_seconds(),
        chunk_size=cfg.sink.remote_write.chunk_size,
        headers=cfg.sink.remote_write.headers,
        max_connections=cfg.sink.remote_write.max_connections,
        max_keepalive_connections=cfg.sink.remote_write.max_keepalive_connections,
    )
    source = PromSource(
        url=cfg.datasource.url,
        timeout_seconds=cfg.datasource.timeout.total_seconds(),
    )

    try:
        stats = asyncio.run(
            _run_async(
                cfg=cfg,
                source=source,
                sink=sink,
                groups=selected_groups,
                lookback_override_seconds=lookback_seconds,
                min_improvement_pct=min_improvement_pct,
            )
        )
    except TunerError as exc:
        write(f"error: {exc}\n")
        return exc.exit_code
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("tuner_unexpected_error")
        write(f"error: {exc}\n")
        return 2

    write(
        f"tuner: groups={len(selected_groups)} "
        f"queries={stats.queries_evaluated} "
        f"candidates={stats.candidates_evaluated} "
        f"recommendations={stats.recommendations_emitted} "
        f"failed={stats.failed_queries} "
        f"elapsed_s={stats.elapsed_seconds:.1f} "
        f"outcome={stats.outcome}\n"
    )
    return 0 if stats.outcome != "error" else 2


def _select_groups(groups: list[GroupConfig], selector: list[str] | None) -> list[GroupConfig]:
    if not selector:
        return list(groups)
    wanted = set(selector)
    return [g for g in groups if g.name in wanted]


async def _run_async(
    *,
    cfg: Config,
    source: PromSource,
    sink: ForecastSink,
    groups: list[GroupConfig],
    lookback_override_seconds: int | None,
    min_improvement_pct: float,
) -> TunerStats:
    start = time.time()
    queries_evaluated = 0
    candidates_evaluated = 0
    failed_queries = 0
    recommendations: list[RecommendedParam] = []

    # Wrap the evaluation + push in a single try/finally so the HTTP
    # clients on the source and sink are reliably released even if an
    # *unexpected* exception slips past ``_QueryEvaluationError``
    # (e.g. a hard import failure on the pandas branch, an asyncio
    # cancellation). Without this, a one-shot tuner pod that crashed
    # mid-walk would leak the keepalive pool until process exit.
    try:
        for group in groups:
            for query in group.queries:
                queries_evaluated += 1
                try:
                    cand_count, recs = await _evaluate_query(
                        cfg=cfg,
                        source=source,
                        group=group,
                        query=query,
                        lookback_override_seconds=lookback_override_seconds,
                        min_improvement_pct=min_improvement_pct,
                    )
                except _QueryEvaluationError as exc:
                    failed_queries += 1
                    logger.warning(
                        "tuner_query_failed",
                        group=group.name,
                        query=query.id,
                        error=str(exc),
                    )
                    continue
                candidates_evaluated += cand_count
                recommendations.extend(recs)

        # Always push the tuner run telemetry (timestamp + duration) — even
        # an all-failing run produces tracking metrics the operator wants
        # to see in Grafana.
        try:
            await _push_results(
                sink=sink,
                recommendations=recommendations,
                outcome=_run_outcome(failed_queries, len(groups)),
                elapsed_seconds=time.time() - start,
                groups=[g.name for g in groups],
            )
        except SinkError as exc:
            logger.error("tuner_sink_push_failed", reason=exc.reason, error=str(exc))
            raise TunerError(f"sink push failed ({exc.reason}): {exc}", exit_code=2) from exc
    finally:
        # Best-effort cleanup. Swallow per-client errors so a transient
        # close failure on one client doesn't mask the actual exception
        # raised by the body of the try block.
        try:
            await sink.aclose()
        except Exception:
            logger.warning("tuner_sink_aclose_failed")
        try:
            await source.aclose()
        except Exception:
            logger.warning("tuner_source_aclose_failed")

    elapsed = time.time() - start
    return TunerStats(
        queries_evaluated=queries_evaluated,
        candidates_evaluated=candidates_evaluated,
        recommendations_emitted=len(recommendations),
        failed_queries=failed_queries,
        elapsed_seconds=elapsed,
        outcome=_run_outcome(failed_queries, len(groups)),
    )


class _QueryEvaluationError(RuntimeError):
    """Internal sentinel — a single-query failure that's worth logging
    but mustn't take the whole tuner run down."""


async def _evaluate_query(  # noqa: PLR0912
    *,
    cfg: Config,
    source: PromSource,
    group: GroupConfig,
    query: QueryConfig,
    lookback_override_seconds: int | None,
    min_improvement_pct: float,
) -> tuple[int, list[RecommendedParam]]:
    """Run the grid over one (group, query) tuple. Returns recommendations."""
    import pandas as pd  # noqa: PLC0415  # heavy import: defer until needed

    defaults = cfg.defaults
    step_td = effective_step(query, defaults)
    lookback_seconds = (
        lookback_override_seconds
        if lookback_override_seconds is not None
        else int(defaults.lookback.total_seconds())
    )
    end = datetime.now(tz=UTC)
    start = end - timedelta(seconds=lookback_seconds)

    try:
        frames = await source.query_range(
            promql=query.promql,
            start=start,
            end=end,
            step_seconds=max(1, int(step_td.total_seconds())),
        )
    except Exception as exc:
        raise _QueryEvaluationError(f"source fetch failed: {exc}") from exc

    if not frames:
        raise _QueryEvaluationError("source returned no series")

    # Take the first series as the representative — the tuner's job is
    # to surface "your configured hyperparams are off for this query",
    # not "this specific series of this query is off". A single series
    # is sufficient to drive the comparative MAPE and keeps the per-run
    # cost bounded; the operator applies any recommendation at the
    # query level anyway.
    frame = frames[0]
    cleaned = preprocess_mod.preprocess_series(
        timestamps=list(frame.timestamps),
        values=list(frame.values),
        step=step_td,
        config=group.preprocess,
    )
    if cleaned.dropped or len(cleaned.values) < defaults.min_points:
        raise _QueryEvaluationError(
            f"series too short after preprocessing ({len(cleaned.values)} "
            f"points < min_points {defaults.min_points})"
        )

    df = pd.DataFrame(
        {
            "ds": cleaned.timestamps,
            "y": cleaned.values,
            "unique_id": ["series"] * len(cleaned.values),
        }
    )
    freq = _pandas_freq(max(1, int(step_td.total_seconds())))
    season_length = effective_season_length(query, defaults)
    horizons = accuracy_mod.horizons_from_durations(defaults.accuracy.horizons, step_td)
    if not horizons:
        raise _QueryEvaluationError("no usable accuracy horizons configured")

    # Evaluate the baseline for each emitted model first so candidates
    # have a number to beat. The baseline is the configured (model,
    # season_length) tuple as it stands today. ``QueryConfig`` has no
    # per-query ``models`` override today — every query inherits
    # ``defaults.models`` — so the grid axis is the defaults list.
    configured_model_names: list[str] = [spec.name for spec in defaults.models]
    candidates_evaluated = 0
    recommendations: list[RecommendedParam] = []
    for model_name in configured_model_names:
        baseline_mape = _fit_and_score(
            model_name=model_name,
            season_length=season_length,
            df=df,
            freq=freq,
            horizons=horizons,
        )
        if baseline_mape is None:
            continue
        candidates_evaluated += 1

        # Candidate axis 1: season_length grid (x0.5, x2.0 of configured).
        season_candidates = _season_length_grid(season_length)
        for candidate_season in season_candidates:
            cand_mape = _fit_and_score(
                model_name=model_name,
                season_length=candidate_season,
                df=df,
                freq=freq,
                horizons=horizons,
            )
            if cand_mape is None:
                continue
            candidates_evaluated += 1
            improvement = _improvement_pct(baseline_mape, cand_mape)
            if improvement >= min_improvement_pct:
                recommendations.append(
                    RecommendedParam(
                        group=group.name,
                        query=query.id,
                        model=model_name,
                        name="season_length",
                        current_value=str(season_length),
                        recommended_value=str(candidate_season),
                        expected_mape_improvement_pct=improvement,
                    )
                )

        # Candidate axis 2: alternative model from the configured list.
        # Compare each non-baseline model's MAPE; a candidate that beats
        # the baseline by ``min_improvement_pct`` becomes a model-choice
        # recommendation.
        for alt_model in configured_model_names:
            if alt_model == model_name:
                continue
            alt_mape = _fit_and_score(
                model_name=alt_model,
                season_length=season_length,
                df=df,
                freq=freq,
                horizons=horizons,
            )
            if alt_mape is None:
                continue
            candidates_evaluated += 1
            improvement = _improvement_pct(baseline_mape, alt_mape)
            if improvement >= min_improvement_pct:
                recommendations.append(
                    RecommendedParam(
                        group=group.name,
                        query=query.id,
                        model=model_name,
                        name="model",
                        current_value=model_name,
                        recommended_value=alt_model,
                        expected_mape_improvement_pct=improvement,
                    )
                )

    return candidates_evaluated, recommendations


def _fit_and_score(
    *,
    model_name: str,
    season_length: int,
    df: pd.DataFrame,
    freq: str,
    horizons: list[accuracy_mod.HorizonSpec],
) -> float | None:
    """Fit one candidate and return its overall MAPE.

    Returns ``None`` when the fit / backtest cannot produce a usable
    score (model unknown, fit error, all-NaN MAPE on a flat series).
    Callers treat ``None`` as "skip this candidate" and never as
    "0 improvement"; an all-NaN MAPE is not the same as a perfect fit.
    """
    try:
        model = model_registry.build(model_name, season_length=season_length)
    except model_registry.UnknownModelError:
        return None

    max_steps = max((h.steps for h in horizons), default=0)
    if max_steps < 1:
        return None
    try:
        bt = accuracy_mod.backtest(
            model=model,
            df=df,
            horizon_steps=max_steps,
            freq=freq,
            levels=[80],
        )
    except Exception:
        return None
    if bt is None:
        return None
    test, predictions = bt
    train = df.iloc[:-max_steps].reset_index(drop=True)
    results = accuracy_mod.evaluate(
        test=test,
        predictions=predictions,
        train=train,
        season_length=season_length,
        horizons=horizons,
        model_name=model_name,
    )
    mape = overall_mape(results)
    if mape is None or not math.isfinite(mape):
        return None
    return mape


#: Hard ceiling on the candidate ``season_length`` the tuner will fit.
#: Anything above this is rejected as a runaway: a configured value of
#: 10000 (already very large for typical 5-min-step daily seasonality)
#: would otherwise produce a doubled candidate at 20000 and burn
#: per-fit time for no real value.
_MAX_CANDIDATE_SEASON_LENGTH = 10000


def _season_length_grid(current: int) -> list[int]:
    """Pick the candidate season_length values to try around ``current``.

    Bounded to two-element grid (half + double) so the tuner stays
    cheap per query. A separate auto-seasonality story is the right
    home for FFT-driven detection; the tuner's grid is deliberately
    small enough to run weekly without dominating the cluster's CPU.
    """
    candidates: list[int] = []
    half = max(2, current // 2)
    double = current * 2
    if half != current:
        candidates.append(half)
    if double != current and double <= _MAX_CANDIDATE_SEASON_LENGTH:
        candidates.append(double)
    return candidates


def _improvement_pct(baseline: float, candidate: float) -> float:
    """Return the percentage improvement of ``candidate`` over ``baseline``.

    Positive values mean the candidate's MAPE is lower (better). When
    the baseline is zero (extremely unusual but possible) we fall back
    to 0 — dividing by zero would either explode or sign-flip into a
    meaningless recommendation.
    """
    if baseline <= 0:
        return 0.0
    return float(((baseline - candidate) / baseline) * 100.0)


def _run_outcome(failed_queries: int, group_count: int) -> str:
    if failed_queries == 0:
        return "ok"
    # The tuner is best-effort; a single bad query in a multi-query
    # group reports ``partial`` (any successful fit + any failure).
    # An "all failed" run reports ``error`` so an alert on
    # ``forecast_tuner_runs_total{outcome="error"}`` can detect the
    # tuner being globally broken (network, auth) rather than a single
    # query being unfit for the grid.
    if group_count == 0:
        return "error"
    return "partial"


async def _push_results(
    *,
    sink: ForecastSink,
    recommendations: list[RecommendedParam],
    outcome: str,
    elapsed_seconds: float,
    groups: list[str],
) -> None:
    """Push the recommendation + run-telemetry rows through ``sink``.

    Uses :class:`ForecastCurvePoint` to ride the existing sink's
    Prometheus text-format encoder; each row carries a timestamp at
    "now" so the recommendation is treated as the snapshot for the
    most recent tuner run.
    """
    timestamp_ms = int(time.time() * 1000)
    points: list[ForecastCurvePoint] = []
    for rec in recommendations:
        points.append(
            ForecastCurvePoint(
                metric="forecast_recommended_param",
                labels={
                    "group": rec.group,
                    "query": rec.query,
                    "model": rec.model,
                    "name": rec.name,
                    "current_value": rec.current_value,
                    "recommended_value": rec.recommended_value,
                },
                value=rec.expected_mape_improvement_pct,
                timestamp_ms=timestamp_ms,
            )
        )
    # Tuner-internal telemetry. The tuner is a stateless one-shot
    # process — pushing a counter sample of ``1.0`` from every run
    # produces a flat ``1.0, 1.0, 1.0, ...`` series in the TSDB whose
    # ``rate()`` is zero, which is not useful. Instead the tuner emits
    # two gauges keyed on the run timestamp:
    #
    # * ``forecast_tuner_last_run_timestamp_seconds{group, outcome}`` —
    #   unix timestamp of the most recent run. Operators alert on
    #   staleness (``time() - max by (group) (forecast_tuner_last_run_timestamp_seconds) > 8d``)
    #   and split-route on ``outcome`` for "tuner is failing".
    # * ``forecast_tuner_last_run_duration_seconds{group}`` —
    #   wall-clock duration of the most recent run.
    #
    # This matches the snapshot/pull pattern the rest of the codebase
    # uses for last-run telemetry and is the right shape for a
    # stateless one-shot pusher.
    timestamp_seconds = timestamp_ms / 1000.0
    for group_name in groups:
        points.append(
            ForecastCurvePoint(
                metric="forecast_tuner_last_run_timestamp_seconds",
                labels={"group": group_name, "outcome": outcome},
                value=timestamp_seconds,
                timestamp_ms=timestamp_ms,
            )
        )
        points.append(
            ForecastCurvePoint(
                metric="forecast_tuner_last_run_duration_seconds",
                labels={"group": group_name},
                value=elapsed_seconds,
                timestamp_ms=timestamp_ms,
            )
        )
    if not points:
        return
    await sink.write(points)
