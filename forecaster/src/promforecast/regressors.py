"""Exogenous regressor resolution.

Two flavours are supported:

* **Calendar features** (``hour_of_day``, ``day_of_week``, ``is_weekend``):
  derived deterministically from the series timestamps. They can never
  fail at runtime — by construction the values are always available, both
  in-sample (during fit) and out-of-sample (during forecast).
* **Custom regressors**: a PromQL query whose returned series is aligned
  to the primary query's time index via nearest-neighbour interpolation
  (``ds`` -> ``mean(value across instances)``). Failures are caught and
  surfaced as either a graceful skip (counter incremented, forecast still
  runs) or a fatal series-skip when ``regressors[].required: true``.

Resolution is split into two phases:

* :func:`prefetch_promql` runs once **per query** to fetch every custom
  regressor's PromQL. The fan-out is independent of how many series the
  primary query returned, which is critical: without this split a query
  returning 100 series with 2 PromQL regressors would issue 200 redundant
  requests against the TSDB every refresh interval.
* :func:`align_for_series` runs **per series** and is cheap: it picks the
  matching pre-fetched frames, aligns them via ``merge_asof``, and
  materialises calendar features.

The module is intentionally narrow: it only resolves *historical* regressor
values aligned to ``df.ds``. Forward (future-horizon) values are not yet
threaded through to the underlying statsforecast call — model wrappers
accept a ``regressors`` keyword today but built-ins ignore it. The plumbing
lives in :class:`Runner._fit_series` so an extension model (e.g.
``promforecast-prophet``) can pick the dataframe up without touching the
runner.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

import structlog

from .config import CALENDAR_FEATURE_TYPES, RegressorConfig

if TYPE_CHECKING:
    import pandas as pd

    from .source import PromSource, SeriesFrame

logger = structlog.get_logger(__name__)


class RegressorError(RuntimeError):
    """Raised when a ``required: true`` regressor cannot be resolved.

    Carries a short ``reason`` token suitable for use as a Prometheus
    counter label; the message is intended for log output.
    """

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class RegressorFailure:
    """Per-regressor failure record returned from ``resolve``.

    ``reason`` is a bounded token (``query_error`` | ``query_timeout`` |
    ``empty_result`` | ``invalid_type``) that the runner uses as a label on
    ``forecast_regressor_failures_total``. We keep the cardinality small so
    the counter doesn't explode on a runaway regressor.
    """

    regressor_id: str
    reason: str
    message: str


@dataclass(frozen=True)
class PromQLPrefetch:
    """Cached PromQL fetch results, ready for per-series alignment.

    Held once per primary query and reused for every series the query
    produces. ``failures`` already accounts for required-regressor errors
    — the caller is expected to re-raise via :class:`RegressorError` when
    appropriate; we surface them as data so the runner can also count
    optional-regressor failures *once* per query rather than per series.

    ``no_source`` carries the regressor-ids whose PromQL was skipped
    because the runner had no datasource (offline tests, dry-run); these
    are reported as a distinct failure reason.
    """

    fetched: dict[str, list[SeriesFrame]]
    failures: list[RegressorFailure]


@dataclass
class ResolveOutcome:
    """Aggregated regressor resolution result for one series.

    ``frame`` is ``None`` when no regressors were configured *or* when every
    optional regressor failed; downstream code treats both cases identically
    (forecast without exogenous variables).

    ``failures`` here only carries *per-series* failures (calendar features
    that mis-typed past the schema validator). PromQL fetch failures live
    on the :class:`PromQLPrefetch` and are surfaced once per query by the
    runner.
    """

    frame: pd.DataFrame | None
    failures: list[RegressorFailure]


async def prefetch_promql(
    *,
    regressors: list[RegressorConfig],
    source: PromSource | None,
    query_start: datetime,
    query_end: datetime,
    step_seconds: int,
    query_timeout_seconds: float,
) -> PromQLPrefetch:
    """Fan out every custom regressor's PromQL **once per primary query**.

    Calendar regressors are deferred to per-series alignment — they're
    pure functions of the timestamps and don't benefit from caching.
    Required-regressor failures raise :class:`RegressorError` so the
    caller (the runner's ``_run_query``) can short-circuit the entire
    query: every series produced by that PromQL would have been skipped
    anyway, and emitting N copies of the same failure into the per-series
    counter would just be noise.
    """
    fetched: dict[str, list[SeriesFrame]] = {}
    failures: list[RegressorFailure] = []
    promql_regs = [r for r in regressors if r.type == "custom"]
    if not promql_regs:
        return PromQLPrefetch(fetched=fetched, failures=failures)

    if source is None:
        # No datasource attached (test/dry-run). Treat every PromQL
        # regressor as "no_source" so optional ones degrade gracefully and
        # required ones short-circuit the query.
        for reg in promql_regs:
            no_source = RegressorFailure(
                regressor_id=reg.id,
                reason="no_source",
                message="datasource is not available for regressor resolution",
            )
            if reg.required:
                raise RegressorError(no_source.reason, no_source.message)
            failures.append(no_source)
        return PromQLPrefetch(fetched=fetched, failures=failures)

    raw_results = await asyncio.gather(
        *(
            _fetch_one(
                source=source,
                promql=str(reg.promql),
                start=query_start,
                end=query_end,
                step_seconds=step_seconds,
                timeout_seconds=query_timeout_seconds,
            )
            for reg in promql_regs
        ),
        return_exceptions=True,
    )
    for reg, outcome in zip(promql_regs, raw_results, strict=True):
        failure = _failure_from_outcome(reg, outcome, source_present=True)
        if failure is not None:
            if reg.required:
                raise RegressorError(failure.reason, failure.message)
            failures.append(failure)
            continue
        assert isinstance(outcome, list)
        fetched[reg.id] = outcome
    return PromQLPrefetch(fetched=fetched, failures=failures)


def align_for_series(
    *,
    regressors: list[RegressorConfig],
    prefetch: PromQLPrefetch,
    timestamps: list[datetime],
) -> ResolveOutcome:
    """Build the per-series regressor frame from cached PromQL + calendar.

    Cheap path: nearest-asof alignment of pre-fetched frames + a few
    column derivations for calendar features. Fans out no network calls.

    A ``required: true`` PromQL regressor whose pre-fetch failed *before*
    this call would already have raised :class:`RegressorError` from
    :func:`prefetch_promql`; here we only handle the (very rare) case
    where a *calendar* regressor is required and its name escapes the
    boot-time validator (hot-reload races).
    """
    if not regressors:
        return ResolveOutcome(frame=None, failures=[])

    import pandas as pd  # noqa: PLC0415

    out = pd.DataFrame({"ds": pd.to_datetime(timestamps)})
    failures: list[RegressorFailure] = []

    for reg in regressors:
        if reg.type == "calendar":
            cal_failure = _apply_calendar(reg, out)
            if cal_failure is not None:
                if reg.required:
                    raise RegressorError(cal_failure.reason, cal_failure.message)
                failures.append(cal_failure)
            continue
        frames = prefetch.fetched.get(reg.id)
        if frames is None:
            # Pre-fetch failed for this optional regressor — already counted
            # in ``prefetch.failures``. Drop the column and move on.
            continue
        out[reg.id] = _aligned_values(frames, out["ds"])

    if len(out.columns) <= 1:  # only the ``ds`` column → no regressors usable
        return ResolveOutcome(frame=None, failures=failures)
    return ResolveOutcome(frame=out, failures=failures)


def _apply_calendar(reg: RegressorConfig, out: pd.DataFrame) -> RegressorFailure | None:
    """Materialise a calendar feature column or return a failure record.

    Calendar features themselves never fail at runtime — the only way to
    land in the failure path here is a config that named a feature outside
    ``CALENDAR_FEATURE_TYPES``. The schema validator catches that at boot,
    so this branch is defence in depth for hot-reload races.
    """
    try:
        out[reg.id] = _calendar_feature(reg.id, out["ds"])
    except ValueError as exc:
        return RegressorFailure(regressor_id=reg.id, reason="invalid_type", message=str(exc))
    return None


def _failure_from_outcome(
    reg: RegressorConfig,
    outcome: object,
    *,
    source_present: bool,
) -> RegressorFailure | None:
    """Translate a per-regressor fetch result into a failure (or None on success)."""
    if not source_present:
        return RegressorFailure(
            regressor_id=reg.id,
            reason="no_source",
            message="datasource is not available for regressor resolution",
        )
    # ``asyncio.TimeoutError`` aliases ``TimeoutError`` on 3.11+, but keep
    # both names listed so the intent is obvious to a reader scanning for
    # "what counts as a timeout here?"
    if isinstance(outcome, (asyncio.TimeoutError, TimeoutError)):
        return RegressorFailure(
            regressor_id=reg.id,
            reason="query_timeout",
            message=f"regressor query timed out: {reg.promql!r}",
        )
    if isinstance(outcome, BaseException):
        return RegressorFailure(
            regressor_id=reg.id,
            reason="query_error",
            message=f"regressor query failed: {outcome}",
        )
    if not outcome:
        return RegressorFailure(
            regressor_id=reg.id,
            reason="empty_result",
            message=f"regressor query returned no series: {reg.promql!r}",
        )
    return None


async def _fetch_one(
    *,
    source: PromSource,
    promql: str,
    start: datetime,
    end: datetime,
    step_seconds: int,
    timeout_seconds: float,
) -> list[SeriesFrame]:
    return await asyncio.wait_for(
        source.query_range(promql, start, end, step_seconds),
        timeout=timeout_seconds,
    )


def _calendar_feature(name: str, ds: pd.Series[Any]) -> pd.Series[Any]:
    """Compute a deterministic calendar feature aligned to ``ds``.

    Only the feature names enumerated in ``CALENDAR_FEATURE_TYPES`` are
    valid; unknown names raise ``ValueError`` so the load-time validator
    in ``config.py`` is the canonical entry point for the allow-list.
    """
    if name not in CALENDAR_FEATURE_TYPES:
        raise ValueError(f"unknown calendar feature: {name!r}")
    if name == "hour_of_day":
        result: pd.Series[Any] = ds.dt.hour.astype(float)
        return result
    if name == "day_of_week":
        result = ds.dt.dayofweek.astype(float)
        return result
    # is_weekend
    weekday_threshold = 5  # Saturday and Sunday are >= 5 in pandas' 0=Monday encoding
    result = (ds.dt.dayofweek >= weekday_threshold).astype(float)
    return result


def _aligned_values(frames: list[SeriesFrame], ds: pd.Series[Any]) -> pd.Series[Any]:
    """Reduce a multi-series PromQL result to a single value-per-timestamp.

    PromQL regressors typically aggregate (e.g. ``count by () (deploys)``)
    but we can't rely on that. We collapse to a per-timestamp mean across
    every returned series, then nearest-neighbour-align to ``ds``. NaNs
    propagate through to the frame so downstream models see a missing
    value rather than a fabricated zero.
    """
    import pandas as pd  # noqa: PLC0415

    rows: list[tuple[pd.Timestamp, float]] = []
    for frame in frames:
        for ts, val in zip(frame.timestamps, frame.values, strict=True):
            rows.append((pd.Timestamp(ts), float(val)))
    if not rows:
        return pd.Series([float("nan")] * len(ds))
    src_grouped: pd.DataFrame = (
        pd.DataFrame(rows, columns=["ds", "v"]).groupby("ds", as_index=False).mean()
    )
    src_sorted: pd.DataFrame = src_grouped.sort_values(by="ds").reset_index(drop=True)
    target_frame: pd.DataFrame = pd.DataFrame({"ds": ds}).reset_index(drop=True)
    target_sorted: pd.DataFrame = target_frame.sort_values(by="ds")
    aligned: pd.DataFrame = pd.merge_asof(target_sorted, src_sorted, on="ds", direction="nearest")
    out: pd.Series[Any] = aligned["v"].reset_index(drop=True)
    return out
