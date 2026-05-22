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

* :func:`prefetch_promql` runs once **per query** to (a) fetch every
  custom regressor's PromQL and (b) collapse each fetched response to a
  sorted ``(ds, v)`` frame. Both pieces of work are independent of how
  many series the primary query returned, which is critical: without
  this split a query returning 100 series with 2 PromQL regressors
  would issue 200 redundant TSDB requests *and* repeat the per-regressor
  groupby/sort 100 times every refresh interval.
* :func:`align_for_series` runs **per series** and is cheap *because*
  the prefetch already did the reduction: it picks the pre-aggregated
  frame, runs ``merge_asof`` against the per-series timestamps, and
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

# Imported lazily inside ``_apply_holidays`` so the ``holidays`` package
# is loaded only when a config actually configures a holiday regressor.
# Module-level import here would add a few MB of country-tables to every
# import of ``regressors`` for the (common) case of no holiday usage.

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

    ``aggregated`` carries the per-regressor reduction (mean across
    returned series, grouped + sorted by ``ds``) that per-series alignment
    needs. We compute it once per query so a primary query returning N
    series doesn't repeat the reduction N times — that loop dominated the
    refresh CPU profile when regressors had many points or many returned
    series. Mapping is keyed by regressor id; an entry is absent when the
    pre-fetch produced no usable rows. ``None`` means the prefetch path
    was bypassed (legacy callers / tests construct ``PromQLPrefetch``
    directly with raw frames); :func:`align_for_series` falls back to
    building it on the fly in that case.
    """

    fetched: dict[str, list[SeriesFrame]]
    failures: list[RegressorFailure]
    aggregated: dict[str, pd.DataFrame] | None = None


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
    # Reduce each regressor's frames to a sorted (ds, v) frame *once* per
    # primary query. Per-series alignment becomes a cheap merge_asof against
    # this pre-aggregated frame instead of re-running groupby+sort N times,
    # where N is the primary query's series count.
    aggregated = _aggregate_fetched(fetched)
    return PromQLPrefetch(fetched=fetched, failures=failures, aggregated=aggregated)


def _aggregate_fetched(
    fetched: dict[str, list[SeriesFrame]],
) -> dict[str, pd.DataFrame]:
    """Collapse each regressor's series frames to a sorted ``(ds, v)`` frame.

    A regressor is omitted from the result when none of its frames yielded
    a usable row — :func:`align_for_series` then treats it the same as a
    missing pre-fetch.
    """
    aggregated: dict[str, pd.DataFrame] = {}
    for reg_id, frames in fetched.items():
        df = _aggregate_frames(frames)
        if df is not None:
            aggregated[reg_id] = df
    return aggregated


def _aggregate_frames(frames: list[SeriesFrame]) -> pd.DataFrame | None:
    """Mean-across-series + sort-by-ds reduction. ``None`` when no rows.

    PromQL regressors typically already aggregate (e.g. ``count by ()``)
    but we don't rely on that — collapsing to a per-timestamp mean is the
    safe default. NaN values were already filtered upstream by
    :func:`source._parse_matrix`, so we don't re-check here.
    """
    import pandas as pd  # noqa: PLC0415

    rows: list[tuple[pd.Timestamp, float]] = []
    for frame in frames:
        for ts, val in zip(frame.timestamps, frame.values, strict=True):
            rows.append((pd.Timestamp(ts), float(val)))
    if not rows:
        return None
    src_grouped: pd.DataFrame = (
        pd.DataFrame(rows, columns=["ds", "v"]).groupby("ds", as_index=False).mean()
    )
    return src_grouped.sort_values(by="ds").reset_index(drop=True)


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
        if reg.type == "holidays":
            hol_failure = _apply_holidays(reg, out)
            if hol_failure is not None:
                if reg.required:
                    raise RegressorError(hol_failure.reason, hol_failure.message)
                failures.append(hol_failure)
            continue
        aggregated = _resolve_aggregated(prefetch, reg.id)
        if aggregated is None:
            # Pre-fetch failed for this optional regressor — already counted
            # in ``prefetch.failures``. Drop the column and move on.
            continue
        out[reg.id] = _aligned_values(aggregated, out["ds"])

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


def _apply_holidays(reg: RegressorConfig, out: pd.DataFrame) -> RegressorFailure | None:
    """Materialise holiday regressor column(s) or return a failure record.

    Two emission shapes are supported:

    * ``expand: false`` (default) — a single binary column named ``<id>``
      that is ``1`` on any holiday in the configured country/region
      calendar and ``0`` otherwise.
    * ``expand: true`` — one column per distinct holiday name seen in the
      ``ds`` window, named ``<id>__<holiday_slug>``. Operators read each
      coefficient independently ("Thanksgiving's effect is X, Black
      Friday's is Y").

    The ``holidays`` PyPI package is imported lazily so a deployment that
    never configures the regressor doesn't pay the country-table import
    cost. The package is shipped as the ``holidays`` extras dependency;
    missing-install errors surface as a graceful per-regressor failure
    with a message that names the extra.

    The ``timezone`` field on the regressor controls calendar-date
    resolution: ``ds`` is converted into the configured zone before
    ``.dt.date`` so a US holiday flips at local midnight rather than
    UTC-midnight. Without the conversion every non-UTC deployment would
    mis-label holidays by several hours.
    """
    if not reg.country:
        # Defensive: the schema validator rejects this at boot, but a
        # hot-reload race could land us here with an empty country.
        return RegressorFailure(
            regressor_id=reg.id,
            reason="invalid_type",
            message="holiday regressor missing country",
        )
    try:
        calendar = _build_holiday_calendar(reg)
    except ImportError as exc:
        return RegressorFailure(
            regressor_id=reg.id,
            reason="invalid_type",
            message=(
                "holidays package not installed; install the 'holidays' extra: "
                f"pip install 'promforecast[holidays]' (underlying error: {exc})"
            ),
        )
    except Exception as exc:
        return RegressorFailure(
            regressor_id=reg.id,
            reason="invalid_type",
            message=f"holidays calendar build failed: {exc}",
        )

    ds = out["ds"]
    # Operators configure ``timezone`` as an IANA name; the source
    # timestamps come from PromQL queries in UTC. ``tz_convert`` rebases
    # without changing the underlying instant so the eventual
    # ``.dt.date`` extraction matches the operator's local calendar.
    if ds.dt.tz is None:
        ds_local = ds.dt.tz_localize("UTC").dt.tz_convert(reg.timezone)
    else:
        ds_local = ds.dt.tz_convert(reg.timezone)
    dates = ds_local.dt.date
    if reg.expand:
        # One-hot: enumerate the distinct holiday names observed in the
        # ``ds`` window and emit one column per name. ``slug`` lower-cases
        # and underscores so the column is a valid Python identifier and
        # plays nicely with downstream model wrappers that introspect
        # ``regressors_frame.columns``.
        names = {calendar.get(d) for d in dates if calendar.get(d)}
        names.discard(None)
        for name in names:
            slug = _slugify_holiday(str(name))
            col = f"{reg.id}__{slug}"
            out[col] = [1.0 if calendar.get(d) == name else 0.0 for d in dates]
        if not names:
            # No holidays in the window — emit the binary column so the
            # frame always carries at least one explicit ``0`` regressor
            # column rather than silently disappearing.
            out[reg.id] = [0.0] * len(dates)
    else:
        out[reg.id] = [1.0 if calendar.get(d) else 0.0 for d in dates]
    return None


def _build_holiday_calendar(reg: RegressorConfig) -> Any:
    """Resolve the ``holidays.country_holidays`` calendar for ``reg``.

    ``regions`` is forwarded as ``subdiv`` because that is the keyword the
    ``holidays`` API uses; multiple regions collapse into the union by
    iterating + ``update()`` so a deployment can opt into, say,
    ``["CA", "NY"]`` and pick up both US states' bank holidays.

    Per-region calendars are memoised on ``(country, subdiv)`` — a config
    with dozens of holiday regressors against the same country (a common
    pattern with one regressor per query) would otherwise re-materialise
    the (multi-megabyte) date tables on every config load.

    Imports ``holidays`` lazily so the wider regressor module doesn't pay
    the country-tables import cost when no holiday regressor is
    configured.
    """
    # ``country`` is non-empty by validation contract (the caller checks),
    # but mypy can't see across that branch — cast through ``str(...)`` so
    # the holidays stubs accept it without a noqa.
    country = str(reg.country)
    if not reg.regions:
        return _holiday_calendar_cache_get(country, None)
    base: Any = _holiday_calendar_cache_get(country, reg.regions[0])
    if len(reg.regions) == 1:
        return base
    # Union of multiple subdivisions. ``HolidayBase.update`` mutates the
    # left operand, so we copy the cached base before merging in extras —
    # otherwise the cached single-region calendar would acquire every
    # other region's holidays after the first multi-region load.
    base = base.copy()
    for subdiv in reg.regions[1:]:
        base.update(_holiday_calendar_cache_get(country, subdiv))
    return base


# Process-wide ``(country, subdiv) -> HolidayBase`` cache. The
# ``holidays`` package returns a fresh, fully-materialised calendar on
# every ``country_holidays(...)`` call; with one regressor per query the
# config load alone would replay that work hundreds of times.
_HOLIDAY_CALENDAR_CACHE: dict[tuple[str, str | None], Any] = {}


def _holiday_calendar_cache_get(country: str, subdiv: str | None) -> Any:
    """Return a cached holiday calendar for ``(country, subdiv)``.

    Lazy-imports ``holidays`` so the wider module stays cheap. The
    returned object is shared — callers that need to mutate the
    calendar (e.g. unioning multiple regions) must ``.copy()`` first.
    """
    key = (country, subdiv)
    cached = _HOLIDAY_CALENDAR_CACHE.get(key)
    if cached is not None:
        return cached
    import holidays  # noqa: PLC0415

    if subdiv is None:
        calendar: Any = holidays.country_holidays(country)
    else:
        calendar = holidays.country_holidays(country, subdiv=subdiv)
    _HOLIDAY_CALENDAR_CACHE[key] = calendar
    return calendar


def _slugify_holiday(name: str) -> str:
    """Lower-case, ASCII-friendly slug suitable for a column name.

    The ``holidays`` package can emit names with apostrophes, parens, or
    multiple spaces (e.g. ``"New Year's Day"``). We collapse runs of
    non-alphanumerics into a single underscore and lowercase so the
    resulting column is stable across runs.
    """
    out_chars: list[str] = []
    last_was_us = False
    for ch in name.lower():
        if ch.isalnum():
            out_chars.append(ch)
            last_was_us = False
        elif not last_was_us:
            out_chars.append("_")
            last_was_us = True
    slug = "".join(out_chars).strip("_")
    return slug or "holiday"


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


def _resolve_aggregated(prefetch: PromQLPrefetch, reg_id: str) -> pd.DataFrame | None:
    """Return the pre-aggregated frame for ``reg_id`` (or build one lazily).

    ``prefetch_promql`` populates ``aggregated`` eagerly so the production
    path skips the reduction entirely. Direct ``PromQLPrefetch(...)``
    callers (tests, future helpers) leave it ``None``; we then fall back
    to building the aggregation on demand from ``fetched`` so behaviour
    stays identical.
    """
    if prefetch.aggregated is not None:
        return prefetch.aggregated.get(reg_id)
    frames = prefetch.fetched.get(reg_id)
    if frames is None:
        return None
    return _aggregate_frames(frames)


def _aligned_values(src_sorted: pd.DataFrame, ds: pd.Series[Any]) -> pd.Series[Any]:
    """Nearest-asof-align a pre-aggregated regressor frame onto ``ds``.

    The expensive collapse-and-sort already happened once per primary
    query in :func:`_aggregate_frames`; this is a per-series ``merge_asof``
    + reset_index. NaNs propagate through to the frame so downstream
    models see a missing value rather than a fabricated zero.
    """
    import pandas as pd  # noqa: PLC0415

    if src_sorted.empty:
        return pd.Series([float("nan")] * len(ds))
    target_frame: pd.DataFrame = pd.DataFrame({"ds": ds}).reset_index(drop=True)
    target_sorted: pd.DataFrame = target_frame.sort_values(by="ds")
    aligned: pd.DataFrame = pd.merge_asof(target_sorted, src_sorted, on="ds", direction="nearest")
    out: pd.Series[Any] = aligned["v"].reset_index(drop=True)
    return out
