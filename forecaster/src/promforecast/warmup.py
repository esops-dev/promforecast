"""Forecast warm-up status.

Reports per-(group, query) "what is still loading?" status for a fresh
install. The output answers the four questions an operator asks during
the first cycle after ``helm install``:

* How many lookback samples does this query already have?
* How many does it need before the runner will fit a model?
* What is currently blocking it (insufficient lookback, missing
  regressor, ...)?
* Roughly how long until the first usable forecast?

The status is computed on demand — there is no background poll. Each
call issues one cheap PromQL probe per configured query (a small
``query_range`` over the lookback window, returning only the
timestamps the source actually has) plus one additional probe per
required custom regressor when the primary's point count has cleared
``min_points``. Required-regressor probes fan out concurrently with
the primary and each other.

The endpoint is opt-in via ``server.expose_warmup_endpoint`` because
even the "one cheap probe per query" cost is wasteful on the steady-
state path; the operator workflow is "hit it a few times during the
first refresh, then never again". For the same reason there is no
caching layer — a stale cached answer is more confusing than a fresh
500ms probe.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Literal

import structlog

from .config import Config, GroupConfig, QueryConfig, RegressorConfig, effective_step

if TYPE_CHECKING:
    from .source import PromSource, SeriesFrame

logger = structlog.get_logger(__name__)


# Bounded enum of ``blocked_by`` reasons the runner can plausibly
# report from a probe. ``regressor:<id>`` is the templated case — we
# pin the prefix here so callers can match on it without scraping.
BlockedBy = Literal[
    "none",
    "min_points",
    "lookback",
    "change_point",
    "discovery",
]


@dataclass(frozen=True)
class QueryWarmupStatus:
    """Per-query warm-up status row in the warmup report."""

    id: str
    status: Literal["ready", "warming", "error"]
    points_available: int
    points_needed: int
    blocked_by: str  # one of ``BlockedBy`` or ``regressor:<id>``
    eta_seconds: float

    def to_dict(self) -> dict[str, Any]:
        """Render as a JSON-friendly dict."""
        return {
            "id": self.id,
            "status": self.status,
            "points_available": self.points_available,
            "points_needed": self.points_needed,
            "blocked_by": self.blocked_by,
            "eta_seconds": self.eta_seconds,
        }


@dataclass(frozen=True)
class GroupWarmupStatus:
    """Per-group warm-up status."""

    name: str
    queries: list[QueryWarmupStatus]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "queries": [q.to_dict() for q in self.queries],
        }


@dataclass(frozen=True)
class WarmupReport:
    """Full warm-up report."""

    groups: list[GroupWarmupStatus]

    def to_dict(self) -> dict[str, Any]:
        return {"groups": [g.to_dict() for g in self.groups]}


async def compute_warmup(
    *,
    config: Config,
    source: PromSource,
    query_timeout_seconds: float | None = None,
) -> WarmupReport:
    """Probe every configured query and return its warm-up status.

    Fan-out is concurrent across queries — the per-query probe is a
    cheap ``query_range`` that returns at most ``lookback / step``
    timestamps, so total wall-time is bounded by ``max(per-probe
    latency)`` rather than the sum.

    ``query_timeout_seconds`` overrides the per-probe timeout; without
    it the call inherits ``config.safety.query_timeout``. Probe
    failures degrade the row to ``status: "error"`` so a single
    bad PromQL never sinks the rest of the report.
    """
    timeout = (
        query_timeout_seconds
        if query_timeout_seconds is not None
        else config.safety.query_timeout.total_seconds()
    )
    tasks = [
        _probe_group(config=config, group=group, source=source, query_timeout_seconds=timeout)
        for group in config.groups
    ]
    groups = await asyncio.gather(*tasks) if tasks else []
    return WarmupReport(groups=list(groups))


async def _probe_group(
    *,
    config: Config,
    group: GroupConfig,
    source: PromSource,
    query_timeout_seconds: float,
) -> GroupWarmupStatus:
    """Probe every query in ``group`` concurrently."""
    tasks = [
        _probe_query(
            config=config,
            query=query,
            source=source,
            query_timeout_seconds=query_timeout_seconds,
        )
        for query in group.queries
    ]
    queries = await asyncio.gather(*tasks)
    return GroupWarmupStatus(name=group.name, queries=list(queries))


async def _probe_query(  # noqa: PLR0911
    *,
    config: Config,
    query: QueryConfig,
    source: PromSource,
    query_timeout_seconds: float,
) -> QueryWarmupStatus:
    """Run one probe and translate the result into a ``QueryWarmupStatus``.

    Event-profile queries (``data_profile: events``) and queries with
    discovery variables take separate runtime paths in the runner;
    here we report them as ``ready`` with no probe — they don't use
    ``min_points`` and a substitution-failing template would just
    surface as a probe error anyway.
    """
    eff_step = effective_step(query, config.defaults)
    step_seconds = max(1, int(eff_step.total_seconds()))
    min_points = config.defaults.min_points

    if query.data_profile == "events":
        # Event-profile queries don't have a min_points contract; they
        # produce the recurrence heatmap on every refresh once the
        # counter has at least one tick.
        return QueryWarmupStatus(
            id=query.id,
            status="ready",
            points_available=0,
            points_needed=0,
            blocked_by="none",
            eta_seconds=0.0,
        )

    if query.discover:
        # Templated queries depend on the discovery variable resolving
        # before we can probe the rendered string. Probing the raw
        # template (with the substitution placeholder still in it)
        # would just surface as a probe error and confuse the
        # operator, so we short-circuit. Surface the dependency
        # explicitly via ``blocked_by="discovery"`` rather than
        # ``none``: a templated query reporting ``ready`` + ``none``
        # implies "actively forecasting" which is misleading until
        # discovery has resolved. The status is still ``ready``
        # (callers that gate on status alone — including the CLI's
        # exit-code loop — continue to treat templated queries as
        # non-blocking) but the operator can read ``blocked_by`` as
        # "I can't probe this; check the discovery counters".
        return QueryWarmupStatus(
            id=query.id,
            status="ready",
            points_available=0,
            points_needed=0,
            blocked_by="discovery",
            eta_seconds=0.0,
        )

    # Probe the primary PromQL with a short query_range. Reusing the
    # configured lookback gives us the most-points-the-fit-would-see
    # number without a second query path.
    end = datetime.now(tz=UTC)
    start = end - config.defaults.lookback
    try:
        frames = await asyncio.wait_for(
            source.query_range(query.promql, start, end, step_seconds),
            timeout=query_timeout_seconds,
        )
    except TimeoutError:
        logger.warning("warmup_probe_timeout", query=query.id)
        return _error_status(query.id, min_points)
    except Exception as exc:
        logger.warning("warmup_probe_failed", query=query.id, error=str(exc))
        return _error_status(query.id, min_points)

    points_available = _points_in_longest_series(frames)

    if points_available < min_points:
        eta = _estimate_eta(
            points_available=points_available,
            points_needed=min_points,
            step_seconds=step_seconds,
        )
        return QueryWarmupStatus(
            id=query.id,
            status="warming",
            points_available=points_available,
            points_needed=min_points,
            blocked_by="min_points" if points_available > 0 else "lookback",
            eta_seconds=eta,
        )

    # Primary has enough points — check required PromQL regressors next.
    # A required regressor that the TSDB can't yet resolve (no samples in
    # the lookback) would cause the runner's per-series fit to skip with
    # ``forecast_failures_total{reason="required_regressor"}``. The
    # warmup endpoint exists to give operators that signal *before* the
    # fit attempt, so we issue one probe per required PromQL regressor
    # and surface the first one that comes back empty as the block.
    blocking_regressor = await _probe_required_regressors(
        query=query,
        source=source,
        step_seconds=step_seconds,
        lookback=config.defaults.lookback,
        query_timeout_seconds=query_timeout_seconds,
    )
    if blocking_regressor is not None:
        return QueryWarmupStatus(
            id=query.id,
            status="warming",
            points_available=points_available,
            points_needed=min_points,
            blocked_by=f"regressor:{blocking_regressor}",
            eta_seconds=float(step_seconds),
        )

    return QueryWarmupStatus(
        id=query.id,
        status="ready",
        points_available=points_available,
        points_needed=min_points,
        blocked_by="none",
        eta_seconds=0.0,
    )


async def _probe_required_regressors(
    *,
    query: QueryConfig,
    source: PromSource,
    step_seconds: int,
    lookback: timedelta,
    query_timeout_seconds: float,
) -> str | None:
    """Probe each required PromQL regressor; return the first empty one's id.

    Calendar / holiday regressors are skipped — they don't query the
    TSDB and so can't be "missing data". Only ``custom`` regressors
    with a PromQL string are probed.

    A probe failure (timeout / 5xx) is treated as "blocking" so the
    operator sees the dependency surface; the runner would log the
    same failure under ``forecast_regressor_failures_total`` so the
    decision is self-consistent.

    Probes are fanned out concurrently — total wall time is bounded by
    the slowest probe rather than the sum. The "blocking" result is
    deterministic across runs: ties resolve by config order (the first
    required regressor that came back empty/failed), independent of
    which network response landed first.
    """
    required = [
        reg for reg in query.regressors if reg.required and reg.type == "custom" and reg.promql
    ]
    if not required:
        return None
    end = datetime.now(tz=UTC)
    start = end - lookback

    async def _probe_one(reg: RegressorConfig) -> bool:
        """``True`` when the regressor is blocking (empty/failed)."""
        try:
            frames = await asyncio.wait_for(
                source.query_range(reg.promql or "", start, end, step_seconds),
                timeout=query_timeout_seconds,
            )
        except Exception:
            logger.warning(
                "warmup_required_regressor_probe_failed",
                query=query.id,
                regressor=reg.id,
            )
            return True
        return not frames or _points_in_longest_series(frames) == 0

    blocked_flags = await asyncio.gather(*(_probe_one(reg) for reg in required))
    for reg, blocked in zip(required, blocked_flags, strict=True):
        if blocked:
            return reg.id
    return None


def _error_status(query_id: str, min_points: int) -> QueryWarmupStatus:
    """Stable error row for a probe that timed out or raised."""
    return QueryWarmupStatus(
        id=query_id,
        status="error",
        points_available=0,
        points_needed=min_points,
        blocked_by="lookback",
        eta_seconds=math.inf,
    )


def _points_in_longest_series(frames: list[SeriesFrame]) -> int:
    """Pick the series with the most samples and return its length.

    A multi-series response is reported by the longest series rather
    than the mean — the runner's per-series fit treats each frame
    independently, so the "best off" series is what unblocks the
    query first. Using the mean would silently smear over a single
    slow-to-warm peer.
    """
    if not frames:
        return 0
    return max(len(frame.values) for frame in frames)


def _estimate_eta(
    *,
    points_available: int,
    points_needed: int,
    step_seconds: int,
) -> float:
    """Rough seconds-to-ready estimate from the current point count.

    Uses the simplest possible model — "each ``step`` adds one
    sample". That underestimates if the underlying series is sparse
    (TSDB downsampling, scrape misses) and overestimates the typical
    case slightly because backfill jobs may insert points faster
    than the scrape cadence. The estimate is meant to answer
    "minutes, hours, or days?" — not to be a hard deadline.
    """
    if points_available >= points_needed:
        return 0.0
    missing = points_needed - points_available
    return float(missing * step_seconds)


def render_table(report: WarmupReport) -> str:
    """Render the report as a plain-text aligned table for the CLI.

    Kept simple — three columns per query row, columns padded by
    rjust/ljust so the typical 80-column terminal is comfortable.
    """
    if not report.groups:
        return "no groups configured\n"
    rows: list[tuple[str, str, str, str, str, str, str]] = [
        ("group", "query", "status", "points", "needed", "eta", "blocked_by")
    ]
    for group in report.groups:
        for query in group.queries:
            eta_str = _format_eta(query.eta_seconds)
            rows.append(
                (
                    group.name,
                    query.id,
                    query.status,
                    str(query.points_available),
                    str(query.points_needed),
                    eta_str,
                    query.blocked_by,
                )
            )
    widths = [max(len(row[i]) for row in rows) for i in range(7)]
    lines: list[str] = []
    for i, row in enumerate(rows):
        # left-align text, right-align numbers
        formatted = " ".join(
            cell.ljust(widths[j]) if j in {0, 1, 2, 6} else cell.rjust(widths[j])
            for j, cell in enumerate(row)
        )
        lines.append(formatted)
        if i == 0:
            lines.append(" ".join("-" * w for w in widths))
    return "\n".join(lines) + "\n"


_SECONDS_PER_MINUTE = 60
_SECONDS_PER_HOUR = 3_600
_SECONDS_PER_DAY = 86_400


def _format_eta(eta_seconds: float) -> str:
    """Format an ETA value as a compact Prometheus-style duration."""
    if not math.isfinite(eta_seconds):
        return "∞"
    if eta_seconds <= 0:
        return "-"
    seconds = round(eta_seconds)
    if seconds >= _SECONDS_PER_DAY:
        return f"{seconds // _SECONDS_PER_DAY}d"
    if seconds >= _SECONDS_PER_HOUR:
        return f"{seconds // _SECONDS_PER_HOUR}h"
    if seconds >= _SECONDS_PER_MINUTE:
        return f"{seconds // _SECONDS_PER_MINUTE}m"
    return f"{seconds}s"
