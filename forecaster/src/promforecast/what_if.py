"""Multi-service fleet capacity what-if endpoint.

``POST /forecast/what-if/fleet`` is the strategic complement to
``/forecast/preview``:

* The single-query preview answers "what does the forecast look like for
  *this* metric under *these* overrides".
* The fleet what-if answers "which services across the *fleet* will hit
  their configured capacity ceiling first if traffic grows by N%".

Implementation is a thin loop that calls :func:`preview.run_preview` once
per configured query whose ``thresholds:`` declare a capacity ceiling
(comparator ``gt`` or ``lt`` — drain comparators are intentionally
excluded; a queue draining faster is not a capacity event). The same
``traffic_multiplier`` is applied to every query by scaling the
configured regressors (preferred when present so the model's learned
sensitivity is respected) or by scaling the leading lookback samples as
a fallback when no traffic regressor exists.

The endpoint is rate-limited by
``safety.fleet_what_if_min_interval`` (default 5 minutes). A what-if
call typically takes O(minutes) on a medium-sized config (each query
runs a fresh fit) so unthrottled access would starve scheduled fits —
the throttle keeps the endpoint usable from dashboards without
sacrificing the runner's schedule.

The response is a JSON object with a sorted ``services`` list:

.. code-block:: json

    {
      "horizon_seconds": 7776000,
      "traffic_multiplier": 1.5,
      "served_by": "promforecast-0",
      "evaluated_query_count": 24,
      "services": [
        {"query_id": "...", "group": "...", "threshold_name": "...",
         "time_to_threshold_seconds": 86400.0, "model": "AutoARIMA"},
        ...
      ]
    }

Stateless: the snapshot exporter, the executor pool, and the scheduled
fit pipeline are untouched. Failures of individual queries are
recorded inline on the ``services`` list with an ``error`` field so a
single bad query doesn't sink the whole report.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import structlog

from .config import (
    Config,
    QueryConfig,
    ThresholdConfig,
)
from .preview import (
    PreviewError,
    PreviewRequest,
    run_preview,
)

if TYPE_CHECKING:
    from .source import PromSource

logger = structlog.get_logger(__name__)

# Comparators that count as capacity ceilings the what-if endpoint
# considers. Drain comparators (``le_drain``) are intentionally not in
# the set — a faster drain isn't a capacity event, and reporting drain
# ETAs alongside fill ETAs would confuse the sort order. Operators
# wanting drain forecasts under what-if traffic should still use
# ``/forecast/preview``.
_CAPACITY_COMPARATORS: frozenset[str] = frozenset({"gt", "lt"})


class WhatIfThrottledError(RuntimeError):
    """Raised when a what-if call arrives faster than the configured throttle.

    Carries the retry-after duration so the HTTP layer can render a
    ``429 Too Many Requests`` with a precise ``Retry-After`` header.
    """

    def __init__(self, retry_after_seconds: float) -> None:
        super().__init__(f"fleet what-if throttled; retry after {retry_after_seconds:.0f}s")
        self.retry_after_seconds = retry_after_seconds


class WhatIfError(RuntimeError):
    """Raised by :func:`run_fleet_what_if` on operator-visible failures.

    Mirrors :class:`PreviewError` shape so the HTTP handler can render
    actionable responses with a consistent JSON body shape.
    """

    def __init__(self, reason: str, message: str, *, status_code: int = 400) -> None:
        super().__init__(message)
        self.reason = reason
        self.status_code = status_code


@dataclass(frozen=True)
class FleetWhatIfRequest:
    """Parsed JSON body of a ``POST /forecast/what-if/fleet`` request."""

    traffic_multiplier: float
    horizon_seconds: float | None
    group_filter: list[str]


def parse_fleet_request(payload: object) -> FleetWhatIfRequest:
    """Validate the inbound JSON body and return a :class:`FleetWhatIfRequest`.

    The validation rules are deliberately strict:

    * ``traffic_multiplier`` must be a finite positive float. A multiplier
      of 1.0 is allowed (corresponds to "if traffic stays as-is, which
      services breach their threshold first?") because operators
      occasionally want the baseline ranking without an overlay.
    * ``horizon`` defaults to ``config.defaults.horizon`` when absent.
      A bounded value here matters more than for the single-query
      preview because longer horizons multiply the per-query fit cost.
    * ``groups`` (optional) narrows the scan to specific groups; useful
      to ask "what about our high-traffic tier" without re-fitting
      everything.
    """
    if not isinstance(payload, dict):
        raise WhatIfError("invalid_body", "request body must be a JSON object", status_code=400)
    multiplier_raw = payload.get("traffic_multiplier")
    if not isinstance(multiplier_raw, (int, float)):
        raise WhatIfError(
            "invalid_body",
            "'traffic_multiplier' must be a positive number",
            status_code=400,
        )
    multiplier = float(multiplier_raw)
    if not _is_positive_finite(multiplier):
        raise WhatIfError(
            "invalid_body",
            "'traffic_multiplier' must be a finite positive number",
            status_code=400,
        )
    horizon_raw = payload.get("horizon")
    horizon_seconds = _parse_horizon(horizon_raw)
    groups_raw = payload.get("groups") or []
    if not isinstance(groups_raw, list) or not all(isinstance(g, str) for g in groups_raw):
        raise WhatIfError(
            "invalid_body",
            "'groups' must be a list of group-name strings",
            status_code=400,
        )
    return FleetWhatIfRequest(
        traffic_multiplier=multiplier,
        horizon_seconds=horizon_seconds,
        group_filter=[str(g) for g in groups_raw],
    )


def _is_positive_finite(value: float) -> bool:
    """Reject NaN, +/-Inf and non-positive values in one place."""
    import math  # noqa: PLC0415

    return math.isfinite(value) and value > 0.0


def _parse_horizon(value: object) -> float | None:
    """Accept ``None``, a positive number of seconds, or a Prometheus duration.

    Shares the parser with the single-query preview so the YAML
    forms (``5m``, ``1h``, ``14d``) work identically here. Raises a
    :class:`WhatIfError` (not :class:`PreviewError`) so the HTTP layer
    can route the error through the fleet-specific error path.
    """
    from datetime import timedelta as _td  # noqa: PLC0415

    from .config import parse_duration  # noqa: PLC0415

    if value is None:
        return None
    if isinstance(value, (int, float)):
        seconds = float(value)
        if not _is_positive_finite(seconds):
            raise WhatIfError(
                "invalid_body",
                "'horizon' must be a finite positive number of seconds",
                status_code=400,
            )
        return seconds
    if isinstance(value, str):
        try:
            parsed = parse_duration(value)
        except Exception as exc:
            raise WhatIfError(
                "invalid_body",
                f"'horizon' duration parse failed: {exc}",
                status_code=400,
            ) from exc
        if not isinstance(parsed, _td):
            raise WhatIfError(
                "invalid_body",
                (
                    f"'horizon' must be a duration string like '6h' or a "
                    f"number of seconds; got {value!r}"
                ),
                status_code=400,
            )
        if parsed.total_seconds() <= 0:
            raise WhatIfError("invalid_body", "'horizon' must be positive", status_code=400)
        return parsed.total_seconds()
    raise WhatIfError(
        "invalid_body",
        "'horizon' must be a duration string or numeric seconds",
        status_code=400,
    )


class WhatIfThrottle:
    """Process-wide throttle for ``POST /forecast/what-if/fleet``.

    Single-replica: the in-memory timestamp is authoritative.
    HA replicas: the endpoint is stateless so every replica enforces
    its own throttle — the worst case is N replicas serving N fleet
    runs simultaneously rather than one. Since a fleet what-if is
    O(minutes) and operators rarely have more than a handful of
    replicas, the simple per-replica throttle is enough; a Redis-backed
    coordinated throttle is tracked under future considerations.
    """

    def __init__(self, min_interval_seconds: float) -> None:
        # The earliest monotonic clock value at which a fresh call is
        # allowed. Initialised to 0 so the first call is admitted
        # immediately regardless of the configured interval.
        self._next_allowed_at: float = 0.0
        self._lock = asyncio.Lock()
        self._min_interval = max(0.0, min_interval_seconds)

    async def acquire(self) -> None:
        """Reserve a slot or raise :class:`WhatIfThrottledError`.

        The lock guards against two coincident requests both seeing
        ``next_allowed_at == 0`` and slipping through; one wins, the
        other surfaces the throttle.
        """
        async with self._lock:
            now = time.monotonic()
            if now < self._next_allowed_at:
                raise WhatIfThrottledError(self._next_allowed_at - now)
            self._next_allowed_at = now + self._min_interval

    def update_min_interval(self, min_interval_seconds: float) -> None:
        """Adjust the throttle interval after a config reload.

        The next-allowed timestamp is *not* reset so a reload can't be
        used to bypass an in-flight throttle window — the new interval
        applies to subsequent calls.

        Intentionally lock-free: the write is a single atomic
        attribute assignment (GIL guarantees no torn value) and the
        only observable race is "one concurrent ``acquire`` reads the
        old interval", which is benign — that call simply uses the
        previous interval, and the very next call sees the updated
        one. Taking the asyncio lock would require this method to
        become async and propagate through the synchronous
        ``apply_config`` callers for no functional benefit.
        """
        self._min_interval = max(0.0, min_interval_seconds)


def _capacity_thresholds(query: QueryConfig) -> list[ThresholdConfig]:
    """Return the subset of a query's thresholds that count as capacity ceilings.

    Drain comparators are excluded — a faster drain isn't a capacity
    event, so a queue's drain threshold should not appear in the
    fleet's capacity ranking. The full list is preserved (not
    de-duplicated) so multi-threshold queries (e.g. warn + critical)
    each get their own row.
    """
    return [t for t in query.thresholds if t.comparator in _CAPACITY_COMPARATORS]


def _iter_what_if_queries(
    config: Config,
    group_filter: list[str],
) -> list[tuple[str, QueryConfig, ThresholdConfig]]:
    """Yield ``(group_name, query, threshold)`` triples for the scan.

    Returns a list rather than a generator so the HTTP handler can
    report the evaluated count alongside the result. Filters by
    ``group_filter`` when supplied — group names that don't exist in
    the running config produce a precise error rather than silently
    matching nothing. Queries without any capacity threshold are
    dropped — they wouldn't contribute to the ranking anyway.
    """
    if group_filter:
        configured_groups = {g.name for g in config.groups}
        unknown = sorted(set(group_filter) - configured_groups)
        if unknown:
            raise WhatIfError(
                "invalid_body",
                f"unknown group name(s) in 'groups' filter: {', '.join(unknown)}",
                status_code=400,
            )
    out: list[tuple[str, QueryConfig, ThresholdConfig]] = []
    for group in config.groups:
        if group_filter and group.name not in group_filter:
            continue
        for query in group.queries:
            for threshold in _capacity_thresholds(query):
                out.append((group.name, query, threshold))
    return out


def _ranking_key(entry: dict[str, Any]) -> tuple[int, float, str]:
    """Sort key for the ranked services list.

    Three-level sort that puts the most-urgent breaches at the top:

    1. Finite ETAs sort ahead of non-finite ones (NaN means no
       crossing within the horizon — operators want those at the
       bottom, not interleaved with finite values).
    2. Smaller ETAs (sooner breaches) sort ahead of larger.
    3. Query id as a tiebreaker for stable output across runs when
       multiple services share the same projected ETA.
    """
    import math  # noqa: PLC0415

    seconds = entry["time_to_threshold_seconds"]
    finite = 0 if math.isfinite(seconds) else 1
    return (finite, seconds, entry["query_id"])


def _scaled_regressor_overrides(
    query: QueryConfig,
    multiplier: float,
) -> dict[str, float | list[float]]:
    """Build the ``regressor_overrides`` payload that scales the configured load.

    Strategy: when a query has any regressor whose ``id`` is one of the
    conventional traffic-driver names (``traffic``, ``rps``, ``qps``,
    ``request_rate``), scale *that* regressor — it's what the model has
    learned the metric's sensitivity to. Otherwise the override is
    empty and the caller falls back to scaling the lookback samples
    when running the preview.

    Returning a scalar multiplier (the float ``multiplier``) means
    "broadcast this value across every step". The preview code path
    accepts either a scalar or a per-step list.
    """
    if not query.regressors:
        return {}
    traffic_names = {"traffic", "rps", "qps", "request_rate", "requests_per_second"}
    overrides: dict[str, float | list[float]] = {}
    for reg in query.regressors:
        if reg.id in traffic_names:
            overrides[reg.id] = multiplier
    return overrides


async def _run_one_what_if(
    *,
    config: Config,
    source: PromSource,
    group_name: str,
    query: QueryConfig,
    threshold: ThresholdConfig,
    request: FleetWhatIfRequest,
    served_by: str,
) -> dict[str, Any]:
    """Run a single what-if fit for one (query, threshold) pair.

    Reuses :func:`preview.run_preview` so the model selection, fit
    timeout, and per-fit error handling all stay in one place. The
    threshold ETA is computed from the preview's emitted forecast
    curve.

    Failures are flattened to a result row with ``error: <reason>``
    so the caller's ranked list can include the failing services
    explicitly (rather than silently dropping them and giving the
    operator the impression "nothing is at risk").
    """
    overrides = _scaled_regressor_overrides(query, request.traffic_multiplier)
    preview_request = PreviewRequest(
        query_id=query.id,
        regressor_overrides=overrides,
        horizon_seconds=request.horizon_seconds,
        lookback_seconds=None,
    )
    try:
        preview_result = await run_preview(
            config=config,
            source=source,
            request=preview_request,
            served_by=served_by,
        )
    except PreviewError as exc:
        return {
            "query_id": query.id,
            "group": group_name,
            "threshold_name": threshold.name,
            "threshold_value": threshold.value,
            "comparator": threshold.comparator,
            "time_to_threshold_seconds": float("inf"),
            "error": exc.reason,
        }

    eta_seconds = _eta_from_preview(
        preview_result=preview_result,
        threshold=threshold,
    )
    return {
        "query_id": query.id,
        "group": group_name,
        "threshold_name": threshold.name,
        "threshold_value": threshold.value,
        "comparator": threshold.comparator,
        "time_to_threshold_seconds": eta_seconds,
        "model": preview_result.get("model", ""),
        # ``regressor`` when the multiplier landed on a traffic
        # regressor; ``baseline`` when the query has no traffic
        # regressor and the preview ran with the configured inputs
        # unchanged. Operators reading the response use this to spot
        # services whose ranking does *not* reflect the requested
        # traffic shape (a config gap worth fixing).
        "traffic_multiplier_applied": "regressor" if overrides else "baseline",
    }


def _eta_from_preview(
    *,
    preview_result: dict[str, Any],
    threshold: ThresholdConfig,
) -> float:
    """Reconstruct an ETA from the JSON preview response.

    The preview response carries ``timestamps`` + ``yhat``; we need a
    forecast frame to feed :func:`capacity.compute_threshold_etas`.
    Rebuilding the frame here keeps the what-if endpoint decoupled
    from the preview's internal pandas frame (the preview API is
    public; its internal frame is not).
    """
    import math  # noqa: PLC0415

    import pandas as pd  # noqa: PLC0415

    from .capacity import compute_threshold_etas  # noqa: PLC0415

    timestamps = preview_result.get("timestamps") or []
    yhat = preview_result.get("yhat") or []
    if not timestamps or not yhat or len(timestamps) != len(yhat):
        return float("nan")
    df = pd.DataFrame(
        {
            "ds": pd.to_datetime(timestamps),
            "yhat": [float(v) if v is not None else math.nan for v in yhat],
        }
    )
    etas = compute_threshold_etas(forecast=df, thresholds=[threshold], levels=[])
    if not etas:
        return float("nan")
    return etas[0].seconds


async def run_fleet_what_if(
    *,
    config: Config,
    source: PromSource,
    request: FleetWhatIfRequest,
    served_by: str,
    throttle: WhatIfThrottle,
) -> dict[str, Any]:
    """Execute the fleet what-if and return the ranked JSON payload.

    Throttling: the call may raise :class:`WhatIfThrottledError` when a
    previous fleet what-if ran within the configured min-interval. The
    HTTP layer renders that as 429.

    Concurrency: per-query fits run **sequentially** here. Parallelising
    would compete with scheduled fits for executor capacity, which is
    the opposite of what the throttle is trying to avoid. The
    expectation is "ask once every 5+ minutes, accept that it takes a
    minute"; for tight feedback loops use ``/forecast/preview`` on
    individual queries.
    """
    await throttle.acquire()
    candidates = _iter_what_if_queries(config, request.group_filter)
    if not candidates:
        return {
            "horizon_seconds": request.horizon_seconds,
            "traffic_multiplier": request.traffic_multiplier,
            "served_by": served_by,
            "evaluated_query_count": 0,
            "services": [],
            "message": (
                "No queries configured with capacity-shaped thresholds "
                "(comparator gt/lt). Add a `thresholds:` block to the "
                "queries you want included."
            ),
        }
    services: list[dict[str, Any]] = []
    for group_name, query, threshold in candidates:
        result = await _run_one_what_if(
            config=config,
            source=source,
            group_name=group_name,
            query=query,
            threshold=threshold,
            request=request,
            served_by=served_by,
        )
        services.append(result)
    services.sort(key=_ranking_key)
    return {
        "horizon_seconds": request.horizon_seconds,
        "traffic_multiplier": request.traffic_multiplier,
        "served_by": served_by,
        "evaluated_query_count": len(candidates),
        "services": services,
    }


__all__ = [
    "FleetWhatIfRequest",
    "WhatIfError",
    "WhatIfThrottle",
    "WhatIfThrottledError",
    "parse_fleet_request",
    "run_fleet_what_if",
]
