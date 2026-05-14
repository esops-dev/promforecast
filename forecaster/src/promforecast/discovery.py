"""Dynamic series discovery via PromQL templating.

Lets a single YAML query fan out across many real-world series without one
config entry per series. A query that mentions ``$variable`` placeholders is
expanded at runtime by:

1. Running each variable's discovery query (PromQL) to enumerate its values.
2. Taking the Cartesian product across variables (when more than one).
3. Substituting placeholders into the primary PromQL — one rendered string
   per combination — and fanning out the resulting fetches.

The same per-query cardinality caps apply to the *expansion* itself: a
variable that returns 10 000 values is truncated to ``max_series_per_query``
according to ``series_overflow`` so a misconfigured discovery query can't
explode the fan-out. The aggregated post-expansion series count is also
charged against ``max_total_series`` exactly like any other group.

The substitution uses ``${var}`` and ``$var`` markers. We intentionally do
not implement a full template language — PromQL is already expressive, and
anything more elaborate would re-invent recording rules.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from itertools import islice, product
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from .config import DiscoveryVariable, QueryConfig
    from .source import PromSource

logger = structlog.get_logger(__name__)


# ``${var}`` and ``$var`` (PromQL grammar already uses ``{``/``}`` for label
# matchers, so we accept both forms — the brace form removes ambiguity when a
# variable name is followed by alphanumerics, e.g. ``$instance_foo`` would
# otherwise be a single 12-char identifier).
_TEMPLATE_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)")


@dataclass(frozen=True)
class DiscoveryFailure:
    """Per-variable discovery failure record.

    ``reason`` is a small bounded token used as a Prometheus counter label
    (``query_error``, ``query_timeout``, ``empty_result``, ``no_source``,
    ``unknown_label``). The runner surfaces these via
    ``forecast_discovery_failures_total``.
    """

    variable: str
    reason: str
    message: str


@dataclass(frozen=True)
class ExpansionResult:
    """Result of a template expansion: one rendered PromQL per combination.

    ``rendered`` is the list of (PromQL string, value-mapping) pairs ready
    to feed into the existing query pipeline. ``failures`` records every
    discovery query that didn't produce values (or hit the cap); the
    runner reports them through the standard failure/dropped counters.
    """

    rendered: list[tuple[str, dict[str, str]]]
    failures: list[DiscoveryFailure]
    dropped: int = 0


def template_variables(promql: str) -> list[str]:
    """Return the ordered list of distinct variable names in the template.

    Order is insertion-stable so a config with ``$a`` then ``$b`` produces
    a deterministic Cartesian product. Duplicates are returned once.
    """
    seen: dict[str, None] = {}
    for match in _TEMPLATE_RE.finditer(promql):
        name = match.group(1) or match.group(2) or ""
        if name and name not in seen:
            seen[name] = None
    return list(seen.keys())


def substitute(promql: str, values: dict[str, str]) -> str:
    """Replace ``$var`` / ``${var}`` markers with ``values`` entries.

    Markers without a matching value pass through unchanged so a partial
    expansion (rare in practice — the runner always passes the full set)
    fails on the next PromQL parse rather than silently producing an
    invalid query without a recognisable trace.
    """

    def _replace(match: re.Match[str]) -> str:
        name = match.group(1) or match.group(2) or ""
        if name in values:
            return values[name]
        return match.group(0)

    return _TEMPLATE_RE.sub(_replace, promql)


async def expand_query(
    *,
    query: QueryConfig,
    source: PromSource | None,
    lookback: timedelta,
    step_seconds: int,
    query_timeout_seconds: float,
    max_series_per_query: int,
    overflow_strategy: str,
) -> ExpansionResult:
    """Resolve every discovery variable and produce the rendered query list.

    Returns an :class:`ExpansionResult` with a single entry (the original
    PromQL, no substitutions) when the query has no discovery variables —
    keeping the caller's code path uniform with the templated case.
    """
    if not query.discover:
        return ExpansionResult(rendered=[(query.promql, {})], failures=[])

    variables = list(query.discover)
    failures: list[DiscoveryFailure] = []

    if source is None:
        # Test / dry-run path: signal "no source" instead of producing an
        # empty list silently (the runner would drop the whole query without
        # context otherwise).
        for var in variables:
            failures.append(
                DiscoveryFailure(
                    variable=var.name,
                    reason="no_source",
                    message="datasource not available for discovery",
                )
            )
        return ExpansionResult(rendered=[], failures=failures)

    resolved: list[list[str]] = []
    total_dropped = 0
    for var in variables:
        values, var_failures, var_dropped = await _resolve_variable(
            var=var,
            source=source,
            lookback=lookback,
            step_seconds=step_seconds,
            query_timeout_seconds=query_timeout_seconds,
            max_values=max_series_per_query,
            overflow_strategy=overflow_strategy,
        )
        failures.extend(var_failures)
        total_dropped += var_dropped
        if not values:
            # Any variable with no values aborts the whole expansion — a
            # partial expansion with one variable hardcoded would silently
            # produce the wrong (over-aggregated) query.
            return ExpansionResult(rendered=[], failures=failures, dropped=total_dropped)
        resolved.append(values)

    # Pre-compute the Cartesian product size so we can cap *before*
    # materialising. ``list(product(...))`` for 4 variables of 500 values
    # would otherwise allocate 62.5B tuples - instant OOM. ``islice`` over
    # the generator gives us the prefix at constant memory.
    total = 1
    for values in resolved:
        total *= len(values)
    cap = max_series_per_query if max_series_per_query > 0 else total
    take = min(cap, total)
    if total > cap:
        total_dropped += total - cap
    rendered: list[tuple[str, dict[str, str]]] = []
    for combo in islice(product(*resolved), take):
        combo_values = {var.name: combo[i] for i, var in enumerate(variables)}
        rendered.append((substitute(query.promql, combo_values), combo_values))
    return ExpansionResult(rendered=rendered, failures=failures, dropped=total_dropped)


async def _resolve_variable(
    *,
    var: DiscoveryVariable,
    source: PromSource,
    lookback: timedelta,
    step_seconds: int,
    query_timeout_seconds: float,
    max_values: int,
    overflow_strategy: str,
) -> tuple[list[str], list[DiscoveryFailure], int]:
    """Fetch the discovery query and extract distinct label values.

    Bounds the per-variable result count by the per-query cap to keep the
    Cartesian product manageable. Drops past the cap are returned to the
    caller so they can be surfaced as ``series_dropped{reason="discovery_cap"}``.
    """
    end = datetime.now(tz=UTC)
    start = end - lookback
    try:
        frames = await asyncio.wait_for(
            source.query_range(var.promql, start, end, step_seconds),
            timeout=query_timeout_seconds,
        )
    except TimeoutError:
        return (
            [],
            [
                DiscoveryFailure(
                    variable=var.name,
                    reason="query_timeout",
                    message=f"discovery query timed out after {query_timeout_seconds}s",
                )
            ],
            0,
        )
    except Exception as exc:
        return (
            [],
            [
                DiscoveryFailure(
                    variable=var.name,
                    reason="query_error",
                    message=str(exc),
                )
            ],
            0,
        )

    label = var.label
    values: list[str] = []
    seen: set[str] = set()
    for frame in frames:
        value = frame.labels.get(label)
        if value is None or value in seen:
            continue
        seen.add(value)
        values.append(value)

    if not values:
        if not frames:
            reason = "empty_result"
            message = "discovery query returned no series"
        else:
            reason = "unknown_label"
            message = f"label {label!r} not present on any discovered series"
        return [], [DiscoveryFailure(variable=var.name, reason=reason, message=message)], 0

    dropped = 0
    if max_values > 0 and len(values) > max_values:
        if overflow_strategy == "reject":
            return (
                [],
                [
                    DiscoveryFailure(
                        variable=var.name,
                        reason="empty_result",
                        message=(
                            f"discovery produced {len(values)} values, "
                            f"above max_series_per_query={max_values}; "
                            "series_overflow=reject"
                        ),
                    )
                ],
                len(values),
            )
        dropped = len(values) - max_values
        if overflow_strategy == "sample":
            import random  # noqa: PLC0415

            values = random.sample(values, max_values)
        else:
            # Default + "drop_lowest_priority": deterministic prefix —
            # discovery has no per-value priority to drop on, so we
            # take the first ``max_values`` in returned order.
            values = values[:max_values]
    # Sort the values for stable output ordering (otherwise the Cartesian
    # product changes between runs as PromQL response order shifts).
    values.sort()
    return values, [], dropped


__all__ = [
    "DiscoveryFailure",
    "ExpansionResult",
    "expand_query",
    "substitute",
    "template_variables",
]
