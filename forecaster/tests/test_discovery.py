"""Tests for dynamic series discovery via PromQL templating."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from promforecast.config import DiscoveryVariable, QueryConfig
from promforecast.discovery import (
    expand_query,
    substitute,
    template_variables,
)
from promforecast.source import PromSource, SeriesFrame


class _StubSource(PromSource):
    """In-memory PromSource — returns canned per-query frames."""

    def __init__(self, by_query: dict[str, list[SeriesFrame]]) -> None:
        self._by_query = by_query
        self.calls: list[str] = []

    async def query_range(
        self,
        promql: str,
        start: datetime,
        end: datetime,
        step_seconds: int,
    ) -> list[SeriesFrame]:
        self.calls.append(promql)
        return self._by_query.get(promql, [])

    async def aclose(self) -> None:
        return None


def _frame(labels: dict[str, str]) -> SeriesFrame:
    """One-point frame whose only purpose is to carry labels."""
    return SeriesFrame(labels=labels, timestamps=[datetime.now(tz=UTC)], values=[1.0])


def test_template_variables_returns_unique_ordered() -> None:
    assert template_variables('foo{a="$x",b="$y"} + $x') == ["x", "y"]
    assert template_variables("static query") == []


def test_substitute_handles_both_marker_forms() -> None:
    out = substitute('foo{a="$a", b="${b}"}', {"a": "1", "b": "2"})
    assert out == 'foo{a="1", b="2"}'


def test_substitute_leaves_unknown_markers_alone() -> None:
    # ``$missing`` doesn't appear in values — should stay untouched so the
    # next PromQL parse blows up with a recognisable trace.
    assert substitute("foo $missing", {}) == "foo $missing"


@pytest.mark.asyncio
async def test_expand_query_no_template_is_passthrough() -> None:
    """Queries without discovery vars produce a single rendered entry."""
    query = QueryConfig(id="m", promql="up")
    src = _StubSource({})
    result = await expand_query(
        query=query,
        source=src,
        lookback=timedelta(minutes=10),
        step_seconds=15,
        query_timeout_seconds=5.0,
        max_series_per_query=50,
        overflow_strategy="drop_lowest_priority",
    )
    assert result.rendered == [("up", {})]
    assert not result.failures
    assert src.calls == []


@pytest.mark.asyncio
async def test_expand_query_fans_out_over_label_values() -> None:
    """Discovery surfaces label values and renders one query per value."""
    query = QueryConfig(
        id="m",
        promql='rate(node_cpu_seconds_total{instance="$instance"}[5m])',
        discover=[
            DiscoveryVariable(
                name="instance",
                label="instance",
                promql='up{job="node"}',
            )
        ],
    )
    src = _StubSource(
        {
            'up{job="node"}': [
                _frame({"instance": "a"}),
                _frame({"instance": "b"}),
                _frame({"instance": "a"}),  # duplicate must be deduped
            ]
        }
    )
    result = await expand_query(
        query=query,
        source=src,
        lookback=timedelta(minutes=10),
        step_seconds=15,
        query_timeout_seconds=5.0,
        max_series_per_query=50,
        overflow_strategy="drop_lowest_priority",
    )
    promqls = [p for p, _ in result.rendered]
    assert promqls == [
        'rate(node_cpu_seconds_total{instance="a"}[5m])',
        'rate(node_cpu_seconds_total{instance="b"}[5m])',
    ]
    assert not result.failures


@pytest.mark.asyncio
async def test_expand_query_cartesian_across_variables() -> None:
    """Two variables produce ``len(a) * len(b)`` rendered queries."""
    query = QueryConfig(
        id="m",
        promql='rate(x{a="$a",b="$b"}[5m])',
        discover=[
            DiscoveryVariable(name="a", label="a", promql="probe_a"),
            DiscoveryVariable(name="b", label="b", promql="probe_b"),
        ],
    )
    src = _StubSource(
        {
            "probe_a": [_frame({"a": "1"}), _frame({"a": "2"})],
            "probe_b": [_frame({"b": "x"}), _frame({"b": "y"})],
        }
    )
    result = await expand_query(
        query=query,
        source=src,
        lookback=timedelta(minutes=10),
        step_seconds=15,
        query_timeout_seconds=5.0,
        max_series_per_query=50,
        overflow_strategy="drop_lowest_priority",
    )
    rendered = sorted(p for p, _ in result.rendered)
    assert rendered == sorted(
        [
            'rate(x{a="1",b="x"}[5m])',
            'rate(x{a="1",b="y"}[5m])',
            'rate(x{a="2",b="x"}[5m])',
            'rate(x{a="2",b="y"}[5m])',
        ]
    )


@pytest.mark.asyncio
async def test_expand_query_caps_combinations() -> None:
    """The Cartesian product is capped at max_series_per_query."""
    query = QueryConfig(
        id="m",
        promql='x{i="$i"}',
        discover=[DiscoveryVariable(name="i", label="i", promql="probe")],
    )
    frames = [_frame({"i": str(n)}) for n in range(10)]
    src = _StubSource({"probe": frames})
    result = await expand_query(
        query=query,
        source=src,
        lookback=timedelta(minutes=10),
        step_seconds=15,
        query_timeout_seconds=5.0,
        max_series_per_query=3,
        overflow_strategy="drop_lowest_priority",
    )
    assert len(result.rendered) == 3
    assert result.dropped == 7


@pytest.mark.asyncio
async def test_expand_query_empty_label_records_failure() -> None:
    """A discovery query that returns frames without the label fails clearly."""
    query = QueryConfig(
        id="m",
        promql='x{i="$i"}',
        discover=[DiscoveryVariable(name="i", label="i", promql="probe")],
    )
    src = _StubSource({"probe": [_frame({"other": "x"})]})
    result = await expand_query(
        query=query,
        source=src,
        lookback=timedelta(minutes=10),
        step_seconds=15,
        query_timeout_seconds=5.0,
        max_series_per_query=10,
        overflow_strategy="drop_lowest_priority",
    )
    assert not result.rendered
    assert len(result.failures) == 1
    assert result.failures[0].reason == "unknown_label"


@pytest.mark.asyncio
async def test_expand_query_does_not_materialise_huge_cartesian() -> None:
    """Four variables of cardinality 500 must not allocate the full product.

    Regression test for the cap-after-materialise bug: building the full
    Cartesian product *before* truncating instantly OOMs the process.
    Each variable is already capped at ``max_series_per_query`` by the
    per-variable resolver, but the Cartesian step also has to cap *during*
    iteration; without it the product is still ``max**N`` tuples in memory.
    """
    variables = [
        DiscoveryVariable(name=f"v{i}", label=f"v{i}", promql=f"probe_{i}") for i in range(4)
    ]
    query = QueryConfig(
        id="m",
        promql='x{a="$v0",b="$v1",c="$v2",d="$v3"}',
        discover=variables,
    )
    src = _StubSource(
        {f"probe_{i}": [_frame({f"v{i}": str(n)}) for n in range(500)] for i in range(4)}
    )
    result = await expand_query(
        query=query,
        source=src,
        lookback=timedelta(minutes=10),
        step_seconds=15,
        query_timeout_seconds=5.0,
        max_series_per_query=10,
        overflow_strategy="drop_lowest_priority",
    )
    # Cap kicks in at 10 expansions even though the per-variable post-cap
    # product would still be 10**4 = 10 000 tuples.
    assert len(result.rendered) == 10
    # Per-variable drops: 4 vars * (500 - 10) = 1960. Plus Cartesian-cap
    # drops: 10**4 - 10 = 9990. Total: 11950.
    assert result.dropped == 4 * (500 - 10) + (10**4 - 10)


@pytest.mark.asyncio
async def test_expand_query_handles_missing_source() -> None:
    """Without a source, every variable surfaces as ``no_source``."""
    query = QueryConfig(
        id="m",
        promql='x{i="$i"}',
        discover=[DiscoveryVariable(name="i", label="i", promql="probe")],
    )
    result = await expand_query(
        query=query,
        source=None,
        lookback=timedelta(minutes=10),
        step_seconds=15,
        query_timeout_seconds=5.0,
        max_series_per_query=10,
        overflow_strategy="drop_lowest_priority",
    )
    assert not result.rendered
    assert [f.reason for f in result.failures] == ["no_source"]
