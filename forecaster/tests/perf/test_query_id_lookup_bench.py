"""Micro-benchmark for ``_query_id_from_metric``.

Gated behind the ``benchmark`` marker so the default ``pytest`` invocation
skips it (the default ``addopts`` selector is ``-m 'not benchmark'``). Opt in
with ``pytest -m benchmark`` from a nightly CI job or local profiling.

The workload mirrors ``_inherit_prior_for_skipped_queries``'s call shape: a
mix of every emission suffix the runner pushes through the helper, repeated
many times to amortise per-call setup. Comparison is against the original
linear-scan implementation (preserved here as ``_linear_scan_reference``).

The threshold is intentionally loose: shared CI runners have wildly
variable single-core throughput and the assertion exists to catch a
*regression* (e.g. someone disables ``lru_cache`` by accident) rather
than to pin a specific speed-up factor. Locally the lru_cache-warmed
path measures multiples faster than the linear scan on a workload of
80-90 distinct metric names; the 1.1x floor catches the "cache stopped
working" failure mode without flaking on slow CI. If this assertion
ever fires spuriously, prefer raising the workload size (so the ratio
is computed over a longer wall-clock window) over loosening the
threshold further.
"""

from __future__ import annotations

import time

import pytest

from promforecast.runner import (
    _FORECAST_METRIC_SUFFIXES,
    _query_id_from_metric,
)

# Regression alarm — anything under 1.0x means the cache stopped working
# or someone reverted the length-bucket probe to the linear scan. Not a
# performance target.
_MIN_SPEEDUP = 1.1


def _linear_scan_reference(metric_name: str) -> str:
    """Reference implementation: the original linear ``endswith`` scan."""
    for suffix in _FORECAST_METRIC_SUFFIXES:
        if metric_name.endswith(suffix):
            stem = metric_name[: -len(suffix)]
            if stem:
                return stem
    return ""


def _build_workload(repeat: int = 4000) -> list[str]:
    """Build a representative workload covering every emission family."""
    ids = (
        "node_cpu_busy_pct",
        "node_filesystem_avail_bytes",
        "kafka_consumer_lag",
        "http_requests_rate",
        "api_forecast_count",  # legitimate id with "forecast" substring
    )
    metrics: list[str] = []
    for qid in ids:
        for suffix in _FORECAST_METRIC_SUFFIXES:
            metrics.append(qid + suffix)
    # Also include a few unknowns so the "no match" branch is exercised.
    metrics.append("node_memory_used_bytes")
    metrics.append("unrelated_metric_name")
    return metrics * repeat


@pytest.mark.benchmark
def test_query_id_lookup_beats_linear_scan() -> None:
    workload = _build_workload()

    # Warm both implementations equally so the timing isn't biased by
    # CPython's first-call interpreter setup. Then clear the lru_cache
    # so the measurement covers cache-miss work too, not just the
    # steady-state hit cost.
    for metric in workload[:50]:
        _linear_scan_reference(metric)
        _query_id_from_metric(metric)
    _query_id_from_metric.cache_clear()

    start_linear = time.perf_counter()
    for metric in workload:
        _linear_scan_reference(metric)
    elapsed_linear = time.perf_counter() - start_linear

    start_current = time.perf_counter()
    for metric in workload:
        _query_id_from_metric(metric)
    elapsed_current = time.perf_counter() - start_current

    speedup = elapsed_linear / max(elapsed_current, 1e-9)
    assert speedup >= _MIN_SPEEDUP, (
        f"current implementation should be >= {_MIN_SPEEDUP}x faster than the "
        f"linear scan; got {speedup:.2f}x "
        f"(linear={elapsed_linear * 1e3:.2f}ms, current={elapsed_current * 1e3:.2f}ms)"
    )


@pytest.mark.benchmark
def test_query_id_lookup_round_trip_matches_linear_scan() -> None:
    """The current variant must agree with the linear scan on every workload row.

    Correctness is also covered indirectly by the broader runner test suite;
    this benchmark-side check pins the equivalence under a workload shape
    that exercises every suffix at least once.
    """
    for metric in _build_workload(repeat=1):
        assert _query_id_from_metric(metric) == _linear_scan_reference(metric), metric
