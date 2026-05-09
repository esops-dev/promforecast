"""Tests for cardinality safety helpers in the runner."""

from __future__ import annotations

import random

from promforecast.config import GroupConfig, QueryConfig
from promforecast.runner import _apply_overflow, _compute_global_budgets
from promforecast.source import SeriesFrame


def _frames(n: int) -> list[SeriesFrame]:
    return [SeriesFrame(labels={"i": str(i)}, timestamps=[], values=[float(i)]) for i in range(n)]


def _group(name: str, priority: int) -> GroupConfig:
    return GroupConfig(
        name=name,
        priority=priority,
        queries=[QueryConfig(id="x", promql="up")],
    )


def test_overflow_passthrough_when_under_cap() -> None:
    frames = _frames(3)
    allowed, dropped = _apply_overflow(frames, cap=10, strategy="drop_lowest_priority")
    assert allowed == frames
    assert dropped == 0


def test_overflow_drop_lowest_is_deterministic_truncation() -> None:
    frames = _frames(10)
    allowed, dropped = _apply_overflow(frames, cap=4, strategy="drop_lowest_priority")
    assert [f.labels["i"] for f in allowed] == ["0", "1", "2", "3"]
    assert dropped == 6


def test_overflow_reject_drops_entire_query() -> None:
    frames = _frames(10)
    allowed, dropped = _apply_overflow(frames, cap=4, strategy="reject")
    assert allowed == []
    assert dropped == 10


def test_overflow_sample_keeps_cap_random_subset() -> None:
    frames = _frames(10)
    random.seed(0)
    allowed, dropped = _apply_overflow(frames, cap=4, strategy="sample")
    assert len(allowed) == 4
    assert dropped == 6
    # Random, but every element should still be from the input set.
    assert all(f in frames for f in allowed)


def test_overflow_zero_cap_drops_everything() -> None:
    frames = _frames(3)
    allowed, dropped = _apply_overflow(frames, cap=0, strategy="drop_lowest_priority")
    assert allowed == []
    assert dropped == 3


def test_global_budget_disabled_returns_zeros() -> None:
    groups = [_group("a", 1), _group("b", 1)]
    assert _compute_global_budgets(groups, max_total_series=0) == {"a": 0, "b": 0}


def test_global_budget_equal_priorities_split_evenly() -> None:
    groups = [_group("a", 1), _group("b", 1), _group("c", 1)]
    assert _compute_global_budgets(groups, max_total_series=300) == {"a": 100, "b": 100, "c": 100}


def test_global_budget_priority_weighted() -> None:
    groups = [_group("hi", 10), _group("lo1", 1), _group("lo2", 1)]
    budgets = _compute_global_budgets(groups, max_total_series=1200)
    assert budgets["hi"] == 1000
    assert budgets["lo1"] == 100
    assert budgets["lo2"] == 100
