"""Tests for canary selector parsing, frame splitting, and aggregation."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from promforecast import canary
from promforecast.source import SeriesFrame


def _frame(labels: dict[str, str], values: list[float]) -> SeriesFrame:
    base = datetime(2026, 1, 1, tzinfo=UTC)
    ts = [base + timedelta(hours=i) for i in range(len(values))]
    return SeriesFrame(labels=labels, timestamps=ts, values=values)


def test_parse_selector_equality_and_inequality() -> None:
    matchers = canary.parse_selector('{version="canary", tier!="db"}')
    assert len(matchers) == 2
    assert matchers[0].key == "version"
    assert matchers[0].op == "="
    assert matchers[1].op == "!="


def test_parse_selector_without_braces() -> None:
    matchers = canary.parse_selector('version="canary"')
    assert matchers[0].value == "canary"


def test_parse_selector_empty_matches_everything() -> None:
    assert canary.parse_selector("") == []
    assert canary.parse_selector("{}") == []


def test_parse_selector_rejects_regex() -> None:
    with pytest.raises(canary.CanarySelectorError):
        canary.parse_selector('{version=~"can.*"}')


def test_matcher_matches() -> None:
    matchers = canary.parse_selector('{version="canary"}')
    assert canary.matches_all({"version": "canary"}, matchers)
    assert not canary.matches_all({"version": "stable"}, matchers)


def test_split_frames_partitions_populations() -> None:
    frames = [
        _frame({"version": "canary", "pod": "a"}, [1.0]),
        _frame({"version": "stable", "pod": "b"}, [2.0]),
        _frame({"version": "stable", "pod": "c"}, [3.0]),
    ]
    canary_frames, baseline_frames = canary.split_frames(
        frames,
        canary_selector='{version="canary"}',
        baseline_selector='{version!="canary"}',
    )
    assert [f.labels["pod"] for f in canary_frames] == ["a"]
    assert [f.labels["pod"] for f in baseline_frames] == ["b", "c"]


def test_split_frames_rejects_overlapping_selectors() -> None:
    frames = [_frame({"version": "canary"}, [1.0])]
    with pytest.raises(canary.CanarySelectorError):
        canary.split_frames(
            frames,
            canary_selector='{version="canary"}',
            baseline_selector='{version="canary"}',
        )


def test_aggregate_baseline_means_across_series() -> None:
    frames = [
        _frame({"pod": "b"}, [2.0, 4.0]),
        _frame({"pod": "c"}, [4.0, 8.0]),
    ]
    ts, values = canary.aggregate_baseline(frames)
    assert len(ts) == 2
    assert values == [3.0, 6.0]


def test_aggregate_baseline_empty() -> None:
    assert canary.aggregate_baseline([]) == ([], [])


def test_latest_actual() -> None:
    frame = _frame({"pod": "a"}, [1.0, 2.0, 5.0])
    assert canary.latest_actual(frame) == 5.0
    assert canary.latest_actual(_frame({"pod": "x"}, [])) is None
