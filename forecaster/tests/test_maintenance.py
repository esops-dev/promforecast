"""Tests for the maintenance-window recommender."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd

from promforecast.config import RecommendedWindowsConfig
from promforecast.maintenance import recommend_windows


def _forecast_frame(values: list[float], *, step_seconds: int = 300) -> pd.DataFrame:
    """Forecast frame with monotone timestamps at the configured step."""
    base = datetime(2026, 5, 21, 12, 0, tzinfo=UTC)
    timestamps = [base + timedelta(seconds=i * step_seconds) for i in range(len(values))]
    return pd.DataFrame({"unique_id": ["s"] * len(values), "ds": timestamps, "yhat": values})


def test_returns_lowest_load_window_first() -> None:
    """Rank 1 corresponds to the lowest-mean-load window in the lookahead."""
    # 5 steps of 5m each → 25 minutes of horizon.
    # Values: [10, 5, 1, 1, 10] — the cheapest two-step window starts at index 2.
    df = _forecast_frame([10.0, 5.0, 1.0, 1.0, 10.0])
    config = RecommendedWindowsConfig(
        enabled=True,
        duration=timedelta(minutes=10),  # 2 steps at 5m
        lookahead=timedelta(hours=1),
        count=1,
    )
    out = recommend_windows(forecast=df, config=config)
    assert len(out) == 1
    assert out[0].rank == 1
    assert out[0].predicted_load == 1.0


def test_returns_top_n_windows_in_ascending_load_order() -> None:
    """``count=3`` returns the three lowest-load windows, ranked ascending."""
    df = _forecast_frame([10.0, 5.0, 1.0, 1.0, 10.0, 8.0, 2.0])
    config = RecommendedWindowsConfig(
        enabled=True,
        duration=timedelta(minutes=5),  # 1 step
        lookahead=timedelta(hours=1),
        count=3,
    )
    out = recommend_windows(forecast=df, config=config)
    assert len(out) == 3
    assert [w.rank for w in out] == [1, 2, 3]
    assert [w.predicted_load for w in out] == sorted(w.predicted_load for w in out)


def test_window_overflowing_horizon_is_skipped() -> None:
    """A window that runs past the forecast's last sample isn't recommended.

    Operators read the panel as "during this period, load will be N";
    half-overlapping windows would advertise a load they can't see the
    full curve of.
    """
    df = _forecast_frame([1.0, 1.0, 1.0])  # 15 minutes of horizon
    config = RecommendedWindowsConfig(
        enabled=True,
        duration=timedelta(hours=1),  # 60m duration > 15m horizon
        lookahead=timedelta(hours=2),
        count=3,
    )
    out = recommend_windows(forecast=df, config=config)
    assert out == []


def test_ties_break_by_start_timestamp() -> None:
    """Equal-load windows rank by earliest start so successive runs are stable."""
    df = _forecast_frame([1.0, 1.0, 1.0, 1.0, 1.0])
    config = RecommendedWindowsConfig(
        enabled=True,
        duration=timedelta(minutes=5),
        lookahead=timedelta(hours=1),
        count=3,
    )
    out = recommend_windows(forecast=df, config=config)
    assert len(out) == 3
    timestamps = [w.start_timestamp for w in out]
    assert timestamps == sorted(timestamps)


def test_lookahead_limits_candidate_window_set() -> None:
    """Anchors past the lookahead don't enter the candidate pool."""
    df = _forecast_frame([10.0, 10.0, 10.0, 1.0, 1.0, 1.0])
    # Lookahead is 10m → only the first three anchors (t=0, t=5m, t=10m) qualify
    # — all of which have load 10. The cheap (load=1) windows past the
    # lookahead must not be picked.
    config = RecommendedWindowsConfig(
        enabled=True,
        duration=timedelta(minutes=5),
        lookahead=timedelta(minutes=10),
        count=1,
    )
    out = recommend_windows(forecast=df, config=config)
    assert len(out) == 1
    assert out[0].predicted_load == 10.0


def test_empty_forecast_yields_empty() -> None:
    df = pd.DataFrame({"ds": [], "yhat": []})
    config = RecommendedWindowsConfig(enabled=True)
    assert recommend_windows(forecast=df, config=config) == []


def test_disabled_count_zero_emits_nothing() -> None:
    """``count=0`` is rejected by pydantic; verify the runtime also short-circuits."""
    # pydantic enforces count >= 1, so we exercise the runtime guard directly
    # against a hand-constructed config.
    df = _forecast_frame([1.0, 1.0])
    # Set count via internal field manipulation since pydantic refuses 0
    config = RecommendedWindowsConfig(enabled=True, duration=timedelta(minutes=5))
    object.__setattr__(config, "count", 0)
    assert recommend_windows(forecast=df, config=config) == []


def test_start_timestamp_anchored_to_forecast_first_row() -> None:
    """The reported ``start_timestamp`` is the Unix seconds of the window's first row."""
    df = _forecast_frame([1.0, 1.0, 1.0, 1.0])
    config = RecommendedWindowsConfig(
        enabled=True,
        duration=timedelta(minutes=5),
        lookahead=timedelta(hours=1),
        count=1,
    )
    out = recommend_windows(forecast=df, config=config)
    expected_first_row_unix = df["ds"].iloc[0].timestamp()
    assert out[0].start_timestamp == expected_first_row_unix
