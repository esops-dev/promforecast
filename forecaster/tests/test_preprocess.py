"""Tests for the preprocessing pipeline."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from promforecast.config import (
    BlacklistWindowConfig,
    ChangePointConfig,
    GapImputationConfig,
    OutlierConfig,
    PreprocessConfig,
)
from promforecast.preprocess import (
    PreprocessResult,
    _bocpd_breakpoint,
    _hampel_mask,
    _iqr_mask,
    _ruptures_breakpoint,
    preprocess_series,
)


def _regular_series(
    n: int,
    *,
    start: datetime | None = None,
    step: timedelta = timedelta(minutes=1),
) -> tuple[list[datetime], list[float]]:
    start = start or datetime(2026, 1, 1, tzinfo=UTC)
    timestamps = [start + step * i for i in range(n)]
    values = [float(i) for i in range(n)]
    return timestamps, values


def _config(
    *,
    gap_strategy: str = "forward_fill",
    max_gap: timedelta = timedelta(minutes=30),
    outliers_enabled: bool = False,
    outlier_method: str = "hampel",
    outlier_sensitivity: str = "medium",
    outlier_replace: str = "median_neighbours",
    change_point_enabled: bool = False,
    change_point_method: str = "ruptures",
    change_point_action: str = "truncate_lookback",
    min_post_break: int = 50,
    reweight_decay: float = 0.5,
) -> PreprocessConfig:
    return PreprocessConfig(
        gaps=GapImputationConfig(strategy=gap_strategy, max_gap=max_gap),  # type: ignore[arg-type]
        outliers=OutlierConfig(
            enabled=outliers_enabled,
            method=outlier_method,  # type: ignore[arg-type]
            sensitivity=outlier_sensitivity,  # type: ignore[arg-type]
            replace_with=outlier_replace,  # type: ignore[arg-type]
        ),
        change_point=ChangePointConfig(
            enabled=change_point_enabled,
            method=change_point_method,  # type: ignore[arg-type]
            action=change_point_action,  # type: ignore[arg-type]
            min_post_break_points=min_post_break,
            reweight_decay=reweight_decay,
        ),
    )


# ---------- gap imputation -------------------------------------------------


def test_no_gaps_passes_through() -> None:
    ts, vs = _regular_series(20)
    result = preprocess_series(
        timestamps=ts, values=vs, step=timedelta(minutes=1), config=_config()
    )
    assert not result.dropped
    assert len(result.values) == len(vs)
    assert result.gaps_filled == {}
    assert result.imputed_fraction == 0.0


def test_forward_fill_fills_short_gap() -> None:
    ts, vs = _regular_series(20)
    # Cut out 3 minutes [10, 11, 12] so the gap is 4 minutes.
    keep_idx = [i for i in range(20) if i < 10 or i > 12]
    sparse_ts = [ts[i] for i in keep_idx]
    sparse_vs = [vs[i] for i in keep_idx]
    result = preprocess_series(
        timestamps=sparse_ts,
        values=sparse_vs,
        step=timedelta(minutes=1),
        config=_config(),
    )
    assert not result.dropped
    # Three filler points + the original 17 = 20 (matches the regular grid).
    assert len(result.values) == 20
    assert result.gaps_filled == {"forward_fill": 3}
    # Forward-fill: filler value equals the last good value before the gap.
    assert result.values[10] == result.values[9]
    assert result.imputed_fraction > 0


def test_linear_interp_interpolates_filler() -> None:
    timestamps = [
        datetime(2026, 1, 1, 0, 0, tzinfo=UTC),
        datetime(2026, 1, 1, 0, 5, tzinfo=UTC),  # 5min gap (4 filler steps)
    ]
    values = [0.0, 5.0]
    result = preprocess_series(
        timestamps=timestamps,
        values=values,
        step=timedelta(minutes=1),
        config=_config(gap_strategy="linear_interp"),
    )
    # Filler is 4 points between two anchors -> regular grid of 6.
    assert len(result.values) == 6
    # Interpolation steps 1/5 .. 4/5 of the range.
    assert math.isclose(result.values[1], 1.0)
    assert math.isclose(result.values[4], 4.0)
    assert result.gaps_filled == {"linear_interp": 4}


def test_large_gap_triggers_drop() -> None:
    # Two-hour gap blows past max_gap=30m -> drop_series fallback.
    timestamps = [
        datetime(2026, 1, 1, 0, 0, tzinfo=UTC),
        datetime(2026, 1, 1, 2, 0, tzinfo=UTC),
    ]
    result = preprocess_series(
        timestamps=timestamps,
        values=[0.0, 1.0],
        step=timedelta(minutes=1),
        config=_config(),
    )
    assert result.dropped is True
    assert result.drop_reason == "gap_too_large"


def test_drop_series_strategy_always_drops_on_gap() -> None:
    timestamps = [
        datetime(2026, 1, 1, 0, 0, tzinfo=UTC),
        datetime(2026, 1, 1, 0, 3, tzinfo=UTC),
    ]
    result = preprocess_series(
        timestamps=timestamps,
        values=[0.0, 1.0],
        step=timedelta(minutes=1),
        config=_config(gap_strategy="drop_series"),
    )
    assert result.dropped is True


# ---------- outlier scrubbing ---------------------------------------------


def test_hampel_flags_isolated_spike() -> None:
    arr = np.zeros(50, dtype=float)
    arr[25] = 50.0  # huge spike
    mask = _hampel_mask(arr, k=3.0)
    assert bool(mask[25])
    assert mask.sum() == 1


def test_iqr_flags_global_outlier() -> None:
    arr = np.concatenate([np.full(99, 1.0), np.array([100.0])])
    mask = _iqr_mask(arr, mult=1.5)
    assert bool(mask[-1])


def test_outlier_scrubbing_replaces_with_median() -> None:
    n = 60
    ts = [datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=i) for i in range(n)]
    values = [1.0] * n
    values[30] = 1000.0  # outlier
    result = preprocess_series(
        timestamps=ts,
        values=values,
        step=timedelta(minutes=1),
        config=_config(outliers_enabled=True),
    )
    assert result.outliers_replaced == 1
    # Spike replaced by something near the local median (1.0).
    assert abs(result.values[30] - 1.0) < 0.1


def test_outlier_scrubbing_off_by_default() -> None:
    n = 60
    ts = [datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=i) for i in range(n)]
    values = [1.0] * n
    values[30] = 1000.0
    result = preprocess_series(
        timestamps=ts, values=values, step=timedelta(minutes=1), config=_config()
    )
    assert result.outliers_replaced == 0
    assert result.values[30] == 1000.0


# ---------- change-point detection -----------------------------------------


def _bimodal(n_each: int = 80) -> np.ndarray:
    rng = np.random.default_rng(seed=42)
    left = rng.normal(0.0, 0.1, n_each)
    right = rng.normal(5.0, 0.1, n_each)
    return np.concatenate([left, right])


def test_ruptures_finds_mean_shift() -> None:
    arr = _bimodal()
    idx = _ruptures_breakpoint(arr)
    assert idx is not None
    # Break should be near the actual split (within 5% of array length).
    assert abs(idx - 80) < 8


def test_bocpd_fallback_finds_mean_shift() -> None:
    arr = _bimodal()
    idx = _bocpd_breakpoint(arr)
    assert idx is not None
    assert abs(idx - 80) < 8


def test_no_change_point_on_stationary_signal() -> None:
    rng = np.random.default_rng(seed=7)
    arr = rng.normal(0.0, 1.0, 200)
    assert _bocpd_breakpoint(arr) is None


def test_change_point_truncate_lookback() -> None:
    arr = _bimodal()
    n = arr.size
    ts = [datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=i) for i in range(n)]
    result = preprocess_series(
        timestamps=ts,
        values=arr.tolist(),
        step=timedelta(minutes=1),
        config=_config(change_point_enabled=True, min_post_break=20),
    )
    assert result.change_points == 1
    # Truncated to the post-break tail.
    assert len(result.values) < n
    # All retained values should look like the second cluster (mean ~5).
    assert float(np.mean(result.values)) > 2.0


def test_change_point_skipped_when_tail_too_short() -> None:
    arr = _bimodal(n_each=40)  # only 40 post-break points
    n = arr.size
    ts = [datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=i) for i in range(n)]
    result = preprocess_series(
        timestamps=ts,
        values=arr.tolist(),
        step=timedelta(minutes=1),
        config=_config(change_point_enabled=True, min_post_break=60),
    )
    # The detector may fire but the truncation is skipped because the
    # post-break tail is shorter than min_post_break_points.
    assert result.change_points == 0
    assert len(result.values) == n


def test_reweight_recent_soft_truncates() -> None:
    arr = _bimodal()
    n = arr.size
    ts = [datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=i) for i in range(n)]
    result = preprocess_series(
        timestamps=ts,
        values=arr.tolist(),
        step=timedelta(minutes=1),
        config=_config(
            change_point_enabled=True,
            change_point_action="reweight_recent",
            reweight_decay=0.5,
            min_post_break=20,
        ),
    )
    assert result.change_points == 1
    # Soft truncation: kept post-break + half of pre-break.
    assert len(result.values) < n
    assert len(result.values) > 80  # more than a hard truncate would leave


def test_preprocess_result_imputed_fraction_zero_clean() -> None:
    # Empty inputs short-circuit and return a no-op result.
    result = preprocess_series(
        timestamps=[],
        values=[],
        step=timedelta(minutes=1),
        config=_config(),
    )
    assert isinstance(result, PreprocessResult)
    assert result.imputed_fraction == 0.0
    assert result.values == []


# ---------- blacklist windows ---------------------------------------------


def _config_with_blacklist(windows: list[BlacklistWindowConfig]) -> PreprocessConfig:
    base = _config()
    return PreprocessConfig(
        gaps=base.gaps,
        outliers=base.outliers,
        change_point=base.change_point,
        blacklist_windows=windows,
    )


def test_blacklist_absolute_window_drops_samples() -> None:
    # 60 1-minute samples; drop the middle 20.
    ts, vs = _regular_series(60)
    start = ts[20].isoformat().replace("+00:00", "Z")
    end = ts[40].isoformat().replace("+00:00", "Z")
    window = BlacklistWindowConfig(start=start, end=end, reason="incident")
    result = preprocess_series(
        timestamps=ts,
        values=vs,
        step=timedelta(minutes=1),
        config=_config_with_blacklist([window]),
    )
    assert not result.dropped
    assert result.blacklisted_samples == {"incident": 20}
    # The default ``forward_fill`` gap strategy imputes the blacklisted
    # span back to a regular grid so the series length is unchanged;
    # what changes is the counter (20 reasoned samples removed) and the
    # effective-lookback gauge, which discounts those samples.
    assert len(result.values) == 60
    expected_total = (ts[-1] - ts[0]).total_seconds()
    assert result.effective_lookback_seconds < expected_total
    # ``20 samples x 60s step`` worth of discount: ~1200s.
    assert result.effective_lookback_seconds == expected_total - 20 * 60


def test_blacklist_default_empty_reason() -> None:
    ts, vs = _regular_series(60)
    start = ts[10].isoformat().replace("+00:00", "Z")
    end = ts[15].isoformat().replace("+00:00", "Z")
    window = BlacklistWindowConfig(start=start, end=end)
    result = preprocess_series(
        timestamps=ts,
        values=vs,
        step=timedelta(minutes=1),
        config=_config_with_blacklist([window]),
    )
    assert result.blacklisted_samples == {"": 5}


def test_blacklist_annual_recurring_fires_per_year() -> None:
    # 3-year lookback at 1h step (around 365*3*24 ≈ 26280 samples is too
    # heavy for a unit test). Build a sparse-but-representative two-year
    # input covering January in each year so the annual window fires twice.
    points: list[datetime] = []
    for year in (2024, 2025):
        for day in range(1, 31):
            points.append(datetime(year, 1, day, 12, 0, tzinfo=UTC))
    values = [float(i) for i in range(len(points))]
    window = BlacklistWindowConfig(
        start="01-01T00:00",
        end="01-02T00:00",
        reason="newyear",
        recurring="annual",
    )
    result = preprocess_series(
        timestamps=points,
        values=values,
        step=timedelta(hours=24),
        config=_config_with_blacklist([window]),
    )
    # One sample per year falls inside the New Year's Day window.
    assert result.blacklisted_samples == {"newyear": 2}


def test_blacklist_consuming_all_samples_drops_series() -> None:
    ts, vs = _regular_series(30)
    start = (ts[0] - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    end = (ts[-1] + timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    window = BlacklistWindowConfig(start=start, end=end, reason="freeze")
    result = preprocess_series(
        timestamps=ts,
        values=vs,
        step=timedelta(minutes=1),
        config=_config_with_blacklist([window]),
    )
    assert result.dropped
    assert result.drop_reason == "blacklist_removed_all_samples"
    assert result.blacklisted_samples == {"freeze": 30}


def test_blacklist_rejects_malformed_window() -> None:
    with pytest.raises(ValueError, match="ISO-8601"):
        BlacklistWindowConfig(start="2024-11-29", end="2024-11-30")

    with pytest.raises(ValueError, match="MM-DDTHH"):
        BlacklistWindowConfig(
            start="11-29",
            end="11-30",
            recurring="annual",
        )


def test_blacklist_annual_window_anchored_in_local_timezone() -> None:
    """An annual window configured with ``timezone: America/New_York``
    should flip at local midnight, not UTC-midnight. A hard test: a
    sample at 04:00 UTC on Jan 1 is 23:00 on Dec 31 in NYC and must
    be EXCLUDED from a ``01-01T00:00`` → ``01-02T00:00`` NYC window;
    a sample at 06:00 UTC (01:00 NYC) is INSIDE the window.
    """
    # Pick two samples around midnight-NYC on New Year's Day 2025.
    # NYC is UTC-5 in January (EST).
    ts = [
        datetime(2025, 1, 1, 4, 0, tzinfo=UTC),  # 23:00 EST on Dec 31 - OUTSIDE
        datetime(2025, 1, 1, 6, 0, tzinfo=UTC),  # 01:00 EST on Jan 1  - INSIDE
        datetime(2025, 1, 2, 4, 0, tzinfo=UTC),  # 23:00 EST on Jan 1  - INSIDE
        datetime(2025, 1, 2, 6, 0, tzinfo=UTC),  # 01:00 EST on Jan 2  - OUTSIDE
    ]
    values = [1.0, 2.0, 3.0, 4.0]
    window = BlacklistWindowConfig(
        start="01-01T00:00",
        end="01-02T00:00",
        recurring="annual",
        timezone="America/New_York",
        reason="ny_newyear",
    )
    result = preprocess_series(
        timestamps=ts,
        values=values,
        step=timedelta(hours=2),
        config=_config_with_blacklist([window]),
    )
    # Two samples (the 06:00 UTC and the 04:00 UTC the next day) fall
    # inside the local-midnight-to-local-midnight window.
    assert result.blacklisted_samples == {"ny_newyear": 2}


def test_blacklist_annual_unknown_timezone_is_skipped() -> None:
    """An unparseable timezone should not crash the fit pipeline.

    The pydantic validator does not (yet) check that the timezone is
    an IANA name; an operator typo would otherwise reach the runtime
    parser. ``_parse_annual_window`` returns ``None`` on
    ``ZoneInfoNotFoundError`` and the window is silently skipped so
    the rest of the series is still fit.
    """
    ts, vs = _regular_series(60)
    window = BlacklistWindowConfig(
        start="01-01T00:00",
        end="01-02T00:00",
        recurring="annual",
        timezone="Not/A_Real_Zone",
    )
    result = preprocess_series(
        timestamps=ts,
        values=vs,
        step=timedelta(minutes=1),
        config=_config_with_blacklist([window]),
    )
    # The window failed to expand; no samples dropped, no counter ticks.
    assert result.blacklisted_samples == {}
    assert not result.dropped
