"""Tests for the forecast component-decomposition primitive."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from promforecast.decomposition import (
    COMPONENT_NAMES,
    NON_DECOMPOSABLE_MODELS,
    DecompositionError,
    decompose,
    is_decomposable,
    reconstruct,
)


def _forecast_frame(values: list[float], *, step_seconds: int = 60) -> pd.DataFrame:
    base = datetime(2026, 5, 21, tzinfo=UTC)
    timestamps = [base + timedelta(seconds=i * step_seconds) for i in range(len(values))]
    return pd.DataFrame({"unique_id": ["s"] * len(values), "ds": timestamps, "yhat": values})


def test_decompose_returns_all_four_components() -> None:
    """Every emission produces the configured four component lists."""
    df = _forecast_frame([10.0, 12.0, 14.0, 16.0])
    result = decompose(forecast=df, in_sample_values=[8.0, 9.0, 10.0], season_length=2)
    assert set(result.components.keys()) == set(COMPONENT_NAMES)
    for values in result.components.values():
        assert len(values) == len(df)


def test_decompose_sums_back_to_yhat() -> None:
    """The contract: trend + seasonal + level + residual ≈ yhat row by row.

    A failure here would silently bias dashboards that stack components.
    """
    df = _forecast_frame([10.0, 12.0, 14.0, 13.0, 15.0, 17.0, 16.0, 18.0])
    result = decompose(
        forecast=df,
        in_sample_values=[5.0, 7.0, 9.0, 11.0],
        season_length=4,
    )
    summed = reconstruct(result.components)
    yhat = df["yhat"].tolist()
    assert len(summed) == len(yhat)
    for s, y in zip(summed, yhat, strict=True):
        assert math.isclose(s, y, abs_tol=1e-9)


def test_decompose_zero_season_length_collapses_seasonal_to_zero() -> None:
    """``season_length=0`` is a valid degenerate case (no cycle to fold)."""
    df = _forecast_frame([1.0, 2.0, 3.0])
    result = decompose(forecast=df, in_sample_values=[1.0, 2.0], season_length=0)
    assert all(v == 0.0 for v in result.components["seasonal"])
    # Even with no seasonal split, the sum-back invariant holds.
    summed = reconstruct(result.components)
    for s, y in zip(summed, df["yhat"].tolist(), strict=True):
        assert math.isclose(s, y, abs_tol=1e-9)


def test_decompose_rejects_empty_frame() -> None:
    """An empty forecast frame raises with ``components_unavailable``."""
    empty = pd.DataFrame({"unique_id": [], "ds": [], "yhat": []})
    with pytest.raises(DecompositionError) as exc:
        decompose(forecast=empty, in_sample_values=[1.0], season_length=2)
    assert exc.value.reason == "components_unavailable"


def test_decompose_rejects_frame_missing_yhat() -> None:
    """A frame without ``yhat`` is unprocessable — that's the central output."""
    df = pd.DataFrame({"unique_id": ["s"], "ds": [datetime(2026, 5, 21, tzinfo=UTC)]})
    with pytest.raises(DecompositionError) as exc:
        decompose(forecast=df, in_sample_values=[1.0], season_length=2)
    assert exc.value.reason == "components_unavailable"


def test_is_decomposable_skips_intermittent_models() -> None:
    """Intermittent-demand model families don't admit a trend/season split."""
    assert "Croston" in NON_DECOMPOSABLE_MODELS
    assert not is_decomposable("Croston")
    assert is_decomposable("AutoARIMA")


def test_decompose_raises_invariant_violation_when_summing_off() -> None:
    """The sum-to-yhat invariant must be enforced at fit time, not only in tests.

    A future change to the math (e.g. dropping the residual term) would
    silently bias the stacked-area panel; the guard turns that into a
    skip with ``reason="invariant_violation"`` so the operator sees it.
    """
    from promforecast.decomposition import _enforce_sum_invariant  # noqa: PLC0415

    with pytest.raises(DecompositionError) as exc:
        _enforce_sum_invariant(
            yhat=[10.0, 20.0],
            components={
                "trend": [0.0, 0.0],
                "seasonal": [0.0, 0.0],
                "level": [5.0, 5.0],  # sums to 5 vs yhat 10 — violation.
                "residual": [0.0, 0.0],
            },
        )
    assert exc.value.reason == "invariant_violation"


def test_decompose_sum_invariant_tolerates_large_magnitude_noise() -> None:
    """The invariant must use a relative tolerance for large-magnitude metrics.

    A pure absolute floor of 1e-6 would falsely flag harmless float
    accumulation on, e.g., disk-bytes forecasts (~10^12). The
    component sum should clear the check at any magnitude where the
    drift is float-noise relative to ``|yhat|``, not raise
    ``invariant_violation``.
    """
    from promforecast.decomposition import _enforce_sum_invariant  # noqa: PLC0415

    yhat = 1.0e12
    # A drift of 1e-3 absolute is float noise next to 10^12 (ratio
    # 1e-15) and must pass; the old absolute-only check would have
    # rejected it.
    _enforce_sum_invariant(
        yhat=[yhat],
        components={
            "trend": [0.0],
            "seasonal": [0.0],
            "level": [yhat],
            "residual": [-1.0e-3],  # not zero — exercises the tolerance
        },
    )


def test_decompose_sum_invariant_still_catches_real_drift() -> None:
    """Relative tolerance still rejects drift that isn't float noise.

    A 10% drift on a 100-unit yhat row is a logic bug, not float
    accumulation; the check must still raise so the runner emits the
    skip counter rather than ship a misleading panel.
    """
    from promforecast.decomposition import _enforce_sum_invariant  # noqa: PLC0415

    with pytest.raises(DecompositionError) as exc:
        _enforce_sum_invariant(
            yhat=[100.0],
            components={
                "trend": [0.0],
                "seasonal": [0.0],
                "level": [90.0],
                "residual": [0.0],
            },
        )
    assert exc.value.reason == "invariant_violation"


def test_decompose_trend_captures_slope() -> None:
    """A pure linear ramp should land its slope on the ``trend`` component."""
    # values rise by 5/step from 0; trend slope = 5
    df = _forecast_frame([0.0, 5.0, 10.0, 15.0])
    result = decompose(forecast=df, in_sample_values=[0.0], season_length=1)
    trend = result.components["trend"]
    # Step 0 is the anchor (0), step k is k * slope.
    assert math.isclose(trend[0], 0.0, abs_tol=1e-9)
    assert math.isclose(trend[3] - trend[0], 15.0, abs_tol=1e-9)
