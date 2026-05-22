"""Tests for the count-aware emission safeguards.

The clipping and rounding logic lives in two pure helpers
(``_classify_emission_correction`` and the snapshot/curve appliers).
We exercise them directly here so the contracts stay sharp — the
runner integration is covered separately in the runner test suite.
"""

from __future__ import annotations

import math

from promforecast.config import QueryConfig, QueryEmissionConfig, effective_emission_safeguards
from promforecast.exporter import ForecastPoint
from promforecast.runner import (
    _apply_safeguards_to_curve_points,
    _apply_safeguards_to_snapshot_points,
    _classify_emission_correction,
)
from promforecast.sink import ForecastCurvePoint


def test_classify_negative_clipped_to_zero_and_counted() -> None:
    """A raw negative value is clamped at zero and bumps ``kind=negative``."""
    value, kinds = _classify_emission_correction(-3.5, clip=True, rounder=False)
    assert value == 0.0
    assert kinds == ["negative"]


def test_classify_positive_value_unchanged_with_clip_only() -> None:
    """Positive values pass through clipping without a counter bump."""
    value, kinds = _classify_emission_correction(3.5, clip=True, rounder=False)
    assert value == 3.5
    assert kinds == []


def test_classify_rounder_only_bumps_for_non_trivial_fractional() -> None:
    """Rounding bumps the counter only when |frac| >= 0.05.

    A continuous model that emits, say, 4.001 must not flood the
    rounding counter every refresh; that's what the 0.05 floor is for.
    """
    value, kinds = _classify_emission_correction(4.001, clip=False, rounder=True)
    assert value == 4.0
    assert kinds == []

    value, kinds = _classify_emission_correction(4.5, clip=False, rounder=True)
    assert value == 4.0  # banker's rounding via round()
    assert kinds == ["rounded"]


def test_classify_both_applied_in_order_clip_then_round() -> None:
    """Clip first (so a negative -> 0 doesn't get rounded into -1), then round."""
    value, kinds = _classify_emission_correction(-0.6, clip=True, rounder=True)
    assert value == 0.0
    # The clip bumped ``negative``; rounding 0.0 leaves it at 0.0 with
    # no fractional component so ``rounded`` is not added.
    assert kinds == ["negative"]


def test_classify_nan_passes_through_untouched() -> None:
    """NaN must not be silently rewritten to 0 (would mask data gaps)."""
    value, kinds = _classify_emission_correction(float("nan"), clip=True, rounder=True)
    assert math.isnan(value)
    assert kinds == []


def test_apply_safeguards_to_snapshot_points_bumps_counter() -> None:
    """Counter receives one entry per ``kind`` per emission."""
    points = [
        ForecastPoint(metric="m_forecast", labels={"id": "m"}, value=-1.0),
        ForecastPoint(metric="m_forecast", labels={"id": "m"}, value=2.6),
    ]
    counter: dict[str, int] = {}
    out = _apply_safeguards_to_snapshot_points(points, clip=True, rounder=True, counter=counter)
    assert [p.value for p in out] == [0.0, 3.0]
    assert counter == {"negative": 1, "rounded": 1}


def test_apply_safeguards_returns_input_when_both_off() -> None:
    """With both safeguards off the helper short-circuits — same object."""
    points = [ForecastPoint(metric="m_forecast", labels={"id": "m"}, value=-1.0)]
    counter: dict[str, int] = {}
    out = _apply_safeguards_to_snapshot_points(points, clip=False, rounder=False, counter=counter)
    assert out is points
    assert counter == {}


def test_apply_safeguards_to_curve_points_mutates_value_only() -> None:
    """Curve pass adjusts values but doesn't bump the counter (snapshot owns it)."""
    points = [
        ForecastCurvePoint(metric="m_forecast", labels={}, value=-2.0, timestamp_ms=1),
        ForecastCurvePoint(metric="m_forecast", labels={}, value=3.4, timestamp_ms=2),
    ]
    out = _apply_safeguards_to_curve_points(points, clip=True, rounder=True)
    assert [p.value for p in out] == [0.0, 3.0]
    # Timestamps survive the rewrite.
    assert [p.timestamp_ms for p in out] == [1, 2]


def test_effective_emission_safeguards_count_profile_opts_in() -> None:
    """``data_profile: count`` flips both unset toggles to True."""
    q = QueryConfig(id="m", promql="up", data_profile="count")
    clip, rounder = effective_emission_safeguards(q)
    assert clip is True
    assert rounder is True


def test_effective_emission_safeguards_explicit_override_wins() -> None:
    """Explicit False on the toggle wins over the count-profile default.

    Operators occasionally want the count profile (for the auto-select
    nudge in the future) without the rounding — the override pathway
    has to actually work.
    """
    q = QueryConfig(
        id="m",
        promql="up",
        data_profile="count",
        emission=QueryEmissionConfig(round_to_integer=False),
    )
    clip, rounder = effective_emission_safeguards(q)
    assert clip is True
    assert rounder is False


def test_effective_emission_safeguards_continuous_profile_keeps_off() -> None:
    """Without ``data_profile: count`` (or explicit True), both safeguards stay off."""
    q = QueryConfig(id="m", promql="up", data_profile="continuous")
    clip, rounder = effective_emission_safeguards(q)
    assert clip is False
    assert rounder is False
