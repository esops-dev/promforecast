"""Tests for the cold-start sibling-borrowing module.

The cold-start helpers are pure-function: most cases exercise them
without spinning up the runner. A separate runner test plumbs cold-
start through the orchestrator so we don't lose coverage on the
integration seams.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

from promforecast import cold_start as cold_start_mod
from promforecast.source import SeriesFrame


def test_normalize_sibling_shape_returns_unit_mean_median() -> None:
    """Three siblings of the same shape but different scale collapse to one prior.

    The normalisation is what allows a tiny service plus a huge one of
    the same shape to contribute equally to the prior. We verify by
    constructing three identical-shape siblings at 3 scales and
    checking the per-step median is shape/mean (~ 1.0 at the peaks).
    """
    base = [1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0]  # period 4 oscillation
    siblings = [
        [v * 1.0 for v in base],
        [v * 10.0 for v in base],
        [v * 100.0 for v in base],
    ]
    shape = cold_start_mod.normalize_sibling_shape(siblings, season_length=4)
    assert shape is not None
    assert len(shape) == 4
    # All three siblings have the same shape after normalisation, so the
    # per-step median equals the per-step normalised value.
    expected_mean = sum(base) / len(base)
    for i in range(4):
        assert math.isclose(shape[i], base[i] / expected_mean, rel_tol=1e-9)


def test_normalize_sibling_shape_drops_constant_siblings() -> None:
    """A sibling with mean ~ 0 must not poison the prior.

    Dividing by ~0 would explode the normalised values into inf/NaN
    and dominate the median; we want the constant sibling silently
    dropped rather than swallow the whole pool.
    """
    siblings = [
        [0.0, 0.0, 0.0, 0.0],  # constant zero — dropped
        [1.0, 2.0, 3.0, 2.0],  # healthy
        [2.0, 4.0, 6.0, 4.0],  # healthy
    ]
    shape = cold_start_mod.normalize_sibling_shape(siblings, season_length=4)
    assert shape is not None
    # The healthy siblings share the same shape; constants didn't
    # contribute to the median.
    expected_mean = sum([1.0, 2.0, 3.0, 2.0]) / 4
    assert math.isclose(shape[0], 1.0 / expected_mean, rel_tol=1e-9)


def test_normalize_sibling_shape_returns_none_when_no_siblings_usable() -> None:
    """Every sibling unusable -> None, not a degenerate prior."""
    assert cold_start_mod.normalize_sibling_shape([], season_length=4) is None
    assert cold_start_mod.normalize_sibling_shape([[0.0] * 4], season_length=4) is None
    # A sibling too short to fill the seasonal cycle is dropped — and
    # if every sibling is dropped we surface ``None``.
    assert cold_start_mod.normalize_sibling_shape([[1.0, 2.0]], season_length=4) is None


def test_build_cold_start_forecast_repeats_shape_over_horizon() -> None:
    """The borrowed shape repeats across the horizon, scaled by the anchor.

    Forecast row ``i`` should carry ``shape[i % len(shape)] * anchor``.
    """
    shape = [0.5, 1.5]  # mean 1.0 — already in unit-mean form
    start = datetime(2026, 5, 21, tzinfo=UTC)
    forecast = cold_start_mod.build_cold_start_forecast(
        prior_shape=shape,
        anchor_value=10.0,
        horizon=5,
        step=timedelta(minutes=5),
        start=start,
    )
    assert len(forecast) == 5
    # The first row is one step after the anchor (the forecast starts
    # *after* now).
    assert forecast["ds"].iloc[0] == start + timedelta(minutes=5)
    # ds spacing matches step.
    from itertools import pairwise  # noqa: PLC0415

    ds = [d.to_pydatetime() for d in forecast["ds"].tolist()]
    diffs = [(b - a) for a, b in pairwise(ds)]
    assert all(d == timedelta(minutes=5) for d in diffs)
    assert forecast["yhat"].tolist() == [5.0, 15.0, 5.0, 15.0, 5.0]


def test_build_cold_start_forecast_returns_empty_for_zero_horizon() -> None:
    """Defensive: a non-positive horizon must not crash the helper."""
    forecast = cold_start_mod.build_cold_start_forecast(
        prior_shape=[1.0],
        anchor_value=1.0,
        horizon=0,
        step=timedelta(minutes=5),
        start=datetime(2026, 5, 21, tzinfo=UTC),
    )
    assert forecast.empty


def _frame(values: list[float]) -> SeriesFrame:
    start = datetime(2026, 5, 21, tzinfo=UTC)
    timestamps = [start + timedelta(minutes=5 * i) for i in range(len(values))]
    return SeriesFrame(labels={"instance": "peer"}, timestamps=timestamps, values=values)


def test_synthesise_cold_start_forecast_caps_on_min_siblings() -> None:
    """Below ``min_siblings`` usable peers, the helper returns ``None``.

    The runner reads ``None`` as "fall back to the standard min_points
    drop" — so this test guards the contract that the cold-start path
    isn't entered with too thin a prior.
    """
    siblings = [_frame([1.0, 2.0, 1.0, 2.0])]
    out = cold_start_mod.synthesise_cold_start_forecast(
        target_values=[0.5],
        sibling_frames=siblings,
        season_length=2,
        horizon=4,
        step=timedelta(minutes=5),
        min_siblings=3,
        last_ts=datetime(2026, 5, 21, tzinfo=UTC),
    )
    assert out is None


def test_synthesise_cold_start_forecast_returns_scaled_forecast() -> None:
    """With enough siblings, the synthesised forecast lives on the target's scale.

    Anchor = target mean. The forecast magnitude should track that
    anchor — small target = small forecast even when peers are large.
    """
    siblings = [
        _frame([1.0, 2.0, 1.0, 2.0]),
        _frame([10.0, 20.0, 10.0, 20.0]),
        _frame([100.0, 200.0, 100.0, 200.0]),
    ]
    target = [50.0, 70.0]  # mean 60.0
    out = cold_start_mod.synthesise_cold_start_forecast(
        target_values=target,
        sibling_frames=siblings,
        season_length=2,
        horizon=2,
        step=timedelta(minutes=5),
        min_siblings=3,
        last_ts=datetime(2026, 5, 21, tzinfo=UTC),
    )
    assert out is not None
    assert out.sibling_count == 3
    # Sibling shape = [1/1.5, 2/1.5] = [0.6667, 1.333]; anchor = 60.
    # Forecast row 0 = 0.6667 * 60 = 40; row 1 = 1.333 * 60 = 80.
    yhat = out.forecast["yhat"].tolist()
    assert math.isclose(yhat[0], 40.0, rel_tol=1e-3)
    assert math.isclose(yhat[1], 80.0, rel_tol=1e-3)
    # Three siblings tracing the exact same normalised shape -> zero
    # dispersion -> agreement sub-score = 1.0; abundance = 3/3 = 1.0.
    assert out.shape_dispersion == 0.0


def test_cold_start_quality_score_perfect_agreement_hits_cap() -> None:
    """Sibling count at the minimum plus zero dispersion saturates to the cap.

    Operators set ``min_siblings`` as the contract; meeting it with full
    agreement should not score below the cap. The cap then prevents the
    quality from running past strict alert gates.
    """
    score = cold_start_mod.cold_start_quality_score(
        sibling_count=3, shape_dispersion=0.0, min_siblings=3, cap=0.5
    )
    assert math.isclose(score, 0.5)


def test_cold_start_quality_score_decays_with_dispersion() -> None:
    """Larger sibling disagreement decays the score below the cap."""
    score = cold_start_mod.cold_start_quality_score(
        sibling_count=3, shape_dispersion=1.0, min_siblings=3, cap=0.5
    )
    # agreement = 1/(1+1) = 0.5; raw = 1.0 * 0.5 = 0.5; capped at 0.5 -> 0.25.
    assert math.isclose(score, 0.25)


def test_cold_start_quality_score_zero_cap_returns_zero() -> None:
    """A non-positive cap disables cold-start quality entirely."""
    assert (
        cold_start_mod.cold_start_quality_score(
            sibling_count=10, shape_dispersion=0.0, min_siblings=3, cap=0.0
        )
        == 0.0
    )
