"""Tests for the composite quality score."""

from __future__ import annotations

import math

from promforecast.quality import compute


def test_perfect_inputs_score_one() -> None:
    score = compute(backtest_mape=0.0, band_width_relative=0.0, coverage=1.0)
    assert score == 1.0


def test_zero_coverage_zeroes_score() -> None:
    # Even with perfect MAPE and a tight band, no data means no quality.
    score = compute(backtest_mape=0.0, band_width_relative=0.0, coverage=0.0)
    assert score == 0.0


def test_high_mape_drags_score_down() -> None:
    # 50% MAPE -> mape_score = 0.5; band score 1.0 (zero-width); coverage 1.0.
    score = compute(backtest_mape=50.0, band_width_relative=0.0, coverage=1.0)
    assert math.isclose(score, 0.5)


def test_wide_band_halves_score_at_unity() -> None:
    # band_width_relative = 1.0 -> band_score = 0.5.
    score = compute(backtest_mape=0.0, band_width_relative=1.0, coverage=1.0)
    assert math.isclose(score, 0.5)


def test_sub_scores_multiply() -> None:
    # 20% MAPE -> 0.8; band 0.5 -> 1/1.5 = 0.6667; coverage 0.9.
    expected = 0.8 * (1.0 / 1.5) * 0.9
    score = compute(backtest_mape=20.0, band_width_relative=0.5, coverage=0.9)
    assert math.isclose(score, expected, rel_tol=1e-9)


def test_score_is_clamped_to_unit_interval() -> None:
    score = compute(backtest_mape=200.0, band_width_relative=0.0, coverage=1.0)
    assert score == 0.0


def test_none_inputs_zero_their_subscore() -> None:
    assert compute(backtest_mape=None, band_width_relative=0.0, coverage=1.0) == 0.0
    assert compute(backtest_mape=0.0, band_width_relative=None, coverage=1.0) == 0.0
    assert compute(backtest_mape=0.0, band_width_relative=0.0, coverage=None) == 0.0


def test_non_finite_inputs_zero_their_subscore() -> None:
    assert compute(backtest_mape=float("nan"), band_width_relative=0.0, coverage=1.0) == 0.0
    assert compute(backtest_mape=0.0, band_width_relative=float("inf"), coverage=1.0) == 0.0
