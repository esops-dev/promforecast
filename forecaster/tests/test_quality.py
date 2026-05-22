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


def test_imputed_fraction_penalises_coverage() -> None:
    # Full coverage but half imputed -> coverage sub-score halved.
    full = compute(backtest_mape=0.0, band_width_relative=0.0, coverage=1.0)
    half_imputed = compute(
        backtest_mape=0.0,
        band_width_relative=0.0,
        coverage=1.0,
        imputed_fraction=0.5,
    )
    assert math.isclose(half_imputed, full * 0.5)


def test_fully_imputed_coverage_zeroes_score() -> None:
    # If every "covered" point is fabricated, quality must collapse to 0.
    score = compute(
        backtest_mape=0.0,
        band_width_relative=0.0,
        coverage=1.0,
        imputed_fraction=1.0,
    )
    assert score == 0.0


def test_imputed_fraction_defaults_to_no_penalty() -> None:
    # Callers that don't run preprocessing pass ``imputed_fraction=None`` (or
    # leave it default) and must keep the historical behaviour.
    legacy = compute(backtest_mape=0.0, band_width_relative=0.0, coverage=0.9)
    explicit_none = compute(
        backtest_mape=0.0,
        band_width_relative=0.0,
        coverage=0.9,
        imputed_fraction=None,
    )
    assert math.isclose(legacy, explicit_none)


def test_mase_score_kind_perfect_mase_maps_to_one() -> None:
    """MASE = 0 (perfect) plus tight band plus full coverage = quality 1.0.

    Mirrors the MAPE = 0 case so an operator who switches profile gets
    the same upper-bound semantics.
    """
    score = compute(backtest_mape=0.0, band_width_relative=0.0, coverage=1.0, score_kind="mase")
    assert math.isclose(score, 1.0)


def test_mase_score_kind_baseline_lands_on_half() -> None:
    """MASE = 1 (matches seasonal naive) maps to 0.5 — the documented midpoint."""
    score = compute(backtest_mape=1.0, band_width_relative=0.0, coverage=1.0, score_kind="mase")
    assert math.isclose(score, 0.5)


def test_mase_score_kind_large_mase_decays_but_stays_positive() -> None:
    """MASE = 4 → accuracy sub-score = 1/5 = 0.2 — never quite zero.

    The smooth ``1 / (1 + mase)`` curve keeps an intermittent series with
    a noisy model from collapsing to a hard zero quality the way the
    MAPE-based path does on NaN input — operators still get a signal.
    """
    score = compute(backtest_mape=4.0, band_width_relative=0.0, coverage=1.0, score_kind="mase")
    assert math.isclose(score, 0.2)


def test_mase_score_kind_treats_none_as_zero() -> None:
    """NaN/None MASE maps to 0 — same defensive behaviour as the MAPE path."""
    assert (
        compute(backtest_mape=None, band_width_relative=0.0, coverage=1.0, score_kind="mase") == 0.0
    )
    assert (
        compute(
            backtest_mape=float("nan"),
            band_width_relative=0.0,
            coverage=1.0,
            score_kind="mase",
        )
        == 0.0
    )
