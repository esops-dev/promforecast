"""Tests for the auto-select / fallback selector."""

from __future__ import annotations

import math

import pytest

from promforecast.selector import ensemble_weights, select


def test_select_picks_lowest_mape() -> None:
    result = select(
        fallback_model="SeasonalNaive",
        fallback_mape_threshold=50.0,
        backtest_mape={"AutoARIMA": 12.0, "AutoETS": 9.5, "SeasonalNaive": 30.0},
    )
    assert result.model == "AutoETS"
    assert result.best_model == "AutoETS"
    assert result.fallback_applied is False
    assert result.mape == 9.5


def test_select_falls_back_when_winner_exceeds_threshold() -> None:
    result = select(
        fallback_model="SeasonalNaive",
        fallback_mape_threshold=20.0,
        backtest_mape={"AutoARIMA": 60.0, "AutoETS": 25.0},
    )
    assert result.model == "SeasonalNaive"
    assert result.best_model == "AutoETS"
    assert result.fallback_applied is True
    assert result.mape == 25.0


def test_select_no_threshold_disables_fallback() -> None:
    result = select(
        fallback_model="SeasonalNaive",
        fallback_mape_threshold=None,
        backtest_mape={"AutoARIMA": 999.0},
    )
    assert result.model == "AutoARIMA"
    assert result.fallback_applied is False


def test_select_falls_back_when_all_mape_nan() -> None:
    result = select(
        fallback_model="SeasonalNaive",
        fallback_mape_threshold=50.0,
        backtest_mape={"AutoARIMA": math.nan, "AutoETS": math.nan},
    )
    assert result.model == "SeasonalNaive"
    assert result.best_model == "SeasonalNaive"
    assert result.fallback_applied is True
    assert result.mape is None


def test_select_falls_back_on_empty_input() -> None:
    result = select(
        fallback_model="SeasonalNaive",
        fallback_mape_threshold=50.0,
        backtest_mape={},
    )
    assert result.model == "SeasonalNaive"
    assert result.fallback_applied is True
    assert result.mape is None


def test_ensemble_weights_inverse_mape_normalised_to_one() -> None:
    weights = ensemble_weights({"AutoARIMA": 10.0, "AutoETS": 20.0})
    # 1/10 vs 1/20 → 2/3 vs 1/3
    assert weights["AutoARIMA"] == pytest.approx(2.0 / 3.0)
    assert weights["AutoETS"] == pytest.approx(1.0 / 3.0)
    assert sum(weights.values()) == pytest.approx(1.0)


def test_ensemble_weights_drops_nan_models() -> None:
    weights = ensemble_weights({"A": 5.0, "B": math.nan})
    assert "B" not in weights
    assert weights["A"] == 1.0


def test_ensemble_weights_falls_back_to_equal_when_all_nan() -> None:
    weights = ensemble_weights({"A": math.nan, "B": math.nan})
    assert weights == {"A": 0.5, "B": 0.5}


def test_ensemble_weights_handles_zero_mape_without_blowup() -> None:
    weights = ensemble_weights({"perfect": 0.0, "ok": 10.0})
    # Zero MAPE is floored to a tiny epsilon so the perfect model dominates
    # without producing inf weights.
    assert weights["perfect"] > 0.99
    assert weights["ok"] < 0.01
    assert sum(weights.values()) == pytest.approx(1.0)


def test_ensemble_weights_empty_input_returns_empty() -> None:
    assert ensemble_weights({}) == {}


def test_select_prefers_candidate_subset_when_any_finite() -> None:
    """``candidate_preference`` nudges the search toward the listed names.

    When at least one preferred candidate has a finite score, the
    winner must come from the preferred set even if a non-preferred
    model has a lower score. This is the mechanism the runner uses to
    prefer intermittent-demand wrappers under
    ``data_profile: intermittent`` without forcing the operator to
    remove the safety-net continuous models from the config.
    """
    result = select(
        fallback_model="SeasonalNaive",
        fallback_mape_threshold=50.0,
        backtest_mape={"AutoARIMA": 5.0, "CrostonSBA": 12.0, "ADIDA": 18.0},
        candidate_preference=frozenset({"CrostonSBA", "ADIDA"}),
    )
    assert result.model == "CrostonSBA"


def test_select_falls_back_to_full_set_when_no_preferred_finite() -> None:
    """Empty preferred-set match -> rank across every model, not the fallback.

    A misconfigured intermittent profile that lists no intermittent
    models should still produce a forecast — picking the lowest of
    what's there is more useful than dropping the series.
    """
    result = select(
        fallback_model="SeasonalNaive",
        fallback_mape_threshold=50.0,
        backtest_mape={"AutoARIMA": 5.0, "AutoETS": 12.0},
        candidate_preference=frozenset({"CrostonSBA", "ADIDA"}),
    )
    assert result.model == "AutoARIMA"


def test_select_candidate_preference_still_honours_fallback_threshold() -> None:
    """A preferred winner whose score exceeds the threshold still falls back."""
    result = select(
        fallback_model="SeasonalNaive",
        fallback_mape_threshold=10.0,
        backtest_mape={"AutoARIMA": 8.0, "CrostonSBA": 60.0},
        candidate_preference=frozenset({"CrostonSBA"}),
    )
    assert result.model == "SeasonalNaive"
    assert result.best_model == "CrostonSBA"
    assert result.fallback_applied is True
