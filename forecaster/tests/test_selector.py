"""Tests for the auto-select / fallback selector."""

from __future__ import annotations

import math

from promforecast.selector import select


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
