"""Tests for :class:`SeriesPointFactory`.

Pulled out of the runner so the per-metric assembly is exercised in
isolation: we can hand it a synthetic backtest outcome and assert exactly
what comes back without spinning up the full async fit pipeline.
"""

from __future__ import annotations

import math

import pandas as pd

from promforecast.accuracy import AccuracyResult
from promforecast.point_factory import (
    BacktestOutcome,
    BandSummary,
    SeriesPointFactory,
)
from promforecast.selector import SelectionResult


def _factory() -> SeriesPointFactory:
    return SeriesPointFactory(
        metric_id="m",
        base_labels={"__name__": "raw_metric", "instance": "host-a"},
        group="g",
        levels=[80],
    )


def test_forecast_points_strip_internal_name_label() -> None:
    """``__name__`` is the raw TSDB metric name; we replace it with the
    forecast metric stem and must not leak it into emitted labels."""
    factory = _factory()
    df = pd.DataFrame(
        {
            "ds": pd.to_datetime(["2024-01-01"]),
            "yhat": [42.0],
            "yhat_lower_80": [40.0],
            "yhat_upper_80": [44.0],
        }
    )
    points = list(factory.forecast_points(df, model_name="SeasonalNaive", selection=None))
    assert {p.metric for p in points} == {
        "m_forecast",
        "m_forecast_lower",
        "m_forecast_upper",
    }
    for point in points:
        assert "__name__" not in point.labels
        assert point.labels["instance"] == "host-a"
        assert point.labels["group"] == "g"
        assert point.labels["model"] == "SeasonalNaive"


def test_forecast_points_attach_selection_labels_in_auto_select_mode() -> None:
    factory = _factory()
    df = pd.DataFrame({"ds": pd.to_datetime(["2024-01-01"]), "yhat": [1.0]})
    selection = SelectionResult(
        model="SeasonalNaive",
        best_model="AutoARIMA",
        fallback_applied=True,
        mape=99.0,
    )
    points = list(factory.forecast_points(df, model_name="SeasonalNaive", selection=selection))
    assert points[0].labels["best_model"] == "AutoARIMA"
    assert points[0].labels["model_fallback"] == "true"


def test_quality_points_skip_models_without_backtest() -> None:
    """Auto-select fallback models that were never backtested have no
    quality data — the factory must skip them rather than emit zero,
    matching the behaviour the runner integration tests lock in."""
    factory = _factory()
    backtests = {
        "AutoARIMA": BacktestOutcome(
            results=[AccuracyResult(model="AutoARIMA", horizon="1h", mape=10.0, mase=0.5)],
            last_actual=10.0,
            last_yhat=11.0,
            last_bands={80: BandSummary(lower=9.0, upper=13.0)},
        ),
    }
    points = list(
        factory.quality_points(
            backtests=backtests,
            emitted_models=["AutoARIMA", "SeasonalNaive"],  # SeasonalNaive missing from backtests
            train_y=[1.0, 2.0, 3.0],
            coverage=1.0,
        )
    )
    # Aggregated (``horizon=""``) row + one per-horizon row from the
    # single backtest result. SeasonalNaive contributes nothing
    # because it has no backtest data.
    assert len(points) == 2
    models = {p.labels["model"] for p in points}
    assert models == {"AutoARIMA"}
    horizons = {p.labels.get("horizon", "") for p in points}
    assert horizons == {"", "1h"}
    for p in points:
        assert math.isfinite(p.score)


def test_deviation_points_skip_models_without_band_at_level() -> None:
    """Per-level fan-out must skip levels that the backtest didn't produce,
    rather than emitting NaN-band rows that would clutter dashboards."""
    factory = SeriesPointFactory(
        metric_id="m",
        base_labels={"instance": "host-a"},
        group="g",
        levels=[80, 95],  # config asks for two levels...
    )
    backtests = {
        "AutoARIMA": BacktestOutcome(
            results=[],
            last_actual=10.0,
            last_yhat=10.0,
            last_bands={80: BandSummary(lower=8.0, upper=12.0)},  # ...but only 80 came back
        ),
    }
    points = list(factory.deviation_points(backtests=backtests, emitted_models=["AutoARIMA"]))
    assert len(points) == 1
    assert points[0].labels["level"] == "80"


def test_accuracy_points_carry_id_and_horizon_labels() -> None:
    factory = _factory()
    backtest_results = {
        "AutoARIMA": [
            AccuracyResult(model="AutoARIMA", horizon="1h", mape=12.0, mase=0.6),
            AccuracyResult(model="AutoARIMA", horizon="24h", mape=20.0, mase=0.9),
        ]
    }
    points = list(factory.accuracy_points(backtest_results))
    assert {p.labels["horizon"] for p in points} == {"1h", "24h"}
    assert all(p.labels["id"] == "m" for p in points)
    assert all(p.labels["group"] == "g" for p in points)
