"""Tests for the snapshot exporter rendering /metrics output."""

from __future__ import annotations

import math

from promforecast.exporter import AccuracyPoint, Exporter, ForecastPoint, GroupSnapshot


def test_render_empty_is_well_formed() -> None:
    out = Exporter().render()
    assert out == b""


def test_render_includes_forecast_metric_and_op_metrics() -> None:
    exporter = Exporter()
    exporter.update(
        GroupSnapshot(
            group="node_capacity",
            points=[
                ForecastPoint(
                    metric="node_cpu_busy_pct_forecast",
                    labels={
                        "instance": "host-a",
                        "group": "node_capacity",
                        "model": "SeasonalNaive",
                        "horizon": "next",
                    },
                    value=42.0,
                ),
                ForecastPoint(
                    metric="node_cpu_busy_pct_forecast_lower",
                    labels={
                        "instance": "host-a",
                        "group": "node_capacity",
                        "model": "SeasonalNaive",
                        "horizon": "next",
                        "level": "95",
                    },
                    value=30.0,
                ),
            ],
            last_run_timestamp=1700000000.0,
            run_duration_seconds=1.5,
            series_count=1,
        )
    )
    body = exporter.render().decode()
    assert "# TYPE node_cpu_busy_pct_forecast gauge" in body
    assert 'instance="host-a"' in body
    assert 'level="95"' in body
    assert "forecast_last_run_timestamp_seconds" in body
    assert 'group="node_capacity"' in body


def test_render_increments_failures() -> None:
    exporter = Exporter()
    exporter.increment_failure("g", "query_error")
    exporter.increment_failure("g", "query_error")
    exporter.increment_failure("g", "fit_error")
    body = exporter.render().decode()
    assert "# TYPE forecast_failures_total counter" in body
    assert 'forecast_failures_total{group="g",reason="query_error"} 2.0' in body
    assert 'forecast_failures_total{group="g",reason="fit_error"} 1.0' in body


def test_render_emits_accuracy_metrics_per_horizon() -> None:
    exporter = Exporter()
    exporter.update(
        GroupSnapshot(
            group="g",
            accuracy_points=[
                AccuracyPoint(
                    labels={
                        "id": "m",
                        "group": "g",
                        "model": "AutoARIMA",
                        "horizon": "1h",
                        "instance": "host-a",
                    },
                    mape=12.0,
                    mase=0.8,
                ),
                AccuracyPoint(
                    labels={
                        "id": "m",
                        "group": "g",
                        "model": "AutoARIMA",
                        "horizon": "24h",
                        "instance": "host-a",
                    },
                    mape=25.0,
                    mase=1.2,
                ),
            ],
        )
    )
    body = exporter.render().decode()
    assert "# TYPE forecast_accuracy_mape gauge" in body
    assert "# TYPE forecast_accuracy_mase gauge" in body
    assert 'horizon="1h"' in body and 'horizon="24h"' in body
    assert "12.0" in body and "25.0" in body


def test_render_drops_nan_accuracy_values() -> None:
    exporter = Exporter()
    exporter.update(
        GroupSnapshot(
            group="g",
            accuracy_points=[
                AccuracyPoint(
                    labels={"id": "m", "group": "g", "model": "X", "horizon": "1h"},
                    mape=math.nan,
                    mase=math.nan,
                ),
            ],
        )
    )
    body = exporter.render().decode()
    # Family is declared but no samples are emitted for the NaN row.
    assert "forecast_accuracy_mape{" not in body
    assert "forecast_accuracy_mase{" not in body


def test_render_escapes_label_values() -> None:
    exporter = Exporter()
    exporter.update(
        GroupSnapshot(
            group="g",
            points=[
                ForecastPoint(
                    metric="x_forecast",
                    labels={"group": "g", "weird": 'has"quotes\\and\nnewline'},
                    value=1.0,
                )
            ],
        )
    )
    body = exporter.render().decode()
    assert 'weird="has\\"quotes\\\\and\\nnewline"' in body
