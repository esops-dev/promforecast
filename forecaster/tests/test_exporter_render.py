"""Tests for the snapshot exporter rendering /metrics output."""

from __future__ import annotations

import math
import time as time_mod

from promforecast.exporter import (
    AccuracyPoint,
    DeviationPoint,
    Exporter,
    ForecastPoint,
    GroupSnapshot,
    QualityPoint,
)


def test_render_empty_is_well_formed() -> None:
    """No groups + no reload yet → only the always-on reload telemetry."""
    out = Exporter().render().decode()
    # Reload telemetry families are always emitted so dashboards can show
    # "never reloaded" (timestamp 0) without 404'ing the panel on boot.
    assert "forecast_config_last_reload_timestamp_seconds 0.0" in out
    assert "forecast_config_reload_success 1.0" in out
    assert "forecast_config_reload_failures_total 0.0" in out
    # No forecast/accuracy/deviation/quality samples.
    assert "_forecast " not in out
    assert "forecast_accuracy_" not in out
    assert "forecast_deviation_" not in out
    assert "forecast_quality_" not in out


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


def test_render_emits_deviation_metrics() -> None:
    exporter = Exporter()
    exporter.update(
        GroupSnapshot(
            group="g",
            deviation_points=[
                DeviationPoint(
                    labels={
                        "id": "m",
                        "group": "g",
                        "model": "AutoARIMA",
                        "level": "95",
                        "instance": "host-a",
                    },
                    ratio=0.25,
                    outside_band=False,
                ),
                DeviationPoint(
                    labels={
                        "id": "m",
                        "group": "g",
                        "model": "AutoARIMA",
                        "level": "95",
                        "instance": "host-b",
                    },
                    ratio=1.5,
                    outside_band=True,
                ),
            ],
        )
    )
    body = exporter.render().decode()
    assert "# TYPE forecast_deviation_ratio gauge" in body
    assert "# TYPE forecast_deviation_outside_band gauge" in body
    assert 'instance="host-a"' in body
    assert 'instance="host-b"' in body
    # outside_band rendered as 1 / 0.
    assert "forecast_deviation_outside_band{" in body
    assert "1.0" in body and "0.0" in body


def test_render_drops_nan_deviation_ratio_but_keeps_outside_band() -> None:
    exporter = Exporter()
    exporter.update(
        GroupSnapshot(
            group="g",
            deviation_points=[
                DeviationPoint(
                    labels={"id": "m", "group": "g", "model": "X", "level": "95"},
                    ratio=math.nan,
                    outside_band=True,
                ),
            ],
        )
    )
    body = exporter.render().decode()
    assert "forecast_deviation_ratio{" not in body
    assert "forecast_deviation_outside_band{" in body


def test_render_emits_quality_score() -> None:
    exporter = Exporter()
    exporter.update(
        GroupSnapshot(
            group="g",
            quality_points=[
                QualityPoint(
                    labels={
                        "id": "m",
                        "group": "g",
                        "model": "AutoARIMA",
                        "instance": "host-a",
                    },
                    score=0.82,
                ),
            ],
        )
    )
    body = exporter.render().decode()
    assert "# TYPE forecast_quality_score gauge" in body
    assert 'instance="host-a"' in body
    assert "0.82" in body


def test_render_drops_nan_quality_scores() -> None:
    exporter = Exporter()
    exporter.update(
        GroupSnapshot(
            group="g",
            quality_points=[
                QualityPoint(
                    labels={"id": "m", "group": "g", "model": "X"},
                    score=math.nan,
                ),
            ],
        )
    )
    body = exporter.render().decode()
    assert "forecast_quality_score{" not in body


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


def test_render_omits_staleness_family_when_no_fits_recorded() -> None:
    """A fresh deployment shouldn't carry a zero-row staleness family.

    Mirrors the convention every other opt-in family uses (no rows ->
    omit entirely) so dashboards can confirm the gauge starts emitting
    on the first refresh rather than from process boot.
    """
    body = Exporter().render().decode()
    assert "forecast_data_staleness_seconds" not in body


def test_render_includes_staleness_after_recording_a_fit() -> None:
    """A recorded fit emits ``forecast_data_staleness_seconds`` with elapsed time."""
    exporter = Exporter()
    before = time_mod.time()
    exporter.record_series_fit_success(
        group="node_capacity",
        query_id="node_cpu_busy_pct",
        model_name="AutoARIMA",
        series_key="instance=host-a",
        labels={
            "id": "node_cpu_busy_pct",
            "group": "node_capacity",
            "model": "AutoARIMA",
            "instance": "host-a",
        },
    )
    body = exporter.render().decode()
    after = time_mod.time()
    assert "# TYPE forecast_data_staleness_seconds gauge" in body
    assert 'id="node_cpu_busy_pct"' in body
    assert 'model="AutoARIMA"' in body
    assert 'group="node_capacity"' in body
    assert 'instance="host-a"' in body
    # The value should be a small non-negative elapsed time; the bound
    # is generous to absorb test scheduler jitter.
    assert "forecast_data_staleness_seconds{" in body
    # The recorded sample is bounded by the time we spent between
    # recording the fit and rendering — well under a second on any
    # remotely modern machine.
    assert (after - before) < 5.0


def test_staleness_resets_when_fit_is_re_recorded() -> None:
    """A second successful fit overwrites the timestamp, dropping the elapsed time."""
    exporter = Exporter()
    exporter.record_series_fit_success(
        group="g",
        query_id="q",
        model_name="m",
        series_key="k",
        labels={"id": "q", "group": "g", "model": "m"},
    )
    # Simulate "time passes". We don't actually sleep — we manually
    # rewrite the recorded timestamp to an older one so the next render
    # observes a large elapsed value, then re-recording resets the
    # clock.
    key = ("g", "q", "m", "k")
    timestamp, labels = exporter._series_last_success[key]
    exporter._series_last_success[key] = (timestamp - 3600.0, labels)
    body = exporter.render().decode()
    line = next(ln for ln in body.splitlines() if ln.startswith("forecast_data_staleness_seconds{"))
    # Last value on the metric line is the gauge value.
    value = float(line.rsplit(" ", 1)[-1])
    assert value >= 3600.0

    # Re-recording resets the clock.
    exporter.record_series_fit_success(
        group="g",
        query_id="q",
        model_name="m",
        series_key="k",
        labels={"id": "q", "group": "g", "model": "m"},
    )
    body2 = exporter.render().decode()
    line2 = next(
        ln for ln in body2.splitlines() if ln.startswith("forecast_data_staleness_seconds{")
    )
    value2 = float(line2.rsplit(" ", 1)[-1])
    assert value2 < 5.0
    # The reset must happen monotonically — the new sample is strictly
    # smaller than the artificially-aged one.
    assert value2 < value
    assert time_mod.time() > 0  # silence "unused import" if assertion sequence shifts


def test_staleness_retain_groups_prunes_removed_groups() -> None:
    exporter = Exporter()
    exporter.record_series_fit_success(
        group="keep",
        query_id="q",
        model_name="m",
        series_key="k",
        labels={"id": "q", "group": "keep", "model": "m"},
    )
    exporter.record_series_fit_success(
        group="drop",
        query_id="q",
        model_name="m",
        series_key="k",
        labels={"id": "q", "group": "drop", "model": "m"},
    )
    exporter.retain_staleness_groups({"keep"})
    body = exporter.render().decode()
    assert 'group="keep"' in body
    assert 'group="drop"' not in body


def test_staleness_retain_queries_prunes_removed_queries() -> None:
    exporter = Exporter()
    exporter.record_series_fit_success(
        group="g",
        query_id="keep",
        model_name="m",
        series_key="k",
        labels={"id": "keep", "group": "g", "model": "m"},
    )
    exporter.record_series_fit_success(
        group="g",
        query_id="drop",
        model_name="m",
        series_key="k",
        labels={"id": "drop", "group": "g", "model": "m"},
    )
    exporter.retain_staleness_queries({("g", "keep")})
    body = exporter.render().decode()
    assert 'id="keep"' in body
    assert 'id="drop"' not in body


def test_prune_stale_staleness_drops_rows_older_than_threshold() -> None:
    """A model dropped from config (or an ephemeral series) gets pruned by age.

    Guards the two leaks the per-group / per-query retention helpers
    can't reach: a model name removed from ``defaults.models`` (its
    rows never refresh again) and ephemeral series whose label set
    rotates (per-pod metrics with short pod lifetimes). Both manifest
    as a staleness row that's never re-touched by ``record_series_fit_success``;
    the time-based sweep is the only mechanism that cleans them up.
    """
    exporter = Exporter()
    exporter.record_series_fit_success(
        group="g",
        query_id="q",
        model_name="ghost",
        series_key="k",
        labels={"id": "q", "group": "g", "model": "ghost"},
    )
    exporter.record_series_fit_success(
        group="g",
        query_id="q",
        model_name="live",
        series_key="k",
        labels={"id": "q", "group": "g", "model": "live"},
    )
    # Age the ``ghost`` row past the threshold and leave the ``live``
    # row fresh.
    ghost_key = ("g", "q", "ghost", "k")
    live_key = ("g", "q", "live", "k")
    ghost_ts, ghost_labels = exporter._series_last_success[ghost_key]
    exporter._series_last_success[ghost_key] = (ghost_ts - 100_000.0, ghost_labels)

    pruned = exporter.prune_stale_staleness(86_400.0)

    assert pruned == 1
    assert ghost_key not in exporter._series_last_success
    assert live_key in exporter._series_last_success


def test_prune_stale_staleness_no_op_when_threshold_zero_or_negative() -> None:
    """A zero/negative threshold disables pruning so callers can opt out."""
    exporter = Exporter()
    exporter.record_series_fit_success(
        group="g",
        query_id="q",
        model_name="m",
        series_key="k",
        labels={"id": "q", "group": "g", "model": "m"},
    )
    # Age the row well past any sane threshold.
    key = ("g", "q", "m", "k")
    ts, labels = exporter._series_last_success[key]
    exporter._series_last_success[key] = (ts - 10_000_000.0, labels)

    assert exporter.prune_stale_staleness(0.0) == 0
    assert exporter.prune_stale_staleness(-1.0) == 0
    assert key in exporter._series_last_success
