"""Unit tests for the hyperparameter tuner.

Covers the pure helpers (grid generation, improvement-percentage
math, run-outcome classification, recommendation push payload) and
the CLI's early-exit paths (missing config, missing sink). The full
async grid walk is exercised against the docker-compose dev stack
manually (``make dev-up`` + ``promforecast tune --config docker/dev-config.yaml``);
no in-process integration test ships because the grid is genuinely
non-deterministic across model versions and would be a flaky-on-CI
regression target.
"""

from __future__ import annotations

import asyncio
import io
import textwrap
from pathlib import Path

import pytest

from promforecast import tuner as tuner_module
from promforecast.tuner import (
    RecommendedParam,
    _improvement_pct,
    _push_results,
    _run_outcome,
    _season_length_grid,
    run_tuner,
)

# ---------- pure helpers ---------------------------------------------------


def test_season_length_grid_returns_half_and_double() -> None:
    grid = _season_length_grid(288)
    assert grid == [144, 576]


def test_season_length_grid_caps_huge_doubles() -> None:
    # ``current = 6000`` -> ``double = 12000`` is above the 10000 cap;
    # only the halved candidate survives.
    grid = _season_length_grid(6000)
    assert grid == [3000]


def test_season_length_grid_floor_below_two() -> None:
    # A configured season_length of 2 cannot be halved further; the
    # grid still emits the doubled candidate.
    grid = _season_length_grid(2)
    assert grid == [4]


def test_improvement_pct_positive_when_candidate_better() -> None:
    assert _improvement_pct(baseline=20.0, candidate=10.0) == pytest.approx(50.0)


def test_improvement_pct_negative_when_candidate_worse() -> None:
    assert _improvement_pct(baseline=10.0, candidate=20.0) == pytest.approx(-100.0)


def test_improvement_pct_zero_when_baseline_zero() -> None:
    # Avoid divide-by-zero when the configured baseline produces an
    # impossible-to-beat 0 MAPE on a degenerate series.
    assert _improvement_pct(baseline=0.0, candidate=5.0) == 0.0


def test_run_outcome_ok_when_no_failures() -> None:
    assert _run_outcome(failed_queries=0, group_count=3) == "ok"


def test_run_outcome_partial_with_some_failures() -> None:
    assert _run_outcome(failed_queries=2, group_count=3) == "partial"


def test_run_outcome_error_when_no_groups() -> None:
    assert _run_outcome(failed_queries=1, group_count=0) == "error"


# ---------- CLI entry-point early exits ------------------------------------


_CONFIG_YAML = textwrap.dedent(
    """\
    apiVersion: promforecast.io/v1
    datasource:
      url: http://localhost:8428
    server:
      listen: ":9091"
      refresh_interval: 1h
    safety:
      max_series_per_query: 100
    defaults:
      lookback: 14d
      step: 5m
      horizon: 24h
      models: [AutoARIMA, SeasonalNaive]
      season_length: 288
      min_points: 200
      confidence_levels: [80]
    sink:
      remote_write:
        enabled: false
    groups:
      - name: node_capacity
        queries:
          - id: node_cpu_busy_pct
            promql: 'avg by (instance)(rate(node_cpu_seconds_total{mode!="idle"}[5m])) * 100'
    """
)


def _write_config(tmp_path: Path) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(_CONFIG_YAML)
    return path


def test_run_tuner_missing_config_returns_1(tmp_path: Path) -> None:
    out = io.StringIO()
    rc = run_tuner(
        config_path=tmp_path / "does-not-exist.yaml",
        group_selector=None,
        sink_url_override=None,
        out=out,
    )
    assert rc == 1
    assert "config not found" in out.getvalue()


def test_run_tuner_missing_sink_returns_1(tmp_path: Path) -> None:
    out = io.StringIO()
    rc = run_tuner(
        config_path=_write_config(tmp_path),
        group_selector=None,
        sink_url_override=None,  # config also has no sink configured
        out=out,
    )
    assert rc == 1
    assert "no sink configured" in out.getvalue()


def test_run_tuner_unknown_group_returns_1(tmp_path: Path) -> None:
    out = io.StringIO()
    rc = run_tuner(
        config_path=_write_config(tmp_path),
        group_selector=["does-not-exist"],
        sink_url_override="http://victoriametrics:8428/api/v1/import/prometheus",
        out=out,
    )
    assert rc == 1
    assert "no groups selected" in out.getvalue()


def test_recommended_param_is_hashable_dataclass() -> None:
    """``frozen=True`` is intentional — recommendations are passed through a
    chain of functions and should be safe to use as dict keys / set members.
    """
    rec = RecommendedParam(
        group="g",
        query="q",
        model="AutoARIMA",
        name="season_length",
        current_value="288",
        recommended_value="576",
        expected_mape_improvement_pct=12.3,
    )
    assert hash(rec)  # should not raise
    assert rec.expected_mape_improvement_pct == 12.3


def test_module_help_is_documented() -> None:
    """The module docstring should mention the metric name a Grafana
    panel author needs to grep for. Regression guard against silent
    metric renames."""
    assert "forecast_recommended_param" in tuner_module.__doc__ or ""


# ---------- push payload shape ---------------------------------------------


class _RecordingSink:
    """A drop-in ``ForecastSink`` that captures the written points.

    Lets us assert the metric names, labels, and values the tuner
    actually pushes without standing up a real TSDB. Matches the
    protocol surface (``write`` + ``aclose``) so it satisfies the
    ``ForecastSink`` Protocol.
    """

    name = "recording"

    def __init__(self) -> None:
        self.written: list = []
        self.closed = False

    async def write(self, points, *, on_retry=None) -> None:
        self.written.extend(list(points))

    async def aclose(self) -> None:
        self.closed = True


def test_push_results_emits_gauge_timestamps_not_counter() -> None:
    """Regression guard for the stateless-pusher semantics.

    Before the fix the tuner pushed ``forecast_tuner_runs_total`` as
    a constant ``1.0`` per run — a no-op for ``rate()`` against the
    flat series the TSDB ended up with. The fix swaps to
    ``forecast_tuner_last_run_timestamp_seconds{group, outcome}`` as
    a gauge of unix-seconds; this test pins both the metric name and
    the gauge-of-timestamp semantics so a future revert is caught at
    CI time.
    """
    sink = _RecordingSink()
    recs = [
        RecommendedParam(
            group="g1",
            query="q1",
            model="AutoARIMA",
            name="season_length",
            current_value="288",
            recommended_value="576",
            expected_mape_improvement_pct=12.5,
        ),
    ]
    asyncio.run(
        _push_results(
            sink=sink,
            recommendations=recs,
            outcome="ok",
            elapsed_seconds=42.0,
            groups=["g1", "g2"],
        )
    )
    metrics = {p.metric for p in sink.written}
    assert "forecast_recommended_param" in metrics
    assert "forecast_tuner_last_run_timestamp_seconds" in metrics
    assert "forecast_tuner_last_run_duration_seconds" in metrics
    # The pre-fix counter must not reappear under either name shape.
    assert "forecast_tuner_runs_total" not in metrics
    assert "forecast_tuner_run_duration_seconds" not in metrics

    # Timestamp gauge value should be approximately now (unix seconds),
    # not the constant 1.0 the pre-fix counter pushed.
    timestamp_points = [
        p for p in sink.written if p.metric == "forecast_tuner_last_run_timestamp_seconds"
    ]
    assert len(timestamp_points) == 2  # one per group
    for p in timestamp_points:
        assert p.value > 1_700_000_000  # roughly post-2023
        assert p.labels["outcome"] == "ok"

    # The recommendation row carries the improvement-pct as the sample value.
    rec_points = [p for p in sink.written if p.metric == "forecast_recommended_param"]
    assert len(rec_points) == 1
    assert rec_points[0].value == 12.5
    assert rec_points[0].labels["current_value"] == "288"
    assert rec_points[0].labels["recommended_value"] == "576"


def test_push_results_empty_recommendations_still_emits_telemetry() -> None:
    """A no-recommendations run must still push the per-group
    telemetry so the staleness alert
    ``time() - max by (group) (forecast_tuner_last_run_timestamp_seconds) > 8d``
    fires on a missing run rather than on a successful "no
    recommendations" run.
    """
    sink = _RecordingSink()
    asyncio.run(
        _push_results(
            sink=sink,
            recommendations=[],
            outcome="ok",
            elapsed_seconds=10.0,
            groups=["g1"],
        )
    )
    metrics = {p.metric for p in sink.written}
    assert metrics == {
        "forecast_tuner_last_run_timestamp_seconds",
        "forecast_tuner_last_run_duration_seconds",
    }
