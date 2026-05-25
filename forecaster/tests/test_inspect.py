"""Tests for the inspector CLI's pure helpers.

The full inspector pipeline (config load -> PromQL fetch -> model
fit) is exercised end-to-end via docker-compose; here we cover the
pure parsing / rendering helpers + a synthetic-source happy path so
CI doesn't need network egress to verify the table shape.
"""

from __future__ import annotations

import io
import math
from pathlib import Path

import pytest

from promforecast.inspect import (
    BacktestFoldRow,
    BacktestSummary,
    InputStats,
    InspectError,
    InspectReport,
    parse_series_filter,
    render_text,
    run_inspect,
)


def test_parse_series_filter_handles_braces() -> None:
    assert parse_series_filter('{instance="host-a",device="sda1"}') == {
        "instance": "host-a",
        "device": "sda1",
    }


def test_parse_series_filter_accepts_bare_pairs() -> None:
    assert parse_series_filter('instance="host-a"') == {"instance": "host-a"}


def test_parse_series_filter_returns_none_on_empty() -> None:
    assert parse_series_filter(None) is None
    assert parse_series_filter("") is None
    assert parse_series_filter("{}") is None


def test_parse_series_filter_rejects_missing_equals() -> None:
    with pytest.raises(InspectError):
        parse_series_filter("{instance}")


def test_render_text_includes_each_section() -> None:
    report = InspectReport(
        query_id="node_cpu_busy_pct",
        group="node_capacity",
        model="SeasonalNaive",
        series_labels={"instance": "host-a"},
        step_seconds=60,
        season_length=1440,
        horizon_steps=12,
        input_stats=InputStats(
            point_count=200,
            gap_count=2,
            largest_gap_seconds=120.0,
            lookback_start="2026-05-23T00:00:00+00:00",
            lookback_end="2026-05-24T00:00:00+00:00",
        ),
        forecast_timestamps=["2026-05-24T01:00:00+00:00"],
        forecast_yhat=[42.0],
        forecast_bands={"95": {"lower": [30.0], "upper": [50.0]}},
        preprocessing={"gaps_filled": {"forward_fill": 2}, "outliers_replaced": 0},
        backtest=BacktestSummary(
            model="SeasonalNaive",
            n_holdout=12,
            mape=4.5,
            mase=0.8,
            rows=[
                BacktestFoldRow(
                    ds="2026-05-24T00:00:00+00:00",
                    actual=41.0,
                    yhat=42.5,
                    bands={"95": {"lower": 30.0, "upper": 50.0}},
                )
            ],
        ),
    )
    text = render_text(report)
    assert "Query:        node_cpu_busy_pct (group node_capacity)" in text
    assert "Model:        SeasonalNaive" in text
    assert "instance=host-a" in text
    assert "Forecast (first 1 of 1 rows):" in text
    assert "yhat=42.0000" in text
    assert "[95: 30.0000..50.0000]" in text
    assert "Backtest" in text
    assert "MAPE     : 4.5000%" in text
    assert "MASE     : 0.8000" in text


def test_render_text_handles_nan_values() -> None:
    report = InspectReport(
        query_id="m",
        group="g",
        model="X",
        series_labels={},
        step_seconds=60,
        season_length=1,
        horizon_steps=1,
        input_stats=InputStats(
            point_count=0,
            gap_count=0,
            largest_gap_seconds=0.0,
            lookback_start="",
            lookback_end="",
        ),
        forecast_timestamps=[],
        forecast_yhat=[],
        forecast_bands={},
        preprocessing={},
        backtest=BacktestSummary(
            model="X",
            n_holdout=0,
            mape=math.nan,
            mase=math.nan,
            rows=[],
        ),
    )
    text = render_text(report)
    assert "MAPE     : NaN%" in text
    assert "MASE     : NaN" in text
    assert "(no forecast rows)" in text


def test_inspect_report_to_dict_round_trip() -> None:
    """The JSON envelope strips non-finite numbers to ``None`` so jq is happy."""
    report = InspectReport(
        query_id="m",
        group="g",
        model="X",
        series_labels={"instance": "h"},
        step_seconds=60,
        season_length=1,
        horizon_steps=1,
        input_stats=InputStats(
            point_count=1,
            gap_count=0,
            largest_gap_seconds=0.0,
            lookback_start="t0",
            lookback_end="t1",
        ),
        forecast_timestamps=["t2"],
        forecast_yhat=[1.0],
        forecast_bands={"95": {"lower": [0.5], "upper": [1.5]}},
        preprocessing={},
        backtest=BacktestSummary(
            model="X",
            n_holdout=1,
            mape=math.nan,
            mase=2.0,
            rows=[
                BacktestFoldRow(
                    ds="t2",
                    actual=math.nan,
                    yhat=1.0,
                    bands={"95": {"lower": 0.5, "upper": 1.5}},
                )
            ],
        ),
    )
    payload = report.to_dict()
    assert payload["query_id"] == "m"
    assert payload["forecast"]["bands"]["95"]["lower"] == [0.5]
    assert payload["backtest"]["mape"] is None  # NaN -> None
    assert payload["backtest"]["mase"] == 2.0
    assert payload["backtest"]["rows"][0]["actual"] is None


def test_inspector_applies_count_aware_safeguards(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A ``data_profile: count`` query must clip negatives in inspector output.

    Guards the runner-parity contract spelled out in
    ``docs/operations/inspect.md``: the inspector exists to debug "what
    the live forecaster would do right now", so count-aware safeguards
    (``clip_non_negative``, ``round_to_integer``) MUST be applied
    before the inspector report renders. Without this, an operator
    inspecting a count-shaped query would see raw model output
    (possibly negative or fractional) that the live exporter would
    have rejected.
    """
    from datetime import UTC  # noqa: PLC0415
    from datetime import datetime as _dt  # noqa: PLC0415
    from datetime import timedelta as _td  # noqa: PLC0415

    import pandas as pd  # noqa: PLC0415

    from promforecast import inspect as inspect_module  # noqa: PLC0415
    from promforecast import models as model_registry  # noqa: PLC0415
    from promforecast.source import PromSource, SeriesFrame  # noqa: PLC0415

    class _NegativeModel:
        name = "AutoARIMA"

        def fit_predict(
            self,
            df: pd.DataFrame,
            horizon_steps: int,
            freq: str,
            levels: list[int],
            regressors: pd.DataFrame | None,
        ) -> pd.DataFrame:
            start_ts = df["ds"].iloc[-1] + pd.Timedelta(freq)
            ts = pd.date_range(start=start_ts, periods=horizon_steps, freq=freq)
            out: dict[str, object] = {"ds": ts, "yhat": [-5.0] * horizon_steps}
            for level in levels:
                out[f"yhat_lower_{level}"] = [-10.0] * horizon_steps
                out[f"yhat_upper_{level}"] = [0.5] * horizon_steps
            return pd.DataFrame(out)

    class _FakeInspectSource(PromSource):
        def __init__(self) -> None:
            now = _dt.now(tz=UTC)
            step = _td(minutes=1)
            self._frames = [
                SeriesFrame(
                    labels={"instance": "host-a"},
                    timestamps=[now - step * (60 - i) for i in range(60)],
                    values=[float(i) for i in range(60)],
                )
            ]

        async def query_range(
            self,
            promql: str,
            start: _dt,
            end: _dt,
            step_seconds: int,
        ) -> list[SeriesFrame]:
            return list(self._frames)

        async def aclose(self) -> None:
            return None

    monkeypatch.setattr(
        model_registry,
        "build",
        lambda name, season_length, params=None: _NegativeModel(),
    )
    monkeypatch.setattr(inspect_module, "PromSource", lambda **_kw: _FakeInspectSource())

    cfg_body = """
apiVersion: promforecast.io/v1
datasource:
  url: http://127.0.0.1:1
defaults:
  min_points: 10
  models: [AutoARIMA]
  confidence_levels: [80]
  step: 1m
  lookback: 1h
  horizon: 5m
  accuracy:
    evaluate: false
    auto_select: false
    horizons: [5m]
  emission:
    deviation: false
    quality: false
    contribution: false
groups:
  - name: g
    queries:
      - id: q
        promql: up
        data_profile: count
"""
    path = tmp_path / "cfg.yaml"
    path.write_text(cfg_body)
    out = io.StringIO()
    rc = run_inspect(
        config_path=path,
        query_id="q",
        series_filter=None,
        horizon_seconds=None,
        lookback_seconds=None,
        model=None,
        backtest=False,
        json_output=True,
        out=out,
    )
    assert rc == 0, out.getvalue()
    import json  # noqa: PLC0415

    payload = json.loads(out.getvalue())
    # ``data_profile: count`` implies ``clip_non_negative`` — every
    # forecast yhat must be >= 0, even though the model returned -5.
    assert all(v >= 0.0 for v in payload["forecast"]["yhat"]), payload["forecast"]["yhat"]
    # Bands must clip too.
    assert all(v >= 0.0 for v in payload["forecast"]["bands"]["80"]["lower"])


def test_run_inspect_returns_2_on_missing_config() -> None:
    """A missing config file is a 2 (datasource/I/O error)."""
    rc = run_inspect(
        config_path=Path("/nonexistent/forecaster/config.yaml"),
        query_id="m",
        series_filter=None,
        horizon_seconds=None,
        lookback_seconds=None,
        model=None,
        backtest=False,
        json_output=False,
        out=io.StringIO(),
    )
    assert rc == 2


def test_run_inspect_returns_1_on_unknown_query(tmp_path: Path) -> None:
    """An unknown query id fails at find_query, returning exit 1."""
    cfg_body = """
apiVersion: promforecast.io/v1
datasource:
  url: http://127.0.0.1:1
groups:
  - name: g
    queries:
      - id: existing
        promql: up
"""
    path = tmp_path / "cfg.yaml"
    path.write_text(cfg_body)
    rc = run_inspect(
        config_path=path,
        query_id="missing",
        series_filter=None,
        horizon_seconds=None,
        lookback_seconds=None,
        model=None,
        backtest=False,
        json_output=False,
        out=io.StringIO(),
    )
    assert rc == 1
