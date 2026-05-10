"""Integration tests for Runner orchestration with a fake source.

The fit path (statsforecast) needs heavy deps and is exercised end-to-end via
docker-compose. These tests cover the orchestration layer only: query
dispatch, the cap pipeline, the min_points gate, failure isolation, and the
snapshot update.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from promforecast.config import (
    AccuracyConfig,
    Config,
    DatasourceConfig,
    DefaultsConfig,
    EmissionConfig,
    GroupConfig,
    QueryConfig,
    SafetyConfig,
    ServerConfig,
)
from promforecast.exporter import Exporter, ForecastPoint
from promforecast.runner import Runner
from promforecast.source import PromSource, SeriesFrame

if TYPE_CHECKING:
    pass


class _FakeSource(PromSource):
    """Returns canned frames; never hits the network."""

    def __init__(self, by_query: dict[str, list[SeriesFrame]]) -> None:
        self._by_query = by_query
        self.calls: list[str] = []

    async def query_range(
        self,
        promql: str,
        start: datetime,
        end: datetime,
        step_seconds: int,
    ) -> list[SeriesFrame]:
        self.calls.append(promql)
        return self._by_query.get(promql, [])

    async def aclose(self) -> None:
        return None


def _config(
    queries: list[QueryConfig],
    *,
    min_points: int = 10,
    max_series_per_query: int = 500,
    max_total_series: int = 0,
) -> Config:
    return Config(
        datasource=DatasourceConfig(url="http://x"),
        server=ServerConfig(),
        safety=SafetyConfig(
            max_series_per_query=max_series_per_query,
            max_total_series=max_total_series,
        ),
        defaults=DefaultsConfig(min_points=min_points, models=["SeasonalNaive"]),
        groups=[GroupConfig(name="g", queries=queries)],
    )


def _frame(label_value: str, n_points: int) -> SeriesFrame:
    return SeriesFrame(
        labels={"instance": label_value},
        timestamps=[datetime.now(tz=UTC)] * n_points,
        values=[1.0] * n_points,
    )


@pytest.mark.asyncio
async def test_run_group_records_min_points_drops() -> None:
    cfg = _config(
        [QueryConfig(id="m", promql="up")],
        min_points=100,
    )
    source = _FakeSource({"up": [_frame("a", n_points=5), _frame("b", n_points=5)]})
    exporter = Exporter()
    runner = Runner(config=cfg, source=source, exporter=exporter)

    await runner.run_group(cfg.groups[0])

    body = exporter.render().decode()
    assert 'forecast_series_dropped_total{group="g",reason="min_points"} 2.0' in body
    assert 'forecast_series_count{group="g"} 2.0' in body


@pytest.mark.asyncio
async def test_run_group_isolates_query_errors() -> None:
    class _ErroringSource(_FakeSource):
        async def query_range(
            self, promql: str, start: datetime, end: datetime, step_seconds: int
        ) -> list[SeriesFrame]:
            if promql == "bad":
                raise RuntimeError("simulated TSDB outage")
            return await super().query_range(promql, start, end, step_seconds)

    cfg = _config(
        [
            QueryConfig(id="ok", promql="up"),
            QueryConfig(id="bd", promql="bad"),
        ],
        min_points=100,
    )
    source = _ErroringSource({"up": [_frame("a", n_points=5)]})
    exporter = Exporter()
    runner = Runner(config=cfg, source=source, exporter=exporter)

    await runner.run_group(cfg.groups[0])

    body = exporter.render().decode()
    assert 'forecast_failures_total{group="g",reason="query_error"} 1.0' in body


@pytest.mark.asyncio
async def test_run_group_full_snapshot_with_canned_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _config(
        [QueryConfig(id="m", promql="up")],
        min_points=1,
    )
    source = _FakeSource({"up": [_frame("host-a", n_points=10)]})
    exporter = Exporter()
    runner = Runner(config=cfg, source=source, exporter=exporter)

    async def _canned(self, query, frame, group_name, prefetch):  # type: ignore[no-untyped-def]
        from promforecast.runner import _FitResult  # noqa: PLC0415

        points = [
            ForecastPoint(
                metric="m_forecast",
                labels={**frame.labels, "group": group_name, "model": "Fake"},
                value=42.0,
            )
        ]
        return _FitResult(points=points)

    monkeypatch.setattr(Runner, "_fit_series", _canned)
    await runner.run_group(cfg.groups[0])

    body = exporter.render().decode()
    assert "m_forecast{" in body
    assert 'instance="host-a"' in body
    assert 'forecast_series_count{group="g"} 1.0' in body


class _FakeModel:
    """ForecastModel that emits a constant ``yhat``. Used in tests instead of
    statsforecast — keeps the runner tests isolated from heavy fit deps."""

    def __init__(self, name: str, value: float, season_length: int = 288) -> None:
        self.name = name
        self._value = value

    def fit_predict(
        self,
        df,  # type: ignore[no-untyped-def]
        horizon: int,
        freq: str,
        levels: list[int],
        regressors=None,  # type: ignore[no-untyped-def]
    ):  # type: ignore[no-untyped-def]
        import pandas as pd  # noqa: PLC0415

        idx = pd.date_range(df["ds"].iloc[-1], periods=horizon + 1, freq=freq)[1:]
        out = pd.DataFrame(
            {
                "unique_id": ["s"] * horizon,
                "ds": idx,
                "yhat": [self._value] * horizon,
            }
        )
        for level in levels:
            out[f"yhat_lower_{level}"] = self._value - 1.0
            out[f"yhat_upper_{level}"] = self._value + 1.0
        return out


def _ramp_frame(n: int) -> SeriesFrame:
    from datetime import timedelta as _td  # noqa: PLC0415

    start = datetime(2024, 1, 1, tzinfo=UTC)
    timestamps = [start + _td(minutes=5 * i) for i in range(n)]
    values = [float(i) for i in range(n)]
    return SeriesFrame(labels={"instance": "host-a"}, timestamps=timestamps, values=values)


@pytest.mark.asyncio
async def test_run_group_auto_select_emits_winner_with_best_model_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from promforecast import models as model_registry  # noqa: PLC0415

    cfg = Config(
        datasource=DatasourceConfig(url="http://x"),
        server=ServerConfig(),
        safety=SafetyConfig(),
        defaults=DefaultsConfig(
            min_points=10,
            models=["AutoARIMA", "SeasonalNaive"],
            confidence_levels=[80],
            accuracy=AccuracyConfig(
                evaluate=True,
                auto_select=True,
                fallback_mape_threshold=None,
            ),
        ),
        groups=[GroupConfig(name="g", queries=[QueryConfig(id="m", promql="up")])],
    )

    # AutoARIMA tracks the actuals exactly → MAPE near 0; SeasonalNaive
    # predicts 0 → much higher MAPE. Auto-select must pick AutoARIMA.
    def _fake_build(name: str, season_length: int, params: dict | None = None) -> _FakeModel:
        if name == "AutoARIMA":
            return _FakeModel("AutoARIMA", value=50.0)
        if name == "SeasonalNaive":
            return _FakeModel("SeasonalNaive", value=0.0)
        raise AssertionError(name)

    monkeypatch.setattr(model_registry, "build", _fake_build)

    # Series of [0..59], hold-out is last 288 (24h) — but len < 288, so we
    # use a short accuracy horizon to make backtest viable.
    cfg.defaults.accuracy.horizons = [timedelta(minutes=10)]
    source = _FakeSource({"up": [_ramp_frame(60)]})
    exporter = Exporter()
    runner = Runner(config=cfg, source=source, exporter=exporter)

    await runner.run_group(cfg.groups[0])

    body = exporter.render().decode()
    # Only one model's forecast should appear.
    assert body.count("m_forecast{") == 1
    assert 'best_model="AutoARIMA"' in body
    assert 'model="AutoARIMA"' in body
    assert 'model_fallback="false"' in body
    # Backtest also produced accuracy metrics for both models.
    assert "forecast_accuracy_mape" in body
    assert 'model="SeasonalNaive"' in body  # in accuracy metrics
    assert 'horizon="10m"' in body


@pytest.mark.asyncio
async def test_run_group_auto_select_falls_back_when_winner_exceeds_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from promforecast import models as model_registry  # noqa: PLC0415

    cfg = Config(
        datasource=DatasourceConfig(url="http://x"),
        server=ServerConfig(),
        safety=SafetyConfig(),
        defaults=DefaultsConfig(
            min_points=10,
            models=["AutoARIMA"],
            confidence_levels=[80],
            accuracy=AccuracyConfig(
                evaluate=True,
                auto_select=True,
                fallback_model="SeasonalNaive",
                fallback_mape_threshold=10.0,
                horizons=[timedelta(minutes=10)],
            ),
        ),
        groups=[GroupConfig(name="g", queries=[QueryConfig(id="m", promql="up")])],
    )

    def _fake_build(name: str, season_length: int, params: dict | None = None) -> _FakeModel:
        if name == "AutoARIMA":
            # Predict 999 against actuals near 50 → MAPE > threshold.
            return _FakeModel("AutoARIMA", value=999.0)
        if name == "SeasonalNaive":
            return _FakeModel("SeasonalNaive", value=42.0)
        raise AssertionError(name)

    monkeypatch.setattr(model_registry, "build", _fake_build)

    source = _FakeSource({"up": [_ramp_frame(60)]})
    exporter = Exporter()
    runner = Runner(config=cfg, source=source, exporter=exporter)

    await runner.run_group(cfg.groups[0])

    body = exporter.render().decode()
    assert 'model="SeasonalNaive"' in body  # forecast
    assert 'best_model="AutoARIMA"' in body  # what won the (bad) horse race
    assert 'model_fallback="true"' in body


@pytest.mark.asyncio
async def test_run_group_explicit_mode_emits_one_per_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from promforecast import models as model_registry  # noqa: PLC0415

    cfg = Config(
        datasource=DatasourceConfig(url="http://x"),
        server=ServerConfig(),
        safety=SafetyConfig(),
        defaults=DefaultsConfig(
            min_points=10,
            models=["AutoARIMA", "SeasonalNaive"],
            confidence_levels=[80],
            accuracy=AccuracyConfig(
                evaluate=False,
                auto_select=False,
                horizons=[timedelta(minutes=10)],
            ),
        ),
        groups=[GroupConfig(name="g", queries=[QueryConfig(id="m", promql="up")])],
    )

    def _fake_build(name: str, season_length: int, params: dict | None = None) -> _FakeModel:
        return _FakeModel(name, value=1.0 if name == "AutoARIMA" else 2.0)

    monkeypatch.setattr(model_registry, "build", _fake_build)

    source = _FakeSource({"up": [_ramp_frame(60)]})
    exporter = Exporter()
    runner = Runner(config=cfg, source=source, exporter=exporter)

    await runner.run_group(cfg.groups[0])

    body = exporter.render().decode()
    # Two forecast rows — one per model.
    assert body.count("m_forecast{") == 2
    assert 'model="AutoARIMA"' in body
    assert 'model="SeasonalNaive"' in body
    # No best_model label in explicit mode.
    assert "best_model=" not in body
    # No accuracy metrics when evaluate=false.
    assert "forecast_accuracy_mape" not in body


@pytest.mark.asyncio
async def test_run_group_skips_quality_for_unbacktested_fallback_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When auto-select falls back to a model that wasn't in ``models``, the
    fallback model has no backtest data. Quality is undefined for that
    series/model pair and we drop the sample rather than emit zero —
    otherwise ``ForecastQualityZero`` alerts would fire for every fallback.
    """
    from promforecast import models as model_registry  # noqa: PLC0415

    cfg = Config(
        datasource=DatasourceConfig(url="http://x"),
        server=ServerConfig(),
        safety=SafetyConfig(),
        defaults=DefaultsConfig(
            min_points=10,
            # Fallback (SeasonalNaive) deliberately not in ``models`` so it
            # never gets backtested.
            models=["AutoARIMA"],
            confidence_levels=[80],
            accuracy=AccuracyConfig(
                evaluate=True,
                auto_select=True,
                fallback_model="SeasonalNaive",
                fallback_mape_threshold=10.0,
                horizons=[timedelta(minutes=10)],
            ),
            emission=EmissionConfig(quality=True, deviation=True),
        ),
        groups=[GroupConfig(name="g", queries=[QueryConfig(id="m", promql="up")])],
    )

    def _fake_build(name: str, season_length: int, params: dict | None = None) -> _FakeModel:
        if name == "AutoARIMA":
            return _FakeModel("AutoARIMA", value=999.0)  # bad MAPE → fallback
        if name == "SeasonalNaive":
            return _FakeModel("SeasonalNaive", value=42.0)
        raise AssertionError(name)

    monkeypatch.setattr(model_registry, "build", _fake_build)

    source = _FakeSource({"up": [_ramp_frame(60)]})
    exporter = Exporter()
    runner = Runner(config=cfg, source=source, exporter=exporter)

    await runner.run_group(cfg.groups[0])

    body = exporter.render().decode()
    assert 'model_fallback="true"' in body
    # No quality_score sample for the fallback'd (SeasonalNaive) emission —
    # the family may exist as a header from other groups, but no SeasonalNaive
    # row should appear here.
    assert "forecast_quality_score{" not in body
    # And no deviation sample either — no backtest data for the fallback.
    assert "forecast_deviation_ratio{" not in body
    assert "forecast_deviation_outside_band{" not in body


@pytest.mark.asyncio
async def test_run_group_emits_deviation_and_quality_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from promforecast import models as model_registry  # noqa: PLC0415

    cfg = Config(
        datasource=DatasourceConfig(url="http://x"),
        server=ServerConfig(),
        safety=SafetyConfig(),
        defaults=DefaultsConfig(
            min_points=10,
            models=["SeasonalNaive"],
            confidence_levels=[80],
            accuracy=AccuracyConfig(
                evaluate=True,
                auto_select=False,
                horizons=[timedelta(minutes=10)],
            ),
            emission=EmissionConfig(deviation=True, quality=True),
        ),
        groups=[GroupConfig(name="g", queries=[QueryConfig(id="m", promql="up")])],
    )

    def _build(name: str, season_length: int, params: dict | None = None) -> _FakeModel:
        return _FakeModel(name, value=42.0)

    monkeypatch.setattr(model_registry, "build", _build)

    source = _FakeSource({"up": [_ramp_frame(60)]})
    exporter = Exporter()
    runner = Runner(config=cfg, source=source, exporter=exporter)

    await runner.run_group(cfg.groups[0])

    body = exporter.render().decode()
    assert "# TYPE forecast_deviation_ratio gauge" in body
    assert "# TYPE forecast_deviation_outside_band gauge" in body
    assert "# TYPE forecast_quality_score gauge" in body
    assert 'level="80"' in body  # deviation carries level
    assert 'id="m"' in body
    assert 'group="g"' in body
    assert 'model="SeasonalNaive"' in body


@pytest.mark.asyncio
async def test_run_group_emit_deviation_and_quality_disabled_suppresses_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from promforecast import models as model_registry  # noqa: PLC0415

    cfg = Config(
        datasource=DatasourceConfig(url="http://x"),
        server=ServerConfig(),
        safety=SafetyConfig(),
        defaults=DefaultsConfig(
            min_points=10,
            models=["SeasonalNaive"],
            confidence_levels=[80],
            accuracy=AccuracyConfig(
                evaluate=False,
                auto_select=False,
                horizons=[timedelta(minutes=10)],
            ),
            emission=EmissionConfig(deviation=False, quality=False),
        ),
        groups=[GroupConfig(name="g", queries=[QueryConfig(id="m", promql="up")])],
    )

    def _build(name: str, season_length: int, params: dict | None = None) -> _FakeModel:
        return _FakeModel(name, value=1.0)

    monkeypatch.setattr(model_registry, "build", _build)

    source = _FakeSource({"up": [_ramp_frame(60)]})
    exporter = Exporter()
    runner = Runner(config=cfg, source=source, exporter=exporter)

    await runner.run_group(cfg.groups[0])

    body = exporter.render().decode()
    assert "forecast_deviation_ratio" not in body
    assert "forecast_deviation_outside_band" not in body
    assert "forecast_quality_score" not in body


@pytest.mark.asyncio
async def test_backtests_skip_summary_when_only_accuracy_consumer_is_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When only accuracy emission is enabled, ``_run_backtests`` must not
    pay the cost of summarising the trailing band — only deviation and
    quality need that step. This locks in the needs-bitmap behaviour."""
    from promforecast import models as model_registry  # noqa: PLC0415
    from promforecast import runner as runner_module  # noqa: PLC0415

    cfg = Config(
        datasource=DatasourceConfig(url="http://x"),
        server=ServerConfig(),
        safety=SafetyConfig(),
        defaults=DefaultsConfig(
            min_points=10,
            models=["SeasonalNaive"],
            confidence_levels=[80],
            accuracy=AccuracyConfig(
                evaluate=True,
                auto_select=False,
                horizons=[timedelta(minutes=10)],
            ),
            emission=EmissionConfig(deviation=False, quality=False),
        ),
        groups=[GroupConfig(name="g", queries=[QueryConfig(id="m", promql="up")])],
    )

    def _build(name: str, season_length: int, params: dict | None = None) -> _FakeModel:
        return _FakeModel(name, value=1.0)

    monkeypatch.setattr(model_registry, "build", _build)

    summary_calls: list[None] = []
    real_summary = runner_module._summarise_backtest

    def _spy_summary(**kwargs: object) -> object:
        summary_calls.append(None)
        return real_summary(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(runner_module, "_summarise_backtest", _spy_summary)

    source = _FakeSource({"up": [_ramp_frame(60)]})
    runner = Runner(config=cfg, source=source, exporter=Exporter())
    await runner.run_group(cfg.groups[0])

    assert summary_calls == [], "trailing-band summary should be skipped when no consumer wants it"


@pytest.mark.asyncio
async def test_run_group_global_cap_truncates_across_queries() -> None:
    cfg = _config(
        [
            QueryConfig(id="a", promql="qa"),
            QueryConfig(id="b", promql="qb"),
        ],
        min_points=1,
        max_series_per_query=100,
        max_total_series=1,
    )
    source = _FakeSource(
        {
            "qa": [_frame("a", n_points=5)],
            "qb": [_frame("b", n_points=5)],
        }
    )
    exporter = Exporter()
    runner = Runner(config=cfg, source=source, exporter=exporter)

    await runner.run_group(cfg.groups[0])

    body = exporter.render().decode()
    assert 'forecast_series_count{group="g"} 1.0' in body
    assert 'reason="total_cap"' in body
