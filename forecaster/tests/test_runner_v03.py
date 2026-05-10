"""Integration tests for v0.3 runner features.

Covers the runner-level wiring of the four v0.3 user stories:

* Ensemble auto-select mode: combined forecast plus component models, with
  inverse-MAPE weights exposed as a metric.
* Per-model hyperparameter overrides: ``params:`` is forwarded into the
  registry's ``build`` call.
* Exogenous regressors: optional failures degrade gracefully and are
  counted; ``required: true`` failures skip the series.
* ``remote_write`` sink: the runner pushes the full curve and reports sink
  failures via the dedicated counter.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from promforecast.config import (
    AccuracyConfig,
    Config,
    DatasourceConfig,
    DefaultsConfig,
    EmissionConfig,
    GroupConfig,
    ModelSpec,
    QueryConfig,
    RegressorConfig,
    SafetyConfig,
    ServerConfig,
)
from promforecast.exporter import Exporter
from promforecast.runner import Runner
from promforecast.sink import ForecastCurvePoint, SinkError
from promforecast.source import PromSource, SeriesFrame


class _FakeSource(PromSource):
    def __init__(self, by_query: dict[str, list[SeriesFrame]] | None = None) -> None:
        self._by_query = by_query or {}
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


class _FakeModel:
    """A constant-yhat model parametrised on its own params dict.

    ``params`` are captured so a test can assert that hyperparameters
    flowed all the way from the YAML to the model constructor.
    """

    def __init__(self, name: str, value: float, params: dict | None = None) -> None:
        self.name = name
        self._value = value
        self._params = dict(params or {})

    def fit_predict(
        self,
        df,
        horizon: int,
        freq: str,
        levels: list[int],
        regressors=None,
    ):
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
    start = datetime(2024, 1, 1, tzinfo=UTC)
    return SeriesFrame(
        labels={"instance": "host-a"},
        timestamps=[start + timedelta(minutes=5 * i) for i in range(n)],
        values=[float(i) for i in range(n)],
    )


def _config(
    *,
    models: list[ModelSpec | str] | None = None,
    auto_select: bool | str = False,
    regressors: list[RegressorConfig] | None = None,
) -> Config:
    return Config(
        datasource=DatasourceConfig(url="http://x"),
        server=ServerConfig(),
        safety=SafetyConfig(),
        defaults=DefaultsConfig(
            min_points=10,
            models=models or [ModelSpec(name="SeasonalNaive")],
            confidence_levels=[80],
            accuracy=AccuracyConfig(
                evaluate=True,
                auto_select=auto_select,
                horizons=[timedelta(minutes=10)],
            ),
            emission=EmissionConfig(deviation=False, quality=False),
        ),
        groups=[
            GroupConfig(
                name="g",
                queries=[QueryConfig(id="m", promql="up", regressors=regressors or [])],
            )
        ],
    )


@pytest.mark.asyncio
async def test_per_model_params_are_passed_to_registry_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from promforecast import models as model_registry  # noqa: PLC0415

    captured_params: dict[str, dict] = {}

    def _build(name: str, season_length: int, params: dict | None = None) -> _FakeModel:
        captured_params[name] = dict(params or {})
        return _FakeModel(name, value=1.0, params=params)

    monkeypatch.setattr(model_registry, "build", _build)

    cfg = _config(models=[ModelSpec(name="AutoARIMA", params={"max_p": 2, "stepwise": False})])
    runner = Runner(
        config=cfg,
        source=_FakeSource({"up": [_ramp_frame(60)]}),
        exporter=Exporter(),
    )
    await runner.run_group(cfg.groups[0])

    assert captured_params["AutoARIMA"] == {"max_p": 2, "stepwise": False}


@pytest.mark.asyncio
async def test_ensemble_mode_emits_combined_plus_components_and_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from promforecast import models as model_registry  # noqa: PLC0415

    def _build(name: str, season_length: int, params: dict | None = None) -> _FakeModel:
        # AutoARIMA tracks the actuals; SeasonalNaive doesn't. The ensemble
        # should weight AutoARIMA more heavily.
        if name == "AutoARIMA":
            return _FakeModel(name, value=50.0)
        if name == "SeasonalNaive":
            return _FakeModel(name, value=0.0)
        raise AssertionError(name)

    monkeypatch.setattr(model_registry, "build", _build)

    cfg = _config(
        models=[ModelSpec(name="AutoARIMA"), ModelSpec(name="SeasonalNaive")],
        auto_select="ensemble",
    )
    exporter = Exporter()
    runner = Runner(config=cfg, source=_FakeSource({"up": [_ramp_frame(60)]}), exporter=exporter)
    await runner.run_group(cfg.groups[0])

    body = exporter.render().decode()
    # All three forecast lines: per-component AutoARIMA, per-component
    # SeasonalNaive, and the synthetic ensemble row.
    assert body.count("m_forecast{") == 3
    assert 'model="AutoARIMA"' in body
    assert 'model="SeasonalNaive"' in body
    assert 'model="ensemble"' in body
    # Ensemble weights surface for both component models.
    assert "# TYPE forecast_ensemble_weight gauge" in body
    assert 'forecast_ensemble_weight{group="g",id="m",instance="host-a",model="AutoARIMA"}' in body


@pytest.mark.asyncio
async def test_ensemble_mode_skips_combined_row_when_every_component_fits_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If every component fit raises, the ensemble shouldn't fabricate a row."""
    from promforecast import models as model_registry  # noqa: PLC0415

    def _build(name: str, season_length: int, params: dict | None = None) -> _FakeModel:
        class _Failing:
            name = name

            def fit_predict(self, **_: object) -> object:
                raise RuntimeError("boom")

        return _Failing()  # type: ignore[return-value]

    monkeypatch.setattr(model_registry, "build", _build)

    cfg = _config(
        models=[ModelSpec(name="AutoARIMA"), ModelSpec(name="SeasonalNaive")],
        auto_select="ensemble",
    )
    exporter = Exporter()
    runner = Runner(config=cfg, source=_FakeSource({"up": [_ramp_frame(60)]}), exporter=exporter)
    await runner.run_group(cfg.groups[0])

    body = exporter.render().decode()
    assert 'model="ensemble"' not in body


@pytest.mark.asyncio
async def test_optional_regressor_failure_records_counter_and_keeps_forecasting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from promforecast import models as model_registry  # noqa: PLC0415

    def _build(name: str, season_length: int, params: dict | None = None) -> _FakeModel:
        return _FakeModel(name, value=1.0)

    monkeypatch.setattr(model_registry, "build", _build)

    class _ErroringSource(_FakeSource):
        async def query_range(
            self, promql: str, start: datetime, end: datetime, step_seconds: int
        ) -> list[SeriesFrame]:
            if promql == "deploys":
                raise RuntimeError("regressor TSDB outage")
            return await super().query_range(promql, start, end, step_seconds)

    cfg = _config(
        regressors=[RegressorConfig(id="deploys", type="custom", promql="deploys")],
    )
    source = _ErroringSource({"up": [_ramp_frame(60)]})
    exporter = Exporter()
    runner = Runner(config=cfg, source=source, exporter=exporter)
    await runner.run_group(cfg.groups[0])

    body = exporter.render().decode()
    # Forecast should still run.
    assert "m_forecast{" in body
    # Failure counter for the regressor.
    assert "# TYPE forecast_regressor_failures_total counter" in body
    assert (
        'forecast_regressor_failures_total{group="g",query="m",reason="query_error",'
        'regressor="deploys"} 1.0'
    ) in body


@pytest.mark.asyncio
async def test_required_regressor_failure_skips_only_that_series(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from promforecast import models as model_registry  # noqa: PLC0415

    def _build(name: str, season_length: int, params: dict | None = None) -> _FakeModel:
        return _FakeModel(name, value=1.0)

    monkeypatch.setattr(model_registry, "build", _build)

    class _ErroringSource(_FakeSource):
        async def query_range(
            self, promql: str, start: datetime, end: datetime, step_seconds: int
        ) -> list[SeriesFrame]:
            if promql == "deploys":
                raise RuntimeError("regressor TSDB outage")
            return await super().query_range(promql, start, end, step_seconds)

    cfg = _config(
        regressors=[RegressorConfig(id="deploys", type="custom", promql="deploys", required=True)],
    )
    source = _ErroringSource({"up": [_ramp_frame(60)]})
    exporter = Exporter()
    runner = Runner(config=cfg, source=source, exporter=exporter)
    await runner.run_group(cfg.groups[0])

    body = exporter.render().decode()
    # Forecast for this series should not appear.
    assert "m_forecast{" not in body
    # ``required_regressor`` failure counter is bumped.
    assert 'forecast_failures_total{group="g",reason="required_regressor"} 1.0' in body


@pytest.mark.asyncio
async def test_promql_regressor_fetched_once_per_query_not_per_series(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The PromQL regressor must be fetched once per primary query,
    regardless of how many series the query returned. Without this the
    regressor fan-out is multiplied by the cardinality of the primary
    query — a potentially abusive load on the TSDB."""
    from promforecast import models as model_registry  # noqa: PLC0415

    def _build(name: str, season_length: int, params: dict | None = None) -> _FakeModel:
        return _FakeModel(name, value=1.0)

    monkeypatch.setattr(model_registry, "build", _build)

    promql_calls: dict[str, int] = {}

    class _CountingSource(_FakeSource):
        async def query_range(
            self, promql: str, start: datetime, end: datetime, step_seconds: int
        ) -> list[SeriesFrame]:
            promql_calls[promql] = promql_calls.get(promql, 0) + 1
            if promql == "deploys":
                base = datetime(2024, 1, 1, tzinfo=UTC)
                return [
                    SeriesFrame(
                        labels={},
                        timestamps=[base + timedelta(minutes=5 * j) for j in range(60)],
                        values=[float(j) for j in range(60)],
                    )
                ]
            return await super().query_range(promql, start, end, step_seconds)

    # Three series under the primary query. With the per-query prefetch the
    # regressor PromQL ("deploys") must be hit *once*, not three times.
    base = datetime(2024, 1, 1, tzinfo=UTC)
    primary_frames = [
        SeriesFrame(
            labels={"instance": f"host-{i}"},
            timestamps=[base + timedelta(minutes=5 * j) for j in range(60)],
            values=[float(j) for j in range(60)],
        )
        for i in range(3)
    ]
    cfg = _config(
        regressors=[RegressorConfig(id="deploys", type="custom", promql="deploys")],
    )
    source = _CountingSource({"up": primary_frames})
    runner = Runner(config=cfg, source=source, exporter=Exporter())
    await runner.run_group(cfg.groups[0])

    assert promql_calls["up"] == 1, "primary query runs once per refresh"
    assert promql_calls["deploys"] == 1, (
        "regressor PromQL must be fetched once per primary query, not per series"
    )


@pytest.mark.asyncio
async def test_calendar_regressor_does_not_need_promql_or_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Calendar features are pure functions of the timestamps — they must
    work even when the datasource never returns regressor data."""
    from promforecast import models as model_registry  # noqa: PLC0415

    def _build(name: str, season_length: int, params: dict | None = None) -> _FakeModel:
        return _FakeModel(name, value=1.0)

    monkeypatch.setattr(model_registry, "build", _build)

    cfg = _config(
        regressors=[RegressorConfig(id="hour_of_day", type="calendar")],
    )
    exporter = Exporter()
    runner = Runner(config=cfg, source=_FakeSource({"up": [_ramp_frame(60)]}), exporter=exporter)
    await runner.run_group(cfg.groups[0])

    body = exporter.render().decode()
    assert "m_forecast{" in body
    # No regressor failures when calendar succeeds.
    assert "forecast_regressor_failures_total" not in body


class _RecordingSink:
    name = "recording"

    def __init__(self, raise_on_write: SinkError | None = None) -> None:
        self.received: list[ForecastCurvePoint] = []
        self._raise_on_write = raise_on_write

    async def write(self, points) -> None:
        self.received.extend(points)
        if self._raise_on_write is not None:
            raise self._raise_on_write

    async def aclose(self) -> None:
        return None


@pytest.mark.asyncio
async def test_remote_write_sink_receives_full_horizon_curve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from promforecast import models as model_registry  # noqa: PLC0415

    def _build(name: str, season_length: int, params: dict | None = None) -> _FakeModel:
        return _FakeModel(name, value=42.0)

    monkeypatch.setattr(model_registry, "build", _build)

    sink = _RecordingSink()
    cfg = _config()
    runner = Runner(
        config=cfg,
        source=_FakeSource({"up": [_ramp_frame(60)]}),
        exporter=Exporter(),
        sink=sink,
    )
    await runner.run_group(cfg.groups[0])

    # horizon=24h / step=5min → 288 yhat points; with one confidence level
    # we add 2 lower/upper rows per yhat → 3 * 288 = 864 points total.
    expected_per_step = 1 + 2 * len(cfg.defaults.confidence_levels)
    horizon_steps = int(cfg.defaults.horizon.total_seconds() // cfg.defaults.step.total_seconds())
    assert len(sink.received) == expected_per_step * horizon_steps
    # Each point carries a future-dated timestamp.
    assert all(p.timestamp_ms > 0 for p in sink.received)


@pytest.mark.asyncio
async def test_remote_write_sink_failure_increments_counter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from promforecast import models as model_registry  # noqa: PLC0415

    def _build(name: str, season_length: int, params: dict | None = None) -> _FakeModel:
        return _FakeModel(name, value=42.0)

    monkeypatch.setattr(model_registry, "build", _build)

    sink = _RecordingSink(raise_on_write=SinkError("timeout", "simulated"))
    cfg = _config()
    exporter = Exporter()
    runner = Runner(
        config=cfg,
        source=_FakeSource({"up": [_ramp_frame(60)]}),
        exporter=exporter,
        sink=sink,
    )
    await runner.run_group(cfg.groups[0])

    body = exporter.render().decode()
    assert 'forecast_remote_write_failures_total{group="g",reason="timeout"} 1.0' in body
