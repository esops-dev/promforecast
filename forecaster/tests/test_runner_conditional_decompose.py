"""Runner-level integration tests for the conditional + decomposition families."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from promforecast.config import (
    AccuracyConfig,
    Config,
    DatasourceConfig,
    DefaultsConfig,
    GroupConfig,
    QueryConfig,
    QueryEmissionConfig,
    SafetyConfig,
    ServerConfig,
)
from promforecast.exporter import Exporter
from promforecast.runner import Runner
from promforecast.source import PromSource, SeriesFrame


class _FakeSource(PromSource):
    def __init__(self, by_query: dict[str, list[SeriesFrame]]) -> None:
        self._by_query = by_query

    async def query_range(
        self,
        promql: str,
        start: datetime,
        end: datetime,
        step_seconds: int,
    ) -> list[SeriesFrame]:
        return self._by_query.get(promql, [])

    async def aclose(self) -> None:
        return None


class _RampForecastModel:
    """Stable ramp model; ignores regressors so conditional == unconditional."""

    def __init__(
        self,
        season_length: int = 288,
        *,
        start: float = 50.0,
        per_step: float = 1.0,
    ) -> None:
        self.name = "RampModel"
        self._start = start
        self._per_step = per_step

    def fit_predict(
        self,
        df: pd.DataFrame,
        horizon: int,
        freq: str,
        levels: list[int],
        regressors: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        idx = pd.date_range(df["ds"].iloc[-1], periods=horizon + 1, freq=freq)[1:]
        values = [self._start + i * self._per_step for i in range(horizon)]
        out = pd.DataFrame({"unique_id": ["s"] * horizon, "ds": idx, "yhat": values})
        for level in levels:
            out[f"yhat_lower_{level}"] = [v - 1.0 for v in values]
            out[f"yhat_upper_{level}"] = [v + 1.0 for v in values]
        return out


def _frame(n: int) -> SeriesFrame:
    base = datetime(2026, 5, 21, tzinfo=UTC)
    timestamps = [base + timedelta(minutes=5 * i) for i in range(n)]
    return SeriesFrame(labels={"instance": "host-a"}, timestamps=timestamps, values=[1.0] * n)


def _config_with_query(query: QueryConfig) -> Config:
    return Config(
        datasource=DatasourceConfig(url="http://x"),
        server=ServerConfig(),
        safety=SafetyConfig(),
        defaults=DefaultsConfig(
            min_points=5,
            models=["RampModel"],
            confidence_levels=[80],
            horizon=timedelta(hours=2),
            step=timedelta(minutes=5),
            accuracy=AccuracyConfig(evaluate=False),
        ),
        groups=[GroupConfig(name="g", queries=[query])],
    )


@pytest.mark.asyncio
async def test_conditional_emission_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """No conditional metrics until ``emission.conditional`` is flipped on."""
    from promforecast import models as model_registry  # noqa: PLC0415

    monkeypatch.setattr(model_registry, "build", lambda *a, **kw: _RampForecastModel())

    cfg = _config_with_query(QueryConfig(id="m", promql="up"))
    source = _FakeSource({"up": [_frame(20)]})
    runner = Runner(config=cfg, source=source, exporter=(exporter := Exporter()))
    await runner.run_group(cfg.groups[0])

    body = exporter.render().decode()
    assert "m_forecast_conditional" not in body
    assert "forecast_conditional_deviation_outside_band" not in body


@pytest.mark.asyncio
async def test_conditional_emission_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Enabling the toggle adds the parallel conditional family + deviation."""
    from promforecast import models as model_registry  # noqa: PLC0415

    monkeypatch.setattr(model_registry, "build", lambda *a, **kw: _RampForecastModel())

    cfg = _config_with_query(
        QueryConfig(
            id="http_requests_rate",
            promql="up",
            emission=QueryEmissionConfig(conditional=True),
        )
    )
    source = _FakeSource({"up": [_frame(20)]})
    runner = Runner(config=cfg, source=source, exporter=(exporter := Exporter()))
    await runner.run_group(cfg.groups[0])

    body = exporter.render().decode()
    # Conditional snapshot family is emitted (one row per (model, level) for
    # upper / lower bands plus the central point).
    assert "http_requests_rate_forecast_conditional" in body
    assert "http_requests_rate_forecast_conditional_lower" in body
    assert "http_requests_rate_forecast_conditional_upper" in body
    # Conditional deviation operates against the trailing actual; the
    # outside-band gauge is emitted regardless of the ratio's finiteness.
    assert "forecast_conditional_deviation_outside_band" in body


@pytest.mark.asyncio
async def test_conditional_emission_applies_count_safeguards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Count-aware clipping must apply to the conditional family too.

    A query with ``data_profile: count`` would otherwise emit clipped
    unconditional values alongside un-clipped conditional values — the
    same operator alert evaluated against both would behave inconsistently.
    The model below emits negative band values; with safeguards on, the
    conditional output should never go below zero.
    """
    from promforecast import models as model_registry  # noqa: PLC0415

    class _NegativeBandModel(_RampForecastModel):
        name = "RampModel"

        def fit_predict(
            self,
            df: pd.DataFrame,
            horizon: int,
            freq: str,
            levels: list[int],
            regressors: pd.DataFrame | None = None,
        ) -> pd.DataFrame:
            idx = pd.date_range(df["ds"].iloc[-1], periods=horizon + 1, freq=freq)[1:]
            values = [0.5 + i * 0.1 for i in range(horizon)]
            out = pd.DataFrame({"unique_id": ["s"] * horizon, "ds": idx, "yhat": values})
            for level in levels:
                # Band is wide enough that the lower edge goes negative.
                out[f"yhat_lower_{level}"] = [v - 5.0 for v in values]
                out[f"yhat_upper_{level}"] = [v + 5.0 for v in values]
            return out

    monkeypatch.setattr(model_registry, "build", lambda *a, **kw: _NegativeBandModel())

    cfg = _config_with_query(
        QueryConfig(
            id="count_metric",
            promql="up",
            data_profile="count",
            emission=QueryEmissionConfig(conditional=True),
        )
    )
    source = _FakeSource({"up": [_frame(20)]})
    runner = Runner(config=cfg, source=source, exporter=(exporter := Exporter()))
    await runner.run_group(cfg.groups[0])

    body = exporter.render().decode()
    # No negative conditional band value should make it out — the
    # count-aware clip must apply.
    negative_lower = [
        line
        for line in body.splitlines()
        if line.startswith("count_metric_forecast_conditional_lower{")
        and line.rsplit(" ", 1)[-1].startswith("-")
    ]
    assert negative_lower == [], f"unexpected negative conditional bands: {negative_lower}"


@pytest.mark.asyncio
async def test_conditional_fit_failure_bumps_failures_counter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failing conditional fit must surface via ``forecast_failures_total``.

    The unconditional fit succeeds (so the forecast emits and the
    snapshot has data), and the conditional re-fit then raises — the
    operator should see ``reason="conditional_fit_error"`` rise without
    needing to scrape logs.

    Needs the refit path to be exercised: that requires (a) regressors
    actually configured on the query so the held-constant frame is
    non-empty *and* (b) ``model_registry.uses_regressors`` returning
    True so the runner doesn't short-circuit to the alias path. We
    patch the registry helper for clarity rather than relying on the
    unknown-model default.
    """
    from promforecast import models as model_registry  # noqa: PLC0415
    from promforecast.config import RegressorConfig  # noqa: PLC0415

    call_count = {"n": 0}

    class _ConditionalFails(_RampForecastModel):
        def fit_predict(
            self,
            df: pd.DataFrame,
            horizon: int,
            freq: str,
            levels: list[int],
            regressors: pd.DataFrame | None = None,
        ) -> pd.DataFrame:
            call_count["n"] += 1
            # First call: succeed (the unconditional pass).
            # Subsequent: fail (the conditional re-fit).
            if call_count["n"] == 1:
                return super().fit_predict(df, horizon, freq, levels, regressors)
            raise RuntimeError("conditional fit blew up")

    monkeypatch.setattr(model_registry, "build", lambda *a, **kw: _ConditionalFails())
    monkeypatch.setattr(model_registry, "uses_regressors", lambda name: True)

    cfg = _config_with_query(
        QueryConfig(
            id="m",
            promql="up",
            regressors=[RegressorConfig(id="hour_of_day", type="calendar")],
            emission=QueryEmissionConfig(conditional=True),
        )
    )
    source = _FakeSource({"up": [_frame(20)]})
    runner = Runner(config=cfg, source=source, exporter=(exporter := Exporter()))
    await runner.run_group(cfg.groups[0])

    body = exporter.render().decode()
    # Counter line appears with reason="conditional_fit_error".
    failure_lines = [
        line
        for line in body.splitlines()
        if "forecast_failures_total" in line
        and 'reason="conditional_fit_error"' in line
        and line.rsplit(" ", 1)[-1].strip() != "0.0"
    ]
    assert failure_lines, f"conditional_fit_error counter never bumped; body=\n{body[:500]}"


@pytest.mark.asyncio
async def test_conditional_short_circuits_refit_for_non_regressor_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A model that ignores regressors must not be refit for conditional emission.

    The built-in models all advertise ``uses_regressors = False``. When
    ``emission.conditional: true`` is on but the configured model is in
    that set, the conditional family should emit at zero extra fit cost
    — the runner aliases the unconditional forecast frame instead of
    refitting.
    """
    from promforecast import models as model_registry  # noqa: PLC0415

    call_count = {"n": 0}

    class _CountingRampModel(_RampForecastModel):
        # Inherits ``uses_regressors`` from _RampForecastModel below;
        # set it explicitly here so the test fails loudly if the
        # attribute disappears from the protocol.
        uses_regressors = False

        def fit_predict(
            self,
            df: pd.DataFrame,
            horizon: int,
            freq: str,
            levels: list[int],
            regressors: pd.DataFrame | None = None,
        ) -> pd.DataFrame:
            call_count["n"] += 1
            return super().fit_predict(df, horizon, freq, levels, regressors)

    monkeypatch.setattr(model_registry, "build", lambda *a, **kw: _CountingRampModel())
    # The runner reads the class-level marker via the registry's
    # ``uses_regressors`` helper, which looks the class up by name.
    # Patch the helper directly so the test doesn't depend on a registered
    # ``RampModel`` entry point.
    monkeypatch.setattr(model_registry, "uses_regressors", lambda name: False)

    cfg = _config_with_query(
        QueryConfig(
            id="m",
            promql="up",
            emission=QueryEmissionConfig(conditional=True),
        )
    )
    source = _FakeSource({"up": [_frame(20)]})
    runner = Runner(config=cfg, source=source, exporter=(exporter := Exporter()))
    await runner.run_group(cfg.groups[0])

    # Exactly one fit: the unconditional pass. No second fit was issued
    # for the conditional emission because the model advertises
    # uses_regressors=False, and the runner aliased the existing
    # forecast frame.
    assert call_count["n"] == 1
    # Conditional family still emitted on /metrics — the short-circuit
    # must not skip the *emission*, only the redundant refit.
    body = exporter.render().decode()
    assert "m_forecast_conditional" in body


@pytest.mark.asyncio
async def test_conditional_aliased_counter_bumps_on_short_circuit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``forecast_conditional_aliased_total`` must tick every time the
    runner skips the refit and aliases the unconditional frame. Without
    this counter, operators have no way to confirm the short-circuit
    is firing in their deployment — a regression that silently disables
    it would only show up as a CPU regression, not an alertable signal.
    """
    from promforecast import models as model_registry  # noqa: PLC0415

    monkeypatch.setattr(model_registry, "build", lambda *a, **kw: _RampForecastModel())
    monkeypatch.setattr(model_registry, "uses_regressors", lambda name: False)

    cfg = _config_with_query(
        QueryConfig(
            id="m",
            promql="up",
            emission=QueryEmissionConfig(conditional=True),
        )
    )
    source = _FakeSource({"up": [_frame(20)]})
    runner = Runner(config=cfg, source=source, exporter=(exporter := Exporter()))
    await runner.run_group(cfg.groups[0])

    body = exporter.render().decode()
    # The counter family is labelled with (group, query, model). The
    # counter must be > 0 because the unconditional fit's emitted model
    # (``RampModel``) advertises no regressor use, so the conditional
    # pass aliased.
    aliased_lines = [
        line
        for line in body.splitlines()
        if line.startswith("forecast_conditional_aliased_total{")
        and 'group="g"' in line
        and 'query="m"' in line
        and 'model="RampModel"' in line
        and line.rsplit(" ", 1)[-1].strip() != "0.0"
    ]
    assert aliased_lines, (
        "expected a forecast_conditional_aliased_total > 0 row for "
        f"(g, m, RampModel); body:\n{body[:600]}"
    )


@pytest.mark.asyncio
async def test_conditional_aliased_counter_quiet_when_refitting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Conversely: when the runner *did* refit, the alias counter must NOT
    show a positive sample — otherwise an operator alert on
    ``rate(forecast_conditional_aliased_total[5m]) == 0`` (the "is the
    short-circuit firing" check) would silently always pass and miss the
    case where every conditional emission is paying the refit cost.
    """
    from promforecast import models as model_registry  # noqa: PLC0415
    from promforecast.config import RegressorConfig  # noqa: PLC0415

    monkeypatch.setattr(model_registry, "build", lambda *a, **kw: _RampForecastModel())
    monkeypatch.setattr(model_registry, "uses_regressors", lambda name: True)

    cfg = _config_with_query(
        QueryConfig(
            id="m",
            promql="up",
            regressors=[RegressorConfig(id="hour_of_day", type="calendar")],
            emission=QueryEmissionConfig(conditional=True),
        )
    )
    source = _FakeSource({"up": [_frame(20)]})
    runner = Runner(config=cfg, source=source, exporter=(exporter := Exporter()))
    await runner.run_group(cfg.groups[0])

    body = exporter.render().decode()
    # When no alias happens the counter family must stay quiet — the
    # exporter only emits ``forecast_conditional_aliased_total`` after
    # at least one alias has occurred.
    assert "forecast_conditional_aliased_total" not in body


@pytest.mark.asyncio
async def test_conditional_short_circuits_when_model_ignores_configured_regressors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The `uses_regressors=False` branch must fire even when regressors are configured.

    This is the realistic built-in case: the operator wires a calendar
    regressor onto a query AND turns on ``emission.conditional``, but
    the configured model (every built-in wrapper) ignores the
    regressors parameter. Without this short-circuit, every query of
    this shape pays a wasted refit per series per model.
    """
    from promforecast import models as model_registry  # noqa: PLC0415
    from promforecast.config import RegressorConfig  # noqa: PLC0415

    call_count = {"n": 0}

    class _CountingRampModel(_RampForecastModel):
        uses_regressors = False

        def fit_predict(
            self,
            df: pd.DataFrame,
            horizon: int,
            freq: str,
            levels: list[int],
            regressors: pd.DataFrame | None = None,
        ) -> pd.DataFrame:
            call_count["n"] += 1
            return super().fit_predict(df, horizon, freq, levels, regressors)

    monkeypatch.setattr(model_registry, "build", lambda *a, **kw: _CountingRampModel())
    monkeypatch.setattr(model_registry, "uses_regressors", lambda name: False)

    cfg = _config_with_query(
        QueryConfig(
            id="m",
            promql="up",
            regressors=[RegressorConfig(id="hour_of_day", type="calendar")],
            emission=QueryEmissionConfig(conditional=True),
        )
    )
    source = _FakeSource({"up": [_frame(20)]})
    runner = Runner(config=cfg, source=source, exporter=(exporter := Exporter()))
    await runner.run_group(cfg.groups[0])

    # Exactly one fit — even though regressors were configured, the
    # model marker said it doesn't use them, so the runner aliased the
    # unconditional frame.
    assert call_count["n"] == 1
    # And the conditional family still landed on /metrics.
    assert "m_forecast_conditional" in exporter.render().decode()


@pytest.mark.asyncio
async def test_conditional_curve_factory_emits_under_conditional_metric_family(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The new ``conditional_forecast_curve_points`` factory method must
    emit curves under the ``_forecast_conditional*`` metric family —
    not the unconditional ``_forecast*`` family, which would collide
    with the unconditional sink push under the same labelset.
    """
    from promforecast import models as model_registry  # noqa: PLC0415
    from promforecast.sink import ForecastCurvePoint, ForecastSink  # noqa: PLC0415

    captured: list[ForecastCurvePoint] = []

    class _CapturingSink(ForecastSink):
        async def write(
            self,
            points: list[ForecastCurvePoint],
            *,
            on_retry: object = None,
        ) -> None:
            captured.extend(points)

        async def aclose(self) -> None:
            return None

    monkeypatch.setattr(model_registry, "build", lambda *a, **kw: _RampForecastModel())

    cfg = _config_with_query(
        QueryConfig(
            id="cpu_busy",
            promql="up",
            emission=QueryEmissionConfig(conditional=True),
        )
    )
    source = _FakeSource({"up": [_frame(20)]})
    runner = Runner(
        config=cfg,
        source=source,
        exporter=Exporter(),
        sink=_CapturingSink(),
    )
    await runner.run_group(cfg.groups[0])

    # The sink received curve points; the conditional ones must be
    # under the ``_forecast_conditional*`` family (not the rewritten
    # ``_forecast`` family that the old post-hoc renamer used to leak).
    conditional_metrics = {p.metric for p in captured if "_forecast_conditional" in p.metric}
    assert "cpu_busy_forecast_conditional" in conditional_metrics
    # Lower / upper band rows present too at the configured 80% level.
    assert "cpu_busy_forecast_conditional_lower" in conditional_metrics
    assert "cpu_busy_forecast_conditional_upper" in conditional_metrics


@pytest.mark.asyncio
async def test_conditional_refits_when_model_uses_regressors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A regressor-aware model must still be refit for the conditional pass.

    The Prophet plug-in (and any future regressor-aware model) reads the
    held-constant regressor frame at fit time, so the conditional
    output legitimately differs from the unconditional. The runner must
    not alias the unconditional frame in that case.
    """
    from promforecast import models as model_registry  # noqa: PLC0415

    call_count = {"n": 0}

    class _RegressorAwareRamp(_RampForecastModel):
        uses_regressors = True

        def fit_predict(
            self,
            df: pd.DataFrame,
            horizon: int,
            freq: str,
            levels: list[int],
            regressors: pd.DataFrame | None = None,
        ) -> pd.DataFrame:
            call_count["n"] += 1
            return super().fit_predict(df, horizon, freq, levels, regressors)

    monkeypatch.setattr(model_registry, "build", lambda *a, **kw: _RegressorAwareRamp())
    monkeypatch.setattr(model_registry, "uses_regressors", lambda name: True)

    # ``conditional`` is on AND a regressor is configured — the runner
    # must run an extra fit per emitted model for the held-constant
    # refit.
    from promforecast.config import RegressorConfig  # noqa: PLC0415

    cfg = _config_with_query(
        QueryConfig(
            id="m",
            promql="up",
            regressors=[RegressorConfig(id="hour_of_day", type="calendar")],
            emission=QueryEmissionConfig(conditional=True),
        )
    )
    source = _FakeSource({"up": [_frame(20)]})
    runner = Runner(config=cfg, source=source, exporter=Exporter())
    await runner.run_group(cfg.groups[0])

    # 2 fits: unconditional + conditional refit.
    assert call_count["n"] == 2


@pytest.mark.asyncio
async def test_decomposition_emission_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """No component family until ``emission.decompose`` is flipped on."""
    from promforecast import models as model_registry  # noqa: PLC0415

    monkeypatch.setattr(model_registry, "build", lambda *a, **kw: _RampForecastModel())

    cfg = _config_with_query(QueryConfig(id="m", promql="up"))
    source = _FakeSource({"up": [_frame(20)]})
    runner = Runner(config=cfg, source=source, exporter=(exporter := Exporter()))
    await runner.run_group(cfg.groups[0])

    body = exporter.render().decode()
    assert "m_forecast_component" not in body
    assert "forecast_decomposition_skipped_total" not in body


@pytest.mark.asyncio
async def test_decomposition_emission_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Enabling the toggle materialises one row per component on /metrics."""
    from promforecast import models as model_registry  # noqa: PLC0415

    monkeypatch.setattr(model_registry, "build", lambda *a, **kw: _RampForecastModel())

    cfg = _config_with_query(
        QueryConfig(
            id="cpu_busy",
            promql="up",
            emission=QueryEmissionConfig(decompose=True),
        )
    )
    source = _FakeSource({"up": [_frame(20)]})
    runner = Runner(config=cfg, source=source, exporter=(exporter := Exporter()))
    await runner.run_group(cfg.groups[0])

    body = exporter.render().decode()
    assert "cpu_busy_forecast_component" in body
    # All four components show up exactly once per (series, model) at the
    # end-of-horizon row.
    for component in ("trend", "seasonal", "level", "residual"):
        assert f'component="{component}"' in body


@pytest.mark.asyncio
async def test_decomposition_curve_pushed_to_sink(monkeypatch: pytest.MonkeyPatch) -> None:
    """Per-step component curves must reach the remote_write sink.

    Without the sink push, dashboards built on the per-step decomposition
    family have no data to render — only the end-of-horizon snapshot is
    available, which is a single point, not a stacked-area panel input.
    """
    from promforecast import models as model_registry  # noqa: PLC0415
    from promforecast.sink import ForecastCurvePoint, ForecastSink  # noqa: PLC0415

    captured: list[ForecastCurvePoint] = []

    class _CapturingSink(ForecastSink):
        async def write(
            self,
            points: list[ForecastCurvePoint],
            *,
            on_retry: object = None,
        ) -> None:
            captured.extend(points)

        async def aclose(self) -> None:
            return None

    monkeypatch.setattr(model_registry, "build", lambda *a, **kw: _RampForecastModel())

    cfg = _config_with_query(
        QueryConfig(
            id="cpu_busy",
            promql="up",
            emission=QueryEmissionConfig(decompose=True),
        )
    )
    source = _FakeSource({"up": [_frame(20)]})
    runner = Runner(
        config=cfg,
        source=source,
        exporter=Exporter(),
        sink=_CapturingSink(),
    )
    await runner.run_group(cfg.groups[0])

    component_curves = [p for p in captured if p.metric == "cpu_busy_forecast_component"]
    assert component_curves, "no component curve points reached the sink"
    # Each step contributes 4 components (trend/seasonal/level/residual).
    seen_components = {p.labels.get("component") for p in component_curves}
    assert seen_components == {"trend", "seasonal", "level", "residual"}


@pytest.mark.asyncio
async def test_ensemble_row_is_decomposed_alongside_component_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The synthetic ``ensemble`` row must produce its own decomposition.

    ``ensemble`` is deliberately *not* in
    :data:`decomposition.NON_DECOMPOSABLE_MODELS` because the averaged
    curve is still a valid forecast shape. Operators looking at the
    ensemble forecast benefit from seeing trend/seasonal/level/residual
    on the *ensemble* curve alongside the same split on each component
    model. Lock the behaviour in here so a future tweak to the registry
    can't silently drop the ensemble decomposition.
    """
    from promforecast import models as model_registry  # noqa: PLC0415

    monkeypatch.setattr(model_registry, "build", lambda *a, **kw: _RampForecastModel())

    cfg = _config_with_query(
        QueryConfig(
            id="cpu_busy",
            promql="up",
            emission=QueryEmissionConfig(decompose=True),
        )
    )
    # Ensemble mode: emit every configured model PLUS the synthetic
    # ``ensemble`` row.
    cfg.defaults.accuracy.auto_select = "ensemble"
    cfg.defaults.accuracy.evaluate = True  # backtests power the weights
    cfg.defaults.models = [
        cfg.defaults.models[0],
        type(cfg.defaults.models[0])(name="RampModelTwo"),
    ]

    source = _FakeSource({"up": [_frame(40)]})
    runner = Runner(config=cfg, source=source, exporter=(exporter := Exporter()))
    await runner.run_group(cfg.groups[0])

    body = exporter.render().decode()
    # The ensemble row's decomposition lands on /metrics alongside the
    # component models'. We assert specifically on the ``ensemble``
    # model label rather than just the metric name so a regression that
    # drops the ensemble decomposition silently is caught.
    component_lines = [
        line for line in body.splitlines() if line.startswith("cpu_busy_forecast_component{")
    ]
    ensemble_decomp_lines = [line for line in component_lines if 'model="ensemble"' in line]
    assert ensemble_decomp_lines, (
        'ensemble row produced no decomposition; expected `model="ensemble"` rows '
        "on cpu_busy_forecast_component. body sample:\n" + "\n".join(component_lines[:5])
    )
    # And no skip counter was bumped for the ensemble row — it must be
    # treated as decomposable, not as an unsupported model.
    skip_lines = [
        line
        for line in body.splitlines()
        if line.startswith("forecast_decomposition_skipped_total{")
        and 'model="ensemble"' in line
        and line.rsplit(" ", 1)[-1].strip() != "0.0"
    ]
    assert skip_lines == [], f"ensemble row was unexpectedly skipped: {skip_lines}"


@pytest.mark.asyncio
async def test_decomposition_skip_counter_bumps_on_unsupported_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-decomposable model name skips emission and bumps the counter.

    The ``Croston`` family is registered as non-decomposable; we point the
    registry stub at a Croston-named model so the runner takes the skip
    path. The forecast itself still emits normally.
    """
    from promforecast import models as model_registry  # noqa: PLC0415

    class _CrostonStub(_RampForecastModel):
        name = "Croston"

    monkeypatch.setattr(model_registry, "build", lambda *a, **kw: _CrostonStub())

    cfg = _config_with_query(
        QueryConfig(
            id="m",
            promql="up",
            emission=QueryEmissionConfig(decompose=True),
        )
    )
    # Inject the unsupported model name into the defaults so the snapshot
    # forecast points label themselves with ``model="Croston"``.
    cfg.defaults.models[0].name = "Croston"

    source = _FakeSource({"up": [_frame(20)]})
    runner = Runner(config=cfg, source=source, exporter=(exporter := Exporter()))
    await runner.run_group(cfg.groups[0])

    body = exporter.render().decode()
    assert "forecast_decomposition_skipped_total" in body
    assert 'model="Croston"' in body
    assert 'reason="unsupported_model"' in body
    # The skip counter now carries (group, query) labels so operators
    # can alert per-query on persistent skip-path tripping.
    assert 'group="g"' in body
    assert 'query="m"' in body
    # Component family is *not* emitted when every emitted model skipped.
    assert "m_forecast_component" not in body
