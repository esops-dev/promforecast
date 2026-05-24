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


def _conformal_conditional_config(
    *, models_list: list[str] | None = None, holdout_fraction: float = 0.2
) -> Config:
    """Helper: a config with conformal + conditional both enabled."""
    from promforecast.config import ConformalConfig, RegressorConfig  # noqa: PLC0415

    return Config(
        datasource=DatasourceConfig(url="http://x"),
        server=ServerConfig(),
        safety=SafetyConfig(),
        defaults=DefaultsConfig(
            min_points=5,
            models=models_list or ["RampModel"],
            confidence_levels=[80],
            horizon=timedelta(hours=2),
            step=timedelta(minutes=5),
            accuracy=AccuracyConfig(
                evaluate=False,
                conformal=ConformalConfig(enabled=True, holdout_fraction=holdout_fraction),
            ),
        ),
        groups=[
            GroupConfig(
                name="g",
                queries=[
                    QueryConfig(
                        id="m",
                        promql="up",
                        regressors=[RegressorConfig(id="hour_of_day", type="calendar")],
                        emission=QueryEmissionConfig(conditional=True),
                    )
                ],
            )
        ],
    )


@pytest.mark.asyncio
async def test_conditional_calibration_gauge_emits_when_both_features_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``forecast_conditional_band_calibration_offset`` must land for opted-in queries.

    Pre-v1.6.5 the conditional band reused the unconditional offsets,
    so there was no separate gauge to expose the per-(model, level)
    widening the conditional band actually received. The v1.6.5 story
    introduces the dedicated calibration pass; verify the gauge family
    shows up on /metrics whenever both ``accuracy.conformal.enabled``
    and ``emission.conditional`` are on for the query and at least one
    emitted model is regressor-aware.
    """
    from promforecast import models as model_registry  # noqa: PLC0415

    class _RegressorAware(_RampForecastModel):
        uses_regressors = True

    monkeypatch.setattr(model_registry, "build", lambda *a, **kw: _RegressorAware())
    monkeypatch.setattr(model_registry, "uses_regressors", lambda name: True)

    cfg = _conformal_conditional_config()
    # A longer series is needed so the conformal holdout split (20% of
    # the lookback) leaves enough points on both sides for a meaningful
    # quantile estimate.
    source = _FakeSource({"up": [_frame(60)]})
    runner = Runner(config=cfg, source=source, exporter=(exporter := Exporter()))
    await runner.run_group(cfg.groups[0])

    body = exporter.render().decode()
    assert "forecast_conditional_band_calibration_offset" in body, (
        f"expected dedicated conditional calibration gauge; body=\n{body[:1000]}"
    )
    # Both the unconditional and conditional gauges should be present so
    # the operator can compare the per-(model, level) widening.
    assert "forecast_band_calibration_offset" in body
    # Conditional band rows pick up ``calibrated="true"`` when fresh
    # offsets were applied — mirrors the unconditional family's
    # post-hoc labelling so a single dashboard filter
    # (``calibrated="true"``) hits both. Without this label the
    # operator can't distinguish a fresh-calibrated conditional band
    # from one that fell back to inherited offsets purely from the
    # labelset.
    calibrated_conditional_bands = [
        line
        for line in body.splitlines()
        if line.startswith(("m_forecast_conditional_lower{", "m_forecast_conditional_upper{"))
        and 'calibrated="true"' in line
    ]
    assert calibrated_conditional_bands, (
        f'expected conditional band rows to carry calibrated="true"; body=\n{body[:2000]}'
    )
    # The yhat row (no band, no calibration) must NOT carry the label.
    central_rows = [
        line for line in body.splitlines() if line.startswith("m_forecast_conditional{")
    ]
    assert central_rows, "expected at least one central conditional row"
    for line in central_rows:
        assert 'calibrated="true"' not in line, (
            f"calibrated label leaked onto central conditional row: {line}"
        )


@pytest.mark.asyncio
async def test_conditional_calibration_offsets_can_differ_from_unconditional(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fresh conditional pass must produce its own offsets, not a copy of the unconditional.

    The point of the v1.6.5 conditional-band calibration is that the
    holdout residuals are scored under flat-held regressors —
    matching the production-time conditional inference contract —
    rather than under the actual regressor trajectory the
    unconditional pass uses. For a regressor-responsive model, the
    two regimes produce different residual distributions, so the two
    offset gauges must be able to carry different per-(model, level)
    values. A test that asserts only "the gauge exists" misses the
    regression where the runner silently aliases the unconditional
    offsets back into the conditional family.
    """
    from promforecast import models as model_registry  # noqa: PLC0415

    class _RegressorSwingModel(_RampForecastModel):
        """Calibration fit whose residuals depend on whether regressors vary.

        When the regressors frame's non-``ds`` columns vary across
        rows (unconditional calibration), the predicted ``yhat`` is
        offset by the regressor's row-by-row delta — so the
        predictions track the actuals (small residuals, narrow
        calibration offset). When the regressors are flat-held
        (conditional calibration), the model predicts a constant —
        so residuals fan out and the calibration offset has to
        widen to match.
        """

        uses_regressors = True

        def fit_predict(
            self,
            df: pd.DataFrame,
            horizon: int,
            freq: str,
            levels: list[int],
            regressors: pd.DataFrame | None = None,
        ) -> pd.DataFrame:
            idx = pd.date_range(df["ds"].iloc[-1], periods=horizon + 1, freq=freq)[1:]
            # Base ramp matching the in-sample tail so unconditional
            # residuals are tiny when regressors vary.
            base = float(df["y"].iloc[-1]) if "y" in df.columns else 0.0
            ramp = [base + i * 0.5 for i in range(horizon)]
            # Inject the regressor contribution row-by-row so a varying
            # regressor *helps* the forecast follow the actual.
            if regressors is not None and len(regressors) >= horizon:
                non_ds = [c for c in regressors.columns if c != "ds"]
                if non_ds:
                    contrib = regressors[non_ds[0]].to_numpy(dtype=float)[:horizon]
                    ramp = [r + float(c) for r, c in zip(ramp, contrib, strict=False)]
            out = pd.DataFrame({"unique_id": ["s"] * horizon, "ds": idx, "yhat": ramp})
            for level in levels:
                # Tight base band; calibration is what widens it to
                # match real residuals.
                out[f"yhat_lower_{level}"] = [v - 0.01 for v in ramp]
                out[f"yhat_upper_{level}"] = [v + 0.01 for v in ramp]
            return out

    monkeypatch.setattr(model_registry, "build", lambda *a, **kw: _RegressorSwingModel())
    monkeypatch.setattr(model_registry, "uses_regressors", lambda name: True)

    cfg = _conformal_conditional_config()
    # Need a long enough series so the holdout has enough samples for
    # the quantile to discriminate between the two regimes.
    source = _FakeSource({"up": [_frame(100)]})
    runner = Runner(config=cfg, source=source, exporter=(exporter := Exporter()))
    await runner.run_group(cfg.groups[0])

    body = exporter.render().decode()

    def _value(line: str) -> float:
        return float(line.rsplit(" ", 1)[-1].strip())

    unconditional = {
        line for line in body.splitlines() if line.startswith("forecast_band_calibration_offset{")
    }
    conditional = {
        line
        for line in body.splitlines()
        if line.startswith("forecast_conditional_band_calibration_offset{")
    }
    assert unconditional and conditional, (
        f"expected both unconditional + conditional gauge families to emit; body=\n{body[:1500]}"
    )

    # The fresh conditional pass must produce its own offsets — at
    # least one (model, level) pair must carry a value distinct from
    # the unconditional reading. Otherwise the new gauge family
    # would just duplicate the unconditional one and the v1.6.5
    # story's contract is silently broken. Compare per-(model, level)
    # rather than across all rows so a coincidental match at one
    # level doesn't mask a real divergence at another.
    def _key_and_value(line: str, marker: str) -> tuple[str, float]:
        """Strip the gauge name and metric value to leave just the labelset for keying."""
        head, _, _value_part = line.partition(" ")
        # ``head`` is ``<metric>{labels}``; drop the metric prefix and
        # any conformal-affected ``calibrated`` label so the two
        # families key the same way.
        labelset = head[len(marker) :]
        return labelset, _value(line)

    unc_by_label = dict(
        _key_and_value(line, "forecast_band_calibration_offset") for line in unconditional
    )
    cond_by_label = dict(
        _key_and_value(line, "forecast_conditional_band_calibration_offset") for line in conditional
    )
    overlapping = unc_by_label.keys() & cond_by_label.keys()
    assert overlapping, (
        f"expected at least one (model, level) pair to appear in both gauges; "
        f"uncond keys={list(unc_by_label)[:3]} cond keys={list(cond_by_label)[:3]}"
    )
    differing = [k for k in overlapping if abs(unc_by_label[k] - cond_by_label[k]) > 1e-6]
    assert differing, (
        f"fresh conditional offsets must differ from unconditional for at least "
        f"one (model, level); both gauges report {unc_by_label} / {cond_by_label}"
    )


@pytest.mark.asyncio
async def test_conditional_curve_carries_calibrated_label_when_fresh_offsets_apply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The per-step conditional curve (sink path) must also carry the calibrated label.

    Without this the snapshot (scrape) and the curve (remote_write
    push) would disagree on labelling: the snapshot's band rows
    would carry ``calibrated="true"`` but the per-step curve
    landing in the long-term TSDB wouldn't. Dashboards joining the
    two by labelset would lose rows. The label must round-trip
    through both emission paths.
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

    class _RegressorAware(_RampForecastModel):
        uses_regressors = True

    monkeypatch.setattr(model_registry, "build", lambda *a, **kw: _RegressorAware())
    monkeypatch.setattr(model_registry, "uses_regressors", lambda name: True)

    cfg = _conformal_conditional_config()
    source = _FakeSource({"up": [_frame(60)]})
    runner = Runner(
        config=cfg,
        source=source,
        exporter=Exporter(),
        sink=_CapturingSink(),
    )
    await runner.run_group(cfg.groups[0])

    band_curve_rows = [
        p
        for p in captured
        if p.metric in {"m_forecast_conditional_lower", "m_forecast_conditional_upper"}
    ]
    assert band_curve_rows, (
        f"expected conditional band curve points in the sink push; "
        f"got metrics={ {p.metric for p in captured} }"
    )
    calibrated_band_rows = [p for p in band_curve_rows if p.labels.get("calibrated") == "true"]
    assert calibrated_band_rows, (
        f'expected band curve rows to carry calibrated="true"; '
        f"sample labelsets={[p.labels for p in band_curve_rows[:3]]}"
    )
    # The central conditional yhat curve must NOT carry the label
    # — calibration shifts the band level, not the point forecast.
    central_curve_rows = [p for p in captured if p.metric == "m_forecast_conditional"]
    assert central_curve_rows, "expected at least one central conditional curve row"
    for p in central_curve_rows:
        assert p.labels.get("calibrated") != "true", (
            f"calibrated label leaked onto central curve row: {p.labels}"
        )


@pytest.mark.asyncio
async def test_conditional_band_uncalibrated_when_fresh_offsets_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No ``calibrated="true"`` label on the conditional band when fresh calibration fell back.

    Failure paths reuse the inherited unconditional offsets, but the
    label is reserved for the *fresh* per-fit pass so an operator can
    distinguish "this band was widened by the production-time
    inference contract's calibration" from "this band inherited the
    unconditional widening because fresh calibration didn't run".
    """
    from promforecast import models as model_registry  # noqa: PLC0415

    class _ConditionalCalibFails(_RampForecastModel):
        uses_regressors = True

        def fit_predict(
            self,
            df: pd.DataFrame,
            horizon: int,
            freq: str,
            levels: list[int],
            regressors: pd.DataFrame | None = None,
        ) -> pd.DataFrame:
            is_train_prefix = len(df) < 60
            is_flat_held = (
                regressors is not None
                and len(regressors) > 1
                and all(
                    regressors[col].nunique(dropna=False) == 1
                    for col in regressors.columns
                    if col != "ds"
                )
            )
            if is_train_prefix and is_flat_held:
                raise RuntimeError("conditional calibration unavailable")
            return super().fit_predict(df, horizon, freq, levels, regressors)

    monkeypatch.setattr(model_registry, "build", lambda *a, **kw: _ConditionalCalibFails())
    monkeypatch.setattr(model_registry, "uses_regressors", lambda name: True)

    cfg = _conformal_conditional_config()
    source = _FakeSource({"up": [_frame(60)]})
    runner = Runner(config=cfg, source=source, exporter=(exporter := Exporter()))
    await runner.run_group(cfg.groups[0])

    body = exporter.render().decode()
    # Conditional bands are emitted (the snapshot still publishes), but
    # they do NOT carry calibrated="true" — fresh calibration failed and
    # the runner fell back to the inherited unconditional offsets.
    conditional_band_rows = [
        line
        for line in body.splitlines()
        if line.startswith(("m_forecast_conditional_lower{", "m_forecast_conditional_upper{"))
    ]
    assert conditional_band_rows, (
        f"expected conditional band rows even when fresh calibration failed; body=\n{body[:1500]}"
    )
    for line in conditional_band_rows:
        assert 'calibrated="true"' not in line, (
            f"calibrated label should be absent on fallback path: {line}"
        )


@pytest.mark.asyncio
async def test_conditional_calibration_skipped_when_no_model_uses_regressors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No conditional gauge emitted when every emitted model ignores regressors.

    The fresh calibration would produce identical numbers to the
    unconditional pass (the held-constant frame has no effect on a
    model that doesn't read regressors), so emitting a parallel offset
    gauge with the same values would just be noise.
    """
    from promforecast import models as model_registry  # noqa: PLC0415

    monkeypatch.setattr(model_registry, "build", lambda *a, **kw: _RampForecastModel())
    monkeypatch.setattr(model_registry, "uses_regressors", lambda name: False)

    cfg = _conformal_conditional_config()
    source = _FakeSource({"up": [_frame(60)]})
    runner = Runner(config=cfg, source=source, exporter=(exporter := Exporter()))
    await runner.run_group(cfg.groups[0])

    body = exporter.render().decode()
    assert "forecast_conditional_band_calibration_offset" not in body
    # Failure counter must also stay quiet — short-circuit isn't a failure.
    assert "forecast_conditional_calibration_failures_total" not in body


@pytest.mark.asyncio
async def test_conditional_calibration_failure_bumps_counter_and_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failing conditional-calibration fit must bump the failures counter and fall back.

    The runner runs an extra fit per emitted model to derive the
    conditional offsets. If that fit raises, the snapshot should still
    publish (with the band silently widened by the inherited
    unconditional offsets), and the failure counter must tick with a
    bounded ``reason``.
    """
    from promforecast import models as model_registry  # noqa: PLC0415

    # Track which fit is which by inspecting the input frame's shape
    # (training prefix is shorter than the full lookback) and whether the
    # regressors frame is flat-held (every row equal). This is more
    # robust than counting calls because the implementation may reorder
    # or parallelise fits without changing what the test is verifying.
    seen_flat_held_calibration = {"n": 0}

    class _ConditionalCalibFails(_RampForecastModel):
        uses_regressors = True

        def fit_predict(
            self,
            df: pd.DataFrame,
            horizon: int,
            freq: str,
            levels: list[int],
            regressors: pd.DataFrame | None = None,
        ) -> pd.DataFrame:
            # The conditional calibration is the only fit whose
            # ``regressors`` frame has its non-``ds`` columns flat-held
            # at a single value AND whose ``df`` length is the
            # train-prefix length (shorter than the full lookback). The
            # conditional refit also gets a flat-held frame but on the
            # full lookback; the unconditional calibration gets the
            # train prefix but with varying regressors. So
            # ``flat AND len(df) < 60`` uniquely identifies the
            # conditional calibration call.
            is_train_prefix = len(df) < 60
            is_flat_held = (
                regressors is not None
                and len(regressors) > 1
                and all(
                    regressors[col].nunique(dropna=False) == 1
                    for col in regressors.columns
                    if col != "ds"
                )
            )
            if is_train_prefix and is_flat_held:
                seen_flat_held_calibration["n"] += 1
                raise RuntimeError("conditional calibration blew up")
            return super().fit_predict(df, horizon, freq, levels, regressors)

    monkeypatch.setattr(model_registry, "build", lambda *a, **kw: _ConditionalCalibFails())
    monkeypatch.setattr(model_registry, "uses_regressors", lambda name: True)

    cfg = _conformal_conditional_config()
    source = _FakeSource({"up": [_frame(60)]})
    runner = Runner(config=cfg, source=source, exporter=(exporter := Exporter()))
    await runner.run_group(cfg.groups[0])

    body = exporter.render().decode()
    # The failures counter must tick with one of the bounded reasons.
    failure_lines = [
        line
        for line in body.splitlines()
        if line.startswith("forecast_conditional_calibration_failures_total{")
        and 'reason="fit_error"' in line
        and line.rsplit(" ", 1)[-1].strip() != "0.0"
    ]
    assert failure_lines, (
        f"expected conditional calibration failure counter to tick; body=\n{body[:1500]}"
    )
    # The dedicated conditional offset gauge must NOT be emitted for
    # the failed model (the runner skipped it cleanly).
    assert "forecast_conditional_band_calibration_offset" not in body
    # The unconditional snapshot must still publish — failure isolation.
    assert "m_forecast_conditional" in body


@pytest.mark.asyncio
async def test_conditional_calibration_degenerate_holdout_bumps_counter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A holdout too small to support a quantile must bump the counter with degenerate_holdout."""
    from promforecast import models as model_registry  # noqa: PLC0415

    class _RegressorAware(_RampForecastModel):
        uses_regressors = True

    monkeypatch.setattr(model_registry, "build", lambda *a, **kw: _RegressorAware())
    monkeypatch.setattr(model_registry, "uses_regressors", lambda name: True)

    # 12-point series with the default 0.2 holdout fraction → holdout
    # would be ~2 points, below the 5-point minimum in
    # ``holdout_split``, so the conditional calibration short-circuits
    # with a degenerate_holdout failure.
    cfg = _conformal_conditional_config()
    source = _FakeSource({"up": [_frame(12)]})
    runner = Runner(config=cfg, source=source, exporter=(exporter := Exporter()))
    await runner.run_group(cfg.groups[0])

    body = exporter.render().decode()
    degenerate_lines = [
        line
        for line in body.splitlines()
        if line.startswith("forecast_conditional_calibration_failures_total{")
        and 'reason="degenerate_holdout"' in line
        and line.rsplit(" ", 1)[-1].strip() != "0.0"
    ]
    assert degenerate_lines, (
        f"expected degenerate_holdout failure to be counted; body=\n{body[:1500]}"
    )


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


@pytest.mark.asyncio
async def test_conditional_calibration_offsets_inherit_on_skipped_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``forecast_conditional_band_calibration_offset`` rows must survive a partial tick.

    The v1.6.5 story registers the conditional calibration offsets in
    ``_INHERITANCE_REGISTRY`` so per-query refresh cadences don't drop
    them off /metrics when the opted-in query is skipped on a tick. The
    registry entry is easy to drop in a future refactor; this test pins
    the contract so a regression would fail loudly. Without this test,
    a future contributor renaming or removing the conditional family
    from the registry would silently strip the gauge from partial-tick
    snapshots.
    """
    from promforecast import models as model_registry  # noqa: PLC0415
    from promforecast.config import ConformalConfig, RegressorConfig  # noqa: PLC0415

    class _RegressorAware(_RampForecastModel):
        uses_regressors = True

    monkeypatch.setattr(model_registry, "build", lambda *a, **kw: _RegressorAware())
    monkeypatch.setattr(model_registry, "uses_regressors", lambda name: True)

    cfg = Config(
        datasource=DatasourceConfig(url="http://x"),
        server=ServerConfig(),
        safety=SafetyConfig(),
        defaults=DefaultsConfig(
            min_points=5,
            models=["RampModel"],
            confidence_levels=[80],
            horizon=timedelta(hours=2),
            step=timedelta(minutes=5),
            accuracy=AccuracyConfig(
                evaluate=False,
                conformal=ConformalConfig(enabled=True, holdout_fraction=0.2),
            ),
        ),
        groups=[
            GroupConfig(
                name="g",
                queries=[
                    QueryConfig(id="active_query", promql="qa"),
                    QueryConfig(
                        id="conditional_query",
                        promql="qb",
                        regressors=[RegressorConfig(id="hour_of_day", type="calendar")],
                        emission=QueryEmissionConfig(conditional=True),
                    ),
                ],
            )
        ],
    )
    source = _FakeSource({"qa": [_frame(60)], "qb": [_frame(60)]})
    runner = Runner(config=cfg, source=source, exporter=(exporter := Exporter()))

    # First, a full run so the conditional calibration gauge lands for
    # ``conditional_query`` and the snapshot has data to inherit from.
    await runner.run_group(cfg.groups[0])
    initial = exporter.render().decode()
    initial_offset_lines = [
        line
        for line in initial.splitlines()
        if line.startswith("forecast_conditional_band_calibration_offset{")
        and 'id="conditional_query"' in line
    ]
    assert initial_offset_lines, (
        f"initial full run must emit the conditional calibration gauge; body=\n{initial[:1500]}"
    )

    # Partial tick that runs *only* ``active_query``. The conditional
    # calibration rows for ``conditional_query`` must survive via the
    # inheritance registry; without the registry entry the next /metrics
    # scrape would silently drop them.
    await runner.run_group(cfg.groups[0], queries=[cfg.groups[0].queries[0]])
    partial = exporter.render().decode()
    partial_offset_lines = [
        line
        for line in partial.splitlines()
        if line.startswith("forecast_conditional_band_calibration_offset{")
        and 'id="conditional_query"' in line
    ]
    assert partial_offset_lines, (
        "conditional_band_calibration_offset rows for the skipped query must "
        "be inherited via _INHERITANCE_REGISTRY; if this fails, check that "
        "conditional_calibration_offsets is still registered.\n"
        f"partial body excerpt:\n{partial[:1500]}"
    )
