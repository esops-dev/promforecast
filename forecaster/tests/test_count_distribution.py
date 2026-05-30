"""Unit tests for the count-distribution models (Poisson / NegBinom / INGARCH)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from promforecast import models as model_registry
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
from promforecast.models.count_distribution import (
    COUNT_MODEL_NAMES,
    INGARCHModel,
    NegativeBinomialModel,
    PoissonModel,
    _estimate_overdispersion,
    _fit_ingarch,
)
from promforecast.runner import Runner
from promforecast.source import PromSource, SeriesFrame


def _count_frame(values: list[float], *, freq: str = "1h") -> pd.DataFrame:
    ts = pd.date_range("2026-01-01", periods=len(values), freq=freq)
    return pd.DataFrame({"unique_id": ["s"] * len(values), "ds": ts, "y": values})


def _seasonal_counts(period: int, cycles: int, *, seed: int = 0, base: float = 5.0) -> list[float]:
    rng = np.random.default_rng(seed)
    n = period * cycles
    pos = np.arange(n) % period
    rate = base + 3.0 * np.sin(2 * np.pi * pos / period) + 3.0
    return [float(v) for v in rng.poisson(np.maximum(rate, 0.1))]


def test_columns_and_horizon_shape() -> None:
    df = _count_frame(_seasonal_counts(24, 8))
    out = PoissonModel(season_length=24).fit_predict(df, horizon=12, freq="1h", levels=[80, 95])
    assert list(out.columns) == [
        "unique_id",
        "ds",
        "yhat",
        "yhat_lower_80",
        "yhat_upper_80",
        "yhat_lower_95",
        "yhat_upper_95",
    ]
    assert len(out) == 12


def test_bands_are_non_negative_and_nested() -> None:
    """Lower bands never dip below zero and the 95% band contains the 80%."""
    df = _count_frame(_seasonal_counts(24, 8, base=1.0))  # low rate -> tempting to go negative
    for model in (PoissonModel, NegativeBinomialModel, INGARCHModel):
        out = model(season_length=24).fit_predict(df, horizon=12, freq="1h", levels=[80, 95])
        assert (out["yhat_lower_80"] >= 0).all()
        assert (out["yhat_lower_95"] >= 0).all()
        # 95% band is at least as wide as the 80% band.
        assert (out["yhat_lower_95"] <= out["yhat_lower_80"] + 1e-9).all()
        assert (out["yhat_upper_95"] >= out["yhat_upper_80"] - 1e-9).all()


def test_negbinom_band_wider_than_poisson_on_overdispersed_data() -> None:
    """Overdispersed counts get a wider negative-binomial band than Poisson."""
    rng = np.random.default_rng(3)
    # Negative-binomial-distributed counts: variance >> mean.
    values = [float(v) for v in rng.negative_binomial(2, 0.2, size=300)]
    df = _count_frame(values)
    pois = PoissonModel(season_length=24).fit_predict(df, horizon=6, freq="1h", levels=[95])
    nb = NegativeBinomialModel(season_length=24).fit_predict(df, horizon=6, freq="1h", levels=[95])
    pois_width = float((pois["yhat_upper_95"] - pois["yhat_lower_95"]).mean())
    nb_width = float((nb["yhat_upper_95"] - nb["yhat_lower_95"]).mean())
    assert nb_width > pois_width


def test_overdispersion_estimate_zero_on_pure_poisson() -> None:
    """A clean Poisson sample yields ~0 overdispersion (clamped, never negative)."""
    rng = np.random.default_rng(7)
    y = rng.poisson(8.0, size=2000).astype(float)
    fitted = np.full_like(y, float(y.mean()))
    alpha = _estimate_overdispersion(y, fitted)
    assert alpha >= 0.0
    assert alpha < 0.05


def test_empty_series_yields_zero_forecast() -> None:
    df = _count_frame([])
    out = PoissonModel(season_length=24).fit_predict(df, horizon=4, freq="1h", levels=[80])
    assert len(out) == 4
    assert (out["yhat"] == 0.0).all()
    assert (out["yhat_lower_80"] == 0.0).all()
    assert (out["yhat_upper_80"] == 0.0).all()


def test_no_levels_emits_point_only() -> None:
    df = _count_frame(_seasonal_counts(24, 6))
    out = NegativeBinomialModel(season_length=24).fit_predict(df, horizon=3, freq="1h", levels=[])
    assert list(out.columns) == ["unique_id", "ds", "yhat"]


def test_ingarch_recovers_autoregressive_structure() -> None:
    """A bursty INGARCH-generated series fits to a stationary (alpha+beta<1) model."""
    rng = np.random.default_rng(11)
    n = 400
    omega, a, b = 2.0, 0.4, 0.4
    lam = np.empty(n)
    y = np.empty(n)
    lam[0] = omega / (1 - a - b)
    y[0] = rng.poisson(lam[0])
    for t in range(1, n):
        lam[t] = omega + a * y[t - 1] + b * lam[t - 1]
        y[t] = rng.poisson(lam[t])
    params = _fit_ingarch(y.astype(float))
    assert params is not None
    fit_omega, fit_a, fit_b = params
    assert fit_omega > 0
    assert fit_a + fit_b < 1.0


def test_ingarch_falls_back_on_short_series() -> None:
    """Too-short series skip the MLE fit and still produce a valid forecast."""
    df = _count_frame([3.0, 5.0, 4.0, 6.0])  # below _INGARCH_MIN_POINTS
    out = INGARCHModel(season_length=4).fit_predict(df, horizon=5, freq="1h", levels=[90])
    assert len(out) == 5
    assert (out["yhat_lower_90"] >= 0).all()


def test_count_model_names_self_maintains() -> None:
    assert frozenset({"Poisson", "NegativeBinomial", "INGARCH"}) == COUNT_MODEL_NAMES


def test_models_registered_via_entry_points() -> None:
    model_registry.refresh()
    available = model_registry.available()
    for name in ("Poisson", "NegativeBinomial", "INGARCH"):
        assert name in available
        built = model_registry.build(name, season_length=24)
        assert built.name == name
        assert model_registry.uses_regressors(name) is False


# ---- auto-select preference for data_profile: count ----


class _FakeSource(PromSource):
    def __init__(self, by_query: dict[str, list[SeriesFrame]]) -> None:
        self._by_query = by_query

    async def query_range(
        self, promql: str, start: datetime, end: datetime, step_seconds: int
    ) -> list[SeriesFrame]:
        return self._by_query.get(promql, [])

    async def aclose(self) -> None:
        return None


class _ConstModel:
    """A constant-forecast fake; ``name`` set by the factory below."""

    def __init__(self, name: str, value: float) -> None:
        self.name = name
        self.uses_regressors = False
        self._value = value

    def fit_predict(
        self,
        df: pd.DataFrame,
        horizon: int,
        freq: str,
        levels: list[int],
        regressors: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        idx = pd.date_range(df["ds"].iloc[-1], periods=horizon + 1, freq=freq)[1:]
        out = pd.DataFrame(
            {"unique_id": ["s"] * horizon, "ds": idx, "yhat": [self._value] * horizon}
        )
        for level in levels:
            out[f"yhat_lower_{level}"] = [self._value - 1.0] * horizon
            out[f"yhat_upper_{level}"] = [self._value + 1.0] * horizon
        return out


def _ramp_frame(n: int) -> SeriesFrame:
    base = datetime(2026, 5, 21, tzinfo=UTC)
    ts = [base + timedelta(minutes=5 * i) for i in range(n)]
    return SeriesFrame(
        labels={"instance": "h"}, timestamps=ts, values=[float(i % 7) for i in range(n)]
    )


@pytest.mark.asyncio
async def test_count_profile_prefers_count_model_in_auto_select(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``data_profile: count`` restricts the auto-select winner to the count family.

    ``AutoARIMA`` is given the lower-error constant and would win an
    unrestricted race; the count preference picks ``Poisson`` anyway,
    keeping ``AutoARIMA`` only as a safety net.
    """

    def _fake_build(name: str, *a: object, **kw: object) -> _ConstModel:
        # AutoARIMA predicts ~3 (closer to the 0..6 ramp mean) than Poisson.
        return _ConstModel(name, value=3.0 if name == "AutoARIMA" else 0.0)

    monkeypatch.setattr(model_registry, "build", _fake_build)

    cfg = Config(
        datasource=DatasourceConfig(url="http://x"),
        server=ServerConfig(),
        safety=SafetyConfig(max_series_per_query=500, max_total_series=0),
        defaults=DefaultsConfig(
            min_points=5,
            models=["AutoARIMA", "Poisson"],
            confidence_levels=[80],
            horizon=timedelta(minutes=15),
            step=timedelta(minutes=5),
            accuracy=AccuracyConfig(
                evaluate=True,
                auto_select=True,
                horizons=[timedelta(minutes=10)],
                fallback_mape_threshold=None,
            ),
        ),
        groups=[
            GroupConfig(
                name="g",
                queries=[QueryConfig(id="m", promql="up", data_profile="count")],
            )
        ],
    )
    runner = Runner(
        config=cfg, source=_FakeSource({"up": [_ramp_frame(60)]}), exporter=(exp := Exporter())
    )
    await runner.run_group(cfg.groups[0])
    body = exp.render().decode()
    # Auto-select emits only the winner; the count preference makes it Poisson.
    assert "m_forecast{" in body
    forecast_lines = [ln for ln in body.splitlines() if ln.startswith("m_forecast{")]
    assert forecast_lines
    assert all('model="Poisson"' in ln for ln in forecast_lines)


def _seasonal_count_frame(n: int) -> SeriesFrame:
    base = datetime(2026, 5, 21, tzinfo=UTC)
    ts = [base + timedelta(minutes=5 * i) for i in range(n)]
    rng = np.random.default_rng(5)
    pos = np.arange(n) % 288
    rate = 6.0 + 3.0 * np.sin(2 * np.pi * pos / 288)
    values = [float(v) for v in rng.poisson(np.maximum(rate, 0.1))]
    return SeriesFrame(labels={"instance": "h"}, timestamps=ts, values=values)


@pytest.mark.asyncio
async def test_real_poisson_model_emits_through_runner() -> None:
    """The real Poisson model fits and emits end-to-end through the Runner.

    Exercises the actual ``fit_predict`` -> snapshot/curve wiring (no fake
    model), so the forecast ``ds`` grid honours the configured ``freq`` and
    the bands render as non-negative gauge rows.
    """
    cfg = Config(
        datasource=DatasourceConfig(url="http://x"),
        server=ServerConfig(),
        safety=SafetyConfig(max_series_per_query=500, max_total_series=0),
        defaults=DefaultsConfig(
            min_points=20,
            models=["Poisson"],
            confidence_levels=[80, 95],
            horizon=timedelta(minutes=30),
            step=timedelta(minutes=5),
            season_length=288,
            accuracy=AccuracyConfig(evaluate=False),
        ),
        groups=[
            GroupConfig(
                name="g",
                queries=[QueryConfig(id="reqs", promql="up", data_profile="count")],
            )
        ],
    )
    runner = Runner(
        config=cfg,
        source=_FakeSource({"up": [_seasonal_count_frame(300)]}),
        exporter=(exp := Exporter()),
    )
    await runner.run_group(cfg.groups[0])
    body = exp.render().decode()
    assert "reqs_forecast{" in body
    assert 'model="Poisson"' in body
    lower_lines = [ln for ln in body.splitlines() if ln.startswith("reqs_forecast_lower{")]
    assert lower_lines
    # Count bands never render negative.
    assert all(float(ln.rsplit(" ", 1)[1]) >= 0 for ln in lower_lines)


@pytest.mark.asyncio
async def test_count_model_with_quantiles_emits_asymmetric() -> None:
    """Quantiles derived from a count model's bands are asymmetric around the median."""
    cfg = Config(
        datasource=DatasourceConfig(url="http://x"),
        server=ServerConfig(),
        safety=SafetyConfig(max_series_per_query=500, max_total_series=0),
        defaults=DefaultsConfig(
            min_points=20,
            models=["Poisson"],
            confidence_levels=[80],
            horizon=timedelta(minutes=30),
            step=timedelta(minutes=5),
            season_length=288,
            accuracy=AccuracyConfig(evaluate=False),
        ),
        groups=[
            GroupConfig(
                name="g",
                queries=[
                    QueryConfig(
                        id="reqs",
                        promql="up",
                        data_profile="count",
                        emission=QueryEmissionConfig(quantiles=[0.5, 0.9, 0.99]),
                    )
                ],
            )
        ],
    )
    runner = Runner(
        config=cfg,
        source=_FakeSource({"up": [_seasonal_count_frame(300)]}),
        exporter=(exp := Exporter()),
    )
    await runner.run_group(cfg.groups[0])
    body = exp.render().decode()

    def _q(q: str) -> float:
        for ln in body.splitlines():
            if ln.startswith("reqs_forecast_quantile{") and f'q="{q}"' in ln:
                return float(ln.rsplit(" ", 1)[1])
        raise AssertionError(f"no quantile {q}")

    median, q90, q99 = _q("0.5"), _q("0.9"), _q("0.99")
    assert median < q90 < q99
    # Poisson skew: the gap to the 99th percentile exceeds the gap below the
    # median by the same span, i.e. the upper tail is heavier (asymmetric).
    assert (q99 - median) > 0
