"""Runner-level integration tests for the arbitrary-quantile emission family."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from promforecast.config import (
    AccuracyConfig,
    Config,
    ConformalConfig,
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


class _WideBandModel:
    """Constant forecast with a per-level half-width equal to the level/10.

    A higher level => a wider band, so the derived upper quantiles are
    strictly increasing in q — easy to assert the mapping end to end.
    """

    def __init__(self, season_length: int = 288) -> None:
        self.name = "WideBand"
        self.uses_regressors = False

    def fit_predict(
        self,
        df: pd.DataFrame,
        horizon: int,
        freq: str,
        levels: list[int],
        regressors: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        idx = pd.date_range(df["ds"].iloc[-1], periods=horizon + 1, freq=freq)[1:]
        out = pd.DataFrame({"unique_id": ["s"] * horizon, "ds": idx, "yhat": [100.0] * horizon})
        for level in levels:
            half = level / 10.0
            out[f"yhat_lower_{level}"] = [100.0 - half] * horizon
            out[f"yhat_upper_{level}"] = [100.0 + half] * horizon
        return out


def _frame(n: int) -> SeriesFrame:
    base = datetime(2026, 5, 21, tzinfo=UTC)
    timestamps = [base + timedelta(minutes=5 * i) for i in range(n)]
    return SeriesFrame(labels={"instance": "host-a"}, timestamps=timestamps, values=[1.0] * n)


def _config(query: QueryConfig, *, conformal: bool = False) -> Config:
    return Config(
        datasource=DatasourceConfig(url="http://x"),
        server=ServerConfig(),
        safety=SafetyConfig(),
        defaults=DefaultsConfig(
            min_points=5,
            models=["WideBand"],
            confidence_levels=[80],
            horizon=timedelta(minutes=30),
            step=timedelta(minutes=5),
            accuracy=AccuracyConfig(
                evaluate=False,
                conformal=ConformalConfig(enabled=conformal, holdout_fraction=0.3),
            ),
        ),
        groups=[GroupConfig(name="g", queries=[query])],
    )


@pytest.mark.asyncio
async def test_quantiles_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    from promforecast import models as model_registry  # noqa: PLC0415

    monkeypatch.setattr(model_registry, "build", lambda *a, **kw: _WideBandModel())
    cfg = _config(QueryConfig(id="m", promql="up"))
    runner = Runner(
        config=cfg, source=_FakeSource({"up": [_frame(20)]}), exporter=(exp := Exporter())
    )
    await runner.run_group(cfg.groups[0])
    assert "m_forecast_quantile" not in exp.render().decode()


@pytest.mark.asyncio
async def test_quantile_family_emitted_with_correct_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from promforecast import models as model_registry  # noqa: PLC0415

    monkeypatch.setattr(model_registry, "build", lambda *a, **kw: _WideBandModel())
    cfg = _config(
        QueryConfig(
            id="m",
            promql="up",
            emission=QueryEmissionConfig(quantiles=[0.5, 0.9, 0.95, 0.99]),
        )
    )
    runner = Runner(
        config=cfg, source=_FakeSource({"up": [_frame(20)]}), exporter=(exp := Exporter())
    )
    await runner.run_group(cfg.groups[0])
    body = exp.render().decode()
    assert "m_forecast_quantile" in body
    # q=0.5 -> yhat=100; q=0.9 -> upper_80 (100+8); q=0.95 -> upper_90 (100+9);
    # q=0.99 -> upper_98 (100+9.8). Each appears as a labelled sample.
    assert "m_forecast_quantile{" in body
    assert 'q="0.5"' in body
    assert 'q="0.9"' in body
    assert 'q="0.99"' in body
    # The standard band family is unchanged (only confidence_levels=[80]).
    assert "m_forecast_lower{" in body
    # No level-90/98 band rows leaked into the band family.
    assert 'm_forecast_upper{group="g",instance="host-a",level="90"' not in body


def _quantile_value(body: str, q: str) -> float:
    for line in body.splitlines():
        if line.startswith("m_forecast_quantile{") and f'q="{q}"' in line:
            return float(line.rsplit(" ", 1)[1])
    raise AssertionError(f"no quantile row for q={q}")


@pytest.mark.asyncio
async def test_quantiles_are_monotonic_in_q(monkeypatch: pytest.MonkeyPatch) -> None:
    from promforecast import models as model_registry  # noqa: PLC0415

    monkeypatch.setattr(model_registry, "build", lambda *a, **kw: _WideBandModel())
    cfg = _config(
        QueryConfig(
            id="m", promql="up", emission=QueryEmissionConfig(quantiles=[0.5, 0.9, 0.95, 0.99])
        )
    )
    runner = Runner(
        config=cfg, source=_FakeSource({"up": [_frame(20)]}), exporter=(exp := Exporter())
    )
    await runner.run_group(cfg.groups[0])
    body = exp.render().decode()
    v50 = _quantile_value(body, "0.5")
    v90 = _quantile_value(body, "0.9")
    v95 = _quantile_value(body, "0.95")
    v99 = _quantile_value(body, "0.99")
    assert v50 < v90 < v95 < v99


@pytest.mark.asyncio
async def test_quantile_conformal_offset_emitted(monkeypatch: pytest.MonkeyPatch) -> None:
    """With conformal on, the band-calibration gauge carries a ``q`` label."""
    from promforecast import models as model_registry  # noqa: PLC0415

    monkeypatch.setattr(model_registry, "build", lambda *a, **kw: _WideBandModel())
    cfg = _config(
        QueryConfig(id="m", promql="up", emission=QueryEmissionConfig(quantiles=[0.9, 0.95])),
        conformal=True,
    )
    runner = Runner(
        config=cfg, source=_FakeSource({"up": [_frame(40)]}), exporter=(exp := Exporter())
    )
    await runner.run_group(cfg.groups[0])
    body = exp.render().decode()
    # The per-quantile offset lands in the existing band-calibration family
    # with a ``q`` label rather than ``level``.
    assert "forecast_band_calibration_offset{" in body
    assert any(
        line.startswith("forecast_band_calibration_offset{") and 'q="0.9"' in line
        for line in body.splitlines()
    )
