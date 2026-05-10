"""Tests for the per-regressor contribution metric.

Two layers:

* The pure-function ``compute_contributions`` (correlation -> normalised
  share). These cover constant inputs, NaN handling, and the multi-
  regressor sum-to-1 invariant without touching the runner.
* The end-to-end runner wiring: with regressors configured, the snapshot
  rendered to /metrics carries one ``<id>_forecast_contribution_ratio``
  sample per regressor under the same labels as the forecast.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
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
from promforecast.contribution import compute_contributions
from promforecast.exporter import Exporter
from promforecast.runner import Runner
from promforecast.source import PromSource, SeriesFrame

# -- pure helper -----------------------------------------------------------


def test_compute_contributions_returns_empty_for_no_regressors() -> None:
    assert compute_contributions(y=[1.0, 2.0, 3.0], regressors=None) == []
    assert compute_contributions(y=[1.0, 2.0, 3.0], regressors=pd.DataFrame()) == []


def test_compute_contributions_normalises_to_sum_of_one() -> None:
    """Two regressors with non-zero correlation: their ratios must sum to 1."""
    y = [1.0, 2.0, 3.0, 4.0, 5.0]
    regs = pd.DataFrame(
        {
            "ds": pd.date_range("2024-01-01", periods=5, freq="h"),
            "perfect": [1.0, 2.0, 3.0, 4.0, 5.0],
            "anti": [5.0, 4.0, 3.0, 2.0, 1.0],
        }
    )
    out = compute_contributions(y=y, regressors=regs)
    by_id = {c.regressor_id: c.ratio for c in out}
    assert "perfect" in by_id and "anti" in by_id
    assert pytest.approx(sum(by_id.values()), abs=1e-6) == 1.0
    # Both correlations are |1.0| absolute, so equal share.
    assert by_id["perfect"] == pytest.approx(0.5)
    assert by_id["anti"] == pytest.approx(0.5)


def test_compute_contributions_zero_for_constant_regressor() -> None:
    """A constant column has undefined Pearson; contribution drops to 0."""
    y = [1.0, 2.0, 3.0, 4.0]
    regs = pd.DataFrame(
        {
            "ds": pd.date_range("2024-01-01", periods=4, freq="h"),
            "useful": [1.0, 2.0, 3.0, 4.0],
            "constant": [7.0, 7.0, 7.0, 7.0],
        }
    )
    out = compute_contributions(y=y, regressors=regs)
    by_id = {c.regressor_id: c.ratio for c in out}
    assert by_id["useful"] == pytest.approx(1.0)
    assert by_id["constant"] == 0.0


def test_compute_contributions_emits_zeros_when_every_regressor_constant() -> None:
    """Two constant regressors: both ratios are 0 (not NaN, not 0.5).

    /metrics needs a sample even when no regressor produced a usable
    correlation; alerts on "contribution dropped to 0" depend on the
    series being present rather than missing.
    """
    y = [1.0, 2.0, 3.0]
    regs = pd.DataFrame(
        {
            "ds": pd.date_range("2024-01-01", periods=3, freq="h"),
            "a": [1.0, 1.0, 1.0],
            "b": [2.0, 2.0, 2.0],
        }
    )
    out = compute_contributions(y=y, regressors=regs)
    assert {c.regressor_id: c.ratio for c in out} == {"a": 0.0, "b": 0.0}


# -- end-to-end runner wiring ---------------------------------------------


class _FakeSource(PromSource):
    def __init__(self, by_query: dict[str, list[SeriesFrame]] | None = None) -> None:
        self._by_query = by_query or {}

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


class _FakeModel:
    def __init__(self, name: str) -> None:
        self.name = name

    def fit_predict(
        self,
        df,  # type: ignore[no-untyped-def]
        horizon: int,
        freq: str,
        levels: list[int],
        regressors=None,  # type: ignore[no-untyped-def]
    ):  # type: ignore[no-untyped-def]
        import pandas as pd  # noqa: PLC0415

        out = pd.DataFrame(
            {
                "unique_id": ["s"] * horizon,
                "ds": pd.date_range(df["ds"].iloc[-1], periods=horizon + 1, freq=freq)[1:],
                "yhat": [1.0] * horizon,
            }
        )
        for level in levels:
            out[f"yhat_lower_{level}"] = 0.0
            out[f"yhat_upper_{level}"] = 2.0
        return out


def _ramp_frame(n: int) -> SeriesFrame:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    return SeriesFrame(
        labels={"instance": "host-a"},
        timestamps=[start + timedelta(minutes=5 * i) for i in range(n)],
        values=[float(i) for i in range(n)],
    )


def _build_cfg(*, contribution_enabled: bool = True) -> Config:
    return Config(
        datasource=DatasourceConfig(url="http://x"),
        server=ServerConfig(),
        safety=SafetyConfig(),
        defaults=DefaultsConfig(
            min_points=10,
            models=[ModelSpec(name="SeasonalNaive")],
            confidence_levels=[80],
            accuracy=AccuracyConfig(evaluate=False, horizons=[]),
            emission=EmissionConfig(
                deviation=False,
                quality=False,
                contribution=contribution_enabled,
            ),
        ),
        groups=[
            GroupConfig(
                name="g",
                queries=[
                    QueryConfig(
                        id="m",
                        promql="up",
                        regressors=[
                            RegressorConfig(id="hour_of_day", type="calendar"),
                            RegressorConfig(id="day_of_week", type="calendar"),
                        ],
                    ),
                ],
            ),
        ],
    )


@pytest.mark.asyncio
async def test_runner_emits_contribution_metric_for_calendar_regressors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from promforecast import models as model_registry  # noqa: PLC0415

    monkeypatch.setattr(
        model_registry, "build", lambda name, season_length, params=None: _FakeModel(name)
    )

    cfg = _build_cfg()
    exporter = Exporter()
    runner = Runner(config=cfg, source=_FakeSource({"up": [_ramp_frame(60)]}), exporter=exporter)
    await runner.run_group(cfg.groups[0])

    body = exporter.render().decode()
    assert "# TYPE m_forecast_contribution_ratio gauge" in body
    assert 'regressor="hour_of_day"' in body
    assert 'regressor="day_of_week"' in body
    # Forecast itself must still be emitted alongside the contribution.
    assert "m_forecast{" in body


@pytest.mark.asyncio
async def test_runner_suppresses_contribution_when_emission_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from promforecast import models as model_registry  # noqa: PLC0415

    monkeypatch.setattr(
        model_registry, "build", lambda name, season_length, params=None: _FakeModel(name)
    )

    cfg = _build_cfg(contribution_enabled=False)
    exporter = Exporter()
    runner = Runner(config=cfg, source=_FakeSource({"up": [_ramp_frame(60)]}), exporter=exporter)
    await runner.run_group(cfg.groups[0])

    body = exporter.render().decode()
    assert "m_forecast_contribution_ratio" not in body


@pytest.mark.asyncio
async def test_runner_skips_contribution_when_no_regressors_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No regressors -> no contribution metric, even when emission.contribution=true."""
    from promforecast import models as model_registry  # noqa: PLC0415

    monkeypatch.setattr(
        model_registry, "build", lambda name, season_length, params=None: _FakeModel(name)
    )

    cfg = Config(
        datasource=DatasourceConfig(url="http://x"),
        server=ServerConfig(),
        safety=SafetyConfig(),
        defaults=DefaultsConfig(
            min_points=10,
            models=[ModelSpec(name="SeasonalNaive")],
            confidence_levels=[80],
            accuracy=AccuracyConfig(evaluate=False, horizons=[]),
            emission=EmissionConfig(deviation=False, quality=False, contribution=True),
        ),
        groups=[
            GroupConfig(name="g", queries=[QueryConfig(id="m", promql="up")]),
        ],
    )
    exporter = Exporter()
    runner = Runner(config=cfg, source=_FakeSource({"up": [_ramp_frame(60)]}), exporter=exporter)
    await runner.run_group(cfg.groups[0])

    body = exporter.render().decode()
    assert "m_forecast_contribution_ratio" not in body
