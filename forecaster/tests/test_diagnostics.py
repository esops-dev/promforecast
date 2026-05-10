"""Per-group diagnostics: fit duration, slow-fit counter, timeout reasons.

These are the metrics operators reach for when a group is misbehaving:
the deviation from the slow-fit threshold tells them *which* model/query
is slow, the timeout-vs-error split tells them *why* a fit failed.
"""

from __future__ import annotations

from datetime import UTC, datetime
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
from promforecast.exporter import Exporter
from promforecast.runner import Runner
from promforecast.source import PromSource, SeriesFrame

if TYPE_CHECKING:
    pass


class _Source(PromSource):
    """Returns canned frames; bypasses the real PromSource constructor."""

    def __init__(self, frames: list[SeriesFrame]) -> None:
        self._frames = frames

    async def query_range(
        self,
        promql: str,
        start: datetime,
        end: datetime,
        step_seconds: int,
    ) -> list[SeriesFrame]:
        return self._frames

    async def aclose(self) -> None:
        return None


class _TimingOutSource(PromSource):
    """Always raises asyncio.TimeoutError to simulate a slow TSDB."""

    def __init__(self) -> None: ...

    async def query_range(
        self,
        promql: str,
        start: datetime,
        end: datetime,
        step_seconds: int,
    ) -> list[SeriesFrame]:
        raise TimeoutError

    async def aclose(self) -> None:
        return None


def _ramp_frame(n: int) -> SeriesFrame:
    from datetime import timedelta as _td  # noqa: PLC0415

    start = datetime(2024, 1, 1, tzinfo=UTC)
    return SeriesFrame(
        labels={"instance": "host-a"},
        timestamps=[start + _td(minutes=5 * i) for i in range(n)],
        values=[float(i) for i in range(n)],
    )


def _config() -> Config:
    return Config(
        datasource=DatasourceConfig(url="http://x"),
        server=ServerConfig(),
        safety=SafetyConfig(),
        defaults=DefaultsConfig(
            min_points=10,
            models=["SeasonalNaive"],
            confidence_levels=[80],
            accuracy=AccuracyConfig(evaluate=False, horizons=[]),
            emission=EmissionConfig(deviation=False, quality=False),
        ),
        groups=[GroupConfig(name="g", queries=[QueryConfig(id="m", promql="up")])],
    )


@pytest.mark.asyncio
async def test_run_group_records_fit_duration_metric(monkeypatch: pytest.MonkeyPatch) -> None:
    """A successful run should expose a fit duration sample for the model."""
    from promforecast import models as model_registry  # noqa: PLC0415

    class _Fake:
        def __init__(self, *_: object, **__: object) -> None: ...

        def fit_predict(
            self,
            df: object,
            horizon: int,
            freq: str,
            levels: list[int],
            regressors: object | None = None,
        ) -> object:
            import pandas as pd  # noqa: PLC0415

            out = pd.DataFrame(
                {
                    "unique_id": ["s"] * horizon,
                    "ds": pd.date_range("2024-01-02", periods=horizon, freq=freq),
                    "yhat": [1.0] * horizon,
                }
            )
            for level in levels:
                out[f"yhat_lower_{level}"] = 0.0
                out[f"yhat_upper_{level}"] = 2.0
            return out

    monkeypatch.setattr(model_registry, "build", lambda name, season_length, params=None: _Fake())

    cfg = _config()
    exporter = Exporter()
    runner = Runner(config=cfg, source=_Source([_ramp_frame(20)]), exporter=exporter)
    await runner.run_group(cfg.groups[0])

    body = exporter.render().decode()
    assert "# TYPE forecast_fit_duration_seconds gauge" in body
    assert 'group="g"' in body
    assert 'query="m"' in body
    assert 'model="SeasonalNaive"' in body


@pytest.mark.asyncio
async def test_query_timeout_uses_distinct_failure_reason() -> None:
    cfg = _config()
    cfg.safety.query_timeout = type(cfg.safety.query_timeout)(seconds=1)
    exporter = Exporter()
    runner = Runner(config=cfg, source=_TimingOutSource(), exporter=exporter)
    await runner.run_group(cfg.groups[0])

    body = exporter.render().decode()
    assert 'reason="query_timeout"' in body
    assert 'reason="query_error"' not in body


@pytest.mark.asyncio
async def test_slow_fit_threshold_increments_counter(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fit that crosses 0.5 * fit_timeout should bump the slow-fit counter."""
    from datetime import timedelta  # noqa: PLC0415

    from promforecast import models as model_registry  # noqa: PLC0415
    from promforecast import runner as runner_module  # noqa: PLC0415

    class _SlowFake:
        def fit_predict(
            self,
            df: object,
            horizon: int,
            freq: str,
            levels: list[int],
            regressors: object | None = None,
        ) -> object:
            import pandas as pd  # noqa: PLC0415

            out = pd.DataFrame(
                {
                    "unique_id": ["s"] * horizon,
                    "ds": pd.date_range("2024-01-02", periods=horizon, freq=freq),
                    "yhat": [1.0] * horizon,
                }
            )
            for level in levels:
                out[f"yhat_lower_{level}"] = 0.0
                out[f"yhat_upper_{level}"] = 2.0
            return out

    monkeypatch.setattr(
        model_registry, "build", lambda name, season_length, params=None: _SlowFake()
    )

    cfg = _config()
    # Make the slow-fit threshold trivially crossable: any fit over 0s is
    # slow. We deliberately tweak the module-level constant rather than
    # adding a config knob — the threshold is an internal heuristic, not
    # part of the user-facing schema.
    cfg.safety.fit_timeout = timedelta(seconds=5)
    # Threshold = 0 → every fit is "slow" without ever actually timing out.
    monkeypatch.setattr(runner_module, "SLOW_FIT_RATIO", 0.0)

    exporter = Exporter()
    runner = Runner(config=cfg, source=_Source([_ramp_frame(20)]), exporter=exporter)
    await runner.run_group(cfg.groups[0])

    body = exporter.render().decode()
    assert "# TYPE forecast_slow_fits_total counter" in body
    assert 'group="g"' in body
    assert 'query="m"' in body
    assert 'model="SeasonalNaive"' in body


@pytest.mark.asyncio
async def test_slow_fits_counter_prunes_removed_query_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A query removed by a config reload must drop its label set.

    The counter is merged across runs to stay monotonic, but when a query
    disappears from the group's config the corresponding (query, model)
    label set should also disappear. Otherwise a config edit that drops a
    query would leak its old slow_fits series in /metrics forever.
    """
    from datetime import timedelta  # noqa: PLC0415

    from promforecast import models as model_registry  # noqa: PLC0415
    from promforecast import runner as runner_module  # noqa: PLC0415

    class _Fake:
        def fit_predict(
            self,
            df: object,
            horizon: int,
            freq: str,
            levels: list[int],
            regressors: object | None = None,
        ) -> object:
            import pandas as pd  # noqa: PLC0415

            out = pd.DataFrame(
                {
                    "unique_id": ["s"] * horizon,
                    "ds": pd.date_range("2024-01-02", periods=horizon, freq=freq),
                    "yhat": [1.0] * horizon,
                }
            )
            for level in levels:
                out[f"yhat_lower_{level}"] = 0.0
                out[f"yhat_upper_{level}"] = 2.0
            return out

    monkeypatch.setattr(model_registry, "build", lambda name, season_length, params=None: _Fake())
    monkeypatch.setattr(runner_module, "SLOW_FIT_RATIO", 0.0)

    cfg = Config(
        datasource=DatasourceConfig(url="http://x"),
        server=ServerConfig(),
        safety=SafetyConfig(),
        defaults=DefaultsConfig(
            min_points=10,
            models=["SeasonalNaive"],
            confidence_levels=[80],
            accuracy=AccuracyConfig(evaluate=False, horizons=[]),
            emission=EmissionConfig(deviation=False, quality=False),
        ),
        groups=[
            GroupConfig(
                name="g",
                queries=[
                    QueryConfig(id="m", promql="up"),
                    QueryConfig(id="m_old", promql="rate(up[5m])"),
                ],
            )
        ],
    )
    cfg.safety.fit_timeout = timedelta(seconds=5)

    exporter = Exporter()
    runner = Runner(config=cfg, source=_Source([_ramp_frame(20)]), exporter=exporter)

    # First run: both queries are present, both record slow fits.
    await runner.run_group(cfg.groups[0])
    body = exporter.render().decode()
    assert 'query="m"' in body
    assert 'query="m_old"' in body

    # Drop ``m_old`` and re-run; pruning must remove its label combo.
    cfg.groups[0].queries.pop()
    await runner.run_group(cfg.groups[0])

    body = exporter.render().decode()
    assert 'query="m"' in body
    assert 'forecast_slow_fits_total{group="g",model="SeasonalNaive",query="m_old"}' not in body


@pytest.mark.asyncio
async def test_slow_fits_counter_is_monotonic_across_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The slow-fit counter must merge with the previous snapshot so
    rate() doesn't see a counter reset every run."""
    from datetime import timedelta  # noqa: PLC0415

    from promforecast import models as model_registry  # noqa: PLC0415
    from promforecast import runner as runner_module  # noqa: PLC0415

    class _Fake:
        def fit_predict(
            self,
            df: object,
            horizon: int,
            freq: str,
            levels: list[int],
            regressors: object | None = None,
        ) -> object:
            import pandas as pd  # noqa: PLC0415

            out = pd.DataFrame(
                {
                    "unique_id": ["s"] * horizon,
                    "ds": pd.date_range("2024-01-02", periods=horizon, freq=freq),
                    "yhat": [1.0] * horizon,
                }
            )
            for level in levels:
                out[f"yhat_lower_{level}"] = 0.0
                out[f"yhat_upper_{level}"] = 2.0
            return out

    monkeypatch.setattr(model_registry, "build", lambda name, season_length, params=None: _Fake())

    cfg = _config()
    cfg.safety.fit_timeout = timedelta(seconds=5)
    # Threshold = 0 → every fit is "slow" without ever actually timing out.
    monkeypatch.setattr(runner_module, "SLOW_FIT_RATIO", 0.0)

    exporter = Exporter()
    runner = Runner(config=cfg, source=_Source([_ramp_frame(20)]), exporter=exporter)
    await runner.run_group(cfg.groups[0])
    await runner.run_group(cfg.groups[0])

    body = exporter.render().decode()
    # Two runs, each one slow fit → counter at 2.0.
    assert 'forecast_slow_fits_total{group="g",model="SeasonalNaive",query="m"} 2.0' in body
