"""Tests for per-query ``step`` and ``season_length`` overrides.

The runner has historically read ``step`` and ``season_length`` from
``defaults`` only; the v0.3 user story adds per-query overrides so a single
forecaster can mix metrics with very different cadences. These tests pin
the schema (parsing + validation) and the runner wiring (the override
actually flows into the query/fit calls).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

import promforecast.config as config_module
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
    effective_season_length,
    effective_step,
)
from promforecast.exporter import Exporter
from promforecast.runner import Runner
from promforecast.source import PromSource, SeriesFrame


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(body)
    return path


# -- schema --------------------------------------------------------------


def test_query_step_override_parses(tmp_path: Path) -> None:
    cfg = config_module.load(
        _write(
            tmp_path,
            """
datasource:
  url: http://vm:8428/
defaults:
  step: 5m
  season_length: 288
groups:
  - name: g
    queries:
      - id: hourly_metric
        promql: up
        step: 1h
        season_length: 24
""",
        )
    )
    q = cfg.groups[0].queries[0]
    assert q.step == timedelta(hours=1)
    assert q.season_length == 24


def test_query_season_length_auto_parses(tmp_path: Path) -> None:
    cfg = config_module.load(
        _write(
            tmp_path,
            """
datasource:
  url: http://vm:8428/
groups:
  - name: g
    queries:
      - id: m
        promql: up
        step: 30m
        season_length: auto
""",
        )
    )
    assert cfg.groups[0].queries[0].season_length == "auto"


def test_query_rejects_zero_step(tmp_path: Path) -> None:
    with pytest.raises((ValueError, ValidationError)):
        config_module.load(
            _write(
                tmp_path,
                """
datasource:
  url: http://vm:8428/
groups:
  - name: g
    queries:
      - id: m
        promql: up
        step: 0s
""",
            )
        )


def test_query_rejects_zero_season_length(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="season_length must be a positive integer"):
        config_module.load(
            _write(
                tmp_path,
                """
datasource:
  url: http://vm:8428/
groups:
  - name: g
    queries:
      - id: m
        promql: up
        season_length: 0
""",
            )
        )


def test_query_rejects_unknown_season_length_string(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        config_module.load(
            _write(
                tmp_path,
                """
datasource:
  url: http://vm:8428/
groups:
  - name: g
    queries:
      - id: m
        promql: up
        season_length: weekly
""",
            )
        )


# -- effective_* helpers -------------------------------------------------


def test_effective_step_falls_back_to_defaults() -> None:
    defaults = DefaultsConfig(step=timedelta(minutes=5), season_length=288)
    q = QueryConfig(id="m", promql="up")
    assert effective_step(q, defaults) == timedelta(minutes=5)


def test_effective_step_uses_query_override() -> None:
    defaults = DefaultsConfig(step=timedelta(minutes=5), season_length=288)
    q = QueryConfig(id="m", promql="up", step=timedelta(hours=1))
    assert effective_step(q, defaults) == timedelta(hours=1)


def test_effective_season_length_falls_back_to_defaults() -> None:
    defaults = DefaultsConfig(step=timedelta(minutes=5), season_length=288)
    q = QueryConfig(id="m", promql="up")
    assert effective_season_length(q, defaults) == 288


def test_effective_season_length_uses_query_override() -> None:
    defaults = DefaultsConfig(step=timedelta(minutes=5), season_length=288)
    q = QueryConfig(id="m", promql="up", season_length=24)
    assert effective_season_length(q, defaults) == 24


def test_effective_season_length_auto_at_5min_step_is_288() -> None:
    defaults = DefaultsConfig(step=timedelta(minutes=5), season_length=288)
    q = QueryConfig(id="m", promql="up", step=timedelta(minutes=5), season_length="auto")
    assert effective_season_length(q, defaults) == 288


def test_effective_season_length_auto_at_1h_step_is_24() -> None:
    defaults = DefaultsConfig(step=timedelta(minutes=5), season_length=288)
    q = QueryConfig(id="m", promql="up", step=timedelta(hours=1), season_length="auto")
    assert effective_season_length(q, defaults) == 24


def test_effective_season_length_auto_at_daily_step_falls_back_to_weekly() -> None:
    defaults = DefaultsConfig(step=timedelta(minutes=5), season_length=288)
    q = QueryConfig(id="m", promql="up", step=timedelta(days=1), season_length="auto")
    assert effective_season_length(q, defaults) == 7


# -- runner wiring -------------------------------------------------------


class _FakeSource(PromSource):
    """Records every (promql, step_seconds) tuple. Bypasses real init."""

    def __init__(self, by_query: dict[str, list[SeriesFrame]] | None = None) -> None:
        self._by_query = by_query or {}
        self.calls: list[tuple[str, int]] = []

    async def query_range(
        self,
        promql: str,
        start: datetime,
        end: datetime,
        step_seconds: int,
    ) -> list[SeriesFrame]:
        self.calls.append((promql, step_seconds))
        return self._by_query.get(promql, [])

    async def aclose(self) -> None:
        return None


class _RecordingFake:
    """Records season_length / freq it was built with so a test can assert
    that per-query overrides reach the model registry call."""

    def __init__(self, name: str, *, season_length: int) -> None:
        self.name = name
        self.season_length = season_length

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


def _hourly_frame(n: int) -> SeriesFrame:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    return SeriesFrame(
        labels={"instance": "host-a"},
        timestamps=[start + timedelta(hours=i) for i in range(n)],
        values=[float(i) for i in range(n)],
    )


@pytest.mark.asyncio
async def test_runner_uses_per_query_step_for_query_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The TSDB query must be executed at the per-query step, not defaults."""
    from promforecast import models as model_registry  # noqa: PLC0415

    captured: dict[str, int] = {}

    def _build(name: str, season_length: int, params: dict | None = None) -> _RecordingFake:
        captured["season_length"] = season_length
        return _RecordingFake(name, season_length=season_length)

    monkeypatch.setattr(model_registry, "build", _build)

    cfg = Config(
        datasource=DatasourceConfig(url="http://x"),
        server=ServerConfig(),
        safety=SafetyConfig(),
        defaults=DefaultsConfig(
            min_points=10,
            models=["SeasonalNaive"],
            step=timedelta(minutes=5),
            season_length=288,
            confidence_levels=[80],
            accuracy=AccuracyConfig(evaluate=False, horizons=[]),
            emission=EmissionConfig(deviation=False, quality=False),
        ),
        groups=[
            GroupConfig(
                name="g",
                queries=[
                    QueryConfig(
                        id="m",
                        promql="up",
                        step=timedelta(hours=1),
                        season_length=24,
                    ),
                ],
            ),
        ],
    )

    source = _FakeSource({"up": [_hourly_frame(60)]})
    runner = Runner(config=cfg, source=source, exporter=Exporter())
    await runner.run_group(cfg.groups[0])

    assert source.calls == [("up", 3600)], "per-query step must drive the TSDB query"
    assert captured["season_length"] == 24


@pytest.mark.asyncio
async def test_runner_falls_back_to_defaults_when_no_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Queries without overrides still pick up the cascaded defaults."""
    from promforecast import models as model_registry  # noqa: PLC0415

    captured: dict[str, int] = {}

    def _build(name: str, season_length: int, params: dict | None = None) -> _RecordingFake:
        captured["season_length"] = season_length
        return _RecordingFake(name, season_length=season_length)

    monkeypatch.setattr(model_registry, "build", _build)

    cfg = Config(
        datasource=DatasourceConfig(url="http://x"),
        server=ServerConfig(),
        safety=SafetyConfig(),
        defaults=DefaultsConfig(
            min_points=10,
            models=["SeasonalNaive"],
            step=timedelta(minutes=5),
            season_length=288,
            confidence_levels=[80],
            accuracy=AccuracyConfig(evaluate=False, horizons=[]),
            emission=EmissionConfig(deviation=False, quality=False),
        ),
        groups=[
            GroupConfig(name="g", queries=[QueryConfig(id="m", promql="up")]),
        ],
    )

    base = datetime(2024, 1, 1, tzinfo=UTC)
    frame = SeriesFrame(
        labels={"instance": "host-a"},
        timestamps=[base + timedelta(minutes=5 * i) for i in range(60)],
        values=[float(i) for i in range(60)],
    )
    source = _FakeSource({"up": [frame]})
    runner = Runner(config=cfg, source=source, exporter=Exporter())
    await runner.run_group(cfg.groups[0])

    assert source.calls == [("up", 300)], "default step (5m) must drive the TSDB query"
    assert captured["season_length"] == 288
