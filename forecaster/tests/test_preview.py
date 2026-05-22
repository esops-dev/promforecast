"""Tests for the ``/forecast/preview`` endpoint and ``preview`` module."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from promforecast import preview
from promforecast.config import (
    AccuracyConfig,
    Config,
    DatasourceConfig,
    DefaultsConfig,
    GroupConfig,
    QueryConfig,
    SafetyConfig,
    ServerConfig,
)
from promforecast.source import PromSource, SeriesFrame


class _StubSource(PromSource):
    def __init__(self, frames: list[SeriesFrame]) -> None:
        self._frames = frames

    async def query_range(
        self,
        promql: str,
        start: datetime,
        end: datetime,
        step_seconds: int,
    ) -> list[SeriesFrame]:
        return list(self._frames)

    async def aclose(self) -> None:
        return None


class _StubModel:
    def __init__(self, season_length: int = 288) -> None:
        self.name = "StubModel"

    def fit_predict(
        self,
        df: pd.DataFrame,
        horizon: int,
        freq: str,
        levels: list[int],
        regressors: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        idx = pd.date_range(df["ds"].iloc[-1], periods=horizon + 1, freq=freq)[1:]
        out = pd.DataFrame({"unique_id": ["s"] * horizon, "ds": idx, "yhat": [1.0] * horizon})
        for level in levels:
            out[f"yhat_lower_{level}"] = [0.5] * horizon
            out[f"yhat_upper_{level}"] = [1.5] * horizon
        return out


def _frame(n: int = 20) -> SeriesFrame:
    base = datetime(2026, 5, 21, tzinfo=UTC)
    timestamps = [base + timedelta(minutes=5 * i) for i in range(n)]
    return SeriesFrame(labels={}, timestamps=timestamps, values=[1.0] * n)


def _config() -> Config:
    return Config(
        datasource=DatasourceConfig(url="http://x"),
        server=ServerConfig(),
        safety=SafetyConfig(),
        defaults=DefaultsConfig(
            min_points=5,
            models=["StubModel"],
            confidence_levels=[80],
            horizon=timedelta(hours=1),
            step=timedelta(minutes=5),
            accuracy=AccuracyConfig(evaluate=False),
        ),
        groups=[GroupConfig(name="g", queries=[QueryConfig(id="m", promql="up")])],
    )


def test_parse_request_minimal() -> None:
    out = preview.parse_request({"query_id": "m"})
    assert out.query_id == "m"
    assert out.regressor_overrides == {}
    assert out.horizon_seconds is None
    assert out.lookback_seconds is None


def test_parse_request_rejects_missing_query_id() -> None:
    with pytest.raises(preview.PreviewError) as exc:
        preview.parse_request({})
    assert exc.value.status_code == 400


def test_parse_request_accepts_duration_strings() -> None:
    out = preview.parse_request({"query_id": "m", "horizon": "30m", "lookback": "1h"})
    assert out.horizon_seconds == 1800.0
    assert out.lookback_seconds == 3600.0


def test_parse_request_accepts_numeric_seconds() -> None:
    out = preview.parse_request({"query_id": "m", "horizon": 60.0})
    assert out.horizon_seconds == 60.0


def test_parse_request_rejects_negative_horizon() -> None:
    with pytest.raises(preview.PreviewError):
        preview.parse_request({"query_id": "m", "horizon": -1})


def test_parse_request_regressor_overrides_scalar_and_list() -> None:
    out = preview.parse_request(
        {
            "query_id": "m",
            "regressor_overrides": {"deploys": 1.0, "load": [0.1, 0.2]},
        }
    )
    assert out.regressor_overrides == {"deploys": 1.0, "load": [0.1, 0.2]}


def test_find_query_returns_query_and_group() -> None:
    cfg = _config()
    query, group_name = preview.find_query(cfg, "m")
    assert query.id == "m"
    assert group_name == "g"


def test_find_query_raises_on_unknown_id() -> None:
    cfg = _config()
    with pytest.raises(preview.PreviewError) as exc:
        preview.find_query(cfg, "nope")
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_run_preview_returns_json_forecast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from promforecast import models as model_registry  # noqa: PLC0415

    monkeypatch.setattr(model_registry, "build", lambda *a, **kw: _StubModel())

    cfg = _config()
    source = _StubSource([_frame(30)])
    request = preview.PreviewRequest(
        query_id="m",
        regressor_overrides={},
        horizon_seconds=None,
        lookback_seconds=None,
    )
    result = await preview.run_preview(config=cfg, source=source, request=request, served_by="test")
    assert result["query_id"] == "m"
    assert result["group"] == "g"
    assert result["model"] == "StubModel"
    assert result["served_by"] == "test"
    assert isinstance(result["yhat"], list)
    assert result["yhat"]
    assert "80" in result["bands"]
    assert result["bands"]["80"]["lower"]
    assert result["bands"]["80"]["upper"]


@pytest.mark.asyncio
async def test_run_preview_unknown_query_raises_404() -> None:
    cfg = _config()
    source = _StubSource([])
    request = preview.PreviewRequest(
        query_id="missing",
        regressor_overrides={},
        horizon_seconds=None,
        lookback_seconds=None,
    )
    with pytest.raises(preview.PreviewError) as exc:
        await preview.run_preview(config=cfg, source=source, request=request, served_by="test")
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_run_preview_empty_response_returns_no_data() -> None:
    cfg = _config()
    source = _StubSource([])  # source returns no series at all
    request = preview.PreviewRequest(
        query_id="m",
        regressor_overrides={},
        horizon_seconds=None,
        lookback_seconds=None,
    )
    with pytest.raises(preview.PreviewError) as exc:
        await preview.run_preview(config=cfg, source=source, request=request, served_by="test")
    assert exc.value.reason == "no_data"
