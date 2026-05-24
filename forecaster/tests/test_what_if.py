"""Tests for the fleet what-if endpoint and the ``what_if`` module."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from promforecast import preview, what_if
from promforecast.config import (
    AccuracyConfig,
    Config,
    DatasourceConfig,
    DefaultsConfig,
    GroupConfig,
    QueryConfig,
    RegressorConfig,
    SafetyConfig,
    ServerConfig,
    ThresholdConfig,
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
    """Forecast model that produces a deterministic rising series.

    Slope is steep enough that the configured threshold (5.0) is
    breached within the horizon under the default multiplier; useful
    for the ranking-by-eta tests.
    """

    def __init__(self, season_length: int = 288, slope: float = 0.1) -> None:
        self.name = "StubModel"
        self._slope = slope

    def fit_predict(
        self,
        df: pd.DataFrame,
        horizon: int,
        freq: str,
        levels: list[int],
        regressors: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        # Multiplier amplifies slope: when the what-if endpoint overrides
        # a traffic regressor the slope rises by the override's mean
        # value. This is what makes traffic_multiplier > 1.0 actually
        # change the predicted ETA.
        mult = 1.0
        if regressors is not None and "traffic" in regressors.columns:
            mult = float(regressors["traffic"].iloc[-1])
        idx = pd.date_range(df["ds"].iloc[-1], periods=horizon + 1, freq=freq)[1:]
        yhat = [(i + 1) * self._slope * mult for i in range(horizon)]
        out = pd.DataFrame({"unique_id": ["s"] * horizon, "ds": idx, "yhat": yhat})
        for level in levels:
            out[f"yhat_lower_{level}"] = [v * 0.9 for v in yhat]
            out[f"yhat_upper_{level}"] = [v * 1.1 for v in yhat]
        return out


def _frame(n: int = 30) -> SeriesFrame:
    base = datetime(2026, 5, 21, tzinfo=UTC)
    timestamps = [base + timedelta(minutes=5 * i) for i in range(n)]
    return SeriesFrame(labels={}, timestamps=timestamps, values=[1.0] * n)


def _config(
    *,
    queries: list[QueryConfig] | None = None,
    extra_group: GroupConfig | None = None,
) -> Config:
    queries = queries or [
        # Threshold deliberately below the stub model's reachable values
        # so both queries produce a finite ETA within the configured
        # 2-hour horizon (24 steps * 0.1 slope = up to 2.4 at multiplier
        # 1.0, up to 4.8 at multiplier 2.0). The fast_metric ranks
        # first because the traffic-regressor override doubles its slope.
        QueryConfig(
            id="slow_metric",
            promql="up{job='a'}",
            thresholds=[ThresholdConfig(name="full", value=2.0, comparator="gt")],
        ),
        QueryConfig(
            id="fast_metric",
            promql="up{job='b'}",
            thresholds=[ThresholdConfig(name="full", value=2.0, comparator="gt")],
            regressors=[RegressorConfig(id="traffic", promql="up")],
        ),
    ]
    groups = [GroupConfig(name="g", queries=queries)]
    if extra_group is not None:
        groups.append(extra_group)
    return Config(
        datasource=DatasourceConfig(url="http://x"),
        server=ServerConfig(expose_what_if_endpoint=True),
        safety=SafetyConfig(fleet_what_if_min_interval=timedelta(seconds=0)),
        defaults=DefaultsConfig(
            min_points=5,
            models=["StubModel"],
            confidence_levels=[80],
            horizon=timedelta(hours=2),
            step=timedelta(minutes=5),
            accuracy=AccuracyConfig(evaluate=False),
        ),
        groups=groups,
    )


# ---- parse_fleet_request ---------------------------------------------------


def test_parse_request_minimal() -> None:
    out = what_if.parse_fleet_request({"traffic_multiplier": 1.5})
    assert out.traffic_multiplier == 1.5
    assert out.horizon_seconds is None
    assert out.group_filter == []


def test_parse_request_accepts_horizon_duration() -> None:
    out = what_if.parse_fleet_request({"traffic_multiplier": 2.0, "horizon": "6h"})
    assert out.horizon_seconds == 6 * 3600


def test_parse_request_accepts_horizon_numeric() -> None:
    out = what_if.parse_fleet_request({"traffic_multiplier": 2.0, "horizon": 600})
    assert out.horizon_seconds == 600.0


def test_parse_request_accepts_group_filter() -> None:
    out = what_if.parse_fleet_request({"traffic_multiplier": 1.0, "groups": ["a", "b"]})
    assert out.group_filter == ["a", "b"]


def test_parse_request_rejects_zero_multiplier() -> None:
    with pytest.raises(what_if.WhatIfError) as exc:
        what_if.parse_fleet_request({"traffic_multiplier": 0})
    assert exc.value.status_code == 400


def test_parse_request_rejects_negative_multiplier() -> None:
    with pytest.raises(what_if.WhatIfError) as exc:
        what_if.parse_fleet_request({"traffic_multiplier": -0.5})
    assert exc.value.status_code == 400


def test_parse_request_rejects_nan_multiplier() -> None:
    with pytest.raises(what_if.WhatIfError):
        what_if.parse_fleet_request({"traffic_multiplier": float("nan")})


def test_parse_request_rejects_non_dict_groups() -> None:
    with pytest.raises(what_if.WhatIfError):
        what_if.parse_fleet_request({"traffic_multiplier": 1.0, "groups": "g1"})


def test_parse_request_rejects_negative_horizon() -> None:
    with pytest.raises(what_if.WhatIfError):
        what_if.parse_fleet_request({"traffic_multiplier": 1.0, "horizon": -10})


def test_parse_request_rejects_unparseable_horizon() -> None:
    with pytest.raises(what_if.WhatIfError) as exc:
        what_if.parse_fleet_request({"traffic_multiplier": 1.0, "horizon": "not-a-duration"})
    assert exc.value.status_code == 400


def test_parse_request_rejects_non_dict_body() -> None:
    with pytest.raises(what_if.WhatIfError):
        what_if.parse_fleet_request("not a dict")


# ---- throttle --------------------------------------------------------------


@pytest.mark.asyncio
async def test_throttle_first_call_admitted() -> None:
    t = what_if.WhatIfThrottle(min_interval_seconds=60.0)
    await t.acquire()  # should not raise


@pytest.mark.asyncio
async def test_throttle_second_call_rejected() -> None:
    t = what_if.WhatIfThrottle(min_interval_seconds=60.0)
    await t.acquire()
    with pytest.raises(what_if.WhatIfThrottledError) as exc:
        await t.acquire()
    assert exc.value.retry_after_seconds > 0


@pytest.mark.asyncio
async def test_throttle_zero_interval_admits_back_to_back() -> None:
    t = what_if.WhatIfThrottle(min_interval_seconds=0.0)
    await t.acquire()
    await t.acquire()  # should not raise — zero interval disables the throttle


# ---- run_fleet_what_if -----------------------------------------------------


@pytest.mark.asyncio
async def test_run_fleet_what_if_returns_ranked_services(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from promforecast import models as model_registry  # noqa: PLC0415

    monkeypatch.setattr(model_registry, "build", lambda *a, **kw: _StubModel())

    cfg = _config()
    source = _StubSource([_frame(30)])
    throttle = what_if.WhatIfThrottle(min_interval_seconds=0.0)
    request = what_if.FleetWhatIfRequest(
        traffic_multiplier=2.0, horizon_seconds=None, group_filter=[]
    )
    result = await what_if.run_fleet_what_if(
        config=cfg,
        source=source,
        request=request,
        served_by="test",
        throttle=throttle,
    )
    assert result["traffic_multiplier"] == 2.0
    assert result["served_by"] == "test"
    assert result["evaluated_query_count"] == 2
    services = result["services"]
    assert len(services) == 2
    # The query with a traffic regressor (``fast_metric``) gets its
    # slope multiplied by 2.0 and so breaches its threshold sooner than
    # the regressor-less ``slow_metric``. The result should rank
    # ``fast_metric`` first.
    assert services[0]["query_id"] == "fast_metric"
    assert services[0]["traffic_multiplier_applied"] == "regressor"
    assert services[1]["query_id"] == "slow_metric"
    assert services[1]["traffic_multiplier_applied"] == "baseline"


@pytest.mark.asyncio
async def test_run_fleet_what_if_skips_drain_thresholds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``le_drain`` thresholds are not capacity events — exclude them."""
    from promforecast import models as model_registry  # noqa: PLC0415

    monkeypatch.setattr(model_registry, "build", lambda *a, **kw: _StubModel())

    drain_only = QueryConfig(
        id="queue_depth",
        promql="kafka_consumer_lag",
        thresholds=[ThresholdConfig(name="drained", value=0.0, comparator="le_drain")],
    )
    cfg = _config(queries=[drain_only])
    source = _StubSource([_frame(30)])
    throttle = what_if.WhatIfThrottle(min_interval_seconds=0.0)
    request = what_if.FleetWhatIfRequest(
        traffic_multiplier=1.5, horizon_seconds=None, group_filter=[]
    )
    result = await what_if.run_fleet_what_if(
        config=cfg, source=source, request=request, served_by="t", throttle=throttle
    )
    assert result["evaluated_query_count"] == 0
    assert result["services"] == []
    assert "No queries" in result["message"]


@pytest.mark.asyncio
async def test_run_fleet_what_if_group_filter_rejects_unknown_group() -> None:
    cfg = _config()
    source = _StubSource([_frame(30)])
    throttle = what_if.WhatIfThrottle(min_interval_seconds=0.0)
    request = what_if.FleetWhatIfRequest(
        traffic_multiplier=1.0, horizon_seconds=None, group_filter=["does-not-exist"]
    )
    with pytest.raises(what_if.WhatIfError) as exc:
        await what_if.run_fleet_what_if(
            config=cfg, source=source, request=request, served_by="t", throttle=throttle
        )
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_run_fleet_what_if_group_filter_narrows_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from promforecast import models as model_registry  # noqa: PLC0415

    monkeypatch.setattr(model_registry, "build", lambda *a, **kw: _StubModel())

    extra = GroupConfig(
        name="other",
        queries=[
            QueryConfig(
                id="ignored",
                promql="up",
                thresholds=[ThresholdConfig(name="full", value=5.0, comparator="gt")],
            )
        ],
    )
    cfg = _config(extra_group=extra)
    source = _StubSource([_frame(30)])
    throttle = what_if.WhatIfThrottle(min_interval_seconds=0.0)
    request = what_if.FleetWhatIfRequest(
        traffic_multiplier=1.0, horizon_seconds=None, group_filter=["g"]
    )
    result = await what_if.run_fleet_what_if(
        config=cfg, source=source, request=request, served_by="t", throttle=throttle
    )
    assert result["evaluated_query_count"] == 2
    assert all(s["group"] == "g" for s in result["services"])


@pytest.mark.asyncio
async def test_run_fleet_what_if_records_query_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bad query should surface in the result, not sink the whole report."""
    from promforecast import models as model_registry  # noqa: PLC0415

    monkeypatch.setattr(model_registry, "build", lambda *a, **kw: _StubModel())

    class _RaisingSource(PromSource):
        def __init__(self) -> None:
            pass

        async def query_range(
            self,
            promql: str,
            start: datetime,
            end: datetime,
            step_seconds: int,
        ) -> list[SeriesFrame]:
            raise RuntimeError("synthetic source failure")

        async def aclose(self) -> None:
            return None

    cfg = _config()
    source = _RaisingSource()
    throttle = what_if.WhatIfThrottle(min_interval_seconds=0.0)
    request = what_if.FleetWhatIfRequest(
        traffic_multiplier=1.0, horizon_seconds=None, group_filter=[]
    )
    result = await what_if.run_fleet_what_if(
        config=cfg, source=source, request=request, served_by="t", throttle=throttle
    )
    assert result["evaluated_query_count"] == 2
    for service in result["services"]:
        assert service["error"] == "query_failed"
        assert service["time_to_threshold_seconds"] == float("inf")


@pytest.mark.asyncio
async def test_run_fleet_what_if_throttle_denies_back_to_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from promforecast import models as model_registry  # noqa: PLC0415

    monkeypatch.setattr(model_registry, "build", lambda *a, **kw: _StubModel())

    cfg = _config()
    source = _StubSource([_frame(30)])
    throttle = what_if.WhatIfThrottle(min_interval_seconds=60.0)
    request = what_if.FleetWhatIfRequest(
        traffic_multiplier=1.0, horizon_seconds=None, group_filter=[]
    )
    # First call admitted.
    await what_if.run_fleet_what_if(
        config=cfg, source=source, request=request, served_by="t", throttle=throttle
    )
    # Second call rejected.
    with pytest.raises(what_if.WhatIfThrottledError):
        await what_if.run_fleet_what_if(
            config=cfg, source=source, request=request, served_by="t", throttle=throttle
        )


# ---- ETA computation --------------------------------------------------------


def test_eta_from_preview_handles_no_crossing() -> None:
    """A forecast that never crosses the threshold yields NaN."""
    import math  # noqa: PLC0415

    preview_result = {
        "timestamps": ["2026-01-01T00:00:00", "2026-01-01T01:00:00"],
        "yhat": [1.0, 1.0],
    }
    threshold = ThresholdConfig(name="full", value=5.0, comparator="gt")
    eta = what_if._eta_from_preview(preview_result=preview_result, threshold=threshold)
    assert math.isnan(eta)


def test_eta_from_preview_handles_empty_response() -> None:
    """A degenerate preview body yields NaN rather than raising."""
    import math  # noqa: PLC0415

    eta = what_if._eta_from_preview(
        preview_result={"timestamps": [], "yhat": []},
        threshold=ThresholdConfig(name="full", value=5.0, comparator="gt"),
    )
    assert math.isnan(eta)


def test_ranking_key_orders_finite_before_nan() -> None:
    """NaN ETAs should land at the bottom of the ranked list."""
    rows = [
        {"query_id": "a", "time_to_threshold_seconds": float("nan")},
        {"query_id": "b", "time_to_threshold_seconds": 100.0},
        {"query_id": "c", "time_to_threshold_seconds": 50.0},
    ]
    rows.sort(key=what_if._ranking_key)
    assert [r["query_id"] for r in rows] == ["c", "b", "a"]


# ---- module surface --------------------------------------------------------


def test_module_exports_documented_names() -> None:
    """Lock in the public surface so accidental renames are caught."""
    assert what_if.parse_fleet_request is not None
    assert what_if.run_fleet_what_if is not None
    assert issubclass(what_if.WhatIfError, RuntimeError)
    assert issubclass(what_if.WhatIfThrottledError, RuntimeError)


# ---- Quick smoke that the public preview module remains compatible ---------


def test_preview_module_still_exports_run_preview() -> None:
    """Regression guard: the what-if module imports through ``preview``."""
    assert callable(preview.run_preview)
    assert asyncio.iscoroutinefunction(preview.run_preview)


# ---- HTTP wiring ----------------------------------------------------------


def _build_test_app(
    *,
    expose_what_if: bool,
    min_interval_seconds: float | None = 0.0,
    cfg: Config | None = None,
    source: PromSource | None = None,
):
    """Wire up a minimal ``FastAPI`` for the HTTP tests.

    The endpoint plumbing in ``main.py`` is the thing under test here —
    parse errors, throttle headers, source-missing guard — so the
    helper instantiates the real ``_build_app`` rather than a stub.
    """
    from fastapi.testclient import TestClient  # noqa: PLC0415

    from promforecast.exporter import Exporter  # noqa: PLC0415
    from promforecast.main import _build_app  # noqa: PLC0415
    from promforecast.reload import ConfigReloader  # noqa: PLC0415
    from promforecast.runner import Runner  # noqa: PLC0415

    # Build the config with the requested throttle interval so the
    # runner's throttle (which we share with the endpoint via
    # ``runner._what_if_throttle``) picks up the test's value.
    config = cfg or _config()
    if min_interval_seconds is not None:
        config = config.model_copy(
            update={
                "safety": config.safety.model_copy(
                    update={"fleet_what_if_min_interval": timedelta(seconds=min_interval_seconds)}
                )
            }
        )
    exporter = Exporter()
    runner = Runner(config=config, source=source or _StubSource([_frame(30)]), exporter=exporter)
    # Use a throwaway tmp path; the reloader doesn't run in these tests
    # but its constructor needs an existing file.
    import tempfile  # noqa: PLC0415

    tmp = tempfile.NamedTemporaryFile(suffix=".yaml", delete=False)  # noqa: SIM115 - left open by design; the file lives for the duration of the test
    tmp.write(b"datasource:\n  url: http://x\ngroups: []\n")
    tmp.flush()
    reloader = ConfigReloader(runner=runner, exporter=exporter, config_path=tmp.name)
    app = _build_app(
        exporter,
        runner,
        reloader,
        reload_enabled=False,
        expose_what_if_endpoint=expose_what_if,
        source=source or _StubSource([_frame(30)]),
    )
    return TestClient(app)


def test_what_if_endpoint_off_by_default() -> None:
    client = _build_test_app(expose_what_if=False)
    # Unmounted route → 404; same shape as /config when its toggle is off.
    response = client.post("/forecast/what-if/fleet", json={"traffic_multiplier": 1.5})
    assert response.status_code == 404


def test_what_if_endpoint_rejects_bad_json() -> None:
    """Malformed JSON body returns a 400 with a precise error reason."""
    client = _build_test_app(expose_what_if=True)
    # Send a non-JSON body without a content-type header so Starlette
    # delegates body parsing to the handler (where our explicit
    # try/except runs). With a forced ``application/json`` header
    # Starlette's request middleware rejects with 422 before the
    # handler ever runs — that path is FastAPI's, not ours.
    response = client.post("/forecast/what-if/fleet", content=b"this is not json")
    assert response.status_code == 400, response.text
    assert response.json()["error"] == "invalid_body"


def test_what_if_endpoint_rejects_invalid_payload() -> None:
    client = _build_test_app(expose_what_if=True)
    response = client.post("/forecast/what-if/fleet", json={"traffic_multiplier": -1})
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_throttle_apply_config_hot_reloads_interval() -> None:
    """Changing safety.fleet_what_if_min_interval via reload must
    take effect for subsequent calls without a process restart."""
    from promforecast.exporter import Exporter as _Exporter  # noqa: PLC0415
    from promforecast.runner import Runner as _Runner  # noqa: PLC0415

    cfg = _config()
    runner = _Runner(
        config=cfg,
        source=_StubSource([_frame(30)]),
        exporter=_Exporter(),
    )
    assert runner._what_if_throttle._min_interval == 0.0
    new_cfg = cfg.model_copy(
        update={
            "safety": cfg.safety.model_copy(
                update={"fleet_what_if_min_interval": timedelta(seconds=120)}
            )
        }
    )
    await runner.apply_config(new_cfg)
    assert runner._what_if_throttle._min_interval == 120.0
    await runner.stop()


def test_what_if_endpoint_emits_admitted_counter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful call bumps forecast_what_if_runs_total{outcome=admitted}."""
    from promforecast import models as model_registry  # noqa: PLC0415

    monkeypatch.setattr(model_registry, "build", lambda *a, **kw: _StubModel())

    client = _build_test_app(expose_what_if=True)
    response = client.post("/forecast/what-if/fleet", json={"traffic_multiplier": 1.0})
    assert response.status_code == 200, response.text
    metrics = client.get("/metrics").content.decode()
    # Outcome counter family emitted with admitted = 1.
    assert 'forecast_what_if_runs_total{outcome="admitted"} 1.0' in metrics


def test_what_if_endpoint_emits_throttled_counter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from promforecast import models as model_registry  # noqa: PLC0415

    monkeypatch.setattr(model_registry, "build", lambda *a, **kw: _StubModel())

    client = _build_test_app(expose_what_if=True, min_interval_seconds=60.0)
    client.post("/forecast/what-if/fleet", json={"traffic_multiplier": 1.0})
    # Second call denied by throttle → bumps throttled counter.
    client.post("/forecast/what-if/fleet", json={"traffic_multiplier": 1.0})
    metrics = client.get("/metrics").content.decode()
    assert 'forecast_what_if_runs_total{outcome="throttled"} 1.0' in metrics


def test_what_if_endpoint_emits_invalid_body_counter() -> None:
    client = _build_test_app(expose_what_if=True)
    # Malformed body → invalid_body counter.
    client.post("/forecast/what-if/fleet", json={"traffic_multiplier": -1})
    metrics = client.get("/metrics").content.decode()
    assert 'forecast_what_if_runs_total{outcome="invalid_body"} 1.0' in metrics


def test_what_if_runs_counter_baseline_emitted_from_boot() -> None:
    """All four outcome rows are emitted at zero from boot so dashboards
    render a flat-zero series before the first call."""
    client = _build_test_app(expose_what_if=True)
    metrics = client.get("/metrics").content.decode()
    for outcome in ("admitted", "throttled", "error", "invalid_body"):
        assert f'forecast_what_if_runs_total{{outcome="{outcome}"}} 0.0' in metrics


def test_what_if_endpoint_returns_429_on_throttle(monkeypatch: pytest.MonkeyPatch) -> None:
    from promforecast import models as model_registry  # noqa: PLC0415

    monkeypatch.setattr(model_registry, "build", lambda *a, **kw: _StubModel())

    client = _build_test_app(expose_what_if=True, min_interval_seconds=60.0)
    first = client.post("/forecast/what-if/fleet", json={"traffic_multiplier": 1.0})
    assert first.status_code == 200, first.text
    second = client.post("/forecast/what-if/fleet", json={"traffic_multiplier": 1.0})
    assert second.status_code == 429
    assert "Retry-After" in second.headers
    body = second.json()
    assert body["error"] == "throttled"
    assert body["retry_after_seconds"] > 0


def test_increment_what_if_run_clamps_unknown_outcome_to_error() -> None:
    """A future code path that passes an unbounded ``outcome`` string
    must not blow the counter's label cardinality — the exporter
    collapses unknown values to ``error``."""
    from promforecast.exporter import Exporter as _Exporter  # noqa: PLC0415

    exporter = _Exporter()
    exporter.increment_what_if_run("totally-bogus-outcome")
    body = exporter.render().decode()
    # No new ``outcome="totally-bogus-outcome"`` row appears.
    assert 'outcome="totally-bogus-outcome"' not in body
    # The error outcome was bumped instead.
    assert 'forecast_what_if_runs_total{outcome="error"} 1.0' in body


def test_what_if_endpoint_catches_unexpected_exception_into_error_counter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An uncaught exception inside ``run_fleet_what_if`` must surface as
    a 500 *and* bump the ``error`` outcome counter — otherwise the
    counter's outcomes don't sum to the total invocation count and
    dashboards under-report polling."""
    from promforecast import what_if as what_if_module  # noqa: PLC0415

    async def boom(*args: object, **kwargs: object) -> object:
        raise RuntimeError("synthetic explode")

    monkeypatch.setattr(what_if_module, "run_fleet_what_if", boom)
    client = _build_test_app(expose_what_if=True)
    response = client.post("/forecast/what-if/fleet", json={"traffic_multiplier": 1.0})
    assert response.status_code == 500
    body = response.json()
    assert body["error"] == "internal"
    metrics = client.get("/metrics").content.decode()
    assert 'forecast_what_if_runs_total{outcome="error"} 1.0' in metrics
