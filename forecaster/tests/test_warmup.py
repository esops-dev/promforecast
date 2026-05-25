"""Tests for the warm-up status endpoint + CLI machinery.

The warmup probe issues one cheap PromQL ``query_range`` per
configured query. These tests use a fake source so they cover the
status-classification logic (ready / warming / error, blocked_by
enum, ETA estimate) without hitting a real TSDB.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from promforecast.config import (
    Config,
    DatasourceConfig,
    DefaultsConfig,
    GroupConfig,
    QueryConfig,
    RegressorConfig,
    SafetyConfig,
    ServerConfig,
)
from promforecast.exporter import Exporter
from promforecast.main import _build_app
from promforecast.reload import ConfigReloader
from promforecast.runner import Runner
from promforecast.source import PromSource, SeriesFrame
from promforecast.warmup import (
    GroupWarmupStatus,
    QueryWarmupStatus,
    WarmupReport,
    compute_warmup,
    render_table,
)


class _FakeSource(PromSource):
    """Returns canned series for each PromQL string; never opens a socket."""

    def __init__(self, by_query: dict[str, list[SeriesFrame]]) -> None:
        self._by_query = by_query
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


def _config(
    queries: list[QueryConfig],
    *,
    min_points: int = 100,
    lookback_minutes: int = 60,
) -> Config:
    return Config(
        datasource=DatasourceConfig(url="http://x"),
        server=ServerConfig(),
        safety=SafetyConfig(),
        defaults=DefaultsConfig(
            min_points=min_points,
            models=["SeasonalNaive"],
            lookback=timedelta(minutes=lookback_minutes),
            step=timedelta(minutes=1),
        ),
        groups=[GroupConfig(name="g", queries=queries)],
    )


def _frame(n_points: int) -> SeriesFrame:
    return SeriesFrame(
        labels={"instance": "host-a"},
        timestamps=[datetime.now(tz=UTC)] * n_points,
        values=[1.0] * n_points,
    )


@pytest.mark.asyncio
async def test_warmup_reports_ready_when_min_points_satisfied() -> None:
    cfg = _config([QueryConfig(id="m", promql="up")], min_points=10)
    source = _FakeSource({"up": [_frame(20)]})
    report = await compute_warmup(config=cfg, source=source)
    assert len(report.groups) == 1
    group = report.groups[0]
    assert group.name == "g"
    assert len(group.queries) == 1
    status = group.queries[0]
    assert status.status == "ready"
    assert status.points_available == 20
    assert status.points_needed == 10
    assert status.blocked_by == "none"
    assert status.eta_seconds == 0.0


@pytest.mark.asyncio
async def test_warmup_reports_warming_with_eta() -> None:
    """A series under min_points reports ETA = missing-points * step."""
    cfg = _config([QueryConfig(id="m", promql="up")], min_points=100)
    source = _FakeSource({"up": [_frame(40)]})
    report = await compute_warmup(config=cfg, source=source)
    status = report.groups[0].queries[0]
    assert status.status == "warming"
    assert status.points_available == 40
    assert status.points_needed == 100
    assert status.blocked_by == "min_points"
    # Default step is 1m = 60s; missing 60 points * 60s = 3600s.
    assert status.eta_seconds == 60 * 60


@pytest.mark.asyncio
async def test_warmup_reports_lookback_when_no_samples() -> None:
    """A query that returned zero samples blocks on lookback, not min_points."""
    cfg = _config([QueryConfig(id="m", promql="up")])
    source = _FakeSource({"up": []})
    report = await compute_warmup(config=cfg, source=source)
    status = report.groups[0].queries[0]
    assert status.status == "warming"
    assert status.points_available == 0
    assert status.blocked_by == "lookback"


@pytest.mark.asyncio
async def test_warmup_reports_longest_series() -> None:
    """A multi-series query reports the best-off (longest) series.

    The runner's per-series fit treats each frame independently, so
    the longest series unblocks the query first. Reporting the mean
    would silently smear over a single slow-to-warm peer.
    """
    cfg = _config([QueryConfig(id="m", promql="up")], min_points=15)
    source = _FakeSource({"up": [_frame(5), _frame(20), _frame(10)]})
    report = await compute_warmup(config=cfg, source=source)
    status = report.groups[0].queries[0]
    assert status.points_available == 20
    assert status.status == "ready"


@pytest.mark.asyncio
async def test_warmup_reports_error_on_probe_failure() -> None:
    class _BadSource(_FakeSource):
        async def query_range(
            self,
            promql: str,
            start: datetime,
            end: datetime,
            step_seconds: int,
        ) -> list[SeriesFrame]:
            raise RuntimeError("boom")

    cfg = _config([QueryConfig(id="m", promql="up")])
    source = _BadSource({})
    report = await compute_warmup(config=cfg, source=source)
    status = report.groups[0].queries[0]
    assert status.status == "error"
    assert status.blocked_by == "lookback"


@pytest.mark.asyncio
async def test_warmup_event_profile_short_circuits_probe() -> None:
    """Event-profile queries don't have a min_points contract."""
    cfg = _config(
        [QueryConfig(id="m", promql="pagerduty_incidents_total", data_profile="events")],
    )
    source = _FakeSource({})
    report = await compute_warmup(config=cfg, source=source)
    status = report.groups[0].queries[0]
    assert status.status == "ready"
    # No probe issued — the event path skips the cost entirely.
    assert source.calls == []


@pytest.mark.asyncio
async def test_warmup_required_regressor_visible_when_no_primary_samples() -> None:
    """When the primary returns zero but a regressor is required, surface the dependency."""
    query = QueryConfig(
        id="m",
        promql="up",
        regressors=[RegressorConfig(id="deploys", promql="changes(deploys[1h])", required=True)],
    )
    cfg = _config([query])
    source = _FakeSource({"up": []})
    report = await compute_warmup(config=cfg, source=source)
    status = report.groups[0].queries[0]
    # Primary returned 0 points -> blocked on lookback first; required-regressor
    # surface kicks in only at the deeper code path where min_points is hit
    # *with* zero samples — the lookback wording dominates here, which matches
    # the runner's failure order.
    assert status.blocked_by == "lookback"


@pytest.mark.asyncio
async def test_warmup_to_dict_round_trip() -> None:
    """The JSON envelope shape is stable and re-parseable."""
    cfg = _config([QueryConfig(id="m", promql="up")], min_points=5)
    source = _FakeSource({"up": [_frame(10)]})
    report = await compute_warmup(config=cfg, source=source)
    payload = report.to_dict()
    assert "groups" in payload
    assert payload["groups"][0]["name"] == "g"
    q0 = payload["groups"][0]["queries"][0]
    assert q0["id"] == "m"
    assert q0["status"] == "ready"
    assert q0["points_available"] == 10
    assert q0["points_needed"] == 5
    assert q0["blocked_by"] == "none"
    assert q0["eta_seconds"] == 0.0


def test_render_table_includes_header_and_rows() -> None:
    report = WarmupReport(
        groups=[
            GroupWarmupStatus(
                name="g",
                queries=[
                    QueryWarmupStatus(
                        id="ready_one",
                        status="ready",
                        points_available=200,
                        points_needed=100,
                        blocked_by="none",
                        eta_seconds=0.0,
                    ),
                    QueryWarmupStatus(
                        id="warming_one",
                        status="warming",
                        points_available=40,
                        points_needed=100,
                        blocked_by="min_points",
                        eta_seconds=3600.0,
                    ),
                ],
            )
        ]
    )
    text = render_table(report)
    assert "group" in text
    assert "ready_one" in text
    assert "warming_one" in text
    assert "1h" in text  # the 3600s ETA renders as 1h


def test_render_table_handles_empty_report() -> None:
    text = render_table(WarmupReport(groups=[]))
    assert "no groups" in text


# ----- HTTP-level wiring tests -------------------------------------------
#
# Cover the route registration in ``main._build_app``: the endpoint
# only mounts when ``expose_warmup_endpoint=True`` AND a ``source`` is
# wired, and the JSON envelope must match what the CLI's
# ``_report_from_payload`` re-parses.


def test_endpoint_off_by_default_returns_404() -> None:
    cfg = _config([QueryConfig(id="m", promql="up")])
    exporter = Exporter()
    runner = Runner(config=cfg, source=_FakeSource({"up": [_frame(20)]}), exporter=exporter)
    reloader = ConfigReloader(
        runner=runner,
        exporter=exporter,
        config_path=None,  # type: ignore[arg-type]
    )
    app = _build_app(
        exporter,
        runner,
        reloader,
        reload_enabled=False,
        expose_warmup_endpoint=False,
    )
    client = TestClient(app)
    assert client.get("/forecast/warmup").status_code == 404


def test_endpoint_returns_report_when_enabled() -> None:
    cfg = _config([QueryConfig(id="m", promql="up")], min_points=10)
    exporter = Exporter()
    source = _FakeSource({"up": [_frame(20)]})
    runner = Runner(config=cfg, source=source, exporter=exporter)
    reloader = ConfigReloader(
        runner=runner,
        exporter=exporter,
        config_path=None,  # type: ignore[arg-type]
    )
    app = _build_app(
        exporter,
        runner,
        reloader,
        reload_enabled=False,
        expose_warmup_endpoint=True,
        source=source,
    )
    client = TestClient(app)
    response = client.get("/forecast/warmup")
    assert response.status_code == 200
    payload = response.json()
    assert payload["groups"][0]["name"] == "g"
    q0 = payload["groups"][0]["queries"][0]
    assert q0["id"] == "m"
    assert q0["status"] == "ready"
    assert q0["points_available"] == 20
    assert q0["blocked_by"] == "none"


def test_endpoint_disabled_when_source_missing() -> None:
    """``expose_warmup_endpoint=True`` + no source logs a warning, no route."""
    cfg = _config([QueryConfig(id="m", promql="up")])
    exporter = Exporter()
    runner = Runner(config=cfg, source=_FakeSource({}), exporter=exporter)
    reloader = ConfigReloader(
        runner=runner,
        exporter=exporter,
        config_path=None,  # type: ignore[arg-type]
    )
    app = _build_app(
        exporter,
        runner,
        reloader,
        reload_enabled=False,
        expose_warmup_endpoint=True,
        source=None,
    )
    client = TestClient(app)
    assert client.get("/forecast/warmup").status_code == 404


# ----- Required-regressor probing ----------------------------------------


@pytest.mark.asyncio
async def test_warmup_blocks_on_missing_required_regressor() -> None:
    """Primary returns enough samples; required regressor returns zero -> block."""
    query = QueryConfig(
        id="m",
        promql="up",
        regressors=[
            RegressorConfig(id="deploys", promql="deploys_total", required=True),
        ],
    )
    cfg = _config([query], min_points=5)
    source = _FakeSource(
        {
            "up": [_frame(10)],
            "deploys_total": [],  # required regressor is empty
        }
    )
    report = await compute_warmup(config=cfg, source=source)
    status = report.groups[0].queries[0]
    assert status.status == "warming"
    assert status.blocked_by == "regressor:deploys"
    # The probe issued one extra query against the regressor PromQL.
    assert "deploys_total" in source.calls


@pytest.mark.asyncio
async def test_warmup_ready_when_required_regressor_has_samples() -> None:
    """A populated required regressor doesn't block."""
    query = QueryConfig(
        id="m",
        promql="up",
        regressors=[
            RegressorConfig(id="deploys", promql="deploys_total", required=True),
        ],
    )
    cfg = _config([query], min_points=5)
    source = _FakeSource(
        {
            "up": [_frame(10)],
            "deploys_total": [_frame(10)],
        }
    )
    report = await compute_warmup(config=cfg, source=source)
    status = report.groups[0].queries[0]
    assert status.status == "ready"
    assert status.blocked_by == "none"


@pytest.mark.asyncio
async def test_warmup_skips_non_required_regressors() -> None:
    """Optional regressors are not probed, even when empty."""
    query = QueryConfig(
        id="m",
        promql="up",
        regressors=[
            RegressorConfig(id="deploys", promql="deploys_total", required=False),
        ],
    )
    cfg = _config([query], min_points=5)
    source = _FakeSource({"up": [_frame(10)]})
    report = await compute_warmup(config=cfg, source=source)
    status = report.groups[0].queries[0]
    assert status.status == "ready"
    # The optional regressor's PromQL is never queried.
    assert "deploys_total" not in source.calls


@pytest.mark.asyncio
async def test_warmup_blocks_when_required_regressor_probe_raises() -> None:
    """A regressor whose probe raises is treated as blocking.

    Mirrors the runner's behaviour: a required regressor the TSDB
    can't serve would land under ``forecast_regressor_failures_total``
    on the live runner, so the warmup endpoint surfaces the same
    dependency upfront rather than letting the operator wait for the
    fit attempt to fail.
    """

    class _PrimaryOkRegressorBoom(_FakeSource):
        async def query_range(
            self,
            promql: str,
            start: datetime,
            end: datetime,
            step_seconds: int,
        ) -> list[SeriesFrame]:
            self.calls.append(promql)
            if promql == "deploys_total":
                raise RuntimeError("regressor probe blew up")
            return self._by_query.get(promql, [])

    query = QueryConfig(
        id="m",
        promql="up",
        regressors=[
            RegressorConfig(id="deploys", promql="deploys_total", required=True),
        ],
    )
    cfg = _config([query], min_points=5)
    source = _PrimaryOkRegressorBoom({"up": [_frame(10)]})
    report = await compute_warmup(config=cfg, source=source)
    status = report.groups[0].queries[0]
    assert status.status == "warming"
    assert status.blocked_by == "regressor:deploys"
    assert "deploys_total" in source.calls


@pytest.mark.asyncio
async def test_warmup_discover_query_blocks_with_discovery_token() -> None:
    """Templated queries short-circuit to status=ready with blocked_by=discovery.

    The probe can't run against an unresolved template, but reporting
    ``blocked_by=none`` would mislead the operator into thinking the
    query is actively forecasting. ``discovery`` makes the dependency
    explicit while keeping status=ready so the CLI's until-loop
    terminates (templated queries are not blocking the install).
    """
    from promforecast.config import DiscoveryVariable  # noqa: PLC0415

    query = QueryConfig(
        id="m",
        promql="up{instance=~'{{ instance }}'}",
        discover=[
            DiscoveryVariable(name="instance", label="instance", promql="up"),
        ],
    )
    cfg = _config([query])
    source = _FakeSource({})  # never reached
    report = await compute_warmup(config=cfg, source=source)
    status = report.groups[0].queries[0]
    assert status.status == "ready"
    assert status.blocked_by == "discovery"
    # The probe was never issued — templated queries don't touch the
    # source at all.
    assert source.calls == []


@pytest.mark.asyncio
async def test_warmup_required_regressor_ordering_is_deterministic() -> None:
    """Multiple blocked regressors resolve to config order, not race order.

    The runner-side ``forecast_regressor_failures_total`` counter is
    keyed by regressor id; the warmup endpoint reports exactly one
    ``blocked_by`` per query, so the first-by-config rule must be
    stable so successive calls don't flap between "blocked on A"
    and "blocked on B" depending on which probe response landed
    first.
    """
    query = QueryConfig(
        id="m",
        promql="up",
        regressors=[
            RegressorConfig(id="deploys", promql="deploys_total", required=True),
            RegressorConfig(id="rollouts", promql="rollouts_total", required=True),
        ],
    )
    cfg = _config([query], min_points=5)
    source = _FakeSource(
        {
            "up": [_frame(10)],
            "deploys_total": [],
            "rollouts_total": [],
        }
    )
    report = await compute_warmup(config=cfg, source=source)
    status = report.groups[0].queries[0]
    assert status.blocked_by == "regressor:deploys"


# ----- CLI plumbing ------------------------------------------------------


def test_run_warmup_cli_exits_zero_on_empty_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``until promforecast warmup`` must terminate against an empty config.

    Returning ``1`` for "no groups configured" would spin a docs-recommended
    ``until promforecast warmup`` loop forever against a fresh install whose
    operator hasn't added any queries yet. The forecaster itself is healthy
    — there's just nothing to wait on — so ``0`` is the right exit code.
    """
    import httpx  # noqa: PLC0415

    from promforecast.main import _run_warmup_cli  # noqa: PLC0415

    class _FakeResponse:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, list[dict[str, object]]]:
            return {"groups": []}

    monkeypatch.setattr(httpx, "get", lambda url, timeout: _FakeResponse())
    rc = _run_warmup_cli(url="http://x", timeout_seconds=1.0, json_output=True)
    assert rc == 0


def test_run_warmup_cli_exits_one_when_a_query_is_warming(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-ready status returns exit ``1`` so the until-loop keeps polling."""
    import httpx  # noqa: PLC0415

    from promforecast.main import _run_warmup_cli  # noqa: PLC0415

    payload = {
        "groups": [
            {
                "name": "g",
                "queries": [{"id": "q", "status": "warming", "blocked_by": "min_points"}],
            }
        ]
    }

    class _FakeResponse:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return payload

    monkeypatch.setattr(httpx, "get", lambda url, timeout: _FakeResponse())
    rc = _run_warmup_cli(url="http://x", timeout_seconds=1.0, json_output=True)
    assert rc == 1
