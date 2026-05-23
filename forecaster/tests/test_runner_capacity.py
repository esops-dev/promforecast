"""Runner-level integration tests for the capacity-planning families.

These exercise the end-to-end wiring through ``Runner.run_group`` so a
regression in the per-fit hook, snapshot merge, or exporter rendering is
caught even if the per-primitive tests pass.
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
    GroupConfig,
    GrowthRateEmissionConfig,
    QueryConfig,
    QueryEmissionConfig,
    SafetyConfig,
    ServerConfig,
    ThresholdConfig,
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
    """Emits a linear ramp forecast so ETA crossings are predictable."""

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
async def test_runner_emits_threshold_eta_family(monkeypatch: pytest.MonkeyPatch) -> None:
    """A configured threshold must materialise on /metrics with the right labels."""
    from promforecast import models as model_registry  # noqa: PLC0415

    monkeypatch.setattr(model_registry, "build", lambda *a, **kw: _RampForecastModel())

    # Horizon = 2h at 5m step → 24 forecast rows starting at yhat=51 and
    # rising +1/step to yhat=74. Pick thresholds that fall within that
    # range so the ETA crossings actually fire.
    cfg = _config_with_query(
        QueryConfig(
            id="disk_used_pct",
            promql="up",
            thresholds=[
                ThresholdConfig(name="warn", value=60.0, comparator="gt"),
                ThresholdConfig(name="critical", value=70.0, comparator="gt"),
            ],
        )
    )
    source = _FakeSource({"up": [_frame(20)]})
    runner = Runner(config=cfg, source=source, exporter=(exporter := Exporter()))

    await runner.run_group(cfg.groups[0])

    body = exporter.render().decode()
    assert "disk_used_pct_forecast_time_to_threshold_seconds" in body
    assert 'name="warn"' in body
    assert 'name="critical"' in body
    assert 'comparator="gt"' in body
    # Per-level band ETA should be emitted alongside the point ETA.
    assert 'level="80"' in body


@pytest.mark.asyncio
async def test_runner_emits_growth_rate_family(monkeypatch: pytest.MonkeyPatch) -> None:
    """Growth-rate emission must produce one sample per configured window."""
    from promforecast import models as model_registry  # noqa: PLC0415

    monkeypatch.setattr(model_registry, "build", lambda *a, **kw: _RampForecastModel())

    cfg = _config_with_query(
        QueryConfig(
            id="requests_per_sec",
            promql="up",
            emission=QueryEmissionConfig(
                growth_rate=GrowthRateEmissionConfig(
                    enabled=True,
                    windows=[timedelta(minutes=30), timedelta(hours=1)],
                    units="units_per_second",
                ),
            ),
        )
    )
    source = _FakeSource({"up": [_frame(20)]})
    runner = Runner(config=cfg, source=source, exporter=(exporter := Exporter()))

    await runner.run_group(cfg.groups[0])

    body = exporter.render().decode()
    assert "requests_per_sec_forecast_growth_rate" in body
    assert 'window="30m"' in body
    assert 'window="1h"' in body
    assert 'units="units_per_second"' in body


@pytest.mark.asyncio
async def test_runner_eta_band_reflects_conformal_calibration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A widening conformal offset must pull the conservative band ETA earlier.

    Without the fix, the band ETA samples are computed from the model's
    raw bands while /metrics exposes the calibrated bands, so an alert
    on ``time_to_threshold{level="80"} < N`` disagrees with the
    ``yhat_upper{level="80"}`` panel sitting right next to it. The two
    must agree.
    """
    from promforecast import models as model_registry  # noqa: PLC0415
    from promforecast.capacity import compute_threshold_etas  # noqa: PLC0415
    from promforecast.runner import _apply_offsets_to_threshold_eta_inputs  # noqa: PLC0415

    monkeypatch.setattr(model_registry, "build", lambda *a, **kw: _RampForecastModel())

    # Build the forecast df the same way the runner does, then verify the
    # ETA helper shifts band crossings when an offset is supplied.
    base = datetime(2026, 5, 21, tzinfo=UTC)
    df = pd.DataFrame(
        {
            "unique_id": ["s"] * 5,
            "ds": [base + timedelta(minutes=5 * i) for i in range(5)],
            "yhat": [50.0, 55.0, 60.0, 65.0, 70.0],
            "yhat_upper_80": [52.0, 57.0, 62.0, 67.0, 72.0],
            "yhat_lower_80": [48.0, 53.0, 58.0, 63.0, 68.0],
        }
    )
    thresholds = [ThresholdConfig(name="warn", value=65.0, comparator="gt")]
    # Without calibration, yhat_upper_80 crosses 65 between step 2 (62) and
    # step 3 (67) → ETA = 600s + (3/5)*300 = 780s.
    raw = compute_threshold_etas(forecast=df, thresholds=thresholds, levels=[80])
    raw_band = next(e for e in raw if e.level == 80)
    # With a positive offset (the calibration widens the band by +5), the
    # *new* yhat_upper crosses 65 sooner.
    calibrated = _apply_offsets_to_threshold_eta_inputs(df, {80: 5.0}, [80])
    cal = compute_threshold_etas(forecast=calibrated, thresholds=thresholds, levels=[80])
    cal_band = next(e for e in cal if e.level == 80)
    assert cal_band.seconds < raw_band.seconds


@pytest.mark.asyncio
async def test_runner_emits_will_breach_family(monkeypatch: pytest.MonkeyPatch) -> None:
    """A threshold with ``alert_within`` + ``severity`` emits the boolean.

    The RampForecastModel's central forecast crosses 60 mid-horizon and 70
    near the end. We pick ``alert_within`` so the warn threshold fires
    (1.0) and critical does not (0.0). Per-level rows are produced for
    each emitted band.
    """
    from promforecast import models as model_registry  # noqa: PLC0415

    monkeypatch.setattr(model_registry, "build", lambda *a, **kw: _RampForecastModel())

    cfg = _config_with_query(
        QueryConfig(
            id="disk_used_pct",
            promql="up",
            thresholds=[
                ThresholdConfig(
                    name="warn",
                    value=60.0,
                    comparator="gt",
                    alert_within=timedelta(hours=2),
                    severity="warning",
                ),
                ThresholdConfig(
                    name="critical",
                    value=70.0,
                    comparator="gt",
                    alert_within=timedelta(minutes=1),
                    severity="critical",
                ),
            ],
        )
    )
    source = _FakeSource({"up": [_frame(20)]})
    runner = Runner(config=cfg, source=source, exporter=(exporter := Exporter()))

    await runner.run_group(cfg.groups[0])

    body = exporter.render().decode()
    assert "disk_used_pct_forecast_will_breach" in body
    assert 'severity="warning"' in body
    assert 'severity="critical"' in body
    # warn alert_within=2h easily contains the ~11min ETA → 1.
    warn_lines = [
        line
        for line in body.splitlines()
        if line.startswith("disk_used_pct_forecast_will_breach{")
        and 'name="warn"' in line
        and 'level="80"' not in line
    ]
    assert warn_lines and warn_lines[0].endswith(" 1.0")
    # critical alert_within=1min cannot contain the ~21min ETA → 0.
    crit_lines = [
        line
        for line in body.splitlines()
        if line.startswith("disk_used_pct_forecast_will_breach{")
        and 'name="critical"' in line
        and 'level="80"' not in line
    ]
    assert crit_lines and crit_lines[0].endswith(" 0.0")


@pytest.mark.asyncio
async def test_runner_omits_will_breach_when_threshold_opts_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A threshold without ``alert_within`` does not emit the boolean."""
    from promforecast import models as model_registry  # noqa: PLC0415

    monkeypatch.setattr(model_registry, "build", lambda *a, **kw: _RampForecastModel())

    cfg = _config_with_query(
        QueryConfig(
            id="quiet_metric",
            promql="up",
            thresholds=[ThresholdConfig(name="warn", value=60.0, comparator="gt")],
        )
    )
    source = _FakeSource({"up": [_frame(20)]})
    runner = Runner(config=cfg, source=source, exporter=(exporter := Exporter()))
    await runner.run_group(cfg.groups[0])
    body = exporter.render().decode()
    # ETA family exists, will_breach does not — the operator opted into
    # one and not the other.
    assert "quiet_metric_forecast_time_to_threshold_seconds" in body
    assert "quiet_metric_forecast_will_breach" not in body


@pytest.mark.asyncio
async def test_runner_omits_capacity_families_when_unconfigured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No threshold + growth-rate disabled = no capacity families on /metrics."""
    from promforecast import models as model_registry  # noqa: PLC0415

    monkeypatch.setattr(model_registry, "build", lambda *a, **kw: _RampForecastModel())

    cfg = _config_with_query(QueryConfig(id="quiet", promql="up"))
    source = _FakeSource({"up": [_frame(20)]})
    runner = Runner(config=cfg, source=source, exporter=(exporter := Exporter()))

    await runner.run_group(cfg.groups[0])

    body = exporter.render().decode()
    assert "quiet_forecast_time_to_threshold_seconds" not in body
    assert "quiet_forecast_growth_rate" not in body
