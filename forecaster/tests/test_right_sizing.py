"""Tests for the right-sizing recommendation primitive + runner wiring."""

from __future__ import annotations

import math
from datetime import UTC, datetime

import pytest

from promforecast.config import (
    AccuracyConfig,
    Config,
    DatasourceConfig,
    DefaultsConfig,
    EmissionConfig,
    GroupConfig,
    QueryConfig,
    RecommendationsConfig,
    ResourceSizingConfig,
    ResourceSizingWorkloadConfig,
    SafetyConfig,
    ServerConfig,
)
from promforecast.exporter import Exporter, ForecastPoint, GroupSnapshot, QualityPoint
from promforecast.right_sizing import (
    compute_predicted_peak,
    compute_recommendation,
    confidence_for,
)
from promforecast.runner import Runner
from promforecast.source import PromSource, SeriesFrame

# ---- Pure-function tests ----------------------------------------------------


def _fp(metric: str, value: float, **labels: str) -> ForecastPoint:
    return ForecastPoint(metric=metric, labels=labels, value=value)


def _qp(score: float, **labels: str) -> QualityPoint:
    return QualityPoint(labels=labels, score=score)


def test_confidence_buckets_quality_score() -> None:
    """Confidence label maps quality score into a categorical band."""
    assert confidence_for(0.9) == "high"
    assert confidence_for(0.8) == "high"
    assert confidence_for(0.7) == "medium"
    assert confidence_for(0.6) == "medium"
    assert confidence_for(0.5) == "low"
    # Non-finite => low so we don't accidentally render a "high" label
    # on a degenerate forecast.
    assert confidence_for(float("nan")) == "low"


def test_peak_prefers_upper_band_at_highest_level() -> None:
    """The peak uses ``yhat_upper`` at the highest configured level.

    Operators read the recommendation as a conservative cap; the
    central forecast would systematically under-provision for the
    variability the upper band represents.
    """
    points = [
        _fp("svc_forecast", 10.0, model="m"),
        _fp("svc_forecast_upper", 12.0, model="m", level="80"),
        _fp("svc_forecast_upper", 15.0, model="m", level="95"),
        _fp("svc_forecast_upper", 14.0, model="m", level="95"),
    ]
    assert compute_predicted_peak(points, "svc") == 15.0


def test_peak_falls_back_to_central_when_no_band() -> None:
    """Without confidence bands, the central forecast peak is the recommendation."""
    points = [
        _fp("svc_forecast", 5.0, model="m"),
        _fp("svc_forecast", 7.0, model="m"),
        _fp("svc_forecast", 6.0, model="m"),
    ]
    assert compute_predicted_peak(points, "svc") == 7.0


def test_peak_returns_none_when_no_finite_samples() -> None:
    """All-NaN forecast => no recommendation."""
    points = [
        _fp("svc_forecast", float("nan"), model="m"),
        _fp("svc_forecast_upper", float("inf"), model="m", level="95"),
    ]
    assert compute_predicted_peak(points, "svc") is None


def test_peak_ignores_other_query_ids() -> None:
    """A peak for ``svc`` must not be polluted by another query's forecast."""
    points = [
        _fp("svc_forecast", 5.0, model="m"),
        _fp("other_forecast", 999.0, model="m"),
    ]
    assert compute_predicted_peak(points, "svc") == 5.0


def _config_with_floor(floor: float = 0.6, headroom: float = 30.0) -> ResourceSizingConfig:
    return ResourceSizingConfig(
        enabled=True,
        headroom_pct=headroom,
        quality_floor=floor,
        workloads=[ResourceSizingWorkloadConfig(workload="api", kind="cpu", forecast_query="svc")],
    )


def test_recommendation_applies_headroom_to_peak() -> None:
    """Recommended = peak * (1 + headroom_pct/100)."""
    cfg = _config_with_floor(floor=0.0, headroom=30.0)
    rec = compute_recommendation(
        workload=cfg.workloads[0],
        config=cfg,
        forecast_points=[_fp("svc_forecast_upper", 10.0, level="95")],
        quality_points=[_qp(0.9, id="svc")],
    )
    assert rec is not None
    assert math.isclose(rec.recommended, 13.0)
    assert rec.confidence == "high"


def test_recommendation_skipped_below_quality_floor() -> None:
    """Mean quality below floor => no row emitted."""
    cfg = _config_with_floor(floor=0.6)
    rec = compute_recommendation(
        workload=cfg.workloads[0],
        config=cfg,
        forecast_points=[_fp("svc_forecast_upper", 10.0, level="95")],
        quality_points=[_qp(0.5, id="svc")],
    )
    assert rec is None


def test_recommendation_quality_gate_uses_mean_not_max() -> None:
    """A single high-quality emission alongside bad ones must NOT pass the gate.

    The peak we compute can come from any emitted model — gating on
    the max-quality model wouldn't constrain which model actually
    contributed the peak. Mean-quality keeps the gate honest: "the
    forecast is broadly trustworthy", not "at least one model is
    trustworthy".
    """
    cfg = _config_with_floor(floor=0.6)
    # Mean of (0.9, 0.3, 0.3) = 0.5 — below the 0.6 floor — even
    # though one model is well above it.
    rec = compute_recommendation(
        workload=cfg.workloads[0],
        config=cfg,
        forecast_points=[_fp("svc_forecast_upper", 10.0, level="95")],
        quality_points=[
            _qp(0.9, id="svc"),
            _qp(0.3, id="svc"),
            _qp(0.3, id="svc"),
        ],
    )
    assert rec is None


def test_recommendation_skipped_when_quality_missing() -> None:
    """A query with no quality_points yet => no recommendation.

    Right-sizing on an unmeasured forecast would systematically
    surprise operators when the score eventually catches up.
    """
    cfg = _config_with_floor()
    rec = compute_recommendation(
        workload=cfg.workloads[0],
        config=cfg,
        forecast_points=[_fp("svc_forecast_upper", 10.0, level="95")],
        quality_points=[],
    )
    assert rec is None


def test_recommendation_with_current_computes_savings_pct() -> None:
    """``current`` enables the sibling savings_pct emission."""
    cfg = _config_with_floor(floor=0.0, headroom=30.0)
    rec = compute_recommendation(
        workload=cfg.workloads[0],
        config=cfg,
        forecast_points=[_fp("svc_forecast_upper", 10.0, level="95")],
        quality_points=[_qp(0.9, id="svc")],
        current=20.0,
    )
    assert rec is not None
    # current=20, recommended=13 → savings = (20-13)/20*100 = 35
    assert math.isclose(rec.savings_pct, 35.0)
    assert rec.current == 20.0


def test_recommendation_negative_savings_means_upsize() -> None:
    """A recommended value higher than current => negative savings.

    Operators read negative savings as "upsize, don't downsize" —
    the forecast predicts a peak above the current limit.
    """
    cfg = _config_with_floor(floor=0.0, headroom=30.0)
    rec = compute_recommendation(
        workload=cfg.workloads[0],
        config=cfg,
        forecast_points=[_fp("svc_forecast_upper", 20.0, level="95")],
        quality_points=[_qp(0.9, id="svc")],
        current=10.0,
    )
    assert rec is not None
    # current=10, recommended=26 → savings = (10-26)/10*100 = -160
    assert rec.savings_pct is not None
    assert rec.savings_pct < 0


def test_recommendation_drops_current_when_non_positive() -> None:
    """``current <= 0`` is treated as missing — the percentage is undefined."""
    cfg = _config_with_floor(floor=0.0)
    rec = compute_recommendation(
        workload=cfg.workloads[0],
        config=cfg,
        forecast_points=[_fp("svc_forecast_upper", 10.0, level="95")],
        quality_points=[_qp(0.9, id="svc")],
        current=0.0,
    )
    assert rec is not None
    # current=0 is meaningless; savings_pct must be unset.
    assert rec.savings_pct is None
    assert rec.current is None


# ---- Runner-level wiring ---------------------------------------------------


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


def _runner_config(
    *,
    workloads: list[ResourceSizingWorkloadConfig] | None = None,
    enabled: bool = True,
) -> Config:
    return Config(
        datasource=DatasourceConfig(url="http://x"),
        server=ServerConfig(),
        safety=SafetyConfig(),
        defaults=DefaultsConfig(
            min_points=5,
            models=["SeasonalNaive"],
            accuracy=AccuracyConfig(evaluate=False, auto_select=False),
            emission=EmissionConfig(quality=True, deviation=False),
        ),
        groups=[GroupConfig(name="g", queries=[QueryConfig(id="svc", promql="up")])],
        recommendations=RecommendationsConfig(
            resource_sizing=ResourceSizingConfig(
                enabled=enabled,
                headroom_pct=30.0,
                quality_floor=0.6,
                workloads=workloads or [],
            )
        ),
    )


@pytest.mark.asyncio
async def test_runner_emits_recommendation_after_refresh() -> None:
    """The runner publishes a recommendation when the forecast snapshot is healthy."""
    cfg = _runner_config(
        workloads=[ResourceSizingWorkloadConfig(workload="api", kind="cpu", forecast_query="svc")]
    )
    exporter = Exporter()
    exporter.update(
        GroupSnapshot(
            group="g",
            points=[ForecastPoint(metric="svc_forecast_upper", labels={"level": "95"}, value=10.0)],
            quality_points=[_qp(0.9, id="svc", group="g", model="SeasonalNaive")],
        )
    )
    runner = Runner(config=cfg, source=_FakeSource(), exporter=exporter)
    await runner._refresh_resource_sizing_recommendations()

    body = exporter.render().decode()
    assert "forecast_recommended_resource_size{" in body
    # 10 * 1.3 = 13.0 (no current_promql configured → no sibling rows).
    rec_line = next(
        line for line in body.splitlines() if line.startswith("forecast_recommended_resource_size{")
    )
    assert " 13.0" in rec_line
    assert 'workload="api"' in rec_line
    assert 'kind="cpu"' in rec_line
    assert 'confidence="high"' in rec_line
    # Sibling current/savings_pct must NOT appear without current_promql.
    assert "forecast_recommended_resource_current{" not in body
    assert "forecast_recommended_resource_savings_pct{" not in body


@pytest.mark.asyncio
async def test_runner_omits_recommendation_when_quality_too_low() -> None:
    """Recommendations gated by ``quality_floor`` aren't published.

    Better to publish nothing than to mislead an operator into
    downsizing on an untrustworthy forecast.
    """
    cfg = _runner_config(
        workloads=[ResourceSizingWorkloadConfig(workload="api", kind="cpu", forecast_query="svc")]
    )
    exporter = Exporter()
    exporter.update(
        GroupSnapshot(
            group="g",
            points=[ForecastPoint(metric="svc_forecast_upper", labels={"level": "95"}, value=10.0)],
            # Below the 0.6 floor configured in _runner_config.
            quality_points=[_qp(0.4, id="svc", group="g", model="SeasonalNaive")],
        )
    )
    runner = Runner(config=cfg, source=_FakeSource(), exporter=exporter)
    await runner._refresh_resource_sizing_recommendations()

    body = exporter.render().decode()
    assert "forecast_recommended_resource_size{" not in body


@pytest.mark.asyncio
async def test_runner_clears_recommendations_when_feature_disabled() -> None:
    """Disabling resource_sizing on reload drops the prior rows immediately."""
    cfg = _runner_config(
        workloads=[ResourceSizingWorkloadConfig(workload="api", kind="cpu", forecast_query="svc")]
    )
    exporter = Exporter()
    exporter.update(
        GroupSnapshot(
            group="g",
            points=[ForecastPoint(metric="svc_forecast_upper", labels={"level": "95"}, value=10.0)],
            quality_points=[_qp(0.9, id="svc", group="g", model="SeasonalNaive")],
        )
    )
    runner = Runner(config=cfg, source=_FakeSource(), exporter=exporter)
    await runner._refresh_resource_sizing_recommendations()
    initial = exporter.render().decode()
    assert "forecast_recommended_resource_size{" in initial

    disabled = _runner_config(enabled=False)
    await runner.apply_config(disabled)
    after = exporter.render().decode()
    assert "forecast_recommended_resource_size{" not in after


@pytest.mark.asyncio
async def test_runner_current_promql_success_emits_all_three_siblings() -> None:
    """A resolved ``current_promql`` produces recommended + current + savings_pct.

    End-to-end verification of the merged sibling family: all three
    metrics share the (workload, kind) label set; current/savings_pct
    only emit when the operator wires the optional fetch.
    """
    cfg = _runner_config(
        workloads=[
            ResourceSizingWorkloadConfig(
                workload="api",
                kind="cpu",
                forecast_query="svc",
                current_promql="kube_pod_container_resource_requests",
            )
        ]
    )
    exporter = Exporter()
    exporter.update(
        GroupSnapshot(
            group="g",
            points=[ForecastPoint(metric="svc_forecast_upper", labels={"level": "95"}, value=10.0)],
            quality_points=[_qp(0.9, id="svc", group="g", model="SeasonalNaive")],
        )
    )
    # Source resolves the current_promql to a single sample of 20.0.
    source = _FakeSource(
        {
            "kube_pod_container_resource_requests": [
                SeriesFrame(
                    labels={"workload": "api"},
                    timestamps=[datetime(2026, 5, 25, tzinfo=UTC)],
                    values=[20.0],
                )
            ]
        }
    )
    runner = Runner(config=cfg, source=source, exporter=exporter)
    await runner._refresh_resource_sizing_recommendations()

    body = exporter.render().decode()
    # Primary recommendation gauge: 10 * 1.3 = 13.0.
    assert any(
        line.startswith("forecast_recommended_resource_size{") and " 13.0" in line
        for line in body.splitlines()
    )
    # Current value from the source: 20.0.
    assert any(
        line.startswith("forecast_recommended_resource_current{") and " 20.0" in line
        for line in body.splitlines()
    )
    # Savings %: (20 - 13) / 20 * 100 = 35.0.
    assert any(
        line.startswith("forecast_recommended_resource_savings_pct{") and " 35.0" in line
        for line in body.splitlines()
    )


@pytest.mark.asyncio
async def test_runner_current_promql_failure_still_emits_recommendation() -> None:
    """A failed ``current_promql`` fetch must not suppress the recommended row.

    The recommendation is the primary signal; the current / savings_pct
    siblings are nice-to-haves. A flaky current query shouldn't take
    the whole row down.
    """
    cfg = _runner_config(
        workloads=[
            ResourceSizingWorkloadConfig(
                workload="api",
                kind="cpu",
                forecast_query="svc",
                current_promql="nonexistent_promql_returns_nothing",
            )
        ]
    )
    exporter = Exporter()
    exporter.update(
        GroupSnapshot(
            group="g",
            points=[ForecastPoint(metric="svc_forecast_upper", labels={"level": "95"}, value=10.0)],
            quality_points=[_qp(0.9, id="svc", group="g", model="SeasonalNaive")],
        )
    )
    # Source returns nothing for the current PromQL — the fetch is a
    # no-op rather than an error, but the resulting None must still
    # produce a recommended row.
    runner = Runner(config=cfg, source=_FakeSource(), exporter=exporter)
    await runner._refresh_resource_sizing_recommendations()

    body = exporter.render().decode()
    assert "forecast_recommended_resource_size{" in body
    # Without a current value, savings_pct must be absent.
    assert "forecast_recommended_resource_savings_pct{" not in body
