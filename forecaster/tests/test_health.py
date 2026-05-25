"""Unit tests for the service-level forecast health composite."""

from __future__ import annotations

import math
from datetime import datetime

import pytest

from promforecast.config import (
    AccuracyConfig,
    Config,
    DatasourceConfig,
    DefaultsConfig,
    EmissionConfig,
    GovernanceConfig,
    GroupConfig,
    HealthScoreConfig,
    QueryConfig,
    SafetyConfig,
    ServerConfig,
)
from promforecast.exporter import Exporter, GroupSnapshot, QualityPoint
from promforecast.health import compute_service_health_scores
from promforecast.runner import Runner
from promforecast.source import PromSource, SeriesFrame


def _qp(score: float, **labels: str) -> QualityPoint:
    return QualityPoint(labels=labels, score=score)


def test_aggregate_buckets_quality_by_single_label() -> None:
    """A single aggregate_by label produces one row per distinct value."""
    cfg = HealthScoreConfig(aggregate_by=["service"])
    points = [
        _qp(0.8, service="checkout", id="latency", group="g"),
        _qp(0.6, service="checkout", id="errors", group="g"),
        _qp(0.4, service="search", id="latency", group="g"),
    ]
    scores = compute_service_health_scores(points, cfg)
    assert len(scores) == 2
    by_service = {s.labels["service"]: s for s in scores}
    assert math.isclose(by_service["checkout"].score, 0.7)
    assert by_service["checkout"].series_count == 2
    assert math.isclose(by_service["search"].score, 0.4)
    assert by_service["search"].series_count == 1


def test_aggregate_by_multiple_labels_distinguishes_buckets() -> None:
    """Two aggregate_by labels produce one row per (service, environment) pair."""
    cfg = HealthScoreConfig(aggregate_by=["service", "environment"])
    points = [
        _qp(0.9, service="checkout", environment="prod"),
        _qp(0.8, service="checkout", environment="prod"),
        _qp(0.5, service="checkout", environment="staging"),
    ]
    scores = compute_service_health_scores(points, cfg)
    assert len(scores) == 2
    prod = next(s for s in scores if s.labels["environment"] == "prod")
    staging = next(s for s in scores if s.labels["environment"] == "staging")
    assert math.isclose(prod.score, 0.85)
    assert math.isclose(staging.score, 0.5)


def test_aggregate_drops_points_missing_required_label() -> None:
    """A series missing one of the aggregate_by labels is dropped.

    Without the drop, an unlabelled forecast would slip into every
    bucket and skew the report.
    """
    cfg = HealthScoreConfig(aggregate_by=["service", "environment"])
    points = [
        _qp(0.9, service="checkout", environment="prod"),
        _qp(0.1, service="checkout"),  # missing environment
        _qp(0.5, environment="prod"),  # missing service
    ]
    scores = compute_service_health_scores(points, cfg)
    assert len(scores) == 1
    assert math.isclose(scores[0].score, 0.9)


def test_aggregate_drops_empty_string_label_values() -> None:
    """``label=""`` is treated as missing — likely a labelling bug."""
    cfg = HealthScoreConfig(aggregate_by=["service"])
    points = [
        _qp(0.9, service="checkout"),
        _qp(0.1, service=""),
    ]
    scores = compute_service_health_scores(points, cfg)
    assert len(scores) == 1
    assert scores[0].labels["service"] == "checkout"


def test_aggregate_drops_non_finite_scores() -> None:
    """A NaN quality score doesn't poison its bucket."""
    cfg = HealthScoreConfig(aggregate_by=["service"])
    points = [
        _qp(0.8, service="checkout"),
        _qp(float("nan"), service="checkout"),
        _qp(float("inf"), service="checkout"),
    ]
    scores = compute_service_health_scores(points, cfg)
    assert len(scores) == 1
    # Only the 0.8 row contributed.
    assert math.isclose(scores[0].score, 0.8)
    assert scores[0].series_count == 1


def test_quality_floor_drops_low_scores_before_aggregation() -> None:
    """``quality_floor`` filters individual scores before the bucket mean.

    Useful when cold-start / low-quality forecasts should not drag a
    service's composite below the alert gate (typically 0.6).
    """
    cfg = HealthScoreConfig(aggregate_by=["service"], quality_floor=0.5)
    points = [
        _qp(0.9, service="checkout"),
        _qp(0.4, service="checkout"),  # dropped — below floor
        _qp(0.7, service="checkout"),
    ]
    scores = compute_service_health_scores(points, cfg)
    assert len(scores) == 1
    assert math.isclose(scores[0].score, 0.8)
    assert scores[0].series_count == 2


def test_quality_floor_can_eliminate_a_bucket_entirely() -> None:
    """A bucket whose every score is below the floor disappears.

    The story: when every series for a service is too low-quality to
    be trustworthy, surfacing a zero-row sample would mislead readers
    of the executive dashboard. Better to omit the bucket and let the
    "bucket missing" be the signal.
    """
    cfg = HealthScoreConfig(aggregate_by=["service"], quality_floor=0.5)
    points = [
        _qp(0.4, service="checkout"),
        _qp(0.3, service="checkout"),
        _qp(0.9, service="payments"),
    ]
    scores = compute_service_health_scores(points, cfg)
    assert len(scores) == 1
    assert scores[0].labels["service"] == "payments"


def test_emitted_labels_carry_aggregator_and_weight_by() -> None:
    """The aggregator / weight_by config rides as documentation labels.

    A Grafana panel can read which interpretation produced the number
    alongside the bucket dimensions.
    """
    cfg = HealthScoreConfig(
        aggregate_by=["service"],
        weight_by="forecast_series_count",
    )
    points = [_qp(0.9, service="checkout"), _qp(0.7, service="checkout")]
    scores = compute_service_health_scores(points, cfg)
    assert scores[0].labels["aggregator"] == "weighted_mean"
    assert scores[0].labels["weight_by"] == "forecast_series_count"
    assert scores[0].labels["series_count"] == "2"


def test_empty_inputs_produce_no_rows() -> None:
    """An empty quality_points list emits nothing, not a NaN row."""
    cfg = HealthScoreConfig(aggregate_by=["service"])
    assert compute_service_health_scores([], cfg) == []


def test_output_is_deterministic_across_runs() -> None:
    """Bucket ordering is stable so Grafana series colours don't churn."""
    cfg = HealthScoreConfig(aggregate_by=["service"])
    points = [
        _qp(0.8, service="zeta"),
        _qp(0.6, service="alpha"),
        _qp(0.4, service="mu"),
    ]
    first = compute_service_health_scores(points, cfg)
    second = compute_service_health_scores(points, cfg)
    assert [s.labels["service"] for s in first] == [s.labels["service"] for s in second]
    assert [s.labels["service"] for s in first] == ["alpha", "mu", "zeta"]


# ---- Runner-level wiring ---------------------------------------------------


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


def _runner_config(governance: GovernanceConfig | None = None) -> Config:
    """Minimal config — the runner test mutates exporter state directly."""
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
        groups=[
            GroupConfig(name="g", queries=[QueryConfig(id="m", promql="up")]),
        ],
        governance=governance or GovernanceConfig(),
    )


def _qsnap(group_name: str, quality_points: list[QualityPoint]) -> GroupSnapshot:
    return GroupSnapshot(group=group_name, quality_points=quality_points)


@pytest.mark.asyncio
async def test_runner_refresh_aggregates_across_groups() -> None:
    """``_refresh_service_health_scores`` aggregates across every live group.

    Bypasses ``run_group`` (which would overwrite our seeded snapshots
    with empty fits) so we can isolate the cross-group aggregation
    semantics. The runner's read-path walks every configured group's
    snapshot, then publishes one row per bucket.
    """
    cfg = _runner_config(GovernanceConfig(health_score=HealthScoreConfig(aggregate_by=["service"])))
    # Two groups in the live config so the aggregator sweeps both
    # snapshots; the runner's source is unused on this path.
    cfg.groups.append(GroupConfig(name="g2", queries=[QueryConfig(id="x", promql="up")]))
    exporter = Exporter()
    exporter.update(
        _qsnap(
            "g",
            [
                _qp(0.9, service="checkout", id="latency"),
                _qp(0.7, service="checkout", id="errors"),
                _qp(0.5, service="payments", id="latency"),
            ],
        )
    )
    exporter.update(_qsnap("g2", [_qp(0.3, service="payments", id="errors")]))

    runner = Runner(config=cfg, source=_FakeSource({"up": []}), exporter=exporter)
    runner._refresh_service_health_scores()

    body = exporter.render().decode()
    assert "forecast_service_health_score{" in body
    # Per-service mean across both groups:
    # checkout = (0.9 + 0.7) / 2 = 0.8;
    # payments = (0.5 + 0.3) / 2 = 0.4.
    checkout_line = next(
        line
        for line in body.splitlines()
        if line.startswith("forecast_service_health_score{") and 'service="checkout"' in line
    )
    assert " 0.8" in checkout_line
    payments_line = next(
        line
        for line in body.splitlines()
        if line.startswith("forecast_service_health_score{") and 'service="payments"' in line
    )
    assert " 0.4" in payments_line


@pytest.mark.asyncio
async def test_runner_clears_health_score_when_feature_disabled() -> None:
    """``apply_config`` clears the gauge when ``governance.health_score`` is removed."""
    cfg = _runner_config(GovernanceConfig(health_score=HealthScoreConfig(aggregate_by=["service"])))
    exporter = Exporter()
    exporter.update(_qsnap("g", [_qp(0.9, service="checkout", id="latency")]))

    runner = Runner(config=cfg, source=_FakeSource({"up": []}), exporter=exporter)
    runner._refresh_service_health_scores()
    initial = exporter.render().decode()
    assert "forecast_service_health_score{" in initial

    # Reload with the feature turned off — the gauge must disappear.
    disabled = _runner_config(GovernanceConfig(health_score=None))
    await runner.apply_config(disabled)
    after = exporter.render().decode()
    assert "forecast_service_health_score{" not in after


@pytest.mark.asyncio
async def test_runner_skips_health_refresh_when_unconfigured() -> None:
    """A config without ``governance.health_score`` emits no gauge rows."""
    cfg = _runner_config()  # no governance block
    exporter = Exporter()
    exporter.update(_qsnap("g", [_qp(0.9, service="checkout", id="latency")]))

    runner = Runner(config=cfg, source=_FakeSource({"up": []}), exporter=exporter)
    runner._refresh_service_health_scores()
    body = exporter.render().decode()
    assert "forecast_service_health_score{" not in body


def test_exporter_health_bucket_key_is_order_stable() -> None:
    """``replace_service_health_scores`` deduplicates regardless of label-insertion order.

    Without sorting (key, value) pairs by key, two emissions for the
    same logical bucket but with different insertion orders would
    produce two distinct rows on /metrics — a silent regression that
    would only surface as an unexplained doubled row count under
    load. Sort-on-key keeps the dedup deterministic.
    """
    from promforecast.exporter import ServiceHealthScorePoint  # noqa: PLC0415

    exporter = Exporter()
    # Two points for the same conceptual bucket — same label values
    # but inserted in different orders. The second emission must
    # REPLACE the first, not add a second row.
    first = ServiceHealthScorePoint(
        labels={
            "service": "checkout",
            "environment": "prod",
            "aggregator": "weighted_mean",
            "weight_by": "forecast_series_count",
            "series_count": "10",
        },
        score=0.8,
    )
    second = ServiceHealthScorePoint(
        labels={
            # Same labels, different insertion order.
            "environment": "prod",
            "service": "checkout",
            "weight_by": "forecast_series_count",
            "aggregator": "weighted_mean",
            "series_count": "12",
        },
        score=0.9,
    )
    exporter.replace_service_health_scores([first])
    exporter.replace_service_health_scores([second])
    state = exporter._service_health_state()
    assert len(state) == 1
    assert state[0].score == 0.9
    # The render path emits exactly one row for the bucket.
    body = exporter.render().decode()
    bucket_lines = [
        line for line in body.splitlines() if line.startswith("forecast_service_health_score{")
    ]
    assert len(bucket_lines) == 1
