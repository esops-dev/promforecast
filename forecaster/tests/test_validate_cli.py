"""Tests for ``promforecast validate``.

The validate path has two halves: schema validation (cheap, always runs)
and ``--dry-run`` (runs a sample query + fit). The dry-run is exercised
end-to-end via docker-compose; here we cover the schema half plus the
``--dry-run`` failure path against a non-routable datasource so CI doesn't
need network egress to verify the error message shape.
"""

from __future__ import annotations

import io
from pathlib import Path

from promforecast.validate import _estimate_cardinality, run_validate


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(body)
    return path


_GOOD_CONFIG = """
apiVersion: promforecast.io/v1
datasource:
  url: http://vm:8428/
safety:
  # Default emission set (accuracy + deviation + quality + drift +
  # band_coverage at default windows) on two 500-series queries
  # projects ~21k lines — comfortably over the 5000-default cap. Bump
  # the cap so the strict-mode happy path is exercised without
  # disabling every default. Configs that genuinely fit under the
  # default cap are exercised separately.
  max_total_series: 100000
defaults:
  models: [SeasonalNaive]
  confidence_levels: [80, 95]
groups:
  - name: g1
    queries:
      - id: my_metric
        promql: up
      - id: another
        promql: rate(up[5m])
"""


def test_validate_returns_zero_on_good_config(tmp_path: Path) -> None:
    path = _write(tmp_path, _GOOD_CONFIG)
    out = io.StringIO()
    rc = run_validate(path, out=out)
    assert rc == 0
    body = out.getvalue()
    assert "OK" in body
    assert "groups:  1" in body
    assert "queries: 2" in body
    assert "Estimated cardinality" in body


def test_validate_returns_one_on_bad_yaml(tmp_path: Path) -> None:
    path = _write(tmp_path, "datasource: url: oops\nthis is: not yaml :")
    out = io.StringIO()
    rc = run_validate(path, out=out)
    assert rc == 1


def test_validate_returns_one_on_schema_error(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
apiVersion: promforecast.io/v1
datasource:
  url: http://vm:8428/
groups:
  - name: g
    queries:
      - id: bad:colon
        promql: up
""",
    )
    out = io.StringIO()
    rc = run_validate(path, out=out)
    assert rc == 1


def test_validate_returns_one_when_file_missing(tmp_path: Path) -> None:
    out = io.StringIO()
    rc = run_validate(tmp_path / "no-such-file.yaml", out=out)
    assert rc == 1


def test_cardinality_estimate_matches_doc_formula() -> None:
    """The estimate is the headline number reviewers eyeball; lock it down."""
    from datetime import timedelta  # noqa: PLC0415

    from promforecast.config import (  # noqa: PLC0415
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

    cfg = Config(
        datasource=DatasourceConfig(url="http://x"),
        server=ServerConfig(),
        safety=SafetyConfig(max_series_per_query=100, max_total_series=0),
        defaults=DefaultsConfig(
            models=["AutoARIMA", "SeasonalNaive"],
            confidence_levels=[80, 95],
            accuracy=AccuracyConfig(
                evaluate=True,
                horizons=[timedelta(hours=1), timedelta(hours=24)],
                auto_select=False,
            ),
            emission=EmissionConfig(deviation=True, quality=True),
        ),
        groups=[GroupConfig(name="g", queries=[QueryConfig(id="m", promql="up")])],
    )
    est = _estimate_cardinality(cfg)
    # 1 query * 100 series * 2 models * (1 + 2*2 levels) = 1000 forecast lines.
    assert est.forecast_lines == 1 * 100 * 2 * (1 + 2 * 2)
    # 1 * 100 * 2 models * 2 horizons * 2 (mape + mase) = 800.
    assert est.accuracy_lines == 1 * 100 * 2 * 2 * 2
    # 1 * 100 * 2 emitted * 2 levels * 2 (ratio + outside) = 800.
    assert est.deviation_lines == 1 * 100 * 2 * 2 * 2
    # 1 * 100 * 2 emitted * (1 aggregated + 2 horizons) = 600.
    # Per-horizon quality segmentation; existing alerts gating on
    # ``horizon=""`` keep matching the aggregated row.
    assert est.quality_lines == 1 * 100 * 2 * (1 + 2)
    # Revision: 1 * 100 * 2 emitted * 2 horizons = 400 (steady-state;
    # first-fit NaN rows are dropped at render).
    assert est.revision_lines == 1 * 100 * 2 * 2
    # Disagreement: 1 * 100 * 2 horizons = 200 (explicit mode, 2+ models).
    assert est.disagreement_lines == 1 * 100 * 2
    # Without thresholds / growth-rate emission, the capacity families
    # contribute zero to the estimate.
    assert est.threshold_eta_lines == 0
    assert est.growth_rate_lines == 0


def test_revision_and_disagreement_lines_dont_gate_on_evaluate() -> None:
    """Revision + disagreement emit on configured horizons, regardless of evaluate.

    Reading the central forecast at each horizon needs no backtest;
    the runtime emits these metrics even when ``accuracy.evaluate``
    is off. The estimator must match the runtime, or the
    ``--strict-cardinality`` gate falsely lets oversized configs
    through when the operator has explicitly disabled backtesting.
    """
    from datetime import timedelta  # noqa: PLC0415

    from promforecast.config import (  # noqa: PLC0415
        AccuracyConfig,
        Config,
        DatasourceConfig,
        DefaultsConfig,
        GroupConfig,
        QueryConfig,
        SafetyConfig,
        ServerConfig,
    )

    cfg = Config(
        datasource=DatasourceConfig(url="http://x"),
        server=ServerConfig(),
        safety=SafetyConfig(max_series_per_query=10, max_total_series=0),
        defaults=DefaultsConfig(
            models=["AutoARIMA", "SeasonalNaive"],
            confidence_levels=[80],
            accuracy=AccuracyConfig(
                evaluate=False,
                horizons=[timedelta(hours=1), timedelta(hours=24)],
                auto_select=False,
            ),
        ),
        groups=[GroupConfig(name="g", queries=[QueryConfig(id="m", promql="up")])],
    )
    est = _estimate_cardinality(cfg)
    # 1 query * 10 series * 2 emitted models * 2 horizons = 40.
    assert est.revision_lines == 40
    # Explicit mode + 2 models: 1 * 10 * 2 horizons = 20.
    assert est.disagreement_lines == 20


def test_disagreement_lines_suppressed_under_auto_select() -> None:
    """auto_select (including ensemble) suppresses model_disagreement entirely."""
    from datetime import timedelta  # noqa: PLC0415

    from promforecast.config import (  # noqa: PLC0415
        AccuracyConfig,
        Config,
        DatasourceConfig,
        DefaultsConfig,
        GroupConfig,
        QueryConfig,
        SafetyConfig,
        ServerConfig,
    )

    for selector in (True, "ensemble"):
        cfg = Config(
            datasource=DatasourceConfig(url="http://x"),
            server=ServerConfig(),
            safety=SafetyConfig(max_series_per_query=10, max_total_series=0),
            defaults=DefaultsConfig(
                models=["AutoARIMA", "SeasonalNaive"],
                confidence_levels=[80],
                accuracy=AccuracyConfig(
                    evaluate=True,
                    horizons=[timedelta(hours=1)],
                    auto_select=selector,
                ),
            ),
            groups=[GroupConfig(name="g", queries=[QueryConfig(id="m", promql="up")])],
        )
        est = _estimate_cardinality(cfg)
        assert est.disagreement_lines == 0, f"auto_select={selector!r} must suppress disagreement"


def test_cardinality_estimate_counts_threshold_and_growth_rate_lines() -> None:
    """Capacity ETA + growth-rate families must show up in the cardinality estimate."""
    from datetime import timedelta  # noqa: PLC0415

    from promforecast.config import (  # noqa: PLC0415
        AccuracyConfig,
        Config,
        DatasourceConfig,
        DefaultsConfig,
        EmissionConfig,
        GroupConfig,
        GrowthRateEmissionConfig,
        QueryConfig,
        QueryEmissionConfig,
        SafetyConfig,
        ServerConfig,
        ThresholdConfig,
    )

    cfg = Config(
        datasource=DatasourceConfig(url="http://x"),
        server=ServerConfig(),
        safety=SafetyConfig(max_series_per_query=100, max_total_series=0),
        defaults=DefaultsConfig(
            models=["AutoARIMA"],
            confidence_levels=[80, 95],
            accuracy=AccuracyConfig(
                evaluate=False,
                auto_select=False,
                horizons=[timedelta(hours=1)],
            ),
            emission=EmissionConfig(deviation=False, quality=False),
        ),
        groups=[
            GroupConfig(
                name="g",
                queries=[
                    QueryConfig(
                        id="m",
                        promql="up",
                        thresholds=[
                            ThresholdConfig(name="warn", value=80, comparator="gt"),
                            ThresholdConfig(name="crit", value=95, comparator="gt"),
                        ],
                        emission=QueryEmissionConfig(
                            growth_rate=GrowthRateEmissionConfig(
                                enabled=True,
                                windows=[timedelta(days=1), timedelta(days=7)],
                            ),
                        ),
                    ),
                ],
            )
        ],
    )
    est = _estimate_cardinality(cfg)
    # 100 series * 1 model * 2 thresholds * (1 point + 2 levels) = 600.
    assert est.threshold_eta_lines == 100 * 1 * 2 * (1 + 2)
    # 100 series * 1 model * 2 windows = 200.
    assert est.growth_rate_lines == 100 * 1 * 2


def test_cardinality_estimate_counts_service_health_only_when_enabled() -> None:
    """``governance.health_score`` configured => bounded estimate; absent => zero."""
    from datetime import timedelta  # noqa: PLC0415

    from promforecast.config import (  # noqa: PLC0415
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

    def _build(governance: GovernanceConfig) -> Config:
        return Config(
            datasource=DatasourceConfig(url="http://x"),
            server=ServerConfig(),
            safety=SafetyConfig(max_series_per_query=100, max_total_series=0),
            defaults=DefaultsConfig(
                models=["AutoARIMA"],
                confidence_levels=[80],
                accuracy=AccuracyConfig(
                    evaluate=True,
                    auto_select=False,
                    horizons=[timedelta(hours=1)],
                ),
                emission=EmissionConfig(quality=True),
            ),
            groups=[GroupConfig(name="g", queries=[QueryConfig(id="m", promql="up")])],
            governance=governance,
        )

    off = _estimate_cardinality(_build(GovernanceConfig()))
    on = _estimate_cardinality(
        _build(GovernanceConfig(health_score=HealthScoreConfig(aggregate_by=["service"])))
    )
    assert off.service_health_lines == 0
    # Upper bound matches the quality_lines total when the feature is on
    # (one bucket per quality_point in the worst case).
    assert on.service_health_lines == on.quality_lines
    assert on.service_health_lines > 0


def test_cardinality_estimate_counts_resource_sizing_rows() -> None:
    """Right-sizing charges 1 row per workload, plus 2 more when ``current_promql`` is set."""
    from promforecast.config import (  # noqa: PLC0415
        Config,
        DatasourceConfig,
        DefaultsConfig,
        GroupConfig,
        QueryConfig,
        RecommendationsConfig,
        ResourceSizingConfig,
        ResourceSizingWorkloadConfig,
        SafetyConfig,
        ServerConfig,
    )

    cfg = Config(
        datasource=DatasourceConfig(url="http://x"),
        server=ServerConfig(),
        safety=SafetyConfig(max_series_per_query=10, max_total_series=0),
        defaults=DefaultsConfig(models=["AutoARIMA"]),
        groups=[GroupConfig(name="g", queries=[QueryConfig(id="svc", promql="up")])],
        recommendations=RecommendationsConfig(
            resource_sizing=ResourceSizingConfig(
                enabled=True,
                workloads=[
                    # No current_promql -> 1 line (recommended only).
                    ResourceSizingWorkloadConfig(workload="api", kind="cpu", forecast_query="svc"),
                    # current_promql wired -> 3 lines (recommended + current + savings_pct).
                    ResourceSizingWorkloadConfig(
                        workload="api",
                        kind="memory",
                        forecast_query="svc",
                        current_promql="kube_pod_container_resource_requests",
                    ),
                ],
            )
        ),
    )
    est = _estimate_cardinality(cfg)
    # 1 (cpu, recommended only) + 3 (memory, recommended + current + savings_pct) = 4.
    assert est.resource_sizing_lines == 4


def test_cardinality_estimate_counts_action_cost_only_for_opted_in() -> None:
    """``forecast_threshold_action_cost`` charges one line per opted-in threshold.

    Independent of model / series / level fan-out — the metric is
    config-derived. Thresholds without ``action_cost`` configured must
    not contribute to the cardinality budget.
    """
    from datetime import timedelta  # noqa: PLC0415

    from promforecast.config import (  # noqa: PLC0415
        AccuracyConfig,
        ActionCostConfig,
        Config,
        DatasourceConfig,
        DefaultsConfig,
        GroupConfig,
        QueryConfig,
        SafetyConfig,
        ServerConfig,
        ThresholdConfig,
    )

    cfg = Config(
        datasource=DatasourceConfig(url="http://x"),
        server=ServerConfig(),
        safety=SafetyConfig(max_series_per_query=100, max_total_series=0),
        defaults=DefaultsConfig(
            models=["AutoARIMA"],
            confidence_levels=[80, 95],
            accuracy=AccuracyConfig(
                evaluate=False,
                auto_select=False,
                horizons=[timedelta(hours=1)],
            ),
        ),
        groups=[
            GroupConfig(
                name="g",
                queries=[
                    QueryConfig(
                        id="m",
                        promql="up",
                        thresholds=[
                            ThresholdConfig(
                                name="warn",
                                value=80,
                                action_cost=ActionCostConfig(type="usd", value=50.0),
                            ),
                            ThresholdConfig(
                                name="crit",
                                value=95,
                                action_cost=ActionCostConfig(type="usd", value=50_000.0),
                            ),
                            # No action_cost — must not contribute.
                            ThresholdConfig(name="info", value=50),
                        ],
                    ),
                ],
            )
        ],
    )
    est = _estimate_cardinality(cfg)
    # 2 thresholds opted in x 1 row each = 2 lines (no series fan-out).
    assert est.action_cost_lines == 2


def test_cardinality_estimate_counts_conditional_calibration_when_conformal_on() -> None:
    """The dedicated conditional-band conformal gauge must show up in the estimate.

    Each (series, model, level) emits one
    ``forecast_conditional_band_calibration_offset`` row when both
    ``accuracy.conformal.enabled`` and ``emission.conditional`` are
    true; ``promforecast validate`` must account for it or the
    estimate under-reports the fresh-calibration footprint.
    """
    from datetime import timedelta  # noqa: PLC0415

    from promforecast.config import (  # noqa: PLC0415
        AccuracyConfig,
        Config,
        ConformalConfig,
        DatasourceConfig,
        DefaultsConfig,
        EmissionConfig,
        GroupConfig,
        QueryConfig,
        QueryEmissionConfig,
        SafetyConfig,
        ServerConfig,
    )

    series_cap = 100
    models = 2
    levels = 2

    def _build(*, conformal_enabled: bool) -> Config:
        return Config(
            datasource=DatasourceConfig(url="http://x"),
            server=ServerConfig(),
            safety=SafetyConfig(max_series_per_query=series_cap, max_total_series=0),
            defaults=DefaultsConfig(
                models=["AutoARIMA", "SeasonalNaive"],
                confidence_levels=[80, 95],
                accuracy=AccuracyConfig(
                    evaluate=False,
                    auto_select=False,
                    horizons=[timedelta(hours=1)],
                    conformal=ConformalConfig(enabled=conformal_enabled),
                ),
                emission=EmissionConfig(deviation=False, quality=False),
            ),
            groups=[
                GroupConfig(
                    name="g",
                    queries=[
                        QueryConfig(
                            id="m",
                            promql="up",
                            emission=QueryEmissionConfig(conditional=True),
                        )
                    ],
                )
            ],
        )

    # Conditional snapshot + deviation are charged either way (the
    # baseline below validates that side of the formula).
    snapshot_rows = series_cap * models * (1 + 2 * levels)
    deviation_rows = series_cap * models * levels * 2
    calibration_rows = series_cap * models * levels

    off = _estimate_cardinality(_build(conformal_enabled=False))
    assert off.conditional_lines == snapshot_rows + deviation_rows
    # No calibration gauge when conformal is off — even with conditional on.
    assert off.calibration_lines == 0

    on = _estimate_cardinality(_build(conformal_enabled=True))
    # With conformal on, the unconditional calibration gauge appears
    # for every emitted (series, model, level) too — that's the
    # pre-fresh-calibration line — *and* the conditional family picks
    # up the matching new gauge.
    assert on.calibration_lines == series_cap * models * levels  # unconditional
    assert on.conditional_lines == snapshot_rows + deviation_rows + calibration_rows
    # Delta between off and on for the conditional family is exactly
    # the new gauge — protects against drift if a future change
    # widens the conditional emission and the test author forgets to
    # update the formula here.
    assert on.conditional_lines - off.conditional_lines == calibration_rows


def test_validate_strict_cardinality_passes_when_under_budget(tmp_path: Path) -> None:
    """A config whose estimated total fits inside the cap exits 0 in strict mode."""
    path = _write(tmp_path, _GOOD_CONFIG)
    out = io.StringIO()
    rc = run_validate(path, strict_cardinality=True, out=out)
    assert rc == 0


def test_validate_strict_cardinality_fails_when_over_budget(tmp_path: Path) -> None:
    """Strict mode flips the over-budget WARN into a non-zero exit code.

    Tiny ``max_total_series`` against the default emission set guarantees
    the estimate overshoots, so the test is deterministic without
    materialising a contrived large config.
    """
    path = _write(
        tmp_path,
        """
apiVersion: promforecast.io/v1
datasource:
  url: http://vm:8428/
safety:
  max_total_series: 1
defaults:
  models: [SeasonalNaive]
groups:
  - name: g
    queries:
      - id: m
        promql: up
""",
    )
    out = io.StringIO()
    rc = run_validate(path, strict_cardinality=True, out=out)
    assert rc == 1


def test_validate_advisory_mode_returns_zero_when_over_budget(tmp_path: Path) -> None:
    """Default (non-strict) mode keeps the historical advisory-only behaviour."""
    path = _write(
        tmp_path,
        """
apiVersion: promforecast.io/v1
datasource:
  url: http://vm:8428/
safety:
  max_total_series: 1
defaults:
  models: [SeasonalNaive]
groups:
  - name: g
    queries:
      - id: m
        promql: up
""",
    )
    out = io.StringIO()
    rc = run_validate(path, out=out)
    assert rc == 0
    assert "WARN: total exceeds safety.max_total_series" in out.getvalue()


def test_validate_dry_run_reports_failure_when_datasource_unreachable(tmp_path: Path) -> None:
    """Dry-run failures must not crash the CLI; they return rc=2."""
    path = _write(
        tmp_path,
        """
apiVersion: promforecast.io/v1
datasource:
  url: http://127.0.0.1:1
  timeout: 1s
defaults:
  models: [SeasonalNaive]
groups:
  - name: g
    queries:
      - id: m
        promql: up
""",
    )
    out = io.StringIO()
    # Override pointing at a deliberately unreachable port so the CLI must
    # surface the failure as exit code 2 rather than blocking on a real
    # network call.
    rc = run_validate(path, dry_run=True, datasource_url_override="http://127.0.0.1:1", out=out)
    assert rc == 2
    body = out.getvalue()
    assert "OK" in body  # schema check still passes
    assert "Dry-run" in body
