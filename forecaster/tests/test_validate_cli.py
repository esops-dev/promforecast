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
    # 1 * 100 * 2 emitted = 200.
    assert est.quality_lines == 1 * 100 * 2
    # Without thresholds / growth-rate emission, the capacity families
    # contribute zero to the estimate.
    assert est.threshold_eta_lines == 0
    assert est.growth_rate_lines == 0


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
