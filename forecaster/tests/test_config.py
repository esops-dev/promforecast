"""Tests for the YAML config loader."""

from __future__ import annotations

import warnings
from datetime import timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

import promforecast.config as config_module
from promforecast.config import (
    CURRENT_API_VERSION,
    UnsupportedApiVersionError,
)
from promforecast.exporter import InvalidMetricNameError


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(body)
    return path


def test_load_minimal(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
apiVersion: promforecast.io/v1
datasource:
  url: http://vm:8428/
groups:
  - name: g1
    queries:
      - id: my_metric
        promql: up
""",
    )
    cfg = config_module.load(path)
    assert cfg.datasource.url == "http://vm:8428/"
    assert cfg.server.refresh_interval == timedelta(hours=1)
    assert [m.name for m in cfg.defaults.models] == ["AutoARIMA", "SeasonalNaive"]
    assert cfg.groups[0].queries[0].id == "my_metric"


def test_load_parses_prometheus_durations(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
apiVersion: promforecast.io/v1
datasource:
  url: http://vm:8428/
  timeout: 45s
server:
  refresh_interval: 30m
defaults:
  lookback: 14d
  step: 5m
  horizon: 6h
groups:
  - name: g1
    queries:
      - id: my_metric
        promql: up
""",
    )
    cfg = config_module.load(path)
    assert cfg.datasource.timeout == timedelta(seconds=45)
    assert cfg.server.refresh_interval == timedelta(minutes=30)
    assert cfg.defaults.lookback == timedelta(days=14)
    assert cfg.defaults.step == timedelta(minutes=5)
    assert cfg.defaults.horizon == timedelta(hours=6)


def test_load_rejects_colon_in_id(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
apiVersion: promforecast.io/v1
datasource:
  url: http://vm:8428/
groups:
  - name: g1
    queries:
      - id: bad:id
        promql: up
""",
    )
    with pytest.raises(InvalidMetricNameError):
        config_module.load(path)


def test_load_rejects_unknown_field(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
apiVersion: promforecast.io/v1
datasource:
  url: http://vm:8428/
groups:
  - name: g1
    queries:
      - id: my_metric
        promql: up
mystery_top_level: 42
""",
    )
    with pytest.raises(ValidationError):
        config_module.load(path)


def test_load_accepts_nested_accuracy_and_emission_blocks(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
apiVersion: promforecast.io/v1
datasource:
  url: http://vm:8428/
defaults:
  accuracy:
    evaluate: false
    horizons: [10m, 1h]
    auto_select: true
    fallback_model: SeasonalNaive
    fallback_mape_threshold: 25
  emission:
    deviation: false
    quality: true
groups:
  - name: g
    queries:
      - id: m
        promql: up
""",
    )
    cfg = config_module.load(path)
    assert cfg.defaults.accuracy.evaluate is False
    assert cfg.defaults.accuracy.auto_select is True
    assert cfg.defaults.accuracy.fallback_mape_threshold == 25
    assert len(cfg.defaults.accuracy.horizons) == 2
    assert cfg.defaults.emission.deviation is False
    assert cfg.defaults.emission.quality is True


def test_load_accepts_models_as_strings_or_dicts(tmp_path: Path) -> None:
    """Both shorthand string entries and dict entries must coerce to ModelSpec."""
    path = _write(
        tmp_path,
        """
apiVersion: promforecast.io/v1
datasource:
  url: http://vm:8428/
defaults:
  models:
    - SeasonalNaive
    - name: AutoARIMA
      params:
        max_p: 2
        stepwise: false
groups:
  - name: g
    queries:
      - id: m
        promql: up
""",
    )
    cfg = config_module.load(path)
    assert [s.name for s in cfg.defaults.models] == ["SeasonalNaive", "AutoARIMA"]
    assert cfg.defaults.models[0].params == {}
    assert cfg.defaults.models[1].params == {"max_p": 2, "stepwise": False}


def test_load_rejects_unknown_field_inside_model_spec(tmp_path: Path) -> None:
    """Strict ``extra=forbid`` keeps typos in ``ModelSpec`` from being silently
    dropped (e.g. ``param`` instead of ``params``)."""
    path = _write(
        tmp_path,
        """
apiVersion: promforecast.io/v1
datasource:
  url: http://vm:8428/
defaults:
  models:
    - name: AutoARIMA
      param: {}
groups:
  - name: g
    queries:
      - id: m
        promql: up
""",
    )
    with pytest.raises(ValidationError):
        config_module.load(path)


def test_load_rejects_calendar_regressor_with_unknown_id(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
apiVersion: promforecast.io/v1
datasource:
  url: http://vm:8428/
groups:
  - name: g
    queries:
      - id: m
        promql: up
        regressors:
          - id: minute_of_hour
            type: calendar
""",
    )
    with pytest.raises(ValueError, match="calendar regressor id"):
        config_module.load(path)


def test_load_rejects_calendar_regressor_with_promql(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
apiVersion: promforecast.io/v1
datasource:
  url: http://vm:8428/
groups:
  - name: g
    queries:
      - id: m
        promql: up
        regressors:
          - id: hour_of_day
            type: calendar
            promql: up
""",
    )
    with pytest.raises(ValueError, match="must not set promql"):
        config_module.load(path)


def test_load_rejects_custom_regressor_without_promql(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
apiVersion: promforecast.io/v1
datasource:
  url: http://vm:8428/
groups:
  - name: g
    queries:
      - id: m
        promql: up
        regressors:
          - id: deploys
            type: custom
""",
    )
    with pytest.raises(ValueError, match="custom regressors require"):
        config_module.load(path)


def test_load_rejects_remote_write_enabled_without_url(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
apiVersion: promforecast.io/v1
datasource:
  url: http://vm:8428/
sink:
  remote_write:
    enabled: true
groups:
  - name: g
    queries:
      - id: m
        promql: up
""",
    )
    with pytest.raises(ValueError, match=r"sink\.remote_write\.url is empty"):
        config_module.load(path)


def test_load_accepts_remote_write_with_headers(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
apiVersion: promforecast.io/v1
datasource:
  url: http://vm:8428/
sink:
  remote_write:
    enabled: true
    url: http://vm:8428/api/v1/import/prometheus
    timeout: 5s
    chunk_size: 5000
    headers:
      X-Forwarded-For: forecaster
groups:
  - name: g
    queries:
      - id: m
        promql: up
""",
    )
    cfg = config_module.load(path)
    assert cfg.sink.remote_write.enabled is True
    assert cfg.sink.remote_write.url.endswith("/api/v1/import/prometheus")
    assert cfg.sink.remote_write.headers == {"X-Forwarded-For": "forecaster"}
    assert cfg.sink.remote_write.chunk_size == 5000


def test_load_rejects_remote_write_chunk_size_below_one(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
apiVersion: promforecast.io/v1
datasource:
  url: http://vm:8428/
sink:
  remote_write:
    enabled: true
    url: http://vm:8428/x
    chunk_size: 0
groups:
  - name: g
    queries:
      - id: m
        promql: up
""",
    )
    with pytest.raises(ValidationError):
        config_module.load(path)


def test_load_rejects_datasource_keepalive_above_concurrent(tmp_path: Path) -> None:
    """``max_keepalive_connections`` capped at ``max_concurrent_requests``.

    Without this validator the operator could set a higher keepalive
    expecting more idle connections to be retained, but httpx caps
    keepalive at the total connection cap at runtime — so the larger
    value silently has no effect. Better to reject at load time.
    """
    path = _write(
        tmp_path,
        """
apiVersion: promforecast.io/v1
datasource:
  url: http://vm:8428/
  max_concurrent_requests: 4
  max_keepalive_connections: 8
groups:
  - name: g
    queries:
      - id: m
        promql: up
""",
    )
    with pytest.raises(ValidationError, match="max_keepalive_connections"):
        config_module.load(path)


def test_load_rejects_remote_write_keepalive_above_connections(tmp_path: Path) -> None:
    """Same intent as the datasource validator, on the sink side."""
    path = _write(
        tmp_path,
        """
apiVersion: promforecast.io/v1
datasource:
  url: http://vm:8428/
sink:
  remote_write:
    enabled: true
    url: http://vm:8428/x
    max_connections: 2
    max_keepalive_connections: 4
groups:
  - name: g
    queries:
      - id: m
        promql: up
""",
    )
    with pytest.raises(ValidationError, match="max_keepalive_connections"):
        config_module.load(path)


def test_load_accepts_declared_api_version(tmp_path: Path) -> None:
    """An explicit ``apiVersion`` matching the current schema must validate
    silently — no warning, no rewrite."""
    path = _write(
        tmp_path,
        """
apiVersion: promforecast.io/v1
datasource:
  url: http://vm:8428/
groups:
  - name: g
    queries:
      - id: m
        promql: up
""",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any DeprecationWarning would fail the test
        cfg = config_module.load(path)
    assert cfg.api_version == CURRENT_API_VERSION


def test_load_warns_when_api_version_missing(tmp_path: Path) -> None:
    """Configs predating the versioning roll-out keep loading but emit a
    DeprecationWarning so operators know to add ``apiVersion`` before
    the schema freezes."""
    path = _write(
        tmp_path,
        """
datasource:
  url: http://vm:8428/
groups:
  - name: g
    queries:
      - id: m
        promql: up
""",
    )
    with pytest.warns(DeprecationWarning, match="missing 'apiVersion'"):
        cfg = config_module.load(path)
    # Back-fill: the loader injects the current default so downstream code
    # never has to handle ``None``.
    assert cfg.api_version == CURRENT_API_VERSION


def test_load_rejects_unknown_api_version(tmp_path: Path) -> None:
    """Unknown apiVersion strings (typos, not-yet-released versions) raise
    a precise error rather than the noisy ``Literal[...]`` mismatch
    pydantic would otherwise produce."""
    path = _write(
        tmp_path,
        """
apiVersion: promforecast.io/v2
datasource:
  url: http://vm:8428/
groups:
  - name: g
    queries:
      - id: m
        promql: up
""",
    )
    with pytest.raises(UnsupportedApiVersionError, match="unsupported apiVersion"):
        config_module.load(path)


def test_load_accepts_deprecated_v1alpha1_with_warning(tmp_path: Path) -> None:
    """``v1alpha1`` is kept as a backward-compatible alias of v1 so configs
    committed during the alpha window keep loading on a stable-schema
    forecaster — but the loader emits a DeprecationWarning so operators
    retag at their next opportunity. The schemas are byte-identical:
    nothing else needs to change."""
    path = _write(
        tmp_path,
        """
apiVersion: promforecast.io/v1alpha1
datasource:
  url: http://vm:8428/
groups:
  - name: g
    queries:
      - id: m
        promql: up
""",
    )
    with pytest.warns(DeprecationWarning, match="v1alpha1.*deprecated"):
        cfg = config_module.load(path)
    # Loader preserves the declared version verbatim so /config still
    # reflects what the operator wrote — they're the ones who decide when
    # to retag.
    assert cfg.api_version == "promforecast.io/v1alpha1"


def test_load_rejects_top_level_non_mapping(tmp_path: Path) -> None:
    """Non-mapping YAML (a list at top level, an empty file, a scalar)
    should fail with a clear error rather than crashing inside pydantic."""
    path = _write(tmp_path, "- foo\n- bar\n")
    with pytest.raises(ValueError, match="YAML mapping"):
        config_module.load(path)


def test_load_rejects_legacy_flat_accuracy_fields(tmp_path: Path) -> None:
    """Old (flat) ``evaluate`` / ``emit_quality`` etc. now live under nested
    blocks; the schema must reject the flat form so stale configs fail loudly
    rather than silently using the field defaults."""
    path = _write(
        tmp_path,
        """
apiVersion: promforecast.io/v1
datasource:
  url: http://vm:8428/
defaults:
  evaluate: true
  emit_quality: true
groups:
  - name: g
    queries:
      - id: m
        promql: up
""",
    )
    with pytest.raises(ValidationError):
        config_module.load(path)


def test_load_accepts_preprocess_block(tmp_path: Path) -> None:
    """Per-group preprocess block accepts gaps/outliers/change_point."""
    path = _write(
        tmp_path,
        """
apiVersion: promforecast.io/v1
datasource:
  url: http://vm:8428/
groups:
  - name: g
    preprocess:
      gaps:
        strategy: linear_interp
        max_gap: 10m
      outliers:
        enabled: true
        method: hampel
        sensitivity: high
      change_point:
        enabled: true
        method: ruptures
        action: truncate_lookback
        min_post_break_points: 100
    queries:
      - id: m
        promql: up
""",
    )
    cfg = config_module.load(path)
    g = cfg.groups[0]
    assert g.preprocess.gaps.strategy == "linear_interp"
    assert g.preprocess.gaps.max_gap == timedelta(minutes=10)
    assert g.preprocess.outliers.enabled is True
    assert g.preprocess.outliers.sensitivity == "high"
    assert g.preprocess.change_point.enabled is True
    assert g.preprocess.change_point.min_post_break_points == 100


def test_load_accepts_conformal_block(tmp_path: Path) -> None:
    """Defaults accept an accuracy.conformal block."""
    path = _write(
        tmp_path,
        """
apiVersion: promforecast.io/v1
datasource:
  url: http://vm:8428/
defaults:
  accuracy:
    conformal:
      enabled: true
      method: split
      holdout_fraction: 0.25
groups:
  - name: g
    queries:
      - id: m
        promql: up
""",
    )
    cfg = config_module.load(path)
    assert cfg.defaults.accuracy.conformal.enabled is True
    assert cfg.defaults.accuracy.conformal.holdout_fraction == 0.25


def test_load_rejects_invalid_conformal_holdout(tmp_path: Path) -> None:
    """``holdout_fraction`` must be in (0, 1)."""
    path = _write(
        tmp_path,
        """
apiVersion: promforecast.io/v1
datasource:
  url: http://vm:8428/
defaults:
  accuracy:
    conformal:
      enabled: true
      holdout_fraction: 1.5
groups:
  - name: g
    queries:
      - id: m
        promql: up
""",
    )
    with pytest.raises(ValidationError):
        config_module.load(path)


def test_load_rejects_unknown_outlier_method(tmp_path: Path) -> None:
    """Outlier method must be one of the Literal choices."""
    path = _write(
        tmp_path,
        """
apiVersion: promforecast.io/v1
datasource:
  url: http://vm:8428/
groups:
  - name: g
    preprocess:
      outliers:
        enabled: true
        method: nonsense
    queries:
      - id: m
        promql: up
""",
    )
    with pytest.raises(ValidationError):
        config_module.load(path)


def test_load_accepts_thresholds_and_growth_rate(tmp_path: Path) -> None:
    """Per-query ``thresholds`` and ``emission.growth_rate`` load correctly."""
    path = _write(
        tmp_path,
        """
apiVersion: promforecast.io/v1
datasource:
  url: http://vm:8428/
groups:
  - name: g
    queries:
      - id: disk_used_pct
        promql: up
        thresholds:
          - name: warn
            value: 80
            comparator: gt
          - name: critical
            value: 95
            comparator: gt
        emission:
          growth_rate:
            enabled: true
            windows: [1d, 7d]
            units: percent
""",
    )
    cfg = config_module.load(path)
    query = cfg.groups[0].queries[0]
    assert [t.name for t in query.thresholds] == ["warn", "critical"]
    assert query.thresholds[0].comparator == "gt"
    assert query.emission.growth_rate.enabled is True
    assert query.emission.growth_rate.windows == [timedelta(days=1), timedelta(days=7)]
    assert query.emission.growth_rate.units == "percent"


def test_load_rejects_duplicate_threshold_name(tmp_path: Path) -> None:
    """Threshold names must be unique within a query (they become label values)."""
    path = _write(
        tmp_path,
        """
apiVersion: promforecast.io/v1
datasource:
  url: http://vm:8428/
groups:
  - name: g
    queries:
      - id: m
        promql: up
        thresholds:
          - name: warn
            value: 80
          - name: warn
            value: 90
""",
    )
    with pytest.raises(ValueError, match="duplicate threshold name"):
        config_module.load(path)


def test_load_accepts_threshold_with_alert_within_and_severity(tmp_path: Path) -> None:
    """A threshold with ``alert_within`` + ``severity`` round-trips through the loader."""
    path = _write(
        tmp_path,
        """
apiVersion: promforecast.io/v1
datasource:
  url: http://vm:8428/
groups:
  - name: g
    queries:
      - id: disk_used_pct
        promql: up
        thresholds:
          - name: warn
            value: 80
            alert_within: 6h
            severity: warning
""",
    )
    cfg = config_module.load(path)
    threshold = cfg.groups[0].queries[0].thresholds[0]
    assert threshold.alert_within == timedelta(hours=6)
    assert threshold.severity == "warning"


def test_load_rejects_threshold_with_severity_but_no_alert_within(tmp_path: Path) -> None:
    """``severity`` without ``alert_within`` is the half-configured-shape error."""
    path = _write(
        tmp_path,
        """
apiVersion: promforecast.io/v1
datasource:
  url: http://vm:8428/
groups:
  - name: g
    queries:
      - id: m
        promql: up
        thresholds:
          - name: warn
            value: 80
            severity: warning
""",
    )
    with pytest.raises(ValueError, match="alert_within and severity must be set together"):
        config_module.load(path)


def test_load_rejects_threshold_with_alert_within_but_no_severity(tmp_path: Path) -> None:
    """``alert_within`` without ``severity`` is the inverse half-configured error."""
    path = _write(
        tmp_path,
        """
apiVersion: promforecast.io/v1
datasource:
  url: http://vm:8428/
groups:
  - name: g
    queries:
      - id: m
        promql: up
        thresholds:
          - name: warn
            value: 80
            alert_within: 6h
""",
    )
    with pytest.raises(ValueError, match="alert_within and severity must be set together"):
        config_module.load(path)


def test_load_rejects_zero_alert_within(tmp_path: Path) -> None:
    """``alert_within: 0s`` is rejected — every finite ETA satisfies < 0 is False."""
    path = _write(
        tmp_path,
        """
apiVersion: promforecast.io/v1
datasource:
  url: http://vm:8428/
groups:
  - name: g
    queries:
      - id: m
        promql: up
        thresholds:
          - name: warn
            value: 80
            alert_within: 0s
            severity: warning
""",
    )
    with pytest.raises(ValueError, match="alert_within must be a positive duration"):
        config_module.load(path)


def test_load_rejects_growth_rate_enabled_without_windows(tmp_path: Path) -> None:
    """``growth_rate.enabled: true`` with empty windows must fail at load time."""
    path = _write(
        tmp_path,
        """
apiVersion: promforecast.io/v1
datasource:
  url: http://vm:8428/
groups:
  - name: g
    queries:
      - id: m
        promql: up
        emission:
          growth_rate:
            enabled: true
            windows: []
""",
    )
    with pytest.raises(ValueError, match="at least one window"):
        config_module.load(path)


def test_load_rejects_cold_start_with_empty_promql(tmp_path: Path) -> None:
    """``cold_start.borrow_from`` requires a non-empty PromQL — pydantic
    accepts ``""`` as a valid ``str``, so the validator catches the
    operator typo at boot rather than letting the borrow query 4xx the
    TSDB at the first cold-start refresh.
    """
    path = _write(
        tmp_path,
        """
apiVersion: promforecast.io/v1
datasource:
  url: http://vm:8428/
groups:
  - name: g
    queries:
      - id: m
        promql: up
        cold_start:
          borrow_from:
            promql: ""
            min_siblings: 3
""",
    )
    with pytest.raises(ValueError, match=r"cold_start\.borrow_from: promql must be"):
        config_module.load(path)


def test_load_accepts_cold_start_with_promql(tmp_path: Path) -> None:
    """A valid cold-start config round-trips through the loader unchanged."""
    path = _write(
        tmp_path,
        """
apiVersion: promforecast.io/v1
datasource:
  url: http://vm:8428/
groups:
  - name: g
    queries:
      - id: m
        promql: up
        cold_start:
          borrow_from:
            promql: peer_metric
            min_siblings: 5
          max_quality_score: 0.4
""",
    )
    cfg = config_module.load(path)
    q = cfg.groups[0].queries[0]
    assert q.cold_start.borrow_from is not None
    assert q.cold_start.borrow_from.promql == "peer_metric"
    assert q.cold_start.borrow_from.min_siblings == 5
    assert q.cold_start.max_quality_score == 0.4


def test_load_rejects_cold_start_with_intermittent_profile(tmp_path: Path) -> None:
    """Cold-start + ``data_profile: intermittent`` is rejected at load time.

    Median-normalising sparse zero-heavy series produces a degenerate
    prior; the runner skips the combination silently, which surprises
    operators who configured both expecting a forecast.
    """
    path = _write(
        tmp_path,
        """
apiVersion: promforecast.io/v1
datasource:
  url: http://vm:8428/
groups:
  - name: g
    queries:
      - id: m
        promql: up
        data_profile: intermittent
        cold_start:
          borrow_from:
            promql: peer_metric
""",
    )
    with pytest.raises(ValueError, match=r"cold_start.*intermittent"):
        config_module.load(path)


def test_load_rejects_cold_start_with_zero_max_quality_score(tmp_path: Path) -> None:
    """``max_quality_score <= 0`` would fire ``ForecastQualityZero`` every cycle.

    Pydantic's ``ge=0`` lets 0 slip through; the semantic check at load
    time rejects it so operators see a clear error rather than a noisy
    alert backlog.
    """
    path = _write(
        tmp_path,
        """
apiVersion: promforecast.io/v1
datasource:
  url: http://vm:8428/
groups:
  - name: g
    queries:
      - id: m
        promql: up
        cold_start:
          borrow_from:
            promql: peer_metric
          max_quality_score: 0.0
""",
    )
    with pytest.raises(ValueError, match=r"max_quality_score must be > 0"):
        config_module.load(path)


def test_load_accepts_damped_models(tmp_path: Path) -> None:
    """The damped-trend models register and parse via the same model array."""
    path = _write(
        tmp_path,
        """
apiVersion: promforecast.io/v1
datasource:
  url: http://vm:8428/
defaults:
  models:
    - name: AutoETSDamped
      params:
        damping_parameter: 0.95
    - name: AutoARIMADamped
      params:
        damping_parameter: 0.98
groups:
  - name: g
    queries:
      - id: m
        promql: up
""",
    )
    cfg = config_module.load(path)
    names = [spec.name for spec in cfg.defaults.models]
    assert "AutoETSDamped" in names
    assert "AutoARIMADamped" in names
    ets = next(s for s in cfg.defaults.models if s.name == "AutoETSDamped")
    assert ets.params == {"damping_parameter": 0.95}


def test_load_rejects_le_drain_with_alert_within(tmp_path: Path) -> None:
    """``alert_within`` + ``le_drain`` is rejected — will_breach would invert.

    The will_breach gauge fires when the ETA is small; for a drain threshold
    that means "queue will drain soon" — the opposite of a predictive
    breach. The combination must fail at load time so operators don't
    silently ship the inverted alert.
    """
    path = _write(
        tmp_path,
        """
apiVersion: promforecast.io/v1
datasource:
  url: http://vm:8428/
groups:
  - name: g
    queries:
      - id: kafka_consumer_lag
        promql: up
        thresholds:
          - name: drained
            value: 0
            comparator: le_drain
            alert_within: 4h
            severity: warning
""",
    )
    with pytest.raises(ValueError, match="alert_within \\+ severity are not supported on le_drain"):
        config_module.load(path)


def test_load_accepts_slo_block(tmp_path: Path) -> None:
    """A query with ``slo:`` and multiple windows loads cleanly."""
    path = _write(
        tmp_path,
        """
apiVersion: promforecast.io/v1
datasource:
  url: http://vm:8428/
groups:
  - name: g
    queries:
      - id: http_error_ratio
        promql: up
        slo:
          objective: 99.9
          windows: [5m, 1h, 30d]
""",
    )
    cfg = config_module.load(path)
    slo = cfg.groups[0].queries[0].slo
    assert slo is not None
    assert slo.objective == 99.9
    assert slo.windows == [timedelta(minutes=5), timedelta(hours=1), timedelta(days=30)]


def test_load_rejects_slo_objective_at_bounds(tmp_path: Path) -> None:
    """``objective`` must be strictly inside ``(0, 100)`` — pydantic enforces."""
    path = _write(
        tmp_path,
        """
apiVersion: promforecast.io/v1
datasource:
  url: http://vm:8428/
groups:
  - name: g
    queries:
      - id: m
        promql: up
        slo:
          objective: 100
""",
    )
    with pytest.raises(ValueError, match="objective"):
        config_module.load(path)


def test_load_rejects_slo_empty_windows(tmp_path: Path) -> None:
    """An explicitly empty ``windows: []`` must fail (default is non-empty)."""
    path = _write(
        tmp_path,
        """
apiVersion: promforecast.io/v1
datasource:
  url: http://vm:8428/
groups:
  - name: g
    queries:
      - id: m
        promql: up
        slo:
          objective: 99.9
          windows: []
""",
    )
    with pytest.raises(ValueError, match="at least one window is required"):
        config_module.load(path)


def test_load_accepts_recommended_windows_block(tmp_path: Path) -> None:
    """A query with ``recommended_windows.enabled: true`` round-trips cleanly."""
    path = _write(
        tmp_path,
        """
apiVersion: promforecast.io/v1
datasource:
  url: http://vm:8428/
groups:
  - name: g
    queries:
      - id: node_cpu_busy_pct
        promql: up
        recommended_windows:
          enabled: true
          duration: 15m
          lookahead: 24h
          count: 5
""",
    )
    cfg = config_module.load(path)
    rw = cfg.groups[0].queries[0].recommended_windows
    assert rw.enabled is True
    assert rw.duration == timedelta(minutes=15)
    assert rw.lookahead == timedelta(hours=24)
    assert rw.count == 5


def test_load_rejects_recommended_windows_duration_larger_than_lookahead(
    tmp_path: Path,
) -> None:
    """A window longer than the lookahead can never be recommended — reject."""
    path = _write(
        tmp_path,
        """
apiVersion: promforecast.io/v1
datasource:
  url: http://vm:8428/
groups:
  - name: g
    queries:
      - id: m
        promql: up
        recommended_windows:
          enabled: true
          duration: 2h
          lookahead: 1h
""",
    )
    with pytest.raises(ValueError, match="must be <= lookahead"):
        config_module.load(path)
