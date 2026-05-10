"""Tests for the YAML config loader."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

import promforecast.config as config_module
from promforecast.exporter import InvalidMetricNameError


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(body)
    return path


def test_load_minimal(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
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


def test_load_rejects_legacy_flat_accuracy_fields(tmp_path: Path) -> None:
    """Old (flat) ``evaluate`` / ``emit_quality`` etc. now live under nested
    blocks; the schema must reject the flat form so stale configs fail loudly
    rather than silently using the field defaults."""
    path = _write(
        tmp_path,
        """
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
