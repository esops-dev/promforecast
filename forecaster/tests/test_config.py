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
    assert cfg.defaults.models == ["AutoARIMA", "SeasonalNaive"]
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
