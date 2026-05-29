"""Config-schema tests for the ecosystem/integration features.

Covers the webhook + calendar_file regressor types, the per-query canary
comparison block, the top-level triggers and annotations blocks, and the
shared signing block.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import promforecast.config as config_module

_HEADER = """
apiVersion: promforecast.io/v1
datasource:
  url: http://vm:8428/
"""


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(_HEADER + body)
    return path


# -- webhook regressor ------------------------------------------------------


def test_webhook_regressor_loads(tmp_path: Path) -> None:
    cfg = config_module.load(
        _write(
            tmp_path,
            """
groups:
  - name: g1
    queries:
      - id: m
        promql: up
        regressors:
          - id: incidents
            type: webhook
            url: https://pagerduty.example/incidents
            timeout: 5s
            cache_ttl: 1h
            headers:
              Authorization: Bearer abc
            signing:
              algorithm: hmac-sha256
              secret_ref: ${HOOK_SECRET}
""",
        )
    )
    reg = cfg.groups[0].queries[0].regressors[0]
    assert reg.type == "webhook"
    assert reg.url == "https://pagerduty.example/incidents"
    assert reg.signing is not None
    assert reg.signing.secret_ref == "${HOOK_SECRET}"


def test_webhook_regressor_requires_url(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="webhook regressors require a non-empty url"):
        config_module.load(
            _write(
                tmp_path,
                """
groups:
  - name: g1
    queries:
      - id: m
        promql: up
        regressors:
          - id: incidents
            type: webhook
""",
            )
        )


def test_webhook_regressor_rejects_promql(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="webhook regressors must not set promql"):
        config_module.load(
            _write(
                tmp_path,
                """
groups:
  - name: g1
    queries:
      - id: m
        promql: up
        regressors:
          - id: incidents
            type: webhook
            url: https://x/y
            promql: up
""",
            )
        )


# -- calendar_file regressor ------------------------------------------------


def test_calendar_file_regressor_loads(tmp_path: Path) -> None:
    cfg = config_module.load(
        _write(
            tmp_path,
            """
groups:
  - name: g1
    queries:
      - id: m
        promql: up
        regressors:
          - id: release_freeze
            type: calendar_file
            path: /etc/promforecast/calendars/release-freezes.yaml
            expand: true
""",
        )
    )
    reg = cfg.groups[0].queries[0].regressors[0]
    assert reg.type == "calendar_file"
    assert reg.path.endswith("release-freezes.yaml")
    assert reg.expand is True


def test_calendar_file_regressor_requires_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="calendar_file regressors require a non-empty path"):
        config_module.load(
            _write(
                tmp_path,
                """
groups:
  - name: g1
    queries:
      - id: m
        promql: up
        regressors:
          - id: cal
            type: calendar_file
""",
            )
        )


# -- canary comparison ------------------------------------------------------


def test_canary_comparison_loads(tmp_path: Path) -> None:
    cfg = config_module.load(
        _write(
            tmp_path,
            """
groups:
  - name: g1
    queries:
      - id: m
        promql: http_requests_total
        canary:
          enabled: true
          canary_selector: '{version="canary"}'
          baseline_selector: '{version!="canary"}'
""",
        )
    )
    canary = cfg.groups[0].queries[0].canary
    assert canary.enabled is True
    assert canary.canary_selector == '{version="canary"}'


def test_canary_comparison_requires_selectors(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="canary_selector is required"):
        config_module.load(
            _write(
                tmp_path,
                """
groups:
  - name: g1
    queries:
      - id: m
        promql: up
        canary:
          enabled: true
          baseline_selector: '{version!="canary"}'
""",
            )
        )


# -- triggers ---------------------------------------------------------------


def test_triggers_load(tmp_path: Path) -> None:
    cfg = config_module.load(
        _write(
            tmp_path,
            """
groups:
  - name: g1
    queries:
      - id: m
        promql: up
triggers:
  - name: disk-fill-soon
    condition: 'node_filesystem_avail_bytes_forecast{horizon="24h"} < 0'
    webhook_url: https://hooks.slack.example/abc
    repeat_interval: 1h
    payload_template: '{"text": "disk fill predicted"}'
    signing:
      secret_ref: ${TRIGGER_SECRET}
""",
        )
    )
    assert cfg.triggers[0].name == "disk-fill-soon"
    assert cfg.triggers[0].signing is not None


def test_triggers_reject_duplicate_names(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="appears twice"):
        config_module.load(
            _write(
                tmp_path,
                """
groups:
  - name: g1
    queries:
      - id: m
        promql: up
triggers:
  - name: dup
    condition: up
    webhook_url: https://x/y
  - name: dup
    condition: down
    webhook_url: https://x/z
""",
            )
        )


# -- annotations ------------------------------------------------------------


def test_annotations_load(tmp_path: Path) -> None:
    cfg = config_module.load(
        _write(
            tmp_path,
            """
groups:
  - name: g1
    queries:
      - id: m
        promql: up
annotations:
  enabled: true
  grafana_url: https://grafana.example.com
  api_token_ref: ${GRAFANA_TOKEN}
  events: [drift, deploy_risk, deviation_outside_band]
""",
        )
    )
    assert cfg.annotations.enabled is True
    assert "drift" in cfg.annotations.events
    # Defaults to the "elevated" gate for deploy-risk annotations.
    assert cfg.annotations.deploy_risk_min_score == 0.5


def test_annotations_deploy_risk_min_score_override(tmp_path: Path) -> None:
    cfg = config_module.load(
        _write(
            tmp_path,
            """
groups:
  - name: g1
    queries:
      - id: m
        promql: up
annotations:
  enabled: true
  grafana_url: https://grafana.example.com
  events: [deploy_risk]
  deploy_risk_min_score: 0.8
""",
        )
    )
    assert cfg.annotations.deploy_risk_min_score == 0.8


def test_annotations_deploy_risk_min_score_out_of_range(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"deploy_risk_min_score"):
        config_module.load(
            _write(
                tmp_path,
                """
groups:
  - name: g1
    queries:
      - id: m
        promql: up
annotations:
  enabled: true
  grafana_url: https://grafana.example.com
  events: [deploy_risk]
  deploy_risk_min_score: 1.5
""",
            )
        )


def test_annotations_enabled_requires_url(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"annotations\.grafana_url is empty"):
        config_module.load(
            _write(
                tmp_path,
                """
groups:
  - name: g1
    queries:
      - id: m
        promql: up
annotations:
  enabled: true
  events: [drift]
""",
            )
        )
