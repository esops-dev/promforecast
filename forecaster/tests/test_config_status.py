"""Tests for config status surfaces: hash metric + ``/config`` endpoint.

The story is "operators must be able to confirm what config the forecaster
is running". That breaks down into:

* ``forecast_config_hash{sha256="..."}`` rendered on /metrics, refreshed
  on every successful reload.
* The optional ``/config`` endpoint returning a JSON view of the parsed
  config with secrets redacted.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from promforecast import config as config_module
from promforecast.exporter import Exporter
from promforecast.main import _build_app, _compute_config_hash
from promforecast.reload import ConfigReloader
from promforecast.runner import Runner
from promforecast.source import PromSource


class _NullSource(PromSource):
    """Stand-in source with no I/O; the hash + endpoint tests don't fit."""

    def __init__(self) -> None: ...

    async def query_range(self, *_args: object, **_kwargs: object) -> list:  # type: ignore[type-arg]
        return []

    async def aclose(self) -> None:
        return None


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(body)
    return path


_BASE_CONFIG = """
apiVersion: promforecast.io/v1
datasource:
  url: http://vm:8428/
server:
  refresh_interval: 1h
defaults:
  models: [SeasonalNaive]
groups:
  - name: g1
    queries:
      - id: m
        promql: up
"""


# -- compute_config_hash ---------------------------------------------------


def test_compute_config_hash_matches_sha256_of_file_bytes(tmp_path: Path) -> None:
    body = _BASE_CONFIG
    path = _write(tmp_path, body)
    expected = hashlib.sha256(body.encode()).hexdigest()
    assert _compute_config_hash(path) == expected


def test_compute_config_hash_returns_empty_when_file_missing(tmp_path: Path) -> None:
    assert _compute_config_hash(tmp_path / "does-not-exist.yaml") == ""


# -- forecast_config_hash metric ------------------------------------------


def test_metric_omits_hash_family_until_set() -> None:
    body = Exporter().render().decode()
    assert "forecast_config_hash" not in body


def test_metric_emits_hash_after_set() -> None:
    exporter = Exporter()
    exporter.set_config_hash("deadbeef")
    body = exporter.render().decode()
    assert "# TYPE forecast_config_hash gauge" in body
    assert 'forecast_config_hash{sha256="deadbeef"} 1.0' in body


@pytest.mark.asyncio
async def test_reload_refreshes_config_hash(tmp_path: Path) -> None:
    """A successful reload bumps the exposed hash to the new file's digest."""
    path = _write(tmp_path, _BASE_CONFIG)
    cfg = config_module.load(path)
    exporter = Exporter()
    exporter.set_config_hash(_compute_config_hash(path))
    runner = Runner(config=cfg, source=_NullSource(), exporter=exporter)
    reloader = ConfigReloader(runner=runner, exporter=exporter, config_path=path)

    # Confirm initial hash is present.
    initial_hash = _compute_config_hash(path)
    body = exporter.render().decode()
    assert f'forecast_config_hash{{sha256="{initial_hash}"}}' in body

    # Modify the file: add a new group. Reload and confirm the metric
    # updates to the new digest.
    new_body = (
        _BASE_CONFIG
        + """  - name: g2
    queries:
      - id: m2
        promql: rate(up[5m])
"""
    )
    path.write_text(new_body)
    new_hash = hashlib.sha256(new_body.encode()).hexdigest()
    await reloader.reload()
    body = exporter.render().decode()
    assert f'forecast_config_hash{{sha256="{new_hash}"}}' in body
    assert f'forecast_config_hash{{sha256="{initial_hash}"}}' not in body
    await runner.stop()


@pytest.mark.asyncio
async def test_failed_reload_keeps_previous_hash(tmp_path: Path) -> None:
    """Validation failure must not advance the hash; /metrics keeps reading
    the running config's identity."""
    from promforecast.reload import ReloadError  # noqa: PLC0415

    path = _write(tmp_path, _BASE_CONFIG)
    cfg = config_module.load(path)
    exporter = Exporter()
    exporter.set_config_hash(_compute_config_hash(path))
    runner = Runner(config=cfg, source=_NullSource(), exporter=exporter)
    reloader = ConfigReloader(runner=runner, exporter=exporter, config_path=path)

    initial_hash = _compute_config_hash(path)
    path.write_text("not yaml: ::::\nbroken: -")
    with pytest.raises(ReloadError):
        await reloader.reload()

    body = exporter.render().decode()
    assert f'forecast_config_hash{{sha256="{initial_hash}"}}' in body
    await runner.stop()


# -- /config endpoint -----------------------------------------------------


def test_config_endpoint_off_by_default(tmp_path: Path) -> None:
    """Endpoint disabled unless server.expose_config_endpoint is set true."""
    path = _write(tmp_path, _BASE_CONFIG)
    cfg = config_module.load(path)
    exporter = Exporter()
    runner = Runner(config=cfg, source=_NullSource(), exporter=exporter)
    reloader = ConfigReloader(runner=runner, exporter=exporter, config_path=path)
    app = _build_app(
        exporter,
        runner,
        reloader,
        reload_enabled=False,
        expose_config_endpoint=False,
    )
    client = TestClient(app)
    # FastAPI returns 404 for an unmounted route by default.
    assert client.get("/config").status_code == 404


def test_config_endpoint_returns_sanitised_payload(tmp_path: Path) -> None:
    body = """
apiVersion: promforecast.io/v1
datasource:
  url: http://vm:8428/
server:
  expose_config_endpoint: true
defaults:
  models: [SeasonalNaive]
groups:
  - name: g
    queries:
      - id: m
        promql: up
"""
    path = _write(tmp_path, body)
    cfg = config_module.load(path)
    exporter = Exporter()
    exporter.set_config_hash(_compute_config_hash(path))
    runner = Runner(config=cfg, source=_NullSource(), exporter=exporter)
    reloader = ConfigReloader(runner=runner, exporter=exporter, config_path=path)
    app = _build_app(
        exporter,
        runner,
        reloader,
        reload_enabled=False,
        expose_config_endpoint=True,
    )
    client = TestClient(app)
    response = client.get("/config")
    assert response.status_code == 200
    payload = response.json()
    assert payload["config_hash"] == _compute_config_hash(path)
    assert payload["config"]["datasource"]["url"] == "http://vm:8428/"
    assert payload["config"]["groups"][0]["queries"][0]["id"] == "m"
    # The dump must use the *alias* spelling (``apiVersion``) so the JSON
    # keys match the YAML the operator wrote — otherwise diffing /config
    # against the source-controlled config requires a key-rename pass.
    assert payload["config"]["apiVersion"] == "promforecast.io/v1"
    assert "api_version" not in payload["config"]


def test_config_endpoint_redacts_remote_write_headers(tmp_path: Path) -> None:
    """Auth-bearing headers must not appear verbatim in the /config payload."""
    body = """
apiVersion: promforecast.io/v1
datasource:
  url: http://vm:8428/
server:
  expose_config_endpoint: true
sink:
  remote_write:
    enabled: true
    url: http://vm:8428/api/v1/import/prometheus
    headers:
      Authorization: Bearer secret-token-12345
defaults:
  models: [SeasonalNaive]
groups:
  - name: g
    queries:
      - id: m
        promql: up
"""
    path = _write(tmp_path, body)
    cfg = config_module.load(path)
    exporter = Exporter()
    runner = Runner(config=cfg, source=_NullSource(), exporter=exporter)
    reloader = ConfigReloader(runner=runner, exporter=exporter, config_path=path)
    app = _build_app(
        exporter,
        runner,
        reloader,
        reload_enabled=False,
        expose_config_endpoint=True,
    )
    client = TestClient(app)
    response = client.get("/config")
    assert response.status_code == 200
    headers = response.json()["config"]["sink"]["remote_write"]["headers"]
    assert "Authorization" in headers
    assert headers["Authorization"] != "Bearer secret-token-12345"
    assert "secret-token-12345" not in response.text


def test_healthz_includes_config_hash(tmp_path: Path) -> None:
    """/healthz exposes the current hash so a quick curl check works without
    scraping /metrics first."""
    path = _write(tmp_path, _BASE_CONFIG)
    cfg = config_module.load(path)
    exporter = Exporter()
    exporter.set_config_hash("abc123")
    runner = Runner(config=cfg, source=_NullSource(), exporter=exporter)
    reloader = ConfigReloader(runner=runner, exporter=exporter, config_path=path)
    app = _build_app(
        exporter,
        runner,
        reloader,
        reload_enabled=False,
    )
    client = TestClient(app)
    payload = client.get("/healthz").json()
    assert payload["status"] == "ok"
    assert payload["config_hash"] == "abc123"
