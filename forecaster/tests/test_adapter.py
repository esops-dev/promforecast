"""Tests for the Kubernetes custom-metrics adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from promforecast import adapter

# ---- Pure helpers ----------------------------------------------------------


def test_format_quantity_renders_integers_without_decimal() -> None:
    assert adapter._format_quantity(42.0) == "42"
    assert adapter._format_quantity(0.0) == "0"


def test_format_quantity_renders_floats_with_precision() -> None:
    assert adapter._format_quantity(1.5) == "1.5"
    assert adapter._format_quantity(0.123456789) == "0.123457"


def test_format_quantity_handles_non_finite() -> None:
    """NaN / Inf become ``"0"`` — HPA treats unparseable values as 'unavailable'."""
    assert adapter._format_quantity(float("nan")) == "0"
    assert adapter._format_quantity(float("inf")) == "0"


def test_apply_label_selector_substitutes_token() -> None:
    out = adapter._apply_label_selector(
        'http_requests_rate_forecast{namespace="$namespace"}',
        "namespace=payments",
    )
    assert out == 'http_requests_rate_forecast{namespace="payments"}'


def test_apply_label_selector_handles_multiple_pairs() -> None:
    out = adapter._apply_label_selector(
        'metric{namespace="$namespace",pod="$pod"}',
        "namespace=payments,pod=p-1",
    )
    assert 'namespace="payments"' in out
    assert 'pod="p-1"' in out


def test_apply_label_selector_passthrough_when_none() -> None:
    assert adapter._apply_label_selector("foo", None) == "foo"


def test_apply_label_selector_leaves_unknown_placeholders() -> None:
    """An unmatched placeholder stays in the query (renders as empty result)."""
    out = adapter._apply_label_selector(
        'foo{ns="$namespace"}',
        "other=bar",
    )
    assert out == 'foo{ns="$namespace"}'


def test_strip_namespace_pair_removes_caller_supplied_namespace() -> None:
    """The namespaced endpoint must not let a caller-supplied ``namespace=``
    override the URL-path namespace — otherwise a workload in namespace A
    could read metrics scoped to namespace B by passing
    ``labelSelector=namespace=B``.
    """
    assert adapter._strip_namespace_pair("namespace=other,service=foo") == "service=foo"
    assert adapter._strip_namespace_pair("service=foo,namespace=other") == "service=foo"
    # Only the namespace pair is dropped; other pairs survive.
    assert adapter._strip_namespace_pair("namespace=admin") is None
    assert adapter._strip_namespace_pair(None) is None
    assert adapter._strip_namespace_pair("") == ""
    assert adapter._strip_namespace_pair("service=foo") == "service=foo"


def test_namespaced_endpoint_rejects_caller_supplied_namespace_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression guard for the namespace clobber: a caller passing
    ``labelSelector=namespace=other`` against the namespaced URL path
    for namespace ``payments`` must end up querying with
    ``namespace="payments"``, not ``namespace="other"``."""
    seen_promql: dict[str, str] = {}

    async def fake_query(
        *, client: object, base_url: str, promql: str, timeout_seconds: float
    ) -> list[adapter._PromValue]:
        from datetime import UTC, datetime  # noqa: PLC0415

        seen_promql["promql"] = promql
        return [adapter._PromValue(value=1.0, timestamp=datetime.now(tz=UTC), labels={})]

    monkeypatch.setattr(adapter, "_query_promql", fake_query)
    app = _build_test_app(
        [
            {
                "external_metric_name": "rps-forecast",
                "promql": 'http_requests_rate_forecast{namespace="$namespace"}',
            }
        ]
    )
    client = TestClient(app)
    response = client.get(
        "/apis/external.metrics.k8s.io/v1beta1/namespaces/payments/rps-forecast",
        params={"labelSelector": "namespace=other,service=payments-api"},
    )
    assert response.status_code == 200
    # The URL path's namespace wins; the caller-supplied override is dropped.
    assert 'namespace="payments"' in seen_promql["promql"]
    assert 'namespace="other"' not in seen_promql["promql"]


def test_parse_listen_handles_colon_prefix() -> None:
    assert adapter._parse_listen(":6443") == ("0.0.0.0", 6443)


def test_parse_listen_handles_host_form() -> None:
    assert adapter._parse_listen("127.0.0.1:9000") == ("127.0.0.1", 9000)


# ---- Config loading --------------------------------------------------------


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "adapter.yaml"
    path.write_text(body)
    return path


def test_load_config_minimal(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
datasource_url: http://vm:8428
metrics:
  - external_metric_name: requests-per-second
    promql: 'avg(http_requests_rate_forecast)'
""",
    )
    cfg = adapter.load_config(path)
    assert cfg.datasource_url == "http://vm:8428"
    assert len(cfg.metrics) == 1
    assert cfg.metrics[0].external_metric_name == "requests-per-second"


def test_load_config_rejects_bad_metric_name(tmp_path: Path) -> None:
    """K8s metric names must match ``[a-z0-9-]+`` — uppercase rejected."""
    path = _write(
        tmp_path,
        """
datasource_url: http://vm:8428
metrics:
  - external_metric_name: BadName
    promql: 'foo'
""",
    )
    with pytest.raises(Exception):  # noqa: B017 - pydantic raises its own ValidationError subclass
        adapter.load_config(path)


def test_load_config_rejects_non_dict_top_level(tmp_path: Path) -> None:
    path = _write(tmp_path, "- a\n- b\n")
    with pytest.raises(ValueError, match="YAML mapping"):
        adapter.load_config(path)


# ---- App wiring ------------------------------------------------------------


def _build_test_app(metrics: list[dict[str, Any]]) -> FastAPI:
    config = adapter._AdapterConfig(
        datasource_url="http://vm:8428",
        metrics=[adapter._MetricMapping(**m) for m in metrics],
    )
    app = adapter.build_app(config)
    return app


def test_apis_endpoint_returns_two_groups() -> None:
    app = _build_test_app([])
    client = TestClient(app)
    response = client.get("/apis")
    assert response.status_code == 200
    body = response.json()
    assert body["kind"] == "APIGroupList"
    group_names = {g["name"] for g in body["groups"]}
    assert group_names == {"external.metrics.k8s.io", "custom.metrics.k8s.io"}


def test_external_metrics_list_returns_configured_metrics() -> None:
    app = _build_test_app(
        [
            {"external_metric_name": "rps-forecast", "promql": "foo"},
            {"external_metric_name": "cpu-forecast", "promql": "bar"},
        ]
    )
    client = TestClient(app)
    response = client.get("/apis/external.metrics.k8s.io/v1beta1")
    assert response.status_code == 200
    body = response.json()
    assert body["kind"] == "APIResourceList"
    names = {r["name"] for r in body["resources"]}
    assert names == {"rps-forecast", "cpu-forecast"}


def test_custom_metrics_list_returns_empty_placeholder() -> None:
    """Custom metrics surface is a stub today — empty resource list."""
    app = _build_test_app([])
    client = TestClient(app)
    response = client.get("/apis/custom.metrics.k8s.io/v1beta1")
    assert response.status_code == 200
    body = response.json()
    assert body["resources"] == []


def test_external_metric_unknown_returns_404() -> None:
    app = _build_test_app([{"external_metric_name": "known", "promql": "foo"}])
    client = TestClient(app)
    response = client.get("/apis/external.metrics.k8s.io/v1beta1/namespaces/default/unknown")
    assert response.status_code == 404
    body = response.json()
    assert body["reason"] == "NotFound"


def test_external_metric_returns_summed_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Multi-series PromQL results are summed into a scalar HPA can read."""

    async def fake_query(
        *, client: object, base_url: str, promql: str, timeout_seconds: float
    ) -> list[adapter._PromValue]:
        from datetime import UTC, datetime  # noqa: PLC0415

        now = datetime.now(tz=UTC)
        return [
            adapter._PromValue(value=2.0, timestamp=now, labels={}),
            adapter._PromValue(value=3.5, timestamp=now, labels={}),
        ]

    monkeypatch.setattr(adapter, "_query_promql", fake_query)
    app = _build_test_app([{"external_metric_name": "rps-forecast", "promql": "sum(rate(...))"}])
    with TestClient(app) as client:
        response = client.get(
            "/apis/external.metrics.k8s.io/v1beta1/namespaces/payments/rps-forecast"
        )
    assert response.status_code == 200
    body = response.json()
    assert body["kind"] == "ExternalMetricValueList"
    assert body["items"][0]["metricName"] == "rps-forecast"
    # 2.0 + 3.5 = 5.5
    assert body["items"][0]["value"] == "5.5"


def test_external_metric_returns_zero_on_empty_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_query(
        *, client: object, base_url: str, promql: str, timeout_seconds: float
    ) -> list[adapter._PromValue]:
        return []

    monkeypatch.setattr(adapter, "_query_promql", fake_query)
    app = _build_test_app([{"external_metric_name": "rps-forecast", "promql": "foo"}])
    client = TestClient(app)
    response = client.get("/apis/external.metrics.k8s.io/v1beta1/namespaces/payments/rps-forecast")
    assert response.status_code == 200
    body = response.json()
    assert body["items"][0]["value"] == "0"


def test_external_metric_returns_503_on_upstream_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_query(
        *, client: object, base_url: str, promql: str, timeout_seconds: float
    ) -> list[adapter._PromValue]:
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(adapter, "_query_promql", fake_query)
    app = _build_test_app([{"external_metric_name": "rps-forecast", "promql": "foo"}])
    client = TestClient(app)
    response = client.get("/apis/external.metrics.k8s.io/v1beta1/namespaces/payments/rps-forecast")
    assert response.status_code == 503
    body = response.json()
    assert body["reason"] == "ServiceUnavailable"


def test_external_metric_cluster_scoped_uses_no_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cluster-scoped (non-namespaced) external metrics use the no-ns path."""
    seen: dict[str, str] = {}

    async def fake_query(
        *, client: object, base_url: str, promql: str, timeout_seconds: float
    ) -> list[adapter._PromValue]:
        from datetime import UTC, datetime  # noqa: PLC0415

        seen["promql"] = promql
        return [adapter._PromValue(value=42.0, timestamp=datetime.now(tz=UTC), labels={})]

    monkeypatch.setattr(adapter, "_query_promql", fake_query)
    app = _build_test_app(
        [
            {
                "external_metric_name": "fleet-load",
                "promql": "sum(fleet_load_forecast)",
                "namespaced": False,
            }
        ]
    )
    client = TestClient(app)
    response = client.get("/apis/external.metrics.k8s.io/v1beta1/fleet-load")
    assert response.status_code == 200
    assert seen["promql"] == "sum(fleet_load_forecast)"


def test_namespaced_endpoint_injects_namespace_label_selector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Namespace from the URL path becomes a synthetic labelSelector."""
    seen_promql: dict[str, str] = {}

    async def fake_query(
        *, client: object, base_url: str, promql: str, timeout_seconds: float
    ) -> list[adapter._PromValue]:
        from datetime import UTC, datetime  # noqa: PLC0415

        seen_promql["promql"] = promql
        return [adapter._PromValue(value=1.0, timestamp=datetime.now(tz=UTC), labels={})]

    monkeypatch.setattr(adapter, "_query_promql", fake_query)
    app = _build_test_app(
        [
            {
                "external_metric_name": "rps-forecast",
                "promql": 'sum(rate(http_requests_total{namespace="$namespace"}[2m]))',
            }
        ]
    )
    client = TestClient(app)
    response = client.get("/apis/external.metrics.k8s.io/v1beta1/namespaces/payments/rps-forecast")
    assert response.status_code == 200
    # The synthetic ``namespace=payments`` selector was injected and
    # substituted into the PromQL template.
    assert 'namespace="payments"' in seen_promql["promql"]


def test_healthz_returns_ok() -> None:
    app = _build_test_app([])
    client = TestClient(app)
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


# ---- _query_promql lifecycle ---------------------------------------------


@pytest.mark.asyncio
async def test_query_promql_parses_vm_instant_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validate the parser against a realistic VM/Prometheus body."""

    class _StubResponse:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {
                "status": "success",
                "data": {
                    "resultType": "vector",
                    "result": [
                        {
                            "metric": {"__name__": "x", "namespace": "payments"},
                            "value": [1715000000.123, "12.5"],
                        }
                    ],
                },
            }

    class _StubClient:
        async def get(self, url: str, params: dict[str, str], timeout: float) -> _StubResponse:
            return _StubResponse()

    out = await adapter._query_promql(
        client=_StubClient(),  # type: ignore[arg-type]
        base_url="http://vm:8428",
        promql="x",
        timeout_seconds=5.0,
    )
    assert len(out) == 1
    assert out[0].value == 12.5
    assert out[0].labels["namespace"] == "payments"


@pytest.mark.asyncio
async def test_query_promql_returns_empty_on_non_success_body() -> None:
    class _StubResponse:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {"status": "error", "errorType": "bad_data"}

    class _StubClient:
        async def get(self, url: str, params: dict[str, str], timeout: float) -> _StubResponse:
            return _StubResponse()

    out = await adapter._query_promql(
        client=_StubClient(),  # type: ignore[arg-type]
        base_url="http://vm:8428",
        promql="x",
        timeout_seconds=5.0,
    )
    assert out == []


# ---- module surface --------------------------------------------------------


def test_module_exports_documented_names() -> None:
    assert adapter.load_config is not None
    assert adapter.run is not None
    assert adapter.build_app is not None
