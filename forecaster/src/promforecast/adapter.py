"""Kubernetes custom-metrics adapter for predictive autoscaling.

A separate sub-component that implements the Kubernetes custom-metrics API
contracts:

* ``external.metrics.k8s.io/v1beta1`` — flat (cluster-scoped or
  namespace-scoped) metrics referenced by HPA's ``external`` metric
  type. The HPA's ``targetValue`` is compared against the value
  returned here.
* ``custom.metrics.k8s.io/v1beta1`` — pod / object-scoped metrics
  with selectors. Less common in practice; the predictive
  autoscaling use-case usually fits the external API better
  because the forecast is per-service rather than per-pod. The
  adapter implements a minimal stub for the custom-metrics surface
  so HPAs that prefer that flavour still get a sensible response.

The adapter reads from VM via PromQL — it does **not** run the
forecast pipeline. Forecasts are produced by the main forecaster
and stored in VM via the sink; the adapter just exposes them under
HPA-compatible names. Stateless, side-effect-free.

Mounted behind the Kubernetes API aggregator via an
``APIService`` resource. The aggregator requires TLS; the adapter's
Helm chart generates a self-signed cert at install time so the
operator doesn't need cert-manager just to stand this up.

**Why a separate process?** The HPA control loop polls the adapter
every 15 seconds. Embedding the adapter into the forecaster would
mean the forecaster's HTTP surface had to be cluster-API-shaped
(TLS, APIService-fronted), which complicates the simple
``/metrics`` scrape model. A small separate adapter, talking to the
same VM, keeps each component's contract tight.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import structlog
import yaml
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

logger = structlog.get_logger(__name__)


class _MetricMapping(BaseModel):
    """One promforecast metric exposed to HPA.

    ``external_metric_name`` is the name HPA references in its
    ``external.metric.name`` field. Must conform to the Kubernetes
    metric-name regex ([a-z0-9-]+) so the API server accepts it.

    ``promql`` is the PromQL query the adapter runs against VM to
    resolve the metric. The query should return a scalar / single
    value — multi-series responses are summed across labels (the
    HPA only consumes a scalar). For per-pod or per-namespace
    autoscaling, parametrise the PromQL with the conventional
    ``namespace`` / ``pod`` labels and let the adapter substitute
    them at query time.

    ``namespaced: true`` (default) routes the metric under the
    namespaced HPA path. Set false for cluster-scoped metrics that
    HPA can reference regardless of namespace.
    """

    model_config = ConfigDict(extra="forbid")

    external_metric_name: str = Field(pattern=r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?$")
    promql: str
    namespaced: bool = True


class _AdapterConfig(BaseModel):
    """Top-level adapter config.

    Mounted via a ConfigMap in the standard install. The schema is
    deliberately minimal — the adapter's job is to translate
    HPA-shaped queries to PromQL; richer behaviour belongs on the
    forecaster.
    """

    model_config = ConfigDict(extra="forbid")

    datasource_url: str
    metrics: list[_MetricMapping] = Field(default_factory=list)
    # HTTP listen address. The chart's APIService points to the
    # service's port; the deployment's pod listens on this port.
    listen: str = ":6443"
    # Per-PromQL request timeout. The HPA's poll cadence is 15s;
    # tighter than that and we'd risk consecutive misses cascading
    # into HPA marking the metric "unavailable".
    query_timeout_seconds: float = 5.0


# ---------------------------------------------------------------------------
# K8s API JSON shapes. These mirror the upstream Go types; the adapter
# emits the exact JSON shape the API aggregator expects.
# ---------------------------------------------------------------------------


def _api_group_list() -> dict[str, Any]:
    """Body for ``GET /apis/`` (group discovery)."""
    return {
        "kind": "APIGroupList",
        "apiVersion": "v1",
        "groups": [
            {
                "name": "external.metrics.k8s.io",
                "versions": [
                    {
                        "groupVersion": "external.metrics.k8s.io/v1beta1",
                        "version": "v1beta1",
                    }
                ],
                "preferredVersion": {
                    "groupVersion": "external.metrics.k8s.io/v1beta1",
                    "version": "v1beta1",
                },
            },
            {
                "name": "custom.metrics.k8s.io",
                "versions": [
                    {
                        "groupVersion": "custom.metrics.k8s.io/v1beta1",
                        "version": "v1beta1",
                    }
                ],
                "preferredVersion": {
                    "groupVersion": "custom.metrics.k8s.io/v1beta1",
                    "version": "v1beta1",
                },
            },
        ],
    }


def _external_resource_list(metrics: list[_MetricMapping]) -> dict[str, Any]:
    """Body for ``GET /apis/external.metrics.k8s.io/v1beta1`` (resource listing)."""
    return {
        "kind": "APIResourceList",
        "apiVersion": "v1",
        "groupVersion": "external.metrics.k8s.io/v1beta1",
        "resources": [
            {
                "name": m.external_metric_name,
                "singularName": "",
                "namespaced": m.namespaced,
                "kind": "ExternalMetricValueList",
                "verbs": ["get"],
            }
            for m in metrics
        ],
    }


def _custom_resource_list() -> dict[str, Any]:
    """Body for ``GET /apis/custom.metrics.k8s.io/v1beta1`` (resource listing).

    Minimal placeholder — the adapter doesn't ship per-object
    custom metrics today. The empty list is a valid response
    that tells the API aggregator "this group exists, no
    metrics under it yet" rather than 404, which would make HPA
    list-time errors noisier than needed.
    """
    return {
        "kind": "APIResourceList",
        "apiVersion": "v1",
        "groupVersion": "custom.metrics.k8s.io/v1beta1",
        "resources": [],
    }


def _external_metric_value_list(
    *,
    metric_name: str,
    value: float,
    metric_labels: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Body for ``GET /apis/external.metrics.k8s.io/v1beta1/.../<metric_name>``.

    The Kubernetes API expects ``value`` as a Quantity string —
    integers without a unit when the value is whole, decimals when
    fractional. Render as a plain string and let the API parse it.
    """
    return {
        "kind": "ExternalMetricValueList",
        "apiVersion": "external.metrics.k8s.io/v1beta1",
        "metadata": {},
        "items": [
            {
                "metricName": metric_name,
                "metricLabels": metric_labels or {},
                "timestamp": datetime.now(tz=UTC)
                .isoformat(timespec="seconds")
                .replace("+00:00", "Z"),
                "value": _format_quantity(value),
            }
        ],
    }


def _format_quantity(value: float) -> str:
    """Render a float as a K8s Quantity string.

    Integers come out without a decimal so HPA's parser accepts them
    cleanly; fractional values get six digits of precision (matches
    Prometheus's ``%g``-flavoured exposition).

    **NaN / Inf -> "0" trade-off.** HPA's ``failedGetMetric`` event
    (the alternative shape: return a non-200) marks the metric
    unavailable, which keeps the autoscaler on its last good
    recommendation indefinitely — *fail open* in HPA's terms. For a
    forecast that hasn't produced a sample yet (cold-start, ramp-up,
    transient query failure) the operator usually wants the HPA to
    *fail closed* — i.e. fall back to whatever ``minReplicas`` says
    rather than freeze in place. Returning ``"0"`` achieves that:
    a target like ``averageValue: 10`` against a current value of
    ``0`` triggers a scale-down toward minReplicas. Operators who
    want the opposite shape (freeze on NaN) should wrap their
    PromQL with ``clamp_min(..., 0.001)`` and a sentinel value
    upstream — that's a downstream config decision the adapter
    can't make safely on its own.
    """
    import math  # noqa: PLC0415

    if not math.isfinite(value):
        return "0"
    if value.is_integer():
        return str(int(value))
    return f"{value:.6f}".rstrip("0").rstrip(".")


# ---------------------------------------------------------------------------
# PromQL fetch shim. The adapter uses httpx directly rather than the
# forecaster's ``PromSource`` because the adapter's contract is
# substantively narrower — single-value queries with a short timeout
# — and pulling in the full source would drag scheduling / retry
# infrastructure the adapter doesn't need.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _PromValue:
    """One scalar result from PromQL."""

    value: float
    timestamp: datetime
    labels: dict[str, str]


async def _query_promql(
    *,
    client: httpx.AsyncClient,
    base_url: str,
    promql: str,
    timeout_seconds: float,
) -> list[_PromValue]:
    """Run a PromQL instant query and return the matching values.

    The adapter uses ``/api/v1/query`` (instant) rather than
    ``/api/v1/query_range`` (range) — HPA wants the most recent
    value, not a time series. VM accepts the same endpoint shape
    as Prometheus.

    Returns an empty list when the query has no matching series.
    Raises ``httpx.HTTPError`` on transport failure; the caller
    translates that into a 500 / 503 response.
    """
    url = base_url.rstrip("/") + "/api/v1/query"
    response = await client.get(url, params={"query": promql}, timeout=timeout_seconds)
    response.raise_for_status()
    body = response.json()
    if body.get("status") != "success":
        return []
    result = body.get("data", {}).get("result", [])
    values: list[_PromValue] = []
    for entry in result:
        metric_labels = {str(k): str(v) for k, v in entry.get("metric", {}).items()}
        value_pair = entry.get("value")
        if not value_pair or len(value_pair) != 2:  # noqa: PLR2004
            continue
        try:
            ts = datetime.fromtimestamp(float(value_pair[0]), tz=UTC)
            val = float(value_pair[1])
        except (TypeError, ValueError):
            continue
        values.append(_PromValue(value=val, timestamp=ts, labels=metric_labels))
    return values


# ---------------------------------------------------------------------------
# FastAPI app construction.
# ---------------------------------------------------------------------------


def _build_app(config: _AdapterConfig) -> FastAPI:
    """Wire up the adapter HTTP surface against ``config``.

    The app is built once per process — the metric list is static
    over the process lifetime. Reloading happens via process
    restart (Helm ``kubectl rollout restart``) so the contract
    stays as simple as a stateless adapter can be.
    """
    metrics_by_name = {m.external_metric_name: m for m in config.metrics}

    @asynccontextmanager
    async def lifespan(fastapi_app: FastAPI) -> AsyncIterator[None]:
        # If a test pre-seeded a client (or a previous lifespan left
        # one behind), close it before installing the one we own —
        # otherwise we leak a connection pool on every restart.
        prior = getattr(fastapi_app.state, "http", None)
        if prior is not None:
            with contextlib.suppress(Exception):
                await prior.aclose()
        async with httpx.AsyncClient() as client:
            fastapi_app.state.http = client
            yield

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    # Lazily build the http client on first request so tests that
    # bypass ``lifespan`` (bare ``TestClient(app)`` without a ``with``
    # block) get a usable client without us leaking a pool that the
    # lifespan would otherwise replace and never close. Production
    # boots through ``lifespan`` and never hits the lazy path.
    app.state.http = None

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/apis")
    async def apis() -> Response:
        return JSONResponse(_api_group_list())

    @app.get("/apis/")
    async def apis_slash() -> Response:
        # The API aggregator may probe either form; both serve the
        # same body so a trailing-slash mismatch doesn't break
        # discovery.
        return JSONResponse(_api_group_list())

    @app.get("/apis/external.metrics.k8s.io/v1beta1")
    async def external_metrics_list() -> Response:
        return JSONResponse(_external_resource_list(config.metrics))

    @app.get("/apis/external.metrics.k8s.io/v1beta1/")
    async def external_metrics_list_slash() -> Response:
        return JSONResponse(_external_resource_list(config.metrics))

    @app.get("/apis/custom.metrics.k8s.io/v1beta1")
    async def custom_metrics_list() -> Response:
        return JSONResponse(_custom_resource_list())

    async def _fetch_external_value(metric_name: str, label_selector: str | None) -> Response:
        mapping = metrics_by_name.get(metric_name)
        if mapping is None:
            return JSONResponse(
                {
                    "kind": "Status",
                    "apiVersion": "v1",
                    "status": "Failure",
                    "message": f"metric {metric_name!r} is not configured",
                    "reason": "NotFound",
                    "code": 404,
                },
                status_code=404,
            )
        promql = _apply_label_selector(mapping.promql, label_selector)
        # ``lifespan`` installs the client on app startup; the lazy
        # path here only fires for tests that skip the lifespan
        # (bare ``TestClient(app)`` without a ``with`` block). The
        # lazy client lives for the rest of the app's lifetime; tests
        # that care about cleanup should still use the context-manager
        # form of TestClient.
        if app.state.http is None:
            app.state.http = httpx.AsyncClient()
        client = app.state.http
        try:
            values = await _query_promql(
                client=client,
                base_url=config.datasource_url,
                promql=promql,
                timeout_seconds=config.query_timeout_seconds,
            )
        except httpx.HTTPError as exc:
            logger.warning(
                "adapter_query_failed",
                metric=metric_name,
                error=str(exc),
            )
            return JSONResponse(
                {
                    "kind": "Status",
                    "apiVersion": "v1",
                    "status": "Failure",
                    "message": f"upstream query failed: {exc}",
                    "reason": "ServiceUnavailable",
                    "code": 503,
                },
                status_code=503,
            )
        # Sum across returned series. HPA wants a scalar; multi-
        # series responses are nearly always per-pod or per-replica
        # fan-outs that the operator wants aggregated for
        # autoscaling purposes. Operators who want a different
        # reduction wrap their PromQL with ``avg()`` / ``max()`` /
        # ``topk()`` at config time.
        aggregate = sum(v.value for v in values) if values else 0.0
        return JSONResponse(
            _external_metric_value_list(
                metric_name=metric_name,
                value=aggregate,
            )
        )

    @app.get("/apis/external.metrics.k8s.io/v1beta1/{metric_name}")
    async def external_metric_cluster(metric_name: str, request: Request) -> Response:
        return await _fetch_external_value(metric_name, request.query_params.get("labelSelector"))

    @app.get("/apis/external.metrics.k8s.io/v1beta1/namespaces/{namespace}/{metric_name}")
    async def external_metric_namespaced(
        namespace: str, metric_name: str, request: Request
    ) -> Response:
        selector = request.query_params.get("labelSelector")
        # The URL path's namespace is authoritative — the K8s API
        # server has already enforced RBAC against it. Strip any
        # caller-supplied ``namespace=`` pair from the labelSelector
        # before appending our own so a caller can't bypass scoping by
        # passing ``labelSelector=namespace=other``. Other pairs pass
        # through.
        cleaned = _strip_namespace_pair(selector)
        ns_selector = f"namespace={namespace}"
        merged = ns_selector if not cleaned else f"{cleaned},{ns_selector}"
        return await _fetch_external_value(metric_name, merged)

    return app


def _strip_namespace_pair(selector: str | None) -> str | None:
    """Drop any ``namespace=...`` pair from a labelSelector.

    The namespaced endpoint reads the namespace from the URL path
    (already authenticated and authorised by the K8s API server) and
    refuses to let a caller-supplied ``namespace=`` pair override it
    — otherwise a workload in namespace A could read metrics scoped
    to namespace B. Other pairs are preserved verbatim.
    """
    if not selector:
        return selector
    kept = [p for p in selector.split(",") if p.partition("=")[0].strip() != "namespace"]
    return ",".join(kept) if kept else None


def _apply_label_selector(promql: str, selector: str | None) -> str:
    """Substitute ``$<label>`` placeholders in ``promql`` with selector values.

    Trivial label-selector parser: comma-separated ``k=v`` pairs.
    Each pair is substituted into the PromQL via ``$k`` token
    replacement. A config that has ``http_requests_rate_forecast{namespace="$namespace"}``
    works against ``labelSelector=namespace=foo`` without the
    adapter needing to understand the Kubernetes label-selector
    grammar in full.

    Unknown placeholders are left in the PromQL — the upstream
    query will return an empty result, which the adapter renders
    as a zero value (the right "metric not yet available" signal).
    """
    if not selector:
        return promql
    out = promql
    for pair in selector.split(","):
        if "=" not in pair:
            continue
        key, _, value = pair.partition("=")
        out = re.sub(rf"\${re.escape(key.strip())}\b", value.strip(), out)
    return out


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------


def load_config(path: Path) -> _AdapterConfig:
    """Load + validate the adapter config YAML."""
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"adapter config must be a YAML mapping: {path}")
    return _AdapterConfig.model_validate(raw)


def _parse_listen(listen: str) -> tuple[str, int]:
    """Accept ``:6443`` or ``host:6443``."""
    if listen.startswith(":"):
        return "0.0.0.0", int(listen[1:])
    host, _, port = listen.rpartition(":")
    return host or "0.0.0.0", int(port)


def run(
    config_path: Path,
    *,
    tls_cert: Path | None = None,
    tls_key: Path | None = None,
) -> None:
    """Boot the adapter HTTP server.

    The Kubernetes API aggregator requires TLS; the chart
    generates a self-signed cert at install time and mounts it
    into the pod. Plain HTTP is permitted (no TLS args) for
    local development and tests but rejected by the aggregator in
    a real cluster.
    """
    import uvicorn  # noqa: PLC0415

    config = load_config(config_path)
    app = _build_app(config)
    host, port = _parse_listen(config.listen)
    ssl_certfile = str(tls_cert) if tls_cert is not None else None
    ssl_keyfile = str(tls_key) if tls_key is not None else None
    if ssl_certfile is not None and ssl_keyfile is not None:
        logger.info("adapter_starting_tls", host=host, port=port, cert=ssl_certfile)
    else:
        logger.warning(
            "adapter_starting_plain",
            host=host,
            port=port,
            note="HTTP only; K8s API aggregator requires TLS in production",
        )
    uvicorn_config = uvicorn.Config(
        app=app,
        host=host,
        port=port,
        log_level="info",
        lifespan="on",
        ssl_certfile=ssl_certfile,
        ssl_keyfile=ssl_keyfile,
    )
    asyncio.run(_serve(uvicorn.Server(uvicorn_config)))


async def _serve(server: Any) -> None:
    """Run uvicorn until SIGTERM / SIGINT."""
    import signal  # noqa: PLC0415

    loop = asyncio.get_running_loop()

    def _shutdown() -> None:
        server.should_exit = True
        # Stamp the time so a deployment can correlate the
        # rolling-restart latency.
        logger.info("adapter_shutdown_requested", at=time.time())

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _shutdown)
    await server.serve()


__all__ = [
    "load_config",
    "run",
]


# Re-exported for tests so they can wire stub clients without
# touching module privates. The leading underscore stays — the
# public surface above is the documented one.
build_app = _build_app
