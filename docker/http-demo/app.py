# Tiny FastAPI app for the dev stack: a handful of HTTP endpoints with varied
# latency / error profiles and a /metrics endpoint that exposes prometheus_client
# counters/histograms/gauges. Driven by docker/http-demo/loadgen.py so the
# forecaster has http_* time series to fit on.
from __future__ import annotations

import asyncio
import os
import random
import time

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

registry = CollectorRegistry()

http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests handled.",
    ["method", "endpoint", "status"],
    registry=registry,
)
http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds.",
    ["method", "endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
    registry=registry,
)
http_requests_in_flight = Gauge(
    "http_requests_in_flight",
    "In-flight HTTP requests.",
    registry=registry,
)
http_payload_bytes = Histogram(
    "http_payload_bytes",
    "Response payload size in bytes.",
    ["endpoint"],
    buckets=(64, 256, 1024, 4096, 16384, 65536),
    registry=registry,
)

APP_NAME = os.getenv("APP_NAME", "http-demo")
app = FastAPI(title=APP_NAME)


@app.middleware("http")
async def observe(request: Request, call_next):
    endpoint = request.scope.get("route").path if request.scope.get("route") else request.url.path
    method = request.method
    http_requests_in_flight.inc()
    start = time.perf_counter()
    try:
        response: Response = await call_next(request)
        status = response.status_code
    except Exception:
        status = 500
        raise
    finally:
        elapsed = time.perf_counter() - start
        http_requests_in_flight.dec()
        http_request_duration_seconds.labels(method=method, endpoint=endpoint).observe(elapsed)
        http_requests_total.labels(method=method, endpoint=endpoint, status=str(status)).inc()
    return response


def _sleep_lognormal(mu: float, sigma: float, cap: float) -> float:
    delay = min(random.lognormvariate(mu, sigma), cap)
    return delay


@app.get("/")
async def root() -> dict:
    body = {"app": APP_NAME, "endpoints": ["/", "/api/users", "/api/products", "/api/slow", "/api/flaky", "/healthz"]}
    http_payload_bytes.labels(endpoint="/").observe(len(str(body)))
    return body


@app.get("/api/users")
async def users() -> dict:
    # Fast endpoint, ~5-20ms.
    await asyncio.sleep(_sleep_lognormal(mu=-4.5, sigma=0.4, cap=0.2))
    payload = {"users": [{"id": i, "name": f"user-{i}"} for i in range(random.randint(5, 25))]}
    http_payload_bytes.labels(endpoint="/api/users").observe(len(str(payload)))
    return payload


@app.get("/api/products")
async def products() -> dict:
    # Medium endpoint, ~30-150ms with occasional spikes.
    await asyncio.sleep(_sleep_lognormal(mu=-3.0, sigma=0.6, cap=1.5))
    payload = {"products": [{"sku": f"sku-{i}", "price": round(random.uniform(1, 99), 2)} for i in range(20)]}
    http_payload_bytes.labels(endpoint="/api/products").observe(len(str(payload)))
    return payload


@app.get("/api/slow")
async def slow() -> dict:
    # Slow endpoint, ~0.3-2s. Useful for histogram_quantile to have something
    # to show.
    await asyncio.sleep(_sleep_lognormal(mu=-1.0, sigma=0.5, cap=3.0))
    return {"status": "ok"}


@app.get("/api/flaky")
async def flaky() -> Response:
    # ~7% of calls return 500. Gives http_requests_total{status="500"} a non-zero rate
    # so the forecaster has an error series to fit.
    await asyncio.sleep(_sleep_lognormal(mu=-3.5, sigma=0.5, cap=1.0))
    if random.random() < 0.07:
        return JSONResponse({"error": "transient"}, status_code=500)
    return JSONResponse({"status": "ok"})


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


@app.get("/metrics")
async def metrics() -> PlainTextResponse:
    return PlainTextResponse(generate_latest(registry).decode("utf-8"), media_type=CONTENT_TYPE_LATEST)
