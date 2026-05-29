"""Tests for the Grafana-annotations sink."""

from __future__ import annotations

import json

import httpx
import pytest

from promforecast.annotations import AnnotationEvent, GrafanaAnnotationSink
from promforecast.config import AnnotationsConfig
from promforecast.exporter import Exporter


def _sink(config: AnnotationsConfig, exporter: Exporter, handler: object) -> GrafanaAnnotationSink:
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))  # type: ignore[arg-type]
    return GrafanaAnnotationSink(config, exporter, client=http, now_ms=lambda: 1_700_000_000_000)


def _config(events: list[str], **kwargs: object) -> AnnotationsConfig:
    base: dict[str, object] = {
        "enabled": True,
        "grafana_url": "https://grafana.example.com",
        "events": events,
    }
    base.update(kwargs)
    return AnnotationsConfig(**base)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_posts_enabled_event() -> None:
    posted: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/annotations"
        posted.append(json.loads(request.content))
        return httpx.Response(200)

    exporter = Exporter()
    sink = _sink(_config(["drift"]), exporter, handler)
    event = AnnotationEvent(
        event_type="drift", dedup_key="g:m", tags=["promforecast", "drift"], text="drift!"
    )
    try:
        await sink.post_events([event])
    finally:
        await sink.aclose()
    assert len(posted) == 1
    assert posted[0]["text"] == "drift!"
    assert posted[0]["time"] == 1_700_000_000_000
    body = exporter.render().decode()
    assert 'forecast_annotation_posts_total{event_type="drift",outcome="success"} 1.0' in body


@pytest.mark.asyncio
async def test_disabled_event_not_posted() -> None:
    posts = {"n": 0}

    def handler(_: httpx.Request) -> httpx.Response:
        posts["n"] += 1
        return httpx.Response(200)

    exporter = Exporter()
    sink = _sink(_config(["drift"]), exporter, handler)
    event = AnnotationEvent(event_type="deploy_risk", dedup_key="svc:1", tags=[], text="x")
    try:
        await sink.post_events([event])
    finally:
        await sink.aclose()
    assert posts["n"] == 0


@pytest.mark.asyncio
async def test_rising_edge_dedup() -> None:
    posts = {"n": 0}

    def handler(_: httpx.Request) -> httpx.Response:
        posts["n"] += 1
        return httpx.Response(200)

    exporter = Exporter()
    sink = _sink(_config(["deviation_outside_band"]), exporter, handler)
    event = AnnotationEvent(
        event_type="deviation_outside_band", dedup_key="g:m:s", tags=[], text="outside"
    )
    try:
        await sink.post_events([event])  # rising edge
        await sink.post_events([event])  # sustained -> suppressed
        assert posts["n"] == 1
        await sink.post_events([])  # clears the active key
        await sink.post_events([event])  # re-fires
        assert posts["n"] == 2
    finally:
        await sink.aclose()


@pytest.mark.asyncio
async def test_rising_edge_dedup_is_scoped_per_group() -> None:
    # The runner calls post_events once per group, each with only that
    # group's events. A sustained event in group "a" must not be re-posted
    # just because group "b" posts its own (different) event in between.
    posts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        posts.append(json.loads(request.content)["text"])
        return httpx.Response(200)

    exporter = Exporter()
    sink = _sink(_config(["drift"]), exporter, handler)
    ev_a = AnnotationEvent(event_type="drift", dedup_key="a:m", tags=[], text="a-drift")
    ev_b = AnnotationEvent(event_type="drift", dedup_key="b:m", tags=[], text="b-drift")
    try:
        # Cycle 1: both groups fire their rising edge.
        await sink.post_events([ev_a], scope="a")
        await sink.post_events([ev_b], scope="b")
        # Cycle 2: both still firing — both must be suppressed, not re-posted.
        await sink.post_events([ev_a], scope="a")
        await sink.post_events([ev_b], scope="b")
    finally:
        await sink.aclose()
    assert posts == ["a-drift", "b-drift"]


@pytest.mark.asyncio
async def test_auth_token_header(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GRAFANA_TOKEN", "glsa_abc")
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["auth"] = request.headers.get("Authorization", "")
        return httpx.Response(200)

    exporter = Exporter()
    sink = _sink(_config(["drift"], api_token_ref="${GRAFANA_TOKEN}"), exporter, handler)
    event = AnnotationEvent(event_type="drift", dedup_key="g:m", tags=[], text="d")
    try:
        await sink.post_events([event])
    finally:
        await sink.aclose()
    assert captured["auth"] == "Bearer glsa_abc"


@pytest.mark.asyncio
async def test_http_error_counts_outcome() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    exporter = Exporter()
    sink = _sink(_config(["drift"]), exporter, handler)
    event = AnnotationEvent(event_type="drift", dedup_key="g:m", tags=[], text="d")
    try:
        await sink.post_events([event])
    finally:
        await sink.aclose()
    body = exporter.render().decode()
    assert 'forecast_annotation_posts_total{event_type="drift",outcome="http_error"} 1.0' in body
