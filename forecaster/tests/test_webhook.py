"""Tests for the webhook regressor client.

Covers JSON parsing, per-(url, range) caching, HMAC verification, and the
bounded failure reasons that drive ``forecast_regressor_failures_total``.
The end-to-end prefetch wiring (graceful degradation, required short-circuit)
is covered in ``test_regressors.py``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import httpx
import pytest

from promforecast import signing
from promforecast.config import RegressorConfig, SigningConfig
from promforecast.webhook import WebhookError, WebhookRegressorClient


def _client(handler: object, clock: object | None = None) -> WebhookRegressorClient:
    transport = httpx.MockTransport(handler)  # type: ignore[arg-type]
    http = httpx.AsyncClient(transport=transport)
    if clock is None:
        return WebhookRegressorClient(client=http)
    return WebhookRegressorClient(client=http, clock=clock)  # type: ignore[arg-type]


def _reg(**kwargs: object) -> RegressorConfig:
    base: dict[str, object] = {"id": "incidents", "type": "webhook", "url": "https://hook/x"}
    base.update(kwargs)
    return RegressorConfig(**base)  # type: ignore[arg-type]


_START = datetime(2026, 1, 1, tzinfo=UTC)
_END = datetime(2026, 1, 2, tzinfo=UTC)


@pytest.mark.asyncio
async def test_fetch_parses_epoch_timestamps() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"timestamps": [_START.timestamp(), _END.timestamp()], "values": [1.0, 3.0]},
        )

    client = _client(handler)
    try:
        frames = await client.fetch(reg=_reg(), query_id="cpu", start=_START, end=_END)
    finally:
        await client.aclose()
    assert len(frames) == 1
    assert frames[0].values == [1.0, 3.0]
    assert frames[0].timestamps[0] == _START


@pytest.mark.asyncio
async def test_fetch_expands_env_refs_in_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PF_BRIDGE_TOKEN", "secret-token")
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["auth"] = request.headers.get("Authorization", "")
        return httpx.Response(200, json={"timestamps": [_START.timestamp()], "values": [1.0]})

    client = _client(handler)
    reg = _reg(headers={"Authorization": "Bearer ${PF_BRIDGE_TOKEN}"})
    try:
        await client.fetch(reg=reg, query_id="cpu", start=_START, end=_END)
    finally:
        await client.aclose()
    assert captured["auth"] == "Bearer secret-token"


@pytest.mark.asyncio
async def test_fetch_parses_iso_timestamps_and_drops_nan() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        # Raw body so we can include a JSON ``NaN`` token (the standard JSON
        # encoder rejects it, but Python's decoder accepts it on parse).
        body = (
            f'{{"timestamps": ["{_START.isoformat()}", "{_END.isoformat()}"], '
            '"values": [2.0, NaN]}'
        )
        return httpx.Response(200, content=body, headers={"Content-Type": "application/json"})

    client = _client(handler)
    try:
        frames = await client.fetch(reg=_reg(), query_id="cpu", start=_START, end=_END)
    finally:
        await client.aclose()
    assert frames[0].values == [2.0]


@pytest.mark.asyncio
async def test_fetch_request_body_carries_query_id_and_range() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json  # noqa: PLC0415

        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"timestamps": [_START.timestamp()], "values": [1.0]})

    client = _client(handler)
    try:
        await client.fetch(reg=_reg(), query_id="cpu_busy", start=_START, end=_END)
    finally:
        await client.aclose()
    assert seen["query_id"] == "cpu_busy"
    assert seen["regressor_id"] == "incidents"
    assert seen["start"] == _START.isoformat()


@pytest.mark.asyncio
async def test_fetch_caches_within_ttl() -> None:
    calls = {"n": 0}

    def handler(_: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json={"timestamps": [_START.timestamp()], "values": [1.0]})

    now = {"t": 1000.0}
    client = _client(handler, clock=lambda: now["t"])
    reg = _reg(cache_ttl=timedelta(hours=1))
    try:
        await client.fetch(reg=reg, query_id="cpu", start=_START, end=_END)
        now["t"] += 60  # still inside the 1h TTL
        await client.fetch(reg=reg, query_id="cpu", start=_START, end=_END)
        assert calls["n"] == 1
        now["t"] += 3600  # past TTL
        await client.fetch(reg=reg, query_id="cpu", start=_START, end=_END)
        assert calls["n"] == 2
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_fetch_http_error_raises_webhook_failure() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="overloaded")

    client = _client(handler)
    try:
        with pytest.raises(WebhookError) as exc:
            await client.fetch(reg=_reg(), query_id="cpu", start=_START, end=_END)
    finally:
        await client.aclose()
    assert exc.value.reason == "webhook_failure"


@pytest.mark.asyncio
async def test_fetch_bad_shape_raises_webhook_failure() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"not": "a timeseries"})

    client = _client(handler)
    try:
        with pytest.raises(WebhookError) as exc:
            await client.fetch(reg=_reg(), query_id="cpu", start=_START, end=_END)
    finally:
        await client.aclose()
    assert exc.value.reason == "webhook_failure"


@pytest.mark.asyncio
async def test_fetch_valid_signature_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PF_HOOK_SECRET", "s3cret")
    body = b'{"timestamps": [1735689600.0], "values": [1.0]}'
    sig = signing.compute_signature("s3cret", body)

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body, headers={signing.SIGNATURE_HEADER: sig})

    reg = _reg(signing=SigningConfig(secret_ref="${PF_HOOK_SECRET}"))
    client = _client(handler)
    try:
        frames = await client.fetch(reg=reg, query_id="cpu", start=_START, end=_END)
    finally:
        await client.aclose()
    assert frames[0].values == [1.0]


@pytest.mark.asyncio
async def test_fetch_invalid_signature_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PF_HOOK_SECRET", "s3cret")

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"timestamps": [_START.timestamp()], "values": [1.0]},
            headers={signing.SIGNATURE_HEADER: "sha256=deadbeef"},
        )

    reg = _reg(signing=SigningConfig(secret_ref="${PF_HOOK_SECRET}"))
    client = _client(handler)
    try:
        with pytest.raises(WebhookError) as exc:
            await client.fetch(reg=reg, query_id="cpu", start=_START, end=_END)
    finally:
        await client.aclose()
    assert exc.value.reason == "signature_invalid"


@pytest.mark.asyncio
async def test_fetch_unresolvable_secret_degrades_gracefully(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A missing signing secret is a config error, but it must route through
    # the graceful-degradation path (WebhookError) rather than crashing the
    # whole query with a SecretResolutionError.
    monkeypatch.delenv("PF_MISSING_HOOK_SECRET", raising=False)

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"timestamps": [_START.timestamp()], "values": [1.0]})

    reg = _reg(signing=SigningConfig(secret_ref="${PF_MISSING_HOOK_SECRET}"))
    client = _client(handler)
    try:
        with pytest.raises(WebhookError) as exc:
            await client.fetch(reg=reg, query_id="cpu", start=_START, end=_END)
    finally:
        await client.aclose()
    assert exc.value.reason == "signature_invalid"
