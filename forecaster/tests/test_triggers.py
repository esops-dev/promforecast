"""Tests for the forecast-driven webhook trigger engine."""

from __future__ import annotations

import json
from datetime import timedelta

import httpx
import pytest

from promforecast import signing
from promforecast.config import SigningConfig, TriggerConfig
from promforecast.exporter import Exporter
from promforecast.source import InstantSample, PromSource
from promforecast.triggers import TriggerEngine, condition_is_true


class _FakeSource(PromSource):
    """Returns a scripted instant-query result per call."""

    def __init__(self, results: list[list[InstantSample]]) -> None:
        self._results = results
        self._i = 0

    async def query_instant(self, promql: str) -> list[InstantSample]:
        result = self._results[min(self._i, len(self._results) - 1)]
        self._i += 1
        return result

    async def aclose(self) -> None:
        return None


def _engine(
    source: PromSource,
    exporter: Exporter,
    handler: object,
    *,
    triggers: list[TriggerConfig],
    clock: object | None = None,
) -> TriggerEngine:
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))  # type: ignore[arg-type]
    kwargs: dict[str, object] = {"now_iso": lambda: "2026-01-01T00:00:00+00:00"}
    if clock is not None:
        kwargs["clock"] = clock
    return TriggerEngine(
        triggers=triggers,
        source=source,
        exporter=exporter,
        client=http,
        **kwargs,  # type: ignore[arg-type]
    )


def _trigger(**kwargs: object) -> TriggerConfig:
    base: dict[str, object] = {
        "name": "t1",
        "condition": "up == 1",
        "webhook_url": "https://hook/x",
    }
    base.update(kwargs)
    return TriggerConfig(**base)  # type: ignore[arg-type]


def test_condition_is_true() -> None:
    assert condition_is_true([InstantSample(labels={}, value=1.0)])
    assert not condition_is_true([InstantSample(labels={}, value=0.0)])
    assert not condition_is_true([])


@pytest.mark.asyncio
async def test_trigger_fires_on_rising_edge() -> None:
    posted: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        posted.append(json.loads(request.content))
        return httpx.Response(200)

    source = _FakeSource([[InstantSample(labels={"model": "AutoARIMA"}, value=1.0)]])
    exporter = Exporter()
    engine = _engine(source, exporter, handler, triggers=[_trigger()])
    try:
        await engine.evaluate()
    finally:
        await engine.aclose()
    assert len(posted) == 1
    assert posted[0]["name"] == "t1"
    assert posted[0]["value"] == 1.0
    body = exporter.render().decode()
    assert 'forecast_trigger_fires_total{name="t1",outcome="success"} 1.0' in body


@pytest.mark.asyncio
async def test_trigger_does_not_refire_while_sustained() -> None:
    posts = {"n": 0}

    def handler(_: httpx.Request) -> httpx.Response:
        posts["n"] += 1
        return httpx.Response(200)

    # Condition stays true on every evaluation.
    source = _FakeSource([[InstantSample(labels={}, value=1.0)]])
    exporter = Exporter()
    now = {"t": 0.0}
    engine = _engine(
        source,
        exporter,
        handler,
        triggers=[_trigger(repeat_interval=timedelta(hours=1))],
        clock=lambda: now["t"],
    )
    try:
        await engine.evaluate()  # rising edge -> fire
        now["t"] += 60  # still within repeat_interval
        await engine.evaluate()  # suppressed
        assert posts["n"] == 1
        now["t"] += 3600  # past repeat_interval
        await engine.evaluate()  # re-fires
        assert posts["n"] == 2
    finally:
        await engine.aclose()


@pytest.mark.asyncio
async def test_trigger_coalesces_per_group_burst() -> None:
    # The runner evaluates triggers once per group run, so a refresh cycle
    # with several groups calls evaluate() back-to-back. Those calls land
    # within the coalescing window and must collapse to a single instant
    # query + delivery rather than one per group.
    queries = {"n": 0}

    class _CountingSource(PromSource):
        def __init__(self) -> None:
            pass

        async def query_instant(self, promql: str) -> list[InstantSample]:
            queries["n"] += 1
            return [InstantSample(labels={}, value=1.0)]

        async def aclose(self) -> None:
            return None

    posts = {"n": 0}

    def handler(_: httpx.Request) -> httpx.Response:
        posts["n"] += 1
        return httpx.Response(200)

    exporter = Exporter()
    now = {"t": 100.0}
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    engine = TriggerEngine(
        triggers=[_trigger(repeat_interval=timedelta(hours=1))],
        source=_CountingSource(),
        exporter=exporter,
        client=http,
        clock=lambda: now["t"],
        now_iso=lambda: "2026-01-01T00:00:00+00:00",
        coalesce_seconds=5.0,
    )
    try:
        # Three group runs in the same instant -> one query, one fire.
        await engine.evaluate()
        await engine.evaluate()
        await engine.evaluate()
        assert queries["n"] == 1
        assert posts["n"] == 1
        # Past the window the trigger is re-evaluated (query issued again),
        # but the sustained condition is still inside repeat_interval -> no
        # second delivery.
        now["t"] += 10.0
        await engine.evaluate()
        assert queries["n"] == 2
        assert posts["n"] == 1
    finally:
        await engine.aclose()


@pytest.mark.asyncio
async def test_trigger_signs_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRIGGER_SECRET", "shh")
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["sig"] = request.headers.get(signing.SIGNATURE_HEADER, "")
        captured["body"] = request.content.decode()
        return httpx.Response(200)

    source = _FakeSource([[InstantSample(labels={}, value=1.0)]])
    exporter = Exporter()
    trigger = _trigger(signing=SigningConfig(secret_ref="${TRIGGER_SECRET}"))
    engine = _engine(source, exporter, handler, triggers=[trigger])
    try:
        await engine.evaluate()
    finally:
        await engine.aclose()
    expected = signing.compute_signature("shh", captured["body"].encode())
    assert captured["sig"] == expected


@pytest.mark.asyncio
async def test_trigger_payload_template_renders() -> None:
    posted: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        posted.append(request.content.decode())
        return httpx.Response(200)

    source = _FakeSource([[InstantSample(labels={}, value=42.0)]])
    exporter = Exporter()
    trigger = _trigger(payload_template='{"text": "value is $value for $name"}')
    engine = _engine(source, exporter, handler, triggers=[trigger])
    try:
        await engine.evaluate()
    finally:
        await engine.aclose()
    assert posted[0] == '{"text": "value is 42.0 for t1"}'


@pytest.mark.asyncio
async def test_trigger_http_error_counts_outcome() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    source = _FakeSource([[InstantSample(labels={}, value=1.0)]])
    exporter = Exporter()
    engine = _engine(source, exporter, handler, triggers=[_trigger()])
    try:
        await engine.evaluate()
    finally:
        await engine.aclose()
    body = exporter.render().decode()
    assert 'forecast_trigger_fires_total{name="t1",outcome="http_error"} 1.0' in body


@pytest.mark.asyncio
async def test_trigger_condition_failure_does_not_fire() -> None:
    class _BadSource(PromSource):
        def __init__(self) -> None:
            pass

        async def query_instant(self, promql: str) -> list[InstantSample]:
            raise RuntimeError("tsdb down")

        async def aclose(self) -> None:
            return None

    posts = {"n": 0}

    def handler(_: httpx.Request) -> httpx.Response:
        posts["n"] += 1
        return httpx.Response(200)

    exporter = Exporter()
    engine = _engine(_BadSource(), exporter, handler, triggers=[_trigger()])
    try:
        await engine.evaluate()
    finally:
        await engine.aclose()
    assert posts["n"] == 0


@pytest.mark.asyncio
async def test_one_bad_trigger_does_not_block_others(monkeypatch: pytest.MonkeyPatch) -> None:
    # First trigger has an unresolvable signing secret (raises mid-fire);
    # the second must still be evaluated and delivered.
    monkeypatch.delenv("PF_MISSING_TRIGGER_SECRET", raising=False)
    posted: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        posted.append(str(request.url))
        return httpx.Response(200)

    source = _FakeSource([[InstantSample(labels={}, value=1.0)]])
    exporter = Exporter()
    bad = _trigger(
        name="bad",
        webhook_url="https://hook/bad",
        signing=SigningConfig(secret_ref="${PF_MISSING_TRIGGER_SECRET}"),
    )
    good = _trigger(name="good", webhook_url="https://hook/good")
    engine = _engine(source, exporter, handler, triggers=[bad, good])
    try:
        await engine.evaluate()
    finally:
        await engine.aclose()
    assert "https://hook/good" in posted
