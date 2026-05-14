"""Tests for the optional OpenTelemetry tracer facade.

We deliberately avoid importing ``opentelemetry`` here — the entire point
of the disabled path is that the SDK is never touched. The "enabled but
SDK missing" branch is exercised by stubbing ``build_tracer`` against a
config that asks for telemetry and asserting the graceful fallback.
"""

from __future__ import annotations

import sys
import types
from contextlib import nullcontext

import pytest
from pydantic import ValidationError

from promforecast import telemetry
from promforecast.config import OTLPTelemetryConfig, TelemetryConfig


def test_disabled_tracer_returns_nullcontext() -> None:
    """The zero-overhead path must return ``nullcontext`` (no SDK import)."""
    tracer = telemetry.disabled()
    assert tracer.enabled is False
    span_cm = tracer.span("anything", group="g", model="m")
    # ``isinstance(span_cm, nullcontext)`` is the contract: any future
    # refactor that swaps it for a custom no-op must still satisfy
    # "context manager that does nothing".
    assert isinstance(span_cm, nullcontext)
    with span_cm:
        pass


def test_disabled_tracer_singleton_across_calls() -> None:
    """``disabled()`` returns the same instance so tests can compare identity."""
    assert telemetry.disabled() is telemetry.disabled()


def test_build_tracer_with_disabled_config_returns_disabled_tracer() -> None:
    config = TelemetryConfig(enabled=False)
    tracer = telemetry.build_tracer(config)
    assert tracer.enabled is False
    assert tracer is telemetry.disabled()


def test_telemetry_enabled_requires_endpoint() -> None:
    with pytest.raises(ValidationError, match=r"otlp\.endpoint is empty"):
        TelemetryConfig(enabled=True, otlp=OTLPTelemetryConfig(endpoint=""))


def test_telemetry_sample_ratio_range() -> None:
    """Sample ratios outside [0, 1] are nonsense and must reject."""
    with pytest.raises(ValidationError):
        TelemetryConfig(sample_ratio=1.5)
    with pytest.raises(ValidationError):
        TelemetryConfig(sample_ratio=-0.1)


def test_build_tracer_handles_missing_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the SDK extra isn't installed we log + degrade to the no-op."""

    def _raise(_: TelemetryConfig) -> object:
        raise ImportError("simulated: opentelemetry not installed")

    monkeypatch.setattr(telemetry, "_initialise_otel", _raise)
    config = TelemetryConfig(
        enabled=True,
        otlp=OTLPTelemetryConfig(endpoint="otel-collector:4317"),
    )
    tracer = telemetry.build_tracer(config)
    assert tracer.enabled is False


def test_disabled_tracer_shutdown_is_safe() -> None:
    """``shutdown()`` on the disabled tracer must be a no-op, not crash."""
    tracer = telemetry.disabled()
    tracer.shutdown()
    # Idempotent: calling twice is fine.
    tracer.shutdown()


def test_real_tracer_shutdown_flushes_provider() -> None:
    """An enabled tracer's shutdown calls the provider's flush hook.

    Without this the BatchSpanProcessor's queue is dropped at SIGTERM,
    so the last few seconds of spans never reach the collector.
    """

    class _FakeProvider:
        def __init__(self) -> None:
            self.shutdown_calls = 0

        def shutdown(self) -> None:
            self.shutdown_calls += 1

    provider = _FakeProvider()
    tracer = telemetry.Tracer(otel_tracer=object(), provider=provider)
    tracer.shutdown()
    assert provider.shutdown_calls == 1


def test_real_tracer_shutdown_swallows_errors() -> None:
    """A shutdown failure must not crash the host process.

    The teardown is the very last thing before process exit; raising here
    would mask whatever actually triggered the shutdown.
    """

    class _BrokenProvider:
        def shutdown(self) -> None:
            raise RuntimeError("collector unreachable")

    tracer = telemetry.Tracer(otel_tracer=object(), provider=_BrokenProvider())
    tracer.shutdown()  # must not raise


def test_real_tracer_dispatches_to_otel(monkeypatch: pytest.MonkeyPatch) -> None:
    """When initialised with a tracer object, ``span`` delegates to it.

    We hand-roll a tiny fake that records ``start_as_current_span`` calls,
    so the test stays free of the OTel SDK while still exercising the
    branch that imports OTel's ``Status`` / ``StatusCode``. The import
    is shimmed at the module level for the same reason.
    """

    class _FakeSpan:
        def __init__(self) -> None:
            self.exceptions: list[BaseException] = []
            self.status: object = None

        def __enter__(self) -> _FakeSpan:
            return self

        def __exit__(self, *_args: object) -> bool:
            return False

        def record_exception(self, exc: BaseException) -> None:
            self.exceptions.append(exc)

        def set_status(self, status: object) -> None:
            self.status = status

    class _FakeOtelTracer:
        def __init__(self) -> None:
            self.spans: list[tuple[str, dict[str, object]]] = []

        def start_as_current_span(self, name: str, attributes: dict[str, object]) -> _FakeSpan:
            self.spans.append((name, attributes))
            return _FakeSpan()

    fake = _FakeOtelTracer()
    tracer = telemetry.Tracer(fake)
    assert tracer.enabled is True

    # Shim the lazy ``opentelemetry.trace`` import inside ``_start_span``
    # so the test doesn't depend on the SDK being installed.
    fake_module = types.ModuleType("opentelemetry.trace")

    class _FakeStatus:
        def __init__(self, code: object, msg: str) -> None:
            self.code = code
            self.msg = msg

    class _FakeStatusCode:
        ERROR = "ERROR"

    fake_module.Status = _FakeStatus  # type: ignore[attr-defined]
    fake_module.StatusCode = _FakeStatusCode  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "opentelemetry.trace", fake_module)

    with tracer.span("op", group="g", model=None):
        pass
    assert fake.spans == [("op", {"group": "g"})]

    with pytest.raises(RuntimeError), tracer.span("bad", group="g"):
        raise RuntimeError("boom")
