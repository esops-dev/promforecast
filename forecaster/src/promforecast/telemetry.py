"""Optional OpenTelemetry tracing for the forecaster.

The forecaster does not require OpenTelemetry. When ``telemetry.enabled`` is
``False`` (the default), :class:`Tracer` returns a ``nullcontext`` from
``span(...)`` and the ``opentelemetry`` package is never imported, so the
import-time cost of the SDK is paid only by deployments that actually want
tracing.

When tracing *is* enabled, :func:`build_tracer` initialises an OTLP-gRPC
exporter and returns a :class:`Tracer` that wraps an OpenTelemetry tracer.
Callers don't need to branch on whether tracing is on — they always invoke
``with tracer.span("name", **attrs) as span:`` and the tracer takes care of
the no-op vs real-span dispatch.

The OpenTelemetry SDK lives behind a setuptools extra (``promforecast[otel]``)
so platforms that don't want the SDK don't have to install it. The container
image bakes the extra in by default for the same reason it ships the HA
extras — flipping ``telemetry.enabled`` shouldn't need a custom build.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

    from .config import TelemetryConfig

logger = structlog.get_logger(__name__)


class Tracer:
    """Tracing facade with a zero-overhead disabled path.

    Holds either an OpenTelemetry tracer instance (when tracing is enabled
    and the SDK installed) or ``None`` (the no-op path). ``span(...)``
    always returns a context manager so the call sites can use a single
    ``with`` shape without branching.
    """

    def __init__(self, otel_tracer: object | None = None, provider: object | None = None) -> None:
        self._tracer = otel_tracer
        # The TracerProvider is owned by this object so shutdown can flush
        # any in-flight BatchSpanProcessor queue at process exit. Kept
        # separate from the tracer reference because the disabled tracer
        # has neither.
        self._provider = provider

    @property
    def enabled(self) -> bool:
        return self._tracer is not None

    def shutdown(self) -> None:
        """Flush queued spans and tear down the OTel pipeline.

        Safe to call when tracing is disabled (does nothing). Safe to call
        twice — OTel's own ``shutdown`` is idempotent. Called from the
        FastAPI lifespan teardown so SIGTERM gives the BatchSpanProcessor
        a chance to flush rather than dropping its queue.
        """
        if self._provider is None:
            return
        shutdown = getattr(self._provider, "shutdown", None)
        if shutdown is None:
            return
        try:
            shutdown()
        except Exception as exc:
            logger.warning("telemetry_shutdown_failed", error=str(exc))

    def span(self, name: str, **attributes: Any) -> AbstractContextManager[object]:
        """Open a span (no-op when tracing is disabled).

        Attributes are set on the span at creation time. We deliberately
        only accept primitive types (str | int | float | bool) — OTel
        rejects everything else at runtime, and silently coercing dicts
        would hide configuration mistakes.
        """
        if self._tracer is None:
            return nullcontext()
        return self._start_span(name, attributes)

    @contextmanager
    def _start_span(self, name: str, attributes: dict[str, Any]) -> Iterator[object]:
        """Yield an active OTel span; recorded errors propagate as usual."""
        # The opentelemetry import lives inside the method, not the module,
        # so deployments that don't enable tracing never load the SDK.
        from opentelemetry.trace import Status, StatusCode  # noqa: PLC0415

        cleaned = {k: v for k, v in attributes.items() if v is not None}
        with self._tracer.start_as_current_span(name, attributes=cleaned) as span:  # type: ignore[union-attr]
            try:
                yield span
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                raise


_DISABLED_TRACER = Tracer()


def disabled() -> Tracer:
    """Return a shared no-op tracer.

    Used by call sites (tests, validate, ad-hoc tooling) that don't have a
    config and just want the disabled-path object.
    """
    return _DISABLED_TRACER


def build_tracer(config: TelemetryConfig) -> Tracer:
    """Initialise OpenTelemetry and return a real :class:`Tracer`.

    Disabled config returns the shared no-op tracer without any imports
    against ``opentelemetry``. When enabled, the OTLP-gRPC exporter is
    wired against ``config.otlp.endpoint`` and a parent-based ratio
    sampler is installed.

    If the SDK import fails (the ``[otel]`` extra wasn't installed in this
    environment), we log a warning and fall back to the disabled tracer
    rather than crash the process — tracing is observability, not a
    correctness dependency.
    """
    if not config.enabled:
        return _DISABLED_TRACER

    try:
        otel_tracer, provider = _initialise_otel(config)
    except ImportError as exc:
        logger.warning(
            "telemetry_disabled_sdk_missing",
            error=str(exc),
            hint=(
                "install the optional dependency: pip install 'promforecast[otel]' "
                "or 'opentelemetry-sdk' + 'opentelemetry-exporter-otlp-proto-grpc'"
            ),
        )
        return _DISABLED_TRACER

    logger.info(
        "telemetry_enabled",
        endpoint=config.otlp.endpoint,
        service_name=config.service_name,
        sample_ratio=config.sample_ratio,
    )
    return Tracer(otel_tracer, provider=provider)


def _initialise_otel(config: TelemetryConfig) -> tuple[object, object]:
    """Construct the OTel tracer provider + OTLP exporter.

    Returns ``(tracer, provider)``: the tracer is the per-call surface,
    the provider is retained so the lifespan can flush queued spans on
    shutdown. Kept private so the public ``build_tracer`` can swallow the
    ``ImportError`` cleanly without callers having to know which SDK
    submodules are involved.
    """
    # Local imports keep the disabled path import-free. Mypy can't follow
    # third-party SDK types accurately without stubs, so we accept the
    # ``Any`` surface here.
    from opentelemetry import trace  # noqa: PLC0415
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (  # noqa: PLC0415
        OTLPSpanExporter,
    )
    from opentelemetry.sdk.resources import Resource  # noqa: PLC0415
    from opentelemetry.sdk.trace import TracerProvider  # noqa: PLC0415
    from opentelemetry.sdk.trace.export import BatchSpanProcessor  # noqa: PLC0415
    from opentelemetry.sdk.trace.sampling import (  # noqa: PLC0415
        ParentBased,
        TraceIdRatioBased,
    )

    resource = Resource.create({"service.name": config.service_name})
    sampler = ParentBased(root=TraceIdRatioBased(config.sample_ratio))
    provider = TracerProvider(resource=resource, sampler=sampler)
    exporter = OTLPSpanExporter(
        endpoint=config.otlp.endpoint,
        insecure=config.otlp.insecure,
        headers=tuple(config.otlp.headers.items()) if config.otlp.headers else None,
        timeout=int(config.otlp.timeout.total_seconds()),
    )
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    # Calm the OTel SDK's stdlib logger so a temporary collector outage
    # doesn't fill stdout with "Failed to export" stack traces; our own
    # structlog pipeline already surfaces critical issues.
    logging.getLogger("opentelemetry").setLevel(logging.WARNING)
    return trace.get_tracer("promforecast"), provider


__all__ = [
    "Tracer",
    "build_tracer",
    "disabled",
]
