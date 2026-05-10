"""Forecast push sinks.

The default ``/metrics`` exporter is pull-based and only carries the *next*
forecast value. The ``remote_write`` sink complements it: after each run
the full per-step forecast curve is pushed with explicit future timestamps
so a TSDB that accepts forward-dated samples (VictoriaMetrics with
``-futureRetention``, Prometheus with the ``out_of_order`` feature flag,
Mimir, Thanos receive) becomes a queryable history of past predictions.

The wire format we use is the Prometheus text exposition with a trailing
millisecond timestamp — the same body shape that VictoriaMetrics' bundled
``/api/v1/import/prometheus`` endpoint accepts. This avoids pulling
``snappy`` + ``protobuf`` into the runtime: the typical deployment writes
to VictoriaMetrics anyway, and shops with stricter ingestion paths can
front the sink with a translator. A future v0.x can swap in proper
remote-write protobuf without changing the runner-facing API.

Failures are bounded to a couple of reason tokens
(``http_error`` | ``timeout``) so the
``forecast_remote_write_failures_total`` counter has a small, predictable
label set.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import httpx
import structlog
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class ForecastCurvePoint:
    """A single forecast row along the horizon, with an explicit future ts.

    Distinct from :class:`promforecast.exporter.ForecastPoint` which carries
    the snapshot's ``next``-step row only and has no notion of a future
    timestamp (Prometheus assigns scrape time on pull).
    """

    metric: str
    labels: dict[str, str]
    value: float
    timestamp_ms: int


class SinkError(RuntimeError):
    """Raised when a sink push fails. ``reason`` is a bounded label token.

    ``status_code`` is set for HTTP-status failures so the retry policy can
    distinguish transient 5xx from deterministic 4xx without re-parsing
    the message string. ``None`` for transport / timeout / encode errors.
    """

    def __init__(self, reason: str, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.reason = reason
        self.status_code = status_code


@runtime_checkable
class ForecastSink(Protocol):
    """Pluggable push sink for the forecast curve.

    ``write`` accepts an optional ``on_retry`` callback that the sink
    invokes once per retry attempt with a bounded reason token. The
    runner uses this to bump ``forecast_remote_write_retries_total``
    with the calling group's label so transient failures are visible
    independently of terminal failures.
    """

    name: str

    async def write(
        self,
        points: Iterable[ForecastCurvePoint],
        *,
        on_retry: Callable[[str], None] | None = None,
    ) -> None: ...

    async def aclose(self) -> None: ...


#: Default rows per HTTP body. At ~70 bytes per Prometheus-text line, 10k
#: rows ≈ 700 KiB — comfortably below the typical TSDB request-size cap
#: (VictoriaMetrics defaults to 64 MiB; Prometheus' remote_write recommends
#: 500 KiB chunks). Tunable via :class:`RemoteWriteSink` for shops behind
#: tight ingress limits.
DEFAULT_CHUNK_SIZE = 10_000


class RemoteWriteSink:
    """Push the full forecast curve to a TSDB ingest endpoint.

    Targets the VictoriaMetrics-compatible Prometheus text-import endpoint
    by default (``/api/v1/import/prometheus``). Each call to :meth:`write`
    chunks the curve into bounded HTTP bodies (default 10k rows) and POSTs
    them sequentially. Sequential rather than concurrent because the
    forecaster usually runs against a single TSDB; parallel pushes would
    only buy throughput when the sink is the bottleneck, and chunking
    already keeps each request fast enough to be irrelevant for typical
    cardinality.

    A failure on any chunk raises immediately and aborts the rest of the
    push for that run; the next refresh interval will re-send the full
    curve, which is fine because the TSDB dedupes by ``(labels,
    timestamp)``.
    """

    name = "remote_write"

    def __init__(
        self,
        url: str,
        timeout_seconds: float,
        headers: dict[str, str] | None = None,
        client: httpx.AsyncClient | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        max_connections: int = 4,
        max_keepalive_connections: int = 2,
    ) -> None:
        if not url:
            raise ValueError("RemoteWriteSink requires a non-empty url")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self._url = url
        self._timeout_seconds = timeout_seconds
        self._headers = {"Content-Type": "text/plain; version=0.0.4", **(headers or {})}
        # Explicit pool limits — without them, httpx's 100/20 default
        # masks any future change to push concurrency. The sink is shared
        # across groups, and groups push concurrently when their refresh
        # intervals overlap, so the pool sizing matters when many groups
        # are configured. ``limits`` is exposed publicly so callers can
        # introspect the resolved values without reaching into httpx's
        # private transport internals; ``None`` when a custom client was
        # injected (the caller owns its limits).
        if client is None:
            self.limits: httpx.Limits | None = httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_keepalive_connections,
            )
            self._client = httpx.AsyncClient(timeout=timeout_seconds, limits=self.limits)
        else:
            self.limits = None
            self._client = client
        # ``_owned_client`` controls whether ``aclose`` shuts the client down.
        # When tests inject a client, the test owns its lifecycle.
        self._owned_client = client is None
        self._chunk_size = chunk_size

    async def write(
        self,
        points: Iterable[ForecastCurvePoint],
        *,
        on_retry: Callable[[str], None] | None = None,
    ) -> None:
        # Materialise into a list so we can index it for chunking. The
        # caller is the runner, which already builds a list per group, so
        # this isn't an extra allocation in the hot path.
        points_list = list(points)
        if not points_list:
            return
        for start in range(0, len(points_list), self._chunk_size):
            await self._post(points_list[start : start + self._chunk_size], on_retry)

    async def _post(
        self,
        chunk: list[ForecastCurvePoint],
        on_retry: Callable[[str], None] | None,
    ) -> None:
        body = encode_prometheus_text(chunk)
        if not body:
            return
        # Retry transient flakes (transport errors, timeouts, 5xx) with
        # exponential backoff so a single TSDB blip doesn't open an
        # entire refresh-interval gap in pushed forecasts. 4xx still
        # fails fast — those are deterministic and won't recover.
        encoded = body.encode("utf-8")
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=0.5, max=4),
            retry=retry_if_exception(_is_retryable_sink_error),
            before_sleep=_retry_recorder(on_retry),
            reraise=True,
        ):
            with attempt:
                await self._post_once(encoded)

    async def _post_once(self, body: bytes) -> None:
        try:
            response = await self._client.post(
                self._url,
                content=body,
                headers=self._headers,
                timeout=self._timeout_seconds,
            )
        except TimeoutError as exc:
            raise SinkError("timeout", f"remote_write timed out: {self._url}") from exc
        except httpx.TimeoutException as exc:
            raise SinkError("timeout", f"remote_write timed out: {self._url}") from exc
        except httpx.HTTPError as exc:
            raise SinkError("http_error", f"remote_write transport error: {exc}") from exc
        if response.status_code >= 400:  # noqa: PLR2004
            raise SinkError(
                "http_error",
                f"remote_write returned {response.status_code}: {response.text[:200]}",
                status_code=response.status_code,
            )

    async def aclose(self) -> None:
        if self._owned_client:
            await self._client.aclose()


def _retry_recorder(
    on_retry: Callable[[str], None] | None,
) -> Callable[[RetryCallState], None] | None:
    """Build a tenacity ``before_sleep`` hook that bumps a retry counter.

    Returning ``None`` when the caller didn't supply a callback skips
    tenacity's ``before_sleep`` machinery entirely on the no-op path.
    """
    if on_retry is None:
        return None

    def _record(retry_state: RetryCallState) -> None:
        if retry_state.outcome is None:
            return
        exc = retry_state.outcome.exception()
        if isinstance(exc, SinkError):
            on_retry(exc.reason)

    return _record


def _is_retryable_sink_error(exc: BaseException) -> bool:
    """Retry transport flakes, timeouts, and 5xx; let 4xx fail fast.

    A 4xx (bad URL, missing auth header, body rejected) is deterministic
    and would just multiply the visible failure latency. 5xx and timeouts
    are exactly the cases where backoff helps.
    """
    if not isinstance(exc, SinkError):
        return False
    if exc.reason == "timeout":
        return True
    if exc.reason == "http_error" and exc.status_code is None:
        # transport-level error (DNS, connection reset) — retryable
        return True
    if exc.reason == "http_error" and exc.status_code is not None:
        return exc.status_code >= 500  # noqa: PLR2004
    return False


def encode_prometheus_text(points: Iterable[ForecastCurvePoint]) -> str:
    """Serialise points as Prometheus text-format with millisecond timestamps.

    Pure function — separated from :class:`RemoteWriteSink` so the formatter
    can be unit-tested without touching the network.
    """
    lines: list[str] = []
    for p in points:
        lines.append(f"{p.metric}{{{_format_labels(p.labels)}}} {p.value} {p.timestamp_ms}")
    return "\n".join(lines)


def _format_labels(labels: dict[str, str]) -> str:
    """Render a label set in the standard ``k="v",k="v"`` form.

    Sorted for deterministic output (hashing-friendly, easier to diff).
    Values are escaped per the Prometheus exposition spec — backslash,
    double quote, and newline are the only required escapes.
    """
    return ",".join(f'{k}="{_escape(v)}"' for k, v in sorted(labels.items()))


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
