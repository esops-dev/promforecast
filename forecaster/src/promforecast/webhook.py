"""Webhook regressor client.

Pulls a regressor signal that does not live in Prometheus and is not on the
built-in calendar list — active incidents from PagerDuty, deploys-in-flight
from a CI/CD API, a custom internal "promotion-day" calendar — by POSTing to
an operator-configured URL at fit time and parsing a JSON time series from
the response.

The request body carries the query id and the timestamp range; the response
is ``{"timestamps": [...], "values": [...]}``. Successful responses are
cached per (url, query_id, lookback) for ``cache_ttl`` so a primary query
returning many series — or a short refresh interval — doesn't hammer the
external service. The forecaster stays stateless: the webhook owns whatever
state it queries.

Failures are bounded to two reason tokens so the
``forecast_regressor_failures_total`` label set stays small:

* ``webhook_failure`` — timeout, transport error, non-2xx status, or an
  unparseable / mis-shaped body.
* ``signature_invalid`` — HMAC verification was requested but the response
  carried a missing or mismatched signature.

Both flow through the existing per-regressor graceful-degradation path: an
optional webhook regressor that fails is dropped (the forecast still runs);
a ``required: true`` one skips the series.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

import httpx
import structlog

from . import signing
from .config import RegressorConfig
from .source import SeriesFrame

logger = structlog.get_logger(__name__)


class WebhookError(RuntimeError):
    """Raised when a webhook regressor cannot be resolved.

    ``reason`` is a bounded token (``webhook_failure`` | ``signature_invalid``)
    suitable for use as a Prometheus counter label.
    """

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass
class _CacheEntry:
    frames: list[SeriesFrame]
    expires_at: float


class WebhookRegressorClient:
    """Fetches + caches webhook-regressor time series.

    ``client`` may be injected (tests pass an ``httpx.AsyncClient`` backed by
    a ``MockTransport``); when omitted a pooled async client is created and
    owned by this instance. ``clock`` is injectable so cache-expiry tests
    don't sleep.
    """

    def __init__(
        self,
        client: httpx.AsyncClient | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if client is None:
            self._client = httpx.AsyncClient(
                limits=httpx.Limits(max_connections=4, max_keepalive_connections=2),
            )
            self._owned_client = True
        else:
            self._client = client
            self._owned_client = False
        self._clock = clock
        self._cache: dict[tuple[str, str, int], _CacheEntry] = {}

    async def aclose(self) -> None:
        if self._owned_client:
            await self._client.aclose()

    async def fetch(
        self,
        *,
        reg: RegressorConfig,
        query_id: str,
        start: datetime,
        end: datetime,
    ) -> list[SeriesFrame]:
        """Return one :class:`SeriesFrame` for the webhook regressor ``reg``.

        Reads from the per-(url, query_id, lookback) cache when a fresh entry
        exists; otherwise POSTs to the configured URL, verifies the optional
        HMAC signature, parses the JSON time series, caches it, and returns
        it. Raises :class:`WebhookError` on any failure.
        """
        url = str(reg.url)
        lookback = max(1, int((end - start).total_seconds()))
        cache_key = (url, query_id, lookback)
        entry = self._cache.get(cache_key)
        now = self._clock()
        if entry is not None and entry.expires_at > now:
            return entry.frames

        frames = await self._request(reg=reg, url=url, query_id=query_id, start=start, end=end)
        ttl = max(1, int(reg.cache_ttl.total_seconds()))
        self._cache[cache_key] = _CacheEntry(frames=frames, expires_at=now + ttl)
        return frames

    async def _request(
        self,
        *,
        reg: RegressorConfig,
        url: str,
        query_id: str,
        start: datetime,
        end: datetime,
    ) -> list[SeriesFrame]:
        body = {
            "query_id": query_id,
            "regressor_id": reg.id,
            "start": start.isoformat(),
            "end": end.isoformat(),
        }
        try:
            response = await self._client.post(
                url,
                json=body,
                headers=signing.expand_header_env_refs(reg.headers),
                timeout=reg.timeout.total_seconds(),
            )
        except (TimeoutError, httpx.TimeoutException) as exc:
            raise WebhookError("webhook_failure", f"webhook regressor timed out: {url}") from exc
        except httpx.HTTPError as exc:
            raise WebhookError(
                "webhook_failure", f"webhook regressor transport error: {exc}"
            ) from exc
        if response.status_code >= 400:  # noqa: PLR2004
            raise WebhookError(
                "webhook_failure",
                f"webhook regressor returned {response.status_code}: {response.text[:200]}",
            )
        if reg.signing is not None:
            self._verify(
                reg=reg, raw=response.content, header=response.headers.get(signing.SIGNATURE_HEADER)
            )
        return [_parse_series(reg_id=reg.id, payload=_decode_json(response))]

    @staticmethod
    def _verify(*, reg: RegressorConfig, raw: bytes, header: str | None) -> None:
        assert reg.signing is not None  # caller-guarded
        try:
            secret = signing.resolve_secret_ref(reg.signing.secret_ref)
        except signing.SecretResolutionError as exc:
            # A missing/empty signing secret is a config error, but we still
            # route it through the regressor's graceful-degradation path
            # (rather than crashing the whole query) so one mis-set Secret
            # doesn't take down every series of the query.
            raise WebhookError(
                "signature_invalid",
                f"webhook regressor {reg.id!r} signing secret could not be resolved: {exc}",
            ) from exc
        if not signing.verify_signature(secret, raw, header):
            raise WebhookError(
                "signature_invalid",
                f"webhook regressor {reg.id!r} response failed HMAC verification",
            )


def _decode_json(response: httpx.Response) -> dict[str, object]:
    try:
        payload = response.json()
    except ValueError as exc:
        raise WebhookError("webhook_failure", f"webhook response was not JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise WebhookError("webhook_failure", "webhook response was not a JSON object")
    return payload


def _parse_series(*, reg_id: str, payload: dict[str, object]) -> SeriesFrame:
    """Build a :class:`SeriesFrame` from ``{"timestamps": [...], "values": [...]}``.

    Timestamps may be epoch seconds (int/float) or ISO-8601 strings. NaN /
    unparseable points are dropped pairwise so a single bad sample doesn't
    poison the whole series.
    """
    raw_ts = payload.get("timestamps")
    raw_vals = payload.get("values")
    if not isinstance(raw_ts, list) or not isinstance(raw_vals, list):
        raise WebhookError(
            "webhook_failure",
            "webhook response must contain 'timestamps' and 'values' arrays",
        )
    if len(raw_ts) != len(raw_vals):
        raise WebhookError(
            "webhook_failure",
            f"webhook 'timestamps' ({len(raw_ts)}) and 'values' ({len(raw_vals)}) "
            "have different lengths",
        )
    timestamps: list[datetime] = []
    values: list[float] = []
    for ts, val in zip(raw_ts, raw_vals, strict=True):
        parsed_ts = _parse_timestamp(ts)
        parsed_val = _parse_value(val)
        if parsed_ts is None or parsed_val is None:
            continue
        timestamps.append(parsed_ts)
        values.append(parsed_val)
    if not values:
        raise WebhookError(
            "webhook_failure", "webhook response carried no usable (timestamp, value) pairs"
        )
    return SeriesFrame(labels={"regressor_id": reg_id}, timestamps=timestamps, values=values)


def _parse_timestamp(ts: object) -> datetime | None:
    from datetime import UTC  # noqa: PLC0415

    if isinstance(ts, bool):
        return None
    if isinstance(ts, (int, float)):
        try:
            return datetime.fromtimestamp(float(ts), tz=UTC)
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(ts, str):
        try:
            parsed = datetime.fromisoformat(ts)
        except ValueError:
            return None
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
    return None


def _parse_value(val: object) -> float | None:
    import math  # noqa: PLC0415

    if isinstance(val, bool):
        return None
    if isinstance(val, (int, float)):
        f = float(val)
        return None if math.isnan(f) else f
    return None
