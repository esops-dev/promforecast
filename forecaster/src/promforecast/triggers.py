"""Forecast-driven webhook triggers.

Evaluates each configured trigger's PromQL ``condition`` against the
datasource after every group run and POSTs a structured forecast event to
the trigger's ``webhook_url`` when the condition transitions from false to
true. For sustained conditions, re-fires are throttled to at most one per
``repeat_interval``.

Different from a PrometheusRule alert: the payload carries forecast context
(the condition's matched series and their values — which naturally include
the forecast value, the model identity, and any band/drift labels the
operator's PromQL selects). Different from a webhook *regressor*: the
direction is inverse — the forecaster pushes out rather than pulls in.

Delivery never blocks or fails a fit: a condition-evaluation error is logged
and skipped, and a delivery failure bumps
``forecast_trigger_fires_total{outcome=http_error|timeout}`` and moves on.
Outbound payloads can be HMAC-signed (the ``X-Promforecast-Signature``
header) when the trigger sets a ``signing`` block.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from string import Template
from typing import TYPE_CHECKING

import httpx
import structlog

from . import signing
from .config import TriggerConfig

if TYPE_CHECKING:
    from .exporter import Exporter
    from .source import InstantSample, PromSource

logger = structlog.get_logger(__name__)


#: Default coalescing window. The runner evaluates triggers from each
#: group's post-run hook, so with G groups and T triggers a naive pass
#: issues T*G instant queries per refresh cycle — and in the common case
#: (all groups sharing ``server.refresh_interval``) those completions land
#: in one burst against an unchanged metric state. Collapsing re-evaluations
#: that fall within this window cuts the burst to one query per trigger
#: while staying far below any realistic refresh interval (minutes-to-hours),
#: so rising-edge detection latency is unaffected in practice.
_DEFAULT_COALESCE_SECONDS = 5.0


@dataclass
class _TriggerState:
    was_true: bool = False
    last_fired: float = field(default=float("-inf"))
    # Monotonic time of the last *evaluation* (query issued), used to
    # coalesce the per-group-hook burst. Distinct from ``last_fired``,
    # which tracks the last *delivery* for repeat-interval throttling.
    last_evaluated: float = field(default=float("-inf"))


def condition_is_true(samples: list[InstantSample]) -> bool:
    """A condition is true when it matches at least one non-zero sample.

    Mirrors how Prometheus alerting treats a vector: an empty result is
    "inactive", and a sample's truthiness is its value being non-zero
    (NaN samples were already dropped by the instant-query parser).
    """
    return any(s.value != 0.0 for s in samples)


class TriggerEngine:
    """Evaluates triggers and delivers their webhooks.

    ``client`` is injectable for tests (an ``httpx.AsyncClient`` over a
    ``MockTransport``); ``clock`` and ``now_iso`` are injectable so
    repeat-interval and timestamp assertions don't depend on wall-clock.
    """

    def __init__(
        self,
        *,
        triggers: list[TriggerConfig],
        source: PromSource,
        exporter: Exporter,
        client: httpx.AsyncClient | None = None,
        clock: Callable[[], float] = time.monotonic,
        now_iso: Callable[[], str] | None = None,
        coalesce_seconds: float = _DEFAULT_COALESCE_SECONDS,
    ) -> None:
        self._triggers = triggers
        self._source = source
        self._exporter = exporter
        self._coalesce_seconds = max(0.0, coalesce_seconds)
        if client is None:
            self._client = httpx.AsyncClient(
                limits=httpx.Limits(max_connections=4, max_keepalive_connections=2),
            )
            self._owned_client = True
        else:
            self._client = client
            self._owned_client = False
        self._clock = clock
        self._now_iso = now_iso or (lambda: datetime.now(tz=UTC).isoformat())
        self._states: dict[str, _TriggerState] = {t.name: _TriggerState() for t in triggers}

    async def aclose(self) -> None:
        if self._owned_client:
            await self._client.aclose()

    async def evaluate(self) -> None:
        """Evaluate every configured trigger once.

        Each trigger is isolated: an unexpected error in one (e.g. a
        signing ``secret_ref`` that resolves to an empty value) is logged
        and does not prevent the remaining triggers from being evaluated.
        """
        for trigger in self._triggers:
            try:
                await self._evaluate_one(trigger)
            except Exception:
                logger.exception("trigger_evaluate_failed", name=trigger.name)

    async def _evaluate_one(self, trigger: TriggerConfig) -> None:
        state = self._states.setdefault(trigger.name, _TriggerState())
        now = self._clock()
        # Coalesce the per-group-hook burst: if this trigger was evaluated
        # within the window, the metric state can't meaningfully have moved,
        # so skip the redundant instant query. State is untouched (no edge
        # decision is made), so the next non-coalesced evaluation sees fresh
        # data — this only defers, never drops.
        if now - state.last_evaluated < self._coalesce_seconds:
            return
        state.last_evaluated = now
        try:
            samples = await self._source.query_instant(trigger.condition)
        except Exception as exc:
            # A bad condition or a TSDB blip must not block the fit loop —
            # it's not a *delivery* outcome, so the counter stays untouched.
            logger.warning("trigger_condition_failed", name=trigger.name, error=str(exc))
            return
        is_true = condition_is_true(samples)
        repeat = trigger.repeat_interval.total_seconds()
        rising_edge = is_true and not state.was_true
        sustained = is_true and state.was_true and (now - state.last_fired) >= repeat
        state.was_true = is_true
        if not (rising_edge or sustained):
            return
        state.last_fired = now
        await self._fire(trigger, samples)

    async def _fire(self, trigger: TriggerConfig, samples: list[InstantSample]) -> None:
        body = self._render_payload(trigger, samples)
        headers = {
            "Content-Type": "application/json",
            **signing.expand_header_env_refs(trigger.headers),
        }
        if trigger.signing is not None:
            secret = signing.resolve_secret_ref(trigger.signing.secret_ref)
            headers[signing.SIGNATURE_HEADER] = signing.compute_signature(secret, body)
        outcome = await self._deliver(trigger, body, headers)
        self._exporter.increment_trigger_fire(trigger.name, outcome)
        logger.info("trigger_fired", name=trigger.name, outcome=outcome, samples=len(samples))

    async def _deliver(self, trigger: TriggerConfig, body: bytes, headers: dict[str, str]) -> str:
        try:
            response = await self._client.post(
                trigger.webhook_url,
                content=body,
                headers=headers,
                timeout=trigger.timeout.total_seconds(),
            )
        except (TimeoutError, httpx.TimeoutException):
            logger.warning("trigger_delivery_timeout", name=trigger.name)
            return "timeout"
        except httpx.HTTPError as exc:
            logger.warning("trigger_delivery_error", name=trigger.name, error=str(exc))
            return "http_error"
        if response.status_code >= 400:  # noqa: PLR2004
            logger.warning(
                "trigger_delivery_status",
                name=trigger.name,
                status=response.status_code,
            )
            return "http_error"
        return "success"

    def _render_payload(self, trigger: TriggerConfig, samples: list[InstantSample]) -> bytes:
        event = _build_event(trigger=trigger, samples=samples, fired_at=self._now_iso())
        if not trigger.payload_template.strip():
            return json.dumps(event).encode("utf-8")
        # ``string.Template`` ($var) substitution — chosen over str.format so
        # a JSON template's literal ``{`` / ``}`` don't need escaping.
        context = {
            "name": event["name"],
            "condition": event["condition"],
            "value": event["value"],
            "fired_at": event["fired_at"],
            "samples_json": json.dumps(event["samples"]),
        }
        rendered = Template(trigger.payload_template).safe_substitute(context)
        return rendered.encode("utf-8")


def _build_event(
    *,
    trigger: TriggerConfig,
    samples: list[InstantSample],
    fired_at: str,
) -> dict[str, object]:
    """Assemble the default JSON event for a fired trigger.

    ``value`` / ``labels`` reflect the first matched series so a simple
    Slack template can interpolate a single number; ``samples`` carries the
    full matched vector (each series' labels — which include the model
    identity and any band/drift discriminators the condition selected — plus
    its value) so richer receivers have the full forecast context.
    """
    first = samples[0] if samples else None
    return {
        "name": trigger.name,
        "condition": trigger.condition,
        "fired_at": fired_at,
        "value": first.value if first is not None else None,
        "labels": dict(first.labels) if first is not None else {},
        "samples": [{"labels": dict(s.labels), "value": s.value} for s in samples],
    }
