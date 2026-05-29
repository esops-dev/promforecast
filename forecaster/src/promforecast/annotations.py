"""Grafana-annotations sink for significant forecast events.

When enabled, the forecaster posts a Grafana annotation each time an enabled
event *newly* fires — a change-point detected, a drift threshold crossed, a
deploy-risk elevation, a predicted threshold breach, or an actual landing
outside its band. Operators correlate forecast events with the dashboards
they already watch, without manually overlaying timelines.

The runner builds a list of :class:`AnnotationEvent` from each group run and
hands it to :meth:`GrafanaAnnotationSink.post_events`. The sink filters to
the configured event families, suppresses repeats with a rising-edge dedup
(so a sustained "outside band" posts once, not every refresh), and POSTs to
Grafana's ``/api/annotations`` endpoint (compatible with both Grafana OSS and
Grafana Cloud). Delivery never blocks a fit: a failure bumps
``forecast_annotation_posts_total{outcome=http_error|timeout}`` and moves on.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import httpx
import structlog

from . import signing
from .config import AnnotationsConfig

if TYPE_CHECKING:
    from .exporter import Exporter

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class AnnotationEvent:
    """A single significant forecast event destined for Grafana.

    ``event_type`` is one of the configured families; ``dedup_key`` makes
    the (type, series) pair unique so the rising-edge dedup can suppress
    repeats while a condition stays active; ``tags`` and ``text`` are what
    Grafana renders.
    """

    event_type: str
    dedup_key: str
    tags: list[str]
    text: str


class GrafanaAnnotationSink:
    """Posts forecast events to Grafana's annotations API.

    ``client`` is injectable for tests; ``now_ms`` is injectable so the
    posted timestamp is deterministic under test.
    """

    def __init__(
        self,
        config: AnnotationsConfig,
        exporter: Exporter,
        *,
        client: httpx.AsyncClient | None = None,
        now_ms: Callable[[], int] = lambda: int(time.time() * 1000),
    ) -> None:
        self._config = config
        self._exporter = exporter
        self._enabled_events = set(config.events)
        self._url = config.grafana_url.rstrip("/") + "/api/annotations"
        if client is None:
            self._client = httpx.AsyncClient(
                limits=httpx.Limits(max_connections=4, max_keepalive_connections=2),
            )
            self._owned_client = True
        else:
            self._client = client
            self._owned_client = False
        self._now_ms = now_ms
        # Keys currently firing -> the scope that posted them. Used to post
        # only the rising edge. The value records *which* scope (group) owns
        # a key so a per-group call only retires its own keys and can't clear
        # another group's still-active annotations (the runner calls
        # ``post_events`` once per group, each with only that group's events
        # plus the global deploy-risk events).
        self._active: dict[tuple[str, str], str] = {}

    async def aclose(self) -> None:
        if self._owned_client:
            await self._client.aclose()

    async def post_events(self, events: list[AnnotationEvent], *, scope: str = "") -> None:
        """Post the rising edge of each enabled event family.

        ``events`` is the set of events firing this run *for ``scope``* (the
        runner passes the group name so concurrent groups don't interfere).
        A key that is freshly firing posts an annotation; a key owned by this
        scope that fired last run but isn't in ``events`` now is cleared so it
        can fire again later. Keys owned by *other* scopes are left untouched
        — without this, a per-group call would clear sibling groups' active
        keys and re-post every refresh, defeating the rising-edge dedup. Keys
        for disabled event families are ignored entirely (they never enter
        the active set, so toggling a family on doesn't retroactively post).
        """
        firing = {
            (e.event_type, e.dedup_key): e for e in events if e.event_type in self._enabled_events
        }
        # Retire only keys this scope owns that are no longer firing (and
        # whose family is still enabled). A key owned by another scope, or
        # whose family was disabled, stays put.
        for key, owner in list(self._active.items()):
            if owner == scope and key not in firing and key[0] in self._enabled_events:
                del self._active[key]
        for key, event in firing.items():
            if key in self._active:
                continue  # already posted on a prior run; suppress the repeat
            self._active[key] = scope
            await self._post_one(event)

    async def _post_one(self, event: AnnotationEvent) -> None:
        ts = self._now_ms()
        payload = {"time": ts, "tags": event.tags, "text": event.text}
        headers = {"Content-Type": "application/json"}
        if self._config.api_token_ref.strip():
            token = signing.resolve_secret_ref(self._config.api_token_ref)
            headers["Authorization"] = f"Bearer {token}"
        outcome = await self._deliver(event, payload, headers)
        self._exporter.increment_annotation_post(event.event_type, outcome)
        logger.info("annotation_posted", event_type=event.event_type, outcome=outcome)

    async def _deliver(
        self, event: AnnotationEvent, payload: dict[str, object], headers: dict[str, str]
    ) -> str:
        try:
            response = await self._client.post(
                self._url,
                json=payload,
                headers=headers,
                timeout=self._config.timeout.total_seconds(),
            )
        except (TimeoutError, httpx.TimeoutException):
            logger.warning("annotation_timeout", event_type=event.event_type)
            return "timeout"
        except httpx.HTTPError as exc:
            logger.warning("annotation_error", event_type=event.event_type, error=str(exc))
            return "http_error"
        if response.status_code >= 400:  # noqa: PLR2004
            logger.warning(
                "annotation_status", event_type=event.event_type, status=response.status_code
            )
            return "http_error"
        return "success"
