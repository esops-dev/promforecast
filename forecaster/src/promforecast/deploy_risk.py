"""Deploy-anchored composite risk score.

When configured via the top-level ``deploy_signals:`` block, the
forecaster emits ``forecast_deploy_risk_score{deployment_id, service}``
as a weighted composite of per-signal deviation ratios — a single
number per active deploy that an SRE can alert on, instead of
manually correlating ``forecast_deviation_ratio`` across latency,
error-rate, saturation, and traffic gauges during a rollout.

The metric is *deploy-anchored*: it only emits during an active deploy
window. Without an active deploy the gauge disappears, so background
deviation noise doesn't accumulate into a permanent dashboard panel.

This module owns three pieces:

* :class:`DeployEvent` — one deploy occurrence with its detection
  timestamp and the configured active window.
* :class:`DeployTracker` — a small ring buffer of recent deploys per
  service, with expiry based on the per-service active window. Bounded
  by ``max_active_deploys`` so a runaway deploy pipeline can't blow
  the cardinality budget.
* :func:`compute_deploy_risk_scores` — pure function that takes a set
  of active deploys + per-service signal definitions + a snapshot of
  current deviation ratios and returns the weighted composite score
  per deploy.

Deploy *detection* is the runner's responsibility — it inspects the
configured anchor regressor's recent values during each fit cycle and
calls :meth:`DeployTracker.record` whenever a new event is observed.
This module stays pure-function so unit tests can exercise the
composite logic without spinning up the runner.

Deviation values are read from the current snapshot via the
``deviation_lookup`` callable, which keeps :func:`compute_deploy_risk_scores`
decoupled from exporter / point-factory internals (those reshuffle
between releases).
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta

import structlog

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class DeployEvent:
    """A single observed deploy.

    ``deployment_id`` is the operator-facing label that lands on the
    emitted metric. The runner derives it from the anchor regressor's
    sample timestamp + service name; deployments inside the same
    second collapse to one event (operators rarely deploy twice in
    one second, and when they do it's the same rollout).

    ``observed_at_seconds`` is a monotonic timestamp from ``time.monotonic()``
    so wall-clock skew or NTP adjustments can't corrupt the
    expiry logic. The active-window clock starts at observation.
    """

    deployment_id: str
    service: str
    observed_at_seconds: float
    active_window_seconds: float

    def expires_at(self) -> float:
        return self.observed_at_seconds + self.active_window_seconds


@dataclass(frozen=True)
class DeployRiskScore:
    """Weighted composite for one active deploy.

    ``score`` is the composite value (non-negative; higher = more risk).
    ``contributing_signals`` is the count of signals whose deviation
    ratio was finite enough to contribute — emitted as the
    ``contributing_signals`` label so dashboards can drop low-coverage
    scores (e.g. dropping rows where ``contributing_signals < 2`` keeps
    the gauge meaningful when half the signals dropped out due to
    NaN deviations).
    """

    deployment_id: str
    service: str
    score: float
    contributing_signals: int


class DeployTracker:
    """In-memory ring buffer of recent deploys, bounded per service.

    Single-replica scoped: each forecaster replica keeps its own
    tracker. In HA mode followers don't run scheduled fits, so they
    don't observe deploys; the leader's emitted scores are what land
    on /metrics via the shared snapshot cache.

    The bound exists so an abnormal deploy pipeline (a flapping
    Argo Rollouts loop, for example) can't run away with cardinality.
    Per service, the most recent ``max_per_service[service]`` events
    are kept; older events are evicted in LRU fashion.

    The cap is per-service rather than global because operators
    naturally tune ``deploy_signals[].max_active_deploys`` per
    service — a chatty internal service may legitimately overlap
    deploys more than a stable infrastructure tier.
    """

    def __init__(
        self,
        *,
        max_per_service: int | dict[str, int] = 16,
        default_max: int = 16,
    ) -> None:
        """Configure per-service caps.

        ``max_per_service`` accepts either an integer (applied to
        every service) or a service-name -> cap mapping. ``default_max``
        is the fallback when the mapping doesn't list a given service
        — relevant when the runner records an event for a service
        that has no spec in the config (a typo, or a deploy detected
        before the next reload registers the spec).
        """
        if isinstance(max_per_service, int):
            if max_per_service < 1:
                raise ValueError("max_per_service must be >= 1")
            self._per_service: dict[str, int] = {}
            self._default_max = max_per_service
        else:
            for service, cap in max_per_service.items():
                if cap < 1:
                    raise ValueError(f"max_per_service[{service!r}] must be >= 1")
            self._per_service = dict(max_per_service)
            if default_max < 1:
                raise ValueError("default_max must be >= 1")
            self._default_max = default_max
        self._events: dict[str, list[DeployEvent]] = {}

    def _cap_for(self, service: str) -> int:
        return self._per_service.get(service, self._default_max)

    def record(self, event: DeployEvent) -> None:
        """Register a new deploy event for the named service.

        De-duplicates on ``(service, deployment_id)`` — if the runner
        observes the same deploy twice (a second fit cycle within the
        active window) the existing event is preserved (and its
        observed_at_seconds is NOT updated; the active window starts
        at *first* observation so the same deploy doesn't extend
        forever).
        """
        bucket = self._events.setdefault(event.service, [])
        for existing in bucket:
            if existing.deployment_id == event.deployment_id:
                return
        bucket.append(event)
        cap = self._cap_for(event.service)
        if len(bucket) > cap:
            evicted = bucket.pop(0)
            logger.info(
                "deploy_tracker_evicted",
                service=event.service,
                deployment_id=evicted.deployment_id,
                reason="max_per_service",
                max_per_service=cap,
            )

    def expire(self, *, now_monotonic: float | None = None) -> int:
        """Drop deploys whose active window has elapsed.

        Returns the number of expired entries. ``now_monotonic``
        defaults to the live monotonic clock; tests can pass a stub
        for deterministic timing.
        """
        now = now_monotonic if now_monotonic is not None else time.monotonic()
        expired = 0
        for service, bucket in list(self._events.items()):
            keep = [e for e in bucket if e.expires_at() > now]
            expired += len(bucket) - len(keep)
            if keep:
                self._events[service] = keep
            else:
                del self._events[service]
        return expired

    def active(
        self,
        *,
        now_monotonic: float | None = None,
    ) -> list[DeployEvent]:
        """Return every still-active deploy across all services.

        Calls :meth:`expire` first so callers always see a fresh view —
        a deploy whose window elapsed between the previous call and
        this one is not returned even if expire hasn't been explicitly
        invoked.
        """
        self.expire(now_monotonic=now_monotonic)
        return [event for bucket in self._events.values() for event in bucket]

    def active_for_service(
        self,
        service: str,
        *,
        now_monotonic: float | None = None,
    ) -> list[DeployEvent]:
        """Return active deploys for one service (post-expiry)."""
        self.expire(now_monotonic=now_monotonic)
        return list(self._events.get(service, []))


# Type alias for the lookup callable: given (service, signal_query_id),
# return the most recent |deviation_ratio| for that signal as a non-
# negative float, or NaN if unavailable. Using the absolute value keeps
# the composite a meaningful magnitude (a -2-sigma deviation contributes
# just as much risk as a +2-sigma deviation).
DeviationLookup = Callable[[str, str], float]


@dataclass(frozen=True)
class SignalSpec:
    """One signal in a deploy's risk composite.

    Mirrors the shape of the :class:`DeploySignalConfig` config block
    but kept separate so pure-function consumers don't need to import
    the config schema. ``query_id`` references a configured forecaster
    query; ``weight`` is multiplied into the absolute deviation ratio
    before summing.
    """

    query_id: str
    weight: float


def compute_deploy_risk_scores(
    *,
    deploys: list[DeployEvent],
    signals_by_service: dict[str, list[SignalSpec]],
    deviation_lookup: DeviationLookup,
) -> list[DeployRiskScore]:
    """Compute the per-deploy composite scores.

    For each active deploy, look up the absolute deviation ratio of
    every configured signal under the deploy's service, multiply by
    the signal's weight, and average. The result is the **weighted
    mean** rather than the weighted sum — that keeps the score
    insensitive to "how many signals are configured" so adding a
    fourth golden signal doesn't mechanically inflate the score.

    Signals whose deviation lookup returns NaN are dropped from the
    average — the composite is computed over the signals that
    actually produced a deviation reading this cycle. The number of
    contributing signals is returned on the score so an alert can
    gate on a minimum (e.g. ``contributing_signals >= 2`` to avoid
    firing on a single-signal blip).

    Returns one score per deploy. A deploy whose service has no
    configured signals returns a score of 0.0 with
    ``contributing_signals = 0`` — visible in /metrics so the operator
    notices the misconfiguration (rather than silently dropping the
    deploy from emission, which would look identical to "no deploy
    happened").
    """
    out: list[DeployRiskScore] = []
    for deploy in deploys:
        signals = signals_by_service.get(deploy.service, [])
        weighted_sum = 0.0
        total_weight = 0.0
        contributing = 0
        for signal in signals:
            raw = deviation_lookup(deploy.service, signal.query_id)
            if not math.isfinite(raw):
                continue
            weighted_sum += abs(raw) * signal.weight
            total_weight += signal.weight
            contributing += 1
        score = weighted_sum / total_weight if total_weight > 0.0 else 0.0
        out.append(
            DeployRiskScore(
                deployment_id=deploy.deployment_id,
                service=deploy.service,
                score=score,
                contributing_signals=contributing,
            )
        )
    return out


def derive_deployment_id(service: str, observed_at_unix_seconds: float) -> str:
    """Deterministic deployment id for an observed event.

    The runner derives ids from wall-clock timestamps so dashboards
    can correlate events across promforecast and operator-side
    tooling (Argo Rollouts events, CI pipeline IDs) via timestamp
    proximity. The id format is intentionally a fixed-width string
    — ``<service>-<unix-seconds>`` — so the label set has bounded
    cardinality (one new value per deploy) and sorts chronologically.
    """
    return f"{service}-{int(observed_at_unix_seconds)}"


def detect_deploy_from_regressor(
    *,
    values: list[float],
    timestamps_unix: list[float],
    threshold: float = 0.0,
) -> tuple[bool, float | None]:
    """Inspect a regressor's recent values for a fresh deploy event.

    Returns ``(observed, timestamp)`` — ``observed`` is True when the
    leading row's value exceeds ``threshold`` (strictly greater than)
    *and* the previous row did not. That defines a "fresh" deploy:
    the regressor must have transitioned from "no recent deploy" to
    "deploy observed" between the two newest samples. Without the
    transition check, a regressor that stays high across multiple
    fit cycles would be re-detected as a new deploy each cycle.

    **Multi-deploy limitation.** Only the leading-pair edge is
    examined, so a sequence ``...0, 1, 0, 1`` between two fit cycles
    collapses to one detected deploy (the most recent transition).
    For services that deploy more often than the runner's
    ``refresh_interval`` the detector under-counts. The fix — walk
    the full slice + a per-(group, query, regressor) memo of the
    previous tail value — is a documented follow-up; the
    current behaviour is correct for the common case of one deploy
    per refresh window.

    ``timestamp`` is the unix-seconds wall-clock timestamp of the
    triggering sample, used by :func:`derive_deployment_id`. When
    ``observed`` is False, ``timestamp`` is None.

    Returns ``(False, None)`` for any of: empty values, fewer than
    two samples, non-finite values, the leading sample not crossing
    the threshold, or the previous sample already above the
    threshold.
    """
    if len(values) < 2 or len(timestamps_unix) != len(values):  # noqa: PLR2004
        return False, None
    leading = values[-1]
    previous = values[-2]
    if not math.isfinite(leading) or not math.isfinite(previous):
        return False, None
    if leading > threshold and previous <= threshold:
        return True, timestamps_unix[-1]
    return False, None


def active_window_seconds(active_window: timedelta) -> float:
    """Single source of truth for clamping the active-window duration.

    Floor at one second so a misconfigured zero window (which would
    expire every event the instant it's recorded) becomes a
    detectable-via-metric mistake instead of a silent no-emission.
    """
    return max(1.0, active_window.total_seconds())


__all__ = [
    "DeployEvent",
    "DeployRiskScore",
    "DeployTracker",
    "DeviationLookup",
    "SignalSpec",
    "active_window_seconds",
    "compute_deploy_risk_scores",
    "derive_deployment_id",
    "detect_deploy_from_regressor",
]
