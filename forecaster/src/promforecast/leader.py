"""Kubernetes Lease-based leader election for HA mode.

Only one replica fits models and writes snapshots at a time. The elected
leader runs the scheduler; followers serve ``/metrics`` from the shared
snapshot cache (see :mod:`promforecast.ha_cache`). Election uses the
``coordination.k8s.io/v1`` Lease API — the same primitive ``kube-scheduler``
and ``cloud-controller-manager`` use, and that the ``kubernetes`` Python
client speaks natively.

Design choices:

* **Sync API in a thread**. The official ``kubernetes`` client is sync-only.
  We dispatch every Lease call through ``loop.run_in_executor`` so the
  asyncio loop is never blocked. The (small, lease_duration-second) cost
  of a thread per renewal is acceptable for one-leader-per-process.
* **Lease params follow client-go defaults**: ``lease_duration=30s``,
  ``renew_deadline=20s``, ``retry_period=4s``. These are battle-tested for
  the Kubernetes control plane and work well for slowly-changing leaders;
  shorten if your environment's API server is unusually fast and you want
  more aggressive failover.
* **Best-effort release on shutdown**. We try to clear the lease's
  ``holderIdentity`` so a graceful pod stop doesn't leave the lease pinned
  to a dead replica for the full duration. A failure here is logged and
  swallowed — followers will pick it up after one lease_duration anyway.
* **Pluggable client** via :class:`LeaseClient`. Tests use a fake that
  never hits the network; production wires in the kubernetes-client-backed
  :class:`KubernetesLeaseClient`.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import structlog

logger = structlog.get_logger(__name__)

# Kubernetes API status codes we map to typed exceptions so the caller can
# branch on them without scraping error messages.
_HTTP_NOT_FOUND = 404
_HTTP_CONFLICT = 409


@dataclass
class LeaseSpec:
    """Subset of the Lease ``spec`` we care about, normalised across backends."""

    holder_identity: str | None
    acquire_time: float | None  # Unix epoch seconds (may be ``None`` on fresh leases)
    renew_time: float | None
    lease_duration_seconds: int
    lease_transitions: int = 0


@dataclass
class LeaseSnapshot:
    """A Lease fetched from the API server — spec plus the resource version.

    The resource version round-trips through update calls so a stale write
    (another replica updated the same lease behind our back) is rejected by
    the API server with a 409 Conflict. We surface that as
    :class:`LeaseConflictError` so the elector can re-read instead of
    overwriting a fresh holder.
    """

    spec: LeaseSpec
    resource_version: str = ""


class LeaseConflictError(RuntimeError):
    """Lease update raced another writer — caller should re-read and retry."""


class LeaseNotFoundError(RuntimeError):
    """Lease doesn't exist yet; caller should create it."""


class LeaseClient(Protocol):
    """Backend contract — both Kubernetes and in-memory fakes implement this."""

    async def get(self, namespace: str, name: str) -> LeaseSnapshot: ...

    async def create(self, namespace: str, name: str, spec: LeaseSpec) -> LeaseSnapshot: ...

    async def update(
        self,
        namespace: str,
        name: str,
        spec: LeaseSpec,
        resource_version: str,
    ) -> LeaseSnapshot: ...


class KubernetesLeaseClient:
    """LeaseClient backed by the official ``kubernetes`` Python client.

    Constructed lazily so importing this module on a machine without
    in-cluster credentials (e.g. CI lint runs) doesn't crash; we only touch
    the kubernetes config when ``connect()`` is called.

    ``request_timeout_seconds`` caps each underlying HTTP call. Without
    it, a hung API server holds the elector's worker thread forever and
    delays SIGTERM by the full Pod terminationGracePeriodSeconds. Default
    is comfortably below the typical 30s grace period.
    """

    def __init__(self, request_timeout_seconds: float = 10.0) -> None:
        self._api: Any = None
        self._request_timeout = request_timeout_seconds

    async def connect(self) -> None:
        """Load in-cluster or kubeconfig credentials and build the API client.

        Idempotent: a second call is a no-op. We try in-cluster first
        (the production path) and fall back to kubeconfig for local dev.
        """
        if self._api is not None:
            return
        # Imports are local so the module loads cleanly when the ``ha`` extra
        # is not installed — leader election only requires kubernetes at HA
        # runtime, not at import time.
        try:
            from kubernetes import client as k8s_client  # noqa: PLC0415
            from kubernetes import config as k8s_config  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "highAvailability requires the 'kubernetes' package; "
                "install with `pip install 'promforecast[ha]'`"
            ) from exc
        try:
            k8s_config.load_incluster_config()
        except Exception:
            # Local dev: fall back to ~/.kube/config. Re-raise the underlying
            # error untouched if both paths fail so operators see the cause.
            k8s_config.load_kube_config()
        self._api = k8s_client.CoordinationV1Api()

    async def get(self, namespace: str, name: str) -> LeaseSnapshot:
        await self.connect()
        return await asyncio.to_thread(self._get_sync, namespace, name)

    async def create(self, namespace: str, name: str, spec: LeaseSpec) -> LeaseSnapshot:
        await self.connect()
        return await asyncio.to_thread(self._create_sync, namespace, name, spec)

    async def update(
        self,
        namespace: str,
        name: str,
        spec: LeaseSpec,
        resource_version: str,
    ) -> LeaseSnapshot:
        await self.connect()
        return await asyncio.to_thread(self._update_sync, namespace, name, spec, resource_version)

    def _get_sync(self, namespace: str, name: str) -> LeaseSnapshot:
        from kubernetes.client.exceptions import ApiException  # noqa: PLC0415

        try:
            lease = self._api.read_namespaced_lease(
                name=name,
                namespace=namespace,
                _request_timeout=self._request_timeout,
            )
        except ApiException as exc:
            if exc.status == _HTTP_NOT_FOUND:
                raise LeaseNotFoundError(name) from exc
            raise
        return _snapshot_from_k8s_lease(lease)

    def _create_sync(self, namespace: str, name: str, spec: LeaseSpec) -> LeaseSnapshot:
        from kubernetes import client as k8s_client  # noqa: PLC0415
        from kubernetes.client.exceptions import ApiException  # noqa: PLC0415

        body = k8s_client.V1Lease(
            metadata=k8s_client.V1ObjectMeta(name=name, namespace=namespace),
            spec=_lease_spec_to_k8s(spec),
        )
        try:
            lease = self._api.create_namespaced_lease(
                namespace=namespace,
                body=body,
                _request_timeout=self._request_timeout,
            )
        except ApiException as exc:
            if exc.status == _HTTP_CONFLICT:
                raise LeaseConflictError(name) from exc
            raise
        return _snapshot_from_k8s_lease(lease)

    def _update_sync(
        self,
        namespace: str,
        name: str,
        spec: LeaseSpec,
        resource_version: str,
    ) -> LeaseSnapshot:
        from kubernetes import client as k8s_client  # noqa: PLC0415
        from kubernetes.client.exceptions import ApiException  # noqa: PLC0415

        meta = k8s_client.V1ObjectMeta(
            name=name,
            namespace=namespace,
            resource_version=resource_version or None,
        )
        body = k8s_client.V1Lease(metadata=meta, spec=_lease_spec_to_k8s(spec))
        try:
            lease = self._api.replace_namespaced_lease(
                name=name,
                namespace=namespace,
                body=body,
                _request_timeout=self._request_timeout,
            )
        except ApiException as exc:
            if exc.status == _HTTP_CONFLICT:
                raise LeaseConflictError(name) from exc
            raise
        return _snapshot_from_k8s_lease(lease)


def _lease_spec_to_k8s(spec: LeaseSpec) -> Any:
    from datetime import UTC, datetime  # noqa: PLC0415

    from kubernetes import client as k8s_client  # noqa: PLC0415

    def _to_dt(epoch: float | None) -> datetime | None:
        if epoch is None:
            return None
        return datetime.fromtimestamp(epoch, tz=UTC)

    return k8s_client.V1LeaseSpec(
        holder_identity=spec.holder_identity,
        lease_duration_seconds=spec.lease_duration_seconds,
        acquire_time=_to_dt(spec.acquire_time),
        renew_time=_to_dt(spec.renew_time),
        lease_transitions=spec.lease_transitions,
    )


def _snapshot_from_k8s_lease(lease: Any) -> LeaseSnapshot:
    s = lease.spec
    return LeaseSnapshot(
        spec=LeaseSpec(
            holder_identity=getattr(s, "holder_identity", None),
            acquire_time=_dt_to_epoch(getattr(s, "acquire_time", None)),
            renew_time=_dt_to_epoch(getattr(s, "renew_time", None)),
            lease_duration_seconds=int(getattr(s, "lease_duration_seconds", 0) or 0),
            lease_transitions=int(getattr(s, "lease_transitions", 0) or 0),
        ),
        resource_version=getattr(lease.metadata, "resource_version", "") or "",
    )


def _dt_to_epoch(value: object) -> float | None:
    if value is None:
        return None
    ts = getattr(value, "timestamp", None)
    if ts is None:
        return None
    try:
        return float(ts())
    except Exception:
        return None


@dataclass
class LeaderElectorConfig:
    namespace: str
    lease_name: str
    identity: str
    lease_duration_seconds: int = 30
    renew_deadline_seconds: int = 20
    retry_period_seconds: int = 4

    def __post_init__(self) -> None:
        # client-go's invariant: renew_deadline < lease_duration. Without
        # this, a slow API server can race a follower acquiring the lease
        # while we're still trying to renew it.
        if self.renew_deadline_seconds >= self.lease_duration_seconds:
            raise ValueError(
                "renew_deadline_seconds must be < lease_duration_seconds "
                f"(got {self.renew_deadline_seconds} >= {self.lease_duration_seconds})"
            )
        if self.retry_period_seconds <= 0:
            raise ValueError("retry_period_seconds must be positive")


Callback = Callable[[], Awaitable[None]]


@dataclass
class _State:
    """Internal mutable state — pulled out so :class:`LeaderElector` stays small."""

    is_leader: bool = False
    leader_identity: str | None = None
    last_observed_renew: float = 0.0
    transitions: int = 0
    # Identities of replicas we logged as "observed-as-leader" so we don't
    # spam logs on every retry tick.
    _logged: set[str] = field(default_factory=set)


class LeaderElector:
    """Owns the acquire/renew loop and surfaces leadership state to the runner.

    Lifecycle:

    1. ``await elector.run()`` — long-running coroutine; either acquires the
       lease, holds it via periodic renews, or waits as a follower while
       another replica holds it.
    2. ``elector.is_leader()`` — synchronous snapshot of leadership state.
    3. ``await elector.stop()`` — sets the stop event; the run loop exits
       after a best-effort release of the lease.

    Callbacks fire **inside** the run loop, so they can ``await`` runner
    methods without re-entrancy worries. The runner's start/stop guards
    against double-start.
    """

    def __init__(
        self,
        config: LeaderElectorConfig,
        client: LeaseClient,
        *,
        on_started_leading: Callback | None = None,
        on_stopped_leading: Callback | None = None,
        on_observed_leader: Callable[[str], None] | None = None,
        release_timeout_seconds: float = 2.0,
    ) -> None:
        self._config = config
        self._client = client
        self._on_started = on_started_leading
        self._on_stopped = on_stopped_leading
        self._on_observed = on_observed_leader
        # Bound the graceful release on stop. The lease will expire on its
        # own if we miss the release window, so we'd rather give up than
        # block SIGTERM.
        self._release_timeout_seconds = release_timeout_seconds
        self._stop = asyncio.Event()
        self._state = _State()

    def is_leader(self) -> bool:
        return self._state.is_leader

    def leader_identity(self) -> str | None:
        return self._state.leader_identity

    async def run(self) -> None:
        log = logger.bind(
            namespace=self._config.namespace,
            lease=self._config.lease_name,
            identity=self._config.identity,
        )
        log.info("leader_elector_started")
        try:
            while not self._stop.is_set():
                acquired = await self._try_acquire(log)
                if acquired:
                    await self._lead_until_lost(log)
                    if self._state.is_leader:
                        await self._transition_to_follower(log)
                await self._sleep_or_stop(self._config.retry_period_seconds)
        finally:
            # Best-effort release: clear the holder if it's still us so a
            # follower can take over within one retry_period rather than
            # one lease_duration. Bounded so a hung API server can't eat
            # the full Pod terminationGracePeriodSeconds (default 30s in
            # Kubernetes); the lease will simply expire on its own if we
            # can't reach the API.
            if self._state.is_leader:
                await self._transition_to_follower(log)
            with contextlib.suppress(Exception):
                await asyncio.wait_for(
                    self._release_if_held(log),
                    timeout=self._release_timeout_seconds,
                )
            log.info("leader_elector_stopped")

    async def stop(self) -> None:
        self._stop.set()

    async def _try_acquire(self, log: structlog.stdlib.BoundLogger) -> bool:
        """Single acquire attempt. Returns True iff we own the lease afterwards."""
        try:
            snapshot = await self._client.get(self._config.namespace, self._config.lease_name)
        except LeaseNotFoundError:
            return await self._try_create(log)
        return await self._maybe_takeover(log, snapshot)

    async def _try_create(self, log: structlog.stdlib.BoundLogger) -> bool:
        now = time.time()
        spec = LeaseSpec(
            holder_identity=self._config.identity,
            acquire_time=now,
            renew_time=now,
            lease_duration_seconds=self._config.lease_duration_seconds,
            lease_transitions=0,
        )
        try:
            await self._client.create(self._config.namespace, self._config.lease_name, spec)
        except LeaseConflictError:
            # Another replica won the create race; we'll re-observe on the
            # next tick and either take over or wait.
            return False
        except Exception as exc:
            log.warning("leader_lease_create_failed", error=str(exc))
            return False
        await self._transition_to_leader(log)
        return True

    async def _maybe_takeover(
        self,
        log: structlog.stdlib.BoundLogger,
        snapshot: LeaseSnapshot,
    ) -> bool:
        """Take the lease if it's expired or already ours; observe otherwise."""
        spec = snapshot.spec
        holder = spec.holder_identity
        now = time.time()
        if holder == self._config.identity:
            # We're still the recorded leader (e.g. after a restart inside
            # the renew deadline). Renew the timestamp and continue leading.
            return await self._renew(log, snapshot)
        if self._is_expired(spec, now):
            return await self._takeover(log, snapshot)
        # Lease is held by someone else and still valid — log once per holder.
        if holder is not None and holder not in self._state._logged:
            self._state._logged.add(holder)
            log.info("leader_observed", holder=holder)
        self._state.leader_identity = holder
        if self._on_observed is not None and holder is not None:
            self._on_observed(holder)
        self._state.last_observed_renew = spec.renew_time or 0.0
        return False

    def _is_expired(self, spec: LeaseSpec, now: float) -> bool:
        renew = spec.renew_time
        if renew is None:
            return True
        return now - renew > spec.lease_duration_seconds

    async def _takeover(
        self,
        log: structlog.stdlib.BoundLogger,
        snapshot: LeaseSnapshot,
    ) -> bool:
        now = time.time()
        new_spec = LeaseSpec(
            holder_identity=self._config.identity,
            acquire_time=now,
            renew_time=now,
            lease_duration_seconds=self._config.lease_duration_seconds,
            lease_transitions=snapshot.spec.lease_transitions + 1,
        )
        try:
            await self._client.update(
                self._config.namespace,
                self._config.lease_name,
                new_spec,
                snapshot.resource_version,
            )
        except LeaseConflictError:
            return False
        except Exception as exc:
            log.warning("leader_lease_takeover_failed", error=str(exc))
            return False
        log.info(
            "leader_taken_over",
            previous_holder=snapshot.spec.holder_identity,
            transitions=new_spec.lease_transitions,
        )
        await self._transition_to_leader(log)
        return True

    async def _renew(
        self,
        log: structlog.stdlib.BoundLogger,
        snapshot: LeaseSnapshot,
    ) -> bool:
        now = time.time()
        renewed = LeaseSpec(
            holder_identity=self._config.identity,
            acquire_time=snapshot.spec.acquire_time or now,
            renew_time=now,
            lease_duration_seconds=self._config.lease_duration_seconds,
            lease_transitions=snapshot.spec.lease_transitions,
        )
        try:
            await self._client.update(
                self._config.namespace,
                self._config.lease_name,
                renewed,
                snapshot.resource_version,
            )
        except LeaseConflictError:
            return False
        except Exception as exc:
            log.warning("leader_lease_renew_failed", error=str(exc))
            return False
        if not self._state.is_leader:
            await self._transition_to_leader(log)
        return True

    async def _lead_until_lost(self, log: structlog.stdlib.BoundLogger) -> None:
        """Tight renew loop: hold the lease until ``stop()`` or a renew failure.

        We re-read the lease on every tick rather than blindly updating —
        if another replica somehow stole the lease (clock skew, manual
        kubectl edit) we want to detect it before the take-over window
        closes.
        """
        deadline_seconds = self._config.renew_deadline_seconds
        retry = self._config.retry_period_seconds
        last_success = time.time()
        while not self._stop.is_set() and self._state.is_leader:
            await self._sleep_or_stop(retry)
            if self._stop.is_set():
                break
            try:
                snapshot = await self._client.get(self._config.namespace, self._config.lease_name)
            except Exception as exc:
                log.warning("leader_lease_read_failed", error=str(exc))
                if time.time() - last_success > deadline_seconds:
                    log.warning("leader_renew_deadline_exceeded_on_read")
                    return
                continue
            if snapshot.spec.holder_identity != self._config.identity:
                log.info(
                    "leader_lease_stolen",
                    new_holder=snapshot.spec.holder_identity,
                )
                return
            renewed = await self._renew(log, snapshot)
            if renewed:
                last_success = time.time()
            elif time.time() - last_success > deadline_seconds:
                log.warning("leader_renew_deadline_exceeded")
                return

    async def _transition_to_leader(self, log: structlog.stdlib.BoundLogger) -> None:
        if self._state.is_leader:
            return
        self._state.is_leader = True
        self._state.leader_identity = self._config.identity
        self._state.transitions += 1
        self._state._logged.clear()
        log.info("leader_acquired", transitions=self._state.transitions)
        if self._on_started is not None:
            try:
                await self._on_started()
            except Exception:
                log.exception("leader_started_callback_failed")

    async def _transition_to_follower(self, log: structlog.stdlib.BoundLogger) -> None:
        if not self._state.is_leader:
            return
        self._state.is_leader = False
        self._state.leader_identity = None
        log.info("leader_lost")
        if self._on_stopped is not None:
            try:
                await self._on_stopped()
            except Exception:
                log.exception("leader_stopped_callback_failed")

    async def _release_if_held(self, log: structlog.stdlib.BoundLogger) -> None:
        """Clear the holder field so a follower can take over quickly.

        Idempotent: if someone has already updated the lease we treat it as
        already-released and exit silently.
        """
        try:
            snapshot = await self._client.get(self._config.namespace, self._config.lease_name)
        except Exception:
            return
        if snapshot.spec.holder_identity != self._config.identity:
            return
        cleared = LeaseSpec(
            holder_identity=None,
            acquire_time=snapshot.spec.acquire_time,
            renew_time=snapshot.spec.renew_time,
            lease_duration_seconds=snapshot.spec.lease_duration_seconds,
            lease_transitions=snapshot.spec.lease_transitions,
        )
        try:
            await self._client.update(
                self._config.namespace,
                self._config.lease_name,
                cleared,
                snapshot.resource_version,
            )
            log.info("leader_lease_released")
        except Exception as exc:
            log.info("leader_lease_release_failed", error=str(exc))

    async def _sleep_or_stop(self, seconds: float) -> None:
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=seconds)
        except TimeoutError:
            return


def default_identity() -> str:
    """Pod-name-or-random identity used when callers don't supply one.

    Operators usually pass the pod name from the downward API
    (``PROMFORECAST_POD_NAME``); this fallback keeps tests and ad-hoc local
    runs functional without manual identity wiring.
    """
    import os  # noqa: PLC0415

    pod = os.environ.get("PROMFORECAST_POD_NAME")
    if pod:
        return pod
    return f"promforecast-{uuid.uuid4().hex[:12]}"


__all__ = [
    "Callback",
    "KubernetesLeaseClient",
    "LeaderElector",
    "LeaderElectorConfig",
    "LeaseClient",
    "LeaseConflictError",
    "LeaseNotFoundError",
    "LeaseSnapshot",
    "LeaseSpec",
    "default_identity",
]
