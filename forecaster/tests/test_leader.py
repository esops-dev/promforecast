"""Tests for Kubernetes-Lease-based leader election.

Covers the elector's state machine without touching the real Kubernetes
API — every interaction goes through a small in-memory ``_FakeLeaseClient``
that satisfies the same :class:`LeaseClient` protocol.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

import pytest

from promforecast.leader import (
    LeaderElector,
    LeaderElectorConfig,
    LeaseConflictError,
    LeaseNotFoundError,
    LeaseSnapshot,
    LeaseSpec,
)


@dataclass
class _Stored:
    spec: LeaseSpec
    resource_version: int = 1


class _FakeLeaseClient:
    """In-memory LeaseClient: enough to drive the elector through every branch."""

    def __init__(self) -> None:
        self._leases: dict[tuple[str, str], _Stored] = {}
        # Allows a test to artificially fail the next ``update`` to simulate
        # a stale write losing the conflict race.
        self.fail_update_with: Exception | None = None

    async def get(self, namespace: str, name: str) -> LeaseSnapshot:
        stored = self._leases.get((namespace, name))
        if stored is None:
            raise LeaseNotFoundError(name)
        return LeaseSnapshot(spec=stored.spec, resource_version=str(stored.resource_version))

    async def create(self, namespace: str, name: str, spec: LeaseSpec) -> LeaseSnapshot:
        if (namespace, name) in self._leases:
            raise LeaseConflictError(name)
        self._leases[(namespace, name)] = _Stored(spec=spec, resource_version=1)
        return await self.get(namespace, name)

    async def update(
        self,
        namespace: str,
        name: str,
        spec: LeaseSpec,
        resource_version: str,
    ) -> LeaseSnapshot:
        if self.fail_update_with is not None:
            err = self.fail_update_with
            self.fail_update_with = None
            raise err
        stored = self._leases.get((namespace, name))
        if stored is None:
            raise LeaseNotFoundError(name)
        if str(stored.resource_version) != resource_version:
            raise LeaseConflictError(name)
        stored.spec = spec
        stored.resource_version += 1
        return await self.get(namespace, name)


def _config(
    identity: str,
    *,
    lease_duration: int = 5,
    renew_deadline: int = 3,
    retry_period: int = 1,
) -> LeaderElectorConfig:
    return LeaderElectorConfig(
        namespace="ns",
        lease_name="lease",
        identity=identity,
        lease_duration_seconds=lease_duration,
        renew_deadline_seconds=renew_deadline,
        retry_period_seconds=retry_period,
    )


@pytest.mark.asyncio
async def test_elector_acquires_unowned_lease() -> None:
    """A freshly-created lease is taken by the first replica that tries."""
    client = _FakeLeaseClient()
    acquired = asyncio.Event()

    async def _on_acq() -> None:
        acquired.set()

    elector = LeaderElector(
        _config("replica-a"),
        client,
        on_started_leading=_on_acq,
    )
    task = asyncio.create_task(elector.run())
    await asyncio.wait_for(acquired.wait(), timeout=2)
    assert elector.is_leader()
    await elector.stop()
    await task
    assert not elector.is_leader()


@pytest.mark.asyncio
async def test_elector_waits_when_lease_held_by_other() -> None:
    """A valid lease held by a different identity blocks acquisition."""
    client = _FakeLeaseClient()
    now = time.time()
    client._leases[("ns", "lease")] = _Stored(
        spec=LeaseSpec(
            holder_identity="incumbent",
            acquire_time=now,
            renew_time=now,
            lease_duration_seconds=60,
        )
    )
    elector = LeaderElector(_config("challenger"), client)
    task = asyncio.create_task(elector.run())
    await asyncio.sleep(0.3)
    assert not elector.is_leader()
    assert elector.leader_identity() == "incumbent"
    await elector.stop()
    await task


@pytest.mark.asyncio
async def test_elector_takes_over_expired_lease() -> None:
    """An expired lease can be taken over with a bumped ``lease_transitions``."""
    client = _FakeLeaseClient()
    long_ago = time.time() - 10_000
    client._leases[("ns", "lease")] = _Stored(
        spec=LeaseSpec(
            holder_identity="dead-replica",
            acquire_time=long_ago,
            renew_time=long_ago,
            lease_duration_seconds=5,
            lease_transitions=3,
        )
    )
    acquired = asyncio.Event()

    async def _on_acq() -> None:
        acquired.set()

    elector = LeaderElector(
        _config("new-replica"),
        client,
        on_started_leading=_on_acq,
    )
    task = asyncio.create_task(elector.run())
    await asyncio.wait_for(acquired.wait(), timeout=2)
    stored = client._leases[("ns", "lease")]
    assert stored.spec.holder_identity == "new-replica"
    assert stored.spec.lease_transitions == 4
    await elector.stop()
    await task


@pytest.mark.asyncio
async def test_elector_drops_leadership_when_lease_stolen() -> None:
    """A holder change observed mid-loop fires ``on_stopped_leading``."""
    client = _FakeLeaseClient()
    started = asyncio.Event()
    stopped = asyncio.Event()

    async def _on_start() -> None:
        started.set()

    async def _on_stop() -> None:
        stopped.set()

    elector = LeaderElector(
        _config("me", lease_duration=5, renew_deadline=3, retry_period=1),
        client,
        on_started_leading=_on_start,
        on_stopped_leading=_on_stop,
    )
    task = asyncio.create_task(elector.run())
    await asyncio.wait_for(started.wait(), timeout=2)
    # Steal the lease behind the elector's back — a different replica
    # forced a write while we were sleeping in the renew loop.
    stored = client._leases[("ns", "lease")]
    stored.spec = LeaseSpec(
        holder_identity="intruder",
        acquire_time=time.time(),
        renew_time=time.time(),
        lease_duration_seconds=stored.spec.lease_duration_seconds,
        lease_transitions=stored.spec.lease_transitions + 1,
    )
    stored.resource_version += 1
    await asyncio.wait_for(stopped.wait(), timeout=5)
    await elector.stop()
    await task


@pytest.mark.asyncio
async def test_elector_renews_lease_periodically() -> None:
    """The renew loop bumps ``renew_time`` while leadership is held."""
    client = _FakeLeaseClient()
    elector = LeaderElector(
        _config("me", lease_duration=5, renew_deadline=3, retry_period=1),
        client,
    )
    task = asyncio.create_task(elector.run())
    # Wait long enough for at least one renewal to land.
    await asyncio.sleep(1.3)
    stored = client._leases[("ns", "lease")]
    assert stored.spec.holder_identity == "me"
    # Initial create + at least one renew ⇒ resource_version > 1.
    assert stored.resource_version >= 2
    await elector.stop()
    await task


def test_renew_deadline_must_be_below_lease_duration() -> None:
    """Mis-ordered timings are rejected at config build time."""
    with pytest.raises(ValueError):
        LeaderElectorConfig(
            namespace="ns",
            lease_name="x",
            identity="me",
            lease_duration_seconds=10,
            renew_deadline_seconds=10,
        )


@pytest.mark.asyncio
async def test_elector_release_is_bounded_on_stop() -> None:
    """A slow API server on shutdown must not block past the release timeout.

    Regression check for the finally-clause behaviour: without a bounded
    wait, SIGTERM blocks until the API server replies (or the pod's
    terminationGracePeriodSeconds expires, killing us).
    """

    class _SlowReleaseClient(_FakeLeaseClient):
        async def update(
            self,
            namespace: str,
            name: str,
            spec: LeaseSpec,
            resource_version: str,
        ) -> LeaseSnapshot:
            # Block far longer than the test's release timeout — without the
            # wait_for around _release_if_held this would hang the test.
            await asyncio.sleep(30)
            return await super().update(namespace, name, spec, resource_version)

    client = _SlowReleaseClient()
    started = asyncio.Event()

    async def _on_acq() -> None:
        started.set()

    elector = LeaderElector(
        _config("me"),
        client,
        on_started_leading=_on_acq,
        release_timeout_seconds=0.2,
    )
    task = asyncio.create_task(elector.run())
    await asyncio.wait_for(started.wait(), timeout=2)
    t0 = asyncio.get_event_loop().time()
    await elector.stop()
    await asyncio.wait_for(task, timeout=2)
    elapsed = asyncio.get_event_loop().time() - t0
    # Release timeout 0.2s plus retry-loop drain — comfortably below the
    # 30s slow-release simulation.
    assert elapsed < 3
