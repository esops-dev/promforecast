"""Real-cluster integration test for the ``leader_assigned`` elector path.

This is the one part of sharding the in-process tests can't reach: the
``KubernetesLeaseClient`` actually talking to a ``coordination.k8s.io/v1``
Lease, a ``LeaderElector`` acquiring/renewing it, and a
``LeaderAssignedCoordinator`` whose ``is_leader`` is driven by that real
election. We assert the end-to-end invariants against a live API server:

* exactly one of two replicas becomes the assigner,
* the assigner publishes a complete + disjoint group assignment,
* the follower reads the same assignment, and
* leadership (and the assignment) fail over when the assigner leaves.

It is **skipped** unless ``PROMFORECAST_KIND_TEST`` is set, so the normal CI
``test`` job (no cluster) is unaffected. To run it locally:

    kind create cluster --name promforecast-shard-test
    PROMFORECAST_KIND_TEST=1 uv run pytest tests/test_leader_assigned_kind.py -m kind
    kind delete cluster --name promforecast-shard-test

The membership registry uses an in-memory TTL store (Redis is validated
elsewhere); the point here is the Kubernetes election wiring.
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from collections.abc import Callable

import pytest

from promforecast import sharding
from promforecast.leader import LeaderElector, LeaderElectorConfig

pytest.importorskip("kubernetes", reason="leader_assigned kind test needs the 'ha' extra")

pytestmark = [
    pytest.mark.kind,
    pytest.mark.skipif(
        not os.environ.get("PROMFORECAST_KIND_TEST"),
        reason="set PROMFORECAST_KIND_TEST=1 (kubeconfig pointing at a throwaway cluster) to run",
    ),
]

_NAMESPACE = os.environ.get("PROMFORECAST_KIND_NAMESPACE", "default")


class _TTLStore:
    """In-memory AssignmentStore that honours heartbeat TTLs.

    Mirrors :class:`RedisAssignmentStore` semantics (members expire, single
    published map) so the coordinator behaves exactly as it would against
    Redis — the cluster dependency here is only the Lease, not the store.
    """

    def __init__(self) -> None:
        self._members: dict[str, tuple[float, int]] = {}
        self._assignment: dict[str, str] = {}

    async def heartbeat(self, shard_id: str, ttl_seconds: int) -> None:
        self._members[shard_id] = (time.monotonic(), ttl_seconds)

    async def members(self) -> list[str]:
        now = time.monotonic()
        return [s for s, (ts, ttl) in self._members.items() if now - ts <= ttl]

    async def publish_assignment(self, mapping: dict[str, str]) -> None:
        self._assignment = dict(mapping)

    async def read_assignment(self) -> dict[str, str]:
        return dict(self._assignment)


async def _wait_for(predicate: Callable[[], bool], timeout: float = 25.0) -> bool:
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.25)
    return predicate()


def _delete_lease(name: str) -> None:
    """Best-effort cleanup so reruns don't collide on a leftover Lease."""
    try:
        from kubernetes import client as k8s_client  # noqa: PLC0415
        from kubernetes import config as k8s_config  # noqa: PLC0415

        try:
            k8s_config.load_incluster_config()
        except Exception:
            k8s_config.load_kube_config()
        k8s_client.CoordinationV1Api().delete_namespaced_lease(name=name, namespace=_NAMESPACE)
    except Exception:
        pass


async def _drive(coord: sharding.LeaderAssignedCoordinator, groups: list[str], rounds: int) -> None:
    for _ in range(rounds):
        await coord.refresh(groups)
        await asyncio.sleep(0.1)


async def test_leader_assigned_elects_and_assigns_against_real_lease() -> None:
    from promforecast.leader import KubernetesLeaseClient  # noqa: PLC0415

    lease_name = f"promforecast-shard-test-{uuid.uuid4().hex[:8]}"
    # Short, valid lease params for a snappy test (renew < duration).
    cfg_a = LeaderElectorConfig(
        namespace=_NAMESPACE,
        lease_name=lease_name,
        identity="shard-a",
        lease_duration_seconds=5,
        renew_deadline_seconds=3,
        retry_period_seconds=1,
    )
    cfg_b = LeaderElectorConfig(
        namespace=_NAMESPACE,
        lease_name=lease_name,
        identity="shard-b",
        lease_duration_seconds=5,
        renew_deadline_seconds=3,
        retry_period_seconds=1,
    )
    elector_a = LeaderElector(config=cfg_a, client=KubernetesLeaseClient())
    elector_b = LeaderElector(config=cfg_b, client=KubernetesLeaseClient())
    store = _TTLStore()
    coord_a = sharding.LeaderAssignedCoordinator(
        shard_id="shard-a", store=store, is_leader=elector_a.is_leader, member_ttl_seconds=3
    )
    coord_b = sharding.LeaderAssignedCoordinator(
        shard_id="shard-b", store=store, is_leader=elector_b.is_leader, member_ttl_seconds=3
    )
    groups = [f"g{i}" for i in range(24)]

    task_a = asyncio.create_task(elector_a.run())
    task_b = asyncio.create_task(elector_b.run())
    tasks: list[asyncio.Task[None]] = [task_a, task_b]
    try:
        # (1) Real Lease election: exactly one assigner.
        assert await _wait_for(lambda: elector_a.is_leader() ^ elector_b.is_leader()), (
            "no single assigner was elected against the real Lease API"
        )

        # (2)+(3) Drive both coordinators; the assigner publishes, the follower
        # reads. A few rounds let membership populate and the map propagate.
        await asyncio.gather(_drive(coord_a, groups, 4), _drive(coord_b, groups, 4))
        owned_a = coord_a.owned_groups(groups)
        owned_b = coord_b.owned_groups(groups)
        assert owned_a | owned_b == set(groups), "assignment is not complete"
        assert not (owned_a & owned_b), "a group is owned by both shards"
        assert owned_a and owned_b, "assignment is not balanced across two live shards"

        # (4) Failover: the assigner leaves. The other replica must take the
        # Lease over and, once the departed member's heartbeat TTL lapses,
        # reassign every group to itself.
        leader_is_a = elector_a.is_leader()
        (leaving, leaving_task) = (elector_a, task_a) if leader_is_a else (elector_b, task_b)
        survivor = elector_b if leader_is_a else elector_a
        survivor_coord = coord_b if leader_is_a else coord_a
        await leaving.stop()
        await leaving_task
        tasks.remove(leaving_task)

        assert await _wait_for(survivor.is_leader), "survivor did not take over the Lease"
        # Let the departed shard's membership expire (ttl=3s), then republish.
        await asyncio.sleep(3.5)
        await _drive(survivor_coord, groups, 4)
        assert survivor_coord.owned_groups(groups) == set(groups), (
            "survivor did not reabsorb all groups after the assigner left"
        )
    finally:
        for elector, task in ((elector_a, task_a), (elector_b, task_b)):
            if task in tasks:
                await elector.stop()
                from contextlib import suppress  # noqa: PLC0415

                with suppress(asyncio.CancelledError, Exception):
                    await task
        _delete_lease(lease_name)
