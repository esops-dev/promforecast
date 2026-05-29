"""Unit tests for the sharding ownership logic and coordinators.

The pure functions (``stable_bucket``, ``consistent_hash_owns``,
``assign_groups``) carry the correctness-critical invariants: ownership must be
disjoint and complete, and stable across processes. The coordinators wrap those
in the ``owns`` / ``refresh`` surface the runner consumes; ``LeaderAssigned`` is
exercised against an in-memory store so no Redis is needed.
"""

from __future__ import annotations

import pytest

from promforecast import sharding
from promforecast.exporter import Exporter


def test_stable_bucket_is_deterministic() -> None:
    # Must NOT depend on PYTHONHASHSEED — two processes have to agree.
    assert sharding.stable_bucket("node_capacity") == sharding.stable_bucket("node_capacity")
    assert sharding.stable_bucket("a") != sharding.stable_bucket("b")


@pytest.mark.parametrize("replicas", [1, 2, 3, 4, 7])
def test_consistent_hash_ownership_is_disjoint_and_complete(replicas: int) -> None:
    groups = [f"group_{i}" for i in range(50)]
    owners_per_group: dict[str, list[int]] = {g: [] for g in groups}
    for shard in range(replicas):
        for g in groups:
            if sharding.consistent_hash_owns(g, shard, replicas):
                owners_per_group[g].append(shard)
    # Every group owned by exactly one shard.
    assert all(len(owners) == 1 for owners in owners_per_group.values())


def test_consistent_hash_owns_rejects_out_of_range_index() -> None:
    assert sharding.consistent_hash_owns("g", shard_index=5, replica_count=3) is False
    assert sharding.consistent_hash_owns("g", shard_index=-1, replica_count=3) is False


def test_consistent_hash_owns_clamps_zero_replica_count() -> None:
    # A misconfigured count of 0 must not divide by zero; shard 0 owns all.
    assert sharding.consistent_hash_owns("g", shard_index=0, replica_count=0) is True


def test_assign_groups_disjoint_complete_and_balanced() -> None:
    groups = [f"g{i}" for i in range(60)]
    members = ["shard-a", "shard-b", "shard-c"]
    mapping = sharding.assign_groups(groups, members)
    assert set(mapping) == set(groups)
    assert set(mapping.values()) <= set(members)
    # Rendezvous hashing is balanced in expectation — no shard starves with
    # 60 groups over 3 members.
    counts = {m: sum(1 for v in mapping.values() if v == m) for m in members}
    assert all(c > 0 for c in counts.values())


def test_assign_groups_empty_without_members() -> None:
    assert sharding.assign_groups(["g1", "g2"], []) == {}


def test_assign_groups_minimal_churn_on_member_join() -> None:
    # Rendezvous hashing's defining property: adding a member moves only the
    # groups that land on the newcomer, never reshuffles the rest.
    groups = [f"g{i}" for i in range(100)]
    before = sharding.assign_groups(groups, ["a", "b", "c"])
    after = sharding.assign_groups(groups, ["a", "b", "c", "d"])
    moved = [g for g in groups if before[g] != after[g]]
    # Everything that moved must have moved TO the new member.
    assert all(after[g] == "d" for g in moved)
    # And the churn is a minority of groups.
    assert len(moved) < len(groups) // 2


def test_derive_shard_index_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PROMFORECAST_SHARD_INDEX", raising=False)
    monkeypatch.delenv("PROMFORECAST_POD_NAME", raising=False)
    # Explicit config value wins.
    assert sharding.derive_shard_index(2) == 2
    # Then the env override.
    monkeypatch.setenv("PROMFORECAST_SHARD_INDEX", "5")
    assert sharding.derive_shard_index(-1) == 5
    monkeypatch.delenv("PROMFORECAST_SHARD_INDEX")
    # Then the StatefulSet pod ordinal.
    monkeypatch.setenv("PROMFORECAST_POD_NAME", "promforecast-3")
    assert sharding.derive_shard_index(-1) == 3
    monkeypatch.delenv("PROMFORECAST_POD_NAME")
    # Nothing resolvable → -1 (caller fails loudly).
    assert sharding.derive_shard_index(-1) == -1


def test_derive_shard_id_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PROMFORECAST_POD_NAME", raising=False)
    assert sharding.derive_shard_id("custom", 0) == "custom"
    monkeypatch.setenv("PROMFORECAST_POD_NAME", "promforecast-2")
    assert sharding.derive_shard_id("", 2) == "promforecast-2"
    monkeypatch.delenv("PROMFORECAST_POD_NAME")
    assert sharding.derive_shard_id("", 2) == "shard-2"


def test_consistent_hash_coordinator_owns_and_owned_groups() -> None:
    groups = [f"g{i}" for i in range(20)]
    coords = [
        sharding.ConsistentHashCoordinator(shard_index=i, replica_count=3, shard_id=f"s{i}")
        for i in range(3)
    ]
    assert not any(c.needs_periodic_refresh() for c in coords)
    # Union of owned groups == all groups; pairwise disjoint.
    owned_sets = [c.owned_groups(groups) for c in coords]
    union: set[str] = set().union(*owned_sets)
    assert union == set(groups)
    assert sum(len(s) for s in owned_sets) == len(groups)


class _FakeStore:
    """In-memory :class:`AssignmentStore` for coordinator tests."""

    def __init__(self) -> None:
        self.members_set: set[str] = set()
        self.assignment: dict[str, str] = {}

    async def heartbeat(self, shard_id: str, ttl_seconds: int) -> None:
        self.members_set.add(shard_id)

    async def members(self) -> list[str]:
        return sorted(self.members_set)

    async def publish_assignment(self, mapping: dict[str, str]) -> None:
        self.assignment = dict(mapping)

    async def read_assignment(self) -> dict[str, str]:
        return dict(self.assignment)


@pytest.mark.asyncio
async def test_leader_assigned_coordinator_leader_publishes() -> None:
    store = _FakeStore()
    # Pretend two peers already heartbeated.
    store.members_set.update({"shard-b", "shard-c"})
    coord = sharding.LeaderAssignedCoordinator(
        shard_id="shard-a",
        store=store,
        is_leader=lambda: True,
        member_ttl_seconds=30,
    )
    groups = [f"g{i}" for i in range(30)]
    await coord.refresh(groups)
    # The leader published a complete assignment over the live membership.
    assert set(store.assignment) == set(groups)
    assert set(store.assignment.values()) <= {"shard-a", "shard-b", "shard-c"}
    # Its own owned set is the subset assigned to it.
    owned = coord.owned_groups(groups)
    assert owned == {g for g, s in store.assignment.items() if s == "shard-a"}


@pytest.mark.asyncio
async def test_leader_assigned_follower_reads_published_map() -> None:
    store = _FakeStore()
    store.assignment = {"g1": "shard-a", "g2": "shard-b"}
    coord = sharding.LeaderAssignedCoordinator(
        shard_id="shard-b",
        store=store,
        is_leader=lambda: False,
        member_ttl_seconds=30,
    )
    await coord.refresh(["g1", "g2"])
    assert coord.owns("g2") is True
    assert coord.owns("g1") is False


@pytest.mark.asyncio
async def test_leader_assigned_owns_nothing_before_assignment_published() -> None:
    # Cold start: no published assignment → own nothing and wait for the
    # assigner, rather than every replica fitting every group (the N-times herd).
    store = _FakeStore()
    coord_a = sharding.LeaderAssignedCoordinator(
        shard_id="a", store=store, is_leader=lambda: False, member_ttl_seconds=30
    )
    assert coord_a.owns("anything") is False


@pytest.mark.asyncio
async def test_leader_assigned_covers_group_missing_from_published_map() -> None:
    # A reload added "new_group" before the assigner republished: the mapping
    # exists but omits it. Every replica must agree (via rendezvous) on exactly
    # one owner so the group is covered, not orphaned and not duplicated.
    store = _FakeStore()
    members = ["a", "b", "c"]
    coords = {
        m: sharding.LeaderAssignedCoordinator(
            shard_id=m, store=store, is_leader=lambda: False, member_ttl_seconds=30
        )
        for m in members
    }
    # Simulate the published-but-stale state each replica has cached.
    for c in coords.values():
        c._assignment = {"g1": "a", "g2": "b"}
        c._members = list(members)
    owners = [m for m, c in coords.items() if c.owns("new_group")]
    assert len(owners) == 1  # exactly one replica covers the un-mapped group


class _FakeRedis:
    """Minimal async Redis stand-in for RedisAssignmentStore.

    Implements ``scan`` (cursor-based, paged) rather than ``keys`` so the test
    exercises the same non-blocking enumeration the production store uses.
    """

    def __init__(self) -> None:
        self.store: dict[str, bytes] = {}

    async def set(self, key: str, value: object, ex: int | None = None) -> None:
        self.store[key] = value if isinstance(value, bytes) else str(value).encode()

    async def get(self, key: str) -> bytes | None:
        return self.store.get(key)

    async def scan(
        self, cursor: int = 0, match: str = "*", count: int = 100
    ) -> tuple[int, list[bytes]]:
        prefix = match.rstrip("*")
        matched = [k.encode() for k in sorted(self.store) if k.startswith(prefix)]
        # Page the results to prove the store's cursor loop terminates.
        page = matched[cursor : cursor + count]
        next_cursor = cursor + count
        if next_cursor >= len(matched):
            next_cursor = 0
        return next_cursor, page


@pytest.mark.asyncio
async def test_redis_assignment_store_roundtrip() -> None:
    client = _FakeRedis()
    store = sharding.RedisAssignmentStore(client)
    await store.heartbeat("shard-a", ttl_seconds=30)
    await store.heartbeat("shard-b", ttl_seconds=30)
    assert sorted(await store.members()) == ["shard-a", "shard-b"]
    await store.publish_assignment({"g1": "shard-a"})
    assert await store.read_assignment() == {"g1": "shard-a"}


@pytest.mark.asyncio
async def test_redis_assignment_store_empty_assignment() -> None:
    store = sharding.RedisAssignmentStore(_FakeRedis())
    assert await store.read_assignment() == {}


@pytest.mark.asyncio
async def test_redis_assignment_store_members_scan_pages() -> None:
    # More members than one SCAN page (count=100) → the cursor loop must walk
    # every page and terminate. Guards against a regression to a single KEYS
    # call or a cursor loop that never advances.
    client = _FakeRedis()
    store = sharding.RedisAssignmentStore(client)
    for i in range(250):
        await store.heartbeat(f"shard-{i:03d}", ttl_seconds=30)
    members = await store.members()
    assert len(members) == 250
    assert "shard-000" in members and "shard-249" in members


def test_exporter_shard_owns_group_gauge() -> None:
    exporter = Exporter()
    # Absent until set.
    assert "forecast_shard_owns_group" not in exporter.render().decode()
    exporter.set_shard_state(shard_id="shard-1", owned_groups={"node_cap", "kube_res"})
    body = exporter.render().decode()
    assert 'forecast_shard_owns_group{group="node_cap",shard_id="shard-1"} 1.0' in body
    assert 'forecast_shard_owns_group{group="kube_res",shard_id="shard-1"} 1.0' in body


def test_exporter_shard_state_empty_owned_renders_no_samples() -> None:
    exporter = Exporter()
    exporter.set_shard_state(shard_id="shard-1", owned_groups=set())
    # Shard id set but nothing owned → the family may appear with no samples;
    # crucially no spurious empty-group series.
    body = exporter.render().decode()
    assert 'group=""' not in body
