"""Horizontal sharding: N active replicas each owning a disjoint slice of groups.

Where :mod:`promforecast.leader` gives a *single* active replica (the leader)
plus hot followers serving a cached snapshot, sharding makes *every* replica an
active fitter that owns a subset of the configured groups. Group ownership is
disjoint (no group is fit twice) and complete (every group is owned by exactly
one live replica), so the aggregate fit budget scales with the replica count.

Two ownership modes, both reducible to a pure ``owns(group) -> bool`` decision:

* **consistent_hash** — fully local. Each replica knows its own 0-based shard
  index and the total replica count; it owns a group iff
  ``stable_bucket(group) % replica_count == shard_index``. No coordination,
  no shared state — the only agreement needed is the replica count, which the
  chart pins by setting both the StatefulSet ``replicas`` and the config's
  ``replica_count`` from one value. The pod ordinal (StatefulSet gives stable
  ``<name>-<n>`` names) is the shard index.

* **leader_assigned** — coordinated. A ``coordination.k8s.io/v1`` Lease elects
  one replica as the *assigner*. Every replica heartbeats its identity into a
  shared Redis membership registry; the assigner reads the live membership,
  computes a balanced group→shard mapping, and publishes it back to Redis.
  Every replica (assigner included) reads the published mapping to learn what
  it owns. Membership changes (a replica joins or leaves) cause the assigner
  to republish on its next tick, so ownership rebalances automatically. The
  mapping uses rendezvous (highest-random-weight) hashing so a membership
  change moves only ~1/N of the groups rather than reshuffling everything.

The ownership decision is deliberately split from the transport: the pure
functions and the in-memory coordinators carry no I/O, and the Redis-backed
:class:`RedisAssignmentStore` is the only piece that touches the network. Tests
exercise the logic against an in-memory store.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable, Iterable
from typing import Protocol, runtime_checkable

import structlog

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Pure ownership logic
# ---------------------------------------------------------------------------


def stable_bucket(name: str) -> int:
    """Deterministic non-negative hash of ``name``, stable across processes.

    Python's built-in ``hash()`` is salted per-process (``PYTHONHASHSEED``),
    so two replicas would disagree on ownership. We hash with SHA-1 and take
    the leading 8 bytes — overkill cryptographically, but it gives every
    replica (and every test run) the same integer for the same string.
    """
    # SHA-1 here is a fast, stable bucketing hash — not a security primitive.
    digest = hashlib.sha1(name.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def consistent_hash_owns(group_name: str, shard_index: int, replica_count: int) -> bool:
    """True iff ``shard_index`` owns ``group_name`` under modulo hashing.

    ``replica_count`` is clamped to at least 1 so a misconfigured zero can't
    divide by zero; a single shard then owns everything, which is the safe
    degenerate behaviour (identical to sharding disabled).
    """
    count = max(1, replica_count)
    if shard_index < 0 or shard_index >= count:
        return False
    return stable_bucket(group_name) % count == shard_index


def consistent_hash_assignment(group_names: Iterable[str], replica_count: int) -> dict[str, int]:
    """Full ``group -> shard_index`` map under modulo hashing (for debugging)."""
    count = max(1, replica_count)
    return {name: stable_bucket(name) % count for name in group_names}


def assign_groups(group_names: Iterable[str], members: Iterable[str]) -> dict[str, str]:
    """Balanced ``group -> shard_id`` assignment via rendezvous hashing.

    For each group, the owner is the member maximising ``stable_bucket`` of the
    ``group\\x00member`` pair. Rendezvous (HRW) hashing is balanced in
    expectation and — crucially for ``leader_assigned`` rebalancing — moves
    only the groups whose previous owner left (or ~1/N of groups when a member
    joins) rather than reshuffling the whole map the way ``index % count``
    round-robin would. Ties (astronomically unlikely with a 64-bit space) break
    on the lexically smaller member id for determinism.

    Returns an empty map when there are no members — callers treat "no
    assignment yet" as "own nothing" and wait for the next assigner tick.
    """
    member_list = sorted(set(members))
    if not member_list:
        return {}
    mapping: dict[str, str] = {}
    for group in group_names:
        best_member = member_list[0]
        best_weight = stable_bucket(f"{group}\x00{best_member}")
        for member in member_list[1:]:
            weight = stable_bucket(f"{group}\x00{member}")
            if weight > best_weight or (weight == best_weight and member < best_member):
                best_member, best_weight = member, weight
        mapping[group] = best_member
    return mapping


def derive_shard_index(configured: int) -> int:
    """Resolve this replica's shard index, deriving from the environment.

    Precedence: an explicit non-negative ``configured`` value wins (local/test
    runs); otherwise ``PROMFORECAST_SHARD_INDEX``; otherwise the trailing
    ordinal of ``PROMFORECAST_POD_NAME`` (``promforecast-3`` → ``3``), which is
    what a StatefulSet provides. Returns ``-1`` when none resolve so the caller
    can fail loudly rather than silently owning nothing.
    """
    if configured >= 0:
        return configured
    env_index = os.environ.get("PROMFORECAST_SHARD_INDEX")
    if env_index is not None and env_index.strip().lstrip("-").isdigit():
        return int(env_index)
    pod = os.environ.get("PROMFORECAST_POD_NAME", "")
    _, _, suffix = pod.rpartition("-")
    if suffix.isdigit():
        return int(suffix)
    return -1


def derive_shard_id(configured: str, shard_index: int) -> str:
    """Resolve the ``shard_id`` label value.

    Precedence: explicit ``configured`` value, then ``PROMFORECAST_POD_NAME``
    (the natural per-replica identity), then ``shard-<index>``.
    """
    if configured:
        return configured
    pod = os.environ.get("PROMFORECAST_POD_NAME", "")
    if pod:
        return pod
    return f"shard-{shard_index}"


# ---------------------------------------------------------------------------
# Coordinators
# ---------------------------------------------------------------------------


@runtime_checkable
class ShardCoordinator(Protocol):
    """What the runner needs to decide and observe group ownership.

    ``owns`` is synchronous and hot-path cheap (called whenever a group loop
    is (re)spawned). ``refresh`` is async and does any I/O (heartbeat, read or
    publish the assignment); it is a no-op for fully-local modes. The runner
    calls ``refresh`` periodically only when ``needs_periodic_refresh`` is true.
    """

    @property
    def shard_id(self) -> str: ...

    def owns(self, group_name: str) -> bool: ...

    def owned_groups(self, group_names: Iterable[str]) -> set[str]: ...

    def needs_periodic_refresh(self) -> bool: ...

    async def refresh(self, group_names: Iterable[str]) -> None: ...

    async def close(self) -> None: ...


class ConsistentHashCoordinator:
    """Local, coordination-free ownership by ``hash(group) % replica_count``."""

    def __init__(self, *, shard_index: int, replica_count: int, shard_id: str) -> None:
        self._shard_index = shard_index
        self._replica_count = max(1, replica_count)
        self._shard_id = shard_id

    @property
    def shard_id(self) -> str:
        return self._shard_id

    def owns(self, group_name: str) -> bool:
        return consistent_hash_owns(group_name, self._shard_index, self._replica_count)

    def owned_groups(self, group_names: Iterable[str]) -> set[str]:
        return {name for name in group_names if self.owns(name)}

    def needs_periodic_refresh(self) -> bool:
        # Ownership is a pure function of (group, index, count) — nothing to
        # refresh. The runner reconciles only when the group list changes.
        return False

    async def refresh(self, group_names: Iterable[str]) -> None:
        return None

    async def close(self) -> None:
        return None


@runtime_checkable
class AssignmentStore(Protocol):
    """Shared membership registry + published assignment for leader_assigned."""

    async def heartbeat(self, shard_id: str, ttl_seconds: int) -> None: ...

    async def members(self) -> list[str]: ...

    async def publish_assignment(self, mapping: dict[str, str]) -> None: ...

    async def read_assignment(self) -> dict[str, str]: ...


class LeaderAssignedCoordinator:
    """Ownership read from a Lease-elected assigner's published mapping.

    Every replica heartbeats into the shared store. The replica currently
    holding the assigner Lease (``is_leader()`` returns true) recomputes the
    balanced mapping over the live membership and publishes it; all replicas
    read the published mapping to decide ownership.

    Two windows need care:

    * **Cold start** — no mapping published yet. We own *nothing* and wait for
      the assigner (the Lease guarantees one is elected within a lease
      duration). Owning everything here would make every replica fit every
      group at once — an N-times thundering herd at exactly the moment sharding is
      supposed to prevent it. A brief startup gap is the better trade.
    * **A just-added group** — the mapping exists but doesn't yet name a group
      added by a reload before the assigner republished. For that single group
      we fall back to a local rendezvous decision over the membership; because
      rendezvous is deterministic given the same membership, every replica
      agrees on the owner, so the group is covered by exactly one replica.
    """

    def __init__(
        self,
        *,
        shard_id: str,
        store: AssignmentStore,
        is_leader: Callable[[], bool],
        member_ttl_seconds: int,
    ) -> None:
        self._shard_id = shard_id
        self._store = store
        self._is_leader = is_leader
        self._member_ttl = member_ttl_seconds
        self._assignment: dict[str, str] = {}
        self._members: list[str] = [shard_id]

    @property
    def shard_id(self) -> str:
        return self._shard_id

    def owns(self, group_name: str) -> bool:
        owner = self._assignment.get(group_name)
        if owner is not None:
            return owner == self._shard_id
        if not self._assignment:
            # Cold start: no mapping at all yet. Own nothing and wait for the
            # assigner rather than have every replica fit every group at once.
            return False
        # Mapping exists but doesn't name this group (a reload added it before
        # the assigner republished). Decide locally via rendezvous over the
        # membership — deterministic, so every replica agrees on one owner.
        fallback = assign_groups([group_name], self._members)
        return fallback.get(group_name) == self._shard_id

    def owned_groups(self, group_names: Iterable[str]) -> set[str]:
        return {name for name in group_names if self.owns(name)}

    def needs_periodic_refresh(self) -> bool:
        return True

    async def refresh(self, group_names: Iterable[str]) -> None:
        # Always heartbeat first so a transient read/publish failure below
        # doesn't also drop us from the membership.
        try:
            await self._store.heartbeat(self._shard_id, self._member_ttl)
        except Exception as exc:  # pragma: no cover - logged, degrades gracefully
            logger.warning("shard_heartbeat_failed", shard_id=self._shard_id, error=str(exc))
        try:
            members = await self._store.members()
        except Exception as exc:
            logger.warning("shard_members_read_failed", error=str(exc))
            members = []
        # Keep ourselves in the membership even if the store hasn't observed
        # our heartbeat yet (read-after-write lag), so the fallback path never
        # decides we own nothing.
        self._members = sorted(set(members) | {self._shard_id})
        if self._is_leader():
            mapping = assign_groups(sorted(group_names), self._members)
            try:
                await self._store.publish_assignment(mapping)
            except Exception as exc:
                logger.warning("shard_assignment_publish_failed", error=str(exc))
            self._assignment = mapping
            logger.info(
                "shard_assignment_published",
                members=self._members,
                groups=len(mapping),
            )
        else:
            try:
                self._assignment = await self._store.read_assignment()
            except Exception as exc:
                logger.warning("shard_assignment_read_failed", error=str(exc))

    async def close(self) -> None:
        return None


# ---------------------------------------------------------------------------
# Redis-backed assignment store
# ---------------------------------------------------------------------------

_MEMBER_KEY_PREFIX = "promforecast:shard:member:"
_ASSIGNMENT_KEY = "promforecast:shard:assignment"


class RedisAssignmentStore:
    """:class:`AssignmentStore` over the shared async Redis client.

    Membership is a set of short-TTL per-replica keys (``member:<id>``) so a
    dead replica simply expires out — no explicit deregistration needed. The
    assignment is a single JSON blob under one key, refreshed by the assigner
    every tick (no TTL — a stale-but-present map is better than none, and the
    assigner overwrites it continuously).

    Membership is enumerated with ``SCAN`` rather than ``KEYS``: this Redis is
    typically shared with the query cache, and ``KEYS`` blocks the server for
    an O(whole-keyspace) match on every heartbeat. ``SCAN`` is cursor-based and
    non-blocking; the matching set (one key per replica) is tiny, so the cursor
    loop is cheap regardless of how large the surrounding keyspace grows.
    """

    def __init__(self, client: object) -> None:
        self._client = client

    async def heartbeat(self, shard_id: str, ttl_seconds: int) -> None:
        key = f"{_MEMBER_KEY_PREFIX}{shard_id}"
        await self._client.set(key, str(time.time()), ex=max(1, ttl_seconds))  # type: ignore[attr-defined]

    async def members(self) -> list[str]:
        pattern = f"{_MEMBER_KEY_PREFIX}*"
        members: list[str] = []
        cursor = 0
        while True:
            cursor, raw_keys = await self._client.scan(  # type: ignore[attr-defined]
                cursor=cursor, match=pattern, count=100
            )
            for raw in raw_keys or []:
                key = raw.decode("utf-8") if isinstance(raw, bytes | bytearray) else str(raw)
                members.append(key[len(_MEMBER_KEY_PREFIX) :])
            # redis-py returns the cursor as an int; 0 marks a completed sweep.
            if int(cursor) == 0:
                break
        return members

    async def publish_assignment(self, mapping: dict[str, str]) -> None:
        await self._client.set(_ASSIGNMENT_KEY, json.dumps(mapping))  # type: ignore[attr-defined]

    async def read_assignment(self) -> dict[str, str]:
        raw = await self._client.get(_ASSIGNMENT_KEY)  # type: ignore[attr-defined]
        if raw is None:
            return {}
        text = raw.decode("utf-8") if isinstance(raw, bytes | bytearray) else str(raw)
        try:
            data = json.loads(text)
        except (ValueError, TypeError):
            return {}
        if not isinstance(data, dict):
            return {}
        return {str(k): str(v) for k, v in data.items()}


__all__ = [
    "AssignmentStore",
    "ConsistentHashCoordinator",
    "LeaderAssignedCoordinator",
    "RedisAssignmentStore",
    "ShardCoordinator",
    "assign_groups",
    "consistent_hash_assignment",
    "consistent_hash_owns",
    "derive_shard_id",
    "derive_shard_index",
    "stable_bucket",
]
