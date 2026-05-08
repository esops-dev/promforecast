"""Kubernetes Lease-based leader election for HA mode.

Only the elected leader runs the scheduler and fits models. Followers serve
``/metrics`` from the shared Redis cache.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager


@asynccontextmanager
async def lead(
    namespace: str,
    name: str,
    identity: str,
    lease_duration_seconds: int = 30,
    renew_deadline_seconds: int = 20,
) -> AsyncIterator[None]:
    """Acquire a coordination.k8s.io/v1 Lease and yield while leader.

    On loss of the lease the context exits and the caller stops scheduling.
    """
    raise NotImplementedError
    yield  # unreachable; marks this as an async generator for type checkers
