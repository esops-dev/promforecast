"""Per-group execution and APScheduler integration.

Each group runs as an asyncio task with a per-group lock so a slow group
cannot be triggered twice concurrently. Failure isolation is layered: per
query, per series, per regressor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import Config, GroupConfig


class Runner:
    """Schedules and executes per-group forecast runs."""

    def __init__(self, config: Config) -> None:
        self._config = config

    async def start(self) -> None:
        """Register groups with APScheduler and run each once immediately."""
        raise NotImplementedError

    async def run_group(self, group: GroupConfig) -> None:
        """Execute a single group: query, fit, select, export."""
        raise NotImplementedError

    async def stop(self) -> None:
        """Shut down the scheduler and cancel in-flight runs."""
        raise NotImplementedError
