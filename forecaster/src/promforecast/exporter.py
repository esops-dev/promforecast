"""Prometheus exposition: snapshot cache + metric-name validator.

The validator enforces Prometheus naming rules strictly. Colons are reserved
for recording rules and rejected here. There is no escape hatch.
"""

from __future__ import annotations

import re

_METRIC_NAME_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


class InvalidMetricNameError(ValueError):
    """Raised when an output metric name violates Prometheus conventions."""


def validate_metric_name(name: str) -> None:
    """Reject metric names that are not safe for Prometheus exposition.

    Rules:
        - Must match ``[a-zA-Z_][a-zA-Z0-9_]*``.
        - Must not contain a colon (reserved for recording rules).

    Raises:
        InvalidMetricNameError: If ``name`` does not satisfy the rules.
    """
    if ":" in name:
        raise InvalidMetricNameError(
            f"metric name {name!r} contains a colon; colons are reserved for recording rules"
        )
    if not _METRIC_NAME_RE.match(name):
        raise InvalidMetricNameError(
            f"metric name {name!r} does not match Prometheus naming regex"
        )


class Exporter:
    """Holds the most recent forecast snapshot and renders /metrics."""

    def __init__(self) -> None:
        pass

    def render(self) -> bytes:
        """Render the current snapshot in Prometheus exposition format."""
        raise NotImplementedError
