"""Prometheus metric-name validation.

Lives in its own module (rather than alongside the exporter) because the
config loader and the chart's CI drift check both need to validate IDs
*without* pulling in ``prometheus_client``. Keeping the rule small and
self-contained also makes it easy to share with future tooling
(e.g. a pre-commit hook).

Output names are part of the public API: see CONTRIBUTING.md.
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
        raise InvalidMetricNameError(f"metric name {name!r} does not match Prometheus naming regex")
