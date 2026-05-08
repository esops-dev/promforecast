"""Tests for the metric-name validator.

Output metric names are part of the public API. The validator rejects colons
unconditionally — they are reserved for recording rules.
"""

from __future__ import annotations

import pytest

from promforecast.exporter import InvalidMetricNameError, validate_metric_name


@pytest.mark.parametrize(
    "name",
    [
        "node_cpu_busy_pct_forecast",
        "_leading_underscore",
        "with_digits_123",
        "a",
    ],
)
def test_valid_names_pass(name: str) -> None:
    validate_metric_name(name)


@pytest.mark.parametrize(
    "name",
    [
        "has:colon",
        "node:cpu:busy:ratio",
        "1leading_digit",
        "with-dash",
        "with space",
        "",
    ],
)
def test_invalid_names_rejected(name: str) -> None:
    with pytest.raises(InvalidMetricNameError):
        validate_metric_name(name)
