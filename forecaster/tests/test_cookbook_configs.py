"""CI guard: every cookbook YAML loads cleanly and yields a finite cardinality.

The docs/cookbook recipes ship copy-pasteable YAML in
``examples/configs/cookbook/``. If one of those snippets drifts from
the schema (new validator added, field renamed, type changed) operators
hit a confusing failure right after copy-pasting the recipe — exactly
the friction the cookbook is meant to remove. This test loads every
snippet through ``promforecast.config.load`` *and* the cardinality
estimator so a schema or estimator regression catches the cookbook on
the same PR as the rest of the test suite.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from promforecast import config as config_module
from promforecast.validate import _estimate_cardinality

REPO_ROOT = Path(__file__).resolve().parents[2]
COOKBOOK_DIR = REPO_ROOT / "examples" / "configs" / "cookbook"

# Enumerated so a recipe that someone forgot to ship under
# ``examples/configs/cookbook/`` is a hard failure rather than a
# silently-skipped test.
EXPECTED_RECIPES = frozenset(
    {
        "node-filesystem-avail-bytes.yaml",
        "http-requests-total.yaml",
        "kube-deployment-status-replicas.yaml",
        "jvm-gc-pause-seconds.yaml",
        "redis-connected-clients.yaml",
        "aws-cloudwatch-estimated-charges-usd.yaml",
        "kafka-consumer-lag.yaml",
    }
)


def test_cookbook_directory_has_every_expected_recipe() -> None:
    """Every recipe listed in the cookbook README ships a YAML snippet."""
    present = {path.name for path in COOKBOOK_DIR.glob("*.yaml")}
    missing = EXPECTED_RECIPES - present
    assert not missing, (
        f"cookbook YAMLs missing for recipes: {sorted(missing)}. "
        f"Add the file under {COOKBOOK_DIR}/."
    )


@pytest.mark.parametrize("filename", sorted(EXPECTED_RECIPES))
def test_cookbook_config_loads_and_estimates(filename: str) -> None:
    """Each cookbook YAML loads through the schema validator without error.

    Then the cardinality estimator must produce a finite, non-zero
    total. A zero total means no forecast / accuracy / quality rows
    will be emitted — which is almost certainly a recipe bug
    (everything disabled).
    """
    path = COOKBOOK_DIR / filename
    cfg = config_module.load(path)
    estimate = _estimate_cardinality(cfg)
    assert estimate.total > 0, (
        f"cookbook recipe {filename} produced zero estimated rows — "
        "check the model / horizon / emission config."
    )
    # Every recipe carries at least one query.
    assert sum(len(g.queries) for g in cfg.groups) >= 1
