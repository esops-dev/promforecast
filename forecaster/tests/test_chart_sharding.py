"""Chart-template tests for sharding.

consistent_hash sharding must render a StatefulSet with stable ordinals (and
no Deployment); leader_assigned must render a Deployment plus the Lease RBAC and
flow the Redis address into the forecaster config. The chart values must
round-trip through the forecaster's own loader (mirrors the CI drift check).

Like the other chart-template tests these shell out to ``helm`` and skip
cleanly when it isn't on PATH.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
STANDALONE_CHART = REPO_ROOT / "charts" / "promforecast"

helm = shutil.which("helm")
pytestmark = pytest.mark.skipif(
    helm is None, reason="`helm` not on PATH — chart-template tests are skipped"
)


def _helm_template(*sets: str) -> list[dict]:
    cmd = [helm, "template", str(STANDALONE_CHART)]
    for s in sets:
        cmd.extend(["--set", s])
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return [d for d in yaml.safe_load_all(result.stdout) if isinstance(d, dict)]


def _kinds(docs: list[dict]) -> set[str]:
    return {d.get("kind") for d in docs}


def _find(docs: list[dict], kind: str) -> dict | None:
    return next((d for d in docs if d.get("kind") == kind), None)


def _config_yaml(docs: list[dict]) -> dict:
    cm = next(
        d for d in docs if d.get("kind") == "ConfigMap" and "config.yaml" in d.get("data", {})
    )
    return yaml.safe_load(cm["data"]["config.yaml"])


def test_default_renders_deployment_not_statefulset() -> None:
    docs = _helm_template()
    assert "Deployment" in _kinds(docs)
    assert "StatefulSet" not in _kinds(docs)


def test_consistent_hash_renders_statefulset() -> None:
    docs = _helm_template(
        "sharding.enabled=true",
        "sharding.mode=consistent_hash",
        "replicaCount=4",
    )
    kinds = _kinds(docs)
    assert "StatefulSet" in kinds
    assert "Deployment" not in kinds
    sts = _find(docs, "StatefulSet")
    assert sts is not None
    assert sts["spec"]["replicas"] == 4
    assert sts["spec"]["serviceName"]
    assert sts["spec"]["podManagementPolicy"] == "Parallel"
    # Config carries the resolved sharding block with replica_count from replicas.
    cfg = _config_yaml(docs)
    assert cfg["sharding"]["enabled"] is True
    assert cfg["sharding"]["mode"] == "consistent_hash"
    assert cfg["sharding"]["replica_count"] == 4
    # No Lease RBAC for consistent_hash (no coordination backend).
    assert "Role" not in kinds


def test_consistent_hash_injects_pod_name_env() -> None:
    docs = _helm_template("sharding.enabled=true", "sharding.mode=consistent_hash")
    sts = _find(docs, "StatefulSet")
    assert sts is not None
    env = sts["spec"]["template"]["spec"]["containers"][0]["env"]
    names = {e["name"] for e in env}
    assert "PROMFORECAST_SHARDING_ENABLED" in names
    assert "PROMFORECAST_POD_NAME" in names


def test_leader_assigned_renders_deployment_and_lease_rbac() -> None:
    docs = _helm_template(
        "sharding.enabled=true",
        "sharding.mode=leader_assigned",
        "replicaCount=3",
        "sharding.redis.address=redis-master.redis.svc:6379",
    )
    kinds = _kinds(docs)
    assert "Deployment" in kinds
    assert "StatefulSet" not in kinds
    assert "Role" in kinds and "RoleBinding" in kinds
    dep = _find(docs, "Deployment")
    assert dep is not None
    assert dep["spec"]["strategy"]["type"] == "RollingUpdate"
    # Redis address flows into the forecaster's query-cache config.
    cfg = _config_yaml(docs)
    assert cfg["sharding"]["mode"] == "leader_assigned"
    assert cfg["query_cache"]["redis_address"] == "redis-master.redis.svc:6379"


def test_leader_assigned_requires_redis_address() -> None:
    # Schema gate: enabling leader_assigned without a redis address fails.
    with pytest.raises(subprocess.CalledProcessError):
        _helm_template("sharding.enabled=true", "sharding.mode=leader_assigned")
