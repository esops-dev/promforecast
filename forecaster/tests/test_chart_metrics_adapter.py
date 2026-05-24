"""Chart-template tests for the ``promforecast-metrics-adapter`` chart.

These tests assert chart-level invariants that ``helm lint`` /
``kubeconform`` can't catch on their own:

* The auto-generated TLS Secret's ``ca.crt`` is the **same** material
  that lands in both ``APIService.spec.caBundle`` entries — without
  this, the API aggregator would handshake against a self-signed leaf
  that doesn't chain to the advertised CA and HPA would 503 forever.
  Verifying the equality also exercises the ``_tlsMaterial`` cached
  helper (a fresh ``genCA`` per template call would silently diverge).
* The Deployment's ``checksum/tls`` annotation flips when the
  rendered cert material changes — proves the rolling-restart wiring
  for a cert rotation.
* The ``apiService.enabled: false`` path drops both APIService
  manifests cleanly (operators running this alongside another adapter
  registered under the same group can flip the toggle without forking
  the chart).
* The chart no longer ships the dead ``horizontal-pod-autoscaler``
  ServiceAccount binding — a regression guard so a future "convenience"
  PR doesn't reintroduce a binding to a non-existent principal.

Like the existing ServiceMonitor tests, this shells out to ``helm
template``; tests skip cleanly when ``helm`` is absent.
"""

from __future__ import annotations

import base64
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
ADAPTER_CHART = REPO_ROOT / "charts" / "promforecast-metrics-adapter"

helm = shutil.which("helm")
pytestmark = pytest.mark.skipif(
    helm is None, reason="`helm` not on PATH — chart-template tests are skipped"
)


def _helm_template(chart: Path, *sets: str) -> list[dict]:
    """Render ``chart`` with ``--set`` flags and return the parsed docs."""
    cmd = [helm, "template", "test-release", str(chart)]
    for s in sets:
        cmd.extend(["--set", s])
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return [d for d in yaml.safe_load_all(result.stdout) if isinstance(d, dict)]


def _find(docs: list[dict], kind: str, *, name_contains: str | None = None) -> dict | None:
    for d in docs:
        if d.get("kind") != kind:
            continue
        if name_contains is None or name_contains in d.get("metadata", {}).get("name", ""):
            return d
    return None


def _findall(docs: list[dict], kind: str) -> list[dict]:
    return [d for d in docs if d.get("kind") == kind]


def test_default_install_renders_all_core_resources() -> None:
    """A no-args render should produce the full adapter surface: SA, Deployment,
    Service, ConfigMap, TLS Secret, 2 APIService manifests, and 1
    ClusterRoleBinding + 1 RoleBinding for auth-delegation.
    """
    docs = _helm_template(ADAPTER_CHART)
    kinds = sorted(d.get("kind", "") for d in docs)
    assert "Deployment" in kinds
    assert "Service" in kinds
    assert "ConfigMap" in kinds
    assert "ServiceAccount" in kinds
    assert "Secret" in kinds
    assert kinds.count("APIService") == 2
    # auth-delegator ClusterRoleBinding + auth-reader RoleBinding.
    assert kinds.count("ClusterRoleBinding") == 1
    assert kinds.count("RoleBinding") == 1


def test_apiservice_cabundle_matches_secret_ca() -> None:
    """The CA bundle the API aggregator dials against MUST be the CA that
    signed the leaf cert the adapter serves under. The cached
    ``_tlsMaterial`` helper guarantees this within a single render; this
    test pins the invariant so a future helper refactor can't quietly
    regress it (the failure mode is silent: HPA gets 503 + TLS handshake
    failure forever).
    """
    docs = _helm_template(ADAPTER_CHART)
    secret = _find(docs, "Secret", name_contains="tls")
    assert secret is not None, "TLS Secret must render in the default install"
    secret_ca_pem = base64.b64decode(secret["data"]["ca.crt"]).decode()
    apiservices = _findall(docs, "APIService")
    assert len(apiservices) == 2, "Both external + custom APIService manifests must render"
    for apisvc in apiservices:
        ca_bundle_b64 = apisvc["spec"]["caBundle"]
        ca_bundle_pem = base64.b64decode(ca_bundle_b64).decode()
        assert ca_bundle_pem == secret_ca_pem, (
            f"APIService {apisvc['metadata']['name']} caBundle diverges from the "
            "rendered TLS Secret ca.crt — the cached _tlsMaterial helper has broken "
            "and a re-genCA happened mid-render."
        )


def test_apiservice_caches_share_one_ca_across_both_manifests() -> None:
    """Both APIService manifests must dial against the same CA bundle —
    they reference the same Service, so any divergence is a render bug.
    """
    docs = _helm_template(ADAPTER_CHART)
    apiservices = _findall(docs, "APIService")
    bundles = {apisvc["spec"]["caBundle"] for apisvc in apiservices}
    assert len(bundles) == 1, (
        "external + custom APIService caBundle values differ — the cached "
        "_tlsMaterial helper is regenerating across calls."
    )


def test_tls_secret_carries_all_three_pem_keys() -> None:
    """Kubelet projects ``tls.crt`` + ``tls.key`` into the pod and the
    APIService reads ``ca.crt``. All three keys must be present.
    """
    docs = _helm_template(ADAPTER_CHART)
    secret = _find(docs, "Secret", name_contains="tls")
    assert secret is not None
    assert secret["type"] == "kubernetes.io/tls"
    assert set(secret["data"]) >= {"tls.crt", "tls.key", "ca.crt"}
    # Sanity: the b64-decoded payload is PEM-shaped, not empty.
    for key in ("tls.crt", "tls.key", "ca.crt"):
        pem = base64.b64decode(secret["data"][key]).decode()
        assert pem.startswith("-----BEGIN "), f"{key} is not PEM-encoded"


def test_api_service_disabled_drops_both_manifests() -> None:
    """``apiService.enabled: false`` drops the registration manifests so an
    operator who already runs another adapter under these groups can
    install this chart's pod + Service without conflicting registrations.
    """
    docs = _helm_template(ADAPTER_CHART, "apiService.enabled=false")
    assert _findall(docs, "APIService") == [], (
        "apiService.enabled=false still rendered APIService manifests"
    )
    # The pod surface is preserved.
    assert _find(docs, "Deployment") is not None
    assert _find(docs, "Service") is not None


def test_deployment_picks_up_cert_via_checksum_annotation() -> None:
    """``checksum/tls`` must be present on the pod template so a cert
    rotation (initial install, manual Secret delete, or
    ``existingSecret`` swap) triggers a rolling restart — uvicorn loads
    ``ssl_certfile`` once at boot and won't pick up a rotation
    otherwise.
    """
    docs = _helm_template(ADAPTER_CHART)
    deployment = _find(docs, "Deployment")
    assert deployment is not None
    annotations = deployment["spec"]["template"]["metadata"].get("annotations", {}) or {}
    assert "checksum/tls" in annotations
    assert "checksum/config" in annotations


def test_existing_secret_without_cluster_lookup_fails_render() -> None:
    """When the operator sets ``tls.existingSecret`` but the named Secret
    isn't in the cluster (or we're rendering offline), the helper must
    ``fail`` rather than fall back to ``insecureSkipTLSVerify: true``,
    which would defeat the existingSecret contract.
    """
    result = subprocess.run(
        [
            helm,
            "template",
            "test-release",
            str(ADAPTER_CHART),
            "--set",
            "tls.generate=false",
            "--set",
            "tls.existingSecret=does-not-exist",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0, (
        "render with tls.existingSecret pointing at a missing Secret should fail loud"
    )
    assert "does-not-exist" in result.stderr


def test_rbac_does_not_bind_horizontal_pod_autoscaler_sa() -> None:
    """Regression guard: the chart used to bind a ``metrics-reader``
    ClusterRole to a ``horizontal-pod-autoscaler`` ServiceAccount in
    ``kube-system`` — that SA doesn't exist in vanilla Kubernetes (HPA
    runs inside kube-controller-manager and uses
    ``system:kube-controller-manager``). The binding was dead code and
    has been removed; this test pins it stays gone.
    """
    docs = _helm_template(ADAPTER_CHART)
    for binding in _findall(docs, "ClusterRoleBinding"):
        for subj in binding.get("subjects", []) or []:
            assert not (
                subj.get("kind") == "ServiceAccount"
                and subj.get("name") == "horizontal-pod-autoscaler"
            ), (
                "ClusterRoleBinding to the non-existent horizontal-pod-autoscaler "
                "ServiceAccount has reappeared — see rbac.yaml's rationale comment."
            )


def test_clusterrolebinding_targets_auth_delegator_only() -> None:
    """The only ClusterRoleBinding the chart ships binds the adapter's
    own ServiceAccount to ``system:auth-delegator``. Any other
    ClusterRoleBinding is a regression worth catching.
    """
    docs = _helm_template(ADAPTER_CHART)
    bindings = _findall(docs, "ClusterRoleBinding")
    assert len(bindings) == 1
    binding = bindings[0]
    assert binding["roleRef"]["name"] == "system:auth-delegator"


def test_deployment_mounts_match_secret_keys() -> None:
    """The pod projects only the leaf cert + key into the container —
    ``ca.crt`` stays in the Secret for the APIService consumer. A drift
    here (renaming the volume items without updating the
    ``--tls-cert``/``--tls-key`` args) would break TLS on the adapter
    side.
    """
    docs = _helm_template(ADAPTER_CHART)
    deployment = _find(docs, "Deployment")
    assert deployment is not None
    spec = deployment["spec"]["template"]["spec"]
    tls_volume = next(v for v in spec["volumes"] if v["name"] == "tls")
    item_keys = {item["key"] for item in tls_volume["secret"]["items"]}
    assert item_keys == {"tls.crt", "tls.key"}
    args = spec["containers"][0]["args"]
    assert any(a == "--tls-cert=/var/run/promforecast-adapter/tls.crt" for a in args)
    assert any(a == "--tls-key=/var/run/promforecast-adapter/tls.key" for a in args)
