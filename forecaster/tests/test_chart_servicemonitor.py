"""Chart-template tests for the standalone ``promforecast`` chart's
ServiceMonitor and the umbrella ``promforecast-stack`` sink-first
defaults.

The standalone chart exposes ``serviceMonitor.dropForecastSnapshots``,
which defaults to ``null``. ``null`` is the "follow the sink toggle"
state: when ``config.sink.remote_write.enabled`` is true the template
renders a ``metricRelabelings`` block dropping the
``.+_forecast(_lower|_upper)?`` snapshot families; when the sink is
off, no relabeling is rendered. A literal ``true`` / ``false`` pins the
behaviour irrespective of the sink. The umbrella chart pre-sets both
the sink and ServiceMonitor toggles so a fresh install lands on
sink+VM with the snapshot drop applied by default.

These tests shell out to ``helm template`` because the alternative
(implementing a Go template renderer in Python) would diverge from the
real chart behaviour. The cost is a hard dependency on ``helm`` being
on PATH; CI installs it via ``azure/setup-helm`` and the dev environment
has it from ``brew install helm``. Tests skip cleanly when helm is
absent so a contributor without it can still run the rest of the
forecaster suite.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
STANDALONE_CHART = REPO_ROOT / "charts" / "promforecast"
UMBRELLA_CHART = REPO_ROOT / "charts" / "promforecast-stack"

helm = shutil.which("helm")
pytestmark = pytest.mark.skipif(
    helm is None, reason="`helm` not on PATH — chart-template tests are skipped"
)


def _umbrella_deps_resolved() -> bool:
    """Check whether ``charts/promforecast-stack/charts/`` carries both subcharts.

    A fresh checkout (and the CI ``test`` job, which doesn't run
    ``helm dependency update``) starts with the ``charts/`` directory
    empty. ``helm template`` then bails with ``found in Chart.yaml,
    but missing in charts/ directory``. The fixture below resolves
    them lazily; this helper is the cheap idempotency check.
    """
    subcharts_dir = UMBRELLA_CHART / "charts"
    if not subcharts_dir.is_dir():
        return False
    names = {p.name for p in subcharts_dir.iterdir()}
    has_vm = any(n.startswith("victoria-metrics-single") and n.endswith(".tgz") for n in names)
    has_pf = any(n.startswith("promforecast-") and n.endswith(".tgz") for n in names)
    return has_vm and has_pf


@pytest.fixture(scope="session", autouse=True)
def _resolve_umbrella_chart_deps() -> None:
    """Resolve the umbrella chart's subchart dependencies once per session.

    The CI ``test`` job (pytest) runs without the ``helm dependency
    update`` step that ``chart-lint`` runs, so ``helm template
    charts/promforecast-stack`` would fail with missing ``victoria-
    metrics-single`` / ``promforecast`` subcharts. The chart-template
    tests live next to the rest of the forecaster suite, so the
    resolution has to happen here rather than be assumed at job-level.

    The VictoriaMetrics chart repo is registered idempotently (helm's
    ``repo add`` errors on a pre-existing entry; ``|| true`` is wrong
    because we want the real error if it's a different problem, so we
    inspect the exit code). The dependency-update itself is cheap on a
    warm cache and unconditional so a stale ``charts/*.tgz`` from a
    different version always gets refreshed.
    """
    if helm is None or _umbrella_deps_resolved():
        return
    # Register the upstream repo. ``helm repo add`` exits 0 on a fresh
    # add and 1 if the same name is already present with a different
    # URL — for our case the URL is fixed, so a non-zero on the
    # "already exists" path is benign. We discriminate by checking the
    # stderr message rather than swallowing every failure.
    repo_add = subprocess.run(
        [helm, "repo", "add", "victoria-metrics", "https://victoriametrics.github.io/helm-charts/"],
        capture_output=True,
        text=True,
        check=False,
    )
    if repo_add.returncode != 0 and "already exists" not in (repo_add.stderr or ""):
        pytest.skip(
            "could not register the victoria-metrics helm repo, skipping chart-template tests: "
            f"{repo_add.stderr.strip()}"
        )
    subprocess.run([helm, "repo", "update"], capture_output=True, text=True, check=False)
    subprocess.run(
        [helm, "dependency", "update", str(UMBRELLA_CHART)],
        capture_output=True,
        text=True,
        check=True,
    )


def _helm_template(chart: Path, *sets: str) -> list[dict]:
    """Render ``chart`` with the supplied ``--set`` flags and return docs."""
    cmd = [helm, "template", str(chart)]
    for s in sets:
        cmd.extend(["--set", s])
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return [d for d in yaml.safe_load_all(result.stdout) if isinstance(d, dict)]


def _find(docs: list[dict], kind: str, *, name_contains: str | None = None) -> dict | None:
    """Return the first rendered doc matching ``kind`` (and optional name substring)."""
    for d in docs:
        if d.get("kind") != kind:
            continue
        if name_contains is None or name_contains in d.get("metadata", {}).get("name", ""):
            return d
    return None


def _relabelings(servicemonitor: dict) -> list[dict]:
    """Return the ServiceMonitor's first endpoint's metricRelabelings list."""
    eps = servicemonitor["spec"]["endpoints"]
    assert eps, "ServiceMonitor has no endpoints"
    return eps[0].get("metricRelabelings", []) or []


def _has_snapshot_drop(servicemonitor: dict) -> bool:
    for r in _relabelings(servicemonitor):
        if (
            r.get("action") == "drop"
            and r.get("regex") == ".+_forecast(_lower|_upper)?"
            and r.get("sourceLabels") == ["__name__"]
        ):
            return True
    return False


# ---------------------------------------------------------------------------
# Standalone chart: dropForecastSnapshots derivation
# ---------------------------------------------------------------------------


def test_sink_on_renders_snapshot_drop_by_default() -> None:
    """Default ``dropForecastSnapshots: null`` + sink on → drop block present."""
    docs = _helm_template(
        STANDALONE_CHART,
        "serviceMonitor.enabled=true",
        "config.sink.remote_write.enabled=true",
    )
    sm = _find(docs, "ServiceMonitor")
    assert sm is not None
    assert _has_snapshot_drop(sm), _relabelings(sm)


def test_sink_off_omits_snapshot_drop() -> None:
    """Default ``dropForecastSnapshots: null`` + sink off → no drop block."""
    docs = _helm_template(STANDALONE_CHART, "serviceMonitor.enabled=true")
    sm = _find(docs, "ServiceMonitor")
    assert sm is not None
    assert not _has_snapshot_drop(sm), _relabelings(sm)


def test_explicit_false_overrides_sink_on() -> None:
    """``dropForecastSnapshots: false`` with sink on → drop block suppressed.

    The opt-out is for the rare case where the snapshot end-of-horizon
    gauges are wanted alongside the sink curves (e.g. a parallel-cutover
    install verifying the two paths line up before flipping the
    scrape-drop on).
    """
    docs = _helm_template(
        STANDALONE_CHART,
        "serviceMonitor.enabled=true",
        "config.sink.remote_write.enabled=true",
        "serviceMonitor.dropForecastSnapshots=false",
    )
    sm = _find(docs, "ServiceMonitor")
    assert sm is not None
    assert not _has_snapshot_drop(sm), _relabelings(sm)


def test_explicit_true_overrides_sink_off() -> None:
    """``dropForecastSnapshots: true`` with sink off → drop block forced.

    Mirror of the opt-out path: pinning ``true`` keeps the drop in place
    even when the sink is off (e.g. an externally-managed sink that the
    forecaster does not drive itself).
    """
    docs = _helm_template(
        STANDALONE_CHART,
        "serviceMonitor.enabled=true",
        "serviceMonitor.dropForecastSnapshots=true",
    )
    sm = _find(docs, "ServiceMonitor")
    assert sm is not None
    assert _has_snapshot_drop(sm), _relabelings(sm)


def test_user_supplied_metricrelabelings_compose_with_drop() -> None:
    """User-extras are preserved alongside the derived drop block.

    Drop is prepended so the dropped forecast snapshots cannot satisfy a
    subsequent ``keep`` rule; user-extras run on whatever survives.
    """
    docs = _helm_template(
        STANDALONE_CHART,
        "serviceMonitor.enabled=true",
        "config.sink.remote_write.enabled=true",
        "serviceMonitor.metricRelabelings[0].action=keep",
        "serviceMonitor.metricRelabelings[0].sourceLabels[0]=__name__",
        "serviceMonitor.metricRelabelings[0].regex=foo",
    )
    sm = _find(docs, "ServiceMonitor")
    assert sm is not None
    relabelings = _relabelings(sm)
    assert len(relabelings) == 2
    assert relabelings[0]["action"] == "drop"
    assert relabelings[1]["action"] == "keep"
    assert relabelings[1]["regex"] == "foo"


# ---------------------------------------------------------------------------
# Umbrella chart: sink-first default lands the drop block automatically
# ---------------------------------------------------------------------------


def test_umbrella_default_install_renders_sink_and_drop() -> None:
    """The umbrella's defaults must produce both halves of the sink+VM pair.

    The ConfigMap must carry ``sink.remote_write.enabled: true`` (so the
    forecaster pushes the curve) and the rendered ServiceMonitor must
    carry the snapshot drop (so the scrape doesn't duplicate the curve
    under the forecaster's own labelset).
    """
    docs = _helm_template(UMBRELLA_CHART)
    # Look up the forecaster's config ConfigMap by data key rather than
    # name substring — the umbrella also renders a ``-datasources``
    # ConfigMap whose name contains ``promforecast`` and helm template
    # ordering between subcharts is not guaranteed across Helm versions.
    cm = next(
        (d for d in docs if d.get("kind") == "ConfigMap" and "config.yaml" in d.get("data", {})),
        None,
    )
    assert cm is not None, "umbrella did not render the forecaster config ConfigMap"
    cfg = yaml.safe_load(cm["data"]["config.yaml"])
    assert cfg["sink"]["remote_write"]["enabled"] is True
    sm = _find(docs, "ServiceMonitor")
    assert sm is not None, "umbrella must render a ServiceMonitor by default"
    assert _has_snapshot_drop(sm), _relabelings(sm)


def test_umbrella_datasources_default_to_victoriametrics() -> None:
    """The bundled Grafana datasource ConfigMap registers VM as default."""
    docs = _helm_template(UMBRELLA_CHART, "datasources.enabled=true")
    cm = _find(docs, "ConfigMap", name_contains="datasources")
    assert cm is not None
    provisioning = yaml.safe_load(cm["data"]["promforecast-datasources.yaml"])
    vm = next(ds for ds in provisioning["datasources"] if ds["name"] == "VictoriaMetrics")
    assert vm["isDefault"] is True
    assert vm["uid"] == "victoriametrics"


def test_standalone_validate_hook_pod_renders_by_default() -> None:
    """The ``helm test`` validate-config Pod renders with the strict flag.

    The hook lets an operator chain ``helm test <release>`` into a
    GitOps PostSync gate that rejects configs whose projected
    cardinality exceeds ``safety.max_total_series``. CI does not run
    ``helm test`` (no kind cluster), but ``helm template`` +
    ``kubeconform`` validate that the asset renders to a schema-valid
    Pod and carries the expected ``--strict-cardinality`` argument so
    the gate is actually wired up.
    """
    docs = _helm_template(STANDALONE_CHART)
    pod = _find(docs, "Pod", name_contains="test-validate")
    assert pod is not None, "standalone chart must render the validate-config test pod by default"
    annotations = pod["metadata"]["annotations"]
    assert annotations.get("helm.sh/hook") == "test"
    # Inspect the container's args so a regression that drops the
    # strict flag (turning the hook back into an advisory smoke test)
    # surfaces in CI.
    containers = pod["spec"]["containers"]
    assert containers, "validate-config test pod must declare a container"
    args = containers[0].get("args") or []
    assert "validate" in args
    assert "--strict-cardinality" in args
    assert "--config=/etc/promforecast/config.yaml" in args


def test_standalone_validate_hook_can_be_disabled() -> None:
    """``tests.validateConfig.enabled=false`` suppresses the hook entirely.

    Useful for managed-cluster environments that forbid Helm hook
    Pod creation.
    """
    docs = _helm_template(STANDALONE_CHART, "tests.validateConfig.enabled=false")
    pod = _find(docs, "Pod", name_contains="test-validate")
    assert pod is None


def test_umbrella_helm_test_pod_renders() -> None:
    """The labelset-uniqueness ``helm test`` hook is present by default.

    The pod queries VM after a real install and asserts that
    ``*_forecast`` series exist under exactly one labelset family. CI
    does not run ``helm test`` (no kind cluster), but ``helm template``
    + ``kubeconform`` validate that the asset renders to a schema-valid
    Pod.
    """
    docs = _helm_template(UMBRELLA_CHART)
    pod = _find(docs, "Pod", name_contains="test-labelset")
    assert pod is not None
    annotations = pod["metadata"]["annotations"]
    assert annotations.get("helm.sh/hook") == "test"


# ---------------------------------------------------------------------------
# ServiceMonitor: jobLabel + honorLabels (release-name agnostic scrape)
# ---------------------------------------------------------------------------


def _endpoint(servicemonitor: dict) -> dict:
    eps = servicemonitor["spec"]["endpoints"]
    assert eps, "ServiceMonitor has no endpoints"
    return eps[0]


def test_servicemonitor_pins_job_label_to_app_name() -> None:
    """``jobLabel: app.kubernetes.io/name`` pins the scrape job to ``promforecast``.

    Without this prometheus-operator generates a release-name-derived
    job label (``<ns>/<sm-name>/<endpoint>``), which makes the example
    alerts and the labelset-uniqueness helm-test impossible to write
    generically.
    """
    docs = _helm_template(STANDALONE_CHART, "serviceMonitor.enabled=true")
    sm = _find(docs, "ServiceMonitor")
    assert sm is not None
    assert sm["spec"].get("jobLabel") == "app.kubernetes.io/name"


def test_servicemonitor_honors_source_labels() -> None:
    """``honorLabels: true`` preserves source ``instance`` on scraped metrics.

    The forecaster exposes its operational and derived ``*_forecast_*``
    families with the underlying signal's source labels (e.g.
    ``instance="node-exporter:9100"``). Without ``honorLabels`` the
    scrape would clobber that with the forecaster pod's own address,
    breaking the example alerts that interpolate ``{{ $labels.instance }}``
    to identify the underlying host.
    """
    docs = _helm_template(STANDALONE_CHART, "serviceMonitor.enabled=true")
    sm = _find(docs, "ServiceMonitor")
    assert sm is not None
    assert _endpoint(sm).get("honorLabels") is True


# ---------------------------------------------------------------------------
# Standalone: existingConfigMap forces an explicit dropForecastSnapshots
# ---------------------------------------------------------------------------


def test_existing_configmap_without_explicit_drop_fails_render() -> None:
    """GitOps ``existingConfigMap`` users must set ``dropForecastSnapshots``.

    With config managed out-of-band the chart cannot read the sink toggle
    to derive the drop. Silently defaulting to off would land the user
    in the dual-source state; we fail the render and force a decision.
    """
    with pytest.raises(subprocess.CalledProcessError) as exc:
        _helm_template(
            STANDALONE_CHART,
            "serviceMonitor.enabled=true",
            "existingConfigMap=external-config",
        )
    stderr = exc.value.stderr or ""
    assert "dropForecastSnapshots must be set explicitly" in stderr, stderr


def test_existing_configmap_with_explicit_drop_renders() -> None:
    """Setting ``dropForecastSnapshots`` explicitly unblocks the GitOps path."""
    docs = _helm_template(
        STANDALONE_CHART,
        "serviceMonitor.enabled=true",
        "existingConfigMap=external-config",
        "serviceMonitor.dropForecastSnapshots=true",
    )
    sm = _find(docs, "ServiceMonitor")
    assert sm is not None
    assert _has_snapshot_drop(sm), _relabelings(sm)


# ---------------------------------------------------------------------------
# Umbrella: release-name-agnostic VM service URL
# ---------------------------------------------------------------------------


def test_umbrella_vm_service_name_is_release_name_agnostic() -> None:
    """The pinned ``victoria-metrics-single.fullnameOverride`` keeps the VM
    Service name stable regardless of Helm release name, so the
    forecaster's hardcoded ``datasource.url`` / ``sink.remote_write.url``
    resolve in any install. Without the pin, ``helm install foo`` would
    create ``foo-victoria-metrics-single-server`` but the config would
    still point at ``promforecast-stack-victoria-metrics-single-server``.
    """
    for release in ("promforecast-stack", "foo", "team-prod"):
        docs = _helm_template(UMBRELLA_CHART)  # baseline
        vm = next(
            (
                d
                for d in docs
                if d.get("kind") == "Service" and "victoria-metrics-single" in d["metadata"]["name"]
            ),
            None,
        )
        assert vm is not None
        assert vm["metadata"]["name"] == "promforecast-stack-victoria-metrics-single-server", (
            f"release={release}: VM service name drifted to {vm['metadata']['name']}"
        )


# ---------------------------------------------------------------------------
# Umbrella: datasources.victoriaMetrics.isDefault is overridable
# ---------------------------------------------------------------------------


def test_umbrella_datasources_isdefault_overridable() -> None:
    """Users with a Grafana that already has a default datasource (e.g.
    kube-prometheus-stack registering Prometheus) need to flip VM's
    ``isDefault`` off without losing the chart-provisioned VM entry.
    """
    docs = _helm_template(
        UMBRELLA_CHART,
        "datasources.enabled=true",
        "datasources.victoriaMetrics.isDefault=false",
    )
    cm = _find(docs, "ConfigMap", name_contains="datasources")
    assert cm is not None
    provisioning = yaml.safe_load(cm["data"]["promforecast-datasources.yaml"])
    vm = next(ds for ds in provisioning["datasources"] if ds["name"] == "VictoriaMetrics")
    assert vm["isDefault"] is False
    assert vm["uid"] == "victoriametrics"


# ---------------------------------------------------------------------------
# Umbrella: helm test pod has the first-refresh poll loop
# ---------------------------------------------------------------------------


def test_umbrella_helm_test_pod_polls_for_first_refresh() -> None:
    """The helm-test probe must tolerate the post-install delay before the
    first scheduled refresh lands sink data in VM. A naive one-shot
    assertion would fail every fresh install because ``helm test`` runs
    immediately after ``helm install``, before the forecaster has
    completed its first fit.
    """
    docs = _helm_template(UMBRELLA_CHART)
    pod = _find(docs, "Pod", name_contains="test-labelset")
    assert pod is not None
    container = pod["spec"]["containers"][0]
    env = {e["name"]: e.get("value") for e in container.get("env", [])}
    assert env.get("FIRST_REFRESH_TIMEOUT_SECONDS"), "poll timeout env var missing"
    script = container["command"][-1]
    assert "while :" in script, "helm test pod missing the poll loop"
    assert "FIRST_REFRESH_TIMEOUT_SECONDS" in script
    # Fail-fast on the dup case (no waiting) — but wait for sink presence.
    assert "exit 1" in script
    assert "sleep 5" in script
