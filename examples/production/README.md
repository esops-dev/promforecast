# Production reference deployment

A real-world reference for running `promforecast` at scale, designed to be
copied wholesale and edited from there rather than read top-to-bottom.

## What's in this folder

| Path                                  | What it is                                                                                                           |
|---------------------------------------|----------------------------------------------------------------------------------------------------------------------|
| `architecture.txt`                    | ASCII reference architecture covering the data flow, failure modes, and production guard rails.                      |
| `forecaster/config.yaml`              | Multi-group forecaster config (business KPIs + node capacity + Kubernetes resources). Use with `existingConfigMap`.  |
| `forecaster/values.yaml`              | Helm values for the `promforecast` chart in HA mode: 2 replicas, RollingUpdate, Redis snapshot cache, NetworkPolicy. |
| `argocd/application.yaml`             | ArgoCD `Application` pointing at the published Helm repo and pulling values from this Git repo.                      |
| `flux/helmrepository.yaml`            | Flux `HelmRepository` for the project's published chart.                                                             |
| `flux/helmrelease.yaml`               | Flux `HelmRelease` consuming the sibling values `ConfigMap`.                                                         |
| `flux/values-configmap.yaml`          | Sibling `ConfigMap` that carries the chart values for the `HelmRelease` above.                                       |

## Reference architecture

```
                      ┌──────────────────────────┐
                      │  Prometheus (prod)       │   never queried by promforecast
                      └────────────┬─────────────┘
                                   │ remote_write
                                   ▼
                      ┌──────────────────────────┐
                      │  VictoriaMetrics         │◀── reads / writes back forecasts
                      │  long-term TSDB          │
                      └────────────┬─────────────┘
                                   │
                ┌──────────────────┴────────────────┐
                │  promforecast (2 replicas, HA)    │
                │  Lease leader election + Redis    │
                │  /metrics on every replica        │
                └──────────────────┬────────────────┘
                                   │ scrape
                                   ▼
                      ┌──────────────────────────┐
                      │  Prometheus / VM         │
                      └────────────┬─────────────┘
                                   ▼
                      ┌──────────────────────────┐
                      │  Grafana                 │
                      └──────────────────────────┘
```

See [`architecture.txt`](architecture.txt) for the full ASCII version,
including failure modes and the per-refresh-tick data flow.

## Sizing assumptions

This reference is tuned for a fleet of roughly:

* ~150 Kubernetes nodes
* ~1,000 services exposing `http_requests_total`-style metrics
* ~3,000 namespaces worth of `container_*` and `kubelet_*` metrics
* ~18,000 output forecast series after fan-out

If you're an order of magnitude smaller, drop:
* `safety.max_total_series` to `5000`
* `resources.limits.cpu` to `"2"` and `memory` to `2Gi`

If you're an order of magnitude larger, see
[`docs/operations/ha.md`](../../docs/operations/ha.md) for sharding guidance.

## A note on duplication

The forecaster config appears in three places: `forecaster/config.yaml`
(standalone, for the `existingConfigMap` GitOps pattern),
`forecaster/values.yaml` (inline under the chart's `config` block), and
`flux/values-configmap.yaml` (embedded as a YAML string inside a
`ConfigMap`). They are kept in sync by hand because each one is a
copy-and-edit starting point for a different install pattern — generating
one from another would make the examples harder to read, not easier.

Pick **one** pattern when you adopt the reference; you don't need all
three in the same cluster. The CI schema check validates every variant on
each PR so they can't silently drift.

## How to apply

### With `helm install` directly

```sh
helm repo add promforecast https://esops-dev.github.io/promforecast
helm repo update
helm install promforecast promforecast/promforecast \
    --namespace observability \
    --create-namespace \
    --values examples/production/forecaster/values.yaml
```

### With ArgoCD

1. Replace the placeholder repo URL in [`argocd/application.yaml`](argocd/application.yaml).
2. `kubectl apply -f argocd/application.yaml`.
3. ArgoCD picks up the chart from the published Helm repo and the values
   from this Git repo. The `RespectIgnoreDifferences` syncOption keeps it
   from fighting with the chart-managed Deployment strategy.

### With Flux

1. `kubectl apply -f flux/helmrepository.yaml` (once per cluster).
2. `kubectl apply -f flux/values-configmap.yaml` to ship the values.
3. `kubectl apply -f flux/helmrelease.yaml` to create the release.

Flux's reconciler watches the `HelmRepository` for new chart versions and
the `ConfigMap` for value changes; you don't need to nudge it.

## Bring your own

These pieces are *not* shipped here — plug your environment-specific
versions in before applying:

* **TSDB.** The reference points at `victoria-metrics-single-server` in
  the `observability` namespace. Replace with your TSDB Service hostname,
  or install the `promforecast-stack` umbrella chart to get VM bundled.
  Whichever TSDB you use, ensure it accepts future-dated samples
  (VictoriaMetrics: `-futureRetention=2d`).
* **Redis.** Required for HA. Deploy `bitnami/redis` (the upstream chart)
  or use a managed instance; reference its hostname under
  `highAvailability.redis.address`.
* **Prometheus Operator.** The `ServiceMonitor` assumes
  `kube-prometheus-stack` with its default selector labels. Edit
  `serviceMonitor.labels` to match your installation.
* **Secrets.** Any TSDB auth or Redis password should land in pre-existing
  Kubernetes Secrets; the chart references them by `existingSecret` /
  `existingConfigMap` keys and never inlines literals.

## What this reference deliberately does *not* enable

| Feature                     | Why off                                                                                                   |
|-----------------------------|-----------------------------------------------------------------------------------------------------------|
| OpenTelemetry tracing       | Adds a hard dep on an in-cluster collector. Flip `telemetry.enabled: true` and set `otlp.endpoint`.       |
| `auto_select: ensemble`     | Default is `true` (per-series winner). Flip to `ensemble` cluster-wide when no single model dominates your workload; expect roughly 2× fit CPU. |
| Per-instance discovery      | The `kube_resources` group already aggregates by namespace; per-pod fan-out is a separate decision.        |
| `series_overflow: reject`   | Stays on `drop_lowest_priority` so a runaway low-priority group can't take down the whole forecaster.       |

See [`docs/`](../../docs/) for the per-feature deep-dives.
