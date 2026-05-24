# promforecast

Forecasts for Prometheus-compatible metrics, exposed as Prometheus metrics.

This chart deploys the forecaster only — bring your own long-term TSDB. For a
batteries-included install (VictoriaMetrics + forecaster + dashboards), use
`promforecast-stack`.

## Install

```bash
helm repo add promforecast https://esops-dev.github.io/promforecast
helm install promforecast promforecast/promforecast \
  --set config.datasource.url=http://victoria-metrics:8428/
```

## Configuration

Either inline the forecaster YAML config under `config:` in `values.yaml`, or
supply an `existingConfigMap`. The latter pairs well with GitOps workflows
that manage the config out of band.

### Single-replica default

The chart defaults to `replicaCount: 1` with `strategy.type: Recreate`. The
in-process scheduler is per-process and running multiple replicas without
coordination would race on the snapshot cache. Set
`highAvailability.enabled: true` and configure Redis to opt into multi-replica
serving with leader election.

## Sink + ServiceMonitor snapshot drop

Set `config.sink.remote_write.enabled: true` to push the full forecast
curve into a long-term TSDB. When `serviceMonitor.enabled: true` the
generated ServiceMonitor automatically drops the `*_forecast` /
`*_forecast_lower` / `*_forecast_upper` snapshot families from the
scrape so the long-term TSDB doesn't end up with both copies under
different labelsets. The derivation key is
`serviceMonitor.dropForecastSnapshots`:

| sink | dropForecastSnapshots | snapshot drop in ServiceMonitor |
| --- | --- | --- |
| on  | `null` (default) | applied |
| off | `null` (default) | not applied |
| on  | `false` (explicit override) | not applied (both copies stored) |
| off | `true` (explicit override) | applied (no snapshot, no sink either) |

When using `existingConfigMap` the chart cannot inspect the sink setting
inside the external ConfigMap. Rather than silently defaulting to the
off side (which would land the GitOps user in the dual-source state),
the template **fails the render** until `serviceMonitor.dropForecastSnapshots`
is pinned to a literal `true` or `false`.

### Scrape labelset pins

The ServiceMonitor pins two scrape-time labels so the bundled example
alerts in `examples/alerts/promforecast-rules.yaml` and the
labelset-uniqueness probe in `charts/promforecast-stack` work
release-name-agnostically:

- `jobLabel: app.kubernetes.io/name` — the scrape `job` label is the
  stable value `promforecast` regardless of Helm release name.
  Without this prometheus-operator generates `<ns>/<sm-name>/<endpoint>`
  as the job, which makes generic PromQL examples impossible to write.
- `honorLabels: true` — the forecaster's `/metrics` exposes operational
  and derived `*_forecast_*` families with the underlying signal's
  source labels (e.g. `instance="node-exporter:9100"`). Honoring those
  prevents Prometheus from clobbering `instance` with the forecaster
  pod's own address.

> **Upgrade note.** Both pins changed scraped labelsets compared to
> earlier releases. Alerts and dashboards that filtered on the
> auto-generated `job` (e.g. `job="<release>-promforecast"`) or the
> forecaster pod's `instance` need to be updated to the new stable
> values before upgrading.

See [`docs/reference/emission-paths.md`](../../docs/reference/emission-paths.md)
for what each emission path emits and the dual-source trap the drop
prevents.

## Values

See [`values.yaml`](values.yaml) for the full schema.
