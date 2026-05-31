# promforecast-stack

One-command bootstrap: VictoriaMetrics + promforecast + (optional) bundled dashboards and `PrometheusRule` examples.

For greenfield installs and demos. Production deployments typically install `promforecast` alongside an existing VictoriaMetrics, Mimir, or Thanos.

## Install

```bash
helm repo add promforecast https://esops-dev.github.io/promforecast
helm install pf promforecast/promforecast-stack
```

## Future timestamps

The forecaster writes points with future timestamps. VictoriaMetrics drops those by default; this chart sets `victoria-metrics-single.server.extraArgs.futureRetention: "2d"` so they're accepted (VM enforces a 2-day minimum). **Tune this higher than your longest configured forecast horizon** or your longest forecasts will be silently dropped.

## Deduplication

The sink rewrites the full forecast curve on every refit, so each future step is re-emitted by ~`horizon/refresh` successive runs. The sink's contract is last-write-wins per `(labels, timestamp)`. This chart sets `victoria-metrics-single.server.extraArgs.dedup.minScrapeInterval: "15s"` (the Prometheus default `scrape_interval`). **Set it to your actual `scrape_interval`**: a value at or below it collapses duplicate curve writes while preserving raw resolution; a larger value additionally down-samples the underlying signals on this long-term analytics plane.

## Sink-first defaults

The chart pre-configures the **sink + VM** emission path: the forecaster pushes the full forecast curve into the bundled VictoriaMetrics with native future timestamps, and the forecaster's ServiceMonitor drops the `*_forecast` / `*_forecast_lower` / `*_forecast_upper` snapshot families from the scrape. Bundled dashboards default to VictoriaMetrics as their datasource.

A `helm test` hook (`<release>-promforecast-stack-test-labelset`) queries VM after install and asserts no `*_forecast` series carry the forecaster's scrape `job` label — the canonical signature of the dual-source state. Run it on demand:

```bash
helm test <release>
```

See [emission-paths.md](../reference/emission-paths.md) for what the two paths emit, why running both unguarded is a silent trap, and the `metric_relabel_configs` snippet for externally-managed Prometheus.

**ServiceMonitor CRD required.** `promforecast.serviceMonitor.enabled` defaults to `true` so the chart-managed snapshot drop is applied. On clusters without prometheus-operator, set `promforecast.serviceMonitor.enabled=false` and apply the scrape-drop manually.

## Subchart toggles

| Toggle | Description |
|--------|-------------|
| `victoria-metrics-single.enabled` | Bundle a VM single-node install |
| `promforecast.enabled` | Bundle the forecaster (sink to VM by default) |
| `dashboards.enabled` | Install Grafana dashboards as ConfigMaps for the sidecar |
| `datasources.enabled` | Ship a Grafana datasource provisioning ConfigMap |
| `prometheusRules.enabled` | Install example `PrometheusRule` manifests |
| `tests.labelsetUniqueness.enabled` | Include the `helm test` probe described above |

See [values.yaml](values.yaml) for the full schema.