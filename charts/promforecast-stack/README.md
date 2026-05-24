# promforecast-stack

One-command bootstrap: VictoriaMetrics + promforecast + (optional) bundled
dashboards and `PrometheusRule` examples.

For greenfield installs and demos. Production deployments typically install
`promforecast` alongside an existing VictoriaMetrics, Mimir, or Thanos.

## Install

```bash
helm repo add promforecast https://esops-dev.github.io/promforecast
helm install pf promforecast/promforecast-stack
```

## Future timestamps

The forecaster writes points with future timestamps. VictoriaMetrics drops
those by default; this chart sets
`victoria-metrics-single.server.extraArgs.futureRetention: "2d"` so they're
accepted (VM enforces a 2-day minimum). **Tune this higher than your longest
configured forecast horizon or your longest forecasts will be silently
dropped.**

## Sink-first defaults

The chart pre-configures the **sink + VM** emission path: the forecaster
pushes the full forecast curve into the bundled VictoriaMetrics with
native future timestamps, and the forecaster's ServiceMonitor drops the
`*_forecast` / `*_forecast_lower` / `*_forecast_upper` snapshot families
from the scrape so the long-term TSDB stores one canonical labelset per
forecast metric. Bundled dashboards default to VictoriaMetrics as their
datasource.

A `helm test` hook
(`<release>-promforecast-stack-test-labelset`) queries VM after install
and asserts that no `*_forecast` series carry the forecaster's scrape
`job` label — the canonical signature of the dual-source state. Run it
on demand:

```bash
helm test <release>
```

See [`docs/reference/emission-paths.md`](../../docs/reference/emission-paths.md)
for what the two paths emit, why running both unguarded is a silent
trap, and the `metric_relabel_configs` snippet to paste into an
externally-managed Prometheus.

**ServiceMonitor CRD required.** `promforecast.serviceMonitor.enabled`
defaults to `true` so the chart-managed snapshot drop is applied. On
clusters without prometheus-operator (no `monitoring.coreos.com/v1`
ServiceMonitor CRD), `helm install` will fail; flip
`promforecast.serviceMonitor.enabled=false` and apply the scrape-drop
from the docs above to your plain-Prometheus config instead.

## Subchart toggles

- `victoria-metrics-single.enabled`: bundle a VM single-node install.
- `promforecast.enabled`: bundle the forecaster (sink to VM by default).
- `dashboards.enabled`: install Grafana dashboards as ConfigMaps tagged
  for the Grafana sidecar.
- `datasources.enabled`: ship a Grafana datasource provisioning
  ConfigMap (also sidecar-tagged) that registers VM as the default
  datasource the bundled dashboards target.
- `prometheusRules.enabled`: install example `PrometheusRule` manifests.
- `tests.labelsetUniqueness.enabled`: include the `helm test` probe
  described above.

See [`values.yaml`](values.yaml) for the full schema.
