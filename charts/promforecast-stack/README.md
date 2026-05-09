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

## Subchart toggles

- `victoria-metrics-single.enabled`: bundle a VM single-node install.
- `promforecast.enabled`: bundle the forecaster.
- `dashboards.enabled`: install Grafana dashboards as ConfigMaps.
- `prometheusRules.enabled`: install example `PrometheusRule` manifests.

See [`values.yaml`](values.yaml) for the full schema.
