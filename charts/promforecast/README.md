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

## Values

See [`values.yaml`](values.yaml) for the full schema.
