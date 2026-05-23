# promforecast

Forecasts for Prometheus-compatible metrics, exposed as Prometheus metrics.

[![CI](https://github.com/esops-dev/promforecast/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/esops-dev/promforecast/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/esops-dev/promforecast?include_prereleases&sort=semver)](https://github.com/esops-dev/promforecast/releases)
[![License](https://img.shields.io/github/license/esops-dev/promforecast)](LICENSE)

promforecast reads a YAML config, queries a Prometheus-compatible TSDB
(VictoriaMetrics by default), runs [`statsforecast`](https://nixtla.github.io/statsforecast/)
models, and exposes the predictions on `/metrics`. Forecasts become first-class
metrics — graphable in Grafana, alertable through Alertmanager, indistinguishable
from anything else in your monitoring stack.

## Why

- **Drop-in.** `helm install` next to an existing Prometheus and point it at a
  long-term TSDB. No production scrape changes required.
- **Speak Prometheus everywhere.** PromQL in, exposition format out. The TSDB
  is swappable (VictoriaMetrics, Mimir, Thanos).
- **Forecasts are just metrics.** Same labels, same retention, same dashboards.
  Compose with recording rules and alerts you already know.
- **Stateless.** No database to operate. Models refit on a schedule; results
  live in `/metrics`.

## Quickstart

```bash
helm repo add promforecast https://esops-dev.github.io/promforecast
helm install promforecast promforecast/promforecast-stack
```

This installs VictoriaMetrics, the forecaster, and example dashboards.
For an existing TSDB, install only `promforecast/promforecast`.

The umbrella chart ships **sink + VictoriaMetrics** as the canonical
forecast write path: the forecaster pushes the full forecast curve into
VM with native future timestamps, and the bundled ServiceMonitor drops
the redundant `*_forecast` / `*_forecast_lower` / `*_forecast_upper`
snapshot families from the scrape so dashboards plot a smooth forward
curve rather than averaging the sink push against a 5-minute snapshot
staircase. See [`docs/architecture/emission-paths.md`](docs/architecture/emission-paths.md)
for what each path emits and the dual-source trap that motivates the
defaults.

Editing the YAML config does **not** require a pod restart: the
forecaster watches the mounted ConfigMap, validates the new config
in-place, and rolls back on validation failure (`kubectl apply` to the
ConfigMap is enough). `POST /-/reload` and `SIGHUP` trigger the same
path. See [`docs/cli.md`](docs/cli.md) for `promforecast validate`,
which CI users can run against a config file before merging.

See [`docs/`](docs/) for configuration, model selection, and operational
guidance. For a real-world deployment — multi-group config, HA values, and
ready-to-apply ArgoCD/Flux manifests with a reference architecture diagram
— see [`examples/production/`](examples/production/).

## Status

Early development. Output metric names are part of the public API and follow
Prometheus conventions strictly (no colons — those are reserved for recording
rules).

## Security

Found a vulnerability? See [SECURITY.md](SECURITY.md). Please do not open public issues for security reports.

## Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md) and the [docs/](docs/) folder for rule authoring.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
