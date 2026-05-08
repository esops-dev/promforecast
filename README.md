# promforecast

Forecasts for Prometheus-compatible metrics, exposed as Prometheus metrics.

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

See [`docs/`](docs/) for configuration, model selection, and operational
guidance.

## Status

Early development. Output metric names are part of the public API and follow
Prometheus conventions strictly (no colons — those are reserved for recording
rules).

## License

[Apache-2.0](LICENSE).
