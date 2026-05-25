# Production-readiness checklist and sizing guide

A single reference for the questions an operator must answer before going live: how much CPU/memory to provision, which deployment shape to pick, and which knobs to tick off before flipping production traffic.

Detailed how-tos live in the linked pages; this page integrates them.

## Sizing (rule of thumb)

CPU cost is dominated by per-series fits. AutoARIMA is the heaviest built-in model; SeasonalNaive is essentially free.

| Series count | Refresh interval | Recommended CPU | Recommended memory | Replicas (HA mode) |
|--------------|------------------|-----------------|---------------------|--------------------|
| < 100        | 1h               | 200m            | 256Mi               | 1 (no HA)          |
| 100 – 500    | 1h               | 500m            | 512Mi               | 1 (no HA)          |
| 500 – 2,000  | 1h               | 1               | 1Gi                 | 2 (HA)             |
| 2,000 – 10,000 | 1h             | 2               | 2Gi                 | 2 (HA)             |
| 10,000 – 50,000 | 30m           | 4               | 4Gi                 | 3 (HA)             |
| > 50,000     | varies           | —               | —                   | Horizontal sharding (v1.10) |

These are upper bounds. Use `promforecast validate --estimate-cost` for a config-specific projection.

**Tightening refresh interval**  
A 30-minute refresh roughly doubles CPU vs. 1h; a 5-minute refresh costs ~12×. Most operators stay at 30m or slower unless short-horizon SLOs require it.

**Adding emission families**  
Each v1.6+ family multiplies cardinality. Always re-run `promforecast validate --strict-cardinality` after enabling a new family. See [cardinality.md](cardinality.md) for the per-family table.

## Reference architectures

| Shape                          | When to use | Helm values |
|--------------------------------|-------------|-------------|
| **A. Single-replica small**    | < 500 series, 30 s `/metrics` gap acceptable | `replicas: 1`<br>`highAvailability.enabled: false` |
| **B. Leader-elected medium**   | > 500 series or `/metrics` availability required | `replicas: 2`<br>`highAvailability.enabled: true` |
| **C. Per-cluster + central**   | Multi-cluster fleets | See [multi-cluster architectures](architecture/multi-cluster.md) |

## Production-readiness checklist

| Item | Expected end state | Reference |
|------|--------------------|-----------|
| `sink.remote_write.enabled: true` | Chart default; standalone installs set explicitly | [emission-paths.md](reference/emission-paths.md) |
| Scrape-drop applied | `serviceMonitor.dropForecastSnapshots: true` (or equivalent metric_relabel_configs) | [emission-paths.md](reference/emission-paths.md) |
| `ServiceMonitor` present | Forecaster operational metrics scraped (not dropped forecast snapshots) | [values.yaml](../charts/promforecast/values.yaml) |
| Alerts loaded | Shipped `promforecast-rules.yaml` rules deployed and tuned | [examples/alerts/promforecast-rules.yaml](../examples/alerts/promforecast-rules.yaml) |
| Dashboards provisioned | Five shipped Grafana dashboards installed | [dashboards/grafana/](../dashboards/grafana/) |
| Backfill considered | Optional 30 d backfill run so SLO dashboards are useful from day one | [backfill.md](operations/backfill.md) |
| Cost estimate run | `promforecast validate --estimate-cost` confirms projected fit time fits inside `refresh_interval` | [validate.md](operations/validate.md) |
| Quality gate on predictive autoscaling | Any HPA reading a forecast metric pairs it with `forecast_quality_score > 0.6` | [predictive-autoscaling.md](predictive-autoscaling.md) |
| Fleet what-if disabled in small installs | `server.expose_what_if_endpoint: false` unless resource headroom exists | [what-if-fleet.md](outputs/what-if-fleet.md) |
| `futureRetention` configured on VM | ≥ 2 d (chart default) | [promforecast-stack/values.yaml](../charts/promforecast-stack/values.yaml) |
| Degraded-mode alerts wired | `ForecastDatasourceUnreachable`, `ForecastSinkWriteFailing`, `ForecastSnapshotStale` mapped to on-call | [degraded-modes.md](operations/degraded-modes.md) |
| Auto-tuner installed (optional) | Weekly CronJob from `promforecast-tuner` sub-chart | [auto-tuning.md](operations/auto-tuning.md) |

## What you can defer

- HA mode (single-replica with `Recreate` is valid for small installs).
- Auto-tuner.
- Backfill (SLO dashboards become useful after a few refresh cycles).

## What you cannot defer

- Sink + scrape-drop combination (prevents dual-source state).
- `safety.max_total_series` cap (re-evaluate after enabling new families).
- Degraded-mode alerts (prevents silent stale snapshots).

## Cross-references

- [Multi-cluster architectures](architecture/multi-cluster.md)
- [Degraded-mode playbook](operations/degraded-modes.md)
- [Cardinality budgeting](operations/cardinality.md)
- [Migration to sink-first](how-to/migrate-to-sink-first.md)
