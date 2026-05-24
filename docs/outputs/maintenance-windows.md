# Maintenance window recommendations

The forecaster already knows future load. This feature turns that knowledge into a ranked list of the **quietest time windows** of a given duration within a lookahead horizon — eliminating guesswork when scheduling maintenance.

## Configuration

Add to any load-related query:

```yaml
groups:
  - name: node_capacity
    queries:
      - id: node_cpu_busy_pct
        promql: |
          100 - (avg by (instance) (rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)
        recommended_windows:
          enabled: true
          duration: 30m
          lookahead: 48h
          count: 3
```

| Field       | Default | Meaning |
|-------------|---------|---------|
| `enabled`   | `false` | Opt-in (family is off by default) |
| `duration`  | `30m`   | Length of each recommended window |
| `lookahead` | `48h`   | How far into the future to search |
| `count`     | `3`     | Number of windows to recommend, ranked by lowest predicted load |

## Emitted metrics

One global family (no per-id prefix):

```
forecast_recommended_maintenance_window{
  id="node_cpu_busy_pct",
  group="node_capacity",
  model="AutoARIMA",
  rank="1",
  duration_seconds="1800",
  start_timestamp_seconds="1763020800"
} 12.3
```

- **Value** = predicted average load during the window (lower is better).
- `rank` — 1 is the best (quietest) window.
- Ties are broken by earliest start time for stable ranking.

## Worked examples

- **Database reboot** — `duration: 5m`, `lookahead: 24h`, `count: 3`
- **Certificate rotation** — `duration: 15m`, `lookahead: 72h`, `count: 5`
- **Fleet-wide security patch** — `duration: 1h`, `lookahead: 7d`, `count: 5`

## Grafana usage

The bundled capacity dashboard includes a table panel showing top-N windows.  
When using multiple models, filter to one (e.g. the winner):

```promql
forecast_recommended_maintenance_window{
  id="node_cpu_busy_pct",
  model="AutoARIMA"
}
```

Convert `start_timestamp_seconds` to a readable time using Grafana value mappings.

## Cardinality

`count` rows per (series, model) per opted-in query.  
With defaults (`count: 3`) and 500 series this adds ~1500 rows per query — bounded by design.

The `start_timestamp_seconds` label changes over time. Drop it via `metric_relabel_configs` if long-term retention becomes an issue.

## What this is not

- **Not automation** — emits metrics only. Wire your scheduler via Prometheus/Grafana alerts if needed.
- **Not workload-aware** — based purely on the forecast. Does not consider dependencies, business impact, or maintenance fatigue. Use as candidate windows, not final commitments.
