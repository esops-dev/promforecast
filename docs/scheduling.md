# Scheduling and per-query refresh intervals

The forecaster refits each query on a configurable interval.  
A global `server.refresh_interval` still exists, but **per-query overrides** let you mix fast KPI refreshes with slow capacity refits inside the same group — no YAML forking required.

## Default behaviour

```yaml
server:
  refresh_interval: 1h
```

Every query refits every hour — the original single-cadence behaviour when no overrides are present.

## Per-query overrides

```yaml
server:
  refresh_interval: 1h
groups:
  - name: capacity
    queries:
      - id: disk_used_pct
        promql: 1 - (node_filesystem_avail_bytes / node_filesystem_size_bytes)
        refresh_interval: 6h         # refit less often

      - id: requests_per_second
        promql: rate(http_requests_total[5m])
        # inherits global 1h interval
```

The runner keeps a per-query next-run timer. On each tick it only executes queries whose interval has elapsed. Queries that skip a tick keep their previous forecast, accuracy, and quality rows on `/metrics` until the next run.

## Concurrency cap

When several queries share the same refresh cadence they fire together. The new safety cap prevents executor-pool saturation:

```yaml
safety:
  max_concurrent_fits: 8           # default
```

Default matches `datasource.max_concurrent_requests` so the forecaster never opens more parallel fits than it has TSDB read slots.

## Observing per-query timing

Two new families complement the existing group-level gauges:

```
forecast_query_run_duration_seconds{group, query}
forecast_query_series_count{group, query}
```

Group-level metrics (`forecast_run_duration_seconds{group}`, `forecast_series_count{group}`) remain unchanged for backward compatibility.  
Use the per-query families for drill-down:

```promql
# top-5 slowest queries
topk(5, forecast_query_run_duration_seconds)
```

These families appear only after the first successful run.

## Worked example: mixing cadences

| Query                  | `refresh_interval` | Reason |
|------------------------|--------------------|--------|
| `disk_used_pct`        | `6h`               | Slow-moving capacity metric; Grafana panels refresh on the same cadence |
| `cert_expiry_days`     | `12h`              | Changes by seconds per refresh — no value in more frequent fits |
| `requests_per_second`  | `1h`               | Default KPI cadence |
| `errors_per_minute`    | `15m`              | Fast-reacting alert source; needs fresh deviation signals |

`forecast_query_run_duration_seconds` quickly confirms whether the cost split matches expectations.

## Interaction with HA mode

Leader election is replica-wide, not per-query. The leader runs the per-query scheduler exactly as described.  
On failover the new leader starts **every** query immediately (same behaviour as a fresh boot), so you see one catch-up run before the normal cadence resumes.
