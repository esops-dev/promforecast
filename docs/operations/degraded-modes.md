# Degraded-mode operations playbook

The forecaster isolates failures per query, per series, and per group, and retries many transient errors automatically. As a result, several degraded states are invisible if you only watch the per-fit failure counter.

This page lists the five operational failure modes worth monitoring, the metric that detects each, and the recovery path.

The shipped example alerts in [`examples/alerts/promforecast-rules.yaml`](../../examples/alerts/promforecast-rules.yaml) include ready-to-tune rules for each row under the `promforecast.degraded.examples` group.

## Failure-mode matrix

| Failure mode                  | Detection signal                                      | Recovery |
|-------------------------------|-------------------------------------------------------|----------|
| Datasource (PromQL read) unreachable | `forecast_source_retries_total` rate + `forecast_failures_total{reason=~"query_error\|query_timeout"}` | Check VM/Mimir/Thanos reachability, network policies, auth |
| Sink (`remote_write`) writing fails | `forecast_sink_write_failures_total` rate            | Check TSDB reachability, `futureRetention`, auth headers |
| Partial group failure (one query degrades) | `forecast_failures_total{group, reason}` + `forecast_regressor_failures_total` | Per-query investigation |
| Sustained fit timeout         | `forecast_slow_fits_total` + `forecast_failures_total{reason="fit_timeout"}` | Tighten `safety.fit_timeout`, lighter model, or shorter lookback |
| Stale snapshot (group hasn't refreshed) | `forecast_snapshot_age_seconds` + `forecast_last_run_timestamp_seconds` | Check pod health and the rows above |

## What the forecaster does on each failure

Failure isolation is layered so a single fault never escalates:

| Layer          | Caught by                          | Continues running? |
|----------------|------------------------------------|--------------------|
| Per-regressor  | `required: false` (default)        | Yes — forecast fits without the failing regressor |
| Per-query      | `try/except` inside `run_group`    | Yes — sibling queries continue |
| Per-series     | `try/except` inside `_run_query`   | Yes — sibling series continue |
| Per-group      | `try/except` at top of `run_group` | Yes — sibling groups continue |
| Sink           | `try/except` around `sink.write`   | Yes — snapshot still served on `/metrics` |

## Datasource read failure

**Detection**  
```promql
rate(forecast_source_retries_total[5m]) > 0.1
or
rate(forecast_failures_total{reason=~"query_error|query_timeout"}[5m]) > 0.1
```

**Recovery**  
Check VictoriaMetrics / Mimir / Thanos health, network policies, and auth. Update `datasource.url` and reload if the endpoint moved.

Shipped alert: `ForecastDatasourceUnreachable`.

## Sink write failure

**Detection**  
```promql
rate(forecast_sink_write_failures_total[5m]) > 0
```

**Recovery**  
Check TSDB reachability, `futureRetention`, and auth headers in `sink.remote_write.headers`.

Shipped alert: `ForecastSinkWriteFailing`.

## Partial group failure

**Detection**  
`forecast_failures_total{group, reason}` and `forecast_regressor_failures_total`.

**Recovery**  
Filter failures by group + query in logs; fix the source (regressor rename, cardinality spike, model timeout) and reload.

## Sustained fit timeout

**Detection**  
`forecast_slow_fits_total` or `forecast_failures_total{reason="fit_timeout"}`.

**Recovery**  
Raise `safety.fit_timeout`, switch to a lighter model, or shorten `lookback`. Isolate problematic series into their own group if needed.

## Stale snapshot

**Detection**  
```promql
forecast_snapshot_age_seconds > 7200
or
time() - forecast_last_run_timestamp_seconds > 7200
```

**Recovery**  
Check pod health (OOMKilled, `Recreate` strategy) and the rows above for the root cause.

Shipped alert: `ForecastSnapshotStale`.

## Putting it together

Typical production alerting pages on `ForecastDatasourceUnreachable`, `ForecastSinkWriteFailing`, and `ForecastSnapshotStale`. Lower-severity rules (`ForecastModelFitFailing`, `ForecastSeriesStale`, quality/deviation) route to tickets.

Each paged alert maps to a concrete recovery action in the matrix above.

## Cross-references

- [Reload verification](reload.md) — catches partial-config issues that lead to stale snapshots.
- [Telemetry](telemetry.md) — full catalogue of the operational metrics referenced here.
- [Cardinality budgeting](cardinality.md) — emission families assumed counted against `safety.max_total_series`.
