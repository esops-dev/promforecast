# Dynamic series discovery

Instead of writing one YAML query per series, declare a discovery variable and let the forecaster expand it at runtime. Every expanded series receives the same model, seasonality, and emission settings.

## Configuring

```yaml
groups:
  - name: per_instance_cpu
    queries:
      - id: node_cpu_busy_pct
        promql: |
          100 - avg(rate(node_cpu_seconds_total{instance="$instance",mode="idle"}[2m])) * 100
        discover:
          - name: instance
            label: instance
            promql: 'up{job="node-exporter"}'
```

On every refresh the forecaster:

1. Runs the discovery PromQL.
2. Extracts the distinct values of the declared `label`.
3. Substitutes `$instance` (or `${instance}`) in the primary PromQL once per value.
4. Fetches and processes each rendered query exactly as if it had been written by hand.

Multiple variables produce the Cartesian product:

```yaml
- id: tcp_estab
  promql: 'sum(rate(node_netstat_TcpExt_TCPSynRetrans{instance="$instance",cluster="$cluster"}[5m]))'
  discover:
    - { name: instance, label: instance, promql: 'up{job="node"}' }
    - { name: cluster,  label: cluster,  promql: 'up{job="kubernetes-apiserver"}' }
```

Use `$name` or `${name}`. The braced form is required when the placeholder is followed by alphanumerics (`$instance_id` would otherwise be treated as a single identifier).

## Cardinality safety

Discovery respects the same limits as static queries:

- Post-expansion count is capped by `safety.max_series_per_query`. Excess values are dropped according to `safety.series_overflow`.
- The result still counts against the global `safety.max_total_series` budget.
- A discovery query that returns zero values or is missing the requested label increments `forecast_discovery_failures_total` (no silent partial expansion).

`promforecast validate` skips discovery queries, so its cardinality estimate is a **lower bound** when `discover:` is present.

## Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `forecast_discovery_expansions{group, query}` | gauge | Number of concrete queries produced by the template in the most recent run |
| `forecast_discovery_failures_total{group, query, variable, reason}` | counter | Failures per variable (`reason`: `query_error`, `query_timeout`, `empty_result`, `unknown_label`, `no_source`) |

## When not to use discovery

- The result set is small and static — hand-written queries are clearer.
- You need per-series settings (different models, seasonality, etc.) — split into separate top-level queries instead.
- You want to embed complex business logic in the discovery PromQL — put that logic in a recording rule in the TSDB.
