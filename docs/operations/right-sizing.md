# Right-sizing recommendations

The forecaster already knows the future trajectory of every workload.  
`recommendations.resource_sizing` turns that into a per-workload “you can downsize from X to Y” gauge that operators can park on a Grafana panel and act on systematically.

The recommendation is read-only — no auto-apply, no controller integration.

## Enable it

```yaml
recommendations:
  resource_sizing:
    enabled: true
    headroom_pct: 30
    quality_floor: 0.6
    workloads:
      - workload: api-server
        kind: cpu
        forecast_query: api_server_cpu_usage_pct
        current_promql: >
          sum(kube_pod_container_resource_requests{
            workload="api-server", resource="cpu"
          })
      - workload: api-server
        kind: memory
        forecast_query: api_server_memory_usage_bytes
        current_promql: >
          sum(kube_pod_container_resource_requests{
            workload="api-server", resource="memory"
          })
      # current_promql is optional — recommendation still emits,
      # just without savings_pct
      - workload: search
        kind: cpu
        forecast_query: search_cpu_usage_pct
```

After the next refresh, three sibling gauges appear:

```
forecast_recommended_resource_size{workload="api-server", kind="cpu", confidence="high"} 2.6
forecast_recommended_resource_current{workload="api-server", kind="cpu"} 4.0
forecast_recommended_resource_savings_pct{workload="api-server", kind="cpu", confidence="high"} 35.0
```

| Field            | Default | Meaning |
|------------------|---------|---------|
| `enabled`        | `false` | Opt-in |
| `headroom_pct`   | `30`    | Added to predicted peak |
| `quality_floor`  | `0.6`   | Minimum mean quality required to emit a recommendation |
| `workloads`      | `[]`    | List of (workload, kind, forecast_query, optional current_promql) |

### How the recommendation is computed

- Reads the forecast snapshot of `forecast_query`.
- Takes the peak (`max yhat_upper` at the highest confidence level, falling back to `yhat`).
- Applies `headroom_pct`.
- Compares to `current_promql` (if provided) to compute `savings_pct`.

## Reading the recommendation

| Value                  | Meaning |
|------------------------|---------|
| Positive `savings_pct` | Predicted peak + headroom is below current limit → safe to downsize |
| Negative `savings_pct` | Predicted peak exceeds current limit → needs upsizing |
| `confidence="low"`     | Forecast quality too low to trust — improve the model first |
| Row missing            | Quality below `quality_floor` for all models on this query |

## Worked examples

**Top-10 cost-saving opportunities** (high-confidence only)

```promql
topk(10, forecast_recommended_resource_savings_pct{confidence="high"} > 0)
```

**Pair with absolute recommended size**

```promql
forecast_recommended_resource_size{confidence="high"}
  * on(workload, kind) group_left()
  forecast_recommended_resource_savings_pct{confidence="high"} > 0
```

**Bursty workloads**  
A daily peak well above average pins the recommendation to the upper band — intentional. If occasional pressure is acceptable, raise `headroom_pct`.

## Cardinality

Up to three rows per configured workload.  
`promforecast validate` accounts for the upper bound under “resource sizing”.
