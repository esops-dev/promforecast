# Service-level forecast health composite

`forecast_quality_score` answers “how reliable is this *individual* forecast?”  
`forecast_service_health_score` collapses every quality score into one composite per service/environment bucket, giving executives and platform teams a single “how healthy are our forecasts today?” number.

## Enable it

```yaml
governance:
  health_score:
    aggregate_by: [service, environment]
    aggregator: weighted_mean
    weight_by: forecast_series_count
    quality_floor: 0.6   # optional
```

After the next group run, one gauge appears per distinct bucket:

```
forecast_service_health_score{
  service="checkout", environment="prod",
  aggregator="weighted_mean", weight_by="forecast_series_count",
  series_count="42"
} 0.83
```

## Configuration

| Field            | Required | Meaning |
|------------------|----------|---------|
| `aggregate_by`   | Yes      | Label keys that define each bucket |
| `aggregator`     | No       | Only `weighted_mean` supported today |
| `weight_by`      | No       | `uniform` or `forecast_series_count` (documentation label only) |
| `quality_floor`  | No       | Drop any individual quality score below this value before aggregating |

- A quality point missing any key in `aggregate_by` is dropped from all buckets.
- `quality_floor` (default `0.6`) excludes low-quality scores so they don’t drag down the composite.

## Worked example: executive dashboard

Top-row panel for “our forecasts are X % reliable today”:

```promql
# Per-service health, production only
forecast_service_health_score{environment="prod"} > 0.6
```

Worst-first table:

```promql
sort(forecast_service_health_score{environment="prod"})
```

## Cardinality

Bounded by the unique combinations of `aggregate_by` labels.  
Typical case (`[service, environment]` on 20 services × 3 environments) produces ≤ 60 rows — easily dashboard-friendly.

`promforecast validate` reports the upper bound under “service health”.

## When to use vs. when not

**Use it** for executive reporting, platform health overviews, or a single “is the prediction infrastructure healthy?” number.

**Do not use it** as a replacement for per-series `forecast_quality_score` when debugging a specific bad forecast — drill into the individual series for that.

It is also **not** an SRE-style SLI/SLO for the underlying services themselves — use `forecast_error_budget_*` for error-budget tracking (see [slo.md](slo.md)).
