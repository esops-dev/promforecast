# Fleet capacity what-if endpoint

`POST /forecast/what-if/fleet` answers the **strategic** question:  
“If traffic grows by X% over the next N days, which services will hit their capacity ceilings first?”

It re-uses the single-query preview machinery internally, looping over every query that defines a capacity threshold (`gt` or `lt` comparators only) and ranking the projected ETAs.

## Enabling

```yaml
server:
  expose_what_if_endpoint: true   # default: false

safety:
  fleet_what_if_min_interval: 5m  # default: 5m
```

The endpoint follows the same network-level access model as `/forecast/preview` and `/-/reload`.

A rate limit prevents the endpoint from starving scheduled fits. Requests that hit the throttle return `429 Too Many Requests` with a `Retry-After` header and this JSON body:

```json
{
  "error": "throttled",
  "message": "fleet what-if throttled; retry after 287s",
  "retry_after_seconds": 287.4
}
```

## Request body

```json
{
  "traffic_multiplier": 1.5,
  "horizon": "90d",
  "groups": ["payments", "search"]
}
```

| Field                | Required | Notes |
|----------------------|----------|-------|
| `traffic_multiplier` | Yes      | Positive number (`1.0` = baseline ranking) |
| `horizon`            | No       | Prometheus duration or seconds; defaults to `defaults.horizon` |
| `groups`             | No       | Restrict scan to listed groups (unknown groups → 400) |

The multiplier is applied by:
1. **Regressor scaling** (preferred) — if the query has a traffic regressor (`traffic`, `rps`, `qps`, `request_rate`, etc.), the value is overridden.
2. **Baseline fallback** — otherwise the preview runs unchanged and the response row flags `traffic_multiplier_applied: "baseline"`.

## Response

```json
{
  "horizon_seconds": 7776000,
  "traffic_multiplier": 1.5,
  "served_by": "promforecast-0",
  "evaluated_query_count": 24,
  "services": [
    {
      "query_id": "payments_db_storage_used_bytes",
      "group": "infra",
      "threshold_name": "warning",
      "threshold_value": 858993459200,
      "comparator": "gt",
      "time_to_threshold_seconds": 86400.0,
      "model": "AutoARIMA",
      "traffic_multiplier_applied": "regressor"
    }
  ]
}
```

- Rows are ranked by ascending `time_to_threshold_seconds` (sooner breaches first).
- Finite ETAs sort ahead of non-finite (`NaN` = no breach within horizon).
- Failed previews appear with an `"error"` field but do not break the rest of the ranking.

## Worked examples

**Q4 capacity planning**  
```bash
curl -s -X POST http://promforecast:9091/forecast/what-if/fleet \
  -H 'content-type: application/json' \
  -d '{"traffic_multiplier": 1.4, "horizon": "90d"}' \
  | jq '.services[] | select(.time_to_threshold_seconds < 7776000)'
```

**Post-acquisition absorption**  
```bash
curl -s -X POST ... \
  -d '{"traffic_multiplier": 1.3, "horizon": "90d", "groups": ["api", "ingest"]}'
```

**Black Friday peak simulation**  
```bash
curl -s -X POST ... \
  -d '{"traffic_multiplier": 3.0, "horizon": "7d"}' \
  | jq '.services[] | select(.time_to_threshold_seconds < 86400)'
```

## Operational notes

- Throttle is per-replica (in-memory). In HA mode each replica enforces its own limit.
- The endpoint is stateless and emits no metrics.
- The bundled “Is it working?” dashboard includes a *Fleet capacity stress test* table that shows the same shape — pair it with a notebook curl for baseline vs. overlay comparisons.

## See also

- [`preview.md`](preview.md) — single-query what-if this endpoint composes over
- [`capacity.md`](capacity.md) — the underlying `time_to_threshold_seconds` metric
- [`maintenance-windows.md`](maintenance-windows.md) — ranked quiet windows instead of ranked services
