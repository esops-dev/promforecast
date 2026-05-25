# Warm-up status

Right after `helm install` (or any restart that clears Prometheus data), the key question is:  
**“What is still loading, and how long until forecasts appear?”**

The `GET /forecast/warmup` endpoint answers exactly that — it shows per-query progress toward the first usable forecast.

## Enabling

```yaml
server:
  expose_warmup_endpoint: true   # default: false
```

Same network-level access control as `/-/reload` and `/forecast/preview`.

## Endpoint

```bash
curl -s http://forecaster:9091/forecast/warmup | jq .
```

```json
{
  "groups": [
    {
      "name": "node_capacity",
      "queries": [
        {
          "id": "node_cpu_busy_pct",
          "status": "ready",
          "points_available": 200,
          "points_needed": 100,
          "blocked_by": "none",
          "eta_seconds": 0
        },
        {
          "id": "node_memory_used_bytes",
          "status": "warming",
          "points_available": 40,
          "points_needed": 100,
          "blocked_by": "min_points",
          "eta_seconds": 3600
        }
      ]
    }
  ]
}
```

**blocked_by** values:

| Value             | Meaning |
|-------------------|---------|
| `none`            | Ready — next refresh will emit a forecast |
| `min_points`      | Below `defaults.min_points` |
| `lookback`        | No samples in the configured lookback window |
| `regressor:<id>`  | Required custom regressor is missing samples |
| `discovery`       | Templated query — discovery probe failed |
| `change_point`    | Reserved for future use |

## Companion CLI

```bash
promforecast warmup --url http://forecaster:9091
```

Prints a clean table:

```
group         query                     status  points needed eta  blocked_by
------------- ------------------------- ------- ------ ------ ---  ----------
node_capacity node_cpu_busy_pct         ready      200    100   -  none
node_capacity node_memory_used_bytes    warming     40    100 1h  min_points
```

Flags:
- `--json` → raw JSON output
- `--timeout <duration>` → HTTP deadline (default 30s)

The CLI exits `0` when every query is `ready`, `1` otherwise, `2` on network error — perfect for CI/pre-flight scripts.

## Cost

Each call issues one cheap `query_range` per non-event, non-templated query (plus one extra per required custom regressor).  
All probes run concurrently, so wall time equals the slowest probe.

The endpoint is opt-in because even these light probes are unnecessary once the system is warm.

## Limitations

- Event-profile queries (`data_profile: events`) and templated queries (`discover:`) always report `ready`.
- `eta_seconds` is a simple linear estimate (“one sample per step”). It can under- or over-estimate on sparse data or backfills.
- Only **required** custom regressors are probed; optional ones are ignored.

## What to expect on a fresh install

A typical 1 h refresh / 14 d lookback / `min_points: 200` setup needs ~17 hours of data before the first fit.  
`warmup` will show `lookback` → `min_points` → `none`.  
Once all queries are `ready`, the next refresh emits forecasts and the normal quality/accuracy/deviation metrics begin populating.
