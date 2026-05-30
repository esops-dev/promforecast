# Arbitrary-quantile forecasts

The standard forecast emits a central value plus symmetric confidence bands (`<id>_forecast_lower` / `<id>_forecast_upper`).  
For risk-based alerting (“P[disk full within 24h] > 0.05”) and fan charts, you want predictions at *arbitrary* quantiles instead.

`emission.quantiles` emits a parallel family from the **same single fit**:

```
<id>_forecast_quantile{q="0.95", id, group, model, horizon}
```

## Configuring

```yaml
groups:
  - name: capacity
    queries:
      - id: node_filesystem_avail_bytes
        promql: node_filesystem_avail_bytes
        emission:
          quantiles: [0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
```

Values must be strictly between 0 and 1. Duplicates are rejected at load time. The existing band family remains for backward compatibility.

## How a quantile is derived

Each requested quantile is read from the appropriate band edge of the level that brackets it:

| Quantile `q` | Reads from |
|--------------|------------|
| `q = 0.5`    | central forecast (`yhat`) |
| `q < 0.5`    | lower edge of level `round((1 − 2q)·100)` |
| `q > 0.5`    | upper edge of level `round((2q − 1)·100)` |

Quantiles beyond the configured confidence levels are clamped to the outermost band.

## Conformal calibration per quantile

When `accuracy.conformal.enabled: true` is also set, each quantile receives its own empirical-coverage correction. The offset is exposed alongside band-level offsets in the existing gauge, discriminated by the `q` label:

```
forecast_band_calibration_offset{id, group, model, q="0.95"}
```

This adds one extra hold-out fit per emitted model per refresh.

## Fan chart & risk-based alerting

The bundled capacity dashboard includes a “Forecast fan chart” panel that overlays all emitted quantiles plus the central forecast.

Risk-based alerts become natural:

```promql
# Optimistic 10th-percentile still predicts disk fill inside horizon
node_filesystem_avail_bytes_forecast_quantile{q="0.1", horizon="24h"} < 0
```

```promql
# 99th-percentile request rate exceeds capacity
http_requests_rate_forecast_quantile{q="0.99"} > 5000
```

## Cardinality

`N_quantiles` rows per (series, model) per opted-in query, plus matching calibration-offset rows when conformal is enabled.  
`promforecast validate` counts the family.
