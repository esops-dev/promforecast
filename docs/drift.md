# Forecast drift telemetry

The forecaster compares each new forecast curve to the previous fit for the same series and emits two families:

- `forecast_drift_score{id, group, model, method, ...}` — gauge showing normalised distance between the two curves.
- `forecast_drift_alerts_total{group, id}` — counter that ticks when the score crosses the configured threshold.

Drift is **prediction-vs-prediction** (how much the forecast itself moved), not prediction-vs-actuals. It acts as a leading indicator of regime changes, surfacing issues before `forecast_deviation_*` alerts fire.

## Configuration

```yaml
groups:
  - name: requests
    queries:
      - id: requests_per_second
        promql: rate(http_requests_total[5m])
        emission:
          drift:
            enabled: true            # default
            method: mape             # mape (default) | wasserstein
            alert_threshold: 0.25    # null disables the alert counter
```

| Option            | Values                     | Default | Effect |
|-------------------|----------------------------|---------|--------|
| `enabled`         | `true` / `false`           | `true`  | Drop the entire family |
| `method`          | `mape` / `wasserstein`     | `mape`  | Symmetric MAPE (default) or earth-mover distance |
| `alert_threshold` | float or `null`            | `0.25`  | Score that triggers the alert counter |

## Reading the gauge

- `forecast_drift_score = 0` → curves are identical (stable).
- `forecast_drift_score > 0` → curve moved. Value is average symmetric percentage change (`mape`) or normalised Wasserstein distance.
- No sample → first successful fit or previous fit failed. Emission starts only after two consecutive successful fits.

## Comparison to deviation alerts

| Signal                  | Question answered                              | Triggered by                              |
|-------------------------|------------------------------------------------|-------------------------------------------|
| `forecast_drift_*`      | “Did the prediction itself shift?”             | New fit very different from previous fit  |
| `forecast_deviation_*`  | “Is reality outside the forecasted band?”      | Actuals outside `yhat_lower`/`yhat_upper` |

A rising deviation with stable drift usually means a real change. A rising deviation with spiking drift often means the model is struggling with a new regime.

## Example PrometheusRules

```yaml
- alert: ForecastDriftHigh
  expr: forecast_drift_score > 0.5
  for: 30m
  labels:
    severity: warning
  annotations:
    summary: "Forecast curve for {{ $labels.id }} shifted significantly between fits"
```

For threshold-crossing bursts:

```yaml
- alert: ForecastDriftAlertsBurst
  expr: increase(forecast_drift_alerts_total[1h]) > 3
  for: 15m
```

## Cardinality

One gauge sample per (series, model).  
With `auto_select: true` → one per series.  
Explicit mode → one per series per model.  
`promforecast validate` charges this against the series cap.

## When to disable

- Query refits very frequently against bursty data (e.g. 5 min refresh) — drift becomes noisy.
- Metric is intentionally regime-shifting (e.g. deployment cadence) — pair with `preprocess.change_point` instead.
