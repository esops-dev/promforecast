# Drift change-correlation explainer

When `forecast_drift_score` crosses `emission.drift.alert_threshold`, the forecaster automatically explains *why* the prediction shifted.

It emits a structured `forecast_drift_explained` log event and bumps the counter `forecast_drift_explained_total{group, query, primary_suspect}` so dashboards can surface “which regressor most often explains drift on this query?” without log scraping.

No extra config is required — the explainer activates whenever `emission.drift.enabled: true` (default).

## What lands in the log event

```json
{
  "event": "forecast_drift_explained",
  "group": "node_capacity",
  "query": "node_cpu_busy_pct",
  "model": "AutoARIMA",
  "series_key": "host-a",
  "drift_score": 0.342,
  "drift_threshold": 0.25,
  "primary_suspect": "deploys",
  "top_value_suspect": "deploys",
  "top_correlation_suspect": "deploys",
  "top_regressor_correlation": 0.82,
  "top_regressor_correlation_confidence": "high",
  "correlation_window_size": 30,
  "regressor_values_at_drift": {
    "deploys": 1.0,
    "hour_of_day": 14
  }
}
```

| Field                               | Meaning |
|-------------------------------------|---------|
| `drift_score` / `drift_threshold`   | The crossing that triggered the explanation |
| `top_value_suspect`                 | Regressor with largest absolute value at drift moment |
| `top_correlation_suspect`           | Regressor whose history best predicts drift over the rolling window |
| `top_regressor_correlation`         | Pearson correlation `[-1, 1]` (`null` if insufficient data) |
| `top_regressor_correlation_confidence` | `high` / `medium` / `low` / `insufficient_data` |
| `primary_suspect`                   | Label used on the counter (prefers correlation when confidence is `high` or `medium`) |
| `regressor_values_at_drift`         | Snapshot of every regressor’s latest value at the drift moment |

## Confidence bands

| Band                | Meaning |
|---------------------|---------|
| `high`              | ≥0.7 correlation **and** ≥20 samples |
| `medium`            | ≥0.4 correlation **and** ≥8 samples |
| `low`               | Correlation exists but is weak |
| `insufficient_data` | Fewer than 4 valid pairs |

## Tuning the rolling window

```yaml
emission:
  drift:
    alert_threshold: 0.25
    correlation_window: 60   # one week at hourly refresh
```

Smaller windows react faster but are noisier. Larger windows give smoother correlations but lag when the driver changes.

## Building alerts on the counter

```promql
# A query keeps drifting and the same regressor explains it
sum by (group, query, primary_suspect) (
  rate(forecast_drift_explained_total[1h])
) > 0.5
```

Or surface the top-N causes fleet-wide:

```promql
topk(10, sum by (primary_suspect) (increase(forecast_drift_explained_total[24h])))
```

## When the counter stays at zero

The counter only bumps when drift actually crosses the threshold.  
A deployment whose forecasts never cross the threshold (or whose threshold is too high) leaves the counter at zero — this is expected, not a bug.

## When `primary_suspect="unknown"`

This appears when:
- No regressors are configured.
- Every regressor value is non-finite at the drift moment.
- No regressor produced a finite correlation or absolute value.

The structured log still fires so you can see drift happened.
