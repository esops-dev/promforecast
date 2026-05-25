# Recipe: `jvm_gc_pause_seconds`

> Long GC pauses correlate with user-facing latency. Forecast the rate so you can recycle pods before SLOs are impacted.

## The metric

`jvm_gc_pause_seconds` is a histogram of GC pause durations. The forecastable signal is the **rate of long pauses**:

```promql
rate(jvm_gc_pause_seconds_count{action="end of major GC"}[5m])
```

A baseline rate is normal; a step change to a sustained higher rate is the precursor to pods dropping requests.

## The config

```yaml
groups:
  - name: jvm
    queries:
      - id: jvm_gc_long_pause_rate
        promql: |
          sum by (instance, pool) (
            rate(jvm_gc_pause_seconds_count{action="end of major GC"}[5m])
          )
        emission:
          drift:
            enabled: true
            alert_threshold: 0.3
```

## The model

* **`AutoARIMA`** captures the baseline + noise pattern well. JVMs rarely show strong seasonality on this metric.
* `SeasonalNaive` rides along but rarely wins auto-select.

## Expected forecast shape

- Mostly flat with occasional spikes (usually deploy-driven cache invalidation or stop-the-world pauses).
- Relatively wide confidence band because the underlying signal is spiky.

## Cardinality

For 50 JVMs × 2 pools × 1 model × 2 levels:

- forecast lines: 500
- deviation lines: 200
- drift lines: 100
- quality lines: 100

## Alerts

```promql
# Long-GC rate outside the predicted band — investigate.
forecast_deviation_outside_band{
  id="jvm_gc_long_pause_rate", level="95"
} == 1
```

```promql
# Forecast itself is unstable — the model can't fit recent data.
forecast_drift_score{id="jvm_gc_long_pause_rate"} > 0.3
for 4 fits
```

The drift alert is especially useful here — the [drift change-correlation explainer](../operations/drift-explained.md) will surface which regressor (deploy timing, hour-of-day, …) most correlates with the change.
