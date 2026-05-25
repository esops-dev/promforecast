# Recipe: `http_requests_total`

> Per-service traffic forecasts for capacity planning and SLO error-budget projections.

## The metric

`http_requests_total` is the canonical request-rate counter. The forecastable shape is its rate (`rate(...[5m])`).

## The config

```yaml
groups:
  - name: http_traffic
    queries:
      - id: http_requests_rate
        promql: |
          sum by (service) (rate(http_requests_total[5m]))
        step: 1m
        thresholds:
          - name: capacity_soft
            value: 5000
            comparator: gt
            alert_within: 30m
            severity: warning
        emission:
          growth_rate:
            enabled: true
            windows: [1h, 1d]
            units: percent
```

## The model

* **`AutoARIMA`** usually wins auto-select: strong daily cycle plus modest week-over-week growth.
* `SeasonalNaive` rides along as the cheap baseline.

`season_length: 1440` matches the daily cycle at the 1-minute step.

## Expected forecast shape

- Smooth daily curve with a clear business-hours peak.
- Visible weekly modulation (weekdays vs. weekends).
- A deploy or marketing push appears as a short-horizon level shift; the 95 % band absorbs it within an hour or two.

## Cardinality

For 100 services × 1 model × 2 levels:

- forecast lines: 500
- threshold ETA: 300
- will-breach lines: 300
- growth rate lines: 200
- deviation lines: 400
- quality lines: 200

## Alerts

```promql
# Predicted to breach 5k req/s within 30 minutes — scale ahead.
http_requests_rate_forecast_will_breach{severity="warning", level="95"} == 1
```

```promql
# Traffic accelerating 100 % week-over-week — investigate.
http_requests_rate_forecast_growth_rate{window="1h", units="percent"} > 1.0
```

## Pair with SLO error-budget forecasting

Add an `slo:` block to the same query and the forecaster projects when the monthly budget exhausts at the predicted error rate. See [slo.md](../outputs/slo.md).
