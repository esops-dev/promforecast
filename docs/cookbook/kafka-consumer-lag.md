# Recipe: `kafka_consumer_lag`

> Capacity *and* drain forecasts on the same backlog.

## The metric

`kafka_consumergroup_lag` is the number of messages a consumer group is behind the partition HEAD. Two questions matter:

1. **Capacity** — at the current rate, when does the backlog exhaust broker retention?
2. **Drain** — when does the backlog reach zero, or is it not actually draining?

`promforecast` answers both with one query block using the standard `thresholds` block plus the `le_drain` comparator.

## The config

```yaml
groups:
  - name: kafka
    queries:
      - id: kafka_consumer_lag
        promql: |
          sum by (topic, consumergroup) (
            kafka_consumergroup_lag
          )
        thresholds:
          - name: backlog_warn
            value: 100000
            comparator: gt
            alert_within: 1h
            severity: warning
          - name: drained
            value: 0
            comparator: le_drain
```

The `le_drain` comparator emits to `<id>_forecast_time_to_drain_seconds` (separate metric name from the threshold ETA family) and reports `+Inf` when the forecast never crosses zero from above — alert on that to catch "queue is not actually draining".

## The model

* **`AutoARIMA`** captures the consumer-lag dynamics (autocorrelated, often piecewise linear).
* `SeasonalNaive` rides along; surprisingly often wins auto-select when consumption follows a daily / weekly pattern.

## Expected forecast shape

* Healthy queue: smooth oscillation around zero with brief positive excursions during traffic bursts.
* Stuck consumer: monotonic growth — capacity alert fires.
* Catching up: monotonic decrease — drain ETA reports a finite value; alerting on `> 4h` reads as "drain projected to take >4h".

## Cardinality

For 50 (topic, consumergroup) pairs × 1 model × 1 level:

- forecast lines: 150
- threshold ETA: 100
- will-breach lines: 100
- drain ETA: 100 (le_drain emits to a parallel family)

## Alerts

```promql
# Backlog projected to breach 100k within 1h — investigate the consumer.
kafka_consumer_lag_forecast_will_breach{severity="warning"} == 1
```

```promql
# Queue is not actually draining.
kafka_consumer_lag_forecast_time_to_drain_seconds{name="drained"} == +Inf
for 30m
```

```promql
# Drain projected to take >4h.
kafka_consumer_lag_forecast_time_to_drain_seconds{name="drained"} > 4 * 60 * 60
and
kafka_consumer_lag_forecast_time_to_drain_seconds{name="drained"} != +Inf
```
