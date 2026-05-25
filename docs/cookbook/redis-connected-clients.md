# Recipe: `redis_connected_clients`

> Forecast Redis pool exhaustion hours before `maxclients` trips.

## The metric

`redis_connected_clients` is the current connection count per Redis instance. The classic leak pattern: gradual accumulation over days, followed by a sudden outage when `maxclients` (typically 10k) is hit.

## The config

```yaml
groups:
  - name: redis
    queries:
      - id: redis_connected_clients
        promql: |
          max by (instance) (redis_connected_clients)
        thresholds:
          - name: capacity
            value: 9000              # tune against your maxclients
            comparator: gt
            alert_within: 4h
            severity: critical
            action_cost:
              type: priority
              value: 100
```

## The model

* **`AutoETSDamped`** (with `damping_parameter: 0.98`) — connection growth is steady but bounded. Damping prevents linear over-extrapolation on long horizons.
* `SeasonalNaive` as fallback when connections show strong daily cycling (aggressive pool recycling).

## Expected forecast shape

- Slow upward drift with a slight daily cycle.
- When a leak is present, `yhat_upper` crosses the threshold hours-to-days before the central forecast.

## Cardinality

For 20 Redis instances × 1 model × 1 level:

- forecast lines: 60
- threshold ETA: 60
- will-breach lines: 60
- action cost lines: 1

## Alerts

```promql
# Predicted to hit 90 % of maxclients within 4 h — investigate or pre-emptively recycle.
redis_connected_clients_forecast_will_breach{severity="critical", level="95"} == 1
```

```promql
# Leak rate looks unusually fast — surface connection-accumulation rate and gate on a sharp uptick.
redis_connected_clients_forecast - redis_connected_clients_forecast offset 24h > 500
```
