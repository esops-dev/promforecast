# Recipe: `kube_deployment_status_replicas`

> Predict autoscaler ramp before it surprises capacity.

## The metric

`kube_deployment_status_replicas` is the current replica count per deployment. The forecastable shape is the daily cycle of HPA-driven scaling — deployments breathe in and out across the day, and predicting the next peak lets capacity planning keep up.

## The config

```yaml
groups:
  - name: kube_workloads
    queries:
      - id: kube_deployment_status_replicas
        promql: |
          sum by (namespace, deployment) (
            kube_deployment_status_replicas
          )
        step: 1m
        data_profile: count
```

`data_profile: count` enables count-aware emission safeguards (`clip_non_negative` and `round_to_integer` default to on) so predicted replica counts are whole integers and never negative.

## The model

* **`SeasonalNaive`** is surprisingly good — the pattern is largely “same shape as last week”.
* **`AutoARIMA`** rides along as the auto-select alternative for deployments with non-stationary ramp (slow week-over-week growth).

## Expected forecast shape

- Step-function pattern within the day — replicas land on whole integers.
- Confidence band typically narrow at level transitions, wider during transition windows.

## Cardinality

For 200 deployments × 1 model × 1 level:

- forecast lines: 600
- accuracy lines: 400
- deviation lines: 200

## Alerts

```promql
# Predicted to need >50 replicas in 24h — pre-warm capacity now.
kube_deployment_status_replicas_forecast{horizon="24h"} > 50
```

```promql
# Predicted replica count drifting unexpectedly from the historical pattern.
forecast_drift_score{id="kube_deployment_status_replicas"} > 0.4
```
