# Deploy risk score

`forecast_deploy_risk_score` turns multiple golden-signal deviation ratios into a single deploy-anchored risk number.  
Instead of manually correlating latency, errors, saturation, and traffic during a rollout, you get one gauge per active deploy that SREs can alert on.

The metric only exists while a deploy is inside its **active window** — background noise never pollutes dashboards.

## Configuration

```yaml
deploy_signals:
  - service: payments-api
    anchor:
      type: regressor
      query: payments_request_rate
      regressor_id: deploys
      threshold: 0          # transition from <=0 to >0
    signals:
      - query: payments_request_rate
        weight: 0.5
      - query: payments_p95_latency
        weight: 1.0
      - query: payments_error_rate
        weight: 2.0
    active_window: 30m
    max_active_deploys: 8
```

| Field            | Default | Meaning |
|------------------|---------|---------|
| `service`        | —       | Human-readable service name |
| `anchor`         | —       | Detects deploy start (via regressor transition) |
| `signals`        | —       | Weighted golden signals (`errors ≥ latency ≥ saturation ≥ traffic` is conventional) |
| `active_window`  | —       | How long the score stays emitted after detection |
| `max_active_deploys` | `8` | LRU cap per service to bound cardinality |

## Emitted metric

```
forecast_deploy_risk_score{
  service="payments-api",
  deployment_id="payments-api-1715000000",
  contributing_signals="3"
} 0.42
```

- **Value** — weighted mean of absolute `forecast_deviation_ratio` across contributing signals (higher = more risk).
- `deployment_id` — stable identifier for the current deploy.
- `contributing_signals` — number of signals that produced a finite deviation this cycle.

## Anchor detection

The anchor watches the configured regressor on the anchor query.  
A deploy is registered when the leading value crosses `threshold` from ≤ threshold to > threshold.  
A persistently high regressor counts as **one** deploy, not repeated triggers.

Typical anchor regressor:

```yaml
regressors:
  - id: deploys
    promql: changes(kube_deployment_status_observed_generation{deployment="payments-api"}[5m])
```

## Worked examples

**Argo Rollouts**  
Expose rollout phase changes as a counter and reference it as the anchor.  
Argo’s analysis backend can read `forecast_deploy_risk_score` directly and mark the rollout `Failed` if the score exceeds your threshold.

**Flagger**  
Define a `MetricTemplate` that queries the risk score and attach it to your Canary resource.  
A sustained high score triggers automatic rollback.

## Cardinality

At most `max_active_deploys` rows per service at any time.  
With 10 services the worst-case total is ~80 rows — safely bounded by LRU eviction.

## Example alerts

```promql
# High risk during rollout
forecast_deploy_risk_score{contributing_signals!~"0|1"} > 0.5

# Critical risk
forecast_deploy_risk_score{contributing_signals!~"0|1"} > 1.0
```

## Operational notes

- With HA enabled, only the leader computes scores (followers serve the cached snapshot).
- Config reloads reset the tracker; active deploys disappear until re-detected.
- Score is computed from the current snapshot’s deviation rows — a deploy registered before the first signal fit will temporarily show `contributing_signals: 0`.

## See also

- [Deviation metrics](deviation.md) — the signals the score is built from
- [Drift telemetry](drift.md) — distinct from deploy-anchored risk
- [Canary / weighted groups](canary.md) — single-metric alternative when you have only one golden signal
