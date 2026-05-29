# Canary deploy forecast comparison

When you run a canary deploy (a small slice of traffic to a new version), you want to know whether the canary is behaving like the stable fleet *would have* under the same conditions.  

promforecast can model that: it fits the **baseline** pods’ history, projects what the canary *should* look like, and surfaces the divergence between the canary’s actuals and that baseline-predicted forecast as a clear, alertable signal.

This turns canary analysis into a data-driven comparison against a quality forecast, rather than manual eyeballing or hand-rolled thresholds.

> **Note:** this is distinct from [canary / weighted sampling](outputs/canary.md), which fits only a fraction of a group’s series to save CPU.

## Configuring

Add a `canary:` block to a query. The selectors partition the already-fetched series into canary and baseline populations:

```yaml
groups:
  - name: checkout
    queries:
      - id: http_requests
        promql: sum by (version, pod) (rate(http_requests_total[5m]))
        canary:
          enabled: true
          canary_selector: '{version="canary"}'
          baseline_selector: '{version!="canary"}'
```

- `canary_selector` / `baseline_selector` are Prometheus-style label matchers (`=` and `!=` only). Regex (`=~`) is rejected at load time.
- A series matching neither selector is ignored; matching both is a config error.

## Emitted metrics

For each canary series, the forecaster emits (in the same shape as existing deviation metrics):

```
<id>_canary_forecast_deviation{deployment, replica, version, model, level, ...}
<id>_canary_forecast_deviation_outside_band{...}
```

- `<id>_canary_forecast_deviation` — signed normalised distance of the canary actual from the centre of the baseline-predicted band: `(actual - yhat) / (yhat_upper - yhat_lower)`.
- `<id>_canary_forecast_deviation_outside_band` — `1` when the canary actual is outside the baseline band, else `0`.

The labels preserve the canary series’ own labels plus the `model` that produced the baseline forecast and the confidence `level`.

## Use with rollout controllers

The emission shape is directly compatible with Argo Rollouts and Flagger.

**Argo Rollouts** (`AnalysisTemplate`):

```yaml
metrics:
  - name: canary-divergence
    successCondition: result[0] <= 0.5
    provider:
      prometheus:
        address: http://prometheus.monitoring:9090
        query: max(http_requests_canary_forecast_deviation{version="canary"})
```

**Flagger** (`MetricTemplate`):

```yaml
spec:
  provider:
    type: prometheus
    address: http://prometheus.monitoring:9090
  query: max(http_requests_canary_forecast_deviation_outside_band{version="canary"})
```

Gate the rollout on the deviation staying inside the band (or the ratio staying below an operator-chosen magnitude).

## Trade-off: baseline history required

Canary analysis depends on the baseline pods having stable history to fit on. For a brand-new service with no baseline, fall back to [cold-start sibling borrowing](inputs/cold-start.md) until the stable fleet has enough history.

If either population is empty or the baseline is too short to fit, no comparison is emitted that tick — the normal forecast path is unaffected.
