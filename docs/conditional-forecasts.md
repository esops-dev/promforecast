# Conditional forecasts

The default forecast answers: “Is the actual inside the historic envelope?”  
That works for most cases, but breaks when a known regressor legitimately shifts the metric (e.g. a deploy that *should* increase request rate).

**Conditional forecasts** answer: “Given the *current* state of the regressors, what should the metric look like now?”  
Use the conditional deviation when you want to detect anomalies *after accounting for known external drivers*.

## Configuration

Per-query, off by default:

```yaml
groups:
  - name: http
    queries:
      - id: http_requests_rate
        promql: sum(rate(http_requests_total[1m])) by (endpoint)
        regressors:
          - id: deploys
            promql: changes(kube_deployment_status_observed_generation[1h])
        emission:
          conditional: true
```

The runner re-fits each emitted model while holding the most recent regressor values constant.

## Emitted metrics

| Metric                                      | Description |
|---------------------------------------------|-----------|
| `<id>_forecast_conditional`                 | Point forecast at horizon |
| `<id>_forecast_conditional_lower{level}`    | Lower band per level |
| `<id>_forecast_conditional_upper{level}`    | Upper band per level |
| `forecast_conditional_deviation_ratio`      | Signed deviation of latest actual |
| `forecast_conditional_deviation_outside_band` | 1 if actual is outside the conditional band |

These carry the same labels as the unconditional deviation family, so you can alert on them directly:

```promql
# Anomaly after accounting for known regressors
forecast_conditional_deviation_outside_band{level="95"} == 1
  and on (id, group, model) forecast_quality_score > 0.6
```

## Model behaviour

Built-in models (`AutoARIMA`, `AutoETS`, `MSTL`, `SeasonalNaive`, etc.) do **not** use regressors.  
They advertise `uses_regressors = False`, so the runner **short-circuits** the conditional refit and aliases the unconditional forecast.  

You still get a stable schema (conditional metrics always exist), but at **zero extra CPU cost**.  

Regressor-aware models (e.g. Prophet plug-in) perform a real second fit with held-constant regressors.

## Cardinality

Roughly doubles the forecast snapshot footprint for opted-in queries.  
`promforecast validate` includes the extra lines in its estimate.

See [operations/cardinality.md](operations/cardinality.md) for the full multiplier table.

## When not to use it

- No regressors on the query (conditional becomes a pure duplicate).
- No model in the set uses regressors (wastes cardinality).
- You want classic anomaly detection that treats regressor shifts as real deviations.
