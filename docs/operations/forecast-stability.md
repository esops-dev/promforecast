# Forecast stability and model agreement

Two gauges catch a thrashing model or a set of models that disagree about the future **before** downstream alerts react.

| Metric | Question it answers |
|--------|---------------------|
| `forecast_revision_magnitude{id, group, model, horizon}` | How much did this model’s `<horizon>` forecast move since the last fit? |
| `forecast_model_disagreement{id, group, horizon}` | How much do the configured models disagree about this `<horizon>` forecast? (explicit mode only) |

## Revision magnitude

The forecaster caches the previous central `yhat` value at each accuracy horizon. On the next fit it emits:

```
forecast_revision_magnitude = |new − old| / max(|old|, 1e-9)
```

A 30 % shift becomes `0.3`. The metric is **unsigned** — operators care about “the forecast moved a lot”, not the direction.

Shipped example alert:

```yaml
- alert: ForecastUnstable
  expr: forecast_revision_magnitude > 0.3
  for: 20m
  labels:
    severity: info
```

**Note:** this is distinct from `forecast_drift_score`, which measures **input** drift (how much the underlying signal moved). Revision measures **output** drift (how much the model’s prediction moved). The two often correlate but can diverge — a stable input that suddenly produces wildly different forecasts usually signals a model over-reacting to noise or a real regime change.

On the very first fit per (series, model, horizon) the value is `NaN` and the row is dropped. Steady-state behaviour starts after two refresh cycles.

## Model disagreement

In explicit mode (`auto_select: false` with 2+ models) each model produces its own forecast. The disagreement gauge is the **coefficient of variation** across the per-model central forecasts at the same horizon:

```
forecast_model_disagreement{horizon="24h"} = CoV(yhat_24h across models)
```

`0.0` = perfect agreement. Higher values indicate no single model is confident — a useful “should I switch to `auto_select`?” signal.

Shipped example alert:

```yaml
- alert: ForecastModelsDisagree
  expr: forecast_model_disagreement > 0.4
  for: 30m
  labels:
    severity: info
```

Suppressed in `auto_select` and ensemble modes because only one model’s `yhat` is emitted per series.

## Cardinality

- `forecast_revision_magnitude`: `N_series × N_emitted_models × N_horizons` per query with `accuracy.evaluate: true`.
- `forecast_model_disagreement`: `N_series × N_horizons` per query in explicit mode with 2+ models.

`promforecast validate` counts both families against the global budget (see [cardinality.md](cardinality.md)).
