# Forecast explainability

Two related questions get two different metrics:

* "Which **regressor** is moving the forecast?" → [`<id>_forecast_contribution_ratio`](../inputs/regressors.md)
* "Which **model component** is driving the curve?" → the decomposition family below

The decomposition projects the forecast onto `trend / seasonal / level / residual` so you can read “disk fills in 14 days because the trend steepened last week” directly from a stacked-area Grafana panel.

## Configuration

Enabled per query, off by default:

```yaml
queries:
  - id: node_filesystem_avail_bytes
    promql: node_filesystem_avail_bytes{fstype!~"tmpfs|overlay|squashfs"}
    emission:
      decompose: true
```

**Output** (one gauge per component at end-of-horizon):

```
<id>_forecast_component{component="trend", group, model, ...} <value>
<id>_forecast_component{component="seasonal", group, model, ...} <value>
<id>_forecast_component{component="level", group, model, ...} <value>
<id>_forecast_component{component="residual", group, model, ...} <value>
```

The four components always sum to `yhat` (within floating-point tolerance) for each series/model.

## Component meanings

| Component  | What it captures |
|------------|------------------|
| **level**  | In-sample baseline (constant across horizon) |
| **trend**  | Linear slope across the horizon (anchored at 0 on first step) |
| **seasonal** | Repeating deviation within the seasonal cycle |
| **residual** | Remaining structure + model-specific effects |

## Non-decomposable models

Intermittent-demand models (`Croston`, `CrostonSBA`, `ADIDA`, `IMAPA`) do not support decomposition. Setting `emission.decompose: true` on queries using them increments:

```
forecast_decomposition_skipped_total{group, query, model, reason="unsupported_model"}
```

Other `reason` values: `fit_failed`, `components_unavailable`, `invariant_violation`.

## Cardinality

Adds **4×** samples per (series, model) that has decomposition enabled.  
`promforecast validate` includes this in the cardinality estimate.

## When not to use it

- On `data_profile: intermittent` series — components lack meaning.
- On already-understood signals (e.g. clean sinusoidal patterns).
- When dashboard space or cardinality budget is tight and no stacked-area panel will consume the data.
