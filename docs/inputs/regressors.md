# Exogenous regressors

A regressor is an extra input fed to the forecasting model alongside the primary series. Three flavours are supported:

- **`calendar`** — deterministic features derived from timestamps (`hour_of_day`, `day_of_week`, `is_weekend`).
- **`custom`** — any PromQL query whose result is aligned to the primary series.
- **`holidays`** — public-holiday calendar from the [`holidays`](https://pypi.org/project/holidays/) package (install with `pip install 'promforecast[holidays]'`).

Regressors are declared per query under `regressors:`. Built-in models ignore them today; plug-in models (e.g. `promforecast-prophet`) receive them via the `regressors` argument to `fit_predict`.

## Calendar regressors

```yaml
queries:
  - id: requests_per_second
    promql: rate(http_requests_total[5m])
    regressors:
      - id: hour_of_day
        type: calendar
      - id: day_of_week
        type: calendar
      - id: is_weekend
        type: calendar
```

The `id` must be one of the three supported names — the schema rejects anything else at boot.

## Custom (PromQL) regressors

```yaml
regressors:
  - id: deploys
    type: custom
    promql: changes(kube_deployment_status_observed_generation[1h])
    required: false          # default
```

- The PromQL is executed once per primary query and aligned via nearest-asof join.
- `required: true` makes a failure fatal for the whole series.
- Failures are counted in `forecast_regressor_failures_total{group, query, regressor, reason}`.

## Holiday regressors

Requires the `holidays` extra.

### Binary mode (default)

```yaml
regressors:
  - id: is_holiday
    type: holidays
    country: US                  # ISO-3166-1 alpha-2
    regions: []                  # country-wide by default
    timezone: America/New_York   # optional, defaults to UTC
    expand: false                # default
```

Emits a single binary column `is_holiday` (1.0 on any holiday, 0.0 otherwise).

### One-hot mode

```yaml
regressors:
  - id: holiday
    type: holidays
    country: US
    regions: [CA, NY]            # union of calendars
    expand: true
```

Emits one column per observed holiday name (`holiday__thanksgiving`, `holiday__independence_day`, …).  
If the lookback window contains no holidays, falls back to a single binary column so the feature schema stays stable.

### Timezone & regions

- `timezone` (IANA, e.g. `America/New_York`) converts UTC timestamps to local dates. Defaults to `UTC`.
- `regions` takes a list of subdivisions (states/provinces); the runner unions their calendars.

Supported countries/regions are whatever the installed `holidays>=0.50` package provides (≥150 countries + most major subdivisions).

## Failure handling

| Failure type                        | Reason label      | `required: true` behaviour          |
|-------------------------------------|-------------------|-------------------------------------|
| Invalid calendar id                 | `invalid_type`    | Series skipped                      |
| Custom PromQL error / timeout       | `query_error` / `query_timeout` | Entire series skipped          |
| Custom PromQL returns no data       | `empty_result`    | Entire series skipped               |
| No datasource (dry-run / test)      | `no_source`       | Entire series skipped               |
| `holidays` package missing          | `invalid_type`    | Series skipped                      |
| Unknown country / region            | `invalid_type`    | Series skipped                      |

Optional regressors (default) only increment the failure counter; the forecast still emits without that feature.

## Contribution metric

When any regressor contributes signal, the runner emits a normalised explainability share:

```
<id>_forecast_contribution_ratio{regressor, ...}   # values in [0, 1], sum to 1 per series
```

Disable via `defaults.emission.contribution: false` (see [`../reference/config-schema.md`](../reference/config-schema.md)).

## Conditional forecasts

The default forecast answers: "Is the actual inside the historic envelope?"
That works for most cases, but breaks when a known regressor legitimately shifts the metric (e.g. a deploy that *should* increase request rate).

**Conditional forecasts** answer: "Given the *current* state of the regressors, what should the metric look like now?"
Use the conditional deviation when you want to detect anomalies *after accounting for known external drivers*.

### Configuration

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

### Emitted metrics

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

### Model behaviour

Built-in models (`AutoARIMA`, `AutoETS`, `MSTL`, `SeasonalNaive`, etc.) do **not** use regressors.
They advertise `uses_regressors = False`, so the runner **short-circuits** the conditional refit and aliases the unconditional forecast.

You still get a stable schema (conditional metrics always exist), but at **zero extra CPU cost**.

Regressor-aware models (e.g. Prophet plug-in) perform a real second fit with held-constant regressors.

### Conditional-band conformal calibration

When `accuracy.conformal.enabled: true` *and* `emission.conditional: true` are both on for a query, the conditional band carries its own empirical-coverage guarantee — derived from a dedicated calibration fit where regressors are held flat at the final pre-holdout value, matching the production-time conditional inference contract.

The per-(model, level) widening applied to the conditional band lands on `/metrics` as `forecast_conditional_band_calibration_offset{id, group, model, level}` (gauge, same units as the underlying metric), sibling of the unconditional `forecast_band_calibration_offset`. Comparing the two reveals whether the conditional band needs different widening than the unconditional one — typically yes, because the unconditional residuals were scored on a holdout where regressors *did* vary.

Per-(model) calibration failures (degenerate holdout, fit timeout, fit error) fall back to the inherited unconditional offsets and bump `forecast_conditional_calibration_failures_total{group, query, reason}` with `reason` one of `degenerate_holdout`, `fit_timeout`, `fit_error`. The fallback path is no worse than the inherited-offset baseline, but a sustained tick rate signals fresh calibration is silently not running.

Models that advertise `uses_regressors = False` skip the extra calibration fit entirely — the conditional fit would produce the same numbers as the unconditional fit, so emitting two parallel offset gauges with identical values would just be noise. This means the gauge family appears only when at least one emitted model is regressor-aware.

### Cardinality

Roughly doubles the forecast snapshot footprint for opted-in queries.
`promforecast validate` includes the extra lines in its estimate.

See [`../operations/cardinality.md`](../operations/cardinality.md) for the full multiplier table.

### When not to use it

- No regressors on the query (conditional becomes a pure duplicate).
- No model in the set uses regressors (wastes cardinality).
- You want classic anomaly detection that treats regressor shifts as real deviations.
