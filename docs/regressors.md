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

Disable via `defaults.emission.contribution: false` (see `config-schema.md`).
