# Models

promforecast ships a small set of statistical forecasters from the
[`statsforecast`](https://github.com/Nixtla/statsforecast) library, plus a
`SeasonalNaive` baseline.

Configure models via the `models:` array (under `defaults` or per group).  
- **Explicit mode** (`accuracy.auto_select: false`): every chosen model runs and emits forecasts side-by-side.  
- **Auto-select mode** (`accuracy.auto_select: true`): only the lowest-MAPE winner is kept (see [Auto-select with fallback](#auto-select-with-fallback)).  
- **Ensemble mode** (`accuracy.auto_select: ensemble`): every model runs *and* a weighted combination is emitted (see [Ensemble auto-select](#ensemble-auto-select)).

External models can be added via the `promforecast.models` entry-point group (see [Extending the registry](#extending-the-registry)).

## Built-in models

| Model          | Description                                                                 | Best for                                      | Avoid when                                      | Fit time                          |
|----------------|-----------------------------------------------------------------------------|-----------------------------------------------|-------------------------------------------------|-----------------------------------|
| **SeasonalNaive** | Repeats the value from `season_length` steps ago. No fitting, no parameters. | Sanity-check baseline or fallback             | Series with trends that cross the seasonal cycle | Negligible                       |
| **AutoARIMA**  | Automatic ARIMA with trend, seasonality, and noise handling.               | Noisy infrastructure metrics (default choice) | Hundreds of series where speed matters          | Seconds per series (slowest)     |
| **AutoETS**    | Exponential smoothing with automatic error/trend/seasonality selection.    | Clean, smooth seasonal metrics                | Zero-heavy series, sharp regime changes, heavy tails | Faster than AutoARIMA            |
| **MSTL**       | Multiple Seasonal-Trend decomposition (LOESS) + trend forecaster on remainder. | Strong, regular seasonality                   | Short series (< 2 full cycles) or no seasonality | Comparable to AutoETS            |

## Multi-horizon accuracy

When `accuracy.evaluate: true` (default), every run also performs a backtest and emits:

```
forecast_accuracy_mape{id, group, model, horizon, ...}
forecast_accuracy_mase{id, group, model, horizon, ...}
```

`horizon` values come from `accuracy.horizons` (default `[1h, 6h, 24h]`). The backtest uses a single hold-out window — the most recent `max(accuracy.horizons)` of the lookback — and slices the forecast accordingly.

## Cost model (per series)

- `accuracy.evaluate: false` + `accuracy.auto_select: false` + `emission.{deviation,quality}: false`: **N** fits (one per model) — cheapest.  
- `accuracy.evaluate: true` + `accuracy.auto_select: false`: **2N** fits (backtest + final forecast per model).  
- `accuracy.auto_select: true` **or** any emission enabled: **N + 1** fits (backtest per model + one final forecast for the winner).

Backtest and forecast fits run concurrently in the executor pool, so wall-clock time per series equals the slowest individual fit. The backtest runs whenever *any* of `accuracy.evaluate`, `accuracy.auto_select`, `emission.deviation`, or `emission.quality` is enabled.

## Auto-select with fallback

Set `accuracy.auto_select: true` to run a horse race and emit only the winner:

- Winner = lowest mean MAPE across the configured models.  
- Final forecast labels: `model="<winner>"`, `best_model="<winner>"`, `model_fallback="false"`.  
- If the winner’s MAPE exceeds `accuracy.fallback_mape_threshold` (default `50`), it is replaced by `accuracy.fallback_model` (default `SeasonalNaive`). Labels become `model="<fallback>"`, `best_model="<original winner>"`, `model_fallback="true"`.

Set `accuracy.fallback_mape_threshold: null` to disable the check.

## Ensemble auto-select

Set `accuracy.auto_select: ensemble` to combine every configured model into a single weighted forecast. Weights are inverse-MAPE (lowest backtest MAPE = largest weight).

Output:

- Component models are emitted exactly as in explicit mode (`model="AutoARIMA"`, `model="SeasonalNaive"`, …).  
- A synthetic `model="ensemble"` row is emitted alongside, carrying the weighted combination.  
- Weights are exposed via `forecast_ensemble_weight{id, group, model, ...}` for dashboards.

Failure handling:  
- Failed components are excluded from the ensemble.  
- If *every* component MAPE is non-finite, equal weights are used as a last resort.  
- The ensemble row has no backtest data, so `forecast_quality_score{model="ensemble"}` and `forecast_deviation_*{model="ensemble"}` are omitted. Use the per-component metrics instead.

## Per-model hyperparameter overrides

Most users never need this — the `Auto*` wrappers exist so you don’t have to tune.  
If you want to lock in a specific configuration, every entry in `models:` accepts an optional `params:` dict:

```yaml
defaults:
  models:
    - SeasonalNaive                              # plain string shorthand still works
    - name: AutoARIMA
      params:
        max_p: 5
        max_q: 5
        stepwise: false
    - name: AutoETS
      params:
        model: "ZAA"
```

Bare strings are equivalent to `{ name: "<model>", params: {} }`.  
Supported keys are whatever the underlying `statsforecast` model accepts. Unknown keys surface as fit-time errors tagged `fit_error` in `forecast_failures_total`.

## Exogenous regressors

Per-query `regressors:` lets you feed external drivers into a forecast. Two types are supported:

```yaml
groups:
  - name: traffic_capacity
    queries:
      - id: requests_per_second
        promql: rate(http_requests_total[5m])
        regressors:
          - id: hour_of_day
            type: calendar
          - id: deploys
            type: custom
            promql: changes(kube_deployment_status_observed_generation[1h])
            required: false   # default; failures degrade gracefully
```

| Type       | Source                              | Failure behavior                              |
|------------|-------------------------------------|-----------------------------------------------|
| `calendar` | Derived from timestamps (`hour_of_day`, `day_of_week`, `is_weekend`) | Never fails                                   |
| `custom`   | Own PromQL query (mean-aggregated)  | Optional → dropped; `required: true` → whole series skipped |

Failures increment `forecast_regressor_failures_total{group, query, regressor, reason}` (optional) or `forecast_failures_total{reason="required_regressor"}` (required).

The built-in `statsforecast` wrappers currently ignore regressors, but the pipeline is ready for third-party models (`promforecast-prophet`, etc.).

## remote_write sink

By default the forecaster only exposes the *next* forecast row via `/metrics`.  
For historical forecast curves (the kind Grafana draws continuously) enable `remote_write`:

```yaml
sink:
  remote_write:
    enabled: true
    url: http://victoriametrics:8428/api/v1/import/prometheus
    timeout: 30s
    headers:
      X-Forecaster: promforecast
```

After every successful run the full forecast curve is pushed as Prometheus text format with explicit millisecond timestamps.  
VictoriaMetrics requires `-futureRetention=2d` (or higher than your `defaults.horizon`) to accept future-dated samples.

The body is chunked (`chunk_size` defaults to 10000 rows) to avoid huge requests.  
A chunk failure aborts the rest of the push for that run; the next cycle re-sends everything.  
`/metrics` continues to expose the latest snapshot. Push failures are tracked in `forecast_remote_write_failures_total{group, reason}`.

## Extending the registry

To add a third-party model:

1. Implement the `ForecastModel` protocol in `forecaster/src/promforecast/models/<name>.py`.
2. Register it under the `promforecast.models` entry point in `pyproject.toml`.
3. Document it here.

CI automatically verifies all registered models.
