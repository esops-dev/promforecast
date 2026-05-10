# Models

promforecast ships a small set of statistical forecasters from the
[`statsforecast`](https://github.com/Nixtla/statsforecast) library, plus a
`SeasonalNaive` baseline.

Configure models via the `models:` array (in `defaults` or per group).  
- **Explicit mode** (`accuracy.auto_select: false`): every chosen model runs and emits forecasts side-by-side.  
- **Auto-select mode** (`accuracy.auto_select: true`): only the lowest-MAPE winner is kept (see [Auto-select with fallback](#auto-select-with-fallback) below).

External models can be added via the `promforecast.models` entry-point group (see [Extending the registry](#extending-the-registry)).

## Built-in models

| Model          | Description                                                                 | Best for                                      | Avoid when                                      | Fit time                          |
|----------------|-----------------------------------------------------------------------------|-----------------------------------------------|-------------------------------------------------|-----------------------------------|
| **SeasonalNaive** | Repeats the value from `season_length` steps ago. No fitting, no parameters. | Sanity-check baseline or fallback             | Series with trends that cross the seasonal cycle | Negligible                       |
| **AutoARIMA**  | Automatic ARIMA with trend, seasonality, and noise handling.               | Noisy infrastructure metrics (default choice) | Hundreds of series where speed matters          | Seconds per series (slowest)     |
| **AutoETS**    | Exponential smoothing with automatic error/trend/seasonality selection.    | Clean, smooth seasonal metrics                | Zero-heavy series, sharp regime changes, heavy tails | Faster than AutoARIMA            |
| **MSTL**       | Multiple Seasonal-Trend decomposition (LOESS) + trend forecaster on remainder. | Strong, regular seasonality                   | Short series (< 2 full cycles) or no seasonality | Comparable to AutoETS            |

## Multi-horizon accuracy

When `accuracy.evaluate: true` (default), every run also performs a backtest and emits accuracy metrics:

```
forecast_accuracy_mape{id, group, model, horizon, ...}
forecast_accuracy_mase{id, group, model, horizon, ...}
```

`horizon` values come from `accuracy.horizons` (default `[1h, 6h, 24h]`). The backtest uses a single hold-out window (the most recent `max(accuracy.horizons)` of the lookback) and slices the forecast accordingly.

## Cost model (per series)

- `accuracy.evaluate: false` + `accuracy.auto_select: false` + `emission.{deviation,quality}: false`: **N** fits (one per model) — cheapest option.
- `accuracy.evaluate: true` + `accuracy.auto_select: false`: **2N** fits (backtest + final forecast per model).
- `accuracy.auto_select: true`: **N + 1** fits (backtest per model + one final forecast for the winner).

Backtest and forecast fits run concurrently in the executor pool, so wall-clock time per series is determined by the slowest individual fit. The backtest is run whenever any of `accuracy.evaluate`, `accuracy.auto_select`, `emission.deviation`, or `emission.quality` is on; turn all four off to skip backtesting entirely.

## Auto-select with fallback

Set `accuracy.auto_select: true` to run a “horse race” and emit only the winner:

- Winner = lowest mean MAPE across the configured models.
- Labels on the final forecast: `model="<winner>"`, `best_model="<winner>"`, `model_fallback="false"`.
- If the winner’s MAPE exceeds `accuracy.fallback_mape_threshold` (default `50`), it is replaced by `accuracy.fallback_model` (default `SeasonalNaive`). Labels become `model="<fallback>"`, `best_model="<original winner>"`, `model_fallback="true"`.

Set `accuracy.fallback_mape_threshold: null` to disable the fallback check.  
`accuracy.auto_select: ensemble` is reserved for a future release (currently treated as `false` and logs a warning).

## Extending the registry

To add a third-party model:

1. Implement the `ForecastModel` protocol in `forecaster/src/promforecast/models/<name>.py`.
2. Register it under the `promforecast.models` entry point in `pyproject.toml`.
3. Document it here.

CI automatically verifies all registered models.
