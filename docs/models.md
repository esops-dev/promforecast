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

| Model               | Description                                                                                  | Best for                                      | Avoid when                                      | Fit time                          |
|---------------------|----------------------------------------------------------------------------------------------|-----------------------------------------------|-------------------------------------------------|-----------------------------------|
| **SeasonalNaive**   | Repeats the value from `season_length` steps ago. No fitting, no parameters.                 | Sanity-check baseline or fallback             | Series with trends that cross the seasonal cycle | Negligible                       |
| **AutoARIMA**       | Automatic ARIMA with trend, seasonality, and noise handling.                                | Noisy infrastructure metrics (default choice) | Hundreds of series where speed matters          | Seconds per series (slowest)     |
| **AutoARIMADamped** | AutoARIMA with a post-fit damped-trend bend.                                                | Capacity metrics over multi-day horizons       | Series with no trend (damping is a no-op)        | Same as AutoARIMA                |
| **AutoETS**         | Exponential smoothing with automatic error/trend/seasonality selection.                     | Clean, smooth seasonal metrics                | Zero-heavy series, sharp regime changes, heavy tails | Faster than AutoARIMA            |
| **AutoETSDamped**   | AutoETS with a forced damped additive trend (native ETS damping + post-fit bend).            | Capacity metrics over multi-day horizons       | Series with no trend (damping is a no-op)        | Same as AutoETS                  |
| **MSTL**            | Multiple Seasonal-Trend decomposition (LOESS) + trend forecaster on remainder.              | Strong, regular seasonality                   | Short series (< 2 full cycles) or no seasonality | Comparable to AutoETS            |
| **Croston**         | Classical Croston decomposition (demand size × inter-demand interval).                       | Sparse / intermittent demand                  | Continuous series with strong seasonality        | Negligible                       |
| **CrostonSBA**      | Syntetos-Boylan bias-corrected Croston (recommended default).                               | Sparse / intermittent demand                  | Continuous series with strong seasonality        | Negligible                       |
| **ADIDA**           | Aggregate-Disaggregate Intermittent Demand Approach.                                        | Very sparse signals where Croston struggles   | Continuous series with strong seasonality        | Negligible                       |
| **IMAPA**           | Intermittent Multiple Aggregation Prediction Algorithm.                                     | Noisy intermittent signals                    | Continuous series with strong seasonality        | Negligible                       |

### When to choose damped

**Rule of thumb:** for any horizon longer than one full seasonal cycle, prefer the damped variant (`AutoARIMADamped` or `AutoETSDamped`). A 14-day disk-fill forecast with daily seasonality (`season_length=288` at 5 min step) is the classic case.

`damping_parameter` (default `0.98`) controls how quickly the trend bends. `1.0` = no damping.

**Extrapolation table** (linear growth of 1 unit/step):

| `damping_parameter` | h=1   | h=10  | h=24  | h=144 (12 h) | h=288 (24 h) | Asymptote |
|---------------------|-------|-------|-------|--------------|--------------|-----------|
| `1.00` (undamped)   | 1.00  | 10.00 | 24.00 | 144.00       | 288.00       | ∞         |
| `0.99`              | 0.99  | 9.47  | 21.22 | 75.71        | 93.52        | 99        |
| `0.98` (default)    | 0.98  | 8.96  | 18.83 | 46.33        | 48.85        | 49        |
| `0.95`              | 0.95  | 7.62  | 13.45 | 18.99        | 19.00        | 19        |
| `0.90`              | 0.90  | 5.86  | 8.28  | 9.00         | 9.00         | 9         |

See [`capacity.md`](capacity.md) for recommended defaults and decision flow.

### When your series is sparse — intermittent-demand models

AutoARIMA/AutoETS/MSTL assume a continuous signal. On series that are zero ≥80 % of the time they collapse to “always zero” or oscillating bands.

**Decision flow**:
1. Is the lookback ≥80 % zero? → treat as intermittent.
2. Prefer **CrostonSBA** (bias-corrected; use plain Croston only with a specific reason).
3. If inter-demand interval is extremely sparse → **ADIDA**.
4. If aggregation bucket isn’t obvious → **IMAPA** (averages ADIDA over buckets).

Set `data_profile: intermittent` on the query to:
- Switch auto-selector from MAPE to MASE (MAPE is undefined on zeros).
- Prefer intermittent wrappers when present.
- Pair with `forecast_accuracy_mase > 1.0` alerts (see `examples/alerts/promforecast-rules.yaml`).

Intermittent models ignore `season_length` and `regressors`. Confidence bands come from `statsforecast` bootstrap (enabled automatically when `confidence_levels:` is set). Tune via `params.conformal_windows`.

**Worked example** (sparse error counter, 60-step lookback, 12-step hold-out):

| Model        | Forecast at horizon | Band      | MAPE | MASE |
|--------------|---------------------|-----------|------|------|
| `AutoARIMA`  | ≈ 0.0               | ±wide     | NaN  | 1.0  |
| `CrostonSBA` | ≈ 0.10              | ±narrow   | NaN  | 0.6  |
| `ADIDA`      | ≈ 0.08              | ±narrow   | NaN  | 0.6  |

## Count-aware emission safeguards

For non-negative/integer metrics (counts, queue depths) add:

```yaml
data_profile: count
emission:
  clip_non_negative: true   # default when data_profile=count
  round_to_integer: true    # default when data_profile=count
```

- `clip_non_negative` clamps forecasts/bands at zero (`forecast_emission_clipped_total{kind="negative"}`).
- `round_to_integer` rounds to nearest integer (`forecast_emission_clipped_total{kind="rounded"}` only when fractional part ≥0.05).

Applies to both `/metrics` and `remote_write`.

## Cold-start seasonality borrowing

See [`cold-start.md`](cold-start.md). Short version: when history < `min_points`, set `cold_start.borrow_from.promql` to synthesise a `SeasonalNaive`-style forecast from peer siblings. Emits with `model="ColdStart"` and `cold_start="true"`. Quality score is capped (default 0.5).

## Multi-horizon accuracy

When `accuracy.evaluate: true` (default), every run also performs a backtest and emits:

```
forecast_accuracy_mape{id, group, model, horizon, ...}
forecast_accuracy_mase{id, group, model, horizon, ...}
```

`horizon` values come from `accuracy.horizons` (default `[1h, 6h, 24h]`). Single hold-out window = most recent `max(accuracy.horizons)` of the lookback.

## Cost model (per series)

- `accuracy.evaluate: false` + `accuracy.auto_select: false` + `emission.{deviation,quality}: false`: **N** fits (cheapest).  
- `accuracy.evaluate: true` + `accuracy.auto_select: false`: **2N** fits.  
- `accuracy.auto_select: true` **or** any emission enabled: **N + 1** fits.

Backtest + forecast fits run concurrently; wall time = slowest fit.

## Auto-select with fallback

Set `accuracy.auto_select: true`:

- Winner = lowest mean MAPE.  
- Labels: `model="<winner>"`, `best_model="<winner>"`, `model_fallback="false"`.  
- If winner MAPE > `accuracy.fallback_mape_threshold` (default 50), replace with `accuracy.fallback_model` (default `SeasonalNaive`).  

Set threshold to `null` to disable.

## Ensemble auto-select

`accuracy.auto_select: ensemble` emits every component **plus** a weighted `model="ensemble"` row (inverse-MAPE weights).  
Weights exposed via `forecast_ensemble_weight{...}`.  
Failed components are dropped; all-non-finite → equal weights. Ensemble row skips backtest/quality/deviation metrics.

## Per-model hyperparameter overrides

```yaml
defaults:
  models:
    - SeasonalNaive
    - name: AutoARIMA
      params:
        max_p: 5
        max_q: 5
        stepwise: false
```

Bare strings = `{name: "...", params: {}}`. Unknown keys surface as `fit_error`.

## Multi-frequency support

Override `step` / `season_length` per query:

```yaml
- id: hourly_business_kpi
  promql: business_kpi
  step: 1h
  season_length: 24          # or auto
```

`season_length: auto` derives daily (`86400 / step_seconds`) or weekly (`7`) fallback.

## Exogenous regressors

```yaml
regressors:
  - id: hour_of_day
    type: calendar
  - id: deploys
    type: custom
    promql: changes(...)
    required: false
```

| Type       | Source                              | Failure behavior                              |
|------------|-------------------------------------|-----------------------------------------------|
| `calendar` | Timestamps (`hour_of_day`, …)       | Never fails                                   |
| `custom`   | Own PromQL (mean-aggregated)        | Optional → dropped; required → series skipped |

When regressors are present, also emits normalised contribution:

```
<id>_forecast_contribution_ratio{regressor="...", ...}
```

Disable via `defaults.emission.contribution: false`.

## remote_write sink

```yaml
sink:
  remote_write:
    enabled: true
    url: http://victoriametrics:8428/api/v1/import/prometheus
    timeout: 30s
```

Full forecast curve is pushed after every run (Prometheus text format + ms timestamps).  
VictoriaMetrics needs `-futureRetention=2d` (or higher than `defaults.horizon`).

Body is chunked (default 10 000 rows). Each chunk retries up to 3× with backoff on transient errors; 4xx fails fast.  
Failures tracked in `forecast_remote_write_failures_total` and `forecast_remote_write_retries_total`.  
Pool sizing via `max_connections` / `max_keepalive_connections`.

`/metrics` always shows the latest snapshot.

## Extending the registry

Models are discovered via the `promforecast.models` entry point.

**Community model example** (`promforecast-prophet`):

```toml
[project.entry-points."promforecast.models"]
prophet = "promforecast_prophet:ProphetModel"
```

**In-tree (built-ins)**:
1. Implement `ForecastModel` protocol in `forecaster/src/promforecast/models/<name>.py`.
2. Register in `forecaster/pyproject.toml`.
3. Document here.

CI verifies the full registry. The registry is lazy and cached; duplicate names raise `ModelRegistrationError` at boot.
