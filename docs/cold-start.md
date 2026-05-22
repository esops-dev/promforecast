# Cold-start sibling-borrowing

When a brand-new service launches, most series have far less than `defaults.min_points` history. By default the runner drops them with `forecast_failures_total{reason="min_points"}`.  

Cold-start sibling-borrowing solves this by synthesising a forecast from comparable peer series instead of failing the fit.

## How it works

When a series has insufficient history:

1. Run the configured `cold_start.borrow_from.promql` to fetch sibling series.
2. Normalise each sibling by dividing by its own mean.
3. Take the per-step median across siblings → smoothed seasonal shape.
4. Re-scale the shape to the target series’ recent average.
5. Emit the result as a `SeasonalNaive`-style forecast with labels `model="ColdStart"` and `cold_start="true"`.

No backtest, deviation, conformal, or growth-rate metrics are produced for cold-start forecasts — they are explicitly best-effort.

## Configuration

```yaml
groups:
  - name: services
    queries:
      - id: http_requests_per_second
        promql: rate(http_requests_total{service="new-service"}[5m])
        cold_start:
          borrow_from:
            promql: |
              rate(http_requests_total{service!="new-service",region="us-east-1"}[5m])
            method: median_normalized   # only option in v1.3
            min_siblings: 3             # default
          max_quality_score: 0.5        # caps quality score (default 0.5)
```

The `borrow_from` block is required to enable cold-start. Without it, short series are still dropped.

## Telemetry

- `forecast_cold_start_siblings{group, query}` (gauge) — number of usable siblings in the most recent run.  
- `forecast_quality_score{cold_start="true"}` — emitted at the configured cap.

## Label routing

All cold-start metrics carry `cold_start="true"`. Regular series render with `cold_start=""` (empty string).  

**PromQL patterns**

```promql
# Hide cold-start forecasts
http_requests_per_second_forecast{cold_start!="true"}

# Alert on thinning sibling pool
forecast_cold_start_siblings < 5
```

Because `prometheus_client` unions label sets, `forecast_quality_score` (and forecast families) always expose the `cold_start` label once any cold-start series appears. Most `sum by` / `avg by` queries are unaffected. For strict matchers, add `cold_start=""` explicitly when you want only regular fits.

## When sibling-borrowing won’t help

- No comparable peers → borrow query returns zero series → falls back to normal `min_points` drop.  
- All siblings are constant or zero-mean → normalisation produces degenerate prior.  
- Target has fundamentally different seasonality than siblings → borrowed shape is misleading.

### Combinations rejected at config load

- `data_profile: intermittent` + `cold_start.borrow_from` (median of sparse zeros collapses).  
- `max_quality_score <= 0` (would trigger `ForecastQualityZero` every cycle).

### Combinations skipped at runtime

- `regressors[].required: true` — cold-start never feeds regressors, so the series falls back to the normal `min_points` drop (logged as `cold_start_disabled reason=required_regressor`).

The example alerts in `examples/alerts/promforecast-rules.yaml` already exclude `cold_start="true"` from `ForecastQualityLow` / `ForecastQualityZero` and provide a dedicated `ForecastColdStartSiblingsLow` rule.

## Non-goal

This is a simple prior, not hierarchical reconciliation. True bottom-up/top-down forecasting across related series is planned for a future model pack.
