# Forecast inspector

`promforecast inspect` is the debugging tool for “why does this one series have a weird forecast?”.

It pulls the exact series from the TSDB, runs the configured fit pipeline exactly as the live forecaster does, and prints the inputs, the fit result, the forecast curve, and (optionally) the per-fold backtest.

## When to use it

- A specific series’s forecast in Grafana looks wrong (too conservative, weirdly seasonal, flat, predicting negatives).
- `forecast_quality_score` for one series is low while neighbours are fine.
- You need to compare model behaviour without a config change (run once with `--model AutoARIMA`, again with `--model SeasonalNaive`).
- You want to see the exact per-fold backtest predictions that produced a high MAPE.

## Basic usage

```bash
promforecast inspect \
  --config /etc/promforecast/config.yaml \
  --query node_cpu_busy_pct \
  --series-filter '{instance="host-a.example.com"}'
```

**Default text output** (first 10 forecast rows):

```
Query:        node_cpu_busy_pct (group node_capacity)
Model:        SeasonalNaive
Series:       device=cpu0, instance=host-a.example.com
Step / season / horizon: 300s / 288 / 288 steps

Input stats:
  point_count        : 4032
  gaps_filled        : 12
  largest_gap_seconds: 600.0
  lookback_start     : 2026-05-10T12:00:00+00:00
  lookback_end       : 2026-05-24T12:00:00+00:00

Preprocessing applied:
  gaps_filled               : {'forward_fill': 12}
  outliers_replaced         : 0
  change_points             : 0
  imputed_fraction          : 0.0029
  effective_lookback_seconds: 1209600.0

Forecast (first 10 of 288 rows):
  2026-05-24T12:05:00+00:00  yhat=42.1234  [95: 30.0000..54.0000]
  ...
```

Use `--json` for machine-readable output (ideal for `jq` or CI artifacts).

## Backtest mode

`--backtest` runs the same backtest harness the live runner uses and prints per-step predictions vs actuals:

```bash
promforecast inspect --config cfg.yaml --query Q --series-filter '{...}' --backtest
```

Adds MAPE/MASE and the hold-out rows.  
Useful pattern for model comparison:

```bash
promforecast inspect ... --model AutoARIMA --backtest --json > a.json
promforecast inspect ... --model SeasonalNaive --backtest --json > b.json
```

## Flags

| Flag                | Purpose |
|---------------------|---------|
| `--config`          | Path to the live forecaster YAML (required) |
| `--query`           | Query `id` to inspect (required) |
| `--series-filter`   | PromQL selector when the query returns multiple series |
| `--horizon`         | Override `defaults.horizon` |
| `--lookback`        | Override `defaults.lookback` |
| `--model`           | Use this specific model (must be in `defaults.models`) |
| `--backtest`        | Also run and print the backtest |
| `--json`            | Output full report as JSON |

## Debugging walkthroughs

**Sparse signal misfit**  
Symptom: forecast collapses to a constant, low quality score.  
Look at “Input stats” for low `point_count` or high `imputed_fraction` → fix on the input side (extend lookback, fix scrape gaps, or add `cold_start`).

**Regressor missing**  
Symptom: forecast ignores a known driver (e.g. deploy spike).  
Check `forecast_regressor_failures_total` for the query/regressor pair. The inspector does **not** fetch regressors (keeps cost bounded).

**Change-point not detected**  
Symptom: forecast anchored to pre-incident level.  
Look at “Preprocessing applied / change_points”. If `0`, the detector is disabled or not sensitive enough.

## Limitations

- Fits **one model** per invocation (run multiple times to compare).
- Conditional forecasts, decomposition, and conformal calibration are **not** shown (inspector mirrors the standard fit only).
- Bands are raw model output (narrower than live conformal bands).
- No regressor values are fetched or applied.
- Count-aware safeguards (`clip_non_negative`, `round_to_integer`) **are** applied so output matches what the exporter would emit.
