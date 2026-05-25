# Hyperparameter auto-tuning

Most production configs pin `season_length`, `defaults.models`, and per-query overrides at install time. As the underlying signal drifts, those values can quietly become suboptimal — the only visible symptom is slowly degrading `forecast_quality_score` or rising `forecast_accuracy_mape`.

`promforecast tune` evaluates a small grid of alternatives against the live backtest harness and emits one recommendation per (group, query, model) pair that beats the current baseline by a configurable margin. The operator reviews the recommendations and applies them manually — there is no auto-apply path.

The recommended deployment is a weekly `CronJob` via the [`promforecast-tuner`](../../charts/promforecast-tuner/) sub-chart. It re-uses the same container image and config as the live forecaster.

## Emitted metrics

| Metric | Type | Description |
|--------|------|-------------|
| `forecast_recommended_param{group, query, model, name, current_value, recommended_value}` | gauge | One row per candidate that beats the baseline. Value = `expected_mape_improvement_pct`. |
| `forecast_tuner_last_run_timestamp_seconds{group, outcome}` | gauge | Timestamp of the most recent run per group (`outcome`: `ok` / `partial` / `error`). |
| `forecast_tuner_last_run_duration_seconds{group}` | gauge | Wall-clock duration of the most recent run per group. |

## The grid (deliberately small)

- **season_length**: half and double the configured value (e.g. `288` → `144`, `576`).
- **model choice**: every other model in `defaults.models`.

The grid is intentionally limited so the weekly run stays cheap. (Larger grids and FFT-driven auto-seasonality are planned for v0.5.)

## Should I apply this recommendation?

Use the `expected_mape_improvement_pct` value:

| Improvement | Recommended action |
|-------------|--------------------|
| < 5 %       | Ignore (likely noise). |
| 5–15 %      | Worth applying at the next config-change window. |
| 15–30 %     | Apply soon — the current value has drifted. |
| > 30 %      | Investigate first — usually indicates the original config was suboptimal. |

## Operator workflow

1. Weekly CronJob runs.
2. Tuner evaluates every group × query × candidate and pushes recommendations.
3. Operator opens the *Tuner recommendations* Grafana panel (ranked by improvement).
4. Operator edits the config, commits, and rolls out.
5. Next tuner run automatically drops the now-applied recommendation.

## When the tuner fails

Alert on `forecast_tuner_last_run_timestamp_seconds{outcome="error"} > 0` (or on staleness across all outcomes).

Common causes:
- Datasource unreachable → fix upstream.
- Sink unreachable → next run regenerates recommendations (tuner is stateless).

`outcome="partial"` is normal (some queries had too little data).

Missing runs are caught by staleness:

```promql
time() - max by (group) (forecast_tuner_last_run_timestamp_seconds) > 8 * 24 * 60 * 60
```

(Adjust multiplier for your schedule.)

## Scope

- No auto-apply.
- No per-series tuning.
- No large hyperparameter optimisation (only the shipped grid).

A future expansion (larger grid, per-model params) will keep the same metric shape and operator workflow.

The tuner is a separate sub-chart so clusters that don’t want weekly jobs can install the forecaster alone and add the tuner later. It is a pure read-side operation against the long-term TSDB.
