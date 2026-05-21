# Input-side change-point detection

When a metric regime-shifts (deploy doubles throughput, autoscaler halves latency, migration cuts disk usage in half), fitting across the old and new regime produces visibly wrong forecasts.  
Input-side change-point detection finds the most recent structural break in the lookback and **adapts the fit window** before the model runs.

> **Note:** this is the *input-side* counterpart of the detector in `promanomaly`. There it emits an alert counter; here the result is consumed internally (truncate or reweight the lookback) and never leaves the runner.

## Configuring

```yaml
groups:
  - name: node_capacity
    preprocess:
      change_point:
        enabled: true
        method: ruptures               # ruptures | bocpd
        action: truncate_lookback      # truncate_lookback | reweight_recent
        min_post_break_points: 50
```

| Setting              | Options                              | Default          | Effect |
|----------------------|--------------------------------------|------------------|--------|
| `method`             | `ruptures` (PELT + L2 cost) / `bocpd` (CUSUM-style) | `ruptures` | `ruptures` is faster and dependency-free in the base image; `bocpd` is a pure-Python fallback |
| `action`             | `truncate_lookback` / `reweight_recent` | `truncate_lookback` | Hard cut vs. soft decay of pre-break data |
| `min_post_break_points` | integer                           | 50               | Never truncate to fewer than this many points (rule of thumb: one full seasonal cycle) |

For `reweight_recent` you can also tune `reweight_decay` (0.0 = hard truncate, 1.0 = no-op).

## Sensitivity vs. stability

The detectors are deliberately conservative — an over-eager truncation hurts MAPE and widens bands. Defaults bias toward stability:

- `ruptures` uses a BIC-style penalty (`3 * log(n)`) to suppress false positives from normal daily/weekly patterns.
- `bocpd` requires a t-statistic > 3.5 (≈ Bonferroni α = 0.01 across ~1000 candidate points).

If the detector is missing obvious breaks you see in Grafana, check:
1. The break is too close to the end of the lookback (< `min_post_break_points`).
2. The shift is small relative to normal noise (both methods target mean shifts).

## Observability

```
forecast_input_change_point_total{group, query}
forecast_input_effective_lookback_seconds{group, query}
```

- `change_point_total` counts detected breaks.
- `effective_lookback_seconds` shows the actual window used for the latest fit (equals configured `lookback` when no truncation occurs).

Dashboard tip: stack the static `defaults.lookback` against `forecast_input_effective_lookback_seconds` to spot chronically truncated series.

## See also

- [Models](models.md) — how the adjusted lookback feeds the fit.
- [Input preprocessing](preprocessing.md) — outlier scrubbing runs *before* change-point detection so a single spike cannot masquerade as a regime shift.
