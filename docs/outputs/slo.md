# SLO error-budget forecasting

Instead of only alerting on “how much error budget have I burned?”, forecast your error-rate metric to answer “at the current predicted burn rate, when will the budget run out?”

This gives you forward-looking SLO exhaustion projections that match the same trajectory style used for capacity forecasting.

## Configuration

Add an `slo:` block to any error-rate query:

```yaml
groups:
  - name: http_slo
    queries:
      - id: http_request_error_ratio
        promql: |
          sum(rate(http_requests_total{status=~"5.."}[5m]))
            / sum(rate(http_requests_total[5m]))
        slo:
          objective: 99.9
          windows: [5m, 1h, 30d]
```

| Field       | Default     | Meaning |
|-------------|-------------|---------|
| `objective` | required    | SLO target as percentage (e.g. `99.9`) |
| `windows`   | `[30d]`     | One or more SLO windows (short + long windows recommended) |

## Emitted metrics

Per `(series, model, window)`:

| Metric | Description |
|--------|-----------|
| `<id>_forecast_error_budget_remaining_seconds{window}` | Static total budget for the window |
| `<id>_forecast_error_budget_burn_rate{window}` | Predicted burn rate (1.0 = exactly on pace) |
| `<id>_forecast_error_budget_exhausted_in_seconds{window}` | Seconds until budget exhaustion (central prediction) |
| `<id>_forecast_error_budget_exhausted_in_seconds{window, level}` | Per-confidence-level exhaustion (uses upper band) |

- `burn_rate > 1` → consuming budget faster than allowed.
- `exhausted_in_seconds = +Inf` when predicted error rate is zero.

## Example alerts

```promql
# Budget projected to exhaust in <24h
http_request_error_ratio_forecast_error_budget_exhausted_in_seconds{window="30d"} < 86400

# Fast burn: 30d budget gone in <2 days
http_request_error_ratio_forecast_error_budget_burn_rate{window="30d"} > 14.4
```

Configure multiple windows (`[5m, 1h, 30d]`) to cover both fast-burn and slow-drift cases.

## Cardinality

Roughly `(3 + N_levels) × N_windows` rows per (series, model) for each opted-in query.  
A query with 500 series, 1 model, 2 levels, and 3 windows adds ~7 500 rows.  
`promforecast validate` accounts for this in its estimate.

## Implementation notes

- Projections are **point-in-time** (based on the leading forecast row), not integrated over the full window. This matches operator intuition for “current predicted rate”.
- Uses `yhat_upper` for pessimistic exhaustion time when confidence levels are configured.
- Works best with `accuracy.conformal.enabled: true` for calibrated bands.

## What this is not

- Not a replacement for proper SLI instrumentation.
- Not a full state machine (does not integrate observed burn so far — pair with recording rules if needed).
- Not an alerting engine — it only emits metrics for Alertmanager rules.
