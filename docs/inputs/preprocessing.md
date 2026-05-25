# Input preprocessing: gaps, outliers, blacklist windows

Production metrics are noisy — scrapes drop, deploys spike, autoscalers shift baselines.  
Before fitting, the forecaster optionally runs four cleaning steps **in this order**:

1. Blacklist windows (operator-known anomalous spans)
2. Gap imputation
3. Outlier scrubbing
4. Change-point detection (see [`change-points.md`](change-points.md))

## Gap imputation

```yaml
groups:
  - name: node_capacity
    preprocess:
      gaps:
        strategy: forward_fill     # forward_fill | linear_interp | nan | drop_series
        max_gap: 30m
```

| Strategy       | Behaviour                                      | Typical use case                  |
|----------------|------------------------------------------------|-----------------------------------|
| `forward_fill` (default) | Copy last known value forward                  | Slowly-changing counters          |
| `linear_interp`          | Straight-line interpolation                    | Monotone metrics (disk, queues)   |
| `nan`                    | Leave gaps as NaN                              | Let the model decide              |
| `drop_series`            | Skip entire series if any gap > `max_gap`      | Strict data-quality enforcement   |

Any gap longer than `max_gap` (default 30 min) forces `drop_series` regardless of strategy — a 4-hour outage never fabricates values.

**Observability**
- `forecast_input_gaps_filled_total{group, query, strategy}` (counter per actual imputation)
- `forecast_input_max_gap_seconds{group, query}` (gauge of largest gap seen)

Heavy imputation reduces the `forecast_quality_score` coverage component by `(1 - imputed_fraction)` so fully fabricated series no longer pass quality filters.

## Outlier scrubbing

```yaml
groups:
  - name: node_capacity
    preprocess:
      outliers:
        enabled: true
        method: hampel              # hampel | iqr | stl_residual
        sensitivity: medium         # low | medium | high
        replace_with: median_neighbours
```

Disabled by default (silent correction surprises operators).

### Methods

| Method          | Profile                              | Best for                                      |
|-----------------|--------------------------------------|-----------------------------------------------|
| `hampel`        | Rolling median + MAD                 | Isolated spikes on local baseline             |
| `iqr`           | Global Tukey IQR                     | Stable-mean metrics (latency, error rates)    |
| `stl_residual`  | STL decomposition + residual filter  | Seasonal data where phase matters             |

### Sensitivity levels

| Sensitivity | hampel | IQR     | STL    |
|-------------|--------|---------|--------|
| `low`       | 4.0σ   | 3.0×    | 4.0σ   |
| `medium`    | 3.0σ   | 1.5×    | 3.0σ   |
| `high`      | 2.0σ   | 1.0×    | 2.0σ   |

### Replacement

- `median_neighbours` (default) — local 7-point median  
- `linear_interp` — straight line between nearest clean samples  
- `nan` — leave as NaN (model sees true missing data)

### Observability

`forecast_input_outliers_replaced_total{group, query, method}` (counter)

### Worked example

A five-day series with one 10× incident spike at hour 36:  
`AutoARIMA` widens confidence bands dramatically.  
With `outliers.enabled: true, method: hampel, sensitivity: medium` the spike is replaced by the local median and bands return to normal width (see `tests/test_preprocess.py::test_outlier_scrubbing_replaces_with_median`).

## Blacklist windows

Sometimes the operator already knows that a chunk of history is anomalous and should not influence the forecast — a known incident week, a Black Friday traffic spike, a planned migration freeze. Sliding the `lookback` boundary discards unrelated normal data; outlier scrubbing is statistical and may miss multi-day periods. `preprocess.blacklist_windows` makes the operator-known case explicit.

```yaml
groups:
  - name: http_capacity
    preprocess:
      blacklist_windows:
        # Absolute window — drop the November 2024 incident week.
        - start: 2024-11-29T00:00:00Z
          end:   2024-12-06T00:00:00Z
          reason: nov2024_incident
        # Annual recurring — drop Black Friday peak every year.
        - start: 11-29T00:00
          end:   12-02T00:00
          reason: black_friday
          recurring: annual
```

### How it works

Samples whose timestamp falls inside any window are dropped *before* gap imputation. The configured `gaps.strategy` then treats the blacklisted span as a gap: a short window is forward-filled (or linearly interpolated) per the strategy; a window longer than `gaps.max_gap` triggers `drop_series` and skips the series for that fit.

The half-open interval convention `[start, end)` lets two adjacent windows touch at the boundary without double-counting it.

### Shapes

| Field | Required | Description |
|---|---|---|
| `start` | Yes | ISO-8601 timestamp (absolute) or `MM-DDTHH:MM` (annual) |
| `end` | Yes | Same shape as `start`; must be strictly after `start` |
| `reason` | No | Free-text label surfaced on the per-window counter |
| `recurring` | No | `null` (default, absolute) or `annual` |
| `timezone` | No | IANA zone name for `recurring: annual` windows; ignored for absolute. Defaults to `UTC` |

Annual windows materialise one absolute span per year present in the lookback so a 30-year retention does not pay the cost of materialising 30 spans on every fit. A window that wraps a year boundary (`12-29T00:00` → `01-02T00:00`) rolls the end forward to the next year.

**Timezones.** Recurring annual windows are anchored in the configured `timezone` so a US Black Friday window flips at local midnight rather than UTC-midnight. Absolute windows are unaffected — the ISO-8601 string carries its own offset. Same convention as the [`holidays` regressor](regressors.md). An unrecognised IANA zone name causes the window to be silently skipped (the rest of the series is still fit); pair this with the per-(group, query) `forecast_input_blacklisted_samples_total` counter so an empty-counter row is the operator-visible signal that the window didn't expand.

### Observability

- `forecast_input_blacklisted_samples_total{group, query, reason}` — counter, monotonic across runs. Useful for confirming that an annual-recurring window actually fired for the most recent year of the lookback.
- `forecast_input_effective_lookback_seconds{group, query}` — gauge. Discounts the blacklisted span (`count × step`) so a 14-day lookback with two 6-hour blacklisted windows reports `14d − 12h`.

### When to reach for it

- A known incident window where the metric was driven by the failure rather than by the underlying workload.
- Recurring high-traffic events the forecaster shouldn't learn as the baseline (Black Friday, fiscal-quarter-end batches).
- A planned migration / cutover that produced a transient regime that has since returned to the baseline.

Use a regressor instead when the anomalous behaviour is recurring *and* expected — the model should learn its effect rather than pretend the period never happened.

## Interaction with conformal calibration

Outlier scrubbing runs **before** conformal calibration (see [`../outputs/conformal.md`](../outputs/conformal.md)).  
A clean hold-out window produces tighter, more honest quantile bands.
