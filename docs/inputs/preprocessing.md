# Input preprocessing: gaps and outliers

Production metrics are noisy — scrapes drop, deploys spike, autoscalers shift baselines.  
Before fitting, the forecaster optionally runs three cleaning steps **in this order**:

1. Gap imputation  
2. Outlier scrubbing  
3. Change-point detection (see [`change-points.md`](change-points.md))

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

## Interaction with conformal calibration

Outlier scrubbing runs **before** conformal calibration (see [`../outputs/conformal.md`](../outputs/conformal.md)).  
A clean hold-out window produces tighter, more honest quantile bands.
