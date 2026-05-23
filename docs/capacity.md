# Capacity planning

Three knobs turn promforecast into a capacity-planning tool you can read
straight off a Grafana panel:

1. **Damped-trend models** (`AutoETSDamped`, `AutoARIMADamped`) keep
   multi-day extrapolations honest.
2. **Time-to-threshold ETAs** answer "when will this hit X?" as a metric.
3. **Forecast growth rate** exposes how fast each series is heading
   somewhere, in `units/second` or as a percent.

All three are configured per query and emitted alongside the regular
forecast metrics — no recording rule plumbing required.

## Damped-trend models

Linear extrapolation pretends disk usage that grew 1%/day for a week
will keep growing 1%/day for the next year. It will not. The
damped-trend variants bend the long-horizon extrapolation toward a level
while keeping near-term forecasts faithful, which is the behaviour
operators expect when planning capacity.

```yaml
defaults:
  models:
    - name: AutoETSDamped
      params:
        damping_parameter: 0.98
    - name: AutoARIMADamped
      params:
        damping_parameter: 0.98
    - SeasonalNaive
```

| Parameter            | Default | Effect                                                                                                       |
|----------------------|---------|--------------------------------------------------------------------------------------------------------------|
| `damping_parameter`  | `0.98`  | Geometric decay applied per forecast step. `1.0` disables damping; lower values bend the curve more sharply. |

The damping is applied as a post-fit transformation: the forecaster
fits the underlying model normally, then decomposes the predicted curve
into a linear trend plus an oscillation component, replaces the linear
extrapolation with a geometric sum, and shifts the confidence bands by
the same amount so they ride along with the bent forecast without
changing width. Seasonal / cyclic structure is preserved.

**Rule of thumb:** turn on damping for any horizon longer than one full
seasonal cycle. For 5-minute step + 24h horizon at `season_length=288`,
damping starts to matter at horizons of multiple days; for hourly
business KPIs at weekly seasonality, damping starts to matter at
multi-week horizons.

The example `examples/configs/node-capacity.yaml` defaults to damped
trend with `auto_select: true` so the forecaster picks between the two
damped variants per series.

## Time-to-threshold ETA

Configure named thresholds per query and the forecaster emits a metric
named `<id>_forecast_time_to_threshold_seconds` whose value is the
seconds until the forecast first crosses the threshold:

```yaml
groups:
  - name: node_capacity
    queries:
      - id: node_cpu_busy_pct
        promql: |
          100 - (avg by (instance) (rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)
        thresholds:
          - name: warn
            value: 80
            comparator: gt
          - name: critical
            value: 95
            comparator: gt
```

| Field         | Required | Meaning                                                                                                        |
|---------------|----------|----------------------------------------------------------------------------------------------------------------|
| `name`        | yes      | Becomes the `name` label on the emitted metric. Must be unique within a query.                                 |
| `value`       | yes      | Threshold the forecast is compared against.                                                                    |
| `comparator`  | no       | `gt` (default) fires on the first sample above; `lt` fires on the first sample below.                          |

Output:

```
node_cpu_busy_pct_forecast_time_to_threshold_seconds{
  name="warn", comparator="gt", group="node_capacity", model="AutoETSDamped", instance="host-a"
} 32400
```

When confidence levels are emitted, a second sample is added per level
using the *conservative* band edge — `yhat_upper` for `gt` thresholds,
`yhat_lower` for `lt` thresholds. The point ETA tells you when the
central forecast crosses; the band ETA tells you the earliest the
crossing could happen at the configured confidence level. Build alerts
against the band sample for "disk will fill no later than X hours" SLOs:

```promql
# Alert: disk will fall below 5 GiB within 6h at 95% confidence
node_filesystem_avail_bytes_forecast_time_to_threshold_seconds{
  name="critical", comparator="lt", level="95"
} < 6 * 60 * 60
```

When the forecast never crosses the threshold within the configured
horizon, no sample is emitted (NaN is dropped at render time) — alerts
on `time_to_threshold < N` work the way you'd expect.

If the forecast starts already on the wrong side of the threshold, the
ETA is reported as **0** (read: already in violation).

### Worked examples

- **Disk fill**: forecast `node_filesystem_avail_bytes` and set
  `comparator: lt` with the free-space floor as `value`. Pair with
  `level=95` in the alert query for a conservative ETA.
- **Certificate expiry**: forecast the cert's `notAfter - time()`
  metric, use `comparator: lt` with `value: 0`. The ETA *is* the
  time-to-expiry.
- **Quota exhaustion**: forecast the in-use side of any quota,
  `comparator: gt`, `value` = the quota cap.

### Predictive will-breach boolean

`time_to_threshold_seconds < 6h` is a perfectly good alert, but every
operator who writes one re-derives the same arithmetic. Set
`alert_within` + `severity` on a threshold and the forecaster emits a
boolean shaped exactly for `forecast_will_breach == 1`:

```yaml
queries:
  - id: node_filesystem_avail_bytes
    promql: node_filesystem_avail_bytes{fstype!~"tmpfs|overlay|squashfs"}
    thresholds:
      - name: critical
        value: 5368709120     # 5 GiB
        comparator: lt
        alert_within: 6h      # opt into the will_breach gauge
        severity: critical
```

| Field            | Required               | Meaning                                                                              |
|------------------|------------------------|--------------------------------------------------------------------------------------|
| `alert_within`   | only when paired       | Lookahead window. The boolean fires whenever the ETA is finite and below this value. |
| `severity`       | only when paired       | Becomes the `severity` label on the emitted metric.                                  |

Both fields must be set together — the predictive metric is a derived
shape and partial config is rejected at load time. The metric is emitted
even when the predicate is `0`, so dashboards have a baseline series to
chart from boot:

```
node_filesystem_avail_bytes_forecast_will_breach{
  name="critical", severity="critical", level="95",
  group="node_capacity", model="AutoETSDamped", instance="host-a"
} 1.0
```

The same fan-out as the ETA family applies: a row for the central
forecast (no `level` label) plus one row per confidence level so
operators can choose how conservative to be. Alerts read like:

```promql
forecast_will_breach{severity="critical", level="95"} == 1
```

The corresponding `forecast_time_to_threshold_seconds` value at the
same labelset is the operator-visible "when" for the breach. See
`examples/alerts/promforecast-rules.yaml` for the `ForecastWillBreach`
rule template.

## Forecast growth rate

Configure per-query growth-rate emission to publish how fast the
forecast curve is rising or falling:

```yaml
queries:
  - id: node_filesystem_avail_bytes
    promql: |
      node_filesystem_avail_bytes{fstype!~"tmpfs|overlay|squashfs"}
    emission:
      growth_rate:
        enabled: true
        windows: [1d, 7d]
        units: units_per_second
```

| Field      | Default                  | Meaning                                                                                                |
|------------|--------------------------|--------------------------------------------------------------------------------------------------------|
| `enabled`  | `false`                  | Opt-in. Off by default to keep cardinality in check.                                                   |
| `windows`  | `[1d, 7d]`               | One sample is emitted per window. The slope is computed across forecast rows inside the window.        |
| `units`    | `units_per_second`       | `units_per_second` is a least-squares slope (matches Grafana axes); `percent` is `(end / start) - 1`.  |

Output:

```
node_filesystem_avail_bytes_forecast_growth_rate{
  window="7d", units="units_per_second", group="node_capacity", model="AutoETSDamped", instance="host-a"
} -120.4
```

Negative values mean the forecast is shrinking. Operators typically
alert on **changes** to growth rate, not absolutes:

```promql
# Alert: growth rate has tripled compared to last week's baseline
abs(metric_forecast_growth_rate{window="1d"})
  / clamp_min(abs(metric_forecast_growth_rate{window="7d"} offset 1d), 1) > 3
```

Top-N panels in Grafana on `topk(10, metric_forecast_growth_rate)` give
an at-a-glance "what's accelerating?" view across a fleet.

Windows wider than the configured horizon emit a `NaN` (and so are
dropped from `/metrics`) rather than silently producing a shorter
window — the alert query you wrote with `window="7d"` keeps meaning
what it said. Configure shorter windows or extend `defaults.horizon`
if you need them.

## Cardinality

Both new families count against the global `safety.max_total_series`
cap; the cardinality estimator in `promforecast validate` includes
them:

- Threshold ETA: `N_series × N_models × N_thresholds × (1 + N_levels)`
  per query that configures thresholds.
- Will-breach: `N_series × N_models × N_thresholds_with_alert × (1 + N_levels)`
  per query — only thresholds with `alert_within` + `severity` count.
- Growth rate: `N_series × N_models × N_windows` per query that opts in.

Run `promforecast validate config.yaml` after adding thresholds or
growth-rate emission to confirm the new totals fit within your safety
budget.
