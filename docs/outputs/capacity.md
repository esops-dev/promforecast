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

### Ranking by cost of action

`time_to_threshold_seconds` alone tells you *which* capacity event is
closest. It does not tell you which one is most expensive to ignore.
Add an `action_cost` annotation to surface a cost gauge that dashboards
and alerts can multiplex against the ETA:

```yaml
queries:
  - id: object_storage_used_bytes
    promql: sum by (bucket) (object_storage_used_bytes)
    thresholds:
      - name: critical
        value: 1099511627776   # 1 TiB
        comparator: gt
        alert_within: 7d
        severity: critical
        action_cost:
          type: usd
          value: 50000          # tier-upgrade cost if we breach
      - name: backup_warn
        value: 5368709120       # 5 GiB
        comparator: gt
        action_cost:
          type: usd
          value: 50              # nightly archive purge cost
```

| Field   | Required | Meaning                                                                                                |
|---------|----------|--------------------------------------------------------------------------------------------------------|
| `type`  | yes      | Free-text label. Use a currency (`usd`, `eur`) or a non-monetary unit (`engineer_hours`, `priority`).  |
| `value` | yes      | Numeric cost. Must be `> 0` — zero or negative values would silently rank below every meaningful row.  |

Output:

```
forecast_threshold_action_cost{
  id="object_storage_used_bytes", group="capacity",
  name="critical", type="usd"
} 50000
```

The gauge is config-derived: one row per `(group, query, threshold)`
with `action_cost` configured, independent of model / series fan-out.
A threshold removed from YAML disappears from `/metrics` on the next
refresh (the family is rebuilt from current config each run).

Rank by cost in Grafana:

```promql
# Top 10 capacity events by cost-to-ignore — gated to "within 24h".
topk(
  10,
  forecast_threshold_action_cost
  * on(id, group, name) group_left()
  (forecast_time_to_threshold_seconds < 86400)
)
```

`type` is free text on purpose: `usd` and `engineer_hours` coexist in
a single family so different teams can rank by their own currency
without forking the metric. Pin the label in your queries
(`forecast_threshold_action_cost{type="usd"}`) so a panel reports a
single currency and never sums apples with oranges.

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

## Queue burn-down / drain forecasting

A queue (Kafka consumer lag, Sidekiq job backlog, alert backlog) that
*should* drain over time is the inverse of capacity: instead of
"when will this hit X from below?", you want "when will this hit
zero from above?". Use the `le_drain` comparator on a threshold to
opt into the dedicated drain ETA family:

```yaml
queries:
  - id: kafka_consumer_lag
    promql: sum by (topic) (kafka_consumergroup_lag{group="orders"})
    thresholds:
      - name: drained
        value: 0
        comparator: le_drain
```

Output (note the different metric name — `_forecast_time_to_drain_seconds`,
not `_time_to_threshold_seconds`):

```
kafka_consumer_lag_forecast_time_to_drain_seconds{
  name="drained", group="kafka", topic="orders"
} 5400
```

`le_drain` is mechanically the same predicate as `lt` (value drops
below threshold), but with three deliberate differences:

1. **Different metric name.** Emits under
   `<id>_forecast_time_to_drain_seconds`, parallel to the
   threshold-ETA family. The two never collide on the same query.
2. **`+Inf` instead of NaN on no-crossing.** When the forecast curve
   never crosses the threshold from above within the horizon, the
   sample reads `+Inf`. Operators alert with
   `forecast_time_to_drain_seconds == +Inf for 30m` to catch "queue
   is not actually draining" as a distinct condition from "draining
   slowly". With NaN (the standard `lt` behaviour), the alert would
   silently never match.
3. **`yhat_upper` for the conservative band.** A larger predicted
   queue drains later, so `yhat_upper` is the *latest* predicted
   drain — the conservative answer when alerting on a stuck queue.
   (`lt` uses `yhat_lower` for the earliest-crossing semantics that
   capacity alerts want.)

The shipped `examples/alerts/promforecast-rules.yaml` includes a
`promforecast.queue.examples` group with two reference alerts:

```promql
# Queue is not actually draining
kafka_consumer_lag_forecast_time_to_drain_seconds{name="drained"} == +Inf

# Drain projected to take >4h (and is happening at all)
kafka_consumer_lag_forecast_time_to_drain_seconds{name="drained"} > 4 * 60 * 60
and
kafka_consumer_lag_forecast_time_to_drain_seconds{name="drained"} != +Inf
```

## Cardinality

All the capacity-style families count against the global
`safety.max_total_series` cap; the cardinality estimator in
`promforecast validate` includes them:

- Threshold ETA: `N_series × N_models × N_thresholds × (1 + N_levels)`
  per query that configures thresholds (excluding `le_drain`).
- Drain ETA: same shape, charged separately. Only `le_drain` thresholds
  count.
- Will-breach: `N_series × N_models × N_thresholds_with_alert × (1 + N_levels)`
  per query — only thresholds with `alert_within` + `severity` count.
- Growth rate: `N_series × N_models × N_windows` per query that opts in.
- Action cost: one line per threshold with `action_cost` configured.
  No model / series fan-out — the metric is config-derived.

Run `promforecast validate config.yaml` after adding thresholds or
growth-rate emission to confirm the new totals fit within your safety
budget.
