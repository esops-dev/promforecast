# Emission paths: `/metrics` snapshot vs. sink + VM

promforecast can publish each forecast through two different write paths.
Production deployments should pick one of them as the canonical store; the
official charts default to **sink + VM** for a fresh install. This page
explains what each path actually emits, what their trade-offs are, why
running both unguarded silently produces wrong-looking dashboards, and
how the chart defaults steer you onto the recommended path.

## The two paths

### 1. `/metrics` snapshot (pull)

Every fit overwrites a single gauge per (series, model, horizon). When
Prometheus scrapes the forecaster's `/metrics` endpoint it sees the
**end-of-horizon** value as a single point at the scrape timestamp; the
intermediate forecast steps from `t+1` to `t+horizon` are not exposed.

- **What gets stored.** Per scrape interval, one sample per emitted
  forecast metric per series, carrying the forecaster's scrape labels
  (`instance="<pod>:9091"`, `job="<release>"`, plus whatever `cluster` /
  `replica` external labels the scraper adds).
- **What it looks like on a Grafana time-series.** A 5-minute staircase
  — each scrape redraws the gauge at the current value, so the line
  steps every refresh interval rather than tracing a smooth curve.
- **When it's the right pick.** Single-replica low-cardinality installs
  where the snapshot is enough to feed alerts on end-of-horizon
  thresholds, and you don't need to render the forecast curve as a
  smooth projection.

### 2. `sink.remote_write` → VictoriaMetrics (push)

After every fit the forecaster pushes the **full forecast curve** (every
step from `t+1` to `t+horizon`) into a long-term TSDB via
`remote_write` with native future timestamps. VictoriaMetrics accepts
future samples when started with `-futureRetention=2d` or higher.

- **What gets stored.** Per fit, one sample *per future step* per
  series — carrying only the source labels of the underlying signal
  (no `instance="<forecaster>:9091"`, no `job="promforecast"`).
- **What it looks like on a Grafana time-series.** A smooth curve
  extending from `now` into the future, with `_lower` / `_upper` bands
  rendered as a shaded interval. No `offset <horizon>` trickery is
  needed to align actual against predicted.
- **When it's the right pick.** Almost always. Smooth curves are what
  capacity-planning and deviation-detection dashboards want; the
  forecast becomes a first-class historical signal queryable with
  normal PromQL.

## The dual-source trap

Running both paths against the same long-term TSDB without filtering
puts **two copies of every `*_forecast` / `*_forecast_lower` /
`*_forecast_upper` series** in the store under different labelsets:

- The scraped snapshot copy carries `job="promforecast"` (set by the
  ServiceMonitor's `jobLabel: app.kubernetes.io/name`), the
  forecaster's source `instance` (preserved by `honorLabels: true`),
  and any external labels Prometheus' `remote_write` adds at ingest
  (`cluster`, `replica`).
- The sink-pushed copy carries the source signal's labels
  (`instance="node-exporter:9100"`, `device`, etc.) and **no `job`** —
  the forecaster pushes directly to VM, bypassing the Prometheus scrape
  pipeline that would otherwise stamp one on.

The labelset difference therefore reduces to the presence (or absence)
of the `job` label, plus any external labels the scrape pipeline adds
at remote_write time. Both the helm-test probe in
`charts/promforecast-stack/templates/tests/labelset-uniqueness.yaml`
and the manual verification snippet below use exactly this asymmetry.

A dashboard panel that does not pre-filter on the labelset renders an
**average** of the two — the smooth sink curve smeared against the
5-minute snapshot staircase. The graph looks plausibly noisy at a
glance; the failure mode is silent.

This is the implicit default state of any install that turns the sink
on *without* dropping the snapshot families from the scrape. Both the
umbrella chart and the standalone chart fix this by default (see
"Recommended path" below).

## Recommended path: sink + VM, snapshot dropped from the scrape

Production deployments should:

1. **Enable the sink.** Set `config.sink.remote_write.enabled: true`
   and point it at the long-term TSDB.
2. **Drop the snapshot families from the scrape.** Apply a
   `metric_relabel_configs` block (or `metricRelabelings` on the
   ServiceMonitor) that drops `.+_forecast(_lower|_upper)?`. The
   operational `forecast_*` metrics (`forecast_quality_score`,
   `forecast_failures_total`, `forecast_accuracy_mape`, …) and the
   derived snapshot-only families (`*_forecast_time_to_threshold_seconds`,
   `*_forecast_growth_rate`, `*_forecast_contribution_ratio`) stay
   scraped — they only exist on `/metrics`.
3. **Point dashboards at VM as the default datasource.** The bundled
   dashboards under `dashboards/grafana/` declare VictoriaMetrics
   (`uid: victoriametrics`) as their templating default. The umbrella
   chart can optionally provision the matching datasource entries —
   set `datasources.enabled: true` (off by default to avoid colliding
   with installations that already manage Grafana datasources
   out-of-band). When enabled, the chart writes a sidecar-loaded
   ConfigMap pointing `uid: victoriametrics` at the bundled VM. If
   Grafana already has a default datasource, set
   `datasources.victoriaMetrics.isDefault: false` so the bundled
   provisioning does not collide with it — the dashboards keep working
   because they reference the explicit UID, not the global default.

The official charts ship that combination on by default:

- The **umbrella `promforecast-stack` chart** sets
  `promforecast.config.sink.remote_write.enabled: true` and
  `promforecast.serviceMonitor.enabled: true`. The forecaster's
  ServiceMonitor (Story 2) then derives the snapshot drop from the
  sink toggle automatically, so a fresh `helm install` lands on
  sink+VM with the dual-source trap closed.
- The **standalone `promforecast` chart** exposes
  `serviceMonitor.dropForecastSnapshots` (default `null`). When
  `null` it resolves to the value of `config.sink.remote_write.enabled`:
  the drop is on when the sink is on, off when the sink is off. Pin
  to `true` / `false` to override the derivation.

For an externally-managed Prometheus (not driven by prometheus-operator),
copy this snippet into the scrape job that targets the forecaster:

```yaml
scrape_configs:
  - job_name: promforecast
    honor_labels: true
    static_configs:
      - targets: ["forecaster:9091"]
    metric_relabel_configs:
      - source_labels: [__name__]
        regex: '.+_forecast(_lower|_upper)?'
        action: drop
```

The two knobs do different things and both matter:

- `metric_relabel_configs` drops the snapshot copy of the curve families
  (`*_forecast`, `*_forecast_lower`, `*_forecast_upper`) so only the
  sink-pushed copy lives in the TSDB. Without this the dual-source trap
  reappears.
- `honor_labels: true` stops Prometheus from *overwriting* the
  `instance` label that the forecaster exposes on its operational and
  derived `*_forecast_*` families (`forecast_quality_score`,
  `*_forecast_time_to_threshold_seconds`, `*_forecast_growth_rate`,
  `*_forecast_contribution_ratio`). Those families inherit the source
  signal's `instance` (e.g. `node-exporter:9100`) and the example
  alerts in `examples/alerts/promforecast-rules.yaml` interpolate
  `{{ $labels.instance }}` to identify the underlying host. Default
  Prometheus scraping would clobber that with the forecaster pod's own
  address; `honor_labels: true` preserves it. The umbrella chart's
  ServiceMonitor sets the equivalent `honorLabels: true`.

### Verifying single-labelset after install

After `helm install` and at least one successful refresh (the forecaster's
`refresh_interval`, one hour by default, so the **first scheduled fit
must have completed** before the sink push lands any data in VM), query
VM directly to confirm the curve families exist under exactly one
labelset. PromQL matcher values must be double-quoted, so the snippets
below use single-quoted shell wrappers to pass them through verbatim:

```
# Expect ``data: []`` — no forecast series under the forecaster scrape job.
curl -s --data-urlencode 'match[]={__name__=~".+_forecast(_lower|_upper)?",job="promforecast"}' \
  "$VM/api/v1/series"

# Expect a non-empty data array — sink-pushed copies (no scrape job).
curl -s --data-urlencode 'match[]={__name__=~".+_forecast(_lower|_upper)?",job=""}' \
  "$VM/api/v1/series"
```

The umbrella chart bundles this exact pair of probes as a `helm test`
hook (`charts/promforecast-stack/templates/tests/labelset-uniqueness.yaml`)
so you can run `helm test <release>` instead of curling by hand. The
hook polls VM for up to `tests.firstRefreshTimeout` (default `5m`) so
it tolerates the post-install delay before the first scheduled refresh
lands the sink push.

## Why operational metrics still carry source-side labels

The forecaster's operational metrics — `forecast_quality_score`,
`forecast_accuracy_mape`, `forecast_deviation_outside_band`,
`forecast_deviation_ratio`, `forecast_failures_total`, and the
threshold / growth-rate derived families — are emitted on `/metrics` and
inherit the same `(id, group, model, instance, …)` source labels as
the underlying series. This is intentional and aligned with design
principle 8 (label alignment with the sibling `promanomaly` project):
keeping the source `instance` (and other join keys) on the operational
metrics lets you write `and on (id, group, instance) <source_metric>`
PromQL joins without label-rewriting recording rules. The scrape labels
(`job="promforecast"`, `instance="<pod>:9091"`) sit alongside as
metadata; PromQL's `on (...)` joins are explicit about which labels
they care about, so the extra metadata does not break the join.

### Cross-metric joins: the labelset asymmetry

The sink-pushed curve families (`*_forecast`, `*_forecast_lower`,
`*_forecast_upper`) carry only source labels — **no `job`, no forecaster
`instance`**. The scraped operational families
(`forecast_quality_score`, `forecast_deviation_outside_band`, …) carry
source labels **plus** `job="promforecast"`. That asymmetry is by
design — it lets you join across them as long as you are explicit about
the join keys.

**Wrong** (relies on the default "everything matches" join):

```promql
# Returns zero results — RHS has ``job="promforecast"`` that LHS lacks,
# so the implicit full-labelset match never finds a peer.
forecast_deviation_outside_band == 1
  and forecast_quality_score > 0.6
```

**Right** (explicit `on(...)` restricts the join to keys present on both
sides):

```promql
forecast_deviation_outside_band{level="95"} == 1
  and on (id, group, model) forecast_quality_score > 0.6
```

The bundled example alerts in `examples/alerts/promforecast-rules.yaml`
use exactly this form. If you copy them and drop the `on(...)` clause
the rule silently matches zero series.

## Trade-off summary

| Concern | Snapshot only | Sink + VM (recommended) |
| --- | --- | --- |
| Cardinality footprint | one sample per series per scrape | one sample per series per *future step* per fit |
| Visual quality on Grafana | 5-minute staircase | smooth forward curve |
| Forecast history queryable as PromQL | only the latest gauge | full curve back to install time |
| Query cost | cheap (single point per series) | proportional to horizon × refresh density |
| Storage cost | minimal | grows with horizon and refresh interval |
| TSDB requirements | any Prometheus-compatible store | must accept future-timestamped writes (VM `-futureRetention`, Mimir, Thanos receive) |
| Setup complexity | one config block (snapshot is on by default) | sink config + future retention + scrape-drop |
| Dual-source risk | n/a | mitigated by scrape-drop |

The cardinality and storage figures cut both ways: the sink path is
strictly more expensive per emission, but it eliminates the need for
`offset <horizon>` Grafana tricks and gives you forecast history as a
queryable time-series instead of just a current-snapshot gauge.

## Cross-references

- The umbrella chart values that ship sink-first: `charts/promforecast-stack/values.yaml`.
- The standalone chart knob that derives the drop: `charts/promforecast/values.yaml` (`serviceMonitor.dropForecastSnapshots`).
- The example alert rules that assume sink labels: `examples/alerts/promforecast-rules.yaml`.
- The bundled dashboards that default to VM: `dashboards/grafana/`.
