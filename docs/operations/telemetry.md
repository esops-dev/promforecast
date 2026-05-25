# Forecaster telemetry

Two layers of observability:

1. **Metrics** — always on. Per-series freshness, per-group run state, per-fit duration, failure counters, and the new per-series staleness gauge (documented below). Most families are described next to the feature that emits them.
2. **Tracing** — opt-in OpenTelemetry spans for the full query → fit → export pipeline.

## Snapshot age (per group)

`forecast_snapshot_age_seconds{group}` is a server-side computed gauge of seconds since the most recent successful run for the group — the same signal as `time() - forecast_last_run_timestamp_seconds` but pre-subtracted so dashboards and the `ForecastSnapshotStale` alert can read a single gauge instead of a PromQL subtraction. When the group has never produced a snapshot the value is 0; alert authors gating on the value should pair it with `forecast_last_run_timestamp_seconds > 0` to avoid treating "never ran" as healthy.

The [degraded-mode playbook](degraded-modes.md) walks through the five operational failure modes this gauge surfaces, including the bounded sink-failure and source-read counters that distinguish the underlying causes.

## Per-series staleness

`forecast_data_staleness_seconds{id, group, model, ...}` is a per-(series, model) gauge of seconds since the last successful forecast emission.

It complements the group-level `forecast_last_run_timestamp_seconds`:

| Signal | Question it answers |
|--------|---------------------|
| `forecast_last_run_timestamp_seconds{group}` | Has the *group* refreshed recently? |
| `forecast_data_staleness_seconds{id, group, model}` | Is *this specific series* still being fit? |

The staleness clock starts at the first successful fit for that (series, model) tuple and resets on every subsequent success. Failed fits leave the clock running, so the gauge grows until the series recovers.

**Shipped alert** (`examples/alerts/promforecast-rules.yaml`):

```yaml
- alert: ForecastSeriesStale
  expr: forecast_data_staleness_seconds > 14400   # ~4 × default refresh interval
  for: 15m
  labels:
    severity: warning
```

Tune the threshold to roughly four times your `server.refresh_interval`.

### Never-succeeded series

The gauge is **absent** until the first successful fit. This prevents false positives on fresh installs. Use these complementary signals for first-attempt failures:

- `forecast_failures_total{reason=~"fit_error|fit_timeout|...", group="..."} > 0`
- `forecast_series_dropped_total{reason="min_points", group="..."} > 0`

For a fresh-install overview, run `promforecast warmup`.

## Self-observability metrics

Three families track the forecaster’s own internal health:

| Metric | Question it answers |
|--------|---------------------|
| `forecast_model_fit_success_ratio{group, model, window}` | What fraction of recent fits for this model in this group succeeded? (1h / 24h windows) |
| `forecast_regressor_stability_score{group, query, regressor_id}` | Has this regressor’s recent value distribution drifted from its lookback baseline? |
| `forecast_group_memory_bytes{group}`<br>`forecast_group_cpu_seconds_total{group}` | Which group dominates memory or CPU usage? |

- **Fit success ratio** rolls successes and failures into rolling windows. A value below ~0.9 for more than one refresh cycle usually indicates a model silently failing on a subset of series.
- **Regressor stability** is a 0–1 score (1.0 = no drift). Low scores point to noisy regressors that degrade conditional forecasts without raising fit errors.
- **Group resource attribution** gives comparative (not absolute) memory and CPU per group.

## Service-level health composite

`forecast_service_health_score{<aggregate_by>, aggregator, weight_by, series_count}` is the per-bucket aggregate of `forecast_quality_score` defined under the operator-configured [`governance.health_score`](../governance/forecast-health.md) block.

| Label              | Meaning                                                                                              |
|--------------------|------------------------------------------------------------------------------------------------------|
| `<aggregate_by>`   | One label per dim listed in `governance.health_score.aggregate_by` (e.g. `service`, `environment`).  |
| `aggregator`       | Currently always `weighted_mean` — documents which aggregation produced the value.                   |
| `weight_by`        | `uniform` or `forecast_series_count` — documents the weight interpretation.                          |
| `series_count`     | Number of `forecast_quality_score` rows that contributed to the bucket.                              |

Off until the block is configured. Use to surface "our forecasts are X% reliable today" per service/environment on an executive dashboard. See [governance/forecast-health.md](../governance/forecast-health.md) for the worked recipe.

## Right-sizing recommendations

Three sibling gauges describe the per-workload sizing recommendation derived from the forecast peak:

| Metric                                       | Meaning                                                                                                          |
|----------------------------------------------|------------------------------------------------------------------------------------------------------------------|
| `forecast_recommended_resource_size{workload, kind, confidence}` | Recommended limit/request. Sample value is in the metric's native units (CPU cores, memory bytes). |
| `forecast_recommended_resource_current{workload, kind}` | Operator-reported current limit/request (only when `current_promql` is wired). |
| `forecast_recommended_resource_savings_pct{workload, kind, confidence}` | `(current − recommended) / current * 100`. Positive = safe to downsize; negative = upsize required. |

The `confidence` label categorises the underlying `forecast_quality_score` (`high|medium|low`). Off until [`recommendations.resource_sizing.enabled: true`](right-sizing.md) is set; the quality gate uses the **mean** of finite scores for the workload's `forecast_query`.

## Cost of action on capacity thresholds

`forecast_threshold_action_cost{id, group, name, type}` is a config-derived gauge whose sample value is the operator-configured cost. The `type` label is free text (`usd`, `eur`, `engineer_hours`, ...) so non-monetary cost framings coexist with currency ones in a single family. Tied to threshold lifecycle: a threshold removed from YAML drops its row on the next refresh. See [outputs/capacity.md](../outputs/capacity.md#ranking-by-cost-of-action) for the join with `forecast_time_to_threshold_seconds`.

## Drift change-correlation explainer

`forecast_drift_explained_total{group, query, primary_suspect}` counts threshold-crossing events on `forecast_drift_score`. The `primary_suspect` label names the regressor most likely to explain the drift — see [operations/drift-explained.md](drift-explained.md). Always paired with a `forecast_drift_explained` structured log event carrying every regressor value at the drift moment, the rolling Pearson correlation, the `last_config_reload_timestamp`, and the cumulative `change_point_count` so the investigation hypothesis isn't limited to regressor changes.

## OpenTelemetry tracing

Opt-in distributed tracing for the query → fit → export pipeline. Useful when metrics alone don’t explain a slow group, stuck regressor, or runaway fit.

### Configuring

```yaml
telemetry:
  enabled: true
  service_name: promforecast
  sample_ratio: 0.1
  otlp:
    endpoint: otel-collector:4317
    insecure: true
    headers: {}
    timeout: 10s
```

Requires the optional `[otel]` extra (`pip install 'promforecast[otel]'`). The official container image ships with it.

### When disabled

`telemetry.enabled: false` (default) is completely free — no package is imported, every span is a no-op.

### Spans emitted

| Span                     | Attributes                                      | Context |
|--------------------------|-------------------------------------------------|---------|
| `promforecast.group_run` | `group`, `query_count`                          | Each group refresh |
| `promforecast.query`     | `group`, `query`                                | Each PromQL fetch |
| `promforecast.fit`       | `model`, `horizon`, `freq`, `season_length`     | Every `fit_predict` |

Exceptions are automatically recorded and marked `ERROR`.

### Sampling

Head-based sampler controlled by `sample_ratio`. Parent-based context propagation keeps child spans together.

### SDK missing

If the `[otel]` extra is absent, the forecaster logs a clear warning and continues without tracing.

### Lifecycle

Initialised once at boot. FastAPI lifespan teardown flushes the `BatchSpanProcessor`. A normal Kubernetes 30 s SIGTERM is sufficient.

**Not hot-reloadable** — changing `telemetry.*` requires a pod restart.

### Non-goals

- Metrics over OTLP (Prometheus exposition on `/metrics` remains the metric API).
- Auto-instrumentation (only the three manual spans above are emitted).
