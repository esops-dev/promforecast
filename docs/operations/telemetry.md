# Forecaster telemetry

Two layers of operator-facing observability:

1. **Metrics** — always on. Per-(series, model) data-freshness, per-group
   run state, per-fit duration, per-reason failure counters; documented
   inline next to the family they describe (see
   [docs/README.md](../README.md) for the per-feature index). The new
   per-series staleness gauge is described below; the rest live next to
   the feature that emits them.
2. **Tracing (OpenTelemetry)** — opt-in. Spans per group / query / fit;
   shipped via OTLP-gRPC to whatever collector you already run.

## Per-series staleness

`forecast_data_staleness_seconds{id, group, model, ...source_labels}`
is a per-(series, model) gauge of seconds since the last successful
forecast emission. It's the per-series complement to the group-level
`forecast_last_run_timestamp_seconds`:

| Signal                                          | Question it answers                          |
|-------------------------------------------------|----------------------------------------------|
| `forecast_last_run_timestamp_seconds{group}`    | Has *the group* refreshed recently?          |
| `forecast_data_staleness_seconds{id, group, model}` | Is *this specific series* still being fit? |

The group-level gauge keeps ticking as long as *any* series of the
group is fit successfully, so a single series whose model crashes
silently is invisible there. The staleness clock tracks the per-(series,
model) emission cadence — successful fits reset it to zero, failed
fits leave the previous timestamp alone so the gauge value grows.

Cardinality: `N_series × N_models` per opted-in query — already
counted as part of the forecast emission, so no new cardinality
budget is required.

The shipped `examples/alerts/promforecast-rules.yaml` carries a
`ForecastSeriesStale` alert keyed on this gauge:

```yaml
- alert: ForecastSeriesStale
  expr: forecast_data_staleness_seconds > 14400   # 4h, ≈ 4 × default refresh
  for: 15m
  labels:
    severity: warning
```

Tune the threshold to four-or-so times your `server.refresh_interval`
— short enough to catch a single missed cycle, long enough to absorb
transient backoff. Pair with `forecast_failures_total` filtered by
`group` + `query` for the root cause.

### Never-succeeded series

The gauge is **absent** until a (series, model) tuple's first
successful fit — the clock is only started on success. A series that
fails on its very first attempt and never succeeds therefore won't
trip `ForecastSeriesStale` because there's no sample for the alert
to bind to. Pair `ForecastSeriesStale` with two complementary signals
for that case:

* `forecast_failures_total{reason=~"fit_error|fit_timeout|...",group="..."} > 0`
  — counts every per-series fit attempt that raised. Use the
  [`/forecast/warmup`](warmup.md) endpoint on a fresh install to
  distinguish "still warming up" from "consistently failing".
* `forecast_series_dropped_total{reason="min_points",group="..."} > 0`
  — counts series that didn't even get a fit attempt because the
  TSDB doesn't have enough points yet.

We picked "absent until first success" deliberately: a synthetic
`+Inf` starting state would either fire the alert for every fresh
install (operator surprise) or hide behind a `for: <warmup>` clause
that masks real first-attempt failures.

# OpenTelemetry tracing

Opt-in distributed tracing for the full query → fit → export pipeline.  
Useful when metrics alone don’t explain a slow group, stuck regressor, or runaway fit.

## Configuring

```yaml
telemetry:
  enabled: true
  service_name: promforecast          # OTel resource service.name
  sample_ratio: 0.1                   # head-based sampling ratio (0..1)
  otlp:
    endpoint: otel-collector:4317     # OTLP/gRPC endpoint
    insecure: true                    # cleartext (common inside cluster)
    headers: {}                       # optional auth/tenant headers
    timeout: 10s
```

Requires the optional `[otel]` extra:

```bash
pip install 'promforecast[otel]'
```

The official container image ships with it; enabling the feature needs no custom build.

## When disabled

`telemetry.enabled: false` (default) is **completely free**:
- The `opentelemetry` package is never imported.
- Every `Tracer.span(...)` becomes a no-op (`contextlib.nullcontext`).
- No extra dependencies are required.

This zero-cost path is explicitly tested in `forecaster/tests/test_telemetry.py`.

## Spans emitted

| Span                     | Attributes                                      | Context |
|--------------------------|-------------------------------------------------|---------|
| `promforecast.group_run` | `group`, `query_count`                          | Each group refresh tick |
| `promforecast.query`     | `group`, `query`                                | Each PromQL fetch (one per rendered template for discovery) |
| `promforecast.fit`       | `model`, `horizon`, `freq`, `season_length`     | Every `fit_predict` (forecast + each backtest) |

Exceptions are automatically recorded; errored spans are marked `ERROR` with the exception message.

## Sampling

`sample_ratio` is a head-based sampler.  
- `1.0` traces everything (great for dev).  
- `0.01` traces 1 % (typical production).  

Parent-based context propagation ensures all child spans of a sampled group run stay together.

## SDK missing

If `telemetry.enabled: true` but the `[otel]` extra is not installed, the forecaster logs a clear warning and continues without tracing:

```json
{
  "event": "telemetry_disabled_sdk_missing",
  "error": "No module named 'opentelemetry'",
  "hint": "install the optional dependency: pip install 'promforecast[otel]'"
}
```

Tracing is observability, never a correctness requirement.

## Lifecycle

Tracing initialises once at boot in `main.py`.  
The FastAPI lifespan teardown calls `tracer.shutdown()` to flush the `BatchSpanProcessor`.  
A normal Kubernetes 30 s SIGTERM grace period is sufficient.

**Not hot-reloadable.** Changing `telemetry.*` via `/-/reload` has no effect on the running tracer. Restart the pod to pick up new settings (deliberate trade-off to avoid losing in-flight spans or double-initialisation issues).

## Non-goals

- Metrics over OTLP — Prometheus exposition on `/metrics` remains the metric API.
- Auto-instrumentation — only the three manual spans above are emitted. Full `opentelemetry-instrumentation-*` packages are deliberately omitted to keep span volume low and focused.
