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
