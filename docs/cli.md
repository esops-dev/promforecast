# CLI reference

The `promforecast` binary provides two subcommands.  
Running `promforecast --config foo.yaml` is identical to `promforecast run --config foo.yaml`, so existing container entrypoints continue to work unchanged.

## `promforecast run`

Starts the forecaster: parses and validates the YAML config, builds the model registry, and launches the FastAPI server on `server.listen` (default `:9091`).

Exposed endpoints:

| Endpoint       | Method | Purpose                                      |
|----------------|--------|----------------------------------------------|
| `/healthz`     | GET    | Liveness probe                               |
| `/metrics`     | GET    | Prometheus exposition format                 |
| `/-/reload`    | POST   | Validate-then-reload config (atomic swap)    |

**Reload behavior**  
- `SIGHUP` triggers the same reload as `/-/reload`.  
- When `server.reload.watch_configmap: true` (default), the `--config` file is polled every second. Any content change triggers an automatic reload.  
- Kubernetes ConfigMap projections are atomic, so `kubectl apply` is enough — no pod restart required.  
- A failed reload leaves the previous config running and increments `forecast_config_reload_failures_total`. The HTTP endpoint returns `400` with the validation error.

## `promforecast validate`

```bash
promforecast validate --config <file> [--dry-run] [--datasource-url URL]
```

Validates a config file without starting the server. Two modes:

* **Schema check (default)**  
  Parses YAML, runs the full pydantic validator, enforces Prometheus metric-name rules (no colons), and prints an *upper-bound* cardinality estimate.  
  Exit code `0` on success, `1` on any failure.

* **`--dry-run`**  
  In addition to the schema check, it issues a sample query against the datasource (or the URL supplied via `--datasource-url`), fits the first model on the first series, and prints the first 10 forecast points plus a single-horizon backtest MAPE.  
  Exit code `2` if the dry-run itself fails (query timeout, fit error, etc.) while the schema passed.

### Use in CI

```yaml
- name: Validate forecaster config
  run: |
    pip install promforecast
    promforecast validate --config configs/forecaster.yaml
```

### Cardinality estimate

The estimate is intentionally pessimistic: it assumes every PromQL query returns the full `safety.max_series_per_query` series.  
Real-world configs usually emit far fewer. The goal is to catch configurations that *cannot* possibly stay under the global limit *before* they reach production.

**Sample output**

```
OK  config valid: configs/forecaster.yaml
    groups:  3
    queries: 8
    models:  AutoARIMA, SeasonalNaive

Estimated cardinality (worst case at max_series_per_query):
  forecast lines        8000  = series(500) * models(2) * (1 + 2*levels(2)) * queries(8)
  accuracy lines       16000
  deviation lines       8000
  quality lines         1600
  operational lines       21
  total                33621
  WARN: total exceeds safety.max_total_series (5000); series_overflow will drop the excess
```
