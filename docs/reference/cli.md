# CLI reference

The `promforecast` binary provides seven subcommands: `run`,
`validate`, `adapter`, `diagnose` (with `duplicates` and `sample-rate`
sub-subcommands), `inspect`, `warmup`, and `backfill`.
Running `promforecast --config foo.yaml` is identical to `promforecast run --config foo.yaml`, so existing container entrypoints continue to work unchanged.

## `promforecast run`

Starts the forecaster: parses and validates the YAML config, builds the model registry, and launches the FastAPI server on `server.listen` (default `:9091`).

Exposed endpoints:

| Endpoint       | Method | Purpose                                                              |
|----------------|--------|----------------------------------------------------------------------|
| `/healthz`     | GET    | Liveness probe; also returns `config_hash` and last-reload telemetry |
| `/metrics`     | GET    | Prometheus exposition format                                         |
| `/-/reload`    | POST   | Validate-then-reload config (atomic swap)                            |
| `/config`      | GET    | Sanitised JSON view of the running config (opt-in)                   |

**Reload behavior**  
- `SIGHUP` triggers the same reload as `/-/reload`.  
- When `server.reload.watch_configmap: true` (default), the `--config` file is polled every second. Any content change triggers an automatic reload.  
- Kubernetes ConfigMap projections are atomic, so `kubectl apply` is enough — no pod restart required.  
- A failed reload leaves the previous config running and increments `forecast_config_reload_failures_total`. The HTTP endpoint returns `400` with the validation error.

**Config identity surfaces**

The forecaster exposes the running config's identity in three places so operators can confirm a deployment is running the expected revision without `kubectl exec`:

- `forecast_config_hash{sha256="<hex>"} 1.0` on `/metrics` (info-style gauge; family is omitted before the first successful load).
- `forecast_config_last_reload_timestamp_seconds`, `forecast_config_reload_success`, and `forecast_config_reload_failures_total` on `/metrics` for reload telemetry.
- `/healthz` returns `{ "status": "ok", "config_hash": "...", "config_last_reload_timestamp": <unix>, "config_last_reload_success": true }`.
- `/config` (opt-in via `server.expose_config_endpoint: true`) returns the parsed config as JSON with `datasource.auth.*` and `sink.remote_write.headers` redacted. The endpoint is unauthenticated like `/metrics`, so only enable it when the metrics port is on a trusted network.

A successful reload refreshes the hash to the new file's digest; failed reloads keep the previous hash so `/metrics` always reflects what the runner is actually executing.

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
  contribution lines    1000
  operational lines       21
  total                34621
  WARN: total exceeds safety.max_total_series (5000); series_overflow will drop the excess
```

## `promforecast diagnose`

Operator diagnostics that run against the long-term TSDB rather than the
config file. Two subcommands:

### `promforecast diagnose duplicates`

```bash
promforecast diagnose duplicates \
    [--datasource <url> | --config <path>] \
    [--selector '{...}'] [--since <duration>] [--json] \
    [--bearer-token <token>] [--basic-user <user> --basic-password <pass>] \
    [--tls-cert <path> --tls-key <path>] [--tls-ca <path>] [--tls-insecure-skip-verify]
```

Scans the configured TSDB for forecast metrics
(`*_forecast` / `*_forecast_lower` / `*_forecast_upper`) that appear
under more than one labelset family — the dual-source state described
in [the emission paths reference](emission-paths.md). Exit codes
are CI-friendly: `0` for clean, `1` for duplicates detected, `2` for
datasource unreachable.

### `promforecast diagnose sample-rate`

```bash
promforecast diagnose sample-rate \
    [--datasource <url> | --config <path>] \
    [--window <duration>] [--max-samples <N>] \
    [--selector '{...}'] [--json] \
    [auth flags as above]
```

Probes the per-series sample rate via
`count_over_time(<metric>[<window>])` and flags any series whose count
over the window exceeds `--max-samples`. Covers the same-labelset case
`duplicates` cannot see from labelset shape alone — two ingest paths
writing to the **same** series produce a count higher than either path
alone would.

See [`../operations/diagnose.md`](../operations/diagnose.md) for the full
reference, authentication recipes, in-cluster invocation
(`kubectl exec`), and CI-integration patterns.

## `promforecast inspect`

```bash
promforecast inspect \
    --config <path> --query <id> \
    [--series-filter '{label="value",...}'] \
    [--horizon <duration>] [--lookback <duration>] [--model <name>] \
    [--backtest] [--json]
```

Debugs a single weird forecast: fetches the configured series from
the TSDB, runs the configured fit pipeline exactly as the live
forecaster would, and prints the inputs, the fit, the forecast curve,
and (with `--backtest`) the per-fold backtest predictions vs.
actuals. See [`../operations/inspect.md`](../operations/inspect.md)
for debugging walkthroughs.

Exit codes: `0` on success, `1` for inspector-level failures (unknown
query, ambiguous series, fit error), `2` for datasource / network
failures.

## `promforecast warmup`

```bash
promforecast warmup \
    [--url http://forecaster:9091] [--timeout 30] [--json]
```

Calls a running forecaster's `GET /forecast/warmup` endpoint and
prints per-(group, query) "what is still loading?" status. The
endpoint must be enabled via `server.expose_warmup_endpoint: true`;
see [`../operations/warmup.md`](../operations/warmup.md).

Exit codes: `0` when every query reports `status=ready`, `1` when at
least one is still warming or errored, `2` for network failures
(suitable for a `until` loop in a pre-flight script).

## `promforecast backfill`

```bash
promforecast backfill \
    --config <path> --from <duration> \
    [--to now|<iso-8601>] [--groups GROUP1 GROUP2 ...] [--sink-url <url>]
```

Replays the fit pipeline at simulated past timestamps and pushes the
resulting forecast curves + accuracy summaries to the configured
`sink.remote_write` URL with explicit historical `timestamp_ms`
values. Use right after `helm install` so SLO / quality dashboards
populate from day one. The window is capped by
`safety.backfill_max_window` (default `90d`); see
[`../operations/backfill.md`](../operations/backfill.md) for the
resource-sizing trade-off.

Exit codes: `0` on success, `1` for invalid input (over-budget
window, no configured sink), `2` for TSDB / network failures.
