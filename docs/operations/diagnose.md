# `promforecast diagnose`

Operator diagnostics that query the long-term TSDB (not just the config file).  
Two subcommands are currently available: `duplicates` (labelset-shape detection) and `sample-rate` (same-labelset blind spot).

Both subcommands share common authentication and output flags.

## `duplicates`

```bash
promforecast diagnose duplicates \
    [--datasource <url> | --config <path>] \
    [--selector '{...}'] \
    [--since <duration>] \
    [auth flags] \
    [--json]
```

Detects forecast metrics (`*_forecast`, `*_forecast_lower`, `*_forecast_upper`) that appear under more than one distinct labelset family — the classic **dual-source state** caused by running both the `/metrics` scrape and `remote_write` sink into the same TSDB.

### Exit codes

| Code | Meaning                                              | Typical use |
|------|------------------------------------------------------|-------------|
| `0`  | No duplicate labelset families found                 | CI / pre-flight pass |
| `1`  | At least one forecast metric has duplicate families  | CI / pre-flight fail |
| `2`  | Datasource unreachable or malformed response         | Infra / transient error |

### Flags

| Flag              | Purpose |
|-------------------|---------|
| `--datasource`    | TSDB base URL (e.g. `http://victoriametrics:8428`). Required unless `--config` is used. |
| `--config`        | Path to forecaster YAML (inherits `datasource.url`). |
| `--selector`      | Optional PromQL label selector (e.g. `'{cluster="prod"}'`). |
| `--since`         | Lookback window (e.g. `6h`, `2d`). |
| `--max-concurrent`| Max in-flight requests (default `8`). |
| `--json`          | Machine-readable JSON output. |

### What it detects

It groups series by their **set of label names** (shape, not values).  
Flags a metric when one family is scrape-shaped (contains `job`, `pod`, `replica`, etc.) and another is sink-shaped (lacks those labels).

Intra-source variance (e.g. `calibrated="true"`) is **not** flagged.

### Text output example

```text
FAIL  1 of 27 forecast metric(s) appear under more than one labelset family.

  node_cpu_busy_pct_forecast:
    [sink-shaped]   count=4   keys=['group', 'instance', 'model', ...]
    [scrape-shaped] count=4   keys=['cluster', 'group', 'instance', 'job', 'replica', ...]

Remediation:
  1. Drop forecast metrics from the forecaster scrape job (recommended).
  2. Or enable the chart-managed drop.
  3. Clean historical scrape-shaped samples with a delete_series call.
```

### JSON output

```json
{
  "has_duplicates": true,
  "metrics_scanned": 27,
  "findings": [
    {
      "metric": "node_cpu_busy_pct_forecast",
      "is_duplicate": true,
      "families": [ ... ]
    }
  ]
}
```

### CI usage

```yaml
- name: Check for dual-source state
  run: |
    promforecast diagnose duplicates --datasource "$VM_URL" --since 1h --json > result.json
    test "$(jq -r '.has_duplicates' < result.json)" = "false"
```

**Note:** `duplicates` catches distinguishable labelset cases. Use `sample-rate` for the silent same-labelset case (most common with `honor_labels: true`).

## `sample-rate`

```bash
promforecast diagnose sample-rate \
    [--datasource <url> | --config <path>] \
    [--window 10m] \
    [--max-samples 5] \
    [--selector '{...}'] \
    [auth flags] \
    [--json]
```

Detects series receiving samples from multiple ingest paths by counting samples over a recent window.

### Key flags

- `--window`: Lookback (default `10m`).
- `--max-samples`: Per-series cap (default `5`). Series exceeding this are flagged.

### Output (text)

```text
FAIL  4 of 312 series exceeded the configured sample-rate cap.

  node_cpu_busy_pct_forecast:
    count=41  labels={...}
```

**Tuning tip:** Raise `--max-samples` for very fine `step` or during catch-up after long stalls.

## Authentication

Both commands support:

- `--bearer-token` / `$PROMFORECAST_BEARER_TOKEN`
- `--basic-user` + `--basic-password`
- `--tls-cert` / `--tls-key`, `--tls-ca`, `--tls-insecure-skip-verify`

## Running in-cluster

```bash
# Preferred: exec into pod
kubectl exec deploy/promforecast -- promforecast diagnose duplicates --datasource http://victoriametrics:8428

# One-shot pod
kubectl run promforecast-diagnose --rm -it --image=ghcr.io/... \
  --env=PROMFORECAST_BEARER_TOKEN=... \
  -- promforecast diagnose duplicates --datasource http://victoriametrics:8428
```

## See also

- [Emission paths](../architecture/emission-paths.md) — why dual-source is a problem.
- [Migrate to sink-first](../how-to/migrate-to-sink-first.md) — recommended remediation.
