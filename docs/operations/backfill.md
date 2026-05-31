# Historical backfill

`promforecast backfill` replays the fit pipeline at simulated past timestamps and pushes the resulting forecast curves + accuracy summaries to the configured `sink.remote_write` URL with explicit historical timestamps.

Use it right after a fresh install (or after a TSDB volume loss) so dashboards built on `forecast_quality_score`, `forecast_accuracy_mape`, and other rolling families become useful immediately instead of waiting weeks for natural accumulation.

## When to use it

- Fresh install on a TSDB that already holds weeks of source data.
- After a TSDB restore or volume loss.
- When extending a quality dashboard’s lookback window.

Backfill is **not** for replaying missed alerts or correcting historical predictions — the sink relies on last-write-wins per `(labels, timestamp)` so a second run simply overwrites the same points. This requires VictoriaMetrics dedup to be enabled (`dedup.minScrapeInterval`, set by the `promforecast-stack` chart); without it the rewritten points accumulate as duplicate samples rather than overwriting. See the chart's [Deduplication](../../charts/promforecast-stack/README.md#deduplication) note.

## CLI usage

```bash
promforecast backfill \
  --config /etc/promforecast/config.yaml \
  --from 30d \
  --to now \
  --groups node_capacity http_slo
```

### Flags

| Flag          | Purpose |
|---------------|---------|
| `--config`    | Required. Same YAML the live runner uses. |
| `--from`      | Required. How far back to walk (e.g. `30d`, `14d`). |
| `--to`        | End timestamp (`now` or ISO-8601). |
| `--groups`    | Optional space-separated list. Defaults to all groups. |
| `--sink-url`  | Override `sink.remote_write.url`. |

Exit codes: `0` success, `1` invalid input, `2` TSDB/network failure.

## Cost

The backfill cadence matches each group’s effective refresh interval. Dominant cost is model fits.

| Window | Refresh | Fits per group | Single-core wall time (SeasonalNaive) | Wall time (AutoARIMA) |
|--------|---------|----------------|---------------------------------------|------------------------|
| 7d     | 1h      | 168            | ≈ 1 min                               | ≈ 5–10 min             |
| 30d    | 1h      | 720            | ≈ 5 min                               | ≈ 30–60 min            |
| 90d    | 1h      | 2160           | ≈ 15–30 min                           | ≈ 2–3 hours            |
| 30d    | 5m      | 8640           | ≈ 1 hour                              | ≈ 6–12 hours           |

See the example Job in `examples/k8s/backfill-job.yaml` for a ready-to-run manifest.

## Safety cap

`safety.backfill_max_window` (default `90d`) is enforced at the CLI boundary so an accidental `--from 1y` fails fast. Raise it explicitly when needed:

```yaml
safety:
  backfill_max_window: 365d
```

## Audit trail

After each group the backfill emits one gauge:

```
forecast_backfill_fits_last{group} <number of fits>
```

It records “this is how many fits the most recent backfill issued for this group”.

## VM futureRetention

Backfill writes historical timestamps (accepted by default in VictoriaMetrics).  
Forecast curves still extend into the future, so the usual `futureRetention=2d` (or higher than `defaults.horizon`) still applies.

## What backfill does **not** emit

- Operational counters (`forecast_failures_total`, `forecast_run_duration_seconds`, leader-election, deploy-risk, etc.).
- Conditional forecasts, decomposition, drift gauges.
- Deviation ratios / outside-band flags (these are tied to live actuals).
- Band-coverage rolling fractions.

These families describe live execution state and would be misleading when replayed at past timestamps.

## Operational notes

- Backfill is single-process and runs once (no scheduler).
- A config reload or change to `deploy_signals` resets active state on the next run.
- The example Kubernetes Job uses the same image as production and has `backoffLimit: 0` so a crashed walk can be retried with a narrower window.
