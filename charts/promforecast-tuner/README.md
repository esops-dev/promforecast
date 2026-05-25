# promforecast-tuner

Weekly Kubernetes `CronJob` that surfaces hyperparameter recommendations for the configured promforecast groups. Pushes one `forecast_recommended_param` sample per candidate that beats the configured baseline by `--min-improvement-pct` (default 5%).

The tuner reuses the live forecaster's container image (`promforecast tune ...` is a one-shot subcommand) and reads the same YAML config the live runner uses, so the baseline `(model, season_length)` values match exactly.

## Install

```bash
helm install promforecast-tuner \
  oci://ghcr.io/esops-dev/promforecast/charts/promforecast-tuner \
  --namespace promforecast \
  --set configMap.name=promforecast-config
```

## Values

See `values.yaml` for the full set of knobs and their defaults. The common ones:

| Key | Default | Description |
|---|---|---|
| `schedule` | `0 2 * * 0` | Cron expression. Weekly Sunday 02:00 UTC by default. |
| `configMap.name` | `promforecast` | Existing ConfigMap carrying the forecaster YAML. Defaults to the ConfigMap emitted by `helm install promforecast charts/promforecast`; override for other install shapes (umbrella: `<release>-promforecast`). |
| `sinkUrlOverride` | `""` | Override `sink.remote_write.url`. Inherits from the config when empty. |
| `minImprovementPct` | `5.0` | Minimum MAPE improvement to surface a candidate. |
| `lookback` | `null` | Prometheus duration override (e.g. `14d`). |
| `groups` | `[]` | Subset of groups to tune. Empty = all. |

## How to read the recommendations

See [`docs/operations/auto-tuning.md`](../../docs/operations/auto-tuning.md) for the "should I apply this recommendation?" decision guide.
