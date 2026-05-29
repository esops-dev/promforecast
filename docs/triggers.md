# Forecast-driven webhook triggers

Triggers let the forecaster **push** a structured event to a webhook when a PromQL condition over forecast metrics becomes true — e.g. auto-archive cleanup when a disk-fill ETA drops below a day, a Slack post when error-budget burn accelerates, or a CI job when drift is detected.

**How this differs from alternatives**:
- vs. **Alertmanager / PrometheusRule** — the payload carries full forecast context (matched series, values, model identity, band/drift labels), not just “an alert fired”.
- vs. **[webhook regressors](inputs/regressors.md#webhook-regressors)** — the direction is reversed: regressors *pull* signals in; triggers *push* events out.

## Configuring

Top-level `triggers:` list:

```yaml
triggers:
  - name: disk-fill-imminent
    condition: 'node_filesystem_avail_bytes_forecast{horizon="24h"} < 0'
    webhook_url: https://hooks.slack.com/services/XXX/YYY/ZZZ
    repeat_interval: 1h          # default
    timeout: 5s                  # default
    headers:                     # optional static auth
      Authorization: Bearer ${SLACK_TOKEN}
    payload_template: |
      {"text": "Disk fill predicted within 24h (value $value)"}
```

- `condition` is a PromQL expression evaluated after group runs (re-evaluations that land within a few seconds of each other — e.g. several groups finishing one refresh cycle together — are coalesced into a single instant query, so trigger evaluation doesn't scale with group count).
- The trigger fires on false → true transitions. Sustained-true conditions re-fire at most once per `repeat_interval`.

## Payload

With no `payload_template`, the forecaster POSTs:

```json
{
  "name": "disk-fill-imminent",
  "condition": "...",
  "fired_at": "2026-05-29T12:00:00+00:00",
  "value": -1.2e9,
  "labels": {"instance": "node-7", "horizon": "24h", "model": "AutoARIMA"},
  "samples": [{"labels": {...}, "value": -1.2e9}]
}
```

`value` / `labels` come from the first matched series; `samples` carries the full matched vector.

### Templating

`payload_template` supports `$`-style placeholders:

```yaml
payload_template: '{"text": "$name fired: value=$value"}'
```

Available variables: `$name`, `$condition`, `$value`, `$fired_at`, `$samples_json`.

## Authentication

Static headers use the `headers:` block. Header values support `${ENV_VAR}` expansion against the process environment (injected from a Kubernetes Secret via `secretKeyRef`), so `Authorization: Bearer ${SLACK_TOKEN}` is resolved at delivery time rather than sent literally. An unset variable expands to an empty string.

For end-to-end verification, add a signing block:

```yaml
signing:
  algorithm: hmac-sha256
  secret_ref: ${TRIGGER_SIGNING_SECRET}
```

The forecaster adds `X-Promforecast-Signature: sha256=<hex>`. Receivers recompute the HMAC over the raw body.

## Delivery telemetry

```
forecast_trigger_fires_total{name, outcome}   # outcome = success | http_error | timeout
```

Delivery failures never block fits or other triggers.

## Example receivers

**Slack** — point `webhook_url` at a Slack incoming-webhook and use a simple `payload_template`.

**Generic JSON receiver** — leave `payload_template` empty and verify the signature if configured.

## Operational notes

- Triggers are built at process start. Config changes require a restart.
- Conditions are evaluated after each group run.
- One-cycle lag is possible (TSDB ingestion).
- Evaluation or delivery errors are logged and never block the fit pipeline.
