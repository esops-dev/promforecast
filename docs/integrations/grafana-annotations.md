# Grafana annotations for forecast events

If you live in Grafana, you want significant forecast events to appear *on the graph* — a vertical marker when a change-point is detected, drift threshold crossed, or a threshold breach is predicted. This lets you correlate forecasts with operational events without manually overlaying timelines.

When enabled, the forecaster POSTs to Grafana’s annotations API on every enabled event that newly fires. Works with both Grafana OSS and Grafana Cloud.

## Configuring

Top-level `annotations:` block:

```yaml
annotations:
  enabled: true
  grafana_url: https://grafana.example.com
  api_token_ref: ${GRAFANA_API_TOKEN}     # resolved from environment
  events:
    - drift
    - change_point
    - deploy_risk
    - threshold_breach_predicted
    - deviation_outside_band
  timeout: 5s                              # default
```

- `grafana_url` is the base URL; posts go to `<grafana_url>/api/annotations`.
- `api_token_ref` resolves a `${ENV_VAR}` (Grafana service-account token, typically from a Kubernetes Secret).
- `events` selects which families post annotations. Empty list = none.

## Events

| `event_type`                  | Fires when… |
|-------------------------------|-------------|
| `drift`                       | Drift score crosses alert threshold |
| `change_point`                | Preprocessing detects a change-point |
| `deploy_risk`                 | Deploy-anchored risk score is active |
| `threshold_breach_predicted`  | A `*_forecast_will_breach` sample predicts a breach |
| `deviation_outside_band`      | Actual lands outside its forecast band |

Each annotation carries a millisecond timestamp, human-readable text, and tags including `promforecast`, the `event_type`, and `group:`/`id:` scoping tags.

## Rising-edge deduplication

Sustained conditions post **once** on the rising edge, not every refresh. When the condition clears and re-fires, a new annotation is posted. This keeps dashboards readable.

## Delivery telemetry

```
forecast_annotation_posts_total{event_type, outcome}   # outcome = success | http_error | timeout
```

Delivery failures never block fits — they are logged and the run continues.

## Operational notes

- Annotations are built at process start. Changes to `enabled`, `events`, or `grafana_url` require a restart.
- Rising-edge state is in-process. After restart (or HA leader failover), still-active events re-post once — acceptable for annotations that mark moments rather than continuous state.
