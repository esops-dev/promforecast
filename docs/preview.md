# What-if forecast preview endpoint

The forecaster exposes an optional `POST /forecast/preview` endpoint for ad-hoc “what-if” forecasts.  
Pick any configured query, optionally override regressors/horizon/lookback, and receive a clean JSON forecast curve.  

The endpoint is **stateless**: nothing is written to `/metrics`, no snapshot is mutated, and the scheduled fits continue uninterrupted. Per-request CPU is capped by `safety.fit_timeout`.

## Enabling

```yaml
server:
  expose_preview_endpoint: true   # default: false
```

The endpoint follows the same network-level access model as `/-/reload` (NetworkPolicy + service exposure).

## Request body

```json
{
  "query_id": "requests_per_second",
  "horizon": "30m",
  "lookback": "1h",
  "model": "AutoARIMA",
  "series_selector": {"instance": "host-a"},
  "regressor_overrides": {
    "deploys": 0,
    "traffic_factor": [0.1, 0.2, 0.3, 0.4]
  }
}
```

| Field                  | Required | Format / notes |
|------------------------|----------|----------------|
| `query_id`             | Yes      | Must match a running query `id` |
| `horizon`              | No       | Prometheus duration or seconds; defaults to `defaults.horizon` |
| `lookback`             | No       | Same format; defaults to `defaults.lookback` |
| `model`                | No       | One of the configured models; defaults to the first entry |
| `series_selector`      | No       | Label map to pick exactly one series when the query returns multiple |
| `regressor_overrides`  | No       | Regressor id → scalar or list (length must match in-sample series) |

Unknown regressor keys are silently ignored.

## Response

```json
{
  "query_id": "requests_per_second",
  "group": "kpis",
  "model": "AutoARIMA",
  "served_by": "promforecast-7d8b6f5c-abcd",
  "timestamps": ["2026-05-22T14:00:00", "2026-05-22T14:05:00", ...],
  "yhat": [1234.5, 1240.2, ...],
  "bands": {
    "80": { "lower": [...], "upper": [...] },
    "95": { "lower": [...], "upper": [...] }
  }
}
```

`served_by` shows the responding replica (useful in HA mode for log correlation).

## HA behaviour

The preview endpoint is available on **any replica** (leader or follower). It performs a one-off fit that never interferes with scheduled work or the shared Redis snapshot.

## Errors

| HTTP | Reason               | Trigger |
|------|----------------------|---------|
| 400  | `invalid_body`       | Malformed JSON or bad override types |
| 400  | `ambiguous_series`   | Multiple series returned and no (or non-unique) `series_selector` |
| 404  | `unknown_query`      | No query with the given `id` |
| 404  | `no_data`            | TSDB query returned zero series |
| 502  | `query_failed`       | TSDB error (non-timeout) |
| 500  | `fit_failed`         | Model raised during `fit_predict` |
| 504  | `query_timeout`      | TSDB request exceeded `safety.query_timeout` |
| 504  | `fit_timeout`        | Fit exceeded `safety.fit_timeout` |

Error body: `{"error": "<reason>", "message": "<details>"}`.

## Worked example: “what if traffic doubles?”

```bash
curl -s -X POST http://promforecast:9091/forecast/preview \
  -H 'Content-Type: application/json' \
  -d '{
    "query_id": "requests_per_second",
    "regressor_overrides": {"traffic_factor": 2.0}
  }' | jq '.yhat[0:5]'
```

## Building a what-if dashboard

Use a Grafana dashboard with a “scenario” variable + the [JSON Datasource plugin](https://github.com/simPod/grafana-json-datasource). The response shape is directly compatible with the Time series panel after a small adapter converts the JSON arrays into a table.

## Not in scope

- Mutating previews (nothing is ever written to `/metrics` or `remote_write`).
- Authentication (same network-level model as `/-/reload`).
- Multi-series previews (one `query_id` per request).
