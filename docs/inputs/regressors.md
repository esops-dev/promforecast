# Exogenous regressors

A regressor is an extra input fed to the forecasting model alongside the primary series. Three flavours are supported:

- **`calendar`** — deterministic features derived from timestamps (`hour_of_day`, `day_of_week`, `is_weekend`).
- **`custom`** — any PromQL query whose result is aligned to the primary series.
- **`holidays`** — public-holiday calendar from the [`holidays`](https://pypi.org/project/holidays/) package (install with `pip install 'promforecast[holidays]'`).

Regressors are declared per query under `regressors:`. Built-in models ignore them today; plug-in models (e.g. `promforecast-prophet`) receive them via the `regressors` argument to `fit_predict`.

## Calendar regressors

```yaml
queries:
  - id: requests_per_second
    promql: rate(http_requests_total[5m])
    regressors:
      - id: hour_of_day
        type: calendar
      - id: day_of_week
        type: calendar
      - id: is_weekend
        type: calendar
```

The `id` must be one of the three supported names — the schema rejects anything else at boot.

## Custom (PromQL) regressors

```yaml
regressors:
  - id: deploys
    type: custom
    promql: changes(kube_deployment_status_observed_generation[1h])
    required: false          # default
```

- The PromQL is executed once per primary query and aligned via nearest-asof join.
- `required: true` makes a failure fatal for the whole series.
- Failures are counted in `forecast_regressor_failures_total{group, query, regressor, reason}`.

## Holiday regressors

Requires the `holidays` extra.

### Binary mode (default)

```yaml
regressors:
  - id: is_holiday
    type: holidays
    country: US                  # ISO-3166-1 alpha-2
    regions: []                  # country-wide by default
    timezone: America/New_York   # optional, defaults to UTC
    expand: false                # default
```

Emits a single binary column `is_holiday` (1.0 on any holiday, 0.0 otherwise).

### One-hot mode

```yaml
regressors:
  - id: holiday
    type: holidays
    country: US
    regions: [CA, NY]            # union of calendars
    expand: true
```

Emits one column per observed holiday name (`holiday__thanksgiving`, `holiday__independence_day`, …).  
If the lookback window contains no holidays, falls back to a single binary column so the feature schema stays stable.

### Timezone & regions

- `timezone` (IANA, e.g. `America/New_York`) converts UTC timestamps to local dates. Defaults to `UTC`.
- `regions` takes a list of subdivisions (states/provinces); the runner unions their calendars.

Supported countries/regions are whatever the installed `holidays>=0.50` package provides (≥150 countries + most major subdivisions).

## Webhook regressors

For a signal that does not live in Prometheus and is not on the built-in calendar list — active incidents from PagerDuty, deploys-in-flight from a CI/CD API, a custom internal "promotion-day" calendar — the `webhook` type lets the forecaster pull it in at fit time without standing up a Prometheus exporter or modifying the forecaster.

```yaml
regressors:
  - id: active_incidents
    type: webhook
    url: https://pagerduty-bridge.internal/incident-count
    timeout: 5s                  # default
    cache_ttl: 1h                # default — reuse responses per (url, query, range)
    headers:                     # optional static auth
      Authorization: Bearer ${PAGERDUTY_TOKEN}
    required: false              # default
```

The forecaster POSTs to `url` once per primary query (not per series) with a JSON body:

```json
{"query_id": "<id>", "regressor_id": "active_incidents", "start": "<ISO-8601>", "end": "<ISO-8601>"}
```

and expects a JSON time series back:

```json
{"timestamps": [1735689600, 1735693200], "values": [2, 1]}
```

Timestamps may be epoch seconds or ISO-8601 strings. The result is aligned to the primary series with the same nearest-asof join as a custom PromQL regressor, and cached per `(url, query_id, lookback)` for `cache_ttl` so a query returning many series — or a short refresh interval — doesn't hammer the external service.

The forecaster stays stateless: the webhook owns whatever state it queries. Webhook failures (timeout, non-2xx, unparseable body) flow through the same graceful-degradation path as any other regressor — the forecast is fit *without* the regressor and `forecast_regressor_failures_total{…, reason="webhook_failure"}` increments. Set `required: true` to skip the series instead.

`headers` values support `${ENV_VAR}` expansion against the process environment (injected from a Kubernetes Secret via `secretKeyRef`), so `Authorization: Bearer ${PAGERDUTY_TOKEN}` is resolved at request time rather than sent literally — keep the token out of the config. An unset variable expands to an empty string.

> The shared webhook HTTP client is created when *any* query configures a `webhook` regressor. A config reload that adds the first webhook regressor wires the client on the fly, so no restart is required.

### Example integrations

**PagerDuty incident lookup** — a small bridge service queries the PagerDuty API for the count of active incidents in the requested window and returns it as a step series. Spikes in incidents become an exogenous input the model can attribute load changes to.

**CI/CD deploy lookup** — a bridge to your CI/CD API (GitHub Actions, Argo, Jenkins) returns 1 on timestamps where a deploy completed. This is the dynamic sibling of the static `calendar_file` regressor below: use the webhook when "did a deploy happen?" is only answerable at query time, the calendar file when it's a known fixed schedule.

### HMAC-verified responses

When the webhook bridge signs its response body, add a `signing` block and the forecaster verifies it before accepting the data:

```yaml
regressors:
  - id: active_incidents
    type: webhook
    url: https://pagerduty-bridge.internal/incident-count
    signing:
      algorithm: hmac-sha256     # only value today
      secret_ref: ${WEBHOOK_SECRET}
```

The forecaster recomputes `HMAC-SHA256(secret, response_body)` and compares it constant-time against the `X-Promforecast-Signature: sha256=<hex>` header the bridge sets. A missing or mismatched signature is treated as a regressor failure with `reason="signature_invalid"` (same graceful-degradation path). `secret_ref` resolves a `${ENV_VAR}` reference against the process environment (injected from a Kubernetes Secret via `secretKeyRef`) or accepts a literal for local testing; the same `signing` block is available on [webhook triggers](../triggers.md) for the outbound direction.

## Calendar-file regressors

For a static internal calendar — release freezes, fiscal-quarter ends, marketing-campaign launches — the `calendar_file` type reads a YAML or iCal file mounted into the pod, without needing a webhook just to read a file.

```yaml
regressors:
  - id: release_freeze
    type: calendar_file
    path: /etc/promforecast/calendars/release-freezes.yaml
    expand: false                # default — single binary column
```

The file format is YAML (preferred, git-friendly) with an `entries:` list:

```yaml
# calendars/release-freezes.yaml
entries:
  - start: 2026-01-01
    end: 2026-01-05
    name: new-year-freeze
  - start: 2026-12-20
    end: 2026-12-31
    name: holiday-freeze
```

or iCal (`.ics`, for tooling compatibility) — each `VEVENT` becomes one window from its `DTSTART` / `DTEND` / `SUMMARY`.

The emission shape matches the `holidays` regressor:

- `expand: false` (default) emits a single binary column `release_freeze` (1.0 on any day inside any window, 0.0 otherwise).
- `expand: true` emits one column per distinct entry `name` (`release_freeze__new_year_freeze`, …).

The calendar file is re-read automatically when its mtime changes, so a ConfigMap update to the mounted file is picked up on the next fit — the same effect as the main config's configmap-watch reload path, without a process restart. A missing or malformed file degrades gracefully with `reason="calendar_file_error"`.

## Failure handling

| Failure type                        | Reason label      | `required: true` behaviour          |
|-------------------------------------|-------------------|-------------------------------------|
| Invalid calendar id                 | `invalid_type`    | Series skipped                      |
| Custom PromQL error / timeout       | `query_error` / `query_timeout` | Entire series skipped          |
| Custom PromQL returns no data       | `empty_result`    | Entire series skipped               |
| No datasource (dry-run / test)      | `no_source`       | Entire series skipped               |
| `holidays` package missing          | `invalid_type`    | Series skipped                      |
| Unknown country / region            | `invalid_type`    | Series skipped                      |
| Webhook timeout / error / bad body  | `webhook_failure` | Entire series skipped               |
| Webhook HMAC verification failed    | `signature_invalid` | Entire series skipped             |
| Calendar file missing / malformed   | `calendar_file_error` | Series skipped                  |

Optional regressors (default) only increment the failure counter; the forecast still emits without that feature.

## Contribution metric

When any regressor contributes signal, the runner emits a normalised explainability share:

```
<id>_forecast_contribution_ratio{regressor, ...}   # values in [0, 1], sum to 1 per series
```

Disable via `defaults.emission.contribution: false` (see [`../reference/config-schema.md`](../reference/config-schema.md)).

## Conditional forecasts

The default forecast answers: "Is the actual inside the historic envelope?"
That works for most cases, but breaks when a known regressor legitimately shifts the metric (e.g. a deploy that *should* increase request rate).

**Conditional forecasts** answer: "Given the *current* state of the regressors, what should the metric look like now?"
Use the conditional deviation when you want to detect anomalies *after accounting for known external drivers*.

### Configuration

Per-query, off by default:

```yaml
groups:
  - name: http
    queries:
      - id: http_requests_rate
        promql: sum(rate(http_requests_total[1m])) by (endpoint)
        regressors:
          - id: deploys
            promql: changes(kube_deployment_status_observed_generation[1h])
        emission:
          conditional: true
```

The runner re-fits each emitted model while holding the most recent regressor values constant.

### Emitted metrics

| Metric                                      | Description |
|---------------------------------------------|-----------|
| `<id>_forecast_conditional`                 | Point forecast at horizon |
| `<id>_forecast_conditional_lower{level}`    | Lower band per level |
| `<id>_forecast_conditional_upper{level}`    | Upper band per level |
| `forecast_conditional_deviation_ratio`      | Signed deviation of latest actual |
| `forecast_conditional_deviation_outside_band` | 1 if actual is outside the conditional band |

These carry the same labels as the unconditional deviation family, so you can alert on them directly:

```promql
# Anomaly after accounting for known regressors
forecast_conditional_deviation_outside_band{level="95"} == 1
  and on (id, group, model) forecast_quality_score > 0.6
```

### Model behaviour

Built-in models (`AutoARIMA`, `AutoETS`, `MSTL`, `SeasonalNaive`, etc.) do **not** use regressors.
They advertise `uses_regressors = False`, so the runner **short-circuits** the conditional refit and aliases the unconditional forecast.

You still get a stable schema (conditional metrics always exist), but at **zero extra CPU cost**.

Regressor-aware models (e.g. Prophet plug-in) perform a real second fit with held-constant regressors.

### Conditional-band conformal calibration

When `accuracy.conformal.enabled: true` *and* `emission.conditional: true` are both on for a query, the conditional band carries its own empirical-coverage guarantee — derived from a dedicated calibration fit where regressors are held flat at the final pre-holdout value, matching the production-time conditional inference contract.

The per-(model, level) widening applied to the conditional band lands on `/metrics` as `forecast_conditional_band_calibration_offset{id, group, model, level}` (gauge, same units as the underlying metric), sibling of the unconditional `forecast_band_calibration_offset`. Comparing the two reveals whether the conditional band needs different widening than the unconditional one — typically yes, because the unconditional residuals were scored on a holdout where regressors *did* vary.

Per-(model) calibration failures (degenerate holdout, fit timeout, fit error) fall back to the inherited unconditional offsets and bump `forecast_conditional_calibration_failures_total{group, query, reason}` with `reason` one of `degenerate_holdout`, `fit_timeout`, `fit_error`. The fallback path is no worse than the inherited-offset baseline, but a sustained tick rate signals fresh calibration is silently not running.

Models that advertise `uses_regressors = False` skip the extra calibration fit entirely — the conditional fit would produce the same numbers as the unconditional fit, so emitting two parallel offset gauges with identical values would just be noise. This means the gauge family appears only when at least one emitted model is regressor-aware.

### Cardinality

Roughly doubles the forecast snapshot footprint for opted-in queries.
`promforecast validate` includes the extra lines in its estimate.

See [`../operations/cardinality.md`](../operations/cardinality.md) for the full multiplier table.

### When not to use it

- No regressors on the query (conditional becomes a pure duplicate).
- No model in the set uses regressors (wastes cardinality).
- You want classic anomaly detection that treats regressor shifts as real deviations.
