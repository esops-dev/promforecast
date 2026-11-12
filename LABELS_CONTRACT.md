# Cross-tool label contract

This file is **committed identically** in both [promforecast](https://github.com/esops-dev/promforecast) and [promanomaly](https://github.com/esops-dev/promanomaly). A CI job in each repo diffs it against the other's copy and fails if they drift. Any change requires a coordinated PR pair against both repos.

## Why this exists

Operators running both tools want to compose their output in PromQL — e.g. "alert when a metric is both **forecast to breach** *and* **scored as anomalous right now**":

```promql
(forecast_deviation_outside_band == 1)
  and on (id, group)
(anomaly_outside_threshold == 1)
```

This `and on (id, group)` only works if both tools emit `id` and `group` labels with the same meaning and the same values for the same underlying signal. If one renames `id` to `metric_id`, or one decides `group` is a free-form tag while the other treats it as a primary key, the join silently drops every series and the alert never fires.

The contract below pins the labels that both tools must respect.

## Contracted labels

### `id` (required, identical semantics in both tools)

The user-chosen identifier for the source metric, taken verbatim from the YAML config's `queries[].id` field. Used as the **prefix** for output metric names in promforecast (`<id>_forecast`, etc.) and as a **label value** on every anomaly metric in promanomaly.

- Type: string.
- Constraints: must satisfy Prometheus metric-name rules (`[a-zA-Z_:][a-zA-Z0-9_:]*`), and additionally **must not contain colons** — colons are reserved for recording rules in both tools' exposition.
- Stability: changing `id` for an existing signal is a breaking change to dashboards and alerts; rename only with a deprecation period.

### `group` (required, identical semantics in both tools)

The user-chosen grouping name from the YAML config's `groups[].name` field. Operationally used to scope refresh cadence, query limits, and failure counters; semantically used by joins to identify "the same signal" across tools.

- Type: string.
- Constraints: must be unique within a config file. Same Prometheus name rules and no-colons rule as `id`.
- Stability: same as `id`.

### Instance / cohort labels (passed through unchanged)

Both tools propagate the underlying time-series labels (`instance`, `pod`, `device`, etc.) from the source PromQL query onto their output metrics. The contract: **neither tool may rename or rewrite these labels**. If the source carries `instance="host-a:9100"`, the output metric carries `instance="host-a:9100"` — not `exported_instance`, not `source_instance`.

This is the reason `honor_labels: true` is required on the Prometheus scrape config for both tools' `/metrics` endpoints.

### `day_of_week` (contracted when present, identical semantics in both tools)

Emitted on recurrence-analysis metrics (`anomaly_recurrence_score` in promanomaly, incident-pattern metrics in promforecast). Represents the ISO 8601 day of week as a three-letter abbreviation: `Mon`, `Tue`, `Wed`, `Thu`, `Fri`, `Sat`, `Sun`.

- Type: string, one of the seven values above.
- Stability: label values are fixed (the seven abbreviations); adding or removing is a breaking change.

### `hour_of_day` (contracted when present, identical semantics in both tools)

Emitted alongside `day_of_week` on recurrence-analysis metrics. Represents the hour in UTC as a string (`"0"` through `"23"`).

- Type: string, `"0"` through `"23"`.
- Stability: same as `day_of_week`.

## Tool-private labels (NOT contracted)

Either tool may add its own labels to its own output metrics without coordination, as long as they don't collide with the contracted set above.

Examples:

- promforecast: `model`, `horizon`, `level`.
- promanomaly: `detector`, `detector_instance`.

Adding a new label to an existing **output metric family** is still a breaking change *within* that tool (it changes cardinality and breaks PromQL joins on the metric), but does not require a contract update.

## How to change this file

1. Open a draft PR against `promforecast` with the proposed change.
2. Open a draft PR against `promanomaly` with the identical change.
3. Link the two PRs.
4. Merge both within the same release cycle. CI will fail on either side until the other is merged.

Resist adding labels to the contract unless they are required for cross-tool composition. Tool-private labels are cheaper and don't require lockstep releases.
