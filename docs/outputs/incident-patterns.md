# Recurring incident pattern detection

`forecast_incident_recurrence_score` turns a stream of discrete incident events into a **7×24 heatmap** of recurrence intensity.  
It answers “when do incidents tend to happen?” so you can proactively staff on-call or harden systems around predictable hot spots.

This is discrete-event forecasting (not continuous metrics) and sits at the boundary with the sibling `promanomaly` project.

## Configuration

Mark the query with `data_profile: events`:

```yaml
groups:
  - name: incidents
    queries:
      - id: pagerduty_incidents_total
        promql: sum(pagerduty_incidents_total)
        data_profile: events
```

The PromQL must return a counter that increments on each incident. The runner aggregates all returned series into a single event stream before bucketing.

Typical sources:
- `pagerduty_incidents_total`
- `alertmanager_notifications_total{severity="critical"}`
- A custom webhook counter

## Emitted metric

A full 168-cell grid is always published:

```
forecast_incident_recurrence_score{
  group="incidents",
  id="pagerduty_incidents_total",
  day_of_week="0",      # 0 = Monday … 6 = Sunday
  hour_of_day="14"
} 2.5
```

- Value = events in this bucket ÷ mean events across all non-empty buckets.  
  `> 1.0` = above-average concentration; `0` = never observed.

## Interpretation guide

| Score     | Meaning                              |
|-----------|--------------------------------------|
| `0`       | No events observed                   |
| `< 1.0`   | Below-average activity               |
| `1.0`     | Exactly average                      |
| `1.5–3.0` | 50–200 % above average               |
| `> 3.0`   | Strong, reliable hot spot            |

Look at both the score **and** the overall distribution shape:
- Few high-score buckets → predictable pattern (act on it).
- Many low-score buckets → random noise (descriptive only).

## Worked examples

**PagerDuty**
```yaml
- id: pagerduty_incidents_total
  promql: sum(pagerduty_incidents_total{service=~"$service_pattern"})
  data_profile: events
```

**Alertmanager**
```yaml
- id: alertmanager_critical_notifications
  promql: sum(alertmanager_notifications_total{severity="critical"})
  data_profile: events
```

**Custom webhook**
Increment a Prometheus `Counter("incidents_total")` in your webhook handler and scrape it normally.

## Operational notes

- No forecast curve, bands, or deviation metrics are emitted — only the heatmap.
- Fixed cardinality: exactly **168 rows** per opted-in query (counted by `promforecast validate`).
- Time-of-day is always UTC.
- Counter resets (negative deltas) are treated as zero events.

The bundled `promforecast-incident-heatmap.json` dashboard renders the heatmap with group/query selectors and includes “Active buckets” + “Top bucket” stat panels for quick triage.

## Cross-tool overlay (roadmap)

A future dashboard will overlay this recurrence score with `promanomaly` outputs to answer “did the pattern hold this week, and did the anomaly detector agree?”.

Until then, the heatmap stands alone as a preemptive-staffing signal.
