# Recipe: `aws_cloudwatch_estimated_charges_usd`

> Month-end AWS bill projection that finance can read alongside engineering.

## The metric

`aws_cloudwatch_estimated_charges_usd` is the AWS billing estimate published by the CloudWatch exporter. It updates roughly every 6 hours and is denominated in USD.

## The config

```yaml
groups:
  - name: cloud_cost
    queries:
      - id: aws_estimated_charges_usd
        promql: |
          aws_cloudwatch_estimated_charges_usd{service_name="Total"}
        step: 6h
        thresholds:
          - name: monthly_budget
            value: 50000
            comparator: gt
            alert_within: 14d
            severity: warning
            action_cost:
              type: usd
              value: 50000
```

`step: 6h` matches CloudWatch’s publish cadence — finer resolution wastes TSDB CPU on duplicate samples.

## Recommended model

* **`AutoETSDamped`** with `damping_parameter: 0.95` (more aggressive than the default 0.98) because billing curves are fundamentally bounded by service contracts — linear extrapolation overstates.
* `SeasonalNaive` is included as a safety net; the weekly cycle is the dominant signal and often wins auto-select.

## Expected forecast shape

- Smooth upward trend within a billing month.
- Sharp drop at the start of each month as the counter resets — the forecast handles the reset cleanly because the lookback (90 d) includes multiple resets.

## Cardinality

For 5 AWS service categories × 1 model × 1 level:

- forecast lines: 15
- threshold ETA lines: 5
- will-breach lines: 5
- action cost lines: 1

## Alerts

```promql
# Projected to exceed monthly budget within 14 d
aws_estimated_charges_usd_forecast_will_breach{severity="warning"} == 1
```

```promql
# Month-end charge moved by > $5k week-over-week
abs(
  aws_estimated_charges_usd_forecast{horizon="14d"}
  - aws_estimated_charges_usd_forecast{horizon="14d"} offset 7d
) > 5000
```

## Pair with service-level health composite

This recipe contributes one row per service category to the `forecast_service_health_score` aggregation, giving finance a single number per AWS service tier alongside the per-tier forecasts.
