# Recipe: `node_filesystem_avail_bytes`

> Predict when each mount runs out of space. Alert before the pager.

## The metric

`node_filesystem_avail_bytes` (from `node_exporter`) reports free bytes per `(instance, device, mountpoint)`. The line that matters is the one that crosses a low watermark and stays there — disk-fill is rarely sudden.

## The config

```yaml
groups:
  - name: disk_capacity
    queries:
      - id: node_filesystem_avail_bytes
        promql: |
          node_filesystem_avail_bytes{fstype!~"tmpfs|overlay|squashfs"}
        thresholds:
          - name: critical
            value: 5368709120        # 5 GiB
            comparator: lt
            alert_within: 24h
            severity: critical
            action_cost:
              type: usd
              value: 500
        emission:
          growth_rate:
            enabled: true
            windows: [1d, 7d]
            units: units_per_second
```

The full file ships a damped-trend model + auto-select to keep long-horizon extrapolation honest.

## The model

* **`AutoETSDamped`** with `damping_parameter: 0.98` — disk usage grows, but rarely at the rate the last 24 h would extrapolate. Damped-trend bends the long-horizon curve toward a level.
* `SeasonalNaive` rides along as the auto-select fallback when the ETS fit blows up.

## Expected forecast shape

- Steady downward slope, narrow band.
- Bursty backups + nightly cleanups appear as stair-steps within a day, smoothed to a weekly trend.
- Post-deploy log spike shows as a sharp local dip with visibly wider bands for the lookahead.

If you see `yhat` going *up*, the model is fitting the wrong direction. Switch the threshold comparator or check the PromQL filter.

## Cardinality

For ~100 hosts × 1 model × 1 level:

- forecast lines: 300
- threshold ETA: 300
- will-breach lines: 300
- growth rate lines: 200
- action cost lines: 1

## Alerts

```promql
# Will breach 5 GiB within 24 h at the 95th percentile band.
node_filesystem_avail_bytes_forecast_will_breach{severity="critical", level="95"} == 1
```

```promql
# Free-space loss rate has tripled compared to last week's baseline — something unusual is consuming the disk.
abs(node_filesystem_avail_bytes_forecast_growth_rate{window="1d"})
  / clamp_min(
    abs(node_filesystem_avail_bytes_forecast_growth_rate{window="7d"} offset 1d),
    1
  ) > 3
```
