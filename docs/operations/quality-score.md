# Forecast quality score

`forecast_quality_score{id, group, model, horizon}` is a composite 0–1 gauge that multiplies three sub-scores:

1. **Accuracy** — backtest MAPE (continuous) or MASE (intermittent/count).
2. **Band score** — penalises wide confidence bands relative to typical actual value.
3. **Coverage** — fraction of expected points present, scaled by `(1 - imputed_fraction)`.

A score near `1` means the model fits cleanly with tight, well-calibrated bands. A score near `0` means “don’t trust this forecast”.

See [`forecaster/src/promforecast/quality.py`](../../forecaster/src/promforecast/quality.py) for the exact formula.

## Per-horizon segmentation

The 1h forecast is often far more reliable than the 24h forecast. The quality score is emitted **per horizon** so you can gate short-horizon deviation alerts and long-horizon capacity alerts independently:

```
forecast_quality_score{id="...", model="...", horizon=""}        # aggregated
forecast_quality_score{id="...", model="...", horizon="1h"}      # 1h only
forecast_quality_score{id="...", model="...", horizon="24h"}     # 24h only
```

The aggregated row (`horizon=""`) is always emitted, so existing alerts that don’t filter on horizon continue to work unchanged. Per-horizon rows appear whenever `accuracy.evaluate: true`.

### Recommended alert gating

```yaml
# Short-horizon deviation alert
- alert: ForecastDeviation
  expr: |
    forecast_deviation_outside_band{level="95"} == 1
    and on (id, group, model) forecast_quality_score{horizon="1h"} > 0.6

# Long-horizon capacity alert
- alert: ForecastCapacityWarning
  expr: |
    node_filesystem_avail_bytes_forecast < 0
    and on (id, group, instance) forecast_quality_score{horizon="24h"} > 0.5
```

The shipped example alerts in [`examples/alerts/promforecast-rules.yaml`](../../examples/alerts/promforecast-rules.yaml) use the aggregated score for general rules (`ForecastQualityLow`, `ForecastQualityZero`) and horizon-specific rules where appropriate.

## Cardinality

Each (series, model) emits **1 aggregated row + 1 row per configured accuracy horizon**.  
With the default 2 horizons (`1h`, `24h`) this triples the per-(series, model) quality footprint (1 + 2 = 3 rows).  
`promforecast validate` counts the multiplier under the “quality lines” row.

## Why multiplicative?

Any single weak dimension drags the overall score down. Common patterns:

| Pattern                        | Likely cause |
|--------------------------------|--------------|
| Score stuck at 0               | Degenerate MAPE (all-zero actuals), 0 % band, or 0 % coverage |
| 1h score >> 24h score          | Expected long-horizon decay — use horizon-specific gating |
| Score dropped after deploy     | Check `forecast_revision_magnitude` and `forecast_drift_score` |
| Cold-start row near cap        | Expected — cold-start score is deliberately capped |
