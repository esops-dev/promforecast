# Cardinality budgeting

promforecast emits many optional metric families. Turning several on at once multiplies the total series count faster than the input data suggests. This page gives you the multiplier table and the recommended workflow to size your budget before shipping a config.

If your `/metrics` scrape is getting heavy or `forecast_series_dropped_total{reason="total_cap"}` is ticking, start here.

## How the safety budget works

- `safety.max_series_per_query` — caps each individual query.
- `safety.max_total_series` — global cap across all groups (sliced by `priority`).
- When exceeded, `series_overflow` (default: `drop_lowest_priority`) drops excess series.

Dropped series are tracked in `forecast_series_dropped_total{reason="per_query_cap" | "total_cap"}`.

## Estimator: `promforecast validate`

```bash
promforecast validate --config /etc/promforecast/config.yaml

# Fail CI on oversized configs
promforecast validate --config config.yaml --strict-cardinality
```

The validator assumes worst-case (`safety.max_series_per_query` per query) and prints a detailed breakdown. Use `--strict-cardinality` to turn the `WARN:` line into a non-zero exit code.

## Multiplier table

Notation:  
- `S` = `safety.max_series_per_query` (worst-case per query)  
- `M` = emitted models per series (1 with `auto_select`, all configured models otherwise)  
- `L` = number of `confidence_levels`  
- `H` = number of `accuracy.horizons`

| Family                                      | Multiplier per opted-in query          | Notes |
|---------------------------------------------|----------------------------------------|-------|
| `<id>_forecast` + lower + upper             | `S × M × (1 + 2L)`                     | Always on |
| `forecast_accuracy_mape` + `mase`           | `S × M_models × H × 2`                 | Requires `accuracy.evaluate: true` |
| `forecast_deviation_*`                      | `S × M × L × 2`                        | `emission.deviation: true` |
| `forecast_quality_score`                    | `S × M`                                | `emission.quality: true` |
| `<id>_forecast_contribution_ratio`          | `S × N_regressors`                     | Only with regressors |
| `forecast_band_calibration_offset`          | `S × M × L`                            | `accuracy.conformal.enabled: true` |
| `<id>_forecast_component`                   | `S × M × 4`                            | `emission.decompose: true` (trend/seasonal/level/residual) |
| `forecast_drift_score`                      | `S × M`                                | `emission.drift.enabled: true` |
| Threshold ETA / will_breach                 | `S × M × N_thresholds × (1 + L)`       | Only configured thresholds |
| Drain ETA (`le_drain`)                      | `S × M × N_drain_thresholds × (1 + L)` | Charged separately from the standard ETA |
| Growth rate                                 | `S × M × N_windows`                    | `emission.growth_rate.enabled: true` |
| Conditional forecast families               | `S × M × (1 + 2L)`                     | `emission.conditional: true` (may add extra fit) |
| `forecast_conditional_band_calibration_offset` | `S × M × L`                          | `emission.conditional: true` *and* `accuracy.conformal.enabled: true` |
| SLO error budget                            | `S × M × N_windows × (3 + L)`          | `slo:` configured (3 sibling families × N_windows, plus level fan-out on exhausted_in) |
| Recommended maintenance windows             | `S × M × count`                        | `recommended_windows.enabled: true` |
| Operational + preprocessing families        | ~7 per group + ~6 per query            | Mostly always on |

## Working example

**Setup:** 4 queries (50 series each), 2 models, 2 confidence levels, deviation + quality + decompose + conditional + 2 thresholds + growth rate.

**Projected total ≈ 17 200 lines**

With default `safety.max_total_series: 5000` this would fail `--strict-cardinality`. Solutions:
- Raise the cap
- Reduce models (`auto_select: true`)
- Turn off unused families
- Add `topk()` to high-cardinality PromQL queries

## Recommended posture

1. **Run `promforecast validate --strict-cardinality` on every config PR.**
2. Treat `safety.max_total_series` as a real knob — the default 5000 is for minimal setups.
3. Prefer `auto_select: true` to keep `M = 1`.
4. Monitor `forecast_series_dropped_total{reason="total_cap"}`.
5. Use `topk(N, ...)` in PromQL for queries prone to label cardinality spikes.

## See also

- [Emission paths](../reference/emission-paths.md)
- [Capacity planning](../outputs/capacity.md)
- [Conditional forecasts](../inputs/regressors.md#conditional-forecasts)
- [Explainability](../outputs/explainability.md)
