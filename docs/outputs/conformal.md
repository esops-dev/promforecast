# Conformal band calibration

Parametric confidence bands often under-cover on real metric data.  
A nominal 95 % AutoARIMA band might empirically cover only 60–80 % of future points. Split-conformal calibration fixes this by widening the bands *after* the fit so the requested coverage level is what you actually observe.

## Configuring

```yaml
defaults:
  accuracy:
    conformal:
      enabled: true        # opt-in; off by default
      method: split        # only method currently shipped
      holdout_fraction: 0.2
```

The forecaster reserves the most recent `holdout_fraction` of the lookback as a calibration set, fits the model on the prefix, computes absolute residuals on the hold-out, and uses the per-level quantile of those residuals to widen the final forecast bands.  

The point forecast (`yhat`) is **unchanged** — only the bands are adjusted.

## Cost

One extra fit per series per emitted model per refresh (same order of magnitude as the existing backtest pass).  
With `auto_select: true` this is just one extra fit per series (the winner).

## Choosing `holdout_fraction`

Default `0.2` matches the conformal-prediction literature and gives a solid sample for the 95 % quantile.  
Too small → a single outlier can dominate the estimate.  
Too large → the fitting history becomes too short.  

If the split leaves fewer than five points on either side, calibration is silently skipped for that series this run.

## Observability

```
forecast_band_calibration_offset{id, group, model, level}
```

Additive offset (in metric units) applied to each side of the band.  
Zero means the original parametric band already met its nominal coverage.  
Emitted bands also carry the label `calibrated="true"` on `_forecast_lower` / `_forecast_upper` so dashboards can filter or color them separately.

### Conditional band

When `emission.conditional: true` is also on for the query, the conditional band carries its own per-fit calibration (regressors held flat at the final pre-holdout value, matching the production-time conditional inference contract). The per-(model, level) widening lands as:

```
forecast_conditional_band_calibration_offset{id, group, model, level}
```

Per-(model) calibration failures fall back to the inherited unconditional offsets and bump `forecast_conditional_calibration_failures_total{group, query, reason}` (`reason` ∈ `degenerate_holdout` | `fit_timeout` | `fit_error`). See [`../inputs/regressors.md`](../inputs/regressors.md#conditional-band-conformal-calibration).

## When conformal can’t help

- **Insufficient lookback** — fewer than five points on each side of the split.
- **Constant or all-zero series** — residuals are zero → offset stays zero.
- **Systematically bad model** — conformal makes the band *cover* the nominal level, but if the underlying model is biased the band can become uselessly wide. Watch large sustained `forecast_band_calibration_offset` values as the signal to fix the model instead.

## Empirical coverage diagnostics

Calibration moves the bands; verifying they actually achieve their
nominal level is a separate concern. The empirical-coverage gauge
`forecast_band_empirical_coverage` reports the fraction of recent
actuals that fell inside the nominal band:

```
forecast_band_empirical_coverage{id, group, model, level, window}
```

Enable it per query:

```yaml
groups:
  - name: kpis
    queries:
      - id: requests_per_second
        promql: rate(http_requests_total[5m])
        emission:
          band_coverage:
            enabled: true            # default
            windows: [1d, 7d]        # default
```

Each (series, model, level) tracks a rolling ring buffer sized to
`window / step`. The gauge reports `inside / total` — for a
well-calibrated 95 % band you expect the gauge to hover near `0.95`.
A reading of `0.4` on a `level="95"` panel tells you the band is
drastically under-covering; the corresponding deviation alerts are
firing on every cycle and an alert on
`forecast_deviation_outside_band == 1` is effectively meaningless
until the bands are fixed.

Coverage and calibration interact:

* With `conformal.enabled: false`, `forecast_band_empirical_coverage`
  is the operator's primary signal that the bands are mis-calibrated.
* With `conformal.enabled: true`, it's the **feedback loop** —
  after calibration runs for a few refreshes the gauge should track
  the nominal level. Persistent under-coverage despite a positive
  `forecast_band_calibration_offset` points to an underlying model
  problem (wrong family, missing seasonality) that calibration can't
  paper over.

### Example PrometheusRule

```yaml
- alert: ForecastBandUnderCovering
  expr: |
    forecast_band_empirical_coverage{window="7d", level="95"} < 0.85
    and on (id, group)
    forecast_quality_score > 0.6
  for: 6h
  annotations:
    summary: "Forecast 95% band for {{ $labels.id }} covers only {{ $value | humanizePercentage }}"
```

The `and on (id, group) forecast_quality_score > 0.6` gate avoids
alerting on series whose underlying model is already known to be bad
— `ForecastQualityLow` is the better signal for those.

### Memory cost

`window / step` booleans per (series, level, window). With the
default `1d` + `7d` and a 5m step that's `288 + 2016 ≈ 2300` bits
per (series, level). At the canonical 5000-series cap with two
levels and two windows that's roughly 2.8 MB of rolling state —
comfortably bounded.

## See also

- [Models](../reference/models.md) — backtest accuracy feeds the calibrator.
- [Input preprocessing](../inputs/preprocessing.md) — outlier scrubbing runs *before* calibration to keep the hold-out clean.
- [Forecast drift telemetry](drift.md) — complements coverage by surfacing prediction-vs-prediction shifts.
