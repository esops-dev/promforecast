# Conformal band calibration

Parametric confidence bands often under-cover on real metric data.  
A nominal 95 % AutoARIMA band might empirically cover only 60–80 % of future points. Split-conformal calibration fixes this by widening the bands *after* the fit so the requested coverage level is what you actually observe.

## Configuring

```yaml
defaults:
  accuracy:
    conformal:
      enabled: true        # opt-in; off by default
      method: split        # only method supported in v1.1
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

## When conformal can’t help

- **Insufficient lookback** — fewer than five points on each side of the split.
- **Constant or all-zero series** — residuals are zero → offset stays zero.
- **Systematically bad model** — conformal makes the band *cover* the nominal level, but if the underlying model is biased the band can become uselessly wide. Watch large sustained `forecast_band_calibration_offset` values as the signal to fix the model instead.

## See also

- [Models](models.md) — backtest accuracy feeds the calibrator.
- [Input preprocessing](preprocessing.md) — outlier scrubbing runs *before* calibration to keep the hold-out clean.
