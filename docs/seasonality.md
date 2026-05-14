# Automatic seasonality detection

Set `season_length: auto` on a query and the forecaster will detect the dominant period from the data itself.  
Useful when:

- You don’t know the natural cycle in advance.
- A group mixes series with different periods (daily + weekly).
- You’re migrating from another stack and want to avoid hand-tuning every series.

## Configuring

```yaml
defaults:
  step: 5m
  season_length: 288   # fallback when detection fails

groups:
  - name: experimental
    queries:
      - id: queue_depth
        promql: sum(queue_depth)
        season_length: auto
```

`season_length` can be an integer (static) or `auto`. The global `defaults.season_length` is used only as fallback.

## Algorithm

For every series with `season_length: auto`:

1. Compute the autocorrelation function (ACF) via FFT, capped at `min(len(values)//2, 2048)` lags.
2. Find the strongest local maximum after lag 2 (absolute values, so anti-correlations are considered).
3. Snap the peak to the nearest natural calendar period (hour / day / week) if it is within 5 %.
4. If no peak exceeds the noise floor (0.2), fall back to the step-derived heuristic (`86400 / step_seconds` for sub-daily, `7` for daily+).

## Observability

The detector logs the result once per series per refresh (INFO for fallback cases, DEBUG otherwise):

```json
{
  "event": "season_length_auto_resolved",
  "query": "queue_depth",
  "period": 288,
  "source": "snapped",
  "correlation": 0.71
}
```

| Source     | Meaning                                      | Log level |
|------------|----------------------------------------------|-----------|
| `detected` | Strong ACF peak                              | DEBUG     |
| `snapped`  | Peak snapped to calendar period              | DEBUG     |
| `fallback` | No usable peak; using configured fallback    | INFO      |
| `too_short`| Series too short / constant                  | INFO      |

Track behaviour with the always-emitted counter:

`forecast_season_length_detections_total{group, query, source}`

Example alert (persistent fallback):

```promql
sum by (group, query) (rate(forecast_season_length_detections_total{source="fallback"}[1h]))
  / sum by (group, query) (rate(forecast_season_length_detections_total[1h]))
> 0.9
```

## Performance

FFT input is limited to 16384 samples (most recent slice only). Detection stays well under a millisecond per series even on long lookbacks — no executor offload needed.

## When not to use auto

- The period is known and stable → use a hard-coded integer (faster, deterministic).
- Series are very short (`min_points` near the minimum) → detector will fall back anyway.
- You need multiple seasonalities (e.g. daily + weekly in MSTL) → auto picks only one period.
