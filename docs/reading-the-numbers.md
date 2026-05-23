# Reading the numbers

This page explains how to interpret promforecast’s quality and accuracy metrics so you can tell at a glance whether the system is healthy, whether a specific forecast is trustworthy, and when a “bad” number isn’t actually a problem.

## The two numbers that matter most

For daily operation, these two metrics tell you almost everything you need:

| Metric                      | What it tells you                          | Good values                  |
|-----------------------------|--------------------------------------------|------------------------------|
| `forecast_quality_score`    | Per-(series, model) composite reliability in `[0, 1]` | `> 0.6` reliable; `< 0.3` don’t trust |
| `forecast_accuracy_mape`    | Backtest MAPE per `(group, horizon)`, in percent | Depends on signal shape — see below |

If you check the dashboard once a day, look at **average quality per group** and **trustworthy-series count** (series scoring > 0.5). Both green = the bulk of your forecasts are useful.

## What MAPE actually tells you

MAPE is the backtest accuracy (held-out actuals vs. predicted, as percentage).  
There is **no universal “good MAPE”** — it depends heavily on the signal.

| Signal shape          | Typical MAPE range | Example |
|-----------------------|--------------------|---------|
| Smooth, predictable   | `< 5 %`            | Stable network throughput |
| Cyclical with noise   | `10–30 %`          | CPU with daily/weekly patterns |
| Bursty                | `50–100 %`         | Disk I/O with occasional spikes |
| Sparse / near-zero    | `100 % +` (or NaN) | Rare error rates |

For sparse/near-zero series, MAPE becomes unreliable. Set `data_profile: intermittent` on the query to switch to MASE scoring (see [models.md](models.md)).

## Mean quality vs. p95 MAPE — why they can disagree

- **Mean quality** averages across *all* series in the group.  
- **p95 MAPE** highlights the worst ~5 % of `(series × horizon)` measurements.

A group can have excellent mean quality while p95 MAPE is high — that just means one or two series are noisy.  
**Rule of thumb:** use mean quality for “is the group healthy overall?”; use p95 MAPE + `bottomk(3, forecast_quality_score)` when hunting the specific bad series.

## The horizon paradox

Shorter horizons are *not* always more accurate on noisy signals.  
Short horizons inherit minute-to-minute noise; long horizons benefit from implicit averaging.  
Seeing `p95 MAPE 3m > p95 MAPE 15m` is usually normal for spiky data — inspect the raw series to confirm.

## After a restart: wait three cycles

Quality and MAPE reset on restart.  
- First cycle: noisy, not representative.  
- Cycles 2–3: numbers stabilise.  
- After ~30 min of uptime: metrics are meaningful.  

Don’t judge model performance from the first cycle.

## Interpretive examples from a typical dev stack

| Group          | Mean quality | What it tells you |
|----------------|--------------|-------------------|
| `node_network` | `~0.99`      | Smooth, predictable — excellent forecasts |
| `node_storage` | `~0.65`      | Bursty I/O — mid-range quality is the ceiling |
| `node_capacity`| `~0.45`      | Moderately noisy CPU/memory — expected |
| `http_demo`    | `~0.30`      | Sparse endpoints + log-normal latency — low quality is honest |

Your own “good” numbers depend on what you forecast: smooth platform metrics sit near 1.0; application latency percentiles rarely do.

## When to actually worry

Use this checklist when a number looks bad:

- **Quality drops 0.5+ in one cycle** (while uptime is high) → regime shift. Check `forecast_drift_score` and `forecast_input_change_point_total`.  
- **All series in a group drop to quality 0** → datasource problem. Verify the PromQL, datasource URL, and target `up` metric.  
- **One series stuck at quality 0** while siblings are fine → isolated failure. Query `forecast_quality_score{group="<name>"} == 0`.  
- **MAPE climbing but quality stays high** → bands are widening (conformal calibration working). Check `forecast_band_empirical_coverage`.  
- **Visual forecast looks wrong but quality is high** → check Grafana variable selection (`model` / `endpoint` dropdown).

## What each panel of “Is it working?” answers

| Panel                     | Question it answers |
|---------------------------|---------------------|
| Quality score per group   | Is the group trustworthy on average? |
| Trustworthy series count  | How many series score above the usable threshold? |
| Last successful run       | Did the scheduler tick recently? (should be < `refresh_interval`) |
| Failures (last 1 h)       | Did anything go wrong? Zero is the happy path |
| Signature panels          | Does the green (actual) line sit inside the blue (95 % forecast band)? |

For per-series drill-down (quality, drift, deviation, durations, errors) switch to the **“Demo tour”** dashboard.

## See also

- [telemetry.md](telemetry.md) — full list of emitted metrics and labels  
- [conformal.md](conformal.md) — how bands are empirically calibrated  
- [drift.md](drift.md) — `forecast_drift_score` semantics  
- [scheduling.md](scheduling.md) — how `refresh_interval` and per-query overrides work
