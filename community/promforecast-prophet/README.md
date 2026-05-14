# promforecast-prophet

Prophet forecasting model plug-in for [promforecast](../../).

## Why a separate package?

`prophet` brings a Stan toolchain and a sizable wheel matrix. Operators who don't need Prophet shouldn't pay that cost; operators who do, install one extra package and `Prophet` becomes available in their existing config.

## Install

```bash
pip install promforecast-prophet
```

In a Helm-deployed forecaster, add the package to your container image (rebuild from `forecaster/Dockerfile` with an extra `pip install promforecast-prophet`) or layer it via an init container that drops the wheel onto a shared `extraVolumes` mount.

## Use

The package registers a `Prophet` model under the `promforecast.models` entry-point group. The forecaster discovers it automatically at startup — no glue code, no config flags.

```yaml
apiVersion: promforecast.io/v1
defaults:
  models:
    - SeasonalNaive
    - name: Prophet
      params:
        growth: linear
        weekly_seasonality: true
        # interval_width can be pinned globally; otherwise every entry in
        # `confidence_levels` produces its own per-level Prophet fit.
        interval_width: 0.95
groups:
  - name: traffic
    queries:
      - id: requests_per_second
        promql: sum by (service) (rate(http_requests_total[5m]))
```

## Caveats

- Prophet fits one series at a time, so cost scales linearly with the number of returned series. Use `safety.max_series_per_query` to cap the worst case.
- One fit per series is enough regardless of how many entries are in `confidence_levels`: Prophet's `interval_width` is consulted at *predict* time, so the wrapper fits once and runs a cheap `predict()` per level. Pin `params.interval_width` to override Prophet's band width for every level (the level still labels the output columns).
- Regressors are passed through to `Prophet.add_regressor`. Future-frame values default to "carry forward the last observation"; supply a richer regressor pipeline yourself if you need genuine future predictions per regressor.

## License

Apache-2.0 — same as the parent project.
