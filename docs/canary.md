# Canary / weighted groups

Fit only a fraction of a group’s series on every refresh tick.  
Ideal when you want to:

- Safely onboard a high-cardinality group before committing full CPU.
- Test an expensive new model on a representative slice of the fleet.
- Profile or iterate on a slow group with a smaller working set.

## Configuring

```yaml
groups:
  - name: experimental_per_pod
    priority: 1
    canary:
      enabled: true
      weight: 0.1           # fit ~10% of returned series
    queries:
      - id: pod_cpu_busy_pct
        promql: |
          rate(container_cpu_usage_seconds_total{pod=~".+"}[5m])
```

`weight` is a float in `(0, 1]`. The runner samples `ceil(weight * N)` series from each query’s result set — even a `weight: 0.1` group with only 3 series will still fit 1.

Sampling uses Python’s `random.sample` (default RNG, no fixed seed). Each refresh rotates through the population, giving broader observability over time than a static slice.

## Pipeline order

Canary sampling happens **first** in the per-query flow:

1. Fetch frames from TSDB  
2. Canary sampling (if enabled)  
3. Per-query cap (`safety.max_series_per_query`)  
4. Global cap (`safety.max_total_series` remaining)  
5. Minimum-points filter  
6. Fit

**Why this order?** Canary is an intentional reduction chosen by the operator. Running the safety caps first would defeat the purpose for groups that are already under the cardinality limits but still too expensive to fit fully.

Dropped series are counted in `forecast_series_dropped_total{group, reason="canary"}`.

## Disabling

```yaml
canary:
  enabled: false   # or simply omit the block
```

`enabled: true` with `weight: 1.0` is a no-op (and rejected by the schema). You can safely leave the block in place while turning it off.

## Important trade-off

Canary sampling happens **after** the PromQL fetch.  
You still pay the full network / TSDB cost for every series — the savings are only on **fit CPU**.

This is deliberate:
- Sampling inside PromQL (e.g. `topk`) would require rewriting the query and could change semantics.
- Model fitting is almost always the dominant cost.

If you also need to reduce fetch volume, add `topk` directly to your `promql:` or split the group and tune per-query caps.

## When not to use canary

- The group is already small (a handful of series) — variance outweighs benefit.
- You need deterministic, reproducible output across runs (use `topk` or label selectors instead).
- You want different sample rates per query inside the same group — split into separate groups.
