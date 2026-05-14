# Query cache

When multiple groups query the same TSDB with overlapping PromQL and lookback windows, the forecaster would otherwise repeat identical fetches every refresh cycle. The query cache deduplicates those calls at the source.

It is **disabled by default** (single-group setups gain nothing) and stateless across restarts.

## Enabling

```yaml
config:
  query_cache:
    enabled: true
    backend: auto      # `redis` when highAvailability.enabled, else `memory`
    ttl: 5m            # how long a cached result stays usable
    max_entries: 1024  # in-memory LRU cap (ignored for redis)
```

The cache sits inside `PromSource`, so **every** PromQL fetch (primary queries, regressors, discovery probes) goes through it.

## Backends

| Backend | When used                          | Scope                  | Notes |
|---------|------------------------------------|------------------------|-------|
| **memory** | Default for single-replica (or `backend: memory`) | Per-process LRU + TTL | Independent per replica; restarts drop cache |
| **redis**  | Automatic when `highAvailability.enabled: true` (or `backend: redis`) | Shared across replicas | Leader writes, followers read; survives leader changes |

## Cache key

SHA-256 of `(normalised_promql, lookback_seconds, step_seconds)`.  
Whitespace, YAML formatting differences, and absolute time windows are **not** part of the key — only the shape of the query matters. Staleness is strictly bounded by `ttl`.

## Sizing TTL

Choose a TTL **below** your `refresh_interval` so every cycle still gets fresh data, but overlapping groups hit the cache.

**Example** (`refresh_interval: 1h`, three groups sharing `node_cpu_seconds_total`):  
- `ttl: 5m` → at most one fetch per group per hour.  
- `ttl: 90m` → one fetch total per hour, but groups may see data up to 30 min stale.

## Metrics

| Metric                                      | Type    | Description |
|---------------------------------------------|---------|-------------|
| `forecast_query_cache_events_total{kind}`   | counter | `hit`, `miss`, or `error` per event (always emitted) |

## Operational notes

- Failures degrade gracefully: a cache error is treated as a miss and the TSDB is queried directly.  
- Redis outage appears only as elevated `miss` rate + log lines — never as a forecast outage.  
- The cache and HA leader-election share the **same** Redis client (no extra connection pool).  
- Stateless by design: restart or Redis key eviction simply causes the next refresh to refetch.

## Validating locally

`make dev-up` starts the full stack including Redis and uses `docker/dev-config.yaml` (which enables `backend: redis`).  
After the first refresh, `make dev-validate` confirms `forecast_query_cache_events_total{kind="miss"}` is non-zero.

For real HA behaviour, deploy the Helm chart into a `kind`/`minikube` cluster.
