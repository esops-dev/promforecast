# Sharded HA — horizontal scaling for very large deployments

[High availability](../operations/ha.md) gives you one active leader plus hot followers serving a cached snapshot. The ceiling is whatever a single leader can fit within a refresh interval — roughly 50–100k output series.

**Sharding** lifts that ceiling. Every replica is active and fits a **disjoint subset of groups**. Ownership is disjoint and complete, so the aggregate fit budget scales with the replica count. The `replicas` field becomes a throughput knob.

Sharding and HA are **mutually exclusive** — HA gates fitting on a single leader; sharding spreads fitting across every replica. The config loader rejects enabling both.

## When to use sharding

| Situation | Recommended approach |
|-----------|----------------------|
| Fits comfortably within interval on one replica | Single replica (default) |
| Need zero-downtime failover but one replica suffices | [HA](../operations/ha.md) |
| One replica can no longer fit everything within `refresh_interval` | **Sharding** (this page) |

Symptom of outgrowing HA: `forecast_run_duration_seconds` approaching `server.refresh_interval` or rising `forecast_slow_fits` on a vertically scaled leader.

## The two modes

### `consistent_hash` (default, no coordination)

Each replica owns a group if `hash(group) % replicas == shard_index`. No shared state — replicas only need to agree on replica count (pinned by the chart).

Uses a **StatefulSet** so each pod gets a stable ordinal. Forecaster reads trailing ordinal of `PROMFORECAST_POD_NAME` as its `shard_index`.

```yaml
sharding:
  enabled: true
  mode: consistent_hash
replicaCount: 4
```

**Pros**: Simple, no Redis, no Lease.  
**Cons**: Rebalancing on replica count change reshuffles most groups. Crashed shard's groups wait for pod reschedule.

### `leader_assigned` (coordinated)

A `coordination.k8s.io/v1` Lease elects an assigner. Replicas heartbeat into a shared Redis membership registry. The assigner computes a balanced mapping and publishes it.

Uses a **Deployment** (no stable ordinal needed). Employs rendezvous hashing so membership changes move only ~1/N groups.

```yaml
sharding:
  enabled: true
  mode: leader_assigned
  redis:
    address: redis-master.redis.svc:6379
replicaCount: 4
```

**Pros**: Automatic rebalancing on join/leave; crashed shard's groups reassigned quickly.  
**Cons**: Requires Redis and Lease. Cold start has brief period where replicas own nothing until first assignment.

## Choosing a mode

Start with `consistent_hash` — simplest and fewest moving parts.  
Move to `leader_assigned` when you need automatic rebalancing on replica changes and already run Redis.

## Output labels and assignment gauge

Every output metric for an owned group gains a `shard_id` label (pod name by default). This label is only present when sharding is enabled.

Assignment gauge for debugging:

```
forecast_shard_owns_group{shard_id="...", group="..."} 1
```

Each replica emits one sample per owned group.

## Sizing guide

- Aim for **2–3× as many groups as shards** for even distribution.
- Target roughly even series-per-shard, not groups-per-shard. Split oversized groups if needed.
- Reasonable target: ≤ 40–60k output series per shard.

Detect hot shards — a shard whose busiest group nears the refresh interval, or whose owned series count sits well above the fleet median:

```promql
# per-shard busiest group run duration
max by (shard_id) (forecast_run_duration_seconds * on(group) group_left(shard_id) forecast_shard_owns_group)

# per-shard owned series count
sum by (shard_id) (forecast_series_count * on(group) group_left(shard_id) forecast_shard_owns_group)
```

When to rebalance:

- `consistent_hash`: rebalancing means changing `replicas`, which reshuffles most ownership — a deliberate, planned operation. Bump the shard count when the *busiest* shard (not the average) is at its fit ceiling. If a single group dominates, split it rather than adding shards.
- `leader_assigned`: rebalancing is automatic on replica join/leave; raising `replicas` redistributes within a heartbeat interval, moving only the groups that must move.

## Reload behaviour

Reloads that add/remove groups are handled per-shard: each replica re-derives ownership and starts/stops loops. In `leader_assigned` mode the assigner republishes the mapping for the new group set on its next tick.

## See also

- [High availability](../operations/ha.md)
- [Production-readiness checklist](../production-checklist.md)
- [Cardinality budgeting](../operations/cardinality.md)