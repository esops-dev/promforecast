# High availability

`promforecast` runs a per-process scheduler. Multiple replicas without coordination would produce duplicate fits, racing writes, and inconsistent `/metrics` output.

HA mode uses Kubernetes Lease-based leader election so **only one replica** runs the scheduler. The others serve the leader’s snapshot from a shared Redis cache.

## Enabling

```yaml
highAvailability:
  enabled: true
  redis:
    address: redis-master.redis.svc:6379
    existingSecret: redis-auth         # optional

replicaCount: 3
```

This switches the Deployment to a rolling update, mounts pod identity via the downward API, creates the necessary `Role`/`RoleBinding` for Lease access, and wires `config.highAvailability.enabled`.

## How it works

1. Every replica boots as a **follower** and starts the elector loop.
2. The elector acquires the `promforecast-leader` Lease (`coordination.k8s.io/v1`).  
   If the lease is expired or missing, acquisition is immediate.
3. The leader runs the scheduler and, after each successful run, writes the full `/metrics` payload to Redis under `promforecast:snapshot`.
4. Followers serve `/metrics` directly from the Redis cache (they only emit leader-election telemetry locally).
5. On graceful shutdown the leader releases the lease so a successor can take over quickly.

## Metrics

| Metric                                      | Type    | Description |
|---------------------------------------------|---------|-------------|
| `forecast_leader_elected{instance}`         | gauge   | 1 if this replica holds the lease |
| `forecast_leader_transitions_total{instance}` | counter | Number of election state changes since boot |

Alert on rapid `rate(forecast_leader_transitions_total[5m])` to detect lease thrashing.

## Tuning lease parameters

```yaml
config:
  highAvailability:
    lease_duration_seconds: 30
    renew_deadline_seconds: 20
    retry_period_seconds: 4
```

Defaults match the Kubernetes control plane. The forecaster rejects any config where `renew_deadline_seconds >= lease_duration_seconds`.

## Why not multi-leader sharding?

True sharding (different replicas fitting different groups) is planned for a future major version. For now, followers carry zero fit cost — the simplest and safest model for v0.x deployments.
