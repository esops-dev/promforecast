# Multi-cluster reference architectures

The chart defaults assume a single forecaster reading a single VictoriaMetrics (or Mimir/Thanos) endpoint. Multi-cluster deployments require additional choices:

- Where the forecaster runs (per cluster or centrally).
- Where its reads and writes land.
- How the `cluster` label is propagated for dashboards.

Three reference patterns cover the common cases.

## Pattern comparison

| Pattern | Autonomy | Cardinality | Network | When to use |
|---------|----------|-------------|---------|-------------|
| **A. Per-cluster forecaster + per-cluster VM → central aggregation** (recommended default) | High (cluster failure doesn't affect others) | Per-cluster (sized locally) | Low (local reads, one outbound push per cluster) | Any deployment with ≥2 autonomous clusters |
| **B. Central forecaster + federated read** | Low (single point of failure) | Union of all clusters | Higher (cross-cluster reads on every fit) | Small fleets (<5 clusters, <5000 series total) |
| **C. Per-cluster forecaster + per-cluster VM, Grafana mixed-datasource** | Highest (full isolation) | Per-cluster | Zero cross-cluster | Strict regulatory/compliance isolation |

## A. Per-cluster forecaster, per-cluster VM, central aggregation (recommended)

```
Cluster A: prometheus(A) ──remote_write──> VM(A) <── forecaster(A) ──remote_write──> VM(central)
Cluster B: prometheus(B) ──remote_write──> VM(B) <── forecaster(B) ──remote_write──> VM(central)
```

Each cluster runs its own forecaster, reads from its local VM, and pushes forecasts to a shared central VM. Central tier is read-only aggregation for dashboards.

**Why this is preferred**  
- High autonomy: one cluster’s failure never blocks others.  
- Local reads keep fit timing predictable.  
- Cardinality is sized per cluster, not the union.

**Sample values (per cluster)**

```yaml
config:
  datasource:
    url: http://victoriametrics.monitoring.svc.cluster.local:8428
  sink:
    remote_write:
      enabled: true
      url: https://vm-central.example.com/api/v1/import/prometheus
      headers:
        Authorization: "Bearer ${VM_CENTRAL_TOKEN}"
```

The `cluster` label flows automatically from each cluster’s Prometheus `external_labels`.

## B. Central forecaster, federated VM read

For small fleets that prefer one forecaster instance:

```
Cluster A: prometheus(A) ──remote_write──> VM(A) ─┐
Cluster B: prometheus(B) ──remote_write──> VM(B) ─┼──> VM(federated read) <── forecaster(central) ──remote_write──> VM(central)
```

One forecaster reads through a federated endpoint (e.g. VM’s `vmselect`).

**Trade-offs**  
- Simpler ops, native cross-cluster correlations.  
- Higher read latency and single point of failure.  
- Cardinality sized for the union of all clusters.

**Sample values**

```yaml
config:
  datasource:
    url: http://vm-federated.monitoring.svc.cluster.local:8481/select/0/prometheus
    timeout: 60s
  sink:
    remote_write:
      enabled: true
      url: http://victoriametrics.monitoring.svc.cluster.local:8428/api/v1/import/prometheus
  safety:
    max_total_series: 50000   # sized for union
```

## C. Per-cluster forecaster + per-cluster VM, Grafana mixed-datasource

Strict isolation (no central aggregation):

```
Cluster A: prometheus(A) ──remote_write──> VM(A) <── forecaster(A)
Cluster B: prometheus(B) ──remote_write──> VM(B) <── forecaster(B)
```

Grafana uses multiple datasources and mixed-datasource queries for cross-cluster views.

**Trade-offs**  
- Maximum isolation and compliance.  
- Cross-cluster dashboards and alerting are more complex.  
- No cross-cluster forecasts possible.

**Sample values** (identical per cluster, sink points to local VM)

```yaml
config:
  datasource:
    url: http://victoriametrics.monitoring.svc.cluster.local:8428
  sink:
    remote_write:
      enabled: true
      url: http://victoriametrics.monitoring.svc.cluster.local:8428/api/v1/import/prometheus
```

## Label alignment

The `cluster` label flows automatically from each cluster’s Prometheus `external_labels` through `remote_write` into the forecaster’s source queries and back out via the sink. No forecaster-side configuration is needed.

The forecaster’s operational metrics (`forecast_*`) describe its own state and are scoped by scrape labels (`job`, `instance`, `namespace`) rather than `cluster`.

## Cross-references

- [Production-readiness checklist](../production-checklist.md)
- [Emission paths](../reference/emission-paths.md) — labelset preservation details
- [High availability](../operations/ha.md) — pairs cleanly with Pattern A
