```markdown
# Cookbook

Recipes that turn *"I want to forecast X"* into *"here's the config, here's the alert, here's the dashboard"*.  

Each page is anchored to a single common Prometheus metric family, picks a sensible model + alert shape, and is exercised by CI (schema validation + cardinality estimator). If a recipe lands on your `/metrics` and behaves nothing like the dashboard panel described here, the test is the first place to look.

## Capacity & infrastructure

* [`node_filesystem_avail_bytes`](node-filesystem-avail-bytes.md) — predict disk-fill time, alert before the pager fires.
* [`redis_connected_clients`](redis-connected-clients.md) — see Redis connection-pool exhaustion hours before `maxclients` trips.
* [`kafka_consumer_lag`](kafka-consumer-lag.md) — capacity *and* drain forecasts on the same backlog.

## Workloads & application metrics

* [`http_requests_total`](http-requests-total.md) — service traffic forecast, SLO error-budget projections.
* [`kube_deployment_status_replicas`](kube-deployment-status-replicas.md) — predict autoscaler ramp before it surprises capacity.
* [`jvm_gc_pause_seconds`](jvm-gc-pause-seconds.md) — long GC pauses, drift-based anomaly detection.

## Cost & financial

* [`aws_cloudwatch_estimated_charges_usd`](aws-cloudwatch-estimated-charges-usd.md) — month-end bill projection.

Each recipe ships a copy-pasteable config in [`examples/configs/cookbook/`](../../examples/configs/cookbook/). Use that as the starting point and tune PromQL, thresholds, and cost framings to your environment.

## How a recipe is structured

Every page covers:

1. **The metric** — what it measures and why you'd forecast it.
2. **The config** — a copy-pasteable YAML snippet sized for a small fleet.
3. **The recommended model** — which `defaults.models` to pick and why.
4. **The expected forecast shape** — what the curve typically looks like in Grafana.
5. **The cardinality footprint** — what `promforecast validate` reports for the snippet.
6. **Example alerts** — one or two `PrometheusRule` snippets paired with the recipe.
```