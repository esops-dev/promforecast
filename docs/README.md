# Documentation

Organised by pipeline position — input → model → output — then operations,
guides, and reference around the edges. If you're not sure where a topic
lives, the section descriptions below should point you in the right direction.

## Reference

Information-oriented, exhaustive details. Look here when you know the name
of the thing and want its full specification.

- [Config schema](reference/config-schema.md) — `apiVersion`, stability guarantees, and the `v1alpha1` → `v1` migration
- [CLI](reference/cli.md) — `promforecast run`, `promforecast validate`, `promforecast diagnose`
- [Models](reference/models.md) — built-in forecasters, auto-select, accuracy metrics
- [Emission paths (snapshot vs. sink + VM)](reference/emission-paths.md) — what each path emits, the dual-source trap, and why the official charts default to sink + VM

## Inputs

Everything that shapes the data flowing into the forecasting model.

- [Preprocessing](inputs/preprocessing.md) — gap imputation and outlier scrubbing
- [Change-points](inputs/change-points.md) — input-side break detection and lookback adaptation
- [Seasonality](inputs/seasonality.md) — periodicity detection
- [Cold-start](inputs/cold-start.md) — sparse-history behaviour and peer-sibling borrowing
- [Regressors](inputs/regressors.md) — calendar, custom PromQL, holiday, webhook (external pull), and calendar-file regressors, plus conditional forecasts (baseline given the current regressor state) and HMAC-signed webhooks
- [Series discovery](inputs/discovery.md) — template-driven auto-fan-out over label values

## Outputs

Everything the forecaster emits beyond raw forecast curves — derived
metrics, calibration, explainability, and ad-hoc what-if endpoints.

- [Capacity planning](outputs/capacity.md) — damped trend, time-to-threshold ETAs, growth rate, will-breach predictive boolean, queue burn-down (drain) forecasting
- [SLO error-budget forecasting](outputs/slo.md) — burn rate, exhaustion projection, multi-window alerting
- [Maintenance window recommendations](outputs/maintenance-windows.md) — top-N lowest-load windows for scheduling reboots, rollouts, and cert rotations
- [Conformal calibration](outputs/conformal.md) — empirical-coverage band correction
- [Explainability (decomposition)](outputs/explainability.md) — trend / seasonal / level / residual split of the forecast curve
- [Forecast drift](outputs/drift.md) — prediction-vs-prediction shift telemetry
- [Forecast preview](outputs/preview.md) — opt-in ad-hoc what-if endpoint
- [Fleet capacity what-if](outputs/what-if-fleet.md) — multi-service stress-test endpoint that ranks projected breaches under a hypothetical traffic multiplier
- [Deploy risk score](outputs/deploy-safety.md) — deploy-anchored composite of deviation across golden signals; pair with Argo Rollouts / Flagger as a rollback gate
- [Recurring incident patterns](outputs/incident-patterns.md) — 7 x 24 heatmap of when incidents tend to land; for queries with `data_profile: events`
- [Canary forecasts](outputs/canary.md)
- [Predictive autoscaling](predictive-autoscaling.md) — Kubernetes custom-metrics adapter that exposes forecast metrics to HorizontalPodAutoscaler for scaling on the projected load curve

## Integrations

- [Canary deploy forecast comparison](canary-analysis.md) — fit the baseline pods, project what the canary should look like, and emit `<id>_canary_forecast_deviation` for Argo Rollouts / Flagger AnalysisRun gates
- [Forecast-driven webhook triggers](triggers.md) — push structured forecast events (predicted value, band, model, drift) to a webhook when a PromQL condition transitions true; optional HMAC signing
- [Grafana annotations](integrations/grafana-annotations.md) — post significant forecast events (drift, change-point, deploy-risk, predicted breach, deviation) as Grafana annotations on the dashboards you already watch

## Operations

Running the forecaster in production — diagnosis, capacity safeguards,
scheduling, and serving telemetry.

- [Reload verification](operations/reload.md) — confirming a config reload actually picked up the expected queries
- [Duplicate-labelset diagnosis](operations/diagnose.md) — detecting and remediating the scrape+sink dual-source state
- [Cardinality budgeting](operations/cardinality.md) — per-family multipliers and the `--strict-cardinality` CI gate
- [Scheduling](operations/scheduling.md) — per-query refresh intervals, max-concurrent-fits cap
- [High availability](operations/ha.md) — Lease-based leader election, Redis snapshot cache
- [Sharded HA (horizontal scaling)](scaling/sharded-ha.md) — N active replicas each fitting a disjoint slice of groups, via `consistent_hash` or `leader_assigned`; the `shard_id` label and `forecast_shard_owns_group` gauge; sizing guide
- [Query cache](operations/cache.md) — in-memory LRU + optional Redis dedup for overlapping PromQL
- [Telemetry](operations/telemetry.md) — internal metrics catalogue + per-series staleness + self-observability (fit success ratio, regressor stability, per-group CPU/memory)
- [Warm-up status](operations/warmup.md) — fresh-install "what is still loading?" endpoint and CLI
- [Forecast inspector](operations/inspect.md) — debug a single weird forecast end-to-end, with optional per-fold backtest exposure
- [Historical backfill](operations/backfill.md) — replay the fit pipeline at past timestamps so SLO dashboards are meaningful from day one
- [Quality score](operations/quality-score.md) — composite reliability gauge, per-horizon segmentation, and which horizon to gate alerts on
- [Forecast stability](operations/forecast-stability.md) — `forecast_revision_magnitude` and `forecast_model_disagreement` for catching a thrashing model or a configured model set that can't agree on the future
- [Validate CLI](operations/validate.md) — `promforecast validate` modes: cardinality estimate, `--strict-cardinality` CI gate, `--estimate-cost` runtime budget, `--dry-run` sample fit
- [Degraded-mode operations](operations/degraded-modes.md) — failure-mode matrix covering datasource read failure, sink write failure, partial group failure, sustained fit timeout, and stale snapshot
- [Hyperparameter auto-tuning](operations/auto-tuning.md) — weekly recommendation CronJob that surfaces alternative `season_length` / model choices when the current config has drifted
- [Right-sizing recommendations](operations/right-sizing.md) — per-workload limit/request recommendations from the forecast peak, gated by quality score, optional savings-pct sibling
- [Drift change-correlation explainer](operations/drift-explained.md) — structured log + per-(group, query, primary_suspect) counter that turns a drift alert into "start your investigation on this regressor"

## Governance

Aggregations and reporting views over the forecaster's own output.

- [Service-level health composite](governance/forecast-health.md) — per-(service, environment) aggregate of `forecast_quality_score` for executive dashboards

## Cookbook

Copy-pasteable recipes for the common Prometheus metric families.
Each recipe ships a working YAML snippet, a recommended model + rationale,
the cardinality footprint, and example alerts.

- [Cookbook index](cookbook/README.md) — index of all recipes

## Guides

Cross-cutting explanations that don't fit cleanly under one feature.

- [Reading the numbers](guides/reading-the-numbers.md) — how to interpret quality and accuracy in practice (mean vs p95, horizon paradox, "wait three cycles", when to actually worry)
- [Production-readiness checklist](production-checklist.md) — sizing table, three reference architectures, and the boxes worth ticking off before flipping production traffic to a new install
- [Multi-cluster reference architectures](architecture/multi-cluster.md) — three patterns for deployments spanning multiple Kubernetes clusters, with sample chart values and cardinality / network trade-offs

## Contributing

- [Development](contributing/development.md) — local toolchain, CI overview, adding models

---

For a copy-and-edit reference deployment (multi-group config, HA values,
ArgoCD/Flux manifests, architecture diagram), see
[`examples/production/`](../examples/production/).

The configuration reference, metrics catalogue, and compatibility
matrix are tracked against later milestones — see `ROADMAP.md`.
