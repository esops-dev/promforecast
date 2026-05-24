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
- [Regressors](inputs/regressors.md) — calendar, custom PromQL, and holiday regressors, plus conditional forecasts (baseline given the current regressor state)
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
- [Canary forecasts](outputs/canary.md)

## Operations

Running the forecaster in production — diagnosis, capacity safeguards,
scheduling, and serving telemetry.

- [Reload verification](operations/reload.md) — confirming a config reload actually picked up the expected queries
- [Duplicate-labelset diagnosis](operations/diagnose.md) — detecting and remediating the scrape+sink dual-source state
- [Cardinality budgeting](operations/cardinality.md) — per-family multipliers and the `--strict-cardinality` CI gate
- [Scheduling](operations/scheduling.md) — per-query refresh intervals, max-concurrent-fits cap
- [High availability](operations/ha.md) — Lease-based leader election, Redis snapshot cache
- [Query cache](operations/cache.md) — in-memory LRU + optional Redis dedup for overlapping PromQL
- [Telemetry](operations/telemetry.md) — internal metrics catalogue

## Guides

Cross-cutting explanations that don't fit cleanly under one feature.

- [Reading the numbers](guides/reading-the-numbers.md) — how to interpret quality and accuracy in practice (mean vs p95, horizon paradox, "wait three cycles", when to actually worry)

## Contributing

- [Development](contributing/development.md) — local toolchain, CI overview, adding models

---

For a copy-and-edit reference deployment (multi-group config, HA values,
ArgoCD/Flux manifests, architecture diagram), see
[`examples/production/`](../examples/production/).

The configuration reference, metrics catalogue, and compatibility
matrix are tracked against later milestones — see `ROADMAP.md`.
