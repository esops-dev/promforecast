# Documentation

The docs follow the [Diataxis](https://diataxis.fr/) framework:

- **Tutorials** — learning-oriented, end-to-end walkthroughs.
- **How-tos** — task-oriented, "how do I X?" recipes.
- **Reference** — information-oriented, exhaustive details.
- **Explanation** — understanding-oriented, design and trade-offs.

## Index

- [Config schema](config-schema.md) — `apiVersion`, stability guarantees, and the `v1alpha1` → `v1` migration
- [Models](models.md) — built-in forecasters, auto-select, accuracy metrics
- [CLI](cli.md) — `promforecast run` and `promforecast validate`
- [High availability](ha.md) — Lease-based leader election, Redis snapshot cache
- [Query cache](cache.md) — in-memory LRU + optional Redis dedup for overlapping PromQL
- [Series discovery](discovery.md) — template-driven auto-fan-out over label values
- [Preprocessing](preprocessing.md) — gap imputation and outlier scrubbing
- [Change-points](change-points.md) — input-side break detection and lookback adaptation
- [Conformal calibration](conformal.md) — empirical-coverage band correction
- [Capacity planning](capacity.md) — damped trend, time-to-threshold ETAs, growth rate
- [Forecast drift](drift.md) — prediction-vs-prediction shift telemetry
- [Scheduling](scheduling.md) — per-query refresh intervals, max-concurrent-fits cap
- [Forecast preview](preview.md) — opt-in ad-hoc what-if endpoint
- [Regressors](regressors.md) — calendar, custom PromQL, and holiday regressors
- [Development](development.md) — local toolchain, CI overview, adding models

For a copy-and-edit reference deployment (multi-group config, HA values,
ArgoCD/Flux manifests, architecture diagram), see
[`examples/production/`](../examples/production/).

The configuration reference, metrics catalogue, and compatibility
matrix are tracked against later milestones — see `ROADMAP.md`.
