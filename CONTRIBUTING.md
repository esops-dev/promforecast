# Contributing

Thanks for your interest in promforecast.

## Before you start

- Open an issue first for non-trivial changes — design discussion is cheaper
  than re-doing a PR.
- "Use a recording rule" is often the right answer for derived-metric
  requests; promforecast emits raw forecasts and lets the wider Prometheus
  ecosystem compose on top.

## Development

```bash
make dev-up           # docker-compose: VM + node-exporter + Prometheus + forecaster
make test             # pytest
make lint             # ruff + mypy --strict
make chart-lint       # helm lint + kubeconform
```

## Pull requests

- One logical change per PR.
- Tests for new behaviour. Synthetic-series fixtures live in
  `forecaster/tests/`.
- Output metric names are part of the public API. Renames are breaking
  changes and require a deprecation period.
- `mypy --strict` and `ruff` must be clean.
- Adding a model? Update three places: `forecaster/src/promforecast/models/`,
  `pyproject.toml` entry points, and `docs/models.md`.
- Adding a Helm value? Regenerate `values.schema.json` (`make chart-schema`).

## Commit style

Conventional Commits are encouraged but not enforced. PR titles flow into
release notes; keep them descriptive.

## Code of conduct

Be kind. Assume good faith. Disagreement on design is welcome; personal
attacks are not.

## License

By contributing you agree your contribution is licensed under
[Apache-2.0](LICENSE).
