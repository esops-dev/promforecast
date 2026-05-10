# Development

Quick reference for contributors.

**Design rationale** lives in the source: see the docstrings in `runner.py`, `exporter.py`, and `reload.py`.

## Toolchain

```bash
make install          # uv sync --all-extras
make lint             # ruff + mypy --strict
make test             # pytest
make chart-lint       # helm lint, kubeconform, schema drift, alert drift
make dev-up           # docker compose: VM + node-exporter + Prometheus + forecaster
make dev-validate     # smoke-test the running dev stack
```

Package manager is [`uv`](https://docs.astral.sh/uv/).

## CI overview

Every PR runs these four jobs (`.github/workflows/ci.yml`):

| Job          | What it runs                                                    |
|--------------|-----------------------------------------------------------------|
| `lint`       | `ruff check`, `ruff format --check`, `mypy --strict src`       |
| `test`       | `pytest --cov=promforecast --cov-report=xml`                   |
| `chart-lint` | Full Helm/Kubernetes validation (see below)                     |
| `container`  | Multi-stage Docker build + Trivy scan                           |

### `chart-lint` in detail

This job is the single source of truth for chart correctness:

1. **`ct lint`** — validates both charts (`Chart.yaml`, `values.yaml`, `helm lint`).
2. **`helm lint --strict`** — treats deprecation warnings as errors.
3. **`kubeconform`** — renders manifests (default values + dashboards/rules) and checks them against the Kubernetes schema.
4. **values.schema.json drift check** — renders the forecaster chart, extracts the generated `config.yaml`, and validates it against the pydantic schema. Catches configs that would crash the forecaster on boot.
5. **Alert-drift check** — ensures `examples/alerts/promforecast-rules.yaml` and the chart’s `PrometheusRule` declare the same core alert names.

**When you add a chart value**:
- Regenerate `values.schema.json`.
- Update the `config` block in `values.yaml`.
- The drift check will fail if the new field breaks the schema.

## Adding a metric

1. Edit `examples/configs/<group>.yaml` (or your `values.yaml`).
2. `make dev-up` and check `http://localhost:9091/metrics`.
3. `make backtest GROUP=<name>` to review MAPE.
4. Open a PR — CI will catch schema, kubeconform, and alert-drift issues.

## Adding a model

Three files must stay in sync:

1. `forecaster/src/promforecast/models/<name>.py` (model wrapper).
2. `forecaster/pyproject.toml` (entry-point registration).
3. `docs/models.md` (user docs).

CI automatically tests the full registered model set.

## Local config validation

Use the same command CI uses:

```bash
uv run promforecast validate --config examples/configs/node-capacity.yaml

# With live datasource check:
uv run promforecast validate --config examples/configs/node-capacity.yaml --dry-run --datasource-url http://localhost:8428/
```

`--dry-run` needs a reachable VictoriaMetrics instance (`make dev-up` is the easiest way).
