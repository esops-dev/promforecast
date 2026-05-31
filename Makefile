# promforecast — top-level Makefile
#
# Convention: targets shell out into forecaster/ for Python work and into
# charts/ for Helm work. Run `make help` for a summary.

SHELL := /usr/bin/env bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

PY        := python3
UV        ?= uv
FORECASTER := forecaster
CHART_DIR := charts

IMAGE_REPO ?= ghcr.io/esops-dev/promforecast
IMAGE_TAG  ?= dev

.PHONY: help
help: ## Show this help.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ---------- Python ----------

.PHONY: install
install: ## Install Python dependencies into forecaster/.venv.
	cd $(FORECASTER) && $(UV) sync --all-extras

.PHONY: lint
lint: ## ruff + mypy --strict.
	cd $(FORECASTER) && $(UV) run ruff check .
	cd $(FORECASTER) && $(UV) run ruff format --check .
	cd $(FORECASTER) && $(UV) run mypy --strict src

.PHONY: format
format: ## Apply ruff formatting and import sorting.
	cd $(FORECASTER) && $(UV) run ruff format .
	cd $(FORECASTER) && $(UV) run ruff check --fix .

.PHONY: test
test: ## Run the Python test suite.
	cd $(FORECASTER) && $(UV) run pytest

.PHONY: backtest
backtest: ## Run a backtest for a group. Usage: make backtest GROUP=node_capacity
	cd $(FORECASTER) && $(UV) run python -m promforecast.accuracy --group $(GROUP)

# ---------- Container ----------

.PHONY: build
build: ## Build the forecaster container image.
	docker build -t $(IMAGE_REPO):$(IMAGE_TAG) $(FORECASTER)

# ---------- Helm ----------

.PHONY: chart-deps
chart-deps: ## Resolve umbrella subchart dependencies.
	helm dependency update $(CHART_DIR)/promforecast-stack

.PHONY: chart-lint
chart-lint: chart-deps ## helm lint + kubeconform on rendered manifests.
	helm lint $(CHART_DIR)/promforecast
	helm lint $(CHART_DIR)/promforecast-stack
	helm lint $(CHART_DIR)/promforecast-metrics-adapter
	helm template $(CHART_DIR)/promforecast                   | kubeconform -strict -summary -ignore-missing-schemas
	helm template $(CHART_DIR)/promforecast-stack             | kubeconform -strict -summary -ignore-missing-schemas
	helm template $(CHART_DIR)/promforecast-metrics-adapter   | kubeconform -strict -summary -ignore-missing-schemas
	helm template $(CHART_DIR)/promforecast-stack \
	    --set dashboards.enabled=true \
	    --set prometheusRules.enabled=true \
	  | kubeconform -strict -summary -ignore-missing-schemas
	@for f in dashboards/grafana/*.json; do python3 -m json.tool "$$f" > /dev/null; done
	@echo "dashboards JSON ok"
	@python3 -c "import re,sys,pathlib,yaml; \
e=yaml.safe_load(pathlib.Path('examples/alerts/promforecast-rules.yaml').read_text()); \
ex={r['alert'] for g in e['spec']['groups'] if not g['name'].endswith('.examples') for r in g['rules'] if 'alert' in r}; \
ch=set(re.findall(r'^\s*-\s+alert:\s+(\S+)\s*\$$', pathlib.Path('charts/promforecast-stack/templates/prometheusrule.yaml').read_text(), re.M)); \
sys.exit(f'alert drift: chart {sorted(ch-ex) or \"-\"}, example {sorted(ex-ch) or \"-\"}') if ex^ch else print(f'alert sets match ({len(ex)})')"

# values.schema.json is hand-authored alongside values.yaml so it can express
# constraints (enums, ranges, the current single-replica cap) that a generator
# can't infer from a sample. Keep them in sync manually when adding fields.

# ---------- Local dev ----------

.PHONY: dev-up
dev-up: ## Start the minimum dev stack: VM, Prometheus, Redis, node-exporter, forecaster.
	docker compose -f docker-compose.dev.yml up -d --build

.PHONY: dev-up-load
dev-up-load: ## Start the full demo stack: minimum + http-demo + loadgen + Grafana.
	docker compose -f docker-compose.dev.yml --profile load up -d --build

.PHONY: dev-down
dev-down: ## Stop the local dev stack (any profile).
	docker compose -f docker-compose.dev.yml --profile load down -v

.PHONY: dev-validate
dev-validate: ## Sanity-check the running dev stack: forecaster fits and exposes all metric families.
	@echo ">> forecaster /metrics summary"
	@python3 -c "import urllib.request as u; \
b=u.urlopen('http://localhost:9081/metrics').read().decode(); \
fams=sorted({l.split()[2] for l in b.splitlines() if l.startswith('# TYPE forecast') or l.startswith('# TYPE node_')}); \
[print(f'  {f}') for f in fams]; \
need={'forecast_accuracy_mape','forecast_quality_score','forecast_deviation_ratio','forecast_deviation_outside_band'}; \
missing=need-set(fams); \
exit(f'!! missing: {sorted(missing)} -- the forecaster has not produced its first fit yet, wait one refresh_interval and retry') if missing else print('>> all expected metric families present')"
	@echo ">> query cache is wired to Redis"
	@python3 -c "import urllib.request as u; \
b=u.urlopen('http://localhost:9081/metrics').read().decode(); \
lines=[l for l in b.splitlines() if l.startswith('forecast_query_cache_events_total')]; \
miss=next((float(l.split()[-1]) for l in lines if 'kind=\"miss\"' in l), 0.0); \
print(f'  miss={miss}'); \
exit('!! forecast_query_cache_events_total{kind=\"miss\"} is zero -- the cache is not on the fetch path') if miss == 0 else print('>> cache is active (misses observed)')"
	@echo ">> Prometheus has scraped the forecaster"
	@python3 -c "import urllib.request as u, json; \
r=json.loads(u.urlopen('http://localhost:9080/api/v1/query?query=forecast_accuracy_mape').read())['data']['result']; \
print(f'  forecast_accuracy_mape: {len(r)} series'); \
exit('!! Prometheus has 0 series - check the prometheus scrape config and forecaster reachability') if not r else None"
	@echo ">> no label-divergent dual-source forecast labelsets in VM (regression guard for the scrape-drop)"
	@cd $(FORECASTER) && .venv/bin/promforecast diagnose duplicates --datasource http://localhost:8429 --since 10m \
	  > /tmp/promforecast-diagnose.txt 2>&1 \
	  && echo ">> diagnose duplicates: clean" \
	  || (echo "!! diagnose duplicates flagged a label-divergent duplicate state; see /tmp/promforecast-diagnose.txt"; tail -30 /tmp/promforecast-diagnose.txt; exit 1)
	@echo ">> no same-labelset dual-source forecast ingest in VM (regression guard for sample-rate contamination)"
	@# The dev config fits a 1m-step curve (docker/dev-config.yaml: step 1m), so
	@# one ingest path legitimately writes ~10 samples per 10m window. Dedup on
	@# VM collapses the refit overlap to one sample per grid timestamp; the cap
	@# sits above that floor (~11) but well below the rate a leaked snapshot
	@# scrape (every 15s ~ 40/10m) would add, so genuine dual-source still trips.
	@cd $(FORECASTER) && .venv/bin/promforecast diagnose sample-rate --datasource http://localhost:8429 --window 10m --max-samples 20 \
	  > /tmp/promforecast-sample-rate.txt 2>&1 \
	  && echo ">> diagnose sample-rate: clean" \
	  || (echo "!! diagnose sample-rate flagged at least one series above the per-window cap; see /tmp/promforecast-sample-rate.txt"; tail -30 /tmp/promforecast-sample-rate.txt; exit 1)
	@if docker compose -f docker-compose.dev.yml ps --status running --services 2>/dev/null | grep -qx grafana; then \
	  echo ">> dashboards reachable in Grafana at http://localhost:3001 (admin / admin)"; \
	else \
	  echo ">> Grafana not running (minimum dev stack). Run 'make dev-up-load' to start the demo stack with Grafana + http-demo + loadgen."; \
	fi

.PHONY: load-test
load-test: ## Synthetic high-cardinality run for safety-control validation.
	cd $(FORECASTER) && $(UV) run python -m promforecast.scripts.load_test

# ---------- Aggregate ----------

.PHONY: ci
ci: lint test chart-lint ## Everything CI runs.
