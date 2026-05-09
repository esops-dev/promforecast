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
	helm template $(CHART_DIR)/promforecast       | kubeconform -strict -summary -ignore-missing-schemas
	helm template $(CHART_DIR)/promforecast-stack | kubeconform -strict -summary -ignore-missing-schemas

# values.schema.json is hand-authored alongside values.yaml so it can express
# constraints (enums, ranges, the v0.1 single-replica cap) that a generator
# can't infer from a sample. Keep them in sync manually when adding fields.

# ---------- Local dev ----------

.PHONY: dev-up
dev-up: ## Start VictoriaMetrics + node-exporter + Prometheus + forecaster locally.
	docker compose -f docker-compose.dev.yml up -d --build

.PHONY: dev-down
dev-down: ## Stop the local dev stack.
	docker compose -f docker-compose.dev.yml down -v

.PHONY: load-test
load-test: ## Synthetic high-cardinality run for safety-control validation.
	cd $(FORECASTER) && $(UV) run python -m promforecast.scripts.load_test

# ---------- Aggregate ----------

.PHONY: ci
ci: lint test chart-lint ## Everything CI runs.
