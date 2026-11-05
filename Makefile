SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help
.DELETE_ON_ERROR:

-include local.mk

MODEL_STORE ?= build
export SINHALALM_STORE := $(abspath $(MODEL_STORE))

UV ?= uv
NPM ?= npx -y npm@11.21.1
WEB := web
REFERENCE_ENV := .venv-reference
PY := env -u VIRTUAL_ENV $(UV) run python
REF_PY := env -u VIRTUAL_ENV UV_PROJECT_ENVIRONMENT=$(REFERENCE_ENV) $(UV) run --no-default-groups --group reference python

GOLDEN_LOGITS := $(SINHALALM_STORE)/golden/reference_logits.npz
GATE1_REPORT := $(SINHALALM_STORE)/gate1/parity.json
GOLDEN_INPUTS := convert/sinhalallm.py convert/make_golden.py golden/prompts.json uv.lock
GATE1_INPUTS := convert/sinhalallm.py convert/parity_check.py golden/reference.json uv.lock
TOKENIZER_CORPUS := golden/generated/tokenizer_corpus.json
TOKENIZER_INPUTS := convert/sinhalallm.py convert/make_golden.py uv.lock

.PHONY: help setup setup-python setup-web lint lint-python lint-web format golden check-golden convert verify test-web ci

help:
	@grep -E '^[a-z][a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-14s %s\n", $$1, $$2}'

setup: setup-python setup-web ## Install Python and Node dependencies

setup-python:
	env -u VIRTUAL_ENV $(UV) sync --frozen
	env -u VIRTUAL_ENV UV_PROJECT_ENVIRONMENT=$(REFERENCE_ENV) $(UV) sync --frozen --no-default-groups --group reference

setup-web:
	cd $(WEB) && $(NPM) ci

lint: lint-python lint-web ## Run all linters and format checks

lint-python:
	$(UV) run ruff check convert
	$(UV) run ruff format --check convert

lint-web:
	cd $(WEB) && $(NPM) run lint

format: ## Apply formatters to Python and web sources
	$(UV) run ruff format convert
	$(UV) run ruff check --fix convert
	cd $(WEB) && $(NPM) run format

$(GOLDEN_LOGITS): $(GOLDEN_INPUTS)
	$(REF_PY) convert/make_golden.py reference

$(TOKENIZER_CORPUS): $(TOKENIZER_INPUTS)
	$(REF_PY) convert/make_golden.py tokenizer

$(GATE1_REPORT): $(GATE1_INPUTS) $(GOLDEN_LOGITS)
	$(PY) convert/parity_check.py

golden: ## Regenerate golden outputs and tokenizer cases with the reference toolchain
	$(REF_PY) convert/make_golden.py reference
	$(REF_PY) convert/make_golden.py tokenizer

check-golden: ## Verify golden outputs and tokenizer cases are reproducible
	$(REF_PY) convert/make_golden.py reference --check
	$(REF_PY) convert/make_golden.py tokenizer --check

convert: $(GATE1_REPORT) ## Gate 1: verify the export toolchain against the reference

verify: check-golden ## Rerun golden generation and gate 1 from scratch
	$(PY) convert/parity_check.py

test-web: $(TOKENIZER_CORPUS) ## Run web unit tests, including gate 3
	cd $(WEB) && $(NPM) test

ci: lint convert test-web ## Run the local CI pipeline
