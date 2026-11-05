SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

UV ?= uv
NPM ?= npx -y npm@11.21.1
WEB := web

.PHONY: help setup setup-python setup-web lint lint-python lint-web format test-web ci

help:
	@grep -E '^[a-z][a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-14s %s\n", $$1, $$2}'

setup: setup-python setup-web ## Install Python and Node dependencies

setup-python:
	$(UV) sync --frozen

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

test-web: ## Run web unit tests
	cd $(WEB) && $(NPM) test

ci: lint test-web ## Run the full local CI pipeline
