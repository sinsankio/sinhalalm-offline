SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help
.DELETE_ON_ERROR:

-include local.mk

MODEL_STORE ?= build
export SINHALALM_STORE := $(abspath $(MODEL_STORE))
EXPORT_MEMORY ?= 11G
export SINHALALM_EXPORT_MEMORY := $(EXPORT_MEMORY)
QUANTIZE_MEMORY ?= 14G

UV ?= uv
NPM ?= npx -y npm@11.21.1
WEB := web
REFERENCE_ENV := .venv-reference
PY := env -u VIRTUAL_ENV $(UV) run python
REF_PY := env -u VIRTUAL_ENV UV_PROJECT_ENVIRONMENT=$(REFERENCE_ENV) $(UV) run --no-default-groups --group reference python
CAPPED_PY := $(if $(shell command -v systemd-run),systemd-run --user --scope --quiet -p MemoryMax=$(QUANTIZE_MEMORY) -p MemorySwapMax=0) $(PY)

STAMPS := build/stamps
GOLDEN_LOGITS := $(SINHALALM_STORE)/golden/reference_logits.npz
TOKENIZER_CORPUS := golden/generated/tokenizer_corpus.json
GATE1_REPORT := $(SINHALALM_STORE)/gate1/parity.json
ONNX_FP32 := $(SINHALALM_STORE)/onnx/fp32/export.json
GATE2_REPORT := $(SINHALALM_STORE)/onnx/fp32/gate2.json
ONNX_FP16 := $(SINHALALM_STORE)/onnx/fp16/export.json
QUANTIZED := $(foreach variant,q4-v0 q4-v0-small q8,$(SINHALALM_STORE)/onnx/$(variant)/quantization.json)

COMMON_INPUTS := convert/sinhalallm.py uv.lock
GOLDEN_INPUTS := $(COMMON_INPUTS) convert/make_golden.py golden/prompts.json
TOKENIZER_INPUTS := $(COMMON_INPUTS) convert/make_golden.py
GATE1_INPUTS := $(COMMON_INPUTS) convert/parity_check.py golden/reference.json
EXPORT_INPUTS := $(COMMON_INPUTS) convert/parity_check.py convert/export_onnx.py
GATE2_INPUTS := $(COMMON_INPUTS) convert/eval_quality.py golden/reference.json
QUANTIZE_INPUTS := $(COMMON_INPUTS) convert/quantize.py

define content_stamp
$(STAMPS)/$(1).sha256: FORCE
	@mkdir -p $(STAMPS)
	@sha256sum $(2) > $$@.new
	@if cmp -s $$@.new $$@; then rm $$@.new; else mv $$@.new $$@; fi
endef

$(eval $(call content_stamp,golden,$(GOLDEN_INPUTS)))
$(eval $(call content_stamp,tokenizer,$(TOKENIZER_INPUTS)))
$(eval $(call content_stamp,gate1,$(GATE1_INPUTS)))
$(eval $(call content_stamp,export,$(EXPORT_INPUTS)))
$(eval $(call content_stamp,gate2,$(GATE2_INPUTS)))
$(eval $(call content_stamp,quantize,$(QUANTIZE_INPUTS)))

.PHONY: FORCE help setup setup-python setup-web lint lint-python lint-web format golden check-golden convert onnx quantize rq1 verify test-web ci

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

$(GOLDEN_LOGITS): $(STAMPS)/golden.sha256
	$(REF_PY) convert/make_golden.py reference

$(TOKENIZER_CORPUS): $(STAMPS)/tokenizer.sha256
	$(REF_PY) convert/make_golden.py tokenizer

$(GATE1_REPORT): $(GOLDEN_LOGITS) $(STAMPS)/gate1.sha256
	$(PY) convert/parity_check.py

$(ONNX_FP32): $(STAMPS)/export.sha256 | $(GATE1_REPORT)
	$(PY) convert/export_onnx.py

$(GATE2_REPORT): $(ONNX_FP32) $(GOLDEN_LOGITS) $(STAMPS)/gate2.sha256
	$(PY) convert/eval_quality.py gate2

$(ONNX_FP16): $(STAMPS)/export.sha256 | $(GATE1_REPORT) $(ONNX_FP32)
	$(PY) convert/export_onnx.py --dtype fp16

$(SINHALALM_STORE)/onnx/%/quantization.json: $(ONNX_FP32) $(STAMPS)/quantize.sha256
	$(CAPPED_PY) convert/quantize.py $*

golden: ## Regenerate golden outputs and tokenizer cases with the reference toolchain
	$(REF_PY) convert/make_golden.py reference
	$(REF_PY) convert/make_golden.py tokenizer

check-golden: ## Verify golden outputs and tokenizer cases are reproducible
	$(REF_PY) convert/make_golden.py reference --check
	$(REF_PY) convert/make_golden.py tokenizer --check

convert: $(GATE1_REPORT) ## Gate 1: verify the export toolchain against the reference

onnx: $(GATE2_REPORT) ## Export fp32 ONNX and run gate 2 against the reference

quantize: $(ONNX_FP16) $(QUANTIZED) ## Build the fp16, q8 and q4 candidate variants

rq1: quantize ## RQ1: perplexity and answer agreement of every candidate variant (slow)
	$(CAPPED_PY) convert/eval_quality.py rq1 --variant fp32
	$(CAPPED_PY) convert/eval_quality.py rq1 --variant q4-v0
	$(CAPPED_PY) convert/eval_quality.py rq1 --variant q4-v0-small
	$(CAPPED_PY) convert/eval_quality.py rq1 --variant fp16 --sentences 200 --questions 3
	$(CAPPED_PY) convert/eval_quality.py rq1 --variant q8 --sentences 50 --questions 1

verify: check-golden ## Rerun golden generation and gate 1 from scratch
	$(PY) convert/parity_check.py

test-web: $(TOKENIZER_CORPUS) ## Run web unit tests, including gate 3
	cd $(WEB) && $(NPM) test

ci: lint convert onnx quantize test-web ## Run the local CI pipeline
