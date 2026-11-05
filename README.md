# SinhalaLM Web — Offline Sinhala Question Answering in the Browser

For a long period of time, running a language model meant sending text to a server and waiting for a response. SinhalaLLM-1.7B is an open, instruction-tuned Sinhala language model built on SmolLM2-1.7B, with a tokenizer extended by 14,192 Sinhala tokens. At 1.74B parameters, it is a candidate for client-side inference on desktop and laptop computers.

This project runs SinhalaLLM-1.7B fully offline in the browser for single-turn Sinhala question answering. After the first load, no server and no network connection are needed.

The project started with the HelaAI SinhalaLM models, which were replaced after their answers did not reach a satisfactory quality. The reasons and the comparison of alternatives are recorded in [model-selection.md](docs/findings/model-selection.md).

The full design is described in the [project proposal](docs/sinhalalm-browser-proposal.md).

### How it works

Rather than porting a custom runtime, the project treats the deployment as a sequence of verifiable transformations. Each step must reproduce the output of the previous step within a defined tolerance before the next step begins.

- **Reference load** — the model is loaded in the export toolchain with a translated configuration, since `transformers` 4.57 otherwise reads its RoPE base as 10000 instead of 130000, and gate 1 verifies the load against `transformers` 5.19.
- **ONNX export and quantization** — the model is exported with key-value caching and quantized into fp16, q8 and q4 variants.
- **Tokenizer** — the transformers.js tokenizer is verified against the Python tokenizer, so that the browser produces the exact token IDs the model was trained on.
- **ZWJ handling** — the zero-width joiner is added to user input where it is missing (for example, `්ර` becomes `්‍ර`), since the model learned conjuncts in that form, and HTML entities are decoded in the output.
- **Browser runtime** — transformers.js runs the model in a Web Worker on WebGPU, with a WebAssembly fallback.
- **Offline layer** — a service worker and the browser's Cache Storage keep the page and the model available without a connection.

### Project status

| Phase | Scope | Status |
|---|---|---|
| 0 | Project bootstrap and toolchain checks | Done |
| 1 | Model selection, reference outputs and configuration translation (gate 1) | In progress |
| 2 | Tokenizer parity (gate 3) and ZWJ handling | Planned |
| 3 | ONNX export (gate 2), quantization and RQ1 | Planned |
| 4 | Browser application, offline mode and gate 4 | Planned |
| 5 | Hosting checks on a live Space and deployment to a Hugging Face static Space | Planned |
| 6 | Benchmarks (RQ2) and ZWJ fidelity evaluation (RQ3) | Planned |

Research results are summarized in [FINDINGS.md](FINDINGS.md), the outputs of both models on the golden prompts are listed in [docs/model-outputs.md](docs/model-outputs.md), alternative models are compared in [docs/model-comparison.md](docs/model-comparison.md), and every change is recorded in [CHANGELOG.md](CHANGELOG.md).

### Project structure

```
sinhalalm-web/
├── convert/          # Python build-time pipeline: conversion, export, quantization, evaluation
├── golden/           # fixtures shared by the Python and JavaScript tests
├── docs/             # proposal and per-topic findings
├── web/
│   ├── src/text/
│   ├── src/worker/
│   ├── src/ui/
│   ├── public/models/<model>/<version>/
│   └── tests/
└── Makefile          # local CI pipeline
```

### Getting started

The project requires `uv`, Python 3.12, Node.js 22 or later and `make`. npm 11 is pinned through `npx` in the Makefile, so the globally installed npm version does not matter.

```bash
make setup   # install Python and Node dependencies
make golden  # regenerate golden reference outputs
make convert # gate 1
make ci      # run the local CI pipeline
make help    # list all targets
```

The first `make golden` or `make ci` downloads the model checkpoints at pinned revisions with verified SHA-256 hashes. Until the pipeline is replaced for SinhalaLLM-1.7B, these targets still run the previous HelaAI pipeline.

### Model store

Downloaded checkpoints, converted models and generated logits are kept in a model store outside the repository, since they are too large for the internal disk. The store defaults to `build/`, and a local `local.mk` file (not committed) can point it to another location, such as an external drive:

```make
MODEL_STORE := /media/<user>/<drive>/sinhalalm-web
```

The store is read and written only while a pipeline target runs, so the drive needs to be mounted only for `make golden`, `make convert` and `make ci`. If the store is not available, these targets stop with a message instead of downloading the models again.

### Development workflow

Every feature or fix is developed on its own branch and merged into `main` only after `make ci` passes. Features that are visible in the browser are also tested manually and refined through change requests before they are merged. In other words, the CI pipeline runs locally through the Makefile rather than on a hosted CI service.

### Credits

SinhalaLLM-1.7B is developed by Namindu Sanchila Nadeeshan and released under the Apache-2.0 license: [namindu94/SinhalaLLM-1.7B](https://huggingface.co/namindu94/SinhalaLLM-1.7B). It is built on [SmolLM2-1.7B](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B) by Hugging Face.

The model often gets facts wrong from memory, its answers are unverified, and it is not suitable for medical, legal or financial questions.

The earlier stages of the project used the HelaAI SinhalaLM models ([HelaAI/SinhalaLM-V.0.0.1](https://huggingface.co/HelaAI/SinhalaLM-V.0.0.1) and [HelaAI/SinhalaLM-V.0.0.1-it](https://huggingface.co/HelaAI/SinhalaLM-V.0.0.1-it)).
