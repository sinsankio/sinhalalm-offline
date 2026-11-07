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
- **ZWJ handling** — the zero-width joiner in the reader's question is normalized to the 15 conjunct clusters the model learned with it (for example, `ප්ර` becomes `ප්‍ර`, while `ව්‍ය` becomes `ව්ය`), and HTML entities are decoded in the output.
- **Browser runtime** — transformers.js runs the model in a Web Worker on WebGPU, with a WebAssembly fallback.
- **Offline layer** — a service worker and the browser's Cache Storage keep the page and the model available without a connection.

### Project status

| Phase | Scope | Status |
|---|---|---|
| 0 | Project bootstrap and toolchain checks | Done |
| 1 | Model selection, reference outputs and configuration translation (gate 1) | Done |
| 2 | Tokenizer parity (gate 3) and ZWJ handling | Done |
| 3 | ONNX export (gate 2), quantization and RQ1 | Done |
| 4 | Browser application, offline mode and gate 4 | In progress |
| 5 | Hosting checks on a live Space and deployment to a Hugging Face static Space | Planned |
| 6 | Benchmarks (RQ2) and ZWJ fidelity evaluation (RQ3) | Planned |

Research results are summarized in [FINDINGS.md](FINDINGS.md), the answers of the quantization candidates are compared in [docs/quantization-comparison.md](docs/quantization-comparison.md), the answers of SinhalaLLM-1.7B on the golden questions are listed in [docs/model-outputs.md](docs/model-outputs.md), alternative models are compared in [docs/model-comparison.md](docs/model-comparison.md), and every change is recorded in [CHANGELOG.md](CHANGELOG.md).

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
make golden  # regenerate golden outputs and tokenizer cases with the reference toolchain (transformers 5.19.0)
make convert # gate 1: verify the export toolchain (transformers 4.57.6) against the reference
make test-web # web unit tests, including gate 3 (tokenizer parity on 10,000 sentences)
make onnx    # export fp32 ONNX and run gate 2 (needs about 11.5 GB of free memory)
make quantize # build the fp16, q8 and q4 candidate variants (needs about 13 GB of free memory)
make quality # quality gate: the default variant (q4-v0-small) against the author's Q4_K_M release
make package # package the default variant for the web app and link it into web/public/models
make check-package # verify the packaged model in transformers.js against Python
make rq1     # RQ1: perplexity and answer agreement of every candidate (slow; needs FLORES+ access)
make verify  # rerun golden generation and gate 1 from scratch
make ci      # run the local CI pipeline
make help    # list all targets
```

The first `make golden` or `make ci` downloads SinhalaLLM-1.7B (about 3.5 GB) at a pinned revision and verifies the SHA-256 hash of every file. The golden outputs, gate 1, the ONNX export and gate 2 each take minutes on a CPU, so `make ci` reruns a step only when the content of its inputs changes. Content hashes of the inputs are kept in `build/stamps/`, so a checkout or merge that only changes file times reruns nothing, while `make verify` reruns golden generation and gate 1 from scratch.

The pipeline uses two Python environments from the same `uv` project, since the model was saved with `transformers` 5.19.0 while the ONNX exporter requires 4.57.6. `make setup` creates both: `.venv` for the export toolchain and `.venv-reference` for the reference.

### Model store

Downloaded checkpoints, converted models and generated logits are kept in a model store outside the repository, since they are too large for the internal disk. The store defaults to `build/`, and a local `local.mk` file (not committed) can point it to another location, such as an external drive:

```make
MODEL_STORE := /media/<user>/<drive>/sinhalalm-web
```

The web application loads the packaged model through a link, `web/public/models/sinhalallm-1.7b`, which points into the model store, so the drive also needs to be mounted while the application runs locally. The store is otherwise read and written only while a pipeline target runs, so the drive needs to be mounted only for the pipeline targets, such as `make golden`, `make convert`, `make verify` and `make ci`. If the store is not available, these targets stop with a message instead of downloading the models again.

### Development workflow

Every feature or fix is developed on its own branch and merged into `main` only after `make ci` passes. Features that are visible in the browser are also tested manually and refined through change requests before they are merged. In other words, the CI pipeline runs locally through the Makefile rather than on a hosted CI service.

### Credits

SinhalaLLM-1.7B is developed by Namindu Sanchila Nadeeshan and released under the Apache-2.0 license: [namindu94/SinhalaLLM-1.7B](https://huggingface.co/namindu94/SinhalaLLM-1.7B). It is built on [SmolLM2-1.7B](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B) by Hugging Face.

The model often gets facts wrong from memory, its answers are unverified, and it is not suitable for medical, legal or financial questions.

The earlier stages of the project used the HelaAI SinhalaLM models ([HelaAI/SinhalaLM-V.0.0.1](https://huggingface.co/HelaAI/SinhalaLM-V.0.0.1) and [HelaAI/SinhalaLM-V.0.0.1-it](https://huggingface.co/HelaAI/SinhalaLM-V.0.0.1-it)).
