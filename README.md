# SinhalaLM Web — Offline Sinhala Question Answering in the Browser

For a long period of time, running a language model meant sending text to a server and waiting for a response. SinhalaLM changes that equation for Sinhala: it is a compact GPT-style model from HelaAI with only 32.85M unique parameters, which is small enough to run entirely inside a web browser.

This project runs both SinhalaLM checkpoints fully offline in the browser. The base model continues Sinhala text, and the instruction-tuned model answers single-turn Sinhala questions. After the first load, no server and no network connection are needed.

The full design is described in the [project proposal](docs/sinhalalm-browser-proposal.md).

### How it works

Rather than porting a custom runtime, the project treats the deployment as a sequence of verifiable transformations. Each step must reproduce the output of the previous step within a defined tolerance before the next step begins.

- **Llama conversion** — the SinhalaGPT weights are mapped to `LlamaForCausalLM`, since the architecture is mathematically equivalent to Llama.
- **ONNX export and quantization** — the converted model is exported with key-value caching and quantized into fp16, q8 and q4 variants.
- **Tokenizer port** — a JavaScript port of `FastBPETokenizer` reproduces the exact token IDs the model was trained on, since the published `tokenizer.json` cannot be loaded.
- **ZWJ handling** — the zero-width joiner is removed from user input and recovered in the output, so that conjuncts such as `ක්‍ර` render correctly.
- **Browser runtime** — transformers.js runs the model in a Web Worker on WebGPU, with a WebAssembly fallback.
- **Offline layer** — a service worker and the browser's Cache Storage keep the page and the model available without a connection.

### Project status

| Phase | Scope | Status |
|---|---|---|
| 0 | Project bootstrap and toolchain checks | In progress |
| 1 | Reference outputs and Llama conversion (gate 1) | Planned |
| 2 | Tokenizer port (gate 3) and ZWJ handling | Planned |
| 3 | ONNX export (gate 2), quantization and RQ1 | Planned |
| 4 | Browser application, offline mode and gate 4 | Planned |
| 5 | Deployment to a Hugging Face static Space | Planned |
| 6 | Benchmarks (RQ2) and ZWJ recovery evaluation (RQ3) | Planned |

Research results are summarized in [FINDINGS.md](FINDINGS.md), and every change is recorded in [CHANGELOG.md](CHANGELOG.md).

### Project structure

```
sinhalalm-web/
├── convert/          # Python build-time pipeline: conversion, export, quantization, evaluation
├── golden/           # fixtures shared by the Python and JavaScript tests
├── docs/             # proposal and per-topic findings
├── web/
│   ├── src/tokenizer/
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
make ci      # run the local CI pipeline
make help    # list all targets
```

### Development workflow

Every feature or fix is developed on its own branch and merged into `main` only after `make ci` passes. Features that are visible in the browser are also tested manually and refined through change requests before they are merged. In other words, the CI pipeline runs locally through the Makefile rather than on a hosted CI service.

### Credits

SinhalaLM is developed by HelaAI. This project builds on both model cards:

- [HelaAI/SinhalaLM-V.0.0.1](https://huggingface.co/HelaAI/SinhalaLM-V.0.0.1) — base model
- [HelaAI/SinhalaLM-V.0.0.1-it](https://huggingface.co/HelaAI/SinhalaLM-V.0.0.1-it) — instruction-tuned question-answering model

Both models are intended for research and prototyping. Their outputs may be fluent but factually incorrect, and they are not suitable for medical, legal or financial questions.
