# Findings

This document summarizes the research and development findings of the project in one place. Each finding is described in detail in its own document under [docs/findings/](docs/findings/), and this summary links to it.

### Index

| Document | Topic | Summary |
|---|---|---|
| [toolchain.md](docs/findings/toolchain.md) | Export, runtime, quantization and hosting constraints | The planned Llama export path works with current tools, but only with pinned versions, and q4 leaves the tied embedding and head in fp32. |

### Toolchain check

A tiny Llama model with the SinhalaLM configuration was exported, run and quantized end to end before any conversion code was written.

- **Export versions are pinned below the latest releases.** `optimum-onnx` 0.1.0 requires `transformers<4.58` and `optimum~=2.1.0`, so the pipeline uses `transformers` 4.57.6 and `optimum` 2.1.0.
- **`accelerate` is required for tied weights.** Without it, the exporter stored a second copy of the embedding, which added 30.72 MB to the tiny model.
- **Parity holds.** ONNX Runtime matched PyTorch with a maximum absolute logit difference of 1.52e-6, and transformers.js 4.3.2 reproduced the PyTorch greedy output exactly.
- **The streamer works as the proposal assumes.** The first `put()` call carries the prompt, and every later call carries one token as a nested array of `BigInt` values.
- **The transformers.js quantize script is gone.** It was removed in version 4, so the pipeline quantizes with ONNX Runtime's own tools.
- **q4 is larger than q8 with tied weights.** The 4-bit quantizer skips the output head because it reads the shared embedding through a `Transpose`, so the tiny model was 34.73 MB in q4 and 14.08 MB in q8. Two fixes are evaluated in RQ1.
- **Space headers are restricted.** Only the COOP, COEP and CORP headers can be set, in lowercase. Iframe isolation and service worker behavior are verified on a live Space in Phase 5.
- **Software WebGPU adapters must be rejected.** Headless Chrome on Linux exposed only a `swiftshader` adapter, so the worker falls back to WASM in that case.
