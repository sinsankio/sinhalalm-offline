# Toolchain Check: Export, Runtime, Quantization and Hosting Constraints

The proposal was written against the tools that existed at the time, and several of them have moved on since then. transformers.js is now at major version 4, the Optimum ONNX exporter lives in a separate `optimum-onnx` package, and the Hugging Face Spaces configuration defines its own rules for response headers. Before writing any conversion code, we therefore checked whether the planned path still holds end to end.

In this document, we report a throwaway end-to-end run of that path on a small model, together with the hosting and browser constraints that shape the later phases. Rather than relying on documentation alone, every runtime claim below comes from a command that was executed and observed.

### 1. Method

We built a **tiny Llama model** with the exact SinhalaLM configuration from Section 5.3 of the proposal (vocabulary 15,000, hidden size 512, intermediate size 1,365, 8 attention heads, 8 key-value heads, RoPE θ = 10000, machine-epsilon RMSNorm, tied embeddings, BOS 2, EOS 3, PAD 0), except that it has 2 layers instead of 8 and random weights (seed 0). This model then went through the same steps the real model will go through:

1. ONNX export with `optimum-cli export onnx --task text-generation-with-past`.
2. Logit and greedy-generation comparison between PyTorch and ONNX Runtime in Python.
3. Loading and greedy generation in `@huggingface/transformers` 4.3.2 under Node.js, with a custom streamer.
4. Quantization into fp16, q8 and q4, followed by loading each variant in transformers.js.

Since the weights are random, this run validates the mechanics of the pipeline, not the quality of any variant. Quality is measured on the real checkpoints in RQ1.

Alongside this, we read the current Hugging Face Spaces documentation and probed WebGPU availability in Chrome on the development machine.

### 2. Python export toolchain

The export path is constrained by a single package. `optimum-onnx` 0.1.0, its latest release, requires `optimum~=2.1.0` and `transformers>=4.36,<4.58`. As a result, the latest `transformers` release (5.22.1) and the latest `optimum` release (2.3.0) cannot be used for export, and the build-time pipeline is pinned to the newest compatible versions instead.

| Package | Version | Note |
|---|---|---|
| Python | 3.12 | |
| torch | 2.14.1 (CPU) | |
| transformers | 4.57.6 | upper bound set by `optimum-onnx` |
| optimum | 2.1.0 | required by `optimum-onnx` |
| optimum-onnx | 0.1.0 | |
| onnx | 1.23.2 | |
| onnxruntime | 1.31.0 | |
| onnxslim | 0.1.98 | |
| accelerate | 1.15.0 | required for weight deduplication, see Section 3 |
| onnx-ir | 1.1.0 | required by the ONNX Runtime 4-bit quantizer, see Section 5 |

This combination exported, ran and quantized the tiny model without errors. In other words, the proposal's risk "version incompatibility between transformers and the ONNX exporter" is real, and it is mitigated by these pins rather than by the latest releases.

### 3. ONNX export

The exported graph uses opset 18. Its inputs are `input_ids`, `attention_mask`, `position_ids` and one `past_key_values.N.key` and `past_key_values.N.value` pair per layer, each of shape `[batch_size, 8, past_sequence_length, 64]`. Its outputs are `logits` and the matching `present.N.key` and `present.N.value` tensors.

**Tied-weight deduplication.** The first export produced a 86.67 MB file. The exporter printed "Weight deduplication check in the ONNX export requires accelerate", and after installing `accelerate`, the same export produced a 55.95 MB file. The difference of 30.72 MB is exactly one fp32 copy of the 15,000 × 512 embedding table. This is because, without `accelerate`, the exporter stores the tied `lm_head` as a second copy of the embedding. In the context of our offline reader, this single missing dependency would have added about 30 MB to every download, which is why `accelerate` is part of the pinned toolchain.

**Parity.** On the prompt `[2, 100, 200, 300, 400]`, the maximum absolute logit difference between PyTorch and ONNX Runtime was 1.52e-6, and 10-token greedy generations were identical. This is well below the 1e-4 tolerance of gate 2, although the real gate runs on the converted checkpoints.

### 4. transformers.js 4.3.2 runtime

transformers.js 4.3.2 loaded the exported model from a local path with `env.allowRemoteModels = false`, and its greedy output was identical to the PyTorch output: `2652` six times, followed by `11109` four times.

**Streamer behavior.** The proposal notes that the streamer interface has changed between releases and should be checked against the pinned version. We observed the following in 4.3.2:

- `put()` receives a plain JavaScript array of arrays of `BigInt` values.
- The first `put()` call carries the full prompt (`[[2, 100, 200, 300, 400]]`).
- Every later call carries exactly one new token (for example, `[[2652]]`).
- `end()` is called once after generation.

For 10 new tokens, this made 12 calls in total: one for the prompt, ten for the tokens and one `end()`. This is the behavior the worker code in Section 5.7 of the proposal assumes, so that code works without changes.

**File naming.** transformers.js selects a file by appending a suffix for the requested `dtype` to `onnx/model`:

| `dtype` | File |
|---|---|
| `fp32` | `onnx/model.onnx` |
| `fp16` | `onnx/model_fp16.onnx` |
| `q8` | `onnx/model_quantized.onnx` |
| `q4` | `onnx/model_q4.onnx` |
| `q4f16` | `onnx/model_q4f16.onnx` |

**Version pin.** transformers.js 4.3.2 was released on 2026-10-28, which is less than two weeks before this check. Under the project's pinning rule, it becomes eligible on 2026-11-11, before the browser application work in Phase 4 begins. The pin is therefore made in Phase 4, after this check is repeated against the version that is current at that time.

### 5. Quantization

The proposal plans to quantize with the script in the transformers.js repository. That script no longer exists. The Python `scripts/convert.py` and `scripts/quantize.py` were removed when the repository was restructured for version 4 (commit `4e1b4b65`, 2026-01-29), and the last release that ships them is 3.8.1, pinned to `transformers` 4.49.0 and `onnxruntime` 1.20.1. The version 4 README instead points to `optimum-onnx` for conversion, and it describes how to load quantized files but not how to produce them.

Rather than depending on the old pins of a removed script, we quantized directly with ONNX Runtime's own tools:

- **fp16** — `onnxruntime.transformers.float16.convert_float_to_float16` with `keep_io_types=True`.
- **q8** — `onnxruntime.quantization.quantize_dynamic` with 8-bit unsigned weights.
- **q4** — `onnxruntime.quantization.matmul_nbits_quantizer.MatMulNBitsQuantizer` with a block size of 32 and symmetric quantization.

*Table: Tiny model (2 layers) after quantization.*

| Variant | Size | Operators of interest |
|---|---|---|
| fp32 | 55.95 MB | `Gather`, `MatMul` |
| fp16 | 28.01 MB | `Gather`, `MatMul` |
| q8 | 14.08 MB | `Gather`, `DequantizeLinear`, `DynamicQuantizeLinear`, `MatMulInteger` |
| q4 | 34.73 MB | `Gather`, `MatMul`, `MatMulNBits` |

All four variants loaded in transformers.js and produced the same greedy output as fp32 on this prompt.

The interesting result is the q4 variant, which is larger than q8. The 4-bit quantizer logged "MatMul doesn't have const weight. Skip to quantize" for `/lm_head/MatMul`. This is because, after deduplication, the output head reads the shared embedding through a `Transpose` node, so the quantizer does not see a constant weight to quantize. As a result, both the embedding lookup and the output head stay in fp32 in q4, while q8 quantizes the shared table as well.

In other words, the proposal's warning that "the embedding table stays fp32" is not just a possibility for q4, it is the default outcome with tied weights. For the real model, the embedding is 23% of all parameters, so this directly affects the q4 download size. Two candidate fixes are evaluated in RQ1 rather than decided here: storing a transposed constant copy of the head before 4-bit quantization, and quantizing the embedding lookup itself. Each candidate must also be checked for operator support in the WebGPU and WASM backends.

### 6. Hosting on a Hugging Face static Space

The current Spaces documentation confirms the following:

- **Headers.** The README front matter accepts a `custom_headers` key. Only `cross-origin-embedder-policy`, `cross-origin-opener-policy` and `cross-origin-resource-policy` are allowed, and all header names and values must be lowercase. This is enough for cross-origin isolation and, as a result, for multi-threaded WASM.
- **Build.** A static Space can run `app_build_command` in a separate job and serve the result from `refs/convert/build`, with `app_file` pointing to the built `index.html`. Without a build command, the Space serves the files as they are in the repository.
- **Large files.** Files larger than 10 MB should be tracked with `git-xet`. The documentation does not state a repository size limit for Spaces.

Two questions are not answered by the documentation: whether a page is cross-origin isolated when it is viewed inside the iframe on `huggingface.co/spaces/<owner>/<name>` rather than on its direct `<owner>-<name>.hf.space` URL, and how service workers behave in either case. Since the iframe page is served by huggingface.co itself, an offline reader most likely needs the direct URL. These questions are verified on a live Space in Phase 5.

For deployment, we upload the application already built by the local pipeline rather than using `app_build_command`. This keeps the deployed files identical to the files that passed `make ci`.

### 7. WebGPU on the development machine

The development machine runs Linux with Google Chrome 154.0.8037.97 and Mozilla Firefox 158.0.1. In headless Chrome, `navigator.gpu` was present but returned no adapter. With `--enable-unsafe-webgpu --enable-features=Vulkan`, it returned only a software adapter with the architecture `swiftshader`.

This has two consequences. First, it confirms the proposal's decision to run end-to-end tests on WASM. Second, backend selection must not treat any adapter as WebGPU: a software adapter is most likely slower than multi-threaded WASM, so the worker should fall back to WASM when the adapter is a fallback or software adapter. Hardware WebGPU in headed Chrome is checked during Phase 4 and the RQ2 benchmarks.

### 8. Implications for the plan

- The build-time pipeline pins the versions in Section 2, including `accelerate` and `onnx-ir`.
- `convert/quantize.sh` uses ONNX Runtime's quantization tools directly, since the transformers.js script was removed.
- RQ1 includes the q4 tied-weight problem and its two candidate fixes.
- The transformers.js version is pinned at the start of Phase 4, after this check is repeated.
- The inference worker rejects software WebGPU adapters.
- The hosting checks that need a live Space move to Phase 5, and the offline reader is expected to use the direct `.hf.space` URL.

Throughout this check, the planned path held: a Llama model with the SinhalaLM configuration exports, runs and quantizes with current tools, and transformers.js reproduces the PyTorch output exactly. The changes it forces are not in the architecture of the system, but in the details of the toolchain, which is exactly where they are cheapest to discover.
