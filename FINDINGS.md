# Findings

This document summarizes the research and development findings of the project in one place. Each finding is described in detail in its own document under [docs/findings/](docs/findings/), and this summary links to it.

### Index

| Document | Topic | Summary |
|---|---|---|
| [model-selection.md](docs/findings/model-selection.md) | Switch from SinhalaLM to SinhalaLLM-1.7B | SinhalaLLM-1.7B is the only screened model whose answers address the questions, and its tokenizer needs 1.76 tokens per Sinhala word. |
| [sinhalallm-inspection.md](docs/findings/sinhalallm-inspection.md) | Inspection of the SinhalaLLM-1.7B repository | The export toolchain reads the RoPE base as 10000 instead of 130000, the tokenizer expects a ZWJ it does not add, and the browser tokenizer matches the Python one. |
| [toolchain.md](docs/findings/toolchain.md) | Export, runtime, quantization and hosting constraints | The planned Llama export path works with current tools, but only with pinned versions, and q4 leaves the tied embedding and head in fp32. |
| [repository-audit.md](docs/findings/repository-audit.md) | Audit of the two HelaAI repositories (previous model) | All nine original findings reproduce, F4 is corrected to two template `<UNK>` tokens, and three new findings are added. |
| [llama-equivalence.md](docs/findings/llama-equivalence.md) | Parity gate 1 for the HelaAI models (previous model) | Both checkpoints convert to `LlamaForCausalLM` with a logit difference of exactly 0.0 and identical greedy output. |

### Model selection

- **The HelaAI models were replaced.** Their answers on the golden prompts did not reach a satisfactory question-answering quality, although their Llama conversion was exact.
- **SinhalaLLM-1.7B was selected.** On ten Sinhala questions, it was the only screened model whose answers addressed the questions, while Qwen3.5-2B and Qwen3.5-0.8B were mostly wrong and ran to the token limit.
- **Its tokenizer is the most efficient for Sinhala.** It needs 1.76 tokens per word, against 2.66 for SinhalaLM, 3.16 for Gemma 4 and 4.92 for Qwen3.5, and keeps the ZWJ.
- **The scope changed with the model.** The application provides question answering only, and the targets were raised to at most 1.2 GB of download and 2 GB of tab memory.

### SinhalaLLM-1.7B inspection

- **S1: the RoPE base is read differently by two `transformers` versions.** Version 5.19.0 stores it as `rope_parameters`, which version 4.57.6 ignores, so the export toolchain would load 10000 instead of 130000 without any error.
- **S2 and S3: the ZWJ must be added to the input.** The tokenizer has no normalizer, ශ්රී ලංකාව ප්රධාන without the ZWJ becomes 6 fragments instead of 3 tokens, and the vocabulary contains the ZWJ only in `්‍ර` (611 tokens) and `්‍ය` (150 tokens).
- **S4: the browser tokenizer matches the Python one.** transformers.js 4.3.2 produced identical token IDs on 210 Sinhala texts and edge cases, so no custom tokenizer port is needed.
- **S5–S8: context, stop tokens, size and accuracy.** The application caps the context at 1,024 tokens, stops at both `<|im_end|>` and `<|endoftext|>`, ships only quantized variants of the 6.96 GB fp32 model, and labels answers as unverified.

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

### Repository audit (previous model)

The nine audit findings of the proposal (F1–F9) were reproduced against the HelaAI repositories at pinned revisions.

- **All nine findings hold.** This includes the 32,850,432 unique parameters (F1), the unloadable `tokenizer.json` with 5,814 of 14,811 broken merges (F3) and the 512-token context length (F7).
- **F4 is corrected.** The instruct template alone encodes as 24 tokens with two `<UNK>`. The proposal's 37 tokens with three `<UNK>` include the model card's example question, whose ZWJ adds the third `<UNK>`.
- **F6 holds on new prompts.** All 8 `<UNK>` tokens in the eight golden instruct generations follow a token ending in the virama, while the base model generates none.
- **F10: the instruct `modeling_sinhalalm.py` cannot be imported.** It loads a local checkpoint with `torch.load` at import time, so the pipeline executes only the required definitions from the hash-verified file.
- **F11: two fine-tuning datasets are named.** The instruct config names the ELI5 dataset, while the training code names the question-answer pairs dataset.
- **F12: ELI5 HTML entities appear in outputs.** `&quot;` is common in the ELI5 answers, absent in the question-answer pairs, and reproduced literally by the instruct model. The application therefore decodes HTML entities in model outputs.

### Llama equivalence, gate 1 (previous model)

- **Both checkpoints pass.** The maximum absolute logit difference is 0.0 on every golden prompt and on a random 512-token input, and all 40-token greedy generations with key-value caching are identical.
- **The converted model is the expected size.** It has 32,850,432 parameters, and the fp32 file is 131,409,848 bytes, which matches the proposal's 131.4 MB estimate.
