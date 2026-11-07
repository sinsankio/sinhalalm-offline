# Findings

This document summarizes the research and development findings of the project in one place. Each finding is described in detail in its own document under [docs/findings/](docs/findings/), and this summary links to it.

### Index

| Document | Topic | Summary |
|---|---|---|
| [model-selection.md](docs/findings/model-selection.md) | Switch from SinhalaLM to SinhalaLLM-1.7B | SinhalaLLM-1.7B is the only screened model whose answers address the questions, and its tokenizer needs 1.76 tokens per Sinhala word. |
| [sinhalallm-inspection.md](docs/findings/sinhalallm-inspection.md) | Inspection of the SinhalaLLM-1.7B repository | The export toolchain reads the RoPE base as 10000 instead of 130000, the tokenizer expects a ZWJ it does not add, and the browser tokenizer matches the Python one. |
| [browser-runtime.md](docs/findings/browser-runtime.md) | Browser runtime compatibility and speed of the q4 model | The browser build of ONNX Runtime lacks the 4-bit embedding operator, two operators stalled WebGPU at every token until they were rewritten, and WebAssembly has no fast 4-bit kernel. |
| [rq1-quantization.md](docs/findings/rq1-quantization.md) | RQ1: quantization quality | fp16 and q8 stay within 2% of fp32, while every 4-bit variant fails; keeping the value projection of layer 0 in fp32 reduces the 4-bit loss from +164% to +33%, better than the author's own 4-bit release. |
| [quantization.md](docs/findings/quantization.md) | fp16, q8 and q4 variants | Only q4 with an untied, quantized head and a quantized embedding fits the 1.2 GB target (1.17 GB), but a first check suggests it costs quality, while q8 is accurate but very slow on the CPU. |
| [onnx-export.md](docs/findings/onnx-export.md) | ONNX export and parity gate 2 | The export fits the machine only as a capped trace plus a separate verify-and-finalize step, and the fp32 ONNX model reproduces every greedy answer with a relative logit difference of at most 3.46e-5. |
| [zwj-normalization.md](docs/findings/zwj-normalization.md) | Input normalization of the ZWJ | The model learned exactly 15 clusters with the ZWJ and all others without it, so normalizing the input to that split reduces tokens by 1.6% on the corpus and by 10.4% on text typed without the ZWJ, while a blanket rule increases them. |
| [tokenizer-parity.md](docs/findings/tokenizer-parity.md) | Parity gate 3 | transformers.js reproduces the Python tokenizer on 10,000 Sinhala sentences and 17 edge cases, once the chat template is embedded for the browser. |
| [config-translation.md](docs/findings/config-translation.md) | Parity gate 1 for SinhalaLLM-1.7B | The translated configuration loads the model in `transformers` 4.57.6 identically to 5.19.0, while the untranslated one differs by up to 11.66 in the logits. |
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

### Browser runtime

- **S10: the 4-bit embedding operator is missing in the browser.** `GatherBlockQuantized` works in ONNX Runtime for Python and Node.js, so every earlier gate passed, while the asyncify build that transformers.js loads in the browser cannot create a session with it.
- **Standard operators give the same model.** The pipeline replaces it with `Gather`, `BitwiseAnd`, `BitShift` and a dequantization with the same scales. The logits are identical, all ten greedy answers match, the quality gate is unchanged at a perplexity of 158.014 and the size stays 1.184 GB.
- **CI now checks the browser build.** `make check-package` creates a session in the asyncify WebAssembly build and checks the first greedy token of each golden question.
- **S11: two operators stalled WebGPU at every token.** `IsNaN` after each softmax and the bitwise unpack of the embedding ran on the CPU, so the GPU waited for a copy 25 times per token. Exact rewrites raised the speed of a plain session on an RTX 3060 Laptop GPU from 4.1 to about 18 tokens per second.
- **S12: WebAssembly has no fast 4-bit kernel.** Without a GPU, 94% of the time goes to the 4-bit matrix product, and the model generates 0.6 tokens per second with seven threads, against a target of 5.

### RQ1: quantization quality

- **fp16 and q8 pass the 2% gate.** On identical FLORES+ sentences, fp16 matches fp32 (+0.0%) and q8 loses 1.2% in perplexity, at 3.68 GB and 2.48 GB.
- **Every plain 4-bit variant fails by far.** Perplexity rises by +164% to +248% across head and embedding precision, asymmetric quantization, block size 16 and HQQ.
- **The cause is not the pipeline.** The 4-bit kernel's compute precision and the dequantized weight error both behave as expected, and the author's own mixed 4/6-bit GGUF release loses +51.5% on the same sentences.
- **One matrix causes most of the loss.** Keeping the 16 MB value projection of layer 0 in fp32 reduces the loss to +33% (q4-v0, 1.54 GB), or +40% with a 4-bit head and embedding (q4-v0-small, 1.18 GB, within the 1.2 GB target).
- **q4-v0-small ships by default.** The 2% gate was replaced by "no worse than the author's Q4_K_M release on the same sentences, plus a native-speaker review of the answers". On all 1,012 sentences, q4-v0-small passes with a perplexity of 158.0 against 195.0 for Q4_K_M. q8 stays an option if it proves fast enough on WebGPU.

### Quantization

- **Only one variant fits the size target.** q4 is 1.17 GB once the output head is untied and the embedding is quantized to 4 bits, while the default 4-bit quantizer gives 1.53 GB, q8 2.48 GB and fp16 3.68 GB.
- **The planned tools did not fit into memory.** ONNX Runtime's fp16 converter and dynamic 8-bit quantizer were killed at 12 GB, so fp16 is exported directly from PyTorch in fp16 and q8 uses the 8-bit block quantizer.
- **First signals on one question.** fp16 and q8 reproduce the fp32 answer, q4-tied keeps the answer correct, and q4 falls into a loop. q8 generates at 0.1 tokens per second on the CPU. RQ1 measures these effects over the full golden set.

### ONNX export (gate 2)

- **The export is split to fit into memory.** A memory-capped child traces the model and writes one file per tensor, and a separate process verifies every file by size, ties the duplicated output head back to the embedding and writes one 6.96 GB data file, exactly 4 bytes per parameter.
- **The fp32 ONNX model is faithful.** Its greedy answers match the reference on all ten golden questions, the top-1 token matches at every position, and the largest logit difference is 3.46e-5 of each position's largest logit.
- **Gate 2 uses a relative tolerance.** The absolute tolerance of 1e-4 failed at 7.6e-4, always at position 0, and diagnostics attributed this to rounding differences between the fp32 kernels of ONNX Runtime and PyTorch rather than to the export.

### ZWJ normalization

- **The vocabulary defines 15 ZWJ clusters.** Rakaransaya in ප්‍ර, ත්‍ර, ක්‍ර, ග්‍ර, ශ්‍ර, ද්‍ර and ඝ්‍ර, and yansaya in ධ්‍ය, ද්‍ය, ශ්‍ය, ජ්‍ය, ඛ්‍ය, භ්‍ය, ට්‍ය and ෂ්‍ය never appear without the ZWJ, while other clusters, such as ව්ය and ත්ය, appear only without it.
- **A blanket rule is counterproductive.** Adding the ZWJ after every virama before ර or ය increased the corpus by 0.26% in tokens.
- **Normalizing to the learned split is best.** Adding the ZWJ in the 15 clusters and removing it elsewhere reduced the corpus by 1.57%, and text typed without the ZWJ by 10.4%. The application uses this rule as `normalizeZwj`.
- **The output may need the opposite step.** The model most likely writes clusters such as සත්ය without the ZWJ, so restoring it in the output is a candidate for RQ3.

### Tokenizer parity (gate 3)

- **The browser tokenizer is identical.** transformers.js 4.3.1 produced the same token IDs as the Python tokenizer on 10,000 Sinhala sentences from four pinned sources and on 17 edge cases, including ZWJ, ZWNJ, a byte order mark and whitespace characters that Python and JavaScript treat differently.
- **S9: the chat template must be embedded for the browser.** The model stores its template in `chat_template.jinja`, which `AutoTokenizer` in transformers.js does not read. With the template embedded in `tokenizer_config.json`, all ten chat prompts matched Python as text and as token IDs.
- **The corpus is reproducible without being committed.** A manifest records the source revisions, the seed and the corpus hash, and the test fails if the corpus on disk does not match it.

### Configuration translation (gate 1)

- **The translated load is identical to the reference.** With the RoPE base translated to 130,000, `transformers` 4.57.6 matches 5.19.0 with a maximum absolute logit difference of 0.0 on ten golden questions and a random 1,024-token input, and all ten 64-token greedy generations are identical.
- **The negative control fails as intended.** The untranslated RoPE base of 10,000 differs from the reference by up to 11.66 in the logits.
- **The error is silent in short answers.** With the wrong RoPE base, the answer to Q2 is identical and Q1 differs by one word, while the longer answer to Q3 falls into repetition.
- **The golden outputs are reproducible.** Regenerating them from scratch matched the committed files exactly.

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
