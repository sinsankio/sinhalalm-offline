# Repository Inspection: SinhalaLLM-1.7B

A model that loads without errors is not necessarily the model its author published. Since SinhalaLLM-1.7B was saved with a newer `transformers` release than the one the ONNX exporter requires, we inspected the repository before writing any pipeline code, to find the places where the two could disagree without an error message.

In this document, we report the findings of that inspection (S1–S8). All findings refer to `namindu94/SinhalaLLM-1.7B` at revision `b24d58b7929e1ed50f683b435a9a629a73e27af5`.

### 1. Configuration

**S1 — The export toolchain loads the wrong RoPE base.** The repository's `config.json` was written by `transformers` 5.19.0, which stores the RoPE base inside a nested field, `rope_parameters: {rope_theta: 130000, rope_type: "default"}`. Loading the same file with `transformers` 4.57.6, the newest version that `optimum-onnx` 0.1.0 accepts, returns `rope_theta` 10000, since that version does not read the nested field and falls back to its default. The GGUF file in the same repository states `llama.rope.freq_base` 130000, which confirms that 130000 is the intended value.

In other words, the export toolchain would build a model that loads, runs and produces Sinhala text, but with positions encoded differently from training. This is the most dangerous kind of error for the project, since nothing fails. The pipeline therefore writes a translated configuration for the export toolchain, and gate 1 compares its logits against `transformers` 5.19.0, with the untranslated configuration as a negative control that must fail.

**S5 — Configured and trained context differ.** `max_position_embeddings` is 8,192, while the model card states a trained context of 2,048 tokens. The application caps prompt plus generation at 1,024 tokens, which stays within the trained context and bounds the key-value cache to about 403 MB in fp32 (24 layers × 2 × 2,048 values × 4 bytes per token).

**S6 — Stop tokens and caching.** `generation_config.json` lists `eos_token_id` as `[2, 0]`, which are `<|im_end|>` and `<|endoftext|>`, while `config.json` lists only 0 and sets `use_cache` to false. Generation stops at either token, and the translated configuration enables the key-value cache for export.

### 2. Tokenizer

**S2 — The tokenizer expects the ZWJ but does not add it.** The model card describes a normalizer that restored the ZWJ in common conjunct clusters before training. However, `tokenizer.json` defines no normalizer, so the repair is not applied at inference time. As a result, ශ්‍රී ලංකාව ප්‍රධාන encodes as 3 tokens, while the same text without the ZWJ, ශ්රී ලංකාව ප්රධාන, encodes as 6 fragments. In the context of the offline reader, a question typed on a keyboard that omits the ZWJ reaches the model in a form it rarely saw during training.

**S3 — The ZWJ appears in exactly 15 clusters.** The card mentions 16 conjunct clusters but does not list them. Of the 765 vocabulary tokens that contain a ZWJ, 611 contain rakaransaya (`්‍ර`) and 150 contain yansaya (`්‍ය`), in 15 distinct clusters, and only four contain the ZWJ in any other position. None of these 15 clusters appears in the vocabulary without the ZWJ, while other clusters, such as ව්ය and ත්ය, appear only without it. The input normalization in the application follows exactly this split, which is analyzed in [zwj-normalization.md](zwj-normalization.md).

**S4 — The browser tokenizer matches the Python tokenizer.** The pre-tokenizer extends the GPT-2 pattern with the Unicode classes `\p{L}` and `\p{M}` and the zero-width characters, which a JavaScript implementation must reproduce exactly. transformers.js 4.3.2 produced the same token IDs as the Python `tokenizers` library on 210 inputs: 200 Sinhala texts from the test split of `ChamaraVishwajithRajapaksha/sinhala-question-answer-pairs` and 10 edge cases, including a byte order mark, U+0085, the zero-width non-joiner and ChatML special tokens. The ZWJ also survived decoding. This removes the need for a custom tokenizer port, and gate 3 repeats the check on 10,000 sentences.

### 3. Size and quality

**S7 — The model is large for a browser.** The configuration gives 1,740,474,368 parameters, of which the tied 63,360 × 2,048 embedding is 7.5%. The published `model.safetensors` is 3.48 GB in bfloat16, and the model is about 6.96 GB in fp32. Since this most likely exceeds what the WASM backend can address, only quantized variants are shipped, and gate 4 compares the same quantized file in Python and in the browser.

**S8 — Factual accuracy is limited.** In the comparison on ten questions, several answers did not match the reference answers, and two long answers became repetitive. The model card also states that the model often gets facts wrong from memory and is weak at multi-step arithmetic and reasoning.

Throughout this inspection, the architecture itself raised no issue, since the model is already a standard `LlamaForCausalLM`. The risks sit in the configuration and in the handling of the ZWJ, which is exactly where the parity gates and the text processing of the application are placed.
