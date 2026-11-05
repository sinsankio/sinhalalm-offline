# Llama Equivalence: Parity Gate 1

The proposal rests on a single architectural claim: SinhalaGPT is mathematically equivalent to Llama, so that the standard Llama export path and runtime can replace a custom one. If the claim holds, the browser system needs no custom model code at all. If it fails, every later step inherits the error.

In this document, we report parity gate 1, which tests that claim on both checkpoints with the conversion in `convert/parity_check.py`.

### 1. Conversion

The conversion maps each SinhalaGPT tensor to its `LlamaForCausalLM` equivalent:

- **Attention** — the fused `qkv` weight of shape 1,536 × 512 is split into `q_proj` (rows 0–511), `k_proj` (rows 512–1,023) and `v_proj` (rows 1,024–1,535), and `proj` becomes `o_proj`. No head permutation is needed, since both implementations use the rotate-half form of RoPE.
- **Feed-forward** — `w1`, `w2` and `w3` become `gate_proj`, `up_proj` and `down_proj`, since SinhalaGPT computes `w3(silu(w1 x) * w2 x)`.
- **Normalization** — `ln1`, `ln2` and `ln_f` become `input_layernorm`, `post_attention_layernorm` and `model.norm`.
- **Dropped tensors** — `lm_head` and the RoPE `cos` and `sin` tables are dropped, after the conversion checks that `lm_head` is identical to `tok_emb` and that every stored table equals the recomputed one.

The `LlamaConfig` follows Section 5.3 of the proposal. Two values matter most: `rms_norm_eps` is the float32 machine epsilon (1.1920928955078125e-07), since SinhalaGPT uses `nn.RMSNorm` without an explicit epsilon, and `intermediate_size` is 1,365, which comes from `int(512 * 8 / 3)`.

### 2. Gate criteria

The gate passes for a checkpoint when all of the following hold:

- the maximum absolute logit difference between SinhalaGPT and Llama is below 1e-4 on every golden prompt and on a random 512-token input (seed 0);
- 40-token greedy generations from Llama, with key-value caching, are identical to the golden SinhalaGPT generations, which recompute the full sequence at every step;
- the converted model has exactly 32,850,432 parameters;
- the saved model, reloaded from disk, gives the same logits.

### 3. Results

| Checkpoint | Max \|logit diff\| | Greedy mismatches | Parameters | Saved model diff | Result |
|---|---|---|---|---|---|
| Base | 0.0 | 0 of 8 | 32,850,432 | 0.0 | Pass |
| Instruct | 0.0 | 0 of 8 | 32,850,432 | 0.0 | Pass |

The logit difference is exactly zero on every case, including the random 512-token input, which covers the full context length. This is most likely because, once the tensors are mapped, both implementations perform the same floating-point operations in the same order on CPU. Each converted fp32 `model.safetensors` is 131,409,848 bytes, which matches the 131.4 MB estimate in Section 5.5 of the proposal.

### 4. Implications

In other words, the converted model is not an approximation of SinhalaGPT, it is the same function in a different format. This confirms F5 of the proposal, and it means that the remaining parity gates measure only what their own step adds: the ONNX export in gate 2, the tokenizer port in gate 3 and the browser runtime in gate 4.
