# RQ1: Quantization Quality of SinhalaLLM-1.7B

RQ1 asks how perplexity on held-out Sinhala text changes across fp32, fp16, q8 and q4, and which is the smallest variant whose quality stays within an acceptable margin of fp32. The proposal set that margin at 2% of perplexity. In this document, we report the measurements, the investigation of a much larger 4-bit loss than expected, and the candidate variants that came out of it.

### 1. Method

- **Held-out text** — the 1,012 sentences of the FLORES+ `sin_Sinh` devtest split (`openlanguagedata/flores_plus`, revision `1147e2dc`, file hash recorded in `eval_quality.py`), 131,080 characters in total. Each sentence is scored on its own after the `<|endoftext|>` token, without ZWJ normalization.
- **Metrics** — perplexity per token and bits per character, together with the top-1 agreement of the prompt logits and the greedy agreement of 64-token answers on the ten golden questions, both against the fp32 reference.
- **Runtime** — ONNX Runtime 1.31.0 on the CPU for every variant. Per-sentence scores are stored, so that variants evaluated on a subset are compared with fp32 on exactly the same sentences.
- **Slow variants** — fp16 was evaluated on the first 200 sentences and 3 questions, and q8 on the first 50 sentences and 1 question, since q8 generates at 0.26 tokens per second on the CPU.

### 2. First results

| Variant | Size | Perplexity change against fp32 | Greedy answers equal to fp32 |
|---|---|---|---|
| fp16 | 3.68 GB | +0.0% (200 sentences) | 3 of 3 |
| q8 | 2.48 GB | +1.2% (50 sentences) | 1 of 1 |
| q4-tied (default 4-bit quantizer) | 1.53 GB | +164% | 0 of 10 |
| q4 (head and embedding also 4-bit) | 1.17 GB | +187% | 0 of 10 |

The fp32 model reached a perplexity of 113.2 (1.695 bits per character) on the full devtest split. fp16 and q8 stay within the 2% gate, while both 4-bit variants lose far more than expected for a 4-bit model.

### 3. Searching for the cause

**Head and embedding.** An ablation over the precision of the output head and the embedding (both fp32, head 4-bit, embedding 4-bit, both 4-bit) changed perplexity only between +164% and +187%. In other words, the head and the embedding are not the main cause.

**Quantization method.** Three alternatives to symmetric round-to-nearest with block size 32 were screened on 200 sentences: asymmetric quantization (+220%), block size 16 (+248%) and the data-free HQQ method (+197%). None of them helped, and the smaller block size was even worse.

**Runtime kernel.** Setting the compute precision of the 4-bit matrix multiplication explicitly to fp32 (`accuracy_level` 1) gave exactly the same perplexity as the default, so the loss does not come from how ONNX Runtime computes with the 4-bit weights.

**Weight error.** Dequantizing every 4-bit matrix showed a mean relative error of 8.7% at block size 32 and 7.7% at block size 16, both lower than a reference round-to-nearest implementation (9.9%). The quantizer therefore works as intended, and a smaller weight error does not translate into a lower perplexity for this model.

**Sensitive weights.** Keeping one group of weight matrices in fp32 at a time and measuring on 50 sentences located the cause:

| Kept in fp32 | Size | Perplexity (fp32: 109.7) |
|---|---|---|
| nothing | 1.53 GB | 279.2 |
| all down projections | 2.89 GB | 286.1 |
| all up and gate projections | 4.25 GB | 382.6 |
| all query and key projections | 2.21 GB | 273.9 |
| all output projections | 1.87 GB | 268.4 |
| all value projections | 1.87 GB | 134.3 |
| layers 0 and 23 | 1.98 GB | 135.1 |
| layer 23 | 1.75 GB | 271.4 |
| layer 0 | 1.75 GB | 137.6 |
| value projection of layer 23 | 1.54 GB | 276.3 |
| **value projection of layer 0** | **1.54 GB** | **143.2** |

A single 2048 × 2048 matrix, the value projection of the first layer, accounts for most of the 4-bit loss. Keeping it in fp32 costs 16 MB and reduces the perplexity from 279.2 to 143.2. The remaining gap is spread across the model: even all value projections and the whole first layer in fp32 leave about +22%.

**Reference point.** The model's author publishes a 4-bit GGUF file made with llama.cpp's Q4_K_M method, which mixes 4-bit and 6-bit tensors. Dequantized and evaluated on the same 50 sentences, it reached a perplexity of 166.2, which is +51.5% against fp32. The 4-bit sensitivity is therefore a property of the model, not of this pipeline.

### 4. Candidate variants

| Variant | Size | Perplexity, 1,012 sentences | Change | Change, 50 sentences | Generation on the CPU |
|---|---|---|---|---|---|
| fp32 | 6.96 GB | 113.2 | — | — | 3.6 tokens/s |
| q4-v0 | 1.54 GB | 150.2 | +32.7% | +30.5% | 11.4 tokens/s |
| q4-v0-small | 1.18 GB | 158.0 | +39.6% | +37.7% | 15.0 tokens/s |
| author's Q4_K_M (GGUF) | 1.08 GB | — | — | +51.5% | — |

Both candidates keep the value projection of layer 0 in fp32. q4-v0 keeps the output head tied to the fp32 embedding, while q4-v0-small also quantizes the head and the embedding to 4 bits and fits the 1.2 GB target. Both are better than the author's own 4-bit release on the same sentences. Their answers to the ten golden questions are listed next to the fp32 answers in [quantization-comparison.md](../quantization-comparison.md).

### 5. Answer to RQ1

Within the 2% gate, the smallest variant is q8 at 2.48 GB, followed by fp16 at 3.68 GB. No 4-bit variant meets the gate, and the evidence points to an inherent sensitivity of this model to 4-bit weights rather than to the quantization method: the author's own mixed 4/6-bit release loses even more.

The best 4-bit result comes from keeping a single, especially sensitive weight matrix in full precision, which reduces the loss from +164% to +33% at almost no cost in size. `[[NEEDS INPUT: which variant to ship by default, and the quality gate that replaces the 2% margin]]`
