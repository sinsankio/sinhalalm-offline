# Quantization: Variants, Sizes and First Signals

A 1.74B model only fits the 1.2 GB download target of the offline reader if most of its weights are stored in 4 bits. In this document, we report how the fp16, q8 and q4 variants of SinhalaLLM-1.7B were produced from the verified fp32 ONNX model, how large they are, and what a first smoke test suggests about their quality. The full quality evaluation follows in RQ1.

### 1. Methods

Each variant is written as a complete transformers.js model folder in the model store: `onnx/model_<suffix>.onnx`, the model configuration, and the browser version of the tokenizer files with the embedded chat template.

| Variant | Method | File |
|---|---|---|
| fp16 | Exported directly from the PyTorch model loaded in fp16, with the same trace, validation and tie steps as the fp32 export | `model_fp16.onnx` |
| q8 | MatMulNBits, 8 bits, block size 32, symmetric, with the output head untied into its own weight | `model_quantized.onnx` |
| q4 | MatMulNBits, 4 bits, block size 32, symmetric, with the output head untied and the embedding lookup quantized to 4 bits (GatherBlockQuantized) | `model_q4.onnx` |
| q4-tied | MatMulNBits, 4 bits, block size 32, symmetric, with the default quantizer settings | `model_q4.onnx` |

Two of these methods differ from the plan, since the planned tools did not fit into the available memory. ONNX Runtime's fp16 converter and its dynamic 8-bit quantizer both hold the full fp32 model and a converted copy at the same time, and both were killed at a 12 GB memory cap. The 4-bit and 8-bit block quantizer peaked at about 10.9 GB and 12.6 GB and fits under a 14 GB cap, and the fp16 export from PyTorch fits under the 11 GB cap of the export.

### 2. Sizes

| Variant | Size | Linear layers | Embedding | Output head |
|---|---|---|---|---|
| fp32 | 6.96 GB | fp32 | fp32 | tied to the embedding |
| fp16 | 3.68 GB | fp16, except one 2048 × 2048 projection per layer in fp32 | fp16 | tied to the embedding |
| q8 | 2.48 GB | 8-bit (168) | fp32 | 8-bit, untied |
| q4 | 1.17 GB | 4-bit (168) | 4-bit | 4-bit, untied |
| q4-tied | 1.53 GB | 4-bit (168) | fp32 | fp32, tied to the embedding |

Three observations follow from the table:

- **Only q4 fits the 1.2 GB target.** It needs the head to be untied and the embedding to be quantized; with the default settings, q4 is 1.53 GB, which confirms the tied-weight finding of the toolchain check.
- **Block quantization costs more than the bit width suggests.** With one fp32 scale per block of 32 weights, 4-bit weights take 0.625 bytes per parameter and 8-bit weights 1.125 bytes, which explains why q4 is 1.17 GB rather than the 0.87 GB lower bound.
- **The 8-bit quantizer did not quantize the embedding.** The embedding lookup was quantized only at 4 bits, so q8 keeps the 519 MB embedding table in fp32.

The fp16 export contains 100.7M fp32 parameters, exactly one 2048 × 2048 matrix-multiplication weight per layer. This is most likely a projection whose input was upcast to fp32 during the fp16 trace and whose weight was then folded into an fp32 constant. It adds about 0.2 GB and keeps that projection in full precision.

### 3. First signals

As a smoke test, each variant answered the first golden question in ONNX Runtime on the CPU with greedy decoding, and its prompt logits were compared with the fp32 reference.

| Variant | Top-1 agreement on the prompt | Greedy answer equal to fp32 | Speed |
|---|---|---|---|
| fp16 | 100% | yes | 1.4 tokens/s |
| q8 | 93% | yes | 0.1 tokens/s |
| q4 | 79% | no, the answer loops | 6.2 tokens/s |
| q4-tied | 71% | no, but the answer stays correct | 4.4 tokens/s |

With q4, the answer to ශ්‍රී ලංකාවේ අගනුවර කුමක්ද? repeats "ශ්‍රී ලංකාවේ අගනුවර වන්නේ" until the token limit, while q4-tied answers with the correct capital and a repeated name in brackets. Since the two differ only in the head and the embedding, this most likely means that 4-bit quantization of the output head, the embedding or both costs noticeable quality. The 8-bit variant is accurate but impractically slow on the CPU, which suggests that ONNX Runtime has no optimized CPU kernel for 8-bit MatMulNBits.

These are observations on a single question, not a quality measurement. RQ1 measures perplexity and greedy agreement over the full golden set, and isolates the effect of the head and the embedding.

### 4. Implications

- The size target and the quality target pull in different directions at 4 bits. RQ1 compares head and embedding choices, and the 1.2 GB target may need to be raised if no 4-bit variant keeps the quality.
- q8 is not a practical fallback for the WASM backend at its current speed, and its browser speed is measured in RQ2.
- The fp16 variant is mainly relevant for WebGPU, which runs fp16 natively.
