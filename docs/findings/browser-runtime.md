# Browser runtime compatibility of the q4 model

The first browser test of the packaged q4-v0-small model failed while the session was created, before any token was generated. In this document, we report the cause (S10), the replacement that makes the model load in the browser runtime, and the checks that show the replacement does not change the model. The first answers in the browser then showed two speed problems, which we report as S11 (WebGPU) and S12 (WebAssembly), and the first manual test showed that Chrome on Linux does not offer hardware WebGPU by default (S13).

### 1. The failure

transformers.js 4.3.1 runs the model through the asyncify build of ONNX Runtime Web. In this build, session creation failed with:

```
Could not find an implementation for GatherBlockQuantized(1) node with name '/model/embed_tokens/Gather_Q4'
```

**S10 — The 4-bit embedding lookup is not available in the browser runtime.** The 4-bit quantizer replaces the embedding lookup with `GatherBlockQuantized`, a contrib operator of ONNX Runtime. ONNX Runtime in Python and the Node.js build used by the package parity check both implement it, so every earlier gate passed, while the browser build does not include it. The same model therefore worked everywhere except in the browser.

### 2. The replacement

After quantization, the pipeline replaces `GatherBlockQuantized` with standard ONNX operators that compute the same values:

1. The 4-bit table is stored as `uint8`, two codes per byte. The signed codes are stored with an offset of 8, so that each code is an unsigned value between 0 and 15.
2. `Gather` selects the rows for the input tokens.
3. `BitwiseAnd` and `BitShift` split each byte into its low and high code, and `Unsqueeze`, `Concat` and `Reshape` restore the original order of the 2,048 values.
4. `Cast` and `Sub` convert the codes to floats and remove the offset, and the per-block scales, gathered for the same rows, are applied with `Mul`.

The table keeps its size, since the codes and the scales are stored exactly as before. The report of each quantized variant records the replacement under `portable`. A later revision unpacks the codes with `Cast`, `Div`, `Mul` and `Sub` on 32-bit integers instead of `BitwiseAnd` and `BitShift`, for the reason given in S11.

### 3. Checks

| Check | Result |
|---|---|
| Logits against the model with `GatherBlockQuantized`, ONNX Runtime in Python | maximum difference 0.0 |
| Greedy answers on the ten golden questions | 10 of 10 identical |
| Quality gate (FLORES+, 1,012 sentences) | perplexity 158.014, unchanged, against 194.965 for the author's Q4_K_M |
| Size of q4-v0-small | 1,184,130,557 bytes, unchanged within 28 bytes of graph metadata |
| Session in the asyncify WebAssembly build, Node.js | loads, and predicts the first greedy token of all ten golden questions |

The last check is now part of `make check-package`, so a future operator that the browser build lacks fails in CI rather than in the browser.

### 4. Open points

- The parity checks in Node.js use the Node.js build of ONNX Runtime for the full answers. The asyncify check covers only the first token, which is sufficient to show that every operator is available.

### 5. WebGPU: copies between the GPU and the CPU

On an NVIDIA GeForce RTX 3060 Laptop GPU in Chrome, the first answers reached 4.6 to 6.3 tokens per second, against a target of 20. A plain ONNX Runtime session without transformers.js was equally slow, at about 245 ms per token, so the time was spent inside the model.

**S11 — Two operators send data back to the CPU at every step.** The session log shows which nodes ONNX Runtime could not place on the GPU:

- **`IsNaN` in every layer.** The exported attention replaces NaN values after the softmax with `Where(IsNaN(x), 0, x)`. The WebGPU backend has no `IsNaN`, so the softmax output of each of the 24 layers was copied to the CPU and back, which makes the GPU wait for all queued work 24 times per token.
- **`BitwiseAnd` and `BitShift` in the embedding.** The 4-bit codes were unpacked on the CPU after the row lookup on the GPU.

Both are replaced with operators that compute the same values on the GPU: `IsNaN(x)` becomes `Not(Equal(x, x))`, since NaN is the only value that is not equal to itself, and the codes are unpacked with integer division and subtraction. The attention mask also passes through the CPU once per step, but casting it did not move it to the GPU and did not change the speed, so it is left as it is.

| Model | Time per token, median | Tokens per second |
|---|---|---|
| Before the rewrite | 245 ms | 4.1 |
| `IsNaN` replaced | 58 ms | 17.2 |
| `IsNaN` and the embedding unpack replaced | 53 to 57 ms | 17.5 to 18.9 |

These are measurements of a plain ONNX Runtime session in headless Chrome over 50 steps, with run-to-run variation of about 10%. In the application, where transformers.js also keeps the key-value cache on the GPU, a 64-token answer was generated at 27.6 tokens per second, with the first token after 0.1 s, which meets both WebGPU targets. The rewritten model gives a maximum logit difference of 0.0 against the previous one, the same greedy answers, and the same perplexity of 158.014. The packaging step now refuses a model that contains `IsNaN`, `BitwiseAnd`, `BitShift` or `GatherBlockQuantized`.

### 6. WebAssembly: no fast kernel for 4-bit matrix products

**S12 — The WebAssembly backend is far below its target.** Without a GPU, the model generated 0.6 tokens per second with seven threads, against a target of 5. A profile of two steps attributes 94% of the time to `MatMulNBits`, the 4-bit matrix product, and its time hardly depends on the number of tokens in the step. This indicates that the WebAssembly build of ONNX Runtime has no optimized kernel for this operator and converts each 4-bit weight matrix to fp32 at every call. Threads do work: one thread needs 8.7 s per token and seven threads 2.3 s. Computing with 8-bit activations (`accuracy_level` 4) made it slower, at 3.7 s per token.

The rewrites of S10 and S11 do not affect this, since they touch only the embedding and the attention. A faster WebAssembly path needs a different weight format, such as 8-bit weights with `MatMulInteger`, which has optimized WebAssembly kernels but is larger to download.

### 7. WebGPU in Chrome on Linux

**S13 — Chrome on Linux does not offer hardware WebGPU by default.** In the first manual test, Chrome 154 on a Linux laptop with an RTX 3060 Laptop GPU and an Intel Iris Xe GPU fell back to WebAssembly, and selecting WebGPU failed. Headless Chrome on the same laptop showed the cause:

| Chrome flags | WebGPU adapter |
|---|---|
| none | none |
| `--enable-unsafe-webgpu` | SwiftShader, a software adapter |
| `--enable-unsafe-webgpu --enable-features=Vulkan --use-angle=vulkan` | NVIDIA, Ampere architecture |

The application rejects software adapters, since they run on the processor and would be slower than WebAssembly. Visitors who use Chrome on Linux therefore get the WebAssembly speed of S12 unless they start Chrome with these flags. The application now says why it runs on the processor and, on Linux, shows the command with a separate profile, since the flags enable WebGPU features that Chrome has not yet approved for Linux.
