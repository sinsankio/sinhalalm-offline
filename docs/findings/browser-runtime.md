# Browser runtime compatibility of the q4 model

The first browser test of the packaged q4-v0-small model failed while the session was created, before any token was generated. In this document, we report the cause (S10), the replacement that makes the model load in the browser runtime, and the checks that show the replacement does not change the model. The first answers in the browser then showed two speed problems, which we report as S11 (WebGPU) and S12 (WebAssembly), and the first manual test showed that Chrome on Linux does not offer hardware WebGPU by default (S13). Measuring the acceptance targets then showed four problems with how the model files are stored and loaded (S14–S17).

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

The rewrites of S10 and S11 do not affect this, since they touch only the embedding and the attention. A faster WebAssembly path needs a different weight format, such as 8-bit weights with `MatMulInteger`, which has optimized WebAssembly kernels but is larger to download. Since WebGPU met its targets in manual testing, WebAssembly is kept as a slow fallback for browsers without hardware WebGPU, and its speed target is dropped rather than adding a second download.

### 7. WebGPU in Chrome on Linux

**S13 — Chrome on Linux does not offer hardware WebGPU by default.** In the first manual test, Chrome 154 on a Linux laptop with an RTX 3060 Laptop GPU and an Intel Iris Xe GPU fell back to WebAssembly, and selecting WebGPU failed. Headless Chrome on the same laptop showed the cause:

| Chrome flags | WebGPU adapter |
|---|---|
| none | none |
| `--enable-unsafe-webgpu` | SwiftShader, a software adapter |
| `--enable-unsafe-webgpu --enable-features=Vulkan --use-angle=vulkan` | NVIDIA, Ampere architecture |

The application rejects software adapters, since they run on the processor and would be slower than WebAssembly. Visitors who use Chrome on Linux therefore get the WebAssembly speed of S12 unless they start Chrome with these flags. The application now says why it runs on the processor and, on Linux, shows the command with a separate profile, since the flags enable WebGPU features that Chrome has not yet approved for Linux.

### 8. Weights inside the graph file double the memory

**S14 — With the weights inside the ONNX file, ONNX Runtime keeps a second copy in its WebAssembly heap.** The Diagnostics tab measured 2.67 GB of page memory for q4 on WebGPU and on WebAssembly, almost all of it in the inference worker, and the renderer process used 3.8 GB. ONNX Runtime reads the whole file into its WebAssembly heap and copies the weights out of it, and WebAssembly memory never shrinks.

Storing the same weights in a separate data file (`model_q4.onnx_data`) avoids the copy, without changing any value:

| q4, greedy answer to the first golden question | Page memory | Answer |
|---|---|---|
| Weights inside the graph file, WebGPU | 2.67 GB | correct |
| Separate data file, WebGPU | 0.06 GB | identical |
| Separate data file, WebAssembly | 1.88 GB | identical |

On WebGPU, the weights now go to the GPU without passing through the heap. The download size is unchanged.

### 9. A 2.48 GB data file never finishes loading

**S15 — transformers.js cannot load a single external data file of 2.48 GB.** The q8 graph file loaded, but its data file never reported any progress, and the load did not finish within 15 minutes, while the server delivered the file in 3 s. Split into two files of 1.24 GB (`model_quantized.onnx_data` and `model_quantized.onnx_data_1`), which transformers.js loads as numbered parts, q8 loaded in 13 s. The packaging step now writes q4 as one data file and q8 as two.

### 10. Large files are not cached from memory

**S16 — Chrome failed to cache the 1.18 GB data file from an in-memory buffer.** transformers.js stores each downloaded file in Cache Storage from the buffer it has read. For the data file, `Cache.put` failed with an internal error, so the model was downloaded again on every visit and was not available offline. The worker now downloads the model files itself and streams each response into the cache, which Chrome stored without errors, and transformers.js then loads the files from the cache. This also gives one progress bar for all files of a variant.

### 11. A nearly full disk

**S17 — Chrome stops storing data well before the reported quota when the disk is nearly full.** With about 6 GB free on the disk, Chrome reported a quota of 11.9 GB, but storing q8 failed after 2.15 GB with a network error. With the browser profile in memory, the same download succeeded. The application now checks the quota before a download, removes the parts of a failed download, and says that the disk may be too full.

### 12. The 8-bit variant

On FLORES+ (1,012 sentences), q8 has a perplexity of 115.5, against 113.2 for fp32 (+2.0%), 158.0 for q4 and 195.0 for the author's Q4_K_M, so it passes the perplexity part of the quality gate. A native speaker also reviewed its answers (confirmed by Sineth on 2026-11-08), so it passes both parts. On WebGPU, with the files split as in S15, it reached the same speed as q4:

| Variant, WebGPU (RTX 3060 Laptop GPU) | Download | Load from the network | Median first token | Median speed | Renderer process |
|---|---|---|---|---|---|
| q4 | 1.18 GB | 7.4 to 28.3 s | 69 ms | 26.5 tokens/s | 1.37 GB |
| q8 | 2.48 GB | 12.8 s | 76 ms | 26.6 tokens/s | 1.99 GB |

The medians come from the benchmark in the Diagnostics tab (five sample questions, greedy decoding, up to 64 tokens). On WebAssembly, q8 is not practical, since its 8-bit matrix products have no fast kernel either (S12).

### 13. Acceptance targets

| Target | q4 on WebGPU | q4 on WebAssembly | Result |
|---|---|---|---|
| Download at most 1.2 GB | 1.18 GB | 1.18 GB | met |
| Peak tab memory at most 2 GB (page memory) | page 0.09 GB, renderer 1.37 GB | page 1.78 GB, renderer 2.35 GB | met |
| Warm load at most 10 s | 3.7 s + 0.7 s warm-up | 2.8 to 3.2 s + 4.1 to 5.3 s warm-up | met |
| First token at most 1 s on WebGPU, about 60 prompt tokens | 484 ms with 52 prompt tokens | 5.2 s with 52 prompt tokens, about 2 s for short questions (no target) | met |
| At least 20 tokens per second on WebGPU | 26.5 to 27.3 | 0.5 (no target, S12) | met |

All values come from headless Chrome on the RTX 3060 Laptop GPU and its 8-core CPU. Page memory is the browser's own measurement of the page and its workers, without GPU memory; renderer is the resident memory of the tab's process.

After these measurements, the WebAssembly target for the first token was dropped, as the speed target before it (S12), and the memory target counts the page memory measured by the browser, with the renderer process recorded alongside. The application warns when the 8-bit model runs on WebAssembly.

At the end of Phase 4, Sineth made q8 the default on hardware WebGPU, with q4 as the fallback without hardware WebGPU or when q8 cannot be downloaded, stored or loaded. The default download is therefore 2.48 GB, and the 1.2 GB target applies to the q4 fallback.
