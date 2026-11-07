# Browser runtime compatibility of the q4 model

The first browser test of the packaged q4-v0-small model failed while the session was created, before any token was generated. In this document, we report the cause (S10), the replacement that makes the model load in the browser runtime, and the checks that show the replacement does not change the model.

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

The table keeps its size, since the codes and the scales are stored exactly as before. The report of each quantized variant records the replacement as `portable_embedding`.

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

- WebGPU support for `BitwiseAnd` and `BitShift` on `uint8` is not yet measured. If the WebGPU backend does not implement them, ONNX Runtime runs these few nodes on the CPU, which affects only the embedding lookup of each step. The browser tests of the inference worker check this.
- The parity checks in Node.js use the Node.js build of ONNX Runtime for the full answers. The asyncify check covers only the first token, which is sufficient to show that every operator is available.
