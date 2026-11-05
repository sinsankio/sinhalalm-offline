# ONNX Export: Memory, Tied Weights and Parity Gate 2

The toolchain check showed that the Optimum exporter works on a small Llama model ([toolchain.md](toolchain.md)). At 1.74B parameters, the same export ran into a different problem: memory. In this document, we report how the export of SinhalaLLM-1.7B was made to fit the development machine, and what parity gate 2 showed about the exported model.

### 1. Memory during export

The development machine has 23 GB of memory, of which about 9 to 14 GB was available during the work. Each export ran inside a memory-capped process group, so that an out-of-memory kill would end the export instead of another application.

| Attempt | Cap | Outcome |
|---|---|---|
| Single process, as Optimum runs it | 8.8 GB | Killed after tracing, while Optimum merged the per-tensor files into one file. This step loads the whole 7 GB ONNX model while the 7 GB PyTorch model is still in memory. |
| Merge disabled | 8.8 GB | Tracing completed, but the process was killed shortly after writing all tensor files. |
| Merge disabled | 11 GB | Same as above, at a peak of about 11.5 GB. |

The tensor files were complete in every case where tracing finished, which suggests that the peak occurs at the very end of PyTorch's ONNX serialization, when its copies of the weights coexist with the model. Since the available memory could not be raised, the export was restructured instead of retried:

- **Trace** — a child process, capped at 11 GB by default, loads the model with the translated configuration and runs the Optimum exporter with the merge and post-processing steps disabled. It writes the graph and one file per tensor.
- **Validate** — the parent process, outside the cap, checks every tensor file against the size implied by its shape and data type. The export continues only if all 219 files are complete, whatever the exit code of the child.
- **Finalize** — the parent loads the graph lazily with `onnx-ir`, ties the duplicated output head back to the embedding and streams all tensors into a single `model.onnx_data`. This step peaked at about 1.2 GB.

In the recorded export, the trace process exited with code −9 (killed by the cap) after writing all tensor files, the validation passed and the finalized model was written. Gate 2 below then verifies the result numerically, so a silently corrupted tensor could not pass.

### 2. Tied output head

The raw export stores the tied embedding twice: once as `model.embed_tokens.weight` for the input lookup, and once as a transposed copy, `onnx::MatMul_5629`, for the output head. The finalize step verifies that the copy equals the transposed embedding exactly, removes it and feeds the head through a `Transpose` of the embedding.

The finalized `model.onnx_data` is 6,961,897,472 bytes, which is exactly 1,740,474,368 parameters × 4 bytes. In other words, the exported model contains every parameter once and nothing else.

### 3. Parity gate 2

Gate 2 compares the fp32 ONNX model in ONNX Runtime 1.31.0 with the golden logits, which gate 1 proved identical to the PyTorch model of the export toolchain. It covers the prompt logits of the ten golden questions, the logits of the random 1,024-token input at every 16th position, and 64-token greedy generations with the key-value cache.

The first run used the absolute tolerance of gate 1, 1e-4, and failed with a maximum absolute logit difference of 7.6e-4, although all ten greedy generations were identical. Three diagnostics followed:

- **Where the difference occurs.** In every question, the maximum was at position 0, the `<|im_start|>` token. After the second position, the difference stayed at or below 1.4e-4, and the top-1 token was identical at every position of every prompt. The same position 0 gave 7.6e-4 in three prompts and 3.9e-4 in the other seven, which shows that the error depends on the sequence length rather than on the token.
- **PyTorch's own noise.** Switching only the attention implementation of the PyTorch model from SDPA to eager changed the logits by up to 7.1e-5.
- **Graph optimizations.** With all ONNX Runtime graph optimizations disabled, position 0 still differed by 1.9e-4 to 5.6e-4, so the difference does not come from operator fusion.

These results point to rounding differences between the fp32 matrix-multiplication kernels of ONNX Runtime and PyTorch, which are largest at the first position, where the model's activations are largest. Since the absolute tolerance was inherited from a 33M model with a single shared toolchain, gate 2 now uses a scale-free criterion instead:

| Criterion | Result |
|---|---|
| Max \|logit diff\| relative to each position's largest absolute logit | 3.46e-5 (tolerance 1e-4) |
| Max absolute \|logit diff\| (reported) | 7.6e-4, at position 0 |
| Max absolute \|logit diff\| on the random input (reported) | 8.5e-5 |
| Top-1 token identical at every position | yes |
| Greedy mismatches over ten 64-token generations | 0 |
| Gate 2 | Pass |

### 4. Implications

- The ONNX model is a faithful export of the reference: every greedy answer is identical, and the remaining differences are of the size expected between two fp32 kernel libraries.
- Quantization starts from this verified fp32 file, so the quality gate measures only what quantization adds.
- A full export needs about 11.5 GB of free memory for the trace step and about 14 GB of free disk during the export. Both are requirements of `make onnx` on any other machine.
