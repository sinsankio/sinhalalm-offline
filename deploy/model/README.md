---
license: apache-2.0
language:
  - si
base_model: namindu94/SinhalaLLM-1.7B
library_name: transformers.js
pipeline_tag: text-generation
tags:
  - onnx
  - sinhala
  - webgpu
---

# SinhalaLLM-1.7B for the browser (ONNX)

Converted and quantized ONNX versions of [SinhalaLLM-1.7B](https://huggingface.co/namindu94/SinhalaLLM-1.7B) (version 2.1, revision `{revision}`) by Namindu Sanchila Nadeeshan, for [Offline සිංහල](https://huggingface.co/spaces/{space}), an application that answers Sinhala questions inside the browser with transformers.js.

| Variant | Files | Size | FLORES+ perplexity |
|---|---|---|---|
| 8-bit (`q8`, default on WebGPU) | `onnx/model_quantized.onnx` with two data files | 2.48 GB | 115.5 |
| 4-bit (`q4`, fallback) | `onnx/model_q4.onnx` with one data file | 1.18 GB | 158.0 |

For comparison, the original model in fp32 has a perplexity of 113.2 on the same 1,012 FLORES+ sentences, and the author's 4-bit GGUF release (Q4_K_M) 195.0. The 4-bit variant keeps the value projection of layer 0 in fp32. Both variants rewrite a few operators into equivalent standard ones, so that they run in ONNX Runtime Web, with identical logits.

`manifest.json` lists the size and SHA-256 hash of every file.

The model often gets facts wrong from memory, and its answers are not suitable for medical, legal or financial decisions.

## License and credits

Released under the Apache License 2.0, like the original model; see `NOTICE.txt`. SinhalaLLM-1.7B is built on [SmolLM2-1.7B](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B).
