---
license: cc-by-sa-4.0
language:
  - si
pretty_name: SinhalaLLM-1.7B in the browser, evaluation results
tags:
  - quantization
  - onnx
  - webgpu
  - sinhala
configs:
  - config_name: perplexity
    data_files: perplexity.csv
    default: true
  - config_name: agreement
    data_files: agreement.csv
  - config_name: per_sentence
    data_files: per_sentence.csv
  - config_name: browser
    data_files: browser.csv
---

# SinhalaLLM-1.7B in the browser: evaluation results

Evaluation results from bringing [SinhalaLLM-1.7B](https://huggingface.co/namindu94/SinhalaLLM-1.7B) (version 2.1) by Namindu Sanchila Nadeeshan into the browser: the quality of every quantized variant that was tried, and the speed and memory of the shipped models in Chrome. The shipped models are in [sinsankio/sinhalallm-1.7b-onnx-web](https://huggingface.co/sinsankio/sinhalallm-1.7b-onnx-web), and the application that uses them is [Offline සිංහල](https://huggingface.co/spaces/sinsankio/sinhalalm-offline).

### Tables

**`perplexity`** — one row per variant.

| Column | Meaning |
|---|---|
| `variant` | How the weights were stored |
| `file_bytes` | Size of the ONNX model with its data files (empty for the GGUF release) |
| `sentences` | Number of FLORES+ sentences scored, always the first sentences of the split |
| `perplexity` | Perplexity per token on those sentences (lower is better) |
| `bits_per_character` | The same loss expressed per character |
| `fp32_perplexity_same_sentences` | Perplexity of the original fp32 model on exactly the same sentences |
| `change_vs_fp32_percent` | Relative change against that fp32 value |

**`agreement`** — how closely each variant follows the fp32 model on ten test questions: the share of positions where the most likely next token of the prompt is the same (`prompt_top1_agreement`), and the number of 64-token greedy answers that are identical (`greedy_exact_matches`). Slow variants were checked on fewer questions.

**`per_sentence`** — the negative log-likelihood, token count and character count of every scored sentence, by its position in the split, so that any two variants can be compared on the same sentences. The sentences themselves are not included; they are in [FLORES+](https://huggingface.co/datasets/openlanguagedata/flores_plus), which requires accepting its terms.

**`browser`** — speed, first-token time, load time and memory of the shipped 4-bit and 8-bit models in headless Chrome on an NVIDIA RTX 3060 Laptop GPU and its 8-core processor.

### Method

- **Text.** The 1,012 sentences of the FLORES+ `sin_Sinh` devtest split (131,080 characters). Each sentence is scored on its own after the `<|endoftext|>` token.
- **Runtime.** ONNX Runtime 1.31.0 on the CPU for every ONNX variant. The GGUF release was dequantized and evaluated on the same sentences.
- **Subsets.** The fp16 and three 4-bit method variants were evaluated on the first 200 sentences, so their change is computed against fp32 on the same 200 sentences.

### Key results

- fp16 matches fp32, and 8-bit weights lose 2.0% in perplexity.
- Every plain 4-bit variant loses 164% to 248%, whatever the precision of the head and the embedding, asymmetric quantization, block size 16 or HQQ [[CITE: HQQ quantization method]].
- Keeping one 2048 × 2048 matrix, the value projection of the first layer, in fp32 reduces the 4-bit loss to 32.7% (head tied to the fp32 embedding) or 39.6% (4-bit head and embedding, 1.18 GB). Both are better than the author's own Q4_K_M release on the same sentences (+72.2%).
- In Chrome, both shipped models generate about 26.5 tokens per second on WebGPU, while the 4-bit model reaches 0.5 tokens per second on WebAssembly.

### License and credits

The per-sentence scores are derived from FLORES+, so this dataset uses the same license, CC BY-SA 4.0.

- **SinhalaLLM-1.7B** by Namindu Sanchila Nadeeshan, Apache License 2.0, including the Q4_K_M release evaluated here.
- **FLORES+** by the Open Language Data Initiative, CC BY-SA 4.0. NLLB Team et al., *No Language Left Behind: Scaling Human-Centered Machine Translation*, arXiv:2207.04672, 2022.
- **ONNX Runtime** by Microsoft, used for every evaluation, and **llama.cpp**, which made the Q4_K_M release.
