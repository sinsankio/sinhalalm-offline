# End-to-End Fidelity in On-Device Language Models for Low-Resource Scripts: An Empirical Study of Sinhala in the Browser

*System design, implementation, testing and deployment plan for client-side Sinhala question answering*

---

### Abstract

For a long period of time, running a language model meant sending text to a server and waiting for a response. However, the combination of compact models, ONNX-based runtimes and WebGPU has begun to make it practical to run a model entirely inside a web browser, with no server and no network connection after the first download. SinhalaLLM-1.7B is an open, instruction-tuned language model built for Sinhala on top of SmolLM2-1.7B, with a Sinhala-aware tokenizer that extends the base vocabulary by 14,192 Sinhala tokens. At 1.74B parameters, it is small enough to be a candidate for client-side inference on desktop and laptop computers.

In this proposal, we present an end-to-end plan to run SinhalaLLM-1.7B offline in the browser for Sinhala question answering. The project first evaluated the HelaAI SinhalaLM models, whose answers did not reach a satisfactory question-answering quality, and then compared alternative models on the same Sinhala questions, where SinhalaLLM-1.7B was the only screened model whose answers addressed the questions. Before designing the system, we inspected its repository and found several issues that directly shape the design: a configuration that the export toolchain silently loads with the wrong RoPE base, a tokenizer that expects the zero-width joiner (ZWJ) although user input often omits it, and a context length that differs from the trained one. The proposed system addresses each of these through a configuration translation verified against the author's toolchain, explicit ZWJ repair of the input and parity gates at every conversion step, and is evaluated on quantization quality, cross-device latency and ZWJ fidelity.

---

### 1. Introduction

Sinhala is spoken by over 20 million people, yet it remains comparatively under-served by mainstream multilingual language models. The common way to bring LLM capability to Sinhala has been continual pretraining of a large existing model, which requires substantial GPU memory and incurs high inference cost. SinhalaLLM-1.7B explores a smaller direction: a 1.7B model whose vocabulary was extended for Sinhala, and which its author reports at about one fifth of the memory of the 8B SinLlama model.

A model of this size does not just reduce server cost, but opens the possibility of removing the server altogether. Consider a scenario where a reader opens a web page once on Wi-Fi, installs it as an app, and later asks a Sinhala question while having no connection at all. In this proposal, we use this **offline reader** as the running example, and every design decision is evaluated against whether it serves that reader.

Rather than treating the browser deployment as a packaging exercise, we treat it as a sequence of verifiable transformations. Each step (reference loading, tokenization, ONNX export, quantization, in-browser execution) must reproduce the output of the previous step within a defined tolerance before the next step begins.

The objectives of this work are:

- To load SinhalaLLM-1.7B in the export toolchain without any loss of fidelity relative to the toolchain the model was published with.
- To build a browser application that runs single-turn Sinhala question answering fully offline after the first load.
- To measure the trade-off between model size and quality across quantization levels.
- To measure load time, time to first token and generation speed across WebGPU and WebAssembly (WASM) backends on desktop and laptop computers.
- To evaluate whether repairing the ZWJ in user input, and preserving it through the browser runtime, improves the fidelity of Sinhala text end to end.

The rest of this proposal is organized as follows. Section 2 summarizes the model, the model selection and the findings of our repository inspection. Section 3 defines the research questions. Section 4 presents the system design, Section 5 the implementation plan, Section 6 the testing and evaluation strategy, and Section 7 the deployment plan. Section 8 covers risks, Section 9 the milestones and community sharing plan, and Section 10 concludes the proposal.

---

### 2. Background and Preliminary Findings

#### 2.1 The SinhalaLLM-1.7B model

**SinhalaLLM-1.7B** (version 2.1, `namindu94/SinhalaLLM-1.7B`) is a decoder-only language model with 1.74B parameters, using the Llama architecture with 24 layers, a hidden size of 2,048, 32 attention heads and tied input and output embeddings. It is built on SmolLM2-1.7B, whose 49,152-token vocabulary is extended with 14,192 Sinhala tokens to 63,344 tokens, and the embedding matrix is padded to 63,360 rows. The model was trained with a context of 2,048 tokens and is instruction-tuned for Sinhala chat, writing, summarization and question answering from given passages, using the ChatML format.

Its tokenizer uses a **Sinhala-aware pre-tokenizer**: the GPT-2 word pattern is extended so that a word consists of letters, combining marks, the zero-width non-joiner and the ZWJ together, which keeps Sinhala vowel signs and conjuncts inside the same word. During training, a normalizer restored the ZWJ in common conjunct clusters, since large web corpora often drop it.

The model card states that the model is not for medical, legal or financial advice, that it often gets facts wrong from memory, and that its benchmark results were produced by its author. It is released under the Apache-2.0 license.

#### 2.2 Model selection

The project originally targeted the HelaAI SinhalaLM models (32.85M parameters). Their architecture was converted to Llama with exact parity, but the answers of both the base and the instruct model on the golden prompts did not reach a satisfactory question-answering quality ([repository-audit.md](findings/repository-audit.md), [model-comparison.md](model-comparison.md)).

We then compared the HelaAI instruct model with SinhalaLLM-1.7B, Qwen3.5-2B and Qwen3.5-0.8B on ten Sinhala questions ([model-comparison.md](model-comparison.md)). Table 1 summarizes the comparison.

*Table 1: Candidate models for offline Sinhala question answering.*

| Model | Parameters | Sinhala tokens per word | ZWJ round trip | Answers on the ten questions |
|---|---|---|---|---|
| HelaAI SinhalaLM-V.0.0.1-it | 32.85M | 2.66 | lost | none addresses its question |
| SinhalaLLM-1.7B | 1.74B | 1.76 | kept | all on topic, several factually wrong |
| Qwen3.5-2B | about 2B | 4.92 | kept | on topic but often wrong, all hit the token limit |
| Qwen3.5-0.8B | about 0.8B | 4.92 | kept | mostly wrong, all hit the token limit |

SinhalaLLM-1.7B was selected, since it is the only screened model whose answers address the questions, its tokenizer is the most efficient for Sinhala, and it uses the Llama architecture that the existing export path supports.

#### 2.3 Repository inspection

Before designing the system, we inspected the configuration files, the tokenizer files, the chat template, the model card and the GGUF metadata of the SinhalaLLM-1.7B repository at revision `b24d58b7929e1ed50f683b435a9a629a73e27af5`. Table 2 summarizes the findings and how each one affects the design.

*Table 2: Inspection findings and their impact on the browser system.*

| # | Finding | Evidence | Design impact |
|---|---|---|---|
| S1 | The export toolchain loads the model with the wrong RoPE base. | `config.json` was written by `transformers` 5.19.0 and stores the base as `rope_parameters: {rope_theta: 130000}`. `transformers` 4.57.6, which the ONNX exporter requires, does not read this field and loads `rope_theta` as 10000. The GGUF file states `llama.rope.freq_base` 130000. | The pipeline translates the configuration for the export toolchain, and gate 1 verifies the translated load against the author's toolchain (Section 5.3). |
| S2 | The tokenizer expects the ZWJ, but has no normalizer. | `tokenizer.json` defines no normalizer, so the ZWJ repair existed only in the training data. ශ්‍රී ලංකාව ප්‍රධාන encodes as 3 tokens, while ශ්රී ලංකාව ප්රධාන, typed without the ZWJ, encodes as 6 fragments. | User input is repaired before encoding (Section 4.2, D), and RQ3 measures the effect. |
| S3 | The ZWJ appears in the vocabulary only in rakaransaya and yansaya. | Of the 765 vocabulary tokens that contain a ZWJ, 611 contain `්‍ර` and 150 contain `්‍ය`. The 16 conjunct clusters of the training normalizer are not published. | The input repair targets `්ර` and `්ය`, and its edge cases are evaluated by native speakers (RQ3). |
| S4 | The browser tokenizer reproduces the Python tokenizer. | transformers.js 4.3.2 produced the same token IDs as the Python `tokenizers` library on 210 inputs: 200 Sinhala texts and 10 edge cases, including a byte order mark, U+0085, the zero-width non-joiner and ChatML special tokens. ZWJ survives decoding. | No custom tokenizer port is needed; gate 3 verifies the transformers.js tokenizer at scale. |
| S5 | The configured context is longer than the trained context. | `max_position_embeddings` is 8,192, while the model card states a trained context of 2,048 tokens. | The application caps prompt plus generation at 1,024 tokens, which also bounds the key-value cache memory. |
| S6 | Two tokens end generation. | `generation_config.json` lists `eos_token_id` as `[2, 0]`, which are `<|im_end|>` and `<|endoftext|>`, while `config.json` lists only 0. `config.json` also sets `use_cache` to false. | Generation stops at either token, and the export enables the key-value cache explicitly. |
| S7 | The model is large for a browser. | 1,740,474,368 parameters, of which the tied 63,360 × 2,048 embedding is 7.5%. The published `model.safetensors` is 3.48 GB in bfloat16; in fp32, the model is about 6.96 GB. | Only quantized variants are shipped, and the fp32 browser comparison is replaced by a same-file comparison (gate 4). |
| S8 | Factual accuracy is limited. | In the comparison, several answers did not match the reference answers, and two long answers became repetitive. The model card also states that the model often gets facts wrong from memory. | The application exposes a repetition penalty and shows a clear notice that answers are unverified. |

In other words, the inspection changes the project from "port a custom model and its tokenizer" into "load a standard model correctly and keep Sinhala text intact". This is because S4 removes the tokenizer port, while S1 and S2 move the risk into configuration and text handling, which is where a Sinhala-specific system is most likely to fail silently.

---

### 3. Research Questions

Since the application itself is an engineering artifact, we frame the measurable parts of the work as research questions. This ensures that the results can be reported, compared and reproduced by others.

- **RQ1 (Quantization):** How does perplexity on held-out Sinhala text change across fp32, fp16, int8 (q8) and 4-bit (q4) variants, and what is the smallest variant whose quality stays within an acceptable margin of fp32?
- **RQ2 (Latency):** How do download size, cold and warm load time, time to first token and generation speed differ between the WebGPU and WASM backends across desktop and laptop computers?
- **RQ3 (ZWJ fidelity):** Does repairing the ZWJ in user input (`්ර` to `්‍ර` and `්ය` to `්‍ය`) reduce tokenization fragments and improve the answers, and is the ZWJ preserved end to end from input to rendered output in the browser? How often do native speakers judge the repaired input and the rendered output as correct?

---

### 4. System Design

#### 4.1 Requirements

**Functional requirements**

- **FR1** — Single-turn Sinhala question answering with SinhalaLLM-1.7B, using its ChatML format.
- **FR2** — Token-by-token streaming of the output.
- **FR3** — User control over temperature, top-k, repetition penalty and maximum new tokens.
- **FR4** — Full offline operation after the first successful load, including after a browser restart.
- **FR5** — Correct rendering of Sinhala text, including conjuncts with the ZWJ.

**Non-functional requirements**

- **NFR1 (Privacy)** — No user text leaves the device. The application sends no analytics by default.
- **NFR2 (Fidelity)** — Every conversion step passes its parity gate (Section 6).
- **NFR3 (Responsiveness)** — Inference never blocks the user interface thread.
- **NFR4 (Portability)** — The application runs on WebGPU where a hardware adapter is available and falls back to WASM everywhere else.
- **NFR5 (Footprint)** — The default variant downloads at most 1.2 GB, and peak tab memory stays at most 2 GB on desktop and laptop computers (Section 6.6).
- **NFR6 (Transparency)** — The interface states the model's intended use and limitations as given in the model card.

#### 4.2 Architecture overview

The system has two halves. A **build-time pipeline** runs once in Python and produces verified, quantized ONNX artifacts. A **runtime application** runs entirely in the browser and consumes those artifacts.

```mermaid
flowchart LR
  subgraph Build["Build-time pipeline (Python, make ci)"]
    A["SinhalaLLM safetensors<br/>+ tokenizer.json"] --> B["A. Reference load and<br/>config translation, gate 1"]
    B --> C["ONNX export with KV cache<br/>parity gate 2"]
    C --> D["Quantization fp16 / q8 / q4<br/>quality gate, RQ1"]
    A --> E["Golden tokenizer cases<br/>10k sentences"]
  end
  subgraph Runtime["Browser runtime"]
    F["E. UI thread"] -->|"question + settings"| G["C. Inference Web Worker"]
    G --> H["B. transformers.js tokenizer<br/>parity gate 3"]
    G --> I["transformers.js + ONNX Runtime Web<br/>WebGPU or WASM"]
    G -->|"token stream"| J["D. Text processing<br/>ZWJ repair, entity decoding"]
    J --> F
    K["F. Service worker + Cache Storage"] -.->|"offline assets"| F
    K -.->|"model files"| I
  end
  D -->|"versioned model files"| L[("G. Static hosting")]
  E -.->|"test fixtures"| H
  L --> K
```

*Figure 1: End-to-end architecture which demonstrates how verified model artifacts flow from the build-time pipeline into an offline browser runtime.*

The components are described below.

**A) Reference load and configuration translation.** The published checkpoint is already a `LlamaForCausalLM`, so no tensor conversion is needed. However, the ONNX exporter requires `transformers` 4.57.6, which reads the configuration differently from `transformers` 5.19.0, the version the model was saved with (S1). This component writes a configuration in the 4.57 format, with `rope_theta` set to 130,000 and `use_cache` enabled (S6), and gate 1 verifies that the translated load produces the same logits as the author's toolchain. In the context of our offline reader, this step matters because a model with the wrong RoPE base still loads and generates fluent-looking text, so the error would otherwise reach the reader unnoticed.

**B) Tokenizer.** The browser uses the transformers.js tokenizer built from the published `tokenizer.json`, rather than a custom port. This is because the early check in S4 showed identical token IDs on 210 inputs, including the Sinhala pre-tokenizer's combining-mark and ZWJ classes. Gate 3 extends this check to 10,000 sentences, so that the browser produces exactly the token IDs the model was trained on.

**C) Inference worker.** A dedicated Web Worker loads the model through transformers.js, selects the WebGPU or WASM backend, runs generation and streams text back. A software WebGPU adapter is treated as unavailable, since it is most likely slower than multi-threaded WASM. This ensures that the page stays responsive while the model generates (NFR3).

**D) Text processing.** This component handles the text on both sides of the model:

- **Input, ZWJ repair:** `්ර` becomes `්‍ර` and `්ය` becomes `්‍ය` before encoding, since the model learned these clusters with the ZWJ (S2, S3). Text that already contains the ZWJ is left unchanged.
- **Output, streaming decode:** the byte-level tokenizer can split a Sinhala character, which takes three bytes in UTF-8, across two tokens. The worker therefore emits text only when complete characters are available.
- **Output, HTML entities:** entities such as `&quot;` and `&amp;` are decoded into the characters they stand for, with a small buffer so that an entity split across two tokens is still recognized.

The following examples show what the input repair does.

*Example 1: why the input is repaired.* Consider a reader who types ශ්රී ලංකාව ප්රධාන without the ZWJ, which is common when text is typed on a keyboard that does not insert it. The tokenizer produces 6 tokens, since the vocabulary learned these clusters in their ZWJ form. After the repair, the same text becomes ශ්‍රී ලංකාව ප්‍රධාන and produces 3 tokens, which is the form the model saw during training.

*Example 2: why the repair is limited to two clusters.* The vocabulary contains the ZWJ only after a virama and before ර or ය (S3). Repairing other sequences would create text the model never saw, so the rule stays within these two clusters, and its edge cases are evaluated by native speakers (RQ3).

```mermaid
flowchart LR
  U["Reader's question<br/>may lack ZWJ, e.g. ශ්රී"] --> R["repairZwj<br/>්ර to ්‍ර, ්ය to ්‍ය"]
  R --> T["ChatML template"]
  T --> ENC["transformers.js tokenizer"]
  ENC --> M["Model generates IDs"]
  M --> DEC["Streaming decode<br/>complete characters only"]
  DEC --> HE["decodeEntities<br/>&quot; to quote"]
  HE --> OUT["Rendered Sinhala text<br/>with correct conjuncts"]
```

*Figure 2: Text processing across input and output, which represents how the ZWJ is restored before encoding and how the output is kept intact while streaming.*

**E) User interface.** The interface provides a question box, generation settings, a model variant selector, a load progress indicator and a limitations notice. A Sinhala font (Noto Sans Sinhala) is bundled locally, so that conjuncts render correctly without a network connection.

**F) Offline layer.** A service worker caches the application shell, and transformers.js stores model files in the browser's Cache Storage. The application also requests persistent storage through `navigator.storage.persist()`, which reduces the chance that the browser evicts the model under storage pressure. In other words, the offline reader depends on two caches: one for the page and one for the model.

```mermaid
sequenceDiagram
  participant R as Offline reader
  participant SW as Service worker
  participant C as Cache Storage
  participant H as Static host
  Note over R,H: First visit, online
  R->>H: Load page
  H-->>SW: App shell and font, precached
  R->>H: Fetch model files from versioned path
  H-->>C: Stored by transformers.js
  R->>R: Verify SHA-256 hashes
  R->>R: navigator.storage.persist()
  Note over R,H: Later visit, offline
  R->>SW: Load page
  SW-->>R: App shell from cache
  R->>C: Request model files
  C-->>R: Model files from cache
  R->>R: Generate locally, no network
```

*Figure 3: First online visit and later offline visit, which demonstrates how the page cache and the model cache together enable FR4.*

**G) Static hosting.** Model files are served under versioned paths (for example, `/models/sinhalallm-1.7b/v2.1/`), so that a model update never mixes files from two versions inside the reader's cache.

#### 4.3 Request lifecycle

When the offline reader asks a question, the request flows through the system as follows:

1. The UI thread sends the question and settings to the worker.
2. The worker repairs the ZWJ in the question, applies the ChatML template and encodes the prompt with the transformers.js tokenizer.
3. The worker checks that the prompt plus `max_new_tokens` does not exceed 1,024 tokens (S5), and reduces `max_new_tokens` if necessary.
4. transformers.js runs generation with key-value caching, stopping at `<|im_end|>` or `<|endoftext|>` (S6).
5. Each new piece of text is decoded once its characters are complete, passed through entity decoding and streamed to the UI.

```mermaid
sequenceDiagram
  participant UI as UI thread
  participant W as Inference worker
  participant T as Tokenizer, transformers.js
  participant RT as transformers.js model
  participant P as Text processing
  UI->>W: Question and settings
  W->>P: repairZwj
  W->>T: ChatML template, encode
  T-->>W: Prompt IDs
  W->>W: Cap max_new_tokens so the total stays within 1,024
  W->>RT: generate with KV cache, stop at im_end or endoftext
  loop Each new token
    RT-->>W: Token ID via streamer
    W->>P: Complete characters, decodeEntities
    P-->>UI: Text chunk
  end
  RT-->>W: Stop token or token limit reached
  W-->>UI: Done
```

*Figure 4: Request lifecycle of a single question, which illustrates how tokenization, generation and text processing stay inside the worker while the UI only receives text.*

#### 4.4 Key design decisions

**Standard Llama export instead of a custom runtime.** SinhalaLLM-1.7B is already a `LlamaForCausalLM`, so the standard export path (Optimum to ONNX, transformers.js runtime) applies directly. This choice is quite reasonable because the path provides key-value caching, sampling and quantization tooling that is already maintained. The downside of this approach would be a dependency on the `transformers` version the exporter requires, which is exactly what S1 exposed, so the configuration translation and gate 1 are part of the design rather than a workaround.

**transformers.js tokenizer instead of a custom port.** Since the browser tokenizer already reproduces the Python tokenizer on Sinhala text (S4), the application uses it directly. The cost is a parity test suite; the benefit is that the tokenizer stays in sync with the published `tokenizer.json` without any custom code.

**Input repair instead of input stripping.** The model learned conjuncts with the ZWJ, so the application adds the missing ZWJ rather than removing it. This keeps the reader's question in the same form as the training data.

**Web Worker instead of main-thread inference.** A 1.74B model can block the UI thread for hundreds of milliseconds per step. Running inference in a worker keeps typing and scrolling responsive.

**Self-hosted model files instead of fetching from the Hub at runtime.** Setting `env.allowRemoteModels = false` and serving files from the same origin means the runtime contacts no third party, which aligns with NFR1 and makes offline behavior predictable. The Apache-2.0 license permits redistributing the converted model, provided that the license and attribution are included.

---

### 5. Implementation Plan

#### 5.1 Phase 0: License and scope

SinhalaLLM-1.7B is released under Apache-2.0, so the converted ONNX files can be redistributed together with the license text and an attribution to the model's author. The repository is pinned to revision `b24d58b7929e1ed50f683b435a9a629a73e27af5`, and the SHA-256 hash of every downloaded file is verified.

#### 5.2 Phase 1: Reference outputs

We load the model with `transformers` 5.19.0, the version recorded in its `config.json`, and store golden outputs:

- logits for a fixed set of questions in the ChatML format;
- 64-token greedy generations for the same questions;
- perplexity on a held-out Sinhala set. We propose the FLORES+ `sin_Sinh` devtest split (1,012 sentences) for this purpose. Since FLORES+ is gated on Hugging Face, this reference perplexity is computed together with the quality evaluation in Section 5.6.

Since the reference and the export toolchain need different `transformers` versions, the Python environment is split into a reference environment and an export environment.

#### 5.3 Phase 2: Configuration translation (parity gate 1)

The export environment loads the checkpoint with a translated configuration:

```python
cfg = LlamaConfig.from_pretrained(source)
cfg.rope_theta = 130000.0
cfg.use_cache = True
cfg.eos_token_id = [2, 0]
```

The gate passes when the maximum absolute logit difference between the export environment (`transformers` 4.57.6, fp32) and the reference environment (`transformers` 5.19.0, fp32) is below 1e-4 on the golden questions and on a random 1,024-token input, and when all greedy generations are identical. As a negative control, the untranslated configuration must fail the same gate, which confirms that the gate detects the RoPE error in S1.

#### 5.4 Phase 3: Tokenizer parity (parity gate 3)

The parity test encodes 10,000 Sinhala sentences with the Python `tokenizers` library and with transformers.js in Node.js and requires identical ID arrays. The sentences are drawn with a fixed seed from four public sources at pinned revisions: 4,000 from the pretraining corpus of the HelaAI models, 3,000 from the test split of a Sinhala question-answer dataset, 1,000 from the Sinhala ELI5 translation and 2,000 from Sinhala Wikipedia. Since the licenses of these sources differ, the sentences are regenerated by `make golden` and are not committed; only a manifest with the dataset revisions, the seed and the SHA-256 hash of the generated file is.

Alongside the sentences, the test deliberately includes named cases: text with and without the ZWJ, the zero-width non-joiner, a byte order mark, the whitespace characters that differ between Python and JavaScript, ChatML special tokens and empty input. The chat template is tested in the same way: the prompt string built in the browser must be byte-for-byte equal to the one built by `transformers` in Python.

#### 5.5 Phase 4: ONNX export and quantization (parity gate 2)

The translated model is exported with key-value caching, using the Optimum ONNX exporter:

```bash
optimum-cli export onnx --model ./sinhalallm-1.7b --task text-generation-with-past ./export/sinhalallm-1.7b
```

The exported model is then quantized into fp16, q8 and q4 (4-bit, MatMulNBits) variants with ONNX Runtime's quantization tools, so that the files follow the `onnx/` folder layout and naming that transformers.js expects. Parity gate 2 compares the fp32 ONNX model with the PyTorch model in the export environment, using the same tolerance as gate 1.

The expected download sizes are derived from the 1.74B parameters:

| Variant | Lower bound | If the embedding table stays fp32 |
|---|---|---|
| fp32 | 6.96 GB | 6.96 GB |
| fp16 | 3.48 GB | 3.48 GB |
| q8 | 1.74 GB | about 2.13 GB |
| q4 | 0.87 GB | about 1.32 GB |

The right-hand column reflects a practical consideration. The toolchain check showed that the 4-bit quantizer leaves a tied embedding and output head in full precision. For SinhalaLLM-1.7B, the embedding is 7.5% of all parameters, so this would raise the q4 file above the 1.2 GB target. Quantizing the embedding and the output head is therefore part of RQ1, and the real sizes are recorded there.

The fp32 export of a 1.74B model needs about 7 GB of disk for the weights and about 7 GB for the ONNX file, together with a comparable amount of memory during export. Large artifacts are kept in a model store on an external drive.

#### 5.6 Phase 5: Quality evaluation before the UI (RQ1)

For each variant, we compute perplexity on the held-out set with ONNX Runtime in Python, together with the greedy agreement rate against fp32 on the golden questions. Since tokenizers differ between models, bits per character are reported alongside perplexity. The shipped default is the smallest variant whose perplexity stays within 2% of fp32 and whose download stays within NFR5. Since the browser application is built only after this step, a variant that fails here never reaches the reader.

#### 5.7 Phase 6: Browser application

The application is built with Vite, with the inference code in a Web Worker. The worker core is shown below:

```js
import { AutoModelForCausalLM, AutoTokenizer, TextStreamer, env } from "@huggingface/transformers";

env.allowRemoteModels = false;
env.localModelPath = "/models/";

const MODEL = "sinhalallm-1.7b/v2.1";
const tokenizer = await AutoTokenizer.from_pretrained(MODEL);
const model = await AutoModelForCausalLM.from_pretrained(MODEL, { device, dtype: "q4" });

async function answer(question, opts, onText) {
  const messages = [{ role: "user", content: repairZwj(question) }];
  const inputs = tokenizer.apply_chat_template(messages, {
    add_generation_prompt: true,
    return_dict: true,
  });
  const maxNew = Math.min(opts.maxNewTokens, 1024 - inputs.input_ids.dims.at(-1));
  const entities = createEntityDecoder();
  const streamer = new TextStreamer(tokenizer, {
    skip_prompt: true,
    skip_special_tokens: true,
    callback_function: (text) => onText(entities.push(text)),
  });

  await model.generate({
    ...inputs, max_new_tokens: maxNew, eos_token_id: [2, 0],
    do_sample: true, temperature: opts.temperature, top_k: opts.topK,
    repetition_penalty: opts.repetitionPenalty, streamer,
  });
  onText(entities.flush());
}
```

The device is selected before loading: WebGPU when `navigator.gpu.requestAdapter()` returns a hardware adapter, and WASM otherwise. The streamer interface and its handling of incomplete characters should be checked against the pinned transformers.js version.

The offline layer is added with `vite-plugin-pwa` (Workbox), which caches the application shell and the bundled font and makes the application installable. Model files stay in the transformers.js cache, and the application calls `navigator.storage.persist()` after the first successful load.

#### 5.8 Project structure

```
sinhalalm-web/
├── convert/                  # Python, build time
│   ├── sinhalallm.py         # pinned source, config translation, loaders
│   ├── parity_check.py       # gate 1
│   ├── export_onnx.sh        # Optimum export
│   ├── quantize.sh           # fp16 / q8 / q4
│   ├── eval_quality.py       # gate 2 + RQ1 perplexity
│   └── make_golden.py        # golden logits + tokenizer cases
├── golden/                   # fixtures shared by Python and JS tests
├── web/
│   ├── src/text/zwj.js       # repairZwj
│   ├── src/text/entities.js  # decodeEntities
│   ├── src/worker/inference.worker.js
│   ├── src/ui/
│   ├── public/models/<model>/<version>/
│   └── tests/                # Vitest unit tests, Playwright E2E
└── Makefile                  # local CI pipeline
```

---

### 6. Testing and Evaluation

The testing strategy follows the chain of transformations. Each gate compares a step with the step before it, so that a failure always points to one component.

#### 6.1 Parity gates

```mermaid
flowchart LR
  A["Reference<br/>transformers 5.19"] -->|"gate 1"| B["Export load<br/>transformers 4.57"]
  B -->|"gate 2"| C["ONNX fp32"]
  C -->|"quality gate"| D["fp16 / q8 / q4"]
  F["Tokenizer<br/>Python"] -->|"gate 3"| G["Tokenizer<br/>transformers.js"]
  D -->|"gate 4"| E["Same file in the browser<br/>WASM"]
  G --> E
```

*Figure 5: Chain of parity gates, which demonstrates how each transformation is verified against the previous one so that a failure always points to a single component.*

| Gate | Compares | Pass criterion |
|---|---|---|
| 1 | Reference (`transformers` 5.19.0) vs export load (`transformers` 4.57.6, translated config) | max \|logit diff\| < 1e-4 and identical greedy output; the untranslated config must fail |
| 2 | Export load (PyTorch) vs ONNX fp32 | max \|logit diff\| < 1e-4 and identical greedy output |
| 3 | Python `tokenizers` vs transformers.js tokenizer | identical IDs on 10,000 sentences and identical chat-template strings |
| 4 | Default quantized variant in Python ONNX Runtime vs the same file in the browser (WASM) | identical greedy output on golden questions |
| Quality | Quantized vs fp32 | perplexity within 2% of fp32 |

Gate 4 compares the same quantized file in two runtimes, rather than the fp32 model, since a 1.74B model in fp32 (about 7 GB) most likely exceeds what the WASM backend can address. In this way, gate 4 still confirms that the browser runtime adds no drift of its own.

#### 6.2 Unit and integration tests

- **Tokenizer** (Vitest): golden cases from gate 3, with named regression cases:
  - ශ්‍රී ලංකාව ප්‍රධාන → 3 tokens, and ශ්රී ලංකාව ප්රධාන → 6 tokens;
  - text with the zero-width non-joiner, a byte order mark and U+0085 → the same IDs as Python;
  - the ChatML prompt for a question → the same string and IDs as Python.
- **Text processing**: `repairZwj` on a curated list of rakaransaya and yansaya words, on text that already contains the ZWJ and on words where the rule should not apply; `decodeEntities` on `&quot;`, `&amp;` and an entity split across two chunks.
- **Context limit**: prompts close to 1,024 tokens never exceed the limit after generation.

#### 6.3 End-to-end tests

End-to-end tests run with Playwright in Chromium using the WASM backend, since WebGPU exposes only a software adapter in headless Chromium. The suite covers the first load with a progress indicator, question answering with streaming, settings changes, and the offline scenario: after the first load, the test switches the browser context offline with `context.setOffline(true)`, reloads the page and confirms that generation still works. This test is the automated version of our offline reader.

#### 6.4 Cross-device benchmarks (RQ2)

WebGPU behavior varies by device and driver, so benchmarks run on physical devices rather than in CI. For each device and backend, we record:

- download size and cold load time (empty cache);
- warm load time (from cache);
- time to first token;
- generation speed in tokens per second, together with Sinhala characters per second, since tokenizers differ;
- peak memory.

The protocol uses a fixed question, greedy decoding, 128 new tokens and the median of five runs. The application includes a benchmark page that runs this protocol and shows the results with a "copy results" button, so that community members can voluntarily contribute measurements from their own devices.

Since the target devices are desktop and laptop computers, the benchmark matrix runs on a laptop with an Intel Core i7-11375H (8 threads), an NVIDIA GeForce RTX 3060 Laptop GPU, Intel Iris Xe graphics and 23 GB of memory: Chrome with WebGPU on the discrete GPU, Chrome with WebGPU on the integrated GPU, Chrome with multi-threaded WASM (8 threads), Chrome with single-threaded WASM, Firefox with WASM, and a low-end stand-in with 2 WASM threads and 4× CPU throttling. Community measurements from the benchmark page extend this matrix to other computers.

#### 6.5 Human evaluation (RQ3)

ZWJ repair and answer quality are judged by native Sinhala speakers. For RQ3, raters see the answers to the same questions with and without input repair, and mark which answer is better and whether the rendered conjuncts are correct. The number of raters and the sample size are decided after a first sample of generations, starting from three native-speaker raters and about 150 items, with Fleiss' kappa for inter-rater agreement.

#### 6.6 Acceptance criteria

The application is ready for release when all parity gates pass, the selected variant passes the quality gate, the offline end-to-end test passes, and the benchmark results meet the targets below. The targets are provisional and are revisited after the first benchmark run:

| Metric | Target |
|---|---|
| Download size | at most 1.2 GB for the default variant |
| Peak tab memory | at most 2 GB, with the context capped at 1,024 tokens |
| Warm load time | at most 10 s |
| Time to first token (about 60 prompt tokens) | at most 1 s on WebGPU and 3 s on multi-threaded WASM |
| Generation speed | at least 20 tokens/s on WebGPU and 5 tokens/s on multi-threaded WASM |

---

### 7. Deployment

#### 7.1 Hosting

Multi-threaded WASM requires `SharedArrayBuffer`, which browsers only enable on cross-origin isolated pages. This makes the response headers `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp` a hard requirement for good WASM performance.

The project is hosted on a Hugging Face static Space, which accepts these headers through the `custom_headers` key of its configuration. Model files larger than 10 MB are tracked with `git-xet`. Whether the Space is cross-origin isolated inside the iframe on huggingface.co, and how service workers behave there, is verified on a live Space before release; the offline reader is expected to use the direct `.hf.space` URL.

#### 7.2 CI/CD pipeline

The pipeline runs locally through a Makefile (`make ci`) rather than on a hosted CI service, in four stages:

1. **Convert**: runs the reference outputs, parity gates 1 and 2, quantization and the quality gate on the CPU, and writes the model artifacts to the model store.
2. **Tokenizer**: generates golden cases in Python and runs the transformers.js parity test in Node.js.
3. **Build and test**: builds the web application and runs unit and Playwright tests, including gate 4 and the offline test.
4. **Deploy**: on a version tag, `make deploy` publishes the application and the model files under a new versioned path.

```mermaid
flowchart LR
  P["make ci"] --> J1["Convert<br/>gates 1-2, quantize, quality gate"]
  P --> J2["Tokenizer<br/>gate 3"]
  J1 --> J3["Build and test<br/>unit, Playwright, gate 4, offline test"]
  J2 --> J3
  J3 --> T{"Version tag?"}
  T -->|"yes"| J4["Deploy<br/>app + model under a new versioned path"]
  T -->|"no"| X["Stop after tests"]
```

*Figure 6: CI/CD pipeline, which represents how every release passes all gates before it reaches the static host.*

#### 7.3 Versioning and integrity

Each release pins the transformers.js, Optimum, ONNX Runtime and PyTorch versions used to produce it, and writes a manifest with the SHA-256 hash of every model file. The application verifies the hashes after download, which detects corrupted or partially cached files before they reach the reader. Since model files live under versioned paths, a new release downloads the new files once and the old cache entries are deleted after the switch.

#### 7.4 Privacy and responsible use

All inference happens on the device, and the application sends no analytics. The interface states the intended use from the model card: the model often gets facts wrong from memory, its answers are unverified, and it is not suitable for medical, legal or financial questions. It also credits the model's author, includes the Apache-2.0 license and links to the model card.

---

### 8. Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Configuration read differently by two `transformers` versions (S1) | Silently wrong model | Configuration translation, gate 1 with a negative control, pinned versions. |
| Tokenizer drift between Python and JS | Silent quality loss | Parity gate 3 on 10,000 sentences, run on every `make ci`. |
| ZWJ repair rules wrong for some words (S3) | Incorrect input | Rules limited to the clusters in the vocabulary, native-speaker evaluation (RQ3) and a setting to disable the repair. |
| Export needs more memory or disk than available (S7) | Export fails | Model store on an external drive; export in a single process with other applications closed; fp16 export as a fallback. |
| q4 leaves the tied embedding in fp32 | Download above 1.2 GB | Quantize the embedding and head as part of RQ1; ship q8 or a larger budget if quality requires it. |
| WASM cannot hold large variants | Gate 4 or the WASM fallback fails | Gate 4 on the quantized default; WASM support verified at the start of the browser work. |
| WebGPU unavailable or unstable | Slower generation | Automatic WASM fallback with cross-origin isolation for multi-threading. |
| Storage eviction | Offline mode fails | `navigator.storage.persist()`, hash checks and a clear re-download prompt. |
| Factual errors and repetition (S8) | User trust | Repetition penalty, limitations notice and correct framing of the model as a research prototype. |
| Model maturity: recently published, author-reported benchmarks | Unknown weaknesses | Own golden outputs, the comparison on ten questions and native-speaker evaluation. |

---

### 9. Milestones and Community Sharing

The work is organized into four milestones. Each milestone ends with a result that can be shared with the community on its own.

| Milestone | Scope | Shareable result |
|---|---|---|
| M1 | Phases 0–2: model selection, reference outputs, configuration translation | The model comparison and the RoPE configuration finding with gate 1 |
| M2 | Phases 3–5: tokenizer parity, ONNX export, quantization | The size versus quality chart (RQ1) |
| M3 | Phase 6 and Section 6: application, tests, offline mode | A demo of question answering in airplane mode |
| M4 | Section 7 and benchmarks | The cross-device latency results (RQ2), the ZWJ fidelity results (RQ3) and the public link |

`[[NEEDS INPUT: available time per week and target dates for each milestone]]`

---

### 10. Conclusion

Throughout this proposal, we presented an end-to-end plan for running SinhalaLLM-1.7B offline in the browser. We began with a model selection, which replaced the HelaAI SinhalaLM models after their answers did not reach a satisfactory quality, and with an inspection of the selected model, which showed that the main risks lie in how the configuration is read and how the ZWJ is handled rather than in the architecture. We then designed a system that separates a verified build-time pipeline from a browser runtime, with a configuration translation, a verified browser tokenizer, explicit ZWJ repair and a parity gate at every transformation. Finally, we defined the testing, deployment and evaluation plan that turns the application into measurable results on quantization quality, cross-device latency and ZWJ fidelity.

In the context of our offline reader, the outcome is a Sinhala question-answering page that works with no server and no connection after its first load, built on a model that was designed for Sinhala from its tokenizer upwards.
