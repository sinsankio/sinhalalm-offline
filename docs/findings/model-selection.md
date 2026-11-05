# Model Selection: From SinhalaLM to SinhalaLLM-1.7B

The project started with the HelaAI SinhalaLM models, since their 32.85M parameters made them an easy fit for the browser. However, a model that fits the browser is only useful if its answers are useful, and the golden outputs showed that they were not. In this document, we record why the project switched models, how the alternatives were compared and why SinhalaLLM-1.7B was selected.

### 1. Why the HelaAI models were replaced

The HelaAI checkpoints were converted to Llama with exact parity (gate 1, [llama-equivalence.md](llama-equivalence.md)), so the conversion itself was not the problem. The problem was the models. On the eight golden prompts, the base model continued text in loops, and none of the instruct model's answers to the four questions addressed the question ([repository-audit.md](repository-audit.md), F8). After reviewing these outputs, the question-answering quality of both models was judged unsatisfactory, and the project looked for a different model with the same R&D intentions.

### 2. Tokenizer efficiency and ZWJ handling

Before generating any answers, we measured how each candidate tokenizer handles Sinhala, since tokens per word directly affect generation speed and how much text fits into the context. The measurement used 200 texts (9,665 words) from the test split of `ChamaraVishwajithRajapaksha/sinhala-question-answer-pairs`, and the ZWJ round trip encoded and decoded ශ්‍රී ලංකාවේ ප්‍රධාන ක්‍රියාව සත්‍ය.

| Tokenizer | Vocabulary | Sinhala tokens per word | ZWJ round trip |
|---|---|---|---|
| SinhalaLLM-1.7B | 63,344 | 1.76 | kept |
| HelaAI FastBPE | 15,000 | 2.66 | lost |
| Gemma 4 E2B | 262,144 | 3.16 | kept |
| Qwen3.5 | 248,070 | 4.92 | kept |
| SmolLM2-1.7B (SinhalaLLM's base) | 49,152 | 15.99 | kept |

The comparison between SinhalaLLM-1.7B and its own base model highlights how much the vocabulary extension matters: the same architecture needs about 9 times fewer tokens per Sinhala word after the extension.

### 3. Answers on ten Sinhala questions

SinhalaLM-it and the three candidates that fitted the available disk space were run on ten Sinhala questions with greedy decoding ([model-comparison.md](../model-comparison.md)). The new candidates ran at 4-bit precision, so the answers reflect the quality after quantization.

- **SinhalaLM-it** — none of the ten answers addresses its question.
- **SinhalaLLM-1.7B** — every answer is on topic and grammatical, and eight of ten stop by themselves, although several factual answers do not match the reference answers.
- **Qwen3.5-2B** — on topic but often factually wrong, with unrelated English glosses, and all ten answers run to the token limit.
- **Qwen3.5-0.8B** — mostly factually wrong, and all ten answers run to the token limit.

Gemma 4 E2B, SinLlama 8B, Tiny Aya and Gemma 3 1B were not screened, for reasons of disk space, size, gating or license.

### 4. Decision

SinhalaLLM-1.7B was selected on 2026-11-05. It is the only screened model whose answers address the questions, its tokenizer is the most efficient for Sinhala and keeps the ZWJ, it uses the Llama architecture that the export path already supports, and it is released under Apache-2.0. At the same time, two related decisions were taken:

- The application provides question answering only. SinhalaLLM-1.7B is published as a single instruction-tuned checkpoint, so the text-continuation mode of the original proposal was dropped.
- The size, memory and speed targets were raised to fit a 1.74B model: at most 1.2 GB of download, at most 2 GB of peak tab memory, at least 20 tokens per second on WebGPU and 5 on multi-threaded WASM.

The downside of this choice is the model's maturity. It was published on 2026-10-13, its benchmark results were produced by its author, and its factual accuracy is limited, so the project's own golden outputs and the native-speaker evaluation in RQ3 carry more weight than before.
