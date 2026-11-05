# Configuration Translation: Parity Gate 1 for SinhalaLLM-1.7B

SinhalaLLM-1.7B was saved with `transformers` 5.19.0, while the ONNX exporter requires `transformers` 4.57.6. The inspection showed that the older version reads the model's RoPE base as 10000 instead of 130000, without any error ([sinhalallm-inspection.md](sinhalallm-inspection.md), S1). Since every later step starts from the model as the export toolchain loads it, this load has to be proven equal to the model as its author's toolchain loads it.

In this document, we report parity gate 1, which compares the two toolchains on the same weights.

### 1. Method

The pipeline uses two Python environments from one `uv` project: a **reference environment** with `transformers` 5.19.0 and an **export environment** with `transformers` 4.57.6, declared as conflicting dependency groups.

- **Golden outputs** — the reference environment loads the pinned checkpoint in fp32 and writes the prompt token IDs, the full prompt logits and 64-token greedy generations for the ten golden questions, together with the logits of a random 1,024-token input at every 16th position. The question prompts are built with the model's own ChatML template.
- **Configuration translation** — the export environment reads `rope_parameters.rope_theta` from the raw `config.json` and writes it into the field that version 4.57.6 reads, `rope_theta`. The translation also enables the key-value cache and sets both stop tokens, `<|im_end|>` (2) and `<|endoftext|>` (0).
- **Gate** — the export environment loads the same weights in fp32 with the translated configuration and compares its logits and greedy generations with the golden outputs.
- **Negative control** — the rotary embedding of the loaded model is replaced with one built from the untranslated configuration, and its logits on the random input must differ from the golden logits by at least the tolerance. This confirms that the gate can detect the error it is meant to catch.

### 2. Results

| Check | Result |
|---|---|
| RoPE base, translated / untranslated | 130,000 / 10,000 |
| Max \|logit diff\|, ten questions and random input | 0.0 |
| Greedy mismatches over ten 64-token generations | 0 |
| Negative control, max \|logit diff\| on the random input | 11.66 |
| Parameters | 1,740,474,368 |
| Gate 1 | Pass |

The translated load is identical to the reference on every case, while the untranslated load differs by up to 11.66 in the logits. The golden outputs were also regenerated from scratch with `make check-golden` and matched the committed files exactly, which confirms that the reference itself is reproducible.

### 3. What the wrong RoPE base does to answers

To see whether the error would be noticed without the gate, we generated answers to the first three golden questions with the untranslated rotary embedding.

| Question | Correct RoPE base (130,000) | Untranslated RoPE base (10,000) |
|---|---|---|
| Q1 | ශ්‍රී ලංකාවේ අගනුවර ශ්‍රී ජයවර්ධනපුර කෝට්ටේ වේ. | ශ්‍රී ලංකාවේ අගනුවර ශ්‍රී ජයවර්ධනපුර කෝට්ටේ ය. |
| Q2 | සිංහල අකුරු 60 ක් පමණ වේ. | සිංහල අකුරු 60 ක් පමණ වේ. |
| Q3 | ගස් වැදගත් වන්නේ ඒවා පරිසරයට සහ මිනිසුන්ට ප්‍රයෝජනවත් වන බැවිනි. ගස්වල ඇති ශාක, ශාකවල ඇති ... | ගස් වැදගත් වන්නේ ඒවා පරිසරයට ප්‍රයෝජනවත් වන බැවිනි. ගස්වල ඇති ශාක, ශාක, ශාක, ශාක, ශාක, ශාක ... |

Short answers stay plausible, and Q2 is even identical, while the longer answer to Q3 falls into repetition after its first sentence. In other words, the wrong RoPE base does not produce obviously broken text, but a model that degrades with answer length. This is exactly the kind of error that a manual check of a few short answers would miss, which is why the gate compares logits rather than answers.

### 4. Implications

- The export in Phase 3 starts from the translated configuration written by gate 1, so gate 2 compares the ONNX model with a load that is already proven equal to the reference.
- The model is identical across the two toolchains in fp32 on CPU, so any difference that later gates measure comes from their own step: the ONNX export, the quantization or the browser runtime.
