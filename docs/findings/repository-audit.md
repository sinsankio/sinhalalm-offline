# Repository Audit: SinhalaLM Base and Instruct Checkpoints

The proposal designs the browser system around nine audit findings (F1–F9) about the two HelaAI repositories. Since every later step depends on them, we reproduced each finding against the exact files the pipeline uses, rather than carrying them over from the proposal as given.

In this document, we report the result of that reproduction. Every finding was checked against the repositories at pinned commit revisions, with the SHA-256 hash of each model file recorded in `convert/helaai.py`. Alongside the nine original findings, the reproduction corrected one detail of F4 and surfaced three additional findings (F10–F12).

### 1. Scope

| Repository | Revision |
|---|---|
| `HelaAI/SinhalaLM-V.0.0.1` (base) | `5f9c6d73e5ee3ec9e3f9cc2d7c03f95f144a2d8c` |
| `HelaAI/SinhalaLM-V.0.0.1-it` (instruct) | `f3b1f5d5b27786688dce99ba11ceac4220150e00` |

We inspected the safetensors headers, `config.json`, `tokenizer.json`, `bpe_vocabulary.json`, `modeling_sinhalalm.py`, the model cards, the training summaries and the validation loss logs. Model behavior was observed through the golden reference outputs in `golden/reference.json`, which contain 40-token greedy generations for eight prompts, each run through both models. These outputs were listed in `docs/model-outputs.md` until the switch to SinhalaLLM-1.7B, and the instruct answers to the four questions also appear in [model-comparison.md](../model-comparison.md).

### 2. Summary

| # | Finding | Status |
|---|---|---|
| F1 | 44.72M stored values, 32.85M unique parameters | Reproduced |
| F2 | The base repository ships no model classes | Reproduced and extended |
| F3 | `tokenizer.json` cannot be loaded | Reproduced |
| F4 | ZWJ is mapped to `<UNK>` | Reproduced, token count corrected |
| F5 | The architecture is equivalent to Llama | Reproduced (see [llama-equivalence.md](llama-equivalence.md)) |
| F6 | `<UNK>` is generated where a ZWJ belongs | Reproduced on new prompts |
| F7 | `config.json` and the model card disagree | Reproduced |
| F8 | Training was data-limited and outputs repeat | Reproduced |
| F9 | Placeholder license and "private" wording | Reproduced |
| F10 | The instruct `modeling_sinhalalm.py` cannot be imported | New |
| F11 | The instruct metadata names two different fine-tuning datasets | New |
| F12 | HTML entities from the ELI5 data appear in instruct outputs | New |

### 3. Reproduced findings

**F1 — Parameter count.** Both safetensors headers contain 44,724,736 stored values: 25,169,920 in the transformer layers, 7,680,000 in `tok_emb`, 512 in `ln_f`, 7,680,000 in `lm_head` and 4,194,304 in the RoPE `cos` and `sin` tables. The unique trainable parameters therefore total 32,850,432. The conversion in gate 1 also confirms that `lm_head.weight` is identical to `tok_emb.weight` and that every stored RoPE table equals the recomputed one, so both can be dropped without any loss.

**F2 — Missing model classes in the base repository.** The base `modeling_sinhalalm.py` is a download-and-run script whose `main()` imports `SinhalaGPT` and `FastBPETokenizer` from `modeling_sinhalalm` itself, which defines neither class. The base model card's loading code uses the same import. As a result, both checkpoints are loaded with the classes from the instruct repository, which fit the base weights since both files have identical tensor names and shapes.

**F3 — Unloadable `tokenizer.json`.** The `tokenizers` library fails on both repositories with `Token 'න්</w>' out of vocabulary at line 29875 column 1`. Of the 14,811 merges in the file, 5,814 reference a symbol that is not in its vocabulary. The file also uses the `Whitespace` pre-tokenizer and `end_of_word_suffix: "</w>"`, which differ from how `FastBPETokenizer` splits words and handles `</w>`. Both repositories ship the same `bpe_vocabulary.json` (identical SHA-256), which remains the single source of truth for tokenization.

**F4 — ZWJ mapped to `<UNK>`.** None of the 15,000 vocabulary entries contains the zero-width joiner. ශ්‍රී encodes as ශ් · `<UNK>` · ර · `ී</w>`, and `decode()` drops the `<UNK>`, so the four tokens of ප්‍රශ්නය decode as ප්රශ්නය. At the same time, 608 vocabulary entries contain the sequence `්ර` without a joiner.

The token count needs one correction. The proposal states that encoding the instruct template yields 37 tokens, 3 of which are `<UNK>`. In fact, the template alone (with an empty question) yields 24 tokens, 2 of which are `<UNK>`, one for each ZWJ in ප්‍රශ්නයට and ප්‍රශ්නය. The figure of 37 tokens with 3 `<UNK>` holds for the template together with the model card's example question ශ්‍රී ලංකාවේ අගනුවර කුමක්ද?, whose own ZWJ contributes the third `<UNK>`. In other words, the template the model learned to continue contains two `<UNK>` tokens, and any third one comes from the user's text, which is exactly the one that `stripZwj` removes.

**F6 — `<UNK>` where a ZWJ belongs.** The eight golden instruct generations contain 8 `<UNK>` tokens, and all 8 directly follow a token ending in the virama: five after ප්, two after සත් and one after ශ්. In the decoded text, they fall exactly where ප්‍ර, සත්‍ය and අවශ්‍ය need a ZWJ. The eight base-model generations contain none, even for the prompt that contains a ZWJ itself. Since these are different prompts from the ones in the proposal (which observed 5 of 6), the pattern now holds on a second, independent sample, although it is still a small one for RQ3.

**F7 — Context length and dropout.** The base `config.json` lists `block_size: 256` and `dropout: 0.3`, while the model card and the base training summary both list 512 and 0.2. The training summary also reports a wall time of 17,877.6 s and an average throughput of 555,302 tokens per second, which gives 9.93B tokens. This matches 150,000 iterations × an effective batch of 128 × 512 tokens = 9.83B, which supports a context length of 512. The instruct `config.json` already lists 512.

**F8 — Data-limited training and repetitive output.** With 9.83B training tokens over a corpus of about 304M tokens, the base model saw each token about 32 times. The validation loss log reads 3.53 at step 19,000, a best of 3.39 at step 115,500 and 3.41 at the last logged step (149,500), while the final training loss is 2.62. The golden outputs show the same behavior: the base prompt ඔබ ගුවන් යානා continues as a loop of මිලදී ගැනීමට පෙර ඔබ ගුවන් යානා, and none of the instruct answers to the four questions (prompts 5–8) addresses its question directly.

**F9 — License and visibility.** Both model cards contain the placeholder "Update this section with your actual terms ..." in their license section, and both describe the repository as private, although it is public. The repository ID `ChamaraVishwajithRajapaksha/SinhalaLM-V.0.0.1` appears on four lines of the instruct card and on one line of the base card. Permission to redistribute the converted weights has since been given by the HelaAI authors.

### 4. New findings

**F10 — The instruct `modeling_sinhalalm.py` cannot be imported.** The file is the fine-tuning script rather than a library. At import time, it checks for a local `model/sinhala_gpt_checkpoint.pt`, exits if the file is missing, and otherwise loads it with `torch.load`. Importing it is therefore not just unreliable, but also executes a pickle load on a local file. For this reason, `convert/helaai.py` parses the hash-verified file and executes only the ten definitions the pipeline needs (the tokenizer, the model classes and the prompt template), without running any of the training code.

**F11 — Two different fine-tuning datasets.** The instruct `config.json` and its training summary both name `ihalage/sinhala-finetune-qa-eli5` as the dataset. However, the training code in `modeling_sinhalalm.py` sets `HF_DATASET_ID` to `ChamaraVishwajithRajapaksha/sinhala-question-answer-pairs`, and the model card lists both. The repository therefore does not state unambiguously which data produced the published checkpoint.

**F12 — HTML entities in instruct outputs.** In three samples of 100 rows each, the Sinhala answers in `ihalage/sinhala-finetune-qa-eli5` contained 94, 89 and 185 occurrences of `&quot;`, while the same samples of `ChamaraVishwajithRajapaksha/sinhala-question-answer-pairs` contained none. The golden instruct answer to ශ්රී ලංකාවේ ජාතික ධජය විස්තර කරන්න. reads එය &quot;ප්රමාණවත් තරම් විශාල නොවේ&quot; යන ප්රශ්නය සමඟ සම්බන්ධ වේ., with the entity reproduced literally. This most likely explains the `&quot;` and `&` artifacts in instruct outputs, although `L;L2`, the other artifact noted in F8, did not appear in any of the sampled rows. It also suggests that the published checkpoint was fine-tuned at least partly on the ELI5 data (F11).

### 5. Implications for the design

- **F4.** The tokenizer regression test in Section 6.2 of the proposal changes from "the instruct template → 37 tokens with three `<UNK>`" to two cases: the template alone gives 24 tokens with two `<UNK>`, and the template with the model card question gives 37 tokens with three.
- **F6.** Decoding `<UNK>` after a virama as ZWJ remains the candidate recovery method, and the larger RQ3 sample is still required.
- **F10.** The reference implementation is executed from the pinned upstream source, not from a copy, so gates 1 and 3 always compare against the exact code that trained the models.
- **F12.** The application decodes HTML entities such as `&quot;` and `&amp;` in model outputs, as part of the text post-processing alongside ZWJ restoration.

### 6. Suggested upstream fixes

The following changes are small on the HelaAI side and would remove most of the workarounds in this project. They are listed here so that they can be shared with the authors.

1. Ship the model and tokenizer classes in the base repository, or point its loading code to the instruct repository (F2).
2. Split the instruct `modeling_sinhalalm.py` into an importable module and a training script (F10).
3. Regenerate `tokenizer.json` from `bpe_vocabulary.json`, or remove it, since the current file cannot be loaded and its pre-tokenizer does not match `FastBPETokenizer` (F3).
4. Remove the duplicate `lm_head` and the RoPE tables from `model.safetensors`, which would reduce the file from 44.72M to 32.85M stored values (F1).
5. Correct `block_size` and `dropout` in the base `config.json` (F7).
6. State which dataset produced the instruct checkpoint (F11).
7. Replace the placeholder license text and the "private" wording in both model cards (F9).

Throughout this audit, every original finding held against the pinned files, and the one correction (F4) makes the ZWJ story simpler rather than harder: the template carries two `<UNK>` tokens that the model learned to expect, and the third one was always the user's ZWJ.
