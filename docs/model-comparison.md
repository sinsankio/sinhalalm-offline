# Model Comparison: Alternatives to SinhalaLM for Offline Sinhala Question Answering

The answers of the HelaAI SinhalaLM models did not reach a satisfactory question-answering quality, so the project needs a different model before the tokenizer, export and browser work continue. The R&D intentions stay the same: a model that runs fully offline in the browser, a chain of verified transformations from the original weights to the browser, and the three research questions on quantization, latency and ZWJ fidelity.

In this document, we compare the candidate models side by side on the same Sinhala questions, so that the choice of model can be made from their actual answers rather than from model cards alone.

### 1. Candidates

| Model | Parameters | License | Architecture | Browser download (q4, text only) | Sinhala tokens per word | ZWJ round trip | Screened |
|---|---|---|---|---|---|---|---|
| HelaAI SinhalaLM-V.0.0.1-it (current) | 32.85M | placeholder | SinhalaGPT, converts to Llama | about 16–47 MB | 2.66 | lost (`<UNK>`) | yes, baseline |
| [namindu94/SinhalaLLM-1.7B](https://huggingface.co/namindu94/SinhalaLLM-1.7B) v2.1 | 1.74B | Apache-2.0 | `LlamaForCausalLM` (SmolLM2-1.7B with a Sinhala vocabulary extension) | about 1.1 GB | 1.76 | kept | yes |
| [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) | about 2B | Apache-2.0 | `Qwen3_5ForConditionalGeneration` (multimodal) | about 1.39 GB | 4.92 | kept | yes |
| [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) | about 0.8B | Apache-2.0 | `Qwen3_5ForConditionalGeneration` (multimodal) | about 0.59 GB | 4.92 | kept | yes |
| [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it) | 2.3B effective | Apache-2.0 | `Gemma4ForConditionalGeneration` (multimodal) | about 3.1 GB | 3.16 | kept | no, see Section 6 |
| [polyglots/SinLlama_v01](https://huggingface.co/polyglots/SinLlama_v01) | 8B | not stated | Llama-3-8B with a Sinhala vocabulary extension | about 4.5 GB or more | not measured | not measured | no, see Section 6 |

The tokens-per-word figures were measured on 200 Sinhala texts (9,665 words) from the test split of `ChamaraVishwajithRajapaksha/sinhala-question-answer-pairs`. The ZWJ round trip encodes and decodes ශ්‍රී ලංකාවේ ප්‍රධාන ක්‍රියාව සත්‍ය and checks whether the ZWJ survives. The browser download is the q4 decoder plus the q4 embedding table, without the vision or audio encoders of the multimodal models. For SinhalaLLM-1.7B, the size of its Q4_K_M GGUF file is used as a stand-in, since no ONNX build exists yet.

### 2. How the answers were produced

- **Questions** — ten Sinhala questions: the four golden questions from `golden/prompts.json` (Q1–Q4) and six questions from the test split of `ChamaraVishwajithRajapaksha/sinhala-question-answer-pairs` (Q5–Q10), together with the reference answer from that dataset.
- **Decoding** — greedy decoding with at most 150 new tokens, using each model's own prompt format: the Alpaca-style Sinhala template for SinhalaLM, and ChatML for SinhalaLLM and Qwen3.5 (with thinking disabled for Qwen3.5).
- **Precision and runtime** — SinhalaLM ran in fp32 with the HelaAI reference implementation. SinhalaLLM-1.7B ran from its Q4_K_M GGUF file in `transformers` 4.57.6, with the repository's own `tokenizer.json`. Both Qwen3.5 models ran from the `onnx-community` q4 builds in transformers.js 4.3.2 under Node.js, which is the same runtime the browser would use.
- **Answers** — shown exactly as each model produced them, with line breaks collapsed into spaces. No ZWJ restoration or HTML entity decoding is applied.

Since the three new candidates ran at 4-bit precision, their answers show the quality a reader would see after quantization, not the best quality of the original weights.

### 3. Summary of the answers

The following is our reading of the answers in Section 5, and it should be confirmed by a native-speaker review.

- **SinhalaLM-it (current)** — none of the ten answers addresses its question, and three of them loop until the token limit.
- **SinhalaLLM-1.7B** — every answer is on topic and grammatical, and eight of ten stop by themselves. Q1 matches the commonly cited administrative capital, while several factual answers (for example, Q5, Q6 and Q8) do not match the reference answers, and Q7 and Q10 become repetitive.
- **Qwen3.5-2B** — the answers are on topic but often factually wrong, mix in unrelated English glosses, and all ten run to the token limit.
- **Qwen3.5-0.8B** — most answers are factually wrong (Q1 names Korea as the capital), and all ten run to the token limit.

In other words, SinhalaLLM-1.7B is the only screened model whose answers read like answers, although its factual accuracy is still limited.

### 4. What changes in the project for each candidate

The R&D intentions carry over to any candidate, but the work inside each step changes.

| Area | SinhalaLLM-1.7B | Qwen3.5-2B or 0.8B |
|---|---|---|
| Gate 1 (reference model) | Already `LlamaForCausalLM`, so the SinhalaGPT conversion is no longer needed. Gate 1 becomes a check that the reference loads correctly, including the RoPE base: the GGUF file states 130,000, while `config.json` leaves `rope_theta` unset. | Not needed in the same form; the reference is the official `transformers` model. |
| Gate 2 (ONNX export) | Own export through the Optimum Llama path, as planned. | Own export is not possible yet, since the transformers.js conversion script for Qwen3.5 is not published. The pipeline would verify the `onnx-community` files instead of producing them. |
| Gate 3 (tokenizer) | Python `tokenizers` against the transformers.js tokenizer. This is a real Sinhala fidelity question, since the custom pre-tokenizer relies on `\p{M}` and ZWJ character classes that the JavaScript regex must reproduce. | Same comparison, on a standard byte-level tokenizer. |
| RQ3 (ZWJ) | Changes direction. `tokenizer.json` has no normalizer, so ZWJ repair existed only in the training data: ශ්රී typed without a ZWJ becomes 6 fragments instead of 3 tokens. The browser would add the missing ZWJ to the input, and the study measures whether that improves the answers. | ZWJ survives tokenization, so RQ3 measures only whether the model writes correct conjuncts. |
| RQ1 and RQ2 | Unchanged in method, at about 50 times the size of SinhalaLM. | Unchanged in method. |
| Worker, offline layer, Space deployment, Makefile CI | Unchanged. | Unchanged, apart from the multimodal model class. |

Two of your earlier decisions no longer fit any candidate and need new values:

- **Size and memory targets (NFR5 and Section 6.6).** The limits of 70 MB per model and 500 MB of tab memory were set for a 33M model. For SinhalaLLM-1.7B, we propose at most 1.2 GB of download for the default variant, at most 2 GB of peak tab memory with a 1,024-token context cap, at least 20 tokens per second on WebGPU and at least 5 on multi-threaded WASM, and a time to first token of at most 1 s on WebGPU and 3 s on WASM. These are provisional until the first benchmark.
- **Gate 4.** It compares the fp32 ONNX model in the browser with Python. A 1.74B model in fp32 is about 7 GB, which most likely exceeds what the WASM backend can address, so gate 4 would compare a quantized variant instead. This is verified at the start of the browser work.

The conversion pipeline also needs about 20 GB of free disk space for a model of this size (about 7 GB each for the fp32 weights and the fp32 ONNX export, plus the quantized variants). The machine currently has about 4 GB free.

### 5. Answers per question

#### Q1. ශ්‍රී ලංකාවේ අගනුවර කුමක්ද?

*Source: golden prompt.*

**HelaAI SinhalaLM-V.0.0.1-it (current)** — 17 tokens, stopped by itself

> මම හිතන්නේ එය බොහෝ දුරට සත්ය විය හැකිය.

**SinhalaLLM-1.7B v2.1** — 9 tokens, stopped by itself

> ශ්‍රී ලංකාවේ අගනුවර ශ්‍රී ජයවර්ධනපුර කෝට්ටේ වේ.

**Qwen3.5-2B** — 150 tokens, cut at the 150-token limit

> ශ්‍රී ලංකාවේ අගනුවර **කොළඹ** වේ. කොළඹ යනු ශ්‍රී ලංකාවේ ප්‍රධාන ජාතික නගරයක් වන අතර, එහි ප්‍රධාන වශයෙන් පවතින ජාතික ප්‍රධාන කටයුතු සහ ජාතික අධ්‍යාපන සේවා ස්ථාන වේ. මෙම නගරය තුළ ශ්�

**Qwen3.5-0.8B** — 150 tokens, cut at the 150-token limit

> ශ්‍රී ලංකාවේ අගනුවරය **කොරියාව** (Koraiya) වේ. කොරියාව යනු ශ්‍රී ලංකාවේ පුද්ග්‍රමයේ (Pudum) අගනුවරයකි. මෙය පුද්ග්‍රමයේ ප්‍රධිමය අගනුවරයකි. මෙම අගනුවරයේ පුද්ග්‍රමයේ ප්‍රධිමය අගනුවරය

#### Q2. සිංහල අකුරු කීයක් තිබේද?

*Source: golden prompt.*

**HelaAI SinhalaLM-V.0.0.1-it (current)** — 32 tokens, stopped by itself

> ඔබ අදහස් කරන්නේ කුමක්ද යන්න පැහැදිලි කිරීමට මට අවශ්ය නැත, නමුත් මම එය කියවා ඇත්තෙමි.

**SinhalaLLM-1.7B v2.1** — 10 tokens, stopped by itself

> සිංහල අකුරු 60ක් පමණ වේ.

**Qwen3.5-2B** — 150 tokens, cut at the 150-token limit

> සිංහල අකුරු කීයක් තිබේද? සිංහල භාෂාවේ **16** අකුරු තිබේ. මෙම 16 අකුරු පහත පරිදි ව්‍යුහාත්මකව සැකසී ඇත: 1. **සම්පූර්ණ 16 අකුරු:** * **සම්පූර්ණ 16 අකුරු:** 1, 2, 3, 4, 5, 6,

**Qwen3.5-0.8B** — 150 tokens, cut at the 150-token limit

> සිංහල අකුරු **1000** ක් තිබේ. මෙම අකුරු පහත පරිදි පිහිටා ඇත: * **සුදුරු අකුරු:** 1000 * **සුදුරු අකුරු:** 1000 * **සුදුරු අකුරු:** 1000 * **සුදුරු අකුරු:** 1000 * **සුදුරු අකුරු:**

#### Q3. ගස් වැදගත් වන්නේ ඇයි?

*Source: golden prompt.*

**HelaAI SinhalaLM-V.0.0.1-it (current)** — 150 tokens, cut at the 150-token limit

> ගස් යනු ගස් වලින් වැසී ඇති අතර ඒවා වියළී යයි. එය ගස් වලින් වැසී ඇති අතර ඒවා වියළී යයි. ගස් වලින් වැසී ඇති අතර ඒවා වියළී යයි. ගස් වලින් වැසී ඇති අතර ඒවා වියළී යයි. ගස් වියළී ඇති අතර ගස් වියළී යයි. ගස් වියළී ඇති අතර ගස් වියළී යයි. ගස් වියළී ඇති අතර ගස් වියළී යයි. ගස් වියළී ඇති අතර ගස්

**SinhalaLLM-1.7B v2.1** — 47 tokens, stopped by itself

> ගස් වැදගත් වන්නේ ඒවා පෘථිවියේ ඇති වටිනාම සම්පත වන බැවිනි. ගස් යනු පෘථිවියේ ඇති වටිනාම සම්පත වන අතර එය පෘථිවියේ ඇති වටිනාම සම්පතකි. ගස් යනු පෘථිවියේ ඇති වටිනාම සම්පත වන අතර එය පෘථිවියේ ඇති වටිනාම සම්පතකි.

**Qwen3.5-2B** — 150 tokens, cut at the 150-token limit

> ගස් (Gast) යනු ශරීරයේ **අවශ්‍යතා ප්‍රතිපත්ති (Digestive System)** හි ප්‍රධාන ක්‍රියාවලියක් සහිත ප්‍රධාන ජලය ප්‍රතිශාලකයයි. මෙයට අදාළ වශයෙන් පහත පරිදි වැදගත් වේ: 1. **අවශ්‍යතා ප්‍රතිපත්ති සහිත ප්‍රධ

**Qwen3.5-0.8B** — 150 tokens, cut at the 150-token limit

> **ගස් වැදගත් වන්නේ** **සමාජයේ (Socialist) සමාජයක් (Socialist State) සමාජයක් (Socialist Society) සමාජයක් (Socialist Community) සමාජයක් (Socialist City) සමාජයක් (Socialist Town) සමාජයක් (Socialist Village) සමාජයක් (Socialist Neighborhood) සමාජයක් (Socialic Village) සමාජයක් (Socialic Neighborhood) සමාජයක් (Socialic Village) සමාජයක් (Socialic Neighborhood) සමාජයක්

#### Q4. ශ්රී ලංකාවේ ජාතික ධජය විස්තර කරන්න.

*Source: golden prompt.*

**HelaAI SinhalaLM-V.0.0.1-it (current)** — 33 tokens, stopped by itself

> එය &quot;ප්රමාණවත් තරම් විශාල නොවේ&quot; යන ප්රශ්නය සමඟ සම්බන්ධ වේ.

**SinhalaLLM-1.7B v2.1** — 13 tokens, stopped by itself

> ශ්රී ලංකාවේ ජාතික ධජය වන්නේ සිංහ ධජයයි.

**Qwen3.5-2B** — 150 tokens, cut at the 150-token limit

> ශ්‍රී ලංකාවේ ජාතික ධජය (ශ්‍රී ලංකා ජාතික ධජය) යනු මෙයින් පළමුව භාවිතා කරන ලද ජාතික ධජයයි. මෙය 1948 දී ශ්‍රී ලංකාවට ප්‍රාථමිකව ප්‍රකාශය ලැබූ විට භාවිතා කරන ලද ධජයයි. මෙහි ප්‍රධ

**Qwen3.5-0.8B** — 150 tokens, cut at the 150-token limit

> **ශ්‍රී ලංකාවේ ජාතික ජාතික ධජය (National Wildlife)** ශ්‍රී ලංකාවේ ජාතික ධජය ප්‍රමාණවත්වන ලෙස **ප්‍රමාණවත් ජාතික ධජය (Panthera pardus)** ලෙස හැඳින්වේ. මෙය ආර්ථිකයේ සහ ආර්ථිකයේ සහ ආර්ථිකයේ සහ ආර්ථි

#### Q5. පටිච්චසමුප්පාදය මුලින්ම මෙනෙහි කළේ කොහෙදිද? කෙටියෙන් පිළිතුරු දෙන්න.

*Source: QA-pairs test split.* *Reference answer from the dataset:* බුද්ධත්වයට පත් වූ මුල් සතියේ අජපල් නුග රුක මුලදී

**HelaAI SinhalaLM-V.0.0.1-it (current)** — 150 tokens, cut at the 150-token limit

> ඔබ අදහස් කරන්නේ කුමන ආකාරයේ ප්රශ්නයක්ද? ඔබ අදහස් කරන්නේ කුමන ආකාරයේ ප්රශ්නයක්ද? ඔබ අදහස් කරන්නේ කුමන ආකාරයේ ප්රශ්නයක්ද? ඔබ අදහස් කරන්නේ කුමන ආකාරයේ ප්රශ්නයක්ද? ඔබ අදහස් කරන්නේ කුමන ආකාරයේ ප්රශ්නයක්ද? ඔබ අදහස් කරන්නේ කුමන ආකාරයේ ප්රශ්නයක්ද? ඔබ අදහස් කරන්නේ කුමන ආකාරයේ ප්රශ්නයක්ද? ඔබ අදහස් කරන්නේ කුමන ආකාරයේ ප්රශ්නයක්ද? ඔබ අදහස් කරන්නේ කුමන ආකාරයේ ප්

**SinhalaLLM-1.7B v2.1** — 26 tokens, stopped by itself

> පටිච්චසමුප්පාදය මුලින්ම මෙනෙහි කළේ මහායාන දේවගැතිවර වන මහායාන සුත්තaryයෙනි.

**Qwen3.5-2B** — 150 tokens, cut at the 150-token limit

> පටිච්චසමුප්පාදය මුලින්ම මෙනෙහි කළේ **සිහිනාය** (සිහිනාය) වැනි මාධ්‍යයෙන්. මෙය සිහිනායෙන් කළ හැකි වන්නේ: 1. **සිහිනාය** (සිහිනාය) 2. **සිහිනාය** (සිහිනාය) 3. **සිහිනාය** (සිහිනාය)

**Qwen3.5-0.8B** — 150 tokens, cut at the 150-token limit

> **පටිච්චසමුප්පාදය (Pattisamuppada)** යනු **මෙනෙහි (මේනෙහි)** පිළිතුරු දෙන්නේ නැත. මේනෙහි පිළිතුරු දෙන්නේ නැත. මේනෙහි පිළිතුරු දෙන්නේ නැත. **පිළිතුරු:** * **පටිච්චසමුප්පාදය (Pattisamuppada):** මෙනෙහි පිළිතු

#### Q6. බෞද්ධ ඉගැන්වීම් අනුව, බුදුරදුන් පිරිනිවන් පෑ දින අවසන් වරට සැතපුණු ඉරියව්ව කුමක්ද?

*Source: QA-pairs test split.* *Reference answer from the dataset:* සිංහ සෙය්‍යාවයි.

**HelaAI SinhalaLM-V.0.0.1-it (current)** — 86 tokens, stopped by itself

> බුදුරදුන් පිරිනිවන් පෑවේ බුදුන් වහන්සේ පිරිනිවන් පෑවේ බුදුන් වහන්සේ පිරිනිවන් පෑවේ කවදාදැයි විශ්වාස නැත, නමුත් බුදුරදුන් පිරිනිවන් පෑවේ බුදුන් වහන්සේ පිරිනිවන් පෑවේ කවදාදැයි විශ්වාස නැත.

**SinhalaLLM-1.7B v2.1** — 28 tokens, stopped by itself

> බුදු දහම අනුව, බුදුරදුන් පිරිනිවන් පෑ දින අවසන් වරට සැතපුණු ඉරියව්ව වූයේ "අභිධජ" ඉරියව්වයි.

**Qwen3.5-2B** — 150 tokens, cut at the 150-token limit

> බෞද්ධ ඉගැන්වීම් අනුව බුදුරදුන් පිරිනිවන් පෑ දින අවසන් වරට සැතපුණු ඉරියව්වක් නොමැතිව පවතින බවට පැහැය දැනෙනවා. මෙයට හේතු වශයෙන් පහත කරුණු සැලකිය හැක: 1. **සාමාන්‍ය ඉරියව්වය:** සාමාන්‍ය ජීවිතයේ

**Qwen3.5-0.8B** — 150 tokens, cut at the 150-token limit

> බෞද්ධ ඉගැන්වීම් අනුව, බුදුරදුන් පිරිනිවන් පෑ දින අවසන් වරට සැතපුණු ඉරියව්ව (බුදුරදුන්) **විශ්‍රමයෙන් 1000 ක්** වේ. මෙම සැලසුම පහත පරිදි පහත පරිදි පහත පරිදි පහත පරිදි පහත පරිදි පහත පරිදි

#### Q7. ධර්මයේ 'අකාලික' ගුණය යනු ගැන පැහැදිලි කරන්න?

*Source: QA-pairs test split.* *Reference answer from the dataset:* ඕනෑම කාලයකට ගැලපෙන සහ වහා ප්‍රතිඵල දෙන ගුණයයි.

**HelaAI SinhalaLM-V.0.0.1-it (current)** — 150 tokens, cut at the 150-token limit

> ඔබ අදහස් කරන්නේ කුමක්ද යන්න පැහැදිලි කිරීමට අපට ක්රමයක් නොමැත. ඔබ අදහස් කරන්නේ කුමක්ද යන්න පැහැදිලි කිරීමට අපට ක්රමයක් නොමැත. ඔබ අදහස් කරන්නේ කුමක්ද යන්න පැහැදිලි කිරීමට අපට ක්රමයක් නොමැත. ඔබ අදහස් කරන්නේ කුමක්ද යන්න පැහැදිලි කිරීමට අපට ක්රමයක් නොමැත. ඔබ අදහස් කරන්නේ කුමක්ද යන්න පැහැදිලි කිරීමට අපට ක්රමයක් නොමැත. ඔබ අදහස් කරන්නේ කුමක්ද යන්න පැහැදිලි කිරීමට අපට ක්රමයක් නොමැත. ඔබ අදහස් කරන්නේ කුමක්ද යන්න පැහැදිලි කිරීමට අපට ක්රමයක් නොමැත. ඔබ අදහස

**SinhalaLLM-1.7B v2.1** — 150 tokens, cut at the 150-token limit

> අකාලික යනු, එය සිදු වූ විට, එය නැවත සිදු නොවන බව හෝ එය සිදු වූ විට, එය නැවත සිදු නොවන බව ය. එය, එය සිදු වූ විට, එය නැවත සිදු නොවන බව හෝ එය සිදු වූ විට, එය නැවත සිදු නොවන බව ය. එය, එය සිදු වූ විට, එය නැවත සිදු නොවන බව හෝ එය සිදු වූ විට, එය නැවත සිදු නොවන බව ය. එය, එය සිදු වූ විට, එය නැවත සිදු නොවන බව හෝ එය සිදු වූ විට, එය නැවත සිදු නොවන බව ය. එය, එය සිදු වූ විට, එය නැවත සිදු නොවන බව හෝ එය සිදු වූ විට, එය නැවත සිදු නොවන බව ය. එය, එය සිදු වූ විට, එය නැවත සිදු නොවන බව හෝ එය සිදු වූ විට, එය නැවත සිදු නොවන

**Qwen3.5-2B** — 150 tokens, cut at the 150-token limit

> ධර්මයේ **'අකාලික'** (සිංහල: අකාලික) යනු ජීවිතයේ සියල්ලා සමාන වන, සියල්ලා සමාන වන, සියල්ලා සමාන වන, සියල්ලා සමාන වන, සියල්ලා සමාන වන, සියල්ලා සමාන වන, සියල්ලා සමාන වන, සියල්ලා සමාන වන, සියල්ලා සමාන වන, සියල්

**Qwen3.5-0.8B** — 150 tokens, cut at the 150-token limit

> **ධර්මයේ 'අකාලික' (අකාලික ගුණය) යනු** සාමාජिकයේ (Mathematics) අකාලික ගුණය යනු **පරමාණු (Numbers) අනුකල (Algebra) සංකල්පය** (Algebraic Expression) හි **අනුකල ගුණය (Summation)** හි **අනුකල ගුණය (Product)** හි **අනුකල ගුණය (Product)** හි **අනුකල ගුණය (Product

#### Q8. බෞද්ධ ඉගැන්වීම් අනුව, සමාධි බුදු පිළිමය නෙළා ඇත්තේ කුමන ගලින්ද?

*Source: QA-pairs test split.* *Reference answer from the dataset:* කළු ගලින් ().

**HelaAI SinhalaLM-V.0.0.1-it (current)** — 20 tokens, stopped by itself

> මම හිතන්නේ එය බුදු පිළිමයක් විය හැකි යැයි මම සිතමි.

**SinhalaLLM-1.7B v2.1** — 26 tokens, stopped by itself

> බුදු පිළිමය නෙළීම සඳහා වන අවසරයේ සඳහන් වන්නේ, එය නෙළා ඇත්තේ කුමන ගලින්ද? යන ප්‍රශ්නයයි.

**Qwen3.5-2B** — 150 tokens, cut at the 150-token limit

> බෞද්ධ ඉගැන්වීම් අනුව, සමාධි (Concentration) බුදු පිළිමය නෙළා ඇත්තේ **සමාධි ගල** (Concentration Path) මගින්ය. බුද්ධිමානාගාරයේ සමාධි පිළිමයන් ප්‍රධාන වශයෙන් පහත පරිදි ව්‍යායාමිත වේ: 1. **සමාධි ගල (Concentration Path):** මෙහි ප්‍

**Qwen3.5-0.8B** — 150 tokens, cut at the 150-token limit

> බෞද්ධ ඉගැන්වීම් අනුව සමාධි බුදු පිළිමය (මුද්‍රුණු පිළිමය) නෙළා ඇත්තේ **කොට්‍රා (කොට්‍රා)** පිළිමයට ප්‍රතික්‍රියා වේ. මෙම ප්‍රතික්‍රියාව පහත පරිදි සිදුරුයි: 1. **කොට්‍රා පිළිමය

#### Q9. ඩයෝඩයක් හරහා ධාරාව ගමන් කරන්නේ කුමන අවස්තාවේදීද?

*Source: QA-pairs test split.* *Reference answer from the dataset:* ඇනෝඩයට ධන (+) විභවයක් ලැබෙන සේ පූර්ව නැඹුරුකරණය ( ) කළ විට පමණි.

**HelaAI SinhalaLM-V.0.0.1-it (current)** — 50 tokens, stopped by itself

> ධාරාව ගමන් කරන්නේ කුමන අවස්ථාවේදීද යන්න මත රඳා පවතී. ඔබ ධාරාව ගමන් කරන්නේ කුමන අවස්ථාවේදීද යන්න මත රඳා පවතී.

**SinhalaLLM-1.7B v2.1** — 17 tokens, stopped by itself

> ඩයෝඩයක් හරහා ධාරාව ගමන් කරන්නේ ධාරාව ගලා යන විට පමණි.

**Qwen3.5-2B** — 150 tokens, cut at the 150-token limit

> ඩයෝඩයක් හරහා ධාරාව ගමන් කරන්නේ **විශේෂිත ධාරා ප්‍රවාහයක් (Current)** සහිතව පමණි. මෙය සාමාන්‍යවලට "විශ්‍රාමයක්" (Rest) සහ "ප්‍රවාහයක්" (Flow) අතර පැහැය දැක්විය හැක. ඩයෝඩයක් හරහා ධාරාව ගමන් කරන විට ප්‍රධාන වශ

**Qwen3.5-0.8B** — 150 tokens, cut at the 150-token limit

> ඩයෝඩයක් (Inductor) හරහා ධාරාව ගමන් කරන්නේ **මුලින්ම ධාරාව (Current) සමතුලිතයක් (Circulation)** සමතුලිතයක් (Circulation) සමතුලිතයක් (Circulation) සමතුලිතයක් (Circulation) සමතුලිතයක් (Circulation) සමතුලිතයක් (Circulation) සමතුලිතයක් (Circulation) සමතුලිතයක් (Circulation) සමතුලිතයක් (

#### Q10. ප්‍රිය වචනය භාවිතා කිරීමේ වැදගත්කම විස්තර කරන්න?

*Source: QA-pairs test split.* *Reference answer from the dataset:* සමාජයේ සාමය සහ අන්‍යෝන්‍ය සුහදතාවය වර්ධනය වීම.

**HelaAI SinhalaLM-V.0.0.1-it (current)** — 43 tokens, stopped by itself

> එය භාවිතා කරන පුද්ගලයා එය භාවිතා කරන ආකාරය විස්තර කරයි. එය භාවිතා කරන පුද්ගලයා එය භාවිතා කරන ආකාරය විස්තර කරයි.

**SinhalaLLM-1.7B v2.1** — 150 tokens, cut at the 150-token limit

> ප්‍රිය වචනය භාවිතා කිරීමේ වැදගත්කම 1. සෞඛ්‍යය වචනවල අර්ථය සහ අර්ථය වැඩි දියුණු කිරීම: ප්‍රිය වචනය භාවිතා කිරීම වචනවල අර්ථය සහ අර්ථය වැඩි දියුණු කිරීමට උපකාරී වේ. 2. සෞඛ්‍යය වචනවල ස්වරය සහ ස්වරය වැඩි දියුණු කිරීම: ප්‍රිය වචනය භාවිතා කිරීමෙන් වචනවල ස්වරය සහ ස්වරය වැඩි දියුණු කළ හැකිය. 3. මෘදු වචනවල ස්වරය සහ ස්වරය වැඩි දියුණු කිරීම: ප්‍රිය වචනය භාවිතා කිරීමෙන් වචනවල ස්වරය සහ ස්වරය වැඩි දියුණු කළ හැකිය. 4. සෞඛ්‍යය වචනවල ස්වරය සහ ස්වරය වැඩි දියුණු කිරීම: ප්‍රිය වචනය භාවිතා කිරීමෙන් වචනවල ස්වරය සහ ස්වරය වැඩි දියුණු කළ හැකිය. පහයි. වචනවල ස්වරය සහ ස්වරය

**Qwen3.5-2B** — 150 tokens, cut at the 150-token limit

> ප්‍රිය වචනය (Preposition) භාවිතා කිරීමේ වැදගත්කම ඉතා විශාලය. මෙය සරලව කියන්නේ "විශේෂයෙන්" හෝ "විශ්ඛේපයෙන්" යැයි සැලකිය හැක. ප්‍රිය වචනයක් භාවිතා නොකිරීම නිසා පදනම් වන සම්පත් හෝ සිදුවන ක්‍රියා ප්‍රතිපත්ති වැට

**Qwen3.5-0.8B** — 150 tokens, cut at the 150-token limit

> **ප්‍රිය වචනය (Proverb)** යනු සිංහල ප්‍රමාණවල (සංකල්පය) ආරම්භක ප්‍රමාණයක් ලෙස ප්‍රකාශන ලෙස පරිවර්තනය කළ හැකි ප්‍රමාණයකි. මෙය සිංහල ප්‍රමාණයේ ආරම්භක ප්‍රමාණයක් ලෙස ප්‍රකාශන ලෙස ප

### 6. Models that were not screened

- **Gemma 4 E2B** — its q4 download is about 3.1 GB, most of which is the per-layer embedding table, and it did not fit the free disk space for screening. Its tokenizer needs 3.16 tokens per Sinhala word.
- **SinLlama (Llama-3-8B)** — the strongest model in the SinhalaLLM author's own comparison, but at 8B parameters it needs about 4.5 GB or more at 4-bit, and its license is not stated in the repository metadata. It remains a quality reference rather than a browser candidate.
- **Tiny Aya (3.35B)** — gated on Hugging Face and released under a non-commercial license.
- **Gemma 3 1B** — gated on Hugging Face, and it scored about chance level (25.7%) on the SinhalaMMLU shared-task screening set.
- **Qwen3.5-4B Sinhala fine-tunes from the SinhalaMMLU shared task** — about 9.3 GB of weights, and the question-answering model states no license.

### 7. Decision needed

Our recommendation is **SinhalaLLM-1.7B**. It is the only screened model whose answers address the questions, its tokenizer is the most efficient for Sinhala (1.76 tokens per word, against 2.66 for SinhalaLM and 4.92 for Qwen3.5), it keeps the ZWJ, it uses the Llama architecture that the existing pipeline already exports, and it is released under Apache-2.0. The downside of this choice is its maturity: the model was published on 2026-10-13, its benchmark figures come from its author, and its factual accuracy is still limited.

Before the project switches, the following decisions are yours:

1. Which model to continue with.
2. Whether the proposed size, memory and speed targets in Section 4 are acceptable.
3. Freeing about 20 GB of disk space before the conversion work starts.
