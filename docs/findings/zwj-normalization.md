# ZWJ Normalization: Matching the Input to the Training Data

SinhalaLLM-1.7B learned Sinhala conjuncts with the zero-width joiner (ZWJ), but its tokenizer has no normalizer, so a question typed without the ZWJ reaches the model in a form it rarely saw ([sinhalallm-inspection.md](sinhalallm-inspection.md), S2). The obvious fix is to add the ZWJ wherever a virama is followed by ර or ය. In this document, we show why that rule is wrong for this model, and which rule the application uses instead.

### 1. The vocabulary defines the clusters

The model card states that a normalizer restored the ZWJ in 16 conjunct clusters before training, but it does not list them. The vocabulary answers the question instead. Of the 765 vocabulary tokens that contain a ZWJ, all but four contain it inside one of exactly 15 clusters:

| Rakaransaya (`්‍ර`) | Yansaya (`්‍ය`) |
|---|---|
| ප්‍ර (254 tokens), ත්‍ර (129), ක්‍ර (107), ග්‍ර (47), ශ්‍ර (40), ද්‍ර (32), ඝ්‍ර (2) | ධ්‍ය (46), ද්‍ය (41), ශ්‍ය (18), ජ්‍ය (14), ඛ්‍ය (11), භ්‍ය (8), ට්‍ය (7), ෂ්‍ය (5) |

Two properties make this list decisive:

- **The 15 clusters never appear without the ZWJ.** For each of them, the vocabulary contains no token with the same consonants joined without the ZWJ.
- **Other clusters appear only without the ZWJ.** Clusters such as ව්ය (54 tokens), ත්ය (52), ර්ය (49), න්ය (27), ට්ර (22) and බ්ර (14) occur in the vocabulary, but never with the ZWJ.

In other words, the training data was normalized cluster by cluster, and the model learned 15 clusters with the ZWJ and all others without it. The 16th cluster of the normalizer most likely occurred too rarely to form a vocabulary token. Real Sinhala text does not follow this split: in the tokenizer corpus, ව්‍ය appears 335 times and ත්‍ය 322 times with the ZWJ, which is the standard spelling.

### 2. Three candidate rules

We compared three input rules on the 10,000-sentence tokenizer corpus from gate 3 ([tokenizer-parity.md](tokenizer-parity.md)), counting tokens with the transformers.js tokenizer:

- **A, blanket** — add the ZWJ after every virama that is followed by ර or ය.
- **B, learned clusters** — add the ZWJ only in the 15 learned clusters.
- **C, normalization** — B, and additionally remove the ZWJ from every other `්ර` and `්ය` cluster, which reproduces the normalization of the training data.

| Rule | Tokens on the corpus (193,322 unchanged) | Change | Sentences changed | Tokens on 4,387 sentences with the ZWJ removed (115,409 unchanged) | Change |
|---|---|---|---|---|---|
| A, blanket | 193,824 | +0.26% | 470 | 106,717 | −7.5% |
| B, learned clusters | 193,119 | −0.11% | 95 | 103,439 | −10.4% |
| C, normalization | 190,278 | −1.57% | 1,198 | 103,439 | −10.4% |

The second pair of columns simulates the offline reader's keyboard: every ZWJ was removed from the 4,387 corpus sentences that contained one, and each rule was applied to the result. For comparison, the same sentences with their original ZWJ needed 107,676 tokens.

The blanket rule makes the corpus longer, since it creates ZWJ clusters that the vocabulary never learned. Rule C gives the fewest tokens on both sets, and on text typed without the ZWJ, it even needs fewer tokens than the original text with its standard spelling. This is because the original text contains ZWJ clusters, such as ව්‍ය, that the model only learned without the ZWJ.

### 3. Decision

The application normalizes the reader's question with rule C, implemented as `normalizeZwj` in `web/src/text/zwj.js`. The rule changes only the text sent to the model; the reader's question is displayed as it was typed. Its unit tests cover the 15 learned clusters, clusters where the ZWJ is removed, a repaya, the zero-width non-joiner, idempotence and the tokenizer itself: for every learned cluster, the joined form needs no more tokens than the unjoined one, and text typed without the ZWJ is tokenized exactly like the golden text with it.

Fewer tokens are a sign that the input matches the training data, not proof that the answers improve. Whether the normalization improves the answers is measured in RQ3 with native-speaker ratings.

### 4. Open question for the output

The same split affects what the model writes. Since it learned clusters such as ව්ය and ත්ය without the ZWJ, it most likely writes them without the ZWJ, as in සත්ය, which renders with a visible virama rather than as the standard conjunct. Restoring the ZWJ in the output for these clusters is a candidate step for RQ3, and it needs native-speaker evaluation before it is added, since some sequences, such as a repaya in කාර්ය, follow different rules.
