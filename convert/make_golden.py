import argparse
import json
import sys

import numpy as np
import torch

from helaai import (
    MODELS,
    ROOT,
    load_reference_model,
    load_tokenizer,
    prompt_ids,
    reference_greedy,
)

GOLDEN_DIR = ROOT / "golden"
PROMPTS_PATH = GOLDEN_DIR / "prompts.json"
REFERENCE_PATH = GOLDEN_DIR / "reference.json"
LOGITS_DIR = GOLDEN_DIR / "generated"
MAX_NEW_TOKENS = 40
TOP_K = 5


def build_reference() -> tuple[dict, dict[str, np.ndarray]]:
    torch.manual_seed(0)
    prompts = json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))
    tokenizer = load_tokenizer()
    cases, logits_by_key = [], {}
    for model, texts in prompts.items():
        gpt = load_reference_model(model)
        for index, text in enumerate(texts):
            ids = prompt_ids(model, text)
            with torch.no_grad():
                logits, _ = gpt(torch.tensor([ids]))
            last = logits[0, -1]
            greedy = reference_greedy(gpt, ids, MAX_NEW_TOKENS)
            key = f"{model}_{index}"
            logits_by_key[key] = logits[0].numpy()
            cases.append(
                {
                    "key": key,
                    "model": model,
                    "prompt": text,
                    "input_ids": ids,
                    "input_tokens": [tokenizer.itos.get(i, "<UNK>") for i in ids],
                    "last_top_ids": last.topk(TOP_K).indices.tolist(),
                    "greedy_ids": greedy,
                    "greedy_text": tokenizer.decode(greedy),
                }
            )
    reference = {
        "sources": {
            name: {k: spec[k] for k in ("repo", "revision")} for name, spec in MODELS.items()
        },
        "max_new_tokens": MAX_NEW_TOKENS,
        "cases": cases,
    }
    return reference, logits_by_key


def dump(reference: dict) -> str:
    return json.dumps(reference, ensure_ascii=False, indent=2) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate golden reference outputs.")
    parser.add_argument("--check", action="store_true", help="fail if outputs differ")
    args = parser.parse_args()

    reference, logits = build_reference()
    text = dump(reference)

    if args.check:
        if not REFERENCE_PATH.exists() or REFERENCE_PATH.read_text(encoding="utf-8") != text:
            print(f"{REFERENCE_PATH.relative_to(ROOT)} is out of date", file=sys.stderr)
            return 1
        print(f"{REFERENCE_PATH.relative_to(ROOT)} is up to date")
    else:
        REFERENCE_PATH.write_text(text, encoding="utf-8")
        print(f"wrote {REFERENCE_PATH.relative_to(ROOT)}")

    LOGITS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(LOGITS_DIR / "reference_logits.npz", **logits)
    return 0


if __name__ == "__main__":
    sys.exit(main())
