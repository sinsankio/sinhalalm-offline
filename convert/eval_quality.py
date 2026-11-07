import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from sinhalallm import STOP_IDS, load_golden, model_dir, random_ids, store_path

RELATIVE_TOLERANCE = 1e-4
FLORES = {
    "repo": "openlanguagedata/flores_plus",
    "revision": "1147e2dc59f37fbb5c22fd8478770de02cd399f9",
    "file": "devtest/sin_Sinh.jsonl",
    "sha256": "111d661ffe8bb7f3ee3f6a4d8d644a8f28856b476df4ef74890f7bc97b1ab890",
}
VARIANTS = {
    "fp32": "fp32/model.onnx",
    "fp16": "fp16/onnx/model_fp16.onnx",
    "q8": "q8/onnx/model_quantized.onnx",
    "q4": "q4/onnx/model_q4.onnx",
    "q4-tied": "q4-tied/onnx/model_q4.onnx",
    "q4-head": "q4-head/onnx/model_q4.onnx",
    "q4-emb": "q4-emb/onnx/model_q4.onnx",
    "q4-asym": "q4-asym/onnx/model_q4.onnx",
    "q4-b16": "q4-b16/onnx/model_q4.onnx",
    "q4-hqq": "q4-hqq/onnx/model_q4.onnx",
    "q4-v0": "q4-v0/onnx/model_q4.onnx",
    "q4-v0-small": "q4-v0-small/onnx/model_q4.onnx",
}
TEXT_START_ID = 0
DEFAULT_VARIANT = "q4-v0-small"
REFERENCE_GGUF = "SinhalaLLM-v2.1-Q4_K_M.gguf"


class OnnxCausalLM:
    def __init__(self, path: Path):
        options = ort.SessionOptions()
        options.enable_cpu_mem_arena = False
        self.session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        self.inputs = {item.name: item for item in self.session.get_inputs()}
        self.past_names = [name for name in self.inputs if name.startswith("past_key_values.")]
        self.present_names = [
            name.replace("past_key_values", "present") for name in self.past_names
        ]
        heads, head_dim = (
            self.inputs[self.past_names[0]].shape[1],
            self.inputs[self.past_names[0]].shape[3],
        )
        dtype = np.float16 if "float16" in self.inputs[self.past_names[0]].type else np.float32
        self.empty_past = np.zeros((1, heads, 0, head_dim), dtype=dtype)

    def forward(self, ids: list[int], past: dict | None = None) -> tuple[np.ndarray, dict]:
        past_length = 0 if past is None else next(iter(past.values())).shape[2]
        feeds = {
            "input_ids": np.array([ids], dtype=np.int64),
            "attention_mask": np.ones((1, past_length + len(ids)), dtype=np.int64),
        }
        if "position_ids" in self.inputs:
            positions = np.arange(past_length, past_length + len(ids), dtype=np.int64)
            feeds["position_ids"] = positions[None]
        for name in self.past_names:
            feeds[name] = self.empty_past if past is None else past[name]
        outputs = self.session.run(["logits", *self.present_names], feeds)
        presents = dict(zip(self.past_names, outputs[1:], strict=True))
        return outputs[0][0].astype(np.float32), presents

    def greedy(self, ids: list[int], max_new_tokens: int) -> list[int]:
        logits, past = self.forward(ids)
        generated = []
        for _ in range(max_new_tokens):
            next_id = int(logits[-1].argmax())
            generated.append(next_id)
            if next_id in STOP_IDS:
                break
            logits, past = self.forward([next_id], past)
        return generated


def compare(logits: np.ndarray, reference: np.ndarray) -> dict:
    diff = np.abs(logits - reference).max(axis=1)
    relative = diff / np.abs(reference).max(axis=1)
    return {
        "max_abs": float(diff.max()),
        "max_abs_position": int(diff.argmax()),
        "max_relative": float(relative.max()),
        "top1_identical": bool((logits.argmax(axis=1) == reference.argmax(axis=1)).all()),
    }


def gate2(model_path: Path) -> dict:
    golden = load_golden()
    expected = np.load(store_path("golden", "reference_logits.npz"))
    model = OnnxCausalLM(model_path)
    cases, mismatches = {}, []
    for case in golden["cases"]:
        logits, _ = model.forward(case["input_ids"])
        cases[case["key"]] = compare(logits, expected[case["key"]])
        if model.greedy(case["input_ids"], golden["max_new_tokens"]) != case["greedy_ids"]:
            mismatches.append(case["key"])
    spec = golden["random"]
    logits, _ = model.forward(random_ids(spec))
    cases["random"] = compare(logits[spec["positions"]], expected["random"])
    max_relative = max(item["max_relative"] for item in cases.values())
    top1 = all(item["top1_identical"] for item in cases.values())
    return {
        "model": str(model_path),
        "relative_tolerance": RELATIVE_TOLERANCE,
        "max_abs_logit_diff": max(item["max_abs"] for item in cases.values()),
        "max_relative_logit_diff": max_relative,
        "top1_identical": top1,
        "greedy_mismatches": mismatches,
        "cases": cases,
        "passed": max_relative < RELATIVE_TOLERANCE and top1 and not mismatches,
    }


def flores_sentences() -> list[str]:
    path = Path(
        hf_hub_download(
            FLORES["repo"],
            FLORES["file"],
            repo_type="dataset",
            revision=FLORES["revision"],
            local_dir=store_path("hf", "datasets", "flores_plus"),
        )
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != FLORES["sha256"]:
        raise RuntimeError(f"{FLORES['file']}: sha256 {digest} != {FLORES['sha256']}")
    return [json.loads(line)["text"] for line in path.open(encoding="utf-8")]


def perplexity(model: OnnxCausalLM, sentences: list[str]) -> dict:
    tokenizer = Tokenizer.from_file(str(model_dir() / "tokenizer.json"))
    nll, tokens, chars, per_sentence = 0.0, 0, 0, []
    started = time.perf_counter()
    for sentence in sentences:
        ids = tokenizer.encode(sentence, add_special_tokens=False).ids
        logits, _ = model.forward([TEXT_START_ID, *ids[:-1]])
        shifted = logits - logits.max(axis=1, keepdims=True)
        log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        sentence_nll = -float(log_probs[np.arange(len(ids)), ids].sum())
        per_sentence.append([round(sentence_nll, 5), len(ids), len(sentence)])
        nll += sentence_nll
        tokens += len(ids)
        chars += len(sentence)
    return {
        "sentences": len(sentences),
        "tokens": tokens,
        "characters": chars,
        "perplexity": math.exp(nll / tokens),
        "bits_per_character": nll / math.log(2) / chars,
        "prefill_tokens_per_second": tokens / (time.perf_counter() - started),
        "per_sentence_nll_tokens_chars": per_sentence,
    }


def agreement(model: OnnxCausalLM, questions: int) -> dict | None:
    if questions == 0:
        return None
    golden = load_golden()
    expected = np.load(store_path("golden", "reference_logits.npz"))
    cases = golden["cases"][:questions]
    top1, positions, exact, prefix = 0, 0, 0, []
    generated = 0
    started = time.perf_counter()
    for case in cases:
        logits, _ = model.forward(case["input_ids"])
        top1 += int((logits.argmax(axis=1) == expected[case["key"]].argmax(axis=1)).sum())
        positions += len(case["input_ids"])
        ids = model.greedy(case["input_ids"], golden["max_new_tokens"])
        generated += len(ids)
        exact += ids == case["greedy_ids"]
        common = next(
            (i for i, (a, b) in enumerate(zip(ids, case["greedy_ids"], strict=False)) if a != b),
            min(len(ids), len(case["greedy_ids"])),
        )
        prefix.append(common)
    return {
        "questions": len(cases),
        "prompt_top1_agreement": top1 / positions,
        "greedy_exact_matches": exact,
        "greedy_common_prefix_tokens": prefix,
        "generation_tokens_per_second": generated / (time.perf_counter() - started),
    }


def rq1(variant: str, sentences: int | None, questions: int) -> dict:
    path = store_path("onnx", *VARIANTS[variant].split("/"))
    data_bytes = sum(item.stat().st_size for item in path.parent.iterdir() if item.is_file())
    model = OnnxCausalLM(path)
    texts = flores_sentences()[:sentences]
    return {
        "variant": variant,
        "file_bytes": data_bytes,
        "flores": {"revision": FLORES["revision"], "file": FLORES["file"]},
        "perplexity": perplexity(model, texts),
        "agreement": agreement(model, questions),
    }


def reference_gguf(sentences: int | None) -> dict:
    import torch
    from transformers import LlamaForCausalLM

    from sinhalallm import REPO, REVISION

    path = Path(
        hf_hub_download(REPO, REFERENCE_GGUF, revision=REVISION, local_dir=store_path("hf", "gguf"))
    )
    model = LlamaForCausalLM.from_pretrained(
        path.parent, gguf_file=path.name, dtype=torch.float32
    ).eval()
    tokenizer = Tokenizer.from_file(str(model_dir() / "tokenizer.json"))
    per_sentence, nll, tokens, chars = [], 0.0, 0, 0
    with torch.no_grad():
        for sentence in flores_sentences()[:sentences]:
            ids = tokenizer.encode(sentence, add_special_tokens=False).ids
            log_probs = model(torch.tensor([[TEXT_START_ID, *ids[:-1]]])).logits[0].log_softmax(-1)
            sentence_nll = -float(log_probs[torch.arange(len(ids)), torch.tensor(ids)].sum())
            per_sentence.append([round(sentence_nll, 5), len(ids), len(sentence)])
            nll, tokens, chars = nll + sentence_nll, tokens + len(ids), chars + len(sentence)
    return {
        "variant": "gguf-q4_k_m",
        "file": REFERENCE_GGUF,
        "rope_theta": model.config.rope_theta,
        "perplexity": {
            "sentences": len(per_sentence),
            "tokens": tokens,
            "characters": chars,
            "perplexity": math.exp(nll / tokens),
            "bits_per_character": nll / math.log(2) / chars,
            "per_sentence_nll_tokens_chars": per_sentence,
        },
    }


def quality_gate(variant: str) -> dict:
    results = store_path("rq1")
    candidate = json.loads((results / f"{variant}.json").read_text())["perplexity"]
    reference = json.loads((results / "gguf-q4_k_m.json").read_text())["perplexity"]
    count = min(len(candidate["per_sentence_nll_tokens_chars"]), reference["sentences"])

    def subset(scores: dict) -> float:
        rows = scores["per_sentence_nll_tokens_chars"][:count]
        return math.exp(sum(row[0] for row in rows) / sum(row[1] for row in rows))

    candidate_ppl, reference_ppl = subset(candidate), subset(reference)
    return {
        "variant": variant,
        "reference": REFERENCE_GGUF,
        "sentences": count,
        "perplexity": candidate_ppl,
        "reference_perplexity": reference_ppl,
        "passed": candidate_ppl <= reference_ppl,
    }


def run_rq1(args: argparse.Namespace) -> int:
    report = rq1(args.variant, args.sentences, args.questions)
    output = store_path("rq1")
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{args.variant}.json").write_text(json.dumps(report, indent=2) + "\n")
    ppl, agree = report["perplexity"], report["agreement"]
    summary = f"rq1 {args.variant}: ppl={ppl['perplexity']:.3f} bpc={ppl['bits_per_character']:.4f}"
    if agree:
        summary += (
            f" top1={agree['prompt_top1_agreement']:.3f}"
            f" greedy={agree['greedy_exact_matches']}/{agree['questions']}"
        )
    print(summary)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Gate 2 and RQ1 quality evaluation.")
    parser.add_argument("target", choices=("gate2", "rq1", "reference-gguf", "quality-gate"))
    parser.add_argument("--variant", choices=VARIANTS, default=DEFAULT_VARIANT)
    parser.add_argument("--sentences", type=int, help="limit the FLORES+ sentences")
    parser.add_argument("--questions", type=int, default=10, help="golden questions to generate")
    args = parser.parse_args()
    if args.target == "rq1":
        return run_rq1(args)
    if args.target == "reference-gguf":
        report = reference_gguf(args.sentences)
        output = store_path("rq1")
        output.mkdir(parents=True, exist_ok=True)
        (output / "gguf-q4_k_m.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"reference {REFERENCE_GGUF}: ppl={report['perplexity']['perplexity']:.3f}")
        return 0
    if args.target == "quality-gate":
        report = quality_gate(args.variant)
        name = (
            "quality-gate.json"
            if args.variant == DEFAULT_VARIANT
            else f"quality-gate-{args.variant}.json"
        )
        store_path("rq1", name).write_text(json.dumps(report, indent=2) + "\n")
        status = "PASS" if report["passed"] else "FAIL"
        print(
            f"quality gate {report['variant']}: {status} ppl={report['perplexity']:.3f} "
            f"reference={report['reference_perplexity']:.3f} on {report['sentences']} sentences"
        )
        return 0 if report["passed"] else 1
    report = gate2(store_path("onnx", "fp32", "model.onnx"))
    path = store_path("onnx", "fp32", "gate2.json")
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    status = "PASS" if report["passed"] else "FAIL"
    print(
        f"gate 2: {status} max relative diff={report['max_relative_logit_diff']:.3e} "
        f"max|logit diff|={report['max_abs_logit_diff']:.3e} "
        f"top-1 identical={report['top1_identical']} "
        f"greedy mismatches={len(report['greedy_mismatches'])}"
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
