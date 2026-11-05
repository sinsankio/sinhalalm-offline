import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

from sinhalallm import STOP_IDS, load_golden, random_ids, store_path

RELATIVE_TOLERANCE = 1e-4


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
        self.empty_past = np.zeros((1, heads, 0, head_dim), dtype=np.float32)

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
        return outputs[0][0], presents

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


def main() -> int:
    parser = argparse.ArgumentParser(description="Gate 2 and RQ1 quality evaluation.")
    parser.add_argument("target", choices=("gate2",))
    parser.parse_args()
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
