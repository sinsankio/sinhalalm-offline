import argparse
import json
import sys

import numpy as np
import torch
import transformers
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

from sinhalallm import STOP_IDS, load_golden, model_dir, random_ids, raw_config, store_path

LOGIT_TOLERANCE = 1e-4
EXPECTED_PARAMETERS = 1_740_474_368


def untranslated_config() -> LlamaConfig:
    return LlamaConfig.from_pretrained(model_dir())


def translated_config() -> LlamaConfig:
    raw = raw_config()
    rope = raw.get("rope_parameters") or {}
    if rope.get("rope_type", "default") != "default":
        raise ValueError(f"unsupported rope_type {rope['rope_type']}")
    config = untranslated_config()
    config.rope_theta = float(rope.get("rope_theta", raw.get("rope_theta", config.rope_theta)))
    config.rope_scaling = None
    config.use_cache = True
    config.eos_token_id = list(STOP_IDS)
    config.pad_token_id = STOP_IDS[1]
    return config


def load_export_model(config: LlamaConfig, dtype: torch.dtype = torch.float32) -> LlamaForCausalLM:
    return LlamaForCausalLM.from_pretrained(model_dir(), config=config, dtype=dtype).eval()


@torch.no_grad()
def logits_for(model: LlamaForCausalLM, ids: list[int]) -> np.ndarray:
    return model(torch.tensor([ids])).logits[0].numpy()


@torch.no_grad()
def greedy(model: LlamaForCausalLM, ids: list[int], max_new_tokens: int) -> list[int]:
    batch = torch.tensor([ids])
    output = model.generate(
        batch,
        attention_mask=torch.ones_like(batch),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=STOP_IDS,
        pad_token_id=STOP_IDS[1],
    )
    return output[0, len(ids) :].tolist()


def run() -> dict:
    golden = load_golden()
    expected = np.load(store_path("golden", "reference_logits.npz"))
    config = translated_config()
    model = load_export_model(config)

    diffs = {
        case["key"]: float(
            np.abs(logits_for(model, case["input_ids"]) - expected[case["key"]]).max()
        )
        for case in golden["cases"]
    }
    spec = golden["random"]
    random_logits = logits_for(model, random_ids(spec))[spec["positions"]]
    diffs["random"] = float(np.abs(random_logits - expected["random"]).max())

    mismatches = [
        case["key"]
        for case in golden["cases"]
        if greedy(model, case["input_ids"], golden["max_new_tokens"]) != case["greedy_ids"]
    ]

    untranslated = untranslated_config()
    translated_rope = model.model.rotary_emb
    model.model.rotary_emb = LlamaRotaryEmbedding(config=untranslated)
    control_logits = logits_for(model, random_ids(spec))[spec["positions"]]
    control_diff = float(np.abs(control_logits - expected["random"]).max())
    model.model.rotary_emb = translated_rope

    parameters = sum(p.numel() for p in model.parameters())
    max_diff = max(diffs.values())
    passed = (
        max_diff < LOGIT_TOLERANCE
        and not mismatches
        and control_diff >= LOGIT_TOLERANCE
        and parameters == EXPECTED_PARAMETERS
    )

    output_dir = store_path("gate1")
    output_dir.mkdir(parents=True, exist_ok=True)
    config.save_pretrained(output_dir)
    return {
        "reference_transformers": golden["source"]["transformers"],
        "export_transformers": transformers.__version__,
        "rope_theta": {"translated": config.rope_theta, "untranslated": untranslated.rope_theta},
        "tolerance": LOGIT_TOLERANCE,
        "max_abs_logit_diff": max_diff,
        "logit_diff_by_case": diffs,
        "greedy_mismatches": mismatches,
        "negative_control_max_abs_logit_diff": control_diff,
        "parameters": parameters,
        "passed": passed,
    }


def main() -> int:
    argparse.ArgumentParser(description="Gate 1: reference vs export toolchain.").parse_args()
    report = run()
    path = store_path("gate1", "parity.json")
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    status = "PASS" if report["passed"] else "FAIL"
    print(
        f"gate 1: {status} max|logit diff|={report['max_abs_logit_diff']:.3e} "
        f"greedy mismatches={len(report['greedy_mismatches'])} "
        f"negative control={report['negative_control_max_abs_logit_diff']:.3e} "
        f"parameters={report['parameters']:,}"
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
