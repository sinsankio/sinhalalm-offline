import argparse
import json
import sys

import torch
from safetensors.torch import load_file
from transformers import GenerationConfig, LlamaConfig, LlamaForCausalLM

from helaai import (
    ARCH,
    BOS_ID,
    EOS_ID,
    ROOT,
    fetch,
    load_reference_model,
    store_path,
)

GOLDEN_PATH = ROOT / "golden" / "reference.json"
LOGIT_TOLERANCE = 1e-4
RANDOM_SEQ_LEN = 512
UNIQUE_PARAMETERS = 32_850_432
PAD_ID = 0


def llama_config() -> LlamaConfig:
    embed = ARCH["embed_dim"]
    return LlamaConfig(
        vocab_size=ARCH["vocab_size"],
        hidden_size=embed,
        intermediate_size=int(embed * 8 / 3),
        num_hidden_layers=ARCH["n_layers"],
        num_attention_heads=ARCH["n_heads"],
        num_key_value_heads=ARCH["n_heads"],
        max_position_embeddings=ARCH["block_size"],
        rope_theta=10000.0,
        rms_norm_eps=torch.finfo(torch.float32).eps,
        tie_word_embeddings=True,
        attention_bias=False,
        mlp_bias=False,
        bos_token_id=BOS_ID,
        eos_token_id=EOS_ID,
        pad_token_id=PAD_ID,
        torch_dtype="float32",
    )


def convert_state(src: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    embed = ARCH["embed_dim"]
    dst = {
        "model.embed_tokens.weight": src["tok_emb.weight"],
        "model.norm.weight": src["ln_f.weight"],
    }
    for i in range(ARCH["n_layers"]):
        block, layer = f"blocks.{i}", f"model.layers.{i}"
        q, k, v = src[f"{block}.attn.qkv.weight"].split(embed, dim=0)
        dst |= {
            f"{layer}.self_attn.q_proj.weight": q,
            f"{layer}.self_attn.k_proj.weight": k,
            f"{layer}.self_attn.v_proj.weight": v,
            f"{layer}.self_attn.o_proj.weight": src[f"{block}.attn.proj.weight"],
            f"{layer}.mlp.gate_proj.weight": src[f"{block}.ffn.w1.weight"],
            f"{layer}.mlp.up_proj.weight": src[f"{block}.ffn.w2.weight"],
            f"{layer}.mlp.down_proj.weight": src[f"{block}.ffn.w3.weight"],
            f"{layer}.input_layernorm.weight": src[f"{block}.ln1.weight"],
            f"{layer}.post_attention_layernorm.weight": src[f"{block}.ln2.weight"],
        }
    return {name: tensor.clone().contiguous() for name, tensor in dst.items()}


def check_dropped(src: dict[str, torch.Tensor], reference: torch.nn.Module) -> None:
    if not torch.equal(src["lm_head.weight"], src["tok_emb.weight"]):
        raise AssertionError("lm_head.weight is not identical to tok_emb.weight")
    for i, block in enumerate(reference.blocks):
        rope = block.attn.rope
        for name in ("cos", "sin"):
            if not torch.equal(src[f"blocks.{i}.attn.rope.{name}"], getattr(rope, name)):
                raise AssertionError(f"blocks.{i}.attn.rope.{name} differs from recomputed table")


def build_llama(model: str) -> tuple[LlamaForCausalLM, torch.nn.Module]:
    src = load_file(fetch(model, "model.safetensors"))
    reference = load_reference_model(model)
    check_dropped(src, reference)
    llama = LlamaForCausalLM(llama_config())
    state = convert_state(src)
    llama.load_state_dict(state | {"lm_head.weight": state["model.embed_tokens.weight"]})
    if llama.lm_head.weight.data_ptr() != llama.model.embed_tokens.weight.data_ptr():
        raise AssertionError("lm_head is not tied to embed_tokens")
    return llama.eval(), reference


@torch.no_grad()
def max_logit_diff(llama: LlamaForCausalLM, reference: torch.nn.Module, ids: list[int]) -> float:
    batch = torch.tensor([ids])
    expected, _ = reference(batch)
    actual = llama(batch).logits
    return float((expected - actual).abs().max())


@torch.no_grad()
def llama_greedy(llama: LlamaForCausalLM, ids: list[int], max_new_tokens: int) -> list[int]:
    out = llama.generate(
        torch.tensor([ids]),
        attention_mask=torch.ones(1, len(ids), dtype=torch.long),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        eos_token_id=EOS_ID,
        pad_token_id=PAD_ID,
    )
    return out[0, len(ids) :].tolist()


def generation_config() -> GenerationConfig:
    return GenerationConfig(
        bos_token_id=BOS_ID,
        eos_token_id=EOS_ID,
        pad_token_id=PAD_ID,
        do_sample=True,
        temperature=0.7,
        top_k=50,
    )


def run(save: bool) -> dict:
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    generator = torch.Generator().manual_seed(0)
    random_ids = torch.randint(0, ARCH["vocab_size"], (RANDOM_SEQ_LEN,), generator=generator)
    report = {"tolerance": LOGIT_TOLERANCE, "models": {}}

    for model in golden["sources"]:
        llama, reference = build_llama(model)
        cases = [c for c in golden["cases"] if c["model"] == model]
        diffs = {c["key"]: max_logit_diff(llama, reference, c["input_ids"]) for c in cases}
        diffs["random_512"] = max_logit_diff(llama, reference, random_ids.tolist())
        mismatches = [
            c["key"]
            for c in cases
            if llama_greedy(llama, c["input_ids"], golden["max_new_tokens"]) != c["greedy_ids"]
        ]
        parameters = sum(p.numel() for p in llama.parameters())
        report["models"][model] = {
            "max_abs_logit_diff": max(diffs.values()),
            "logit_diff_by_case": diffs,
            "greedy_mismatches": mismatches,
            "unique_parameters": parameters,
            "passed": max(diffs.values()) < LOGIT_TOLERANCE
            and not mismatches
            and parameters == UNIQUE_PARAMETERS,
        }
        if save:
            target = store_path("llama", model)
            llama.save_pretrained(target, safe_serialization=True)
            generation_config().save_pretrained(target)
            reloaded = LlamaForCausalLM.from_pretrained(target).eval()
            reload_diff = max(max_logit_diff(reloaded, reference, c["input_ids"]) for c in cases)
            report["models"][model]["saved_max_abs_logit_diff"] = reload_diff
            report["models"][model]["passed"] &= reload_diff < LOGIT_TOLERANCE

    report["passed"] = all(m["passed"] for m in report["models"].values())
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Gate 1: SinhalaGPT vs LlamaForCausalLM.")
    parser.add_argument("--no-save", action="store_true", help="skip writing converted models")
    args = parser.parse_args()

    report = run(save=not args.no_save)
    output_dir = store_path("llama")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "parity.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for name, result in report["models"].items():
        status = "PASS" if result["passed"] else "FAIL"
        print(
            f"gate 1 {name}: {status} max|logit diff|={result['max_abs_logit_diff']:.3e} "
            f"greedy mismatches={len(result['greedy_mismatches'])} "
            f"parameters={result['unique_parameters']:,}"
        )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
