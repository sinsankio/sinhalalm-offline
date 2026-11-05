import ast
import contextlib
import functools
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "build" / "hf"

VOCAB_SHA256 = "5883d40718e284bcb6b4522c3f8ebda03bbeb6ab45e4c7a3cbda44274633ff38"
BASE_WEIGHTS_SHA256 = "268958f1d7487260fe933cacd473c0c3707b192308386c383a7954e8f3905a85"
IT_WEIGHTS_SHA256 = "9b896174107a850a57fa92895581f6798d6eaaf976fc4478b21a956633cf8be4"
IT_MODELING_SHA256 = "d0c555eec70caecd08999d27dc8454d8d40325db43d40fdce2877b803a860437"

MODELS = {
    "base": {
        "repo": "HelaAI/SinhalaLM-V.0.0.1",
        "revision": "5f9c6d73e5ee3ec9e3f9cc2d7c03f95f144a2d8c",
        "sha256": {
            "model.safetensors": BASE_WEIGHTS_SHA256,
            "bpe_vocabulary.json": VOCAB_SHA256,
        },
    },
    "it": {
        "repo": "HelaAI/SinhalaLM-V.0.0.1-it",
        "revision": "f3b1f5d5b27786688dce99ba11ceac4220150e00",
        "sha256": {
            "model.safetensors": IT_WEIGHTS_SHA256,
            "bpe_vocabulary.json": VOCAB_SHA256,
            "modeling_sinhalalm.py": IT_MODELING_SHA256,
        },
    },
}

REFERENCE_SOURCE = ("it", "modeling_sinhalalm.py")
REFERENCE_DEFINITIONS = frozenset(
    {
        "SPECIAL_TOKENS",
        "SYSTEM_HEADER",
        "build_prompt",
        "get_prompt_prefix",
        "FastBPETokenizer",
        "RoPEEmbedding",
        "SwiGLU",
        "CausalSelfAttention",
        "TransformerBlock",
        "SinhalaGPT",
    }
)

ARCH = {
    "vocab_size": 15000,
    "embed_dim": 512,
    "n_heads": 8,
    "n_layers": 8,
    "block_size": 512,
    "dropout": 0.0,
}

BOS_ID = 2
EOS_ID = 3


def fetch(model: str, filename: str) -> Path:
    spec = MODELS[model]
    local_dir = CACHE_DIR / spec["repo"].split("/")[1]
    path = Path(
        hf_hub_download(spec["repo"], filename, revision=spec["revision"], local_dir=local_dir)
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = spec["sha256"][filename]
    if digest != expected:
        raise RuntimeError(f"{spec['repo']}/{filename}: sha256 {digest} != {expected}")
    return path


def _definition_names(node: ast.stmt) -> set[str]:
    if isinstance(node, ast.FunctionDef | ast.ClassDef):
        return {node.name}
    if isinstance(node, ast.Assign):
        return {t.id for t in node.targets if isinstance(t, ast.Name)}
    return set()


@functools.cache
def reference_namespace() -> dict:
    path = fetch(*REFERENCE_SOURCE)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    selected = [node for node in tree.body if _definition_names(node) & REFERENCE_DEFINITIONS]
    found = set().union(*(_definition_names(node) for node in selected))
    missing = REFERENCE_DEFINITIONS - found
    if missing:
        raise RuntimeError(f"definitions not found in {path.name}: {sorted(missing)}")
    namespace = {
        "json": json,
        "math": math,
        "torch": torch,
        "nn": nn,
        "F": F,
        "Optional": Optional,
    }
    module = ast.Module(body=selected, type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


@functools.cache
def load_tokenizer():
    tokenizer_cls = reference_namespace()["FastBPETokenizer"]
    with contextlib.redirect_stdout(io.StringIO()):
        return tokenizer_cls.load(str(fetch("base", "bpe_vocabulary.json")))


def load_reference_model(model: str) -> nn.Module:
    gpt = reference_namespace()["SinhalaGPT"](**ARCH)
    gpt.load_state_dict(load_file(fetch(model, "model.safetensors")), strict=True)
    return gpt.eval()


def prompt_ids(model: str, text: str) -> list[int]:
    tokenizer = load_tokenizer()
    if model == "it":
        text = reference_namespace()["get_prompt_prefix"](text)
    return [BOS_ID, *tokenizer.encode(text)]


@torch.no_grad()
def reference_greedy(gpt: nn.Module, ids: list[int], max_new_tokens: int) -> list[int]:
    seq = list(ids)
    for _ in range(max_new_tokens):
        logits, _ = gpt(torch.tensor([seq]))
        next_id = int(logits[0, -1].argmax())
        seq.append(next_id)
        if next_id == EOS_ID:
            break
    return seq[len(ids) :]
