import hashlib
import json
import shutil
import sys
from pathlib import Path

import onnx
from onnx import TensorProto
from onnx.external_data_helper import set_external_data

from eval_quality import DEFAULT_VARIANT, VARIANTS, OnnxCausalLM
from sinhalallm import REPO, REVISION, load_golden, store_path

MODEL_NAME = "sinhalallm-1.7b"
MODEL_VERSION = "v2.1"
COMMON_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
PACKAGED = {
    "q4": {"variant": DEFAULT_VARIANT, "model": "model_q4.onnx", "chunks": 1},
    "q8": {"variant": "q8", "model": "model_quantized.onnx", "chunks": 2},
}
BROWSER_UNSUPPORTED = {"BitShift", "BitwiseAnd", "GatherBlockQuantized", "IsNaN"}
NOTICE = f"""SinhalaLLM-1.7B (version 2.1) by Namindu Sanchila Nadeeshan, released under the
Apache License 2.0: https://huggingface.co/{REPO}

This folder contains converted and quantized ONNX versions of the model at revision
{REVISION}, produced by the sinhalalm-web build pipeline: a 4-bit version, which keeps the
value projection of layer 0 in fp32, and an 8-bit version. Answers may be factually wrong and
are not suitable for medical, legal or financial questions.
"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_browser_ops(path: Path) -> None:
    graph = onnx.load(str(path), load_external_data=False).graph
    found = sorted({node.op_type for node in graph.node} & BROWSER_UNSUPPORTED)
    if found:
        raise RuntimeError(f"{path.name} contains operators the browser cannot run fast: {found}")


def chunk_names(name: str, chunks: int) -> list[str]:
    return [name if index == 0 else f"{name}_{index}" for index in range(chunks)]


def split_external_data(source: Path, target: Path, chunks: int) -> list[str]:
    model = onnx.load(str(source))
    tensors = [t for t in model.graph.initializer if len(t.raw_data) > 1024]
    limit = sum(len(t.raw_data) for t in tensors) / chunks
    names = chunk_names(f"{target.name}_data", chunks)
    sizes = [0] * chunks
    handles = [(target.parent / name).open("wb") for name in names]
    part = 0
    try:
        for tensor in tensors:
            data = tensor.raw_data
            if part < chunks - 1 and sizes[part] + len(data) > limit:
                part += 1
            set_external_data(tensor, location=names[part], offset=sizes[part], length=len(data))
            handles[part].write(data)
            sizes[part] += len(data)
            tensor.ClearField("raw_data")
            tensor.data_location = TensorProto.EXTERNAL
    finally:
        for handle in handles:
            handle.close()
    onnx.save(model, str(target))
    return names


def expected_greedy(variant_dir: Path) -> list[dict]:
    golden = load_golden()
    model = OnnxCausalLM(variant_dir / "onnx" / "model_q4.onnx")
    return [
        {"key": case["key"], "ids": model.greedy(case["input_ids"], golden["max_new_tokens"])}
        for case in golden["cases"]
    ]


def main() -> int:
    if VARIANTS[DEFAULT_VARIANT] != f"{DEFAULT_VARIANT}/onnx/model_q4.onnx":
        raise RuntimeError(f"unexpected file layout for {DEFAULT_VARIANT}")
    for spec in PACKAGED.values():
        check_browser_ops(store_path("onnx", spec["variant"], "onnx", spec["model"]))

    target = store_path("web", MODEL_NAME, MODEL_VERSION)
    shutil.rmtree(target, ignore_errors=True)
    (target / "onnx").mkdir(parents=True)
    default_source = store_path("onnx", DEFAULT_VARIANT)
    names = list(COMMON_FILES)
    for name in COMMON_FILES:
        shutil.copyfile(default_source / name, target / name)
    for spec in PACKAGED.values():
        source = store_path("onnx", spec["variant"], "onnx", spec["model"])
        parts = split_external_data(source, target / "onnx" / spec["model"], spec["chunks"])
        spec["files"] = [spec["model"], *parts]
        names.extend(f"onnx/{name}" for name in spec["files"])
    (target / "NOTICE.txt").write_text(NOTICE, encoding="utf-8")
    names.append("NOTICE.txt")

    files = {
        name: {"bytes": (target / name).stat().st_size, "sha256": sha256(target / name)}
        for name in names
    }
    variants = {
        dtype: {
            "variant": spec["variant"],
            "external_data_chunks": len(spec["files"]) - 1,
            "bytes": sum(files[f"onnx/{name}"]["bytes"] for name in spec["files"]),
        }
        for dtype, spec in PACKAGED.items()
    }
    manifest = {
        "model": MODEL_NAME,
        "version": MODEL_VERSION,
        "default": "q4",
        "variants": variants,
        "source": {"repo": REPO, "revision": REVISION},
        "files": files,
    }
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    expected = {"model": f"{MODEL_NAME}/{MODEL_VERSION}", "cases": expected_greedy(default_source)}
    store_path("web", "expected-greedy.json").write_text(json.dumps(expected) + "\n")

    for dtype, item in variants.items():
        print(
            f"packaged {MODEL_NAME}/{MODEL_VERSION} {dtype} ({item['variant']}): "
            f"{item['bytes'] / 1e9:.3f} GB"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
