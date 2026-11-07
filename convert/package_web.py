import hashlib
import json
import shutil
import sys
from pathlib import Path

from eval_quality import DEFAULT_VARIANT, VARIANTS, OnnxCausalLM
from sinhalallm import REPO, REVISION, load_golden, store_path

MODEL_NAME = "sinhalallm-1.7b"
MODEL_VERSION = "v2.1"
FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "onnx/model_q4.onnx",
)
NOTICE = f"""SinhalaLLM-1.7B (version 2.1) by Namindu Sanchila Nadeeshan, released under the
Apache License 2.0: https://huggingface.co/{REPO}

This folder contains a converted and 4-bit quantized ONNX version of the model at revision
{REVISION}, produced by the sinhalalm-web build pipeline. The quantization keeps the value
projection of layer 0 in fp32. Answers may be factually wrong and are not suitable for
medical, legal or financial questions.
"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_greedy(variant_dir: Path) -> list[dict]:
    golden = load_golden()
    model = OnnxCausalLM(variant_dir / "onnx" / "model_q4.onnx")
    return [
        {"key": case["key"], "ids": model.greedy(case["input_ids"], golden["max_new_tokens"])}
        for case in golden["cases"]
    ]


def main() -> int:
    source = store_path("onnx", DEFAULT_VARIANT)
    if VARIANTS[DEFAULT_VARIANT] != f"{DEFAULT_VARIANT}/onnx/model_q4.onnx":
        raise RuntimeError(f"unexpected file layout for {DEFAULT_VARIANT}")
    target = store_path("web", MODEL_NAME, MODEL_VERSION)
    shutil.rmtree(target, ignore_errors=True)
    (target / "onnx").mkdir(parents=True)
    for name in FILES:
        shutil.copyfile(source / name, target / name)
    (target / "NOTICE.txt").write_text(NOTICE, encoding="utf-8")

    files = {
        name: {"bytes": (target / name).stat().st_size, "sha256": sha256(target / name)}
        for name in (*FILES, "NOTICE.txt")
    }
    manifest = {
        "model": MODEL_NAME,
        "version": MODEL_VERSION,
        "variant": DEFAULT_VARIANT,
        "dtype": "q4",
        "source": {"repo": REPO, "revision": REVISION},
        "files": files,
    }
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    expected = {"model": f"{MODEL_NAME}/{MODEL_VERSION}", "cases": expected_greedy(source)}
    store_path("web", "expected-greedy.json").write_text(json.dumps(expected) + "\n")

    total = sum(item["bytes"] for item in files.values())
    print(f"packaged {MODEL_NAME}/{MODEL_VERSION} ({DEFAULT_VARIANT}): {total / 1e9:.3f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
