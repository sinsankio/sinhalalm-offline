import hashlib
import json
import os
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
STORE_DIR = Path(os.environ.get("SINHALALM_STORE") or ROOT / "build").expanduser().resolve()

REPO = "namindu94/SinhalaLLM-1.7B"
REVISION = "b24d58b7929e1ed50f683b435a9a629a73e27af5"
SHA256 = {
    "config.json": "3d8a16c288b4cb0b4d0827999359549c925d8f2702bc92c599b1225941bc99ad",
    "generation_config.json": "8ad24185fd161baeadd067630ef6bc2f640e788bc77dc465d2c10ab11573f20a",
    "tokenizer.json": "92eb9e4b8595d2c99deb4bae19e7127c9744e7bb9c7580e8e3c6beace806671e",
    "tokenizer_config.json": "95dad24518a3849fdc57045001a6faca05fbc2fb7a9471dba37fdcdc9bc2b52e",
    "chat_template.jinja": "f91864928fccff928b3068ca784db6275e655e400ccbb28f61d92c0090caf3b3",
    "model.safetensors": "715de1edf5d34a4282e960b5d02c6f982e3acbc8a38f3e08b998c9ae0b91c3f7",
}

STOP_IDS = [2, 0]
GOLDEN_PATH = ROOT / "golden" / "reference.json"
PROMPTS_PATH = ROOT / "golden" / "prompts.json"


def store_path(*parts: str) -> Path:
    if not STORE_DIR.is_dir():
        if not STORE_DIR.is_relative_to(ROOT):
            raise SystemExit(
                f"model store {STORE_DIR} is not available; mount it or set MODEL_STORE"
            )
        STORE_DIR.mkdir(parents=True)
    return STORE_DIR.joinpath(*parts)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify(path: Path, expected: str) -> None:
    stat = path.stat()
    stamp = f"{expected} {stat.st_size} {stat.st_mtime_ns}"
    marker = path.with_name(f".{path.name}.verified")
    if marker.exists() and marker.read_text() == stamp:
        return
    actual = _sha256(path)
    if actual != expected:
        raise RuntimeError(f"{REPO}/{path.name}: sha256 {actual} != {expected}")
    marker.write_text(stamp)


def model_dir() -> Path:
    local_dir = store_path("hf", REPO.split("/")[1])
    for filename, expected in SHA256.items():
        path = Path(hf_hub_download(REPO, filename, revision=REVISION, local_dir=local_dir))
        _verify(path, expected)
    return local_dir


def raw_config() -> dict:
    return json.loads((model_dir() / "config.json").read_text(encoding="utf-8"))


def load_golden() -> dict:
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def browser_tokenizer_config() -> dict:
    directory = model_dir()
    config = json.loads((directory / "tokenizer_config.json").read_text(encoding="utf-8"))
    template = (directory / "chat_template.jinja").read_text(encoding="utf-8")
    return config | {"chat_template": template}


def write_browser_tokenizer(target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    (target / "tokenizer.json").write_bytes((model_dir() / "tokenizer.json").read_bytes())
    config = json.dumps(browser_tokenizer_config(), ensure_ascii=False, indent=2) + "\n"
    (target / "tokenizer_config.json").write_text(config, encoding="utf-8")


def random_ids(spec: dict) -> list[int]:
    import torch

    generator = torch.Generator().manual_seed(spec["seed"])
    shape = (spec["length"],)
    return torch.randint(spec["low"], spec["high"], shape, generator=generator).tolist()


def write_model_folder(directory: Path) -> None:
    fp32 = store_path("onnx", "fp32")
    for name in ("config.json", "generation_config.json"):
        (directory / name).write_bytes((fp32 / name).read_bytes())
    write_browser_tokenizer(directory)
