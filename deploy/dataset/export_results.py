import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "convert"))

from sinhalallm import store_path  # noqa: E402

CARD = Path(__file__).resolve().parent / "README.md"

VARIANTS = {
    "fp32": "fp32, original weights",
    "fp16": "fp16",
    "q8": "8-bit weights",
    "q4-tied": "4-bit, default quantizer (head tied to the fp32 embedding)",
    "q4-head": "4-bit, 4-bit head, fp32 embedding",
    "q4-emb": "4-bit, fp32 head, 4-bit embedding",
    "q4": "4-bit, 4-bit head and embedding",
    "q4-asym": "4-bit, asymmetric quantization",
    "q4-b16": "4-bit, block size 16",
    "q4-hqq": "4-bit, HQQ",
    "q4-v0": "4-bit, first-layer value projection in fp32, head tied to the fp32 embedding",
    "q4-v0-small": "4-bit, first-layer value projection in fp32 (shipped as the 4-bit model)",
    "gguf-q4_k_m": "Q4_K_M GGUF release by the model author (llama.cpp)",
}

BROWSER = [
    ("4-bit", "WebGPU", "download", 1.18, "GB"),
    ("4-bit", "WebGPU", "median speed", 26.5, "tokens/s"),
    ("4-bit", "WebGPU", "median first token", 69, "ms"),
    ("4-bit", "WebGPU", "first token, 52-token question", 484, "ms"),
    ("4-bit", "WebGPU", "warm load", 3.7, "s"),
    ("4-bit", "WebGPU", "warm-up", 0.7, "s"),
    ("4-bit", "WebGPU", "page memory", 0.09, "GB"),
    ("4-bit", "WebGPU", "renderer process", 1.37, "GB"),
    ("4-bit", "WebAssembly", "speed", 0.5, "tokens/s"),
    ("4-bit", "WebAssembly", "first token, 52-token question", 5200, "ms"),
    ("4-bit", "WebAssembly", "page memory", 1.78, "GB"),
    ("4-bit", "WebAssembly", "renderer process", 2.35, "GB"),
    ("8-bit", "WebGPU", "download", 2.48, "GB"),
    ("8-bit", "WebGPU", "median speed", 26.6, "tokens/s"),
    ("8-bit", "WebGPU", "median first token", 76, "ms"),
    ("8-bit", "WebGPU", "renderer process", 1.99, "GB"),
]


def perplexity(rows):
    return math.exp(sum(row[0] for row in rows) / sum(row[1] for row in rows))


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export the evaluation results as dataset tables.")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    results = store_path("rq1")
    out = args.out or store_path("dataset")
    out.mkdir(parents=True, exist_ok=True)
    reports = {}
    for name in VARIANTS:
        path = results / f"{name}.json"
        if path.exists():
            reports[name] = json.loads(path.read_text(encoding="utf-8"))
    if "fp32" not in reports:
        raise SystemExit(f"no fp32 report in {results}; check MODEL_STORE or run make rq1")
    reference = reports["fp32"]["perplexity"]["per_sentence_nll_tokens_chars"]

    quality, agreement, sentences = [], [], []
    for name, report in reports.items():
        scores = report["perplexity"]["per_sentence_nll_tokens_chars"]
        own = perplexity(scores)
        base = perplexity(reference[: len(scores)])
        quality.append(
            [
                VARIANTS[name],
                report.get("file_bytes") or "",
                len(scores),
                round(own, 2),
                round(report["perplexity"]["bits_per_character"], 3),
                round(base, 2),
                round(100 * (own / base - 1), 1),
            ]
        )
        sentences += [[VARIANTS[name], index, *row] for index, row in enumerate(scores)]
        if report.get("agreement"):
            item = report["agreement"]
            agreement.append(
                [
                    VARIANTS[name],
                    item["questions"],
                    round(item["prompt_top1_agreement"], 3),
                    item["greedy_exact_matches"],
                ]
            )

    write_csv(
        out / "perplexity.csv",
        [
            "variant",
            "file_bytes",
            "sentences",
            "perplexity",
            "bits_per_character",
            "fp32_perplexity_same_sentences",
            "change_vs_fp32_percent",
        ],
        quality,
    )
    write_csv(
        out / "agreement.csv",
        ["variant", "questions", "prompt_top1_agreement", "greedy_exact_matches"],
        agreement,
    )
    write_csv(
        out / "per_sentence.csv",
        ["variant", "sentence", "negative_log_likelihood", "tokens", "characters"],
        sentences,
    )
    write_csv(out / "browser.csv", ["model", "backend", "metric", "value", "unit"], BROWSER)
    shutil.copyfile(CARD, out / "README.md")
    print(f"wrote {len(quality)} variants and {len(sentences)} sentence scores to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
