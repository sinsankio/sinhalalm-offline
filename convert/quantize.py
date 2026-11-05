import argparse
import collections
import json
import shutil
import sys
from pathlib import Path

import onnx
from onnx import numpy_helper
from onnxruntime.quantization.matmul_nbits_quantizer import (
    HQQWeightOnlyQuantConfig,
    MatMulNBitsQuantizer,
)

from sinhalallm import store_path, write_model_folder

PROTOBUF_LIMIT = 2_000_000_000
BLOCK_SIZE = 32
VARIANTS = {
    "q8": {"file": "model_quantized.onnx", "bits": 8, "head": "quantized", "embedding": True},
    "q4": {"file": "model_q4.onnx", "bits": 4, "head": "quantized", "embedding": True},
    "q4-tied": {"file": "model_q4.onnx", "bits": 4, "head": "tied", "embedding": False},
    "q4-head": {"file": "model_q4.onnx", "bits": 4, "head": "quantized", "embedding": False},
    "q4-emb": {"file": "model_q4.onnx", "bits": 4, "head": "fp32", "embedding": True},
    "q4-asym": {"file": "model_q4.onnx", "bits": 4, "head": "tied", "symmetric": False},
    "q4-b16": {"file": "model_q4.onnx", "bits": 4, "head": "tied", "block_size": 16},
    "q4-hqq": {"file": "model_q4.onnx", "bits": 4, "head": "tied", "algorithm": "hqq"},
    "q4-v0": {
        "file": "model_q4.onnx",
        "bits": 4,
        "head": "tied",
        "keep": ["layers.0/self_attn/v_proj"],
    },
    "q4-v0-small": {
        "file": "model_q4.onnx",
        "bits": 4,
        "head": "quantized",
        "embedding": True,
        "keep": ["layers.0/self_attn/v_proj"],
    },
}
DEFAULTS = {
    "embedding": False,
    "symmetric": True,
    "block_size": BLOCK_SIZE,
    "algorithm": "rtn",
    "keep": [],
}


def fp32_model() -> onnx.ModelProto:
    return onnx.load(str(store_path("onnx", "fp32", "model.onnx")))


def save(model: onnx.ModelProto, path: Path) -> None:
    external = sum(len(tensor.raw_data) for tensor in model.graph.initializer) >= PROTOBUF_LIMIT
    onnx.save(
        model,
        str(path),
        save_as_external_data=external,
        all_tensors_to_one_file=True,
        location=f"{path.name}_data",
    )


def untie_head(model: onnx.ModelProto) -> list[str]:
    graph = model.graph
    initializers = {tensor.name: tensor for tensor in graph.initializer}
    consumers = collections.defaultdict(list)
    for node in graph.node:
        for name in node.input:
            consumers[name].append(node)
    transposes = [
        node
        for node in graph.node
        if node.op_type == "Transpose"
        and node.input[0] in initializers
        and any(c.op_type == "MatMul" for c in consumers[node.output[0]])
    ]
    if len(transposes) != 1:
        raise RuntimeError(f"expected one tied head transpose, found {len(transposes)}")
    transpose = transposes[0]
    embedding = numpy_helper.to_array(initializers[transpose.input[0]])
    head = numpy_helper.from_array(embedding.T.copy(), name="lm_head.weight")
    graph.initializer.append(head)
    head_nodes = consumers[transpose.output[0]]
    for node in head_nodes:
        for index, name in enumerate(node.input):
            if name == transpose.output[0]:
                node.input[index] = head.name
    graph.node.remove(transpose)
    return [node.name for node in head_nodes]


def quantize_nbits(target: Path, spec: dict) -> None:
    model = fp32_model()
    options = {
        "bits": spec["bits"],
        "block_size": spec["block_size"],
        "is_symmetric": spec["symmetric"],
    }
    excluded = [
        node.name
        for node in model.graph.node
        if node.op_type == "MatMul" and any(pattern in node.name for pattern in spec["keep"])
    ]
    if spec["head"] != "tied":
        head_nodes = untie_head(model)
        if spec["head"] == "fp32":
            excluded += head_nodes
    if excluded:
        options["nodes_to_exclude"] = excluded
    if spec["embedding"]:
        options |= {
            "op_types_to_quantize": ("MatMul", "Gather"),
            "quant_axes": (("MatMul", 0), ("Gather", 1)),
        }
    if spec["algorithm"] == "hqq":
        options["algo_config"] = HQQWeightOnlyQuantConfig(
            block_size=spec["block_size"],
            bits=spec["bits"],
            op_types_to_quantize=options.get("op_types_to_quantize"),
            quant_axes=options.get("quant_axes"),
        )
    quantizer = MatMulNBitsQuantizer(model, **options)
    quantizer.process()
    save(quantizer.model.model, target)


def op_counts(path: Path) -> dict[str, int]:
    graph = onnx.load(str(path), load_external_data=False).graph
    counts = collections.Counter(node.op_type for node in graph.node)
    keep = ("MatMul", "MatMulNBits", "MatMulInteger", "Gather", "GatherBlockQuantized", "Cast")
    return {name: counts[name] for name in keep if counts[name]}


def main() -> int:
    parser = argparse.ArgumentParser(description="Quantize the fp32 ONNX model.")
    parser.add_argument("variant", choices=VARIANTS)
    args = parser.parse_args()

    directory = store_path("onnx", args.variant)
    shutil.rmtree(directory, ignore_errors=True)
    (directory / "onnx").mkdir(parents=True)
    spec = DEFAULTS | VARIANTS[args.variant]
    target = directory / "onnx" / spec["file"]
    quantize_nbits(target, spec)

    write_model_folder(directory)
    files = {path.name: path.stat().st_size for path in sorted(target.parent.iterdir())}
    report = {
        "variant": args.variant,
        "settings": spec,
        "file": target.name,
        "bytes": sum(files.values()),
        "files": files,
        "ops": op_counts(target),
    }
    (directory / "quantization.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"{args.variant}: {report['bytes'] / 1e9:.3f} GB {report['ops']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
