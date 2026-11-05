import argparse
import collections
import json
import shutil
import sys
from pathlib import Path

import onnx
from onnx import numpy_helper
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

from sinhalallm import store_path, write_model_folder

PROTOBUF_LIMIT = 2_000_000_000
BLOCK_SIZE = 32
FILE_NAMES = {
    "q8": "model_quantized.onnx",
    "q4-tied": "model_q4.onnx",
    "q4": "model_q4.onnx",
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


def untie_head(model: onnx.ModelProto) -> str:
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
    for node in consumers[transpose.output[0]]:
        for index, name in enumerate(node.input):
            if name == transpose.output[0]:
                node.input[index] = head.name
    graph.node.remove(transpose)
    return transpose.input[0]


def quantize_nbits(target: Path, bits: int, untie: bool) -> None:
    model = fp32_model()
    options = {"bits": bits, "block_size": BLOCK_SIZE, "is_symmetric": True}
    if untie:
        untie_head(model)
        options |= {
            "op_types_to_quantize": ("MatMul", "Gather"),
            "quant_axes": (("MatMul", 0), ("Gather", 1)),
        }
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
    parser.add_argument("variant", choices=FILE_NAMES)
    args = parser.parse_args()

    directory = store_path("onnx", args.variant)
    shutil.rmtree(directory, ignore_errors=True)
    (directory / "onnx").mkdir(parents=True)
    target = directory / "onnx" / FILE_NAMES[args.variant]
    if args.variant == "q8":
        quantize_nbits(target, bits=8, untie=True)
    else:
        quantize_nbits(target, bits=4, untie=args.variant == "q4")

    write_model_folder(directory)
    files = {path.name: path.stat().st_size for path in sorted(target.parent.iterdir())}
    report = {
        "variant": args.variant,
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
