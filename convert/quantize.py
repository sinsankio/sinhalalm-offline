import argparse
import collections
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
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


def portable_embedding(model: onnx.ModelProto) -> bool:
    graph = model.graph
    nodes = [node for node in graph.node if node.op_type == "GatherBlockQuantized"]
    if not nodes:
        return False
    (node,) = nodes
    block = next(a.i for a in node.attribute if a.name == "block_size")
    data_name, ids_name, scales_name = node.input[:3]
    data = next(t for t in graph.initializer if t.name == data_name)
    if data.data_type != TensorProto.INT4:
        raise ValueError(f"expected an INT4 embedding, found {data.data_type}")
    vocab, hidden = data.dims
    packed = np.frombuffer(data.raw_data, dtype=np.uint8).reshape(vocab, hidden // 2) ^ 0x88
    graph.initializer.remove(data)

    def name(suffix: str) -> str:
        return f"{node.name}/{suffix}"

    graph.initializer.extend(
        [
            numpy_helper.from_array(packed.copy(), name("packed")),
            numpy_helper.from_array(np.array(15, np.uint8), name("low_mask")),
            numpy_helper.from_array(np.array(4, np.uint8), name("shift")),
            numpy_helper.from_array(np.array([-1], np.int64), name("last_axis")),
            numpy_helper.from_array(np.array([0, 0, hidden], np.int64), name("hidden_shape")),
            numpy_helper.from_array(
                np.array([0, 0, hidden // block, block], np.int64), name("block_shape")
            ),
            numpy_helper.from_array(np.array(8.0, np.float32), name("offset")),
        ]
    )
    replacement = [
        helper.make_node("Gather", [name("packed"), ids_name], [name("rows")], axis=0),
        helper.make_node("BitwiseAnd", [name("rows"), name("low_mask")], [name("low")]),
        helper.make_node(
            "BitShift", [name("rows"), name("shift")], [name("high")], direction="RIGHT"
        ),
        helper.make_node("Unsqueeze", [name("low"), name("last_axis")], [name("low_axis")]),
        helper.make_node("Unsqueeze", [name("high"), name("last_axis")], [name("high_axis")]),
        helper.make_node("Concat", [name("low_axis"), name("high_axis")], [name("pairs")], axis=-1),
        helper.make_node("Reshape", [name("pairs"), name("block_shape")], [name("codes")]),
        helper.make_node("Cast", [name("codes")], [name("codes_float")], to=TensorProto.FLOAT),
        helper.make_node("Sub", [name("codes_float"), name("offset")], [name("centered")]),
        helper.make_node("Gather", [scales_name, ids_name], [name("row_scales")], axis=0),
        helper.make_node(
            "Unsqueeze", [name("row_scales"), name("last_axis")], [name("row_scales_axis")]
        ),
        helper.make_node("Mul", [name("centered"), name("row_scales_axis")], [name("scaled")]),
        helper.make_node("Reshape", [name("scaled"), name("hidden_shape")], [node.output[0]]),
    ]
    index = list(graph.node).index(node)
    graph.node.remove(node)
    for offset, new_node in enumerate(replacement):
        graph.node.insert(index + offset, new_node)
    return True


def quantize_nbits(target: Path, spec: dict) -> bool:
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
    portable = portable_embedding(quantizer.model.model)
    save(quantizer.model.model, target)
    return portable


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
    portable = quantize_nbits(target, spec)

    write_model_folder(directory)
    files = {path.name: path.stat().st_size for path in sorted(target.parent.iterdir())}
    report = {
        "variant": args.variant,
        "settings": spec,
        "portable_embedding": portable,
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
