import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
import onnx_ir as ir

from sinhalallm import store_path, write_model_folder

TASK = "text-generation-with-past"
OPSET = 18
MODEL_FILE = "model.onnx"
OUTPUTS = {
    "fp32": ("fp32", "model.onnx"),
    "fp16": ("fp16/onnx", "model_fp16.onnx"),
}


def trace(target: Path, dtype: str) -> None:
    import optimum.exporters.onnx.convert as convert
    import torch
    from optimum.exporters.onnx import onnx_export_from_model

    from parity_check import load_export_model, translated_config

    convert.check_model_uses_external_data = lambda _: False
    torch_dtype = torch.float16 if dtype == "fp16" else torch.float32
    model = load_export_model(translated_config(), torch_dtype)
    onnx_export_from_model(
        model,
        output=target,
        task=TASK,
        opset=OPSET,
        no_post_process=True,
        do_validation=False,
        device="cpu",
    )


def initializer_inputs(graph: ir.Graph, op_type: str, index: int) -> list[tuple[ir.Node, ir.Value]]:
    return [
        (node, node.inputs[index])
        for node in graph
        if node.op_type == op_type
        and node.inputs[index] is not None
        and node.inputs[index].is_initializer()
    ]


def tie_embedding(model: ir.Model) -> dict:
    graph = model.graph
    ((_, embedding),) = [
        (node, value)
        for node, value in initializer_inputs(graph, "Gather", 0)
        if len(value.shape) == 2
    ]
    vocab, hidden = embedding.shape
    heads = [
        (node, value)
        for node, value in initializer_inputs(graph, "MatMul", 1)
        if tuple(value.shape) == (hidden, vocab)
    ]
    if len(heads) != 1:
        raise RuntimeError(f"expected one output head initializer, found {len(heads)}")
    head_node, head = heads[0]
    if not np.array_equal(embedding.const_value.numpy().T, head.const_value.numpy()):
        raise RuntimeError("output head is not the transposed embedding")
    transpose = ir.node("Transpose", inputs=[embedding], attributes={"perm": [1, 0]})
    graph.insert_before(head_node, transpose)
    head_node.replace_input_with(1, transpose.outputs[0])
    del graph.initializers[head.name]
    return {"embedding": embedding.name, "removed_duplicate": head.name, "shape": [vocab, hidden]}


def validate_raw(raw: Path) -> int:
    graph = onnx.load(str(raw / MODEL_FILE), load_external_data=False).graph
    for tensor in graph.initializer:
        location = next((e.value for e in tensor.external_data if e.key == "location"), None)
        if location is None:
            continue
        expected = (
            int(np.prod(tensor.dims))
            * onnx.helper.tensor_dtype_to_np_dtype(tensor.data_type).itemsize
        )
        path = raw / location
        if not path.is_file() or path.stat().st_size != expected:
            raise RuntimeError(f"incomplete raw export: {location}")
    return len(graph.initializer)


def finalize(raw: Path, target: Path, file_name: str) -> dict:
    model = ir.load(raw / MODEL_FILE)
    tied = tie_embedding(model)
    target.mkdir(parents=True)
    ir.save(model, target / file_name, external_data=f"{file_name}_data")
    return tied


def trace_command(raw: Path, dtype: str) -> list[str]:
    command = [sys.executable, __file__, "--dtype", dtype, "--trace-only", str(raw)]
    memory = os.environ.get("SINHALALM_EXPORT_MEMORY")
    if not memory or shutil.which("systemd-run") is None:
        return command
    limits = ["-p", f"MemoryMax={memory}", "-p", "MemorySwapMax=0"]
    return ["systemd-run", "--user", "--scope", "--quiet", *limits, *command]


def main() -> int:
    parser = argparse.ArgumentParser(description="Export SinhalaLLM-1.7B to ONNX.")
    parser.add_argument("--dtype", choices=OUTPUTS, default="fp32")
    parser.add_argument("--trace-only", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.trace_only:
        trace(args.trace_only, args.dtype)
        return 0

    subdirectory, file_name = OUTPUTS[args.dtype]
    raw = store_path("onnx", f"raw-{args.dtype}")
    root = store_path("onnx", args.dtype)
    target = store_path("onnx", *subdirectory.split("/"))
    for path in (raw, root):
        shutil.rmtree(path, ignore_errors=True)
    raw.mkdir(parents=True)
    trace_exit = subprocess.run(trace_command(raw, args.dtype)).returncode
    initializers = validate_raw(raw)
    tied = finalize(raw, target, file_name)
    if args.dtype == "fp32":
        for name in ("config.json", "generation_config.json"):
            shutil.copyfile(raw / name, root / name)
    else:
        write_model_folder(root)
    shutil.rmtree(raw)

    files = {path.name: path.stat().st_size for path in sorted(target.iterdir())}
    report = {
        "task": TASK,
        "opset": OPSET,
        "dtype": args.dtype,
        "trace_exit_code": trace_exit,
        "initializers": initializers,
        "tied_embedding": tied,
        "files": files,
    }
    (root / "export.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for name, size in files.items():
        print(f"{name}: {size / 1e9:.3f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
