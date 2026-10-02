#!/usr/bin/env python3
# Export MobileNetV2 and lower supported graph regions to XNNPACK.

from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any
import warnings


def main() -> None:
    # Script setup: command-line options, paths, and known dependency warnings.
    parser = argparse.ArgumentParser(description="Export MobileNetV2 and lower it to ExecuTorch XNNPACK.")
    parser.add_argument("--spec", type=Path, default=Path("module-2-cnn-inference-with-quantization/artifacts/mobilenetv2/model_spec.json"))
    parser.add_argument("--out", type=Path, default=None, help="Output .pte path.")
    parser.add_argument(
        "--profiling-artifacts",
        action="store_true",
        help="Also write matching ETRecord files and a portable PTE for the optional profiling activity.",
    )
    args = parser.parse_args()

    configure_quiet_output()

    import torch
    from torch.export import export
    from torchvision import models

    spec = json.loads(args.spec.read_text())
    artifact_dir = args.spec.parent
    out = args.out or artifact_dir / "mobilenetv2_xnnpack.pte"
    weights_path = artifact_dir / spec["weights"]

    # 1. Rebuild the PyTorch model and create an input with the deployed shape.
    print(f"Loading MobileNetV2 weights from {weights_path}...")
    model = models.mobilenet_v2(weights=None)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    example_inputs = (torch.randn(*spec["input"]["shape"]),)

    # 2. PyTorch API: capture the model as an ExportedProgram.
    print("Exporting MobileNetV2...")
    with torch.inference_mode():
        exported = export(model, example_inputs)

    portable_exported = None
    if args.profiling_artifacts:
        with torch.inference_mode():
            portable_exported = export(model, example_inputs)

    # 3. Course wrapper: lower_xnnpack() contains the ExecuTorch AOT and
    # XNNPACK partitioner APIs used to delegate supported regions.
    print("Lowering with XNNPACK...")
    edge = lower_xnnpack(exported)
    edge_for_etrecord = copy.deepcopy(edge) if args.profiling_artifacts else None
    delegation = inspect_delegation(edge)
    # ExecuTorch API: convert the lowered edge program for the runtime.
    program = edge.to_executorch()
    # Course wrapper: write_pte() handles ExecuTorch version-specific serialization.
    write_pte(program, out)

    # Optional profiling output; this is not part of model lowering.
    profiling_artifacts: dict[str, str] = {}
    if args.profiling_artifacts:
        etrecord_path = out.with_suffix(".etrecord")
        write_etrecord(edge_for_etrecord, program, etrecord_path)

        portable_path = out.parent / "mobilenetv2_portable.pte"
        # Course wrapper: lower_portable() contains the ExecuTorch edge API.
        portable_edge = lower_portable(portable_exported)
        portable_edge_for_etrecord = copy.deepcopy(portable_edge)
        portable_program = portable_edge.to_executorch()
        write_pte(portable_program, portable_path)
        portable_etrecord_path = portable_path.with_suffix(".etrecord")
        write_etrecord(portable_edge_for_etrecord, portable_program, portable_etrecord_path)
        profiling_artifacts = {
            "xnnpack_etrecord": str(etrecord_path),
            "portable_pte": str(portable_path),
            "portable_etrecord": str(portable_etrecord_path),
        }

    # Supporting output: record artifact sizes and delegation for comparison.
    metadata = {
        "model": spec["name"],
        "backend": "xnnpack",
        "precision": "fp32",
        "pte": str(out),
        "pte_size_mb": out.stat().st_size / 1_000_000.0,
        "delegation": delegation,
        "profiling_artifacts": profiling_artifacts,
    }
    metadata_path = out.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"Wrote {out}")
    print(f"PTE size: {metadata['pte_size_mb']:.2f} MB")
    print(f"Wrote {metadata_path}")
    for path in profiling_artifacts.values():
        print(f"Wrote {path}")
    print_delegation(delegation)


def lower_xnnpack(exported: Any) -> Any:
    # Course wrapper around ExecuTorch's to_edge_transform_and_lower() API and
    # XnnpackPartitioner.
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.exir import to_edge_transform_and_lower

    return to_edge_transform_and_lower(exported, partitioner=[XnnpackPartitioner()])


def lower_portable(exported: Any) -> Any:
    # Course wrapper around ExecuTorch's to_edge() API, with no backend partitioner.
    from executorch.exir import to_edge

    if exported is None:
        raise ValueError("Portable lowering requires a separate exported program.")
    return to_edge(exported)


def inspect_delegation(edge: Any) -> dict[str, Any]:
    # Count XNNPACK regions and operators left on the portable path.
    import torch

    graph = edge.exported_program().graph_module.graph
    delegate_op = torch.ops.higher_order.executorch_call_delegate
    delegate_calls = 0
    fallback_targets: list[str] = []
    for node in graph.nodes:
        if node.op != "call_function":
            continue
        if node.target == delegate_op:
            delegate_calls += 1
        else:
            fallback_targets.append(str(node.target))

    fallback_counts = Counter(fallback_targets)
    return {
        "delegate_call_count": delegate_calls,
        "fallback_call_function_count": len(fallback_targets),
        "fallback_ops": [f"{name} ({count})" for name, count in sorted(fallback_counts.items())],
    }


def print_delegation(summary: dict[str, Any], *, max_ops: int = 30) -> None:
    # Print the delegation evidence used later with benchmark results.
    print("\nDelegation")
    print("==========")
    print(f"XNNPACK delegate call sites: {summary['delegate_call_count']}")
    print(f"Fallback call_function ops:  {summary['fallback_call_function_count']}")
    for op in summary["fallback_ops"][:max_ops]:
        print(f"  {op}")
    if len(summary["fallback_ops"]) > max_ops:
        print(f"  ... {len(summary['fallback_ops']) - max_ops} more")


def write_pte(program: Any, path: Path) -> None:
    # Course wrapper around the serialization APIs in the pinned ExecuTorch version.
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        if hasattr(program, "write_to_file"):
            program.write_to_file(f)
        else:
            f.write(getattr(program, "buffer", None) or getattr(program, "_buffer"))


def write_etrecord(edge: Any, program: Any, path: Path) -> None:
    # Write graph metadata that matches the packaged ExecuTorch program.
    from executorch.devtools import generate_etrecord

    if edge is None:
        raise ValueError("ETRecord generation requires the pre-packaging edge program.")
    generate_etrecord(str(path), edge, program)


def configure_quiet_output() -> None:
    # Supporting setup: filter known warnings without hiding export failures.
    ensure_interpreter_bin_on_path()
    os.environ.setdefault("GLOG_minloglevel", "2")
    os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning, message="The given buffer is not writable.*")
    logging.getLogger().setLevel(logging.ERROR)
    for name in (
        "torch",
        "torch.distributed",
        "torch.distributed.elastic",
        "torch.utils._pytree",
        "torchvision",
        "executorch",
    ):
        logging.getLogger(name).setLevel(logging.ERROR)


def ensure_interpreter_bin_on_path() -> None:
    # Make venv-provided tools such as flatc visible when using venv/bin/python directly.
    bin_path = Path(sys.executable).parent
    flatc_path = bin_path / "flatc"
    if flatc_path.exists():
        os.environ.setdefault("FLATC_EXECUTABLE", str(flatc_path))
    bin_dir = str(bin_path)
    path_parts = [part for part in os.environ.get("PATH", "").split(os.pathsep) if part and part != bin_dir]
    os.environ["PATH"] = os.pathsep.join([bin_dir, *path_parts])

if __name__ == "__main__":
    main()
