#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any
import warnings


def main() -> None:
    parser = argparse.ArgumentParser(description="Export MobileNetV2 and lower it to ExecuTorch XNNPACK.")
    parser.add_argument("--spec", type=Path, default=Path("module_2/artifacts/mobilenetv2/model_spec.json"))
    parser.add_argument("--out", type=Path, default=None, help="Output .pte path.")
    args = parser.parse_args()

    configure_quiet_output()

    with suppress_stderr():
        import torch
        from torch.export import export
        from torchvision import models

    spec = json.loads(args.spec.read_text())
    artifact_dir = args.spec.parent
    out = args.out or artifact_dir / "mobilenetv2_xnnpack.pte"
    weights_path = artifact_dir / spec["weights"]

    print(f"Loading MobileNetV2 weights from {weights_path}...")
    with suppress_stderr():
        model = models.mobilenet_v2(weights=None)
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        model.eval()

    example_inputs = (torch.randn(*spec["input"]["shape"]),)
    print("Exporting MobileNetV2...")
    with torch.inference_mode(), suppress_stderr():
        exported = export(model, example_inputs)

    print("Lowering with XNNPACK...")
    program, delegation = lower_xnnpack(exported)
    write_pte(program, out)

    metadata = {
        "model": spec["name"],
        "backend": "xnnpack",
        "precision": "fp32",
        "pte": str(out),
        "pte_size_mb": out.stat().st_size / 1_000_000.0,
        "delegation": delegation,
    }
    metadata_path = out.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"Wrote {out}")
    print(f"PTE size: {metadata['pte_size_mb']:.2f} MB")
    print(f"Wrote {metadata_path}")
    print_delegation(delegation)


def lower_xnnpack(exported: Any) -> tuple[Any, dict[str, Any]]:
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.exir import to_edge_transform_and_lower

    edge = to_edge_transform_and_lower(exported, partitioner=[XnnpackPartitioner()])
    return edge.to_executorch(), inspect_delegation(edge)


def inspect_delegation(edge: Any) -> dict[str, Any]:
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
    print("\nDelegation")
    print("==========")
    print(f"XNNPACK delegate call sites: {summary['delegate_call_count']}")
    print(f"Fallback call_function ops:  {summary['fallback_call_function_count']}")
    for op in summary["fallback_ops"][:max_ops]:
        print(f"  {op}")
    if len(summary["fallback_ops"]) > max_ops:
        print(f"  ... {len(summary['fallback_ops']) - max_ops} more")


def write_pte(program: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        if hasattr(program, "write_to_file"):
            program.write_to_file(f)
        else:
            f.write(getattr(program, "buffer", None) or getattr(program, "_buffer"))


def configure_quiet_output() -> None:
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
    """Make venv-provided tools such as flatc visible when using venv/bin/python directly."""
    bin_path = Path(sys.executable).parent
    flatc_path = bin_path / "flatc"
    if flatc_path.exists():
        os.environ.setdefault("FLATC_EXECUTABLE", str(flatc_path))
    bin_dir = str(bin_path)
    path_parts = [part for part in os.environ.get("PATH", "").split(os.pathsep) if part and part != bin_dir]
    os.environ["PATH"] = os.pathsep.join([bin_dir, *path_parts])


@contextmanager
def suppress_stderr() -> Any:
    """Temporarily hide noisy native dependency logs written directly to stderr."""
    saved_stderr = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved_stderr, 2)
        os.close(saved_stderr)
        os.close(devnull)


if __name__ == "__main__":
    main()
