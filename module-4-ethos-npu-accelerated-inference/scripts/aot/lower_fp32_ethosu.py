#!/usr/bin/env python3
# Test how an FP32 MobileNetV2 graph partitions for an Ethos-U target.

from __future__ import annotations

import argparse
from collections import Counter
import json
import logging
import os
from pathlib import Path
from typing import Any
import warnings


def main() -> None:
    # Script setup: command-line options, paths, and known dependency warnings.
    parser = argparse.ArgumentParser(description="Attempt FP32 MobileNetV2 lowering with the Ethos-U partitioner.")
    parser.add_argument("--spec", type=Path, default=Path("module-4-ethos-npu-accelerated-inference/artifacts/mobilenetv2/model_spec.json"))
    parser.add_argument("--out", type=Path, default=None)
    add_ethosu_args(parser)
    args = parser.parse_args()

    configure_quiet_output()

    import torch
    from torchvision.models import mobilenet_v2

    spec = json.loads(args.spec.read_text())
    artifact_dir = args.spec.parent
    out = args.out or artifact_dir / "mobilenetv2_fp32_ethosu.pte"

    # 1. Rebuild the PyTorch model and create an input with the deployed shape.
    model = mobilenet_v2(weights=None).eval()
    model.load_state_dict(torch.load(artifact_dir / spec["weights"], map_location="cpu"))

    example_inputs = (torch.randn(*spec["input"]["shape"]),)

    # 2. PyTorch API: capture the model as an ExportedProgram.
    print("Exporting FP32 MobileNetV2...")
    with torch.inference_mode():
        # Course wrapper: export_no_guards() contains the PyTorch export APIs.
        exported = export_no_guards(model, example_inputs)

    # 3. Course wrappers: these functions contain the ExecuTorch APIs for
    # Ethos-U delegation, runtime-program conversion, and serialization.
    print(f"Lowering with Ethos-U target {args.target}...")
    edge = lower_to_ethosu(exported, args)
    program = to_executorch(edge)
    write_pte(program, out)

    # Supporting output: record target settings and delegation for comparison.
    metadata = {
        "model": spec["name"],
        "backend": "ethosu",
        "precision": "fp32",
        "target": args.target,
        "system_config": args.system_config,
        "memory_mode": args.memory_mode,
        "pte": str(out),
        "pte_size_mb": out.stat().st_size / 1_000_000.0,
        "delegation": inspect_delegation(edge),
    }
    metadata_path = out.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"Wrote {out}")
    print(f"PTE size: {metadata['pte_size_mb']:.2f} MB")
    print_delegation(metadata["delegation"])
    if metadata["delegation"]["delegate_call_count"] == 0:
        print("Observation: no Ethos-U delegation was detected for FP32.")


def add_ethosu_args(parser: argparse.ArgumentParser) -> None:
    # Add the target, memory, and Vela options shared by Ethos-U lowering.
    parser.add_argument("--target", default="ethos-u85-256")
    parser.add_argument("--system-config", default="Ethos_U85_SYS_DRAM_Mid")
    parser.add_argument("--memory-mode", default="Shared_Sram")
    parser.add_argument("--extra-flag", action="append", default=["--output-format=raw"])


def export_no_guards(model: Any, example_inputs: tuple[Any, ...]) -> Any:
    # Course wrapper around two torch.export.export() calls. The second export
    # removes input-value guards from the first exported graph module.
    import torch

    first = torch.export.export(model, example_inputs)
    graph_module = first.module(check_guards=False)
    return torch.export.export(graph_module, example_inputs)


def lower_to_ethosu(exported: Any, args: argparse.Namespace) -> Any:
    # Course wrapper around ExecuTorch's EthosUPartitioner and
    # to_edge_transform_and_lower() APIs.
    from executorch.backends.arm.ethosu import EthosUCompileSpec, EthosUPartitioner
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower

    compile_spec = EthosUCompileSpec(
        target=args.target,
        system_config=args.system_config,
        memory_mode=args.memory_mode,
        extra_flags=args.extra_flag,
    )
    return to_edge_transform_and_lower(
        exported,
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
        partitioner=[EthosUPartitioner(compile_spec)],
    )


def to_executorch(edge: Any) -> Any:
    # Course wrapper around edge.to_executorch() and ExecutorchBackendConfig.
    from executorch.exir import ExecutorchBackendConfig

    return edge.to_executorch(ExecutorchBackendConfig(extract_delegate_segments=True))


def inspect_delegation(edge: Any) -> dict[str, Any]:
    # Count Ethos-U delegate calls and operations left on the default path.
    import torch

    graph = edge.exported_program().graph_module.graph
    delegate_op = torch.ops.higher_order.executorch_call_delegate
    delegate_calls = 0
    fallback_targets: list[str] = []
    for node in graph.nodes:
        if node.op != "call_function":
            continue
        if node.target == delegate_op or "delegate" in str(node.target).lower():
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
    print(f"Ethos-U delegate call sites: {summary['delegate_call_count']}")
    print(f"Fallback call_function ops:  {summary['fallback_call_function_count']}")
    for op in summary["fallback_ops"][:max_ops]:
        print(f"  {op}")


def write_pte(program: Any, path: Path) -> None:
    # Course wrapper around the serialization APIs in the pinned ExecuTorch version.
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(program, "save"):
        program.save(str(path))
        return
    with path.open("wb") as f:
        if hasattr(program, "write_to_file"):
            program.write_to_file(f)
        else:
            f.write(getattr(program, "buffer", None) or getattr(program, "_buffer"))


def configure_quiet_output() -> None:
    # Supporting setup: filter known warnings without hiding export failures.
    os.environ.setdefault("GLOG_minloglevel", "2")
    os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    warnings.filterwarnings("ignore", category=FutureWarning)
    logging.getLogger().setLevel(logging.ERROR)

if __name__ == "__main__":
    main()
