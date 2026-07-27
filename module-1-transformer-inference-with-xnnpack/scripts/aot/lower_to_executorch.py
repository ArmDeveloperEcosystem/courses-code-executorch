#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys
from typing import Any


MODULE1_SCRIPTS = Path(__file__).resolve().parents[1]
if str(MODULE1_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(MODULE1_SCRIPTS))

from _support.common import configure_quiet_output


def main() -> None:
    parser = argparse.ArgumentParser(description="Load an exported program, lower it, and write an ExecuTorch .pte artifact.")
    parser.add_argument("--spec", type=Path, required=True, help="Path to model_spec.json.")
    parser.add_argument("--backend", choices=["portable", "xnnpack"], required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output .pte path, or the exact directory to write into.",
    )
    args = parser.parse_args()

    configure_quiet_output(include_interpreter_bin=True)

    from torch.export import load

    spec = json.loads(args.spec.read_text())
    exported_path = resolve_exported_path(args.spec, spec)
    out = resolve_output_path(args.out, spec_path=args.spec, spec=spec, backend=args.backend)

    print(f"Loading exported program from {exported_path}...")
    exported_program = load(exported_path)

    print(f"Lowering {spec['name']} with {args.backend} backend...")
    program, delegation = lower(
        exported_program,
        backend=args.backend,
    )
    write_pte(program, out)

    metadata = {
        "model": spec["name"],
        "model_id": spec["model_id"],
        "backend": args.backend,
        "exported_program": str(exported_path),
        "pte": str(out),
        "pte_size_mb": out.stat().st_size / 1_000_000.0,
        "seq_len": spec["seq_len"],
        "delegation": delegation,
    }
    metadata_path = out.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"Wrote {out}")
    print(f"PTE size: {metadata['pte_size_mb']:.2f} MB")
    print(f"Wrote {metadata_path}")
    if delegation:
        print_delegation(delegation)


def resolve_exported_path(spec_path: Path, spec: dict[str, Any]) -> Path:
    if "exported_program" not in spec:
        raise KeyError("Spec does not contain exported_program. Re-run the prepare script.")
    path = Path(spec["exported_program"])
    return path if path.is_absolute() else spec_path.parent / path


def resolve_output_path(out: Path | None, *, spec_path: Path, spec: dict[str, Any], backend: str) -> Path:
    default_name = default_pte_name(spec, backend)
    if out is None:
        return spec_path.parent / default_name
    if out.suffix == ".pte":
        return out
    return out / default_name


def lower(
    exported_program: Any,
    *,
    backend: str,
) -> tuple[Any, dict[str, Any] | None]:
    from executorch.exir import to_edge_transform_and_lower

    if backend == "portable":
        edge = to_edge_transform_and_lower(exported_program)
        return edge.to_executorch(), None

    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

    edge = to_edge_transform_and_lower(
        exported_program,
        partitioner=[XnnpackPartitioner()],
    )
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
            return
        buffer = getattr(program, "buffer", None) or getattr(program, "_buffer", None)
        if buffer is None:
            raise AttributeError("ExecuTorch program has no write_to_file(), buffer, or _buffer")
        f.write(buffer)


def default_pte_name(spec: dict[str, Any], backend: str) -> str:
    name = spec["name"].replace("/", "_").replace("-", "_")
    return f"{name}_{backend}.pte"


if __name__ == "__main__":
    main()
