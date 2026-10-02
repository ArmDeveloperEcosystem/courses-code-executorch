#!/usr/bin/env python3
# Export LR-ASPP logits and lower them to portable or XNNPACK ExecuTorch.

from __future__ import annotations

import argparse
from collections import Counter
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any
import warnings


def main() -> None:
    # Script setup: command-line options, paths, and known dependency warnings.
    parser = argparse.ArgumentParser(description="Export LR-ASPP MobileNetV3 and lower to ExecuTorch.")
    parser.add_argument("--spec", type=Path, default=Path("module-3-executorch-on-android-semantic-segmentation/artifacts/lraspp_mobilenetv3/model_spec.json"))
    parser.add_argument("--backend", choices=["portable", "xnnpack"], default="xnnpack")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    configure_quiet_output()

    import torch
    from torch.export import export
    from torchvision.models.segmentation import lraspp_mobilenet_v3_large

    spec = json.loads(args.spec.read_text())
    artifact_dir = args.spec.parent
    default_name = f"lraspp_mobilenetv3_{args.backend}.pte"
    out = args.out or artifact_dir / default_name

    # 1. Rebuild the PyTorch model and expose its logits tensor as one output.
    print(f"Loading LR-ASPP weights from {artifact_dir / spec['weights']}")
    base_model = lraspp_mobilenet_v3_large(
        weights=None,
        weights_backbone=None,
        num_classes=len(load_labels(args.spec, spec)),
    )
    base_model.load_state_dict(torch.load(artifact_dir / spec["weights"], map_location="cpu"))
    model = SegmentationLogits(base_model).eval()

    example_inputs = (torch.randn(*spec["input"]["shape"]),)

    # 2. PyTorch API: capture the model as an ExportedProgram.
    print("Exporting LR-ASPP segmentation logits")
    with torch.inference_mode():
        exported = export(model, example_inputs)

    # 3. ExecuTorch APIs: select backend delegation, then package the .pte.
    if args.backend == "xnnpack":
        print("Lowering with XNNPACK")
        # Course wrapper: lower_xnnpack() contains the ExecuTorch AOT and
        # XNNPACK partitioner APIs.
        program, delegation = lower_xnnpack(exported)
    else:
        print("Lowering with portable ExecuTorch")
        # Course wrapper: lower_portable() contains the ExecuTorch AOT APIs.
        program, delegation = lower_portable(exported), None
    # Course wrapper: write_pte() handles ExecuTorch version-specific serialization.
    write_pte(program, out)

    # Supporting output: record artifact size and delegation for comparison.
    metadata = {
        "model": spec["name"],
        "backend": args.backend,
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
    if delegation is None:
        print("\nDelegation")
        print("==========")
        print("Portable ExecuTorch artifact: no backend delegate regions")
    else:
        print_delegation(delegation)


class SegmentationLogits:
    # TorchVision returns a dictionary; the Android app expects one tensor.

    def __init__(self, model: Any):
        import torch

        class Wrapper(torch.nn.Module):
            def __init__(self, inner: Any):
                super().__init__()
                self.inner = inner

            def forward(self, image: Any) -> Any:
                return self.inner(image)["out"]

        self.module = Wrapper(model)

    def eval(self) -> Any:
        return self.module.eval()


def load_labels(spec_path: Path, spec: dict[str, Any]) -> list[str]:
    # Load the class order that defines the output tensor's channel meaning.
    return json.loads((spec_path.parent / spec["labels"]).read_text())


def lower_xnnpack(exported: Any) -> tuple[Any, dict[str, Any]]:
    # Course wrapper around ExecuTorch's to_edge_transform_and_lower() and
    # edge.to_executorch() APIs, using XnnpackPartitioner.
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower

    edge = to_edge_transform_and_lower(
        exported,
        partitioner=[XnnpackPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )
    return edge.to_executorch(), inspect_delegation(edge)


def lower_portable(exported: Any) -> Any:
    # Course wrapper around ExecuTorch's to_edge_transform_and_lower() and
    # edge.to_executorch() APIs, with no backend partitioner.
    from executorch.exir import to_edge_transform_and_lower

    edge = to_edge_transform_and_lower(exported)
    return edge.to_executorch()


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
    bin_path = Path(sys.executable).parent
    flatc_path = bin_path / "flatc"
    if flatc_path.exists():
        os.environ.setdefault("FLATC_EXECUTABLE", str(flatc_path))
    bin_dir = str(bin_path)
    path_parts = [part for part in os.environ.get("PATH", "").split(os.pathsep) if part and part != bin_dir]
    os.environ["PATH"] = os.pathsep.join([bin_dir, *path_parts])

if __name__ == "__main__":
    main()
