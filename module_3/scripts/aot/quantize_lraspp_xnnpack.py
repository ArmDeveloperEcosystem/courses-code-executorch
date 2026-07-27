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

from segmentation_preprocessing import preprocess_tensor


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantize LR-ASPP MobileNetV3 with PT2E and lower to XNNPACK.")
    parser.add_argument("--spec", type=Path, default=Path("module_3/artifacts/lraspp_mobilenetv3/model_spec.json"))
    parser.add_argument(
        "--calibration-images",
        type=Path,
        default=Path("module_3/artifacts/lraspp_mobilenetv3/calibration_images"),
        help="ImageFolder directory used for PTQ calibration.",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--num-calib", type=int, default=100)
    args = parser.parse_args()

    configure_quiet_output()

    with suppress_stderr():
        import torch
        import torchvision.datasets as datasets
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
        from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
            get_symmetric_quantization_config,
            XNNPACKQuantizer,
        )
        from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
        from torch.export import export
        from torchao.quantization.pt2e import move_exported_model_to_eval
        from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
        from torchvision.models.segmentation import lraspp_mobilenet_v3_large

    spec = json.loads(args.spec.read_text())
    artifact_dir = args.spec.parent
    out = args.out or artifact_dir / "lraspp_mobilenetv3_int8_xnnpack.pte"

    if not args.calibration_images.exists():
        raise FileNotFoundError(
            f"Calibration images not found at {args.calibration_images}. "
            "Run create_segmentation_calibration_images.py first."
        )

    labels = json.loads((artifact_dir / spec["labels"]).read_text())
    with suppress_stderr():
        base_model = lraspp_mobilenet_v3_large(
            weights=None,
            weights_backbone=None,
            num_classes=len(labels),
        )
        base_model.load_state_dict(torch.load(artifact_dir / spec["weights"], map_location="cpu"))
        model = SegmentationLogits(base_model).eval()

    example_inputs = (torch.randn(*spec["input"]["shape"]),)
    quantizer = XNNPACKQuantizer()
    quantizer.set_global(get_symmetric_quantization_config(is_per_channel=True))

    print("Exporting and preparing LR-ASPP for PT2E quantization")
    with suppress_stderr():
        exported = export(model, example_inputs)
        prepared = prepare_pt2e(exported.module(), quantizer)
        move_exported_model_to_eval(prepared)

    calibration_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.calibration_images, transform=image_transform(spec)),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    print(f"Calibrating on up to {args.num_calib} images from {args.calibration_images}")
    with torch.inference_mode():
        for index, (image, _label) in enumerate(calibration_loader):
            if index >= args.num_calib:
                break
            with suppress_stderr():
                prepared(image)

    print("Converting to INT8 and lowering with XNNPACK")
    quantized_model = convert_pt2e(prepared)
    exported_quantized = export(quantized_model, example_inputs)
    edge = to_edge_transform_and_lower(
        exported_quantized,
        partitioner=[XnnpackPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )
    program = edge.to_executorch()
    write_pte(program, out)

    fp32_pte = artifact_dir / "lraspp_mobilenetv3_xnnpack.pte"
    metadata = {
        "model": spec["name"],
        "backend": "xnnpack",
        "precision": "int8",
        "pte": str(out),
        "pte_size_mb": out.stat().st_size / 1_000_000.0,
        "fp32_pte_size_mb": fp32_pte.stat().st_size / 1_000_000.0 if fp32_pte.exists() else None,
        "num_calib": min(args.num_calib, len(calibration_loader.dataset)),
        "calibration_images": str(args.calibration_images),
        "delegation": inspect_delegation(edge),
    }
    metadata_path = out.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"Wrote {out}")
    print(f"INT8 PTE size: {metadata['pte_size_mb']:.2f} MB")
    if metadata["fp32_pte_size_mb"] is not None:
        print(f"FP32 PTE size: {metadata['fp32_pte_size_mb']:.2f} MB")
    print(f"Wrote {metadata_path}")
    print_delegation(metadata["delegation"])


class SegmentationLogits:
    """Thin wrapper so the exported program returns only the segmentation logits tensor."""

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


def image_transform(spec: dict[str, Any]) -> Any:
    return lambda image: preprocess_tensor(image, spec)


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
        "torchao",
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


@contextmanager
def suppress_stderr() -> Any:
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
