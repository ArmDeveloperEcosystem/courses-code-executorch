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
    parser = argparse.ArgumentParser(description="Quantize MobileNetV2 with PT2E and lower to XNNPACK.")
    parser.add_argument("--spec", type=Path, default=Path("module_2/artifacts/mobilenetv2/model_spec.json"))
    parser.add_argument(
        "--calibration-images",
        type=Path,
        default=Path("module_2/artifacts/mobilenetv2/calibration_images"),
        help="ImageFolder directory used for PTQ calibration.",
    )
    parser.add_argument("--out", type=Path, default=None, help="Output INT8 .pte path.")
    parser.add_argument("--num-calib", type=int, default=100)
    parser.add_argument("--quantize-classifier-head", action="store_true")
    args = parser.parse_args()

    configure_quiet_output()

    with suppress_stderr():
        import torch
        import torchvision.datasets as datasets
        import torchvision.models as models
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
        from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
            get_symmetric_quantization_config,
            XNNPACKQuantizer,
        )
        from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
        from torch.export import export
        from torchao.quantization.pt2e import move_exported_model_to_eval
        from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    spec = json.loads(args.spec.read_text())
    artifact_dir = args.spec.parent
    out = args.out or artifact_dir / "mobilenetv2_int8_xnnpack.pte"
    weights_path = artifact_dir / spec["weights"]

    if not args.calibration_images.exists():
        raise FileNotFoundError(
            f"Calibration images not found at {args.calibration_images}. "
            "Run create_calibration_images.py first."
        )

    with suppress_stderr():
        model = models.mobilenet_v2(weights=None)
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        model.eval()

    example_inputs = (torch.randn(*spec["input"]["shape"]),)
    quantizer = XNNPACKQuantizer()
    quantizer.set_global(get_symmetric_quantization_config(is_per_channel=True))
    if not args.quantize_classifier_head:
        quantizer.set_filter_function(skip_classifier_head)

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

    print(f"Calibrating on up to {args.num_calib} images from {args.calibration_images}...")
    with torch.inference_mode():
        for index, (image, _) in enumerate(calibration_loader):
            if index >= args.num_calib:
                break
            with suppress_stderr():
                prepared(image)

    quantized_model = convert_pt2e(prepared)
    exported_quantized = export(quantized_model, example_inputs)
    edge = to_edge_transform_and_lower(
        exported_quantized,
        partitioner=[XnnpackPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )
    program = edge.to_executorch()
    write_pte(program, out)

    fp32_pte = artifact_dir / "mobilenetv2_xnnpack.pte"
    metadata = {
        "model": spec["name"],
        "backend": "xnnpack",
        "precision": "int8",
        "pte": str(out),
        "pte_size_mb": out.stat().st_size / 1_000_000.0,
        "fp32_pte_size_mb": fp32_pte.stat().st_size / 1_000_000.0 if fp32_pte.exists() else None,
        "num_calib": min(args.num_calib, len(calibration_loader.dataset)),
        "delegation": inspect_delegation(edge),
    }
    metadata_path = out.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"Wrote {out}")
    print(f"INT8 PTE size: {metadata['pte_size_mb']:.2f} MB")
    if metadata["fp32_pte_size_mb"] is not None:
        print(f"FP32 PTE size: {metadata['fp32_pte_size_mb']:.2f} MB")
    print(f"Wrote {metadata_path}")


def skip_classifier_head(node: Any) -> bool:
    """Keep the final classifier head in fp32 for a steadier top-5 comparison."""
    for fqn, _typ in node.meta.get("nn_module_stack", {}).values():
        if "classifier.1" in str(fqn):
            return False
    return True


def image_transform(spec: dict[str, Any]) -> Any:
    import torchvision.transforms as transforms

    prep = spec["input"]["preprocessing"]
    return transforms.Compose(
        [
            transforms.Resize(prep["resize"]),
            transforms.CenterCrop(prep["center_crop"]),
            transforms.ToTensor(),
            transforms.Normalize(mean=prep["mean"], std=prep["std"]),
        ]
    )


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
