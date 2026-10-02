#!/usr/bin/env python3
# Quantize MobileNetV2 and lower it to Ethos-U.

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import json
import logging
import os
from pathlib import Path
from typing import Any, Iterator
import warnings


def main() -> None:
    # Script setup: command-line options, paths, and known dependency warnings.
    parser = argparse.ArgumentParser(description="Quantize MobileNetV2 and lower to ExecuTorch Ethos-U.")
    parser.add_argument("--spec", type=Path, default=Path("module-4-ethos-npu-accelerated-inference/artifacts/mobilenetv2/model_spec.json"))
    parser.add_argument("--calibration-images", type=Path, required=True)
    parser.add_argument("--num-calib", type=int, default=100)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--validate-image",
        type=Path,
        default=None,
        help="Compare FP32 and converted INT8 predictions before Ethos-U lowering.",
    )
    parser.add_argument("--insert-lrn", action="store_true", help="Insert LocalResponseNorm to demonstrate fragmentation.")
    parser.add_argument(
        "--so-library",
        type=Path,
        action="append",
        default=[],
        help="Optional custom op library to load before export. May be passed more than once.",
    )
    add_ethosu_args(parser)
    args = parser.parse_args()

    configure_quiet_output()
    if args.num_calib <= 0:
        raise ValueError("--num-calib must be greater than zero")

    import torch
    import torchvision.datasets as datasets
    from torch.export import export
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
    from torchvision.models import mobilenet_v2

    for so_library in args.so_library:
        print(f"Loading custom ops from {so_library}...")
        torch.ops.load_library(str(so_library))

    spec = json.loads(args.spec.read_text())
    artifact_dir = args.spec.parent
    suffix = "lrn_int8_ethosu" if args.insert_lrn else "int8_ethosu"
    out = args.out or artifact_dir / f"mobilenetv2_{suffix}.pte"

    if not args.calibration_images.exists():
        raise FileNotFoundError(f"Calibration images not found: {args.calibration_images}")

    # 1. Rebuild the PyTorch model and optionally insert an unsupported operation.
    base_model = mobilenet_v2(weights=None).eval()
    base_model.load_state_dict(torch.load(artifact_dir / spec["weights"], map_location="cpu"))
    model = MobileNetV2WithLRN(base_model).eval() if args.insert_lrn else base_model

    example_inputs = (torch.randn(*spec["input"]["shape"]),)

    # 2. PyTorch API: capture the model graph used by PT2E.
    print("Exporting model for PT2E quantization...")
    with torch.inference_mode():
        exported = export(model, example_inputs)
        graph_module = exported.module(check_guards=False)

    print("Preparing Ethos-U quantization...")
    # 3. ExecuTorch backend and PT2E APIs: configure Ethos-U quantization and
    # insert observers that record activation ranges during calibration.
    # Course wrapper: make_quantizer() contains the ExecuTorch Ethos-U
    # quantizer APIs. prepare_pt2e() below is a direct PyTorch API call.
    quantizer = make_quantizer(args)
    prepared = prepare_pt2e(graph_module, quantizer)

    calibration_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.calibration_images, transform=image_transform(spec)),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    if len(calibration_loader.dataset) == 0:
        raise ValueError(f"No calibration images found in {args.calibration_images}")

    print(f"Calibrating on up to {args.num_calib} images from {args.calibration_images}...")
    with torch.inference_mode():
        for index, (image, _) in enumerate(calibration_loader):
            if index >= args.num_calib:
                break
            # Native libraries can repeat the same message for every image.
            with suppress_stderr():
                prepared(image)

    print("Converting to INT8...")
    # 4. PT2E API: replace the calibrated observers with quantized operations.
    quantized_model = convert_pt2e(prepared)

    validation = None
    if args.validate_image is not None:
        validation = validate_predictions(
            fp32_model=model,
            int8_model=quantized_model,
            image_path=args.validate_image,
            spec=spec,
            labels_path=artifact_dir / spec["labels"],
        )

    with torch.inference_mode():
        # Course wrapper: export_no_guards() contains the PyTorch export APIs.
        exported_quantized = export_no_guards(quantized_model, example_inputs)

    # 5. Course wrappers: these functions contain the ExecuTorch APIs that
    # delegate supported regions and package delegate data inside the .pte.
    # replace_boundary_quant_nodes() is course support code for Cortex-M boundaries.
    print(f"Lowering with Ethos-U target {args.target}...")
    edge = lower_to_ethosu(exported_quantized, args)
    edge = replace_boundary_quant_nodes(edge)
    program = to_executorch(edge)
    write_pte(program, out)

    # Supporting output: record target settings and delegation for comparison.
    metadata = {
        "model": spec["name"],
        "backend": "ethosu",
        "precision": "int8",
        "insert_lrn": args.insert_lrn,
        "target": args.target,
        "system_config": args.system_config,
        "memory_mode": args.memory_mode,
        "pte": str(out),
        "pte_size_mb": out.stat().st_size / 1_000_000.0,
        "num_calib": min(args.num_calib, len(calibration_loader.dataset)),
        "validation": validation,
        "delegation": inspect_delegation(edge),
    }
    metadata_path = out.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"Wrote {out}")
    print(f"PTE size: {metadata['pte_size_mb']:.2f} MB")
    print_delegation(metadata["delegation"])


class MobileNetV2WithLRN:
    # Insert LocalResponseNorm between the feature extractor and classifier to
    # show how one unsupported operation splits an Ethos-U delegated region.

    def __init__(self, base_model: Any):
        import torch
        import torch.nn.functional as F

        class Wrapper(torch.nn.Module):
            def __init__(self, inner: Any):
                super().__init__()
                self.features = inner.features
                self.classifier = inner.classifier
                self.lrn = torch.nn.LocalResponseNorm(5)

            def forward(self, image: Any) -> Any:
                x = self.features(image)
                x = self.lrn(x)
                x = F.adaptive_avg_pool2d(x, (1, 1))
                x = torch.flatten(x, 1)
                return self.classifier(x)

        self.module = Wrapper(base_model)

    def eval(self) -> Any:
        return self.module.eval()


def add_ethosu_args(parser: argparse.ArgumentParser) -> None:
    # Add the target, memory, and Vela options shared by quantization and lowering.
    parser.add_argument("--target", default="ethos-u85-256")
    parser.add_argument("--system-config", default="Ethos_U85_SYS_DRAM_Mid")
    parser.add_argument("--memory-mode", default="Shared_Sram")
    parser.add_argument("--extra-flag", action="append", default=["--output-format=raw"])


def make_quantizer(args: argparse.Namespace) -> Any:
    # Course wrapper around ExecuTorch's EthosUCompileSpec and EthosUQuantizer APIs.
    from executorch.backends.arm.ethosu import EthosUCompileSpec
    from executorch.backends.arm.quantizer import EthosUQuantizer, get_symmetric_quantization_config

    compile_spec = EthosUCompileSpec(
        target=args.target,
        system_config=args.system_config,
        memory_mode=args.memory_mode,
        extra_flags=args.extra_flag,
    )
    quantizer = EthosUQuantizer(compile_spec)
    quantizer.set_global(get_symmetric_quantization_config())
    return quantizer


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

    return edge.to_executorch(
        config=ExecutorchBackendConfig(extract_delegate_segments=False)
    )


def replace_boundary_quant_nodes(edge: Any) -> Any:
    # Rewrite delegate-boundary quantization into the Cortex-M runtime form.
    from executorch.backends.cortex_m.passes.replace_quant_nodes_pass import (
        ReplaceQuantNodesPass,
    )

    return edge.transform([ReplaceQuantNodesPass()])


def export_no_guards(model: Any, example_inputs: tuple[Any, ...]) -> Any:
    # Course wrapper around two torch.export.export() calls. The second export
    # removes input-value guards from the first exported graph module.
    import torch

    first = torch.export.export(model, example_inputs)
    graph_module = first.module(check_guards=False)
    return torch.export.export(graph_module, example_inputs)


def image_transform(spec: dict[str, Any]) -> Any:
    # Use the same resize, crop, and normalization as deployment.
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


def validate_predictions(
    *,
    fp32_model: Any,
    int8_model: Any,
    image_path: Path,
    spec: dict[str, Any],
    labels_path: Path,
) -> dict[str, Any]:
    # Compare FP32 and converted INT8 predictions before backend lowering.
    import torch
    from PIL import Image

    if not image_path.exists():
        raise FileNotFoundError(f"Validation image not found: {image_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    labels = json.loads(labels_path.read_text())
    image = image_transform(spec)(Image.open(image_path).convert("RGB")).unsqueeze(0)

    with torch.inference_mode():
        fp32_output = fp32_model(image)[0]
        int8_output = int8_model(image)[0]

    result = {
        "image": str(image_path),
        "fp32": prediction_summary(fp32_output, labels),
        "converted_int8": prediction_summary(int8_output, labels),
    }
    print_prediction_summary("FP32 reference", result["fp32"])
    print_prediction_summary("Converted INT8 before Ethos-U lowering", result["converted_int8"])
    return result


def prediction_summary(output: Any, labels: list[str], *, top_k: int = 5) -> dict[str, Any]:
    # Summarize logits and labelled probabilities for a validation check.
    import torch

    probabilities = torch.softmax(output, dim=0)
    scores, indices = torch.topk(probabilities, top_k)
    sorted_logits, _ = torch.sort(output, descending=True)
    return {
        "logit_min": float(output.min()),
        "logit_max": float(output.max()),
        "top1_top2_logit_margin": float(sorted_logits[0] - sorted_logits[1]),
        "predictions": [
            {"label": labels[int(index)], "score": float(score)}
            for score, index in zip(scores, indices)
        ],
    }


def print_prediction_summary(title: str, summary: dict[str, Any]) -> None:
    print(f"\n{title}")
    print("=" * len(title))
    for rank, prediction in enumerate(summary["predictions"], start=1):
        print(f"{rank}. {prediction['label']}: {prediction['score'] * 100:.2f}%")
    print(
        "Logits: "
        f"min={summary['logit_min']:.4f}, "
        f"max={summary['logit_max']:.4f}, "
        f"top-1/top-2 margin={summary['top1_top2_logit_margin']:.4f}"
    )


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
    warnings.filterwarnings("ignore", category=UserWarning, message="The given buffer is not writable.*")
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=r"erase_node\(.*\) on an already erased node",
    )
    logging.getLogger().setLevel(logging.ERROR)


@contextmanager
def suppress_stderr() -> Iterator[None]:
    # Used only in repeated calibration loops where native output would appear
    # once per image. Python tracebacks are printed after stderr is restored.
    # Set COURSE_DEBUG=1 to keep native diagnostics visible.
    if os.environ.get("COURSE_DEBUG") == "1":
        yield
        return
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
