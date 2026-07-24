#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
from collections import Counter
from contextlib import contextmanager
import json
import logging
import os
from pathlib import Path
from typing import Any
import warnings


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantize MobileNetV2 and lower to ExecuTorch Ethos-U.")
    parser.add_argument("--spec", type=Path, default=Path("module_4/artifacts/mobilenetv2/model_spec.json"))
    parser.add_argument("--calibration-images", type=Path, required=True)
    parser.add_argument("--num-calib", type=int, default=16)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--validate-image",
        type=Path,
        default=None,
        help="Compare FP32 and converted INT8 predictions before Ethos-U lowering.",
    )
    parser.add_argument("--insert-lrn", action="store_true", help="Insert LocalResponseNorm to demonstrate fragmentation.")
    parser.add_argument("--etrecord", action="store_true", help="Write an ETRecord matching the generated PTE.")
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

    with suppress_stderr():
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

    with suppress_stderr():
        base_model = mobilenet_v2(weights=None).eval()
        base_model.load_state_dict(torch.load(artifact_dir / spec["weights"], map_location="cpu"))
        model = MobileNetV2WithLRN(base_model).eval() if args.insert_lrn else base_model

    example_inputs = (torch.randn(*spec["input"]["shape"]),)

    print("Exporting model for PT2E quantization...")
    with torch.inference_mode(), suppress_stderr():
        exported = export(model, example_inputs)
        graph_module = exported.module(check_guards=False)

    print("Preparing Ethos-U quantization...")
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
            with suppress_stderr():
                prepared(image)

    print("Converting to INT8...")
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

    with torch.inference_mode(), suppress_stderr():
        exported_quantized = export_no_guards(quantized_model, example_inputs)

    print(f"Lowering with Ethos-U target {args.target}...")
    edge = lower_to_ethosu(exported_quantized, args)
    edge = replace_boundary_quant_nodes(edge)
    edge_for_etrecord = copy.deepcopy(edge) if args.etrecord else None
    program = to_executorch(edge)
    write_pte(program, out)

    etrecord_path = None
    if args.etrecord:
        etrecord_path = out.with_suffix(".etrecord")
        write_etrecord(edge_for_etrecord, program, etrecord_path)

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
        "etrecord": str(etrecord_path) if etrecord_path else None,
        "num_calib": min(args.num_calib, len(calibration_loader.dataset)),
        "validation": validation,
        "delegation": inspect_delegation(edge),
    }
    metadata_path = out.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"Wrote {out}")
    print(f"PTE size: {metadata['pte_size_mb']:.2f} MB")
    if etrecord_path:
        print(f"Wrote {etrecord_path}")
    print_delegation(metadata["delegation"])


class MobileNetV2WithLRN:
    """MobileNetV2 variant used to show how one model change can split delegation."""

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
    parser.add_argument("--target", default="ethos-u85-256")
    parser.add_argument("--system-config", default="Ethos_U85_SYS_DRAM_Mid")
    parser.add_argument("--memory-mode", default="Shared_Sram")
    parser.add_argument("--extra-flag", action="append", default=["--output-format=raw"])


def make_quantizer(args: argparse.Namespace) -> Any:
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
    from executorch.exir import ExecutorchBackendConfig

    return edge.to_executorch(
        config=ExecutorchBackendConfig(extract_delegate_segments=False)
    )


def replace_boundary_quant_nodes(edge: Any) -> Any:
    from executorch.backends.cortex_m.passes.replace_quant_nodes_pass import (
        ReplaceQuantNodesPass,
    )

    return edge.transform([ReplaceQuantNodesPass()])


def export_no_guards(model: Any, example_inputs: tuple[Any, ...]) -> Any:
    import torch

    first = torch.export.export(model, example_inputs)
    graph_module = first.module(check_guards=False)
    return torch.export.export(graph_module, example_inputs)


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


def validate_predictions(
    *,
    fp32_model: Any,
    int8_model: Any,
    image_path: Path,
    spec: dict[str, Any],
    labels_path: Path,
) -> dict[str, Any]:
    import torch
    from PIL import Image

    if not image_path.exists():
        raise FileNotFoundError(f"Validation image not found: {image_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    labels = json.loads(labels_path.read_text())
    image = image_transform(spec)(Image.open(image_path).convert("RGB")).unsqueeze(0)

    with torch.inference_mode(), suppress_stderr():
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
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(program, "save"):
        program.save(str(path))
        return
    with path.open("wb") as f:
        if hasattr(program, "write_to_file"):
            program.write_to_file(f)
        else:
            f.write(getattr(program, "buffer", None) or getattr(program, "_buffer"))


def write_etrecord(edge: Any, program: Any, path: Path) -> None:
    from executorch.devtools import generate_etrecord

    if edge is None:
        raise ValueError("ETRecord generation needs a copy of the edge program.")
    path.parent.mkdir(parents=True, exist_ok=True)
    generate_etrecord(str(path), edge, program)


def configure_quiet_output() -> None:
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
