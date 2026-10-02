#!/usr/bin/env python3
# Benchmark MobileNetV2 with PyTorch, FP32 XNNPACK, or INT8 XNNPACK.

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE2_SCRIPTS = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(MODULE2_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(MODULE2_SCRIPTS))

from _support.common import configure_quiet_output, load_executorch_program
from course_support.benchmarking import benchmark, print_stats, save_plots, write_results


def main() -> None:
    # Script setup: command-line options, model settings, and output paths.
    parser = argparse.ArgumentParser(description="Benchmark MobileNetV2 with PyTorch or ExecuTorch.")
    parser.add_argument("--runtime", choices=["pytorch", "executorch"], required=True)
    parser.add_argument("--backend", choices=["xnnpack", "int8_xnnpack"], default=None)
    parser.add_argument("--spec", type=Path, default=Path("module-2-cnn-inference-with-quantization/artifacts/mobilenetv2/model_spec.json"))
    parser.add_argument("--pte", type=Path, default=None)
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--plots", type=Path, default=None)
    args = parser.parse_args()

    configure_quiet_output()

    import torch

    spec = json.loads(args.spec.read_text())
    image = args.image or resolve_sample_image(args.spec, spec)
    if image is None or not image.exists():
        raise FileNotFoundError(
            "No benchmark image found. Pass --image /path/to/image.jpg, "
            "or rerun prepare_mobilenetv2.py in an environment that can download the sample image."
        )

    if args.runtime == "pytorch":
        if args.backend is not None:
            raise ValueError("--backend is only valid for --runtime executorch")
        backend = "eager"
    else:
        backend = args.backend or infer_backend(args.pte, args.out)

    runtime_label = format_runtime_label(args.runtime, backend)
    result_name = format_result_name(spec["name"], args.runtime, backend)
    labels = json.loads((args.spec.parent / spec["labels"]).read_text())
    input_tensor = preprocess_image(image, spec)

    if args.runtime == "pytorch":
        module = load_pytorch_model(args.spec, spec)

        def forward() -> Any:
            with torch.inference_mode():
                # PyTorch eager baseline: call the nn.Module directly.
                return module(input_tensor)

    else:
        if args.pte is None:
            raise ValueError("--pte is required for --runtime executorch")
        # ExecuTorch Python runtime binding: load the packaged .pte.
        module = load_executorch_program(args.pte)

        def forward() -> Any:
            with torch.inference_mode():
                # ExecuTorch runtime: invoke the exported forward method.
                return module.forward((input_tensor,))[0]

    output = forward()
    predictions = topk_predictions(output, labels)
    # The shared benchmark helper suppresses repeated native output only while
    # timing each forward call.
    times_ms, stats = benchmark(forward, warmups=args.warmups, runs=args.runs)
    plot_dir = args.plots or args.out.parent / "plots"
    plot_paths = save_plots(times_ms, title=result_name, out_dir=plot_dir)

    write_results(
        out=args.out,
        runtime=args.runtime,
        model=spec["name"],
        warmups=args.warmups,
        times_ms=times_ms,
        stats=stats,
        extra={
            "backend": backend,
            "runtime_label": runtime_label,
            "result_name": result_name,
            "precision": format_precision(backend),
            "image": str(image),
            "pte": str(args.pte) if args.pte else None,
            "predictions": predictions,
            "plots": [str(path) for path in plot_paths],
        },
    )
    print_predictions(predictions)
    print_stats(result_name, stats, warmups=args.warmups)
    print(f"Wrote {args.out}")


def load_pytorch_model(spec_path: Path, spec: dict[str, Any]) -> Any:
    # Reconstruct the eager baseline from the saved state dictionary.
    import torch
    from torchvision import models

    model = models.mobilenet_v2(weights=None)
    model.load_state_dict(torch.load(spec_path.parent / spec["weights"], map_location="cpu"))
    return model.eval()


def resolve_sample_image(spec_path: Path, spec: dict[str, Any]) -> Path | None:
    sample = spec.get("sample_image")
    return spec_path.parent / sample if sample else None


def preprocess_image(path: Path, spec: dict[str, Any]) -> Any:
    # Apply the resize, crop, and normalization recorded in model_spec.json.
    from PIL import Image
    import torchvision.transforms as transforms

    prep = spec["input"]["preprocessing"]
    transform = transforms.Compose(
        [
            transforms.Resize(prep["resize"]),
            transforms.CenterCrop(prep["center_crop"]),
            transforms.ToTensor(),
            transforms.Normalize(mean=prep["mean"], std=prep["std"]),
        ]
    )
    return transform(Image.open(path).convert("RGB")).unsqueeze(0)


def topk_predictions(output: Any, labels: list[str], *, k: int = 5) -> list[dict[str, float | str]]:
    # Convert classifier logits into labelled top-k probabilities.
    import torch

    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    scores, indices = torch.topk(probabilities, k)
    return [
        {
            "label": labels[int(index)],
            "score": float(score),
        }
        for score, index in zip(scores, indices)
    ]


def infer_backend(pte: Path | None, out: Path) -> str:
    # Infer a result label from artifact names when --backend is omitted.
    for path in (pte, out):
        if path is None:
            continue
        name = path.stem.lower()
        if "int8" in name and "xnnpack" in name:
            return "int8_xnnpack"
        if "xnnpack" in name:
            return "xnnpack"
    return "executorch"


def format_runtime_label(runtime: str, backend: str) -> str:
    if runtime == "pytorch":
        return "PyTorch eager"
    if backend == "xnnpack":
        return "ExecuTorch with XNNPACK"
    if backend == "int8_xnnpack":
        return "INT8 ExecuTorch with XNNPACK"
    return "ExecuTorch"


def format_result_name(model_name: str, runtime: str, backend: str) -> str:
    if runtime == "pytorch":
        return f"{model_name}_pytorch"
    return f"{model_name}_{backend}"


def format_precision(backend: str) -> str:
    if backend == "int8_xnnpack":
        return "int8"
    if backend == "xnnpack":
        return "fp32"
    return "fp32"


def print_predictions(predictions: list[dict[str, float | str]]) -> None:
    print("\nTop-5 predictions")
    print("=================")
    for index, pred in enumerate(predictions, start=1):
        print(f"{index}. {pred['label']}: {float(pred['score']) * 100:.2f}%")


if __name__ == "__main__":
    main()
