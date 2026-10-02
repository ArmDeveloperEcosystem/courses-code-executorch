#!/usr/bin/env python3
# Measure agreement between FP32 and INT8 segmentation cutouts.

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from segmentation_preprocessing import letterbox_image, preprocess_tensor
from validate_cutout import configure_quiet_output, largest_non_background


def main() -> None:
    # Run both ExecuTorch models and compare their selected foreground masks.
    parser = argparse.ArgumentParser(
        description="Compare FP32 and INT8 LR-ASPP segmentation outputs."
    )
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--fp32-pte", type=Path, required=True)
    parser.add_argument("--int8-pte", type=Path, required=True)
    parser.add_argument("--image", type=Path, default=None)
    args = parser.parse_args()

    configure_quiet_output()
    import torch
    from executorch.extension.pybindings.portable_lib import _load_for_executorch

    spec = json.loads(args.spec.read_text())
    labels = json.loads((args.spec.parent / spec["labels"]).read_text())
    image_path = args.image or args.spec.parent / spec["sample_image"]
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    input_tensor, crop = prepare_input(image_path, spec)
    fp32_mask = run_model(args.fp32_pte, input_tensor, spec, torch, _load_for_executorch)
    int8_mask = run_model(args.int8_pte, input_tensor, spec, torch, _load_for_executorch)

    background = spec["output"]["postprocessing"]["background_class"]
    fp32_class = largest_non_background(fp32_mask, background=background)
    int8_class = largest_non_background(int8_mask, background=background)
    fp32_foreground = crop_mask(fp32_mask == fp32_class, crop)
    int8_foreground = crop_mask(int8_mask == int8_class, crop)

    agreement, iou, dice = binary_agreement(fp32_foreground, int8_foreground)
    print(f"FP32 selected class: {labels[fp32_class]} ({fp32_class})")
    print(f"INT8 selected class: {labels[int8_class]} ({int8_class})")
    if fp32_class != int8_class:
        print("Warning: the models selected different foreground classes.")
    print(f"Pixel agreement: {agreement:.2%}")
    print(f"Foreground IoU: {iou:.2%}")
    print(f"Foreground Dice: {dice:.2%}")


def prepare_input(path: Path, spec: dict[str, Any]) -> tuple[Any, Any]:
    # Create one model input and retain the bounds of the unpadded image.
    from PIL import Image

    with Image.open(path) as source:
        image = source.convert("RGB")
    letterboxed = letterbox_image(image, spec)
    return preprocess_tensor(image, spec).unsqueeze(0), letterboxed


def run_model(
    pte_path: Path,
    input_tensor: Any,
    spec: dict[str, Any],
    torch: Any,
    loader: Any,
) -> Any:
    # Run a PTE and return the winning class at each pixel.
    if not pte_path.exists():
        raise FileNotFoundError(f"PTE not found: {pte_path}")
    # ExecuTorch Python runtime: load the .pte and invoke its forward method.
    model = loader(str(pte_path))
    with torch.inference_mode():
        output = model.forward((input_tensor,))[0]

    expected_shape = tuple(spec["output"]["shape"])
    if tuple(output.shape) != expected_shape:
        raise ValueError(
            f"Unexpected output shape {tuple(output.shape)} from {pte_path}; "
            f"expected {expected_shape}"
        )
    return output[0].argmax(dim=0).cpu().numpy()


def crop_mask(mask: Any, letterboxed: Any) -> Any:
    # Remove letterbox padding before calculating agreement.
    left = letterboxed.pad_left
    top = letterboxed.pad_top
    right = left + letterboxed.content_width
    bottom = top + letterboxed.content_height
    return mask[top:bottom, left:right]


def binary_agreement(first: Any, second: Any) -> tuple[float, float, float]:
    # Return pixel agreement, foreground IoU, and foreground Dice.
    import numpy as np

    intersection = np.logical_and(first, second).sum()
    union = np.logical_or(first, second).sum()
    foreground_pixels = first.sum() + second.sum()
    agreement = np.equal(first, second).mean()
    iou = intersection / union if union else 1.0
    dice = 2 * intersection / foreground_pixels if foreground_pixels else 1.0
    return float(agreement), float(iou), float(dice)


if __name__ == "__main__":
    main()
