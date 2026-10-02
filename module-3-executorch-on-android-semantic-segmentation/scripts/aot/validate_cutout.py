#!/usr/bin/env python3
# Validate a segmentation .pte and render its mask, overlay, and cutout.

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import time
from typing import Any
import warnings

from segmentation_preprocessing import letterbox_image, preprocess_tensor


def main() -> None:
    # Script setup: command-line options, paths, and known dependency warnings.
    parser = argparse.ArgumentParser(description="Validate LR-ASPP .pte locally and write cutout previews.")
    parser.add_argument("--spec", type=Path, default=Path("module-3-executorch-on-android-semantic-segmentation/artifacts/lraspp_mobilenetv3/model_spec.json"))
    parser.add_argument("--pte", type=Path, default=Path("module-3-executorch-on-android-semantic-segmentation/artifacts/lraspp_mobilenetv3/lraspp_mobilenetv3_xnnpack.pte"))
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("module-3-executorch-on-android-semantic-segmentation/artifacts/lraspp_mobilenetv3/validation"))
    args = parser.parse_args()

    configure_quiet_output()

    import torch
    from executorch.extension.pybindings.portable_lib import _load_for_executorch

    spec = json.loads(args.spec.read_text())
    labels = json.loads((args.spec.parent / spec["labels"]).read_text())
    image_path = args.image or args.spec.parent / spec["sample_image"]
    if image_path is None or not image_path.exists():
        raise FileNotFoundError("Pass --image or run prepare_lraspp_mobilenetv3.py to create sample.jpg.")

    # Application preprocessing: create the NCHW tensor and retain crop geometry.
    input_tensor, letterboxed = preprocess_image(image_path, spec)

    # ExecuTorch Python runtime binding: load the packaged .pte program.
    model = _load_for_executorch(str(args.pte))

    start = time.perf_counter()
    # ExecuTorch Python runtime: invoke the exported forward method.
    with torch.inference_mode():
        output = model.forward((input_tensor,))[0]
    inference_ms = (time.perf_counter() - start) * 1000.0

    expected_shape = tuple(spec["output"]["shape"])
    if tuple(output.shape) != expected_shape:
        raise ValueError(f"Unexpected output shape {tuple(output.shape)}; expected {expected_shape}")

    # Each output channel is one Pascal VOC class; argmax selects the winning
    # class independently for every pixel.
    mask = output[0].argmax(dim=0).cpu().numpy().astype("uint8")
    background_class = spec["output"]["postprocessing"]["background_class"]
    class_id = largest_non_background(mask, background=background_class)
    if class_id == background_class:
        print("No non-background class found; writing background mask previews.")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_outputs(letterboxed, mask, class_id, args.out_dir)

    print(f"Selected class: {labels[class_id]} ({class_id})")
    print(f"ExecuTorch forward: {inference_ms:.2f} ms")
    print(f"Wrote {args.out_dir / 'mask.png'}")
    print(f"Wrote {args.out_dir / 'overlay.png'}")
    print(f"Wrote {args.out_dir / 'cutout.png'}")


def preprocess_image(path: Path, spec: dict[str, Any]) -> tuple[Any, Any]:
    # Return both the model tensor and geometry needed to crop the output.
    from PIL import Image

    with Image.open(path) as source:
        image = source.convert("RGB")
    return preprocess_tensor(image, spec).unsqueeze(0), letterbox_image(image, spec)


def largest_non_background(mask: Any, *, background: int) -> int:
    # Choose the non-background class occupying the most pixels.
    import numpy as np

    ids, counts = np.unique(mask, return_counts=True)
    candidates = [(int(i), int(c)) for i, c in zip(ids, counts) if int(i) != background]
    if not candidates:
        return background
    return max(candidates, key=lambda item: item[1])[0]


def write_outputs(letterboxed: Any, mask: Any, class_id: int, out_dir: Path) -> None:
    # Remove letterbox padding and write mask, overlay, and transparent cutout.
    from PIL import Image
    import numpy as np

    left = letterboxed.pad_left
    top = letterboxed.pad_top
    right = left + letterboxed.content_width
    bottom = top + letterboxed.content_height
    image = letterboxed.image.crop((left, top, right, bottom))
    selected = mask[top:bottom, left:right] == class_id
    mask_image = Image.fromarray((selected.astype("uint8") * 255), mode="L")

    rgba = image.convert("RGBA")
    cutout = Image.new("RGBA", rgba.size, (255, 255, 255, 0))
    cutout.paste(rgba, mask=mask_image)

    overlay_rgba = np.asarray(rgba).copy()
    tint = np.array([42, 112, 196, 120], dtype=np.uint8)
    overlay_rgba[selected] = (0.55 * overlay_rgba[selected] + 0.45 * tint).astype(np.uint8)
    overlay = Image.fromarray(overlay_rgba, mode="RGBA")

    mask_image.save(out_dir / "mask.png")
    overlay.save(out_dir / "overlay.png")
    cutout.save(out_dir / "cutout.png")


def configure_quiet_output() -> None:
    # Supporting setup: filter known dependency warnings without hiding errors.
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

if __name__ == "__main__":
    main()
