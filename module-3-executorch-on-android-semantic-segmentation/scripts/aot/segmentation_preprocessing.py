#!/usr/bin/env python3
# Shared letterbox and normalization logic for calibration and validation.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LetterboxedImage:
    # Keep the resized image and padding needed to restore output coordinates.
    image: Any
    pad_left: int
    pad_top: int
    content_width: int
    content_height: int


def letterbox_image(image: Any, spec: dict[str, Any]) -> LetterboxedImage:
    # Resize without distortion and center the image on the model canvas.
    from PIL import Image

    prep = spec["input"]["preprocessing"]
    if prep.get("resize_mode") != "letterbox":
        raise ValueError("Module 3 expects preprocessing.resize_mode to be 'letterbox'.")

    target_width, target_height = prep["resize"]
    source_width, source_height = image.size
    scale = min(target_width / source_width, target_height / source_height)
    content_width = max(1, min(target_width, round(source_width * scale)))
    content_height = max(1, min(target_height, round(source_height * scale)))
    pad_left = (target_width - content_width) // 2
    pad_top = (target_height - content_height) // 2

    resized = image.convert("RGB").resize(
        (content_width, content_height),
        Image.Resampling.BILINEAR,
    )
    canvas = Image.new("RGB", (target_width, target_height), tuple(prep["pad_rgb"]))
    canvas.paste(resized, (pad_left, pad_top))
    return LetterboxedImage(canvas, pad_left, pad_top, content_width, content_height)


def preprocess_tensor(image: Any, spec: dict[str, Any]) -> Any:
    # Convert the letterboxed image to the normalized CHW model input.
    import torchvision.transforms.functional as functional

    letterboxed = letterbox_image(image, spec)
    prep = spec["input"]["preprocessing"]
    tensor = functional.to_tensor(letterboxed.image)
    return functional.normalize(tensor, mean=prep["mean"], std=prep["std"])
