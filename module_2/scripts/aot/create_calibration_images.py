#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import shutil
from typing import Any
import warnings


def main() -> None:
    parser = argparse.ArgumentParser(description="Create ImageFolder calibration images from local images.")
    parser.add_argument("--out", type=Path, default=Path("module_2/artifacts/mobilenetv2/calibration_images"))
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-image", type=Path)
    source.add_argument("--source-dir", type=Path)
    parser.add_argument("--count", type=int, default=16)
    args = parser.parse_args()

    configure_quiet_output()
    if args.count <= 0:
        raise ValueError("--count must be greater than zero")
    source_paths = [args.source_image] if args.source_image is not None else find_images(args.source_dir)
    write_local_subset(source_paths, args.out, args.count)


def find_images(source_dir: Path | None) -> list[Path]:
    if source_dir is None:
        return []
    image_paths = sorted(
        path
        for path in source_dir.rglob("*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found under {source_dir}")
    return image_paths


def write_local_subset(source_paths: list[Path], out_dir: Path, count: int) -> None:
    from PIL import Image

    if not source_paths:
        raise FileNotFoundError("No calibration source images were provided")
    for source_path in source_paths:
        if source_path is None or not source_path.exists():
            raise FileNotFoundError(f"Calibration source image not found: {source_path}")

    if out_dir.exists():
        shutil.rmtree(out_dir)
    class_dir = out_dir / "local"
    class_dir.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        source = source_paths[index % len(source_paths)]
        with Image.open(source) as image:
            variant = make_variant(image.convert("RGB"), index)
            variant.save(class_dir / f"img_{index:04d}.jpg", quality=95)

    print(f"Wrote {count} calibration images to {out_dir}")
    print(f"Source images: {len(source_paths)}")


def make_variant(image: Any, index: int) -> Any:
    from PIL import Image

    width, height = image.size
    scale = (1.0, 0.92, 0.84, 0.76)[index % 4]
    crop_width = max(1, int(width * scale))
    crop_height = max(1, int(height * scale))
    x_slots = max(1, width - crop_width)
    y_slots = max(1, height - crop_height)
    left = ((index // 4) * 37) % x_slots
    top = ((index // 4) * 23) % y_slots
    resized = image.crop((left, top, left + crop_width, top + crop_height)).resize(
        (width, height),
        Image.Resampling.BICUBIC,
    )
    if index % 2:
        resized = resized.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    return resized


def configure_quiet_output() -> None:
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    warnings.filterwarnings("ignore", category=FutureWarning)
    logging.getLogger().setLevel(logging.ERROR)


if __name__ == "__main__":
    main()
