#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import contextmanager
import logging
import os
from pathlib import Path
from typing import Any
import warnings


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a small ImageFolder calibration set from CIFAR-10.")
    parser.add_argument("--out", type=Path, default=Path("module_2/artifacts/mobilenetv2/calibration_images"))
    parser.add_argument("--data-root", type=Path, default=Path("module_2/artifacts/mobilenetv2/data"))
    parser.add_argument("--images-per-class", type=int, default=10)
    parser.add_argument("--classes", type=int, default=10)
    args = parser.parse_args()

    configure_quiet_output()

    with suppress_stderr():
        from torchvision import datasets, transforms
        from torchvision.utils import save_image

    args.out.mkdir(parents=True, exist_ok=True)

    # This mirrors the Lab 2 notebook: download CIFAR-10 once, then save a small
    # ImageFolder-style subset for PTQ calibration.
    transform = transforms.ToTensor()
    dataset = datasets.CIFAR10(
        root=args.data_root,
        train=True,
        download=True,
        transform=transform,
    )

    class_names = dataset.classes[: args.classes]
    counts = {class_name: 0 for class_name in class_names}

    for image, label in dataset:
        class_name = dataset.classes[label]
        if class_name not in counts:
            continue
        if counts[class_name] >= args.images_per_class:
            continue

        class_dir = args.out / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        index = counts[class_name]
        save_image(image, class_dir / f"img_{index:04d}.jpg")
        counts[class_name] += 1

        if all(count >= args.images_per_class for count in counts.values()):
            break

    print("Saved calibration images:")
    for class_name, count in counts.items():
        print(f"  {class_name}: {count} images")
    print(f"\nRoot folder: {args.out.resolve()}")


def configure_quiet_output() -> None:
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    warnings.filterwarnings("ignore", category=FutureWarning)
    logging.getLogger().setLevel(logging.ERROR)
    for name in ("torch", "torchvision"):
        logging.getLogger(name).setLevel(logging.ERROR)


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
