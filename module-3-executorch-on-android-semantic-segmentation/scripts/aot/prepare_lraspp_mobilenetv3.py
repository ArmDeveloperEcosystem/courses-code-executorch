#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen
import warnings


SAMPLE_URL = "https://upload.wikimedia.org/wikipedia/commons/0/04/Labrador_Retriever_%281210559%29.jpg"
SAMPLE_SHA256 = "b01fefdd381ef37925a52b72501e299068045700f8e593a4fee70e04ef0713a8"
SAMPLE_SOURCE_PAGE = "https://commons.wikimedia.org/wiki/File:Labrador_Retriever_(1210559).jpg"


VOC_SEGMENTATION_LABELS = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare LR-ASPP MobileNetV3 assets for Android cutout.")
    parser.add_argument("--out", type=Path, default=Path("module-3-executorch-on-android-semantic-segmentation/artifacts/lraspp_mobilenetv3"))
    parser.add_argument("--weights-name", default="lraspp_mobilenetv3.pth")
    parser.add_argument("--labels-name", default="voc_labels.json")
    parser.add_argument("--sample-name", default="sample.jpg")
    parser.add_argument(
        "--sample-url",
        default=SAMPLE_URL,
        help="Image used by the validation, calibration, and Android examples.",
    )
    parser.add_argument("--sample-sha256", default=None)
    args = parser.parse_args()

    configure_quiet_output()

    with suppress_stderr():
        import torch
        from torchvision.models.segmentation import LRASPP_MobileNet_V3_Large_Weights, lraspp_mobilenet_v3_large

    args.out.mkdir(parents=True, exist_ok=True)
    weights_path = args.out / args.weights_name
    labels_path = args.out / args.labels_name
    sample_path = args.out / args.sample_name

    weights = LRASPP_MobileNet_V3_Large_Weights.DEFAULT
    with suppress_stderr():
        model = lraspp_mobilenet_v3_large(weights=weights).eval()
    torch.save(model.state_dict(), weights_path)
    labels_path.write_text(json.dumps(VOC_SEGMENTATION_LABELS, indent=2) + "\n")

    expected_sample_sha256 = args.sample_sha256
    if expected_sample_sha256 is None and args.sample_url == SAMPLE_URL:
        expected_sample_sha256 = SAMPLE_SHA256
    print(f"Downloading sample image to {sample_path}...")
    sample_sha256 = download_sample(args.sample_url, sample_path, expected_sample_sha256)

    spec = {
        "adapter": "lraspp_mobilenetv3_cutout",
        "name": "lraspp_mobilenetv3",
        "model_family": "semantic_segmentation",
        "model_variant": "lraspp_mobilenet_v3_large",
        "task": "object_cutout",
        "weights": args.weights_name,
        "labels": args.labels_name,
        "sample_image": args.sample_name,
        "sample_image_source": {
            "url": args.sample_url,
            "description": "Labrador Retriever (1210559).jpg",
            "license": "CC0 1.0 Universal Public Domain Dedication",
            "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
            "source_page": SAMPLE_SOURCE_PAGE if args.sample_url == SAMPLE_URL else args.sample_url,
            "source": "Wikimedia Commons",
            "author": "Chiemsee2024",
            "sha256": sample_sha256,
        },
        "input": {
            "shape": [1, 3, 520, 520],
            "dtype": "float32",
            "layout": "nchw",
            "preprocessing": {
                "resize": [520, 520],
                "resize_mode": "letterbox",
                "pad_rgb": [0, 0, 0],
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        },
        "output": {
            "shape": [1, len(VOC_SEGMENTATION_LABELS), 520, 520],
            "dtype": "float32",
            "layout": "nchw",
            "postprocessing": {
                "background_class": 0,
                "default_class": "largest_non_background",
            },
        },
    }
    spec_path = args.out / "model_spec.json"
    spec_path.write_text(json.dumps(spec, indent=2) + "\n")

    print(f"Wrote {weights_path}")
    print(f"Wrote {labels_path}")
    print(f"Wrote {sample_path}")
    print(f"Wrote {spec_path}")


def configure_quiet_output() -> None:
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


def download_sample(url: str, path: Path, expected_sha256: str | None) -> str:
    if path.exists():
        current_sha256 = file_sha256(path)
        if expected_sha256 is None or current_sha256 == expected_sha256:
            return current_sha256

    request = Request(url, headers={"User-Agent": "Mozilla/5.0 (ExecuTorch course setup)"})
    with urlopen(request, timeout=30) as response:
        data = response.read()

    downloaded_sha256 = hashlib.sha256(data).hexdigest()
    if expected_sha256 is not None and downloaded_sha256 != expected_sha256:
        raise ValueError(
            f"Sample image SHA256 mismatch: expected {expected_sha256}, got {downloaded_sha256}"
        )
    path.write_bytes(data)
    return downloaded_sha256


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
