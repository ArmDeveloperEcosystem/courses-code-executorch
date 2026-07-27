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
from urllib.error import URLError
from urllib.request import Request, urlopen
import warnings


SAMPLE_URL = "https://upload.wikimedia.org/wikipedia/commons/0/04/Labrador_Retriever_%281210559%29.jpg"
SAMPLE_SHA256 = "b01fefdd381ef37925a52b72501e299068045700f8e593a4fee70e04ef0713a8"
SAMPLE_SOURCE_PAGE = "https://commons.wikimedia.org/wiki/File:Labrador_Retriever_(1210559).jpg"


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare MobileNetV2 assets and write a model spec.")
    parser.add_argument("--out", type=Path, default=Path("module_2/artifacts/mobilenetv2"))
    parser.add_argument("--weights-name", default="mobilenet_v2.pth")
    parser.add_argument("--labels-name", default="imagenet_labels.json")
    parser.add_argument("--sample-name", default="sample.jpg")
    parser.add_argument(
        "--sample-url",
        default=SAMPLE_URL,
        help="Image used by the workflow examples and web demo.",
    )
    parser.add_argument("--sample-sha256", default=None)
    args = parser.parse_args()

    configure_quiet_output()

    with suppress_stderr():
        import torch
        from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

    args.out.mkdir(parents=True, exist_ok=True)
    weights_path = args.out / args.weights_name
    labels_path = args.out / args.labels_name
    sample_path = args.out / args.sample_name

    weights = MobileNet_V2_Weights.IMAGENET1K_V1
    model = mobilenet_v2(weights=weights).eval()
    torch.save(model.state_dict(), weights_path)

    labels = weights.meta["categories"]
    labels_path.write_text(json.dumps(labels, indent=2) + "\n")

    expected_sample_sha256 = args.sample_sha256
    if expected_sample_sha256 is None and args.sample_url == SAMPLE_URL:
        expected_sample_sha256 = SAMPLE_SHA256

    sample_sha256 = None
    if args.sample_url:
        print(f"Downloading sample image to {sample_path}...")
        try:
            sample_sha256 = download_sample(args.sample_url, sample_path, expected_sample_sha256)
        except URLError as exc:
            print(f"Sample image download skipped: {exc}")

    spec = {
        "adapter": "mobilenetv2",
        "name": "mobilenetv2",
        "model_family": "mobilenet",
        "model_variant": "mobilenet_v2",
        "task": "image_classification",
        "weights": args.weights_name,
        "labels": args.labels_name,
        "sample_image": args.sample_name if sample_path.exists() else None,
        "sample_image_source": {
            "url": args.sample_url,
            "description": "Yellow Labrador retriever portrait",
            "license": "CC0 1.0 Universal Public Domain Dedication",
            "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
            "source_page": SAMPLE_SOURCE_PAGE if args.sample_url == SAMPLE_URL else args.sample_url,
            "source": "Wikimedia Commons",
            "author": "Chiemsee2024",
            "sha256": sample_sha256,
        }
        if sample_path.exists()
        else None,
        "input": {
            "shape": [1, 3, 224, 224],
            "dtype": "float32",
            "layout": "nchw",
            "preprocessing": {
                "resize": 256,
                "center_crop": 224,
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        },
    }

    spec_path = args.out / "model_spec.json"
    spec_path.write_text(json.dumps(spec, indent=2) + "\n")

    print(f"Wrote {weights_path}")
    print(f"Wrote {labels_path}")
    if sample_path.exists():
        print(f"Wrote {sample_path}")
    print(f"Wrote {spec_path}")


def configure_quiet_output() -> None:
    """Keep dependency warnings from hiding the useful lab output."""
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


def download_sample(url: str, path: Path, expected_sha256: str | None) -> str:
    if path.exists():
        current_sha256 = file_sha256(path)
        if expected_sha256 is None or current_sha256 == expected_sha256:
            return current_sha256

    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (ExecuTorch course setup)",
        },
    )
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
