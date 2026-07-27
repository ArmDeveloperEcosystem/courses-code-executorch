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


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare MobileNetV2 artifacts for the Ethos-U lab.")
    parser.add_argument("--out", type=Path, default=Path("module-4-ethos-npu-accelerated-inference/artifacts/mobilenetv2"))
    args = parser.parse_args()

    configure_quiet_output()

    with suppress_stderr():
        import torch
        from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

    args.out.mkdir(parents=True, exist_ok=True)
    weights_path = args.out / "mobilenet_v2.pth"
    labels_path = args.out / "imagenet_labels.json"
    sample_path = args.out / "sample.jpg"
    spec_path = args.out / "model_spec.json"

    print("Loading TorchVision MobileNetV2 weights...")
    with suppress_stderr():
        weights = MobileNet_V2_Weights.IMAGENET1K_V1
        model = mobilenet_v2(weights=weights).eval()
        torch.save(model.state_dict(), weights_path)

    labels_path.write_text(json.dumps(weights.meta["categories"], indent=2) + "\n")
    sample_sha256 = download_sample(SAMPLE_URL, sample_path, SAMPLE_SHA256)

    spec = {
        "name": "mobilenetv2",
        "weights_variant": "IMAGENET1K_V1",
        "weights": weights_path.name,
        "labels": labels_path.name,
        "sample_image": sample_path.name if sample_path.exists() else None,
        "sample_image_source": {
            "url": SAMPLE_URL,
            "description": "Yellow Labrador retriever portrait",
            "license": "CC0 1.0 Universal Public Domain Dedication",
            "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
            "source_page": SAMPLE_SOURCE_PAGE,
            "source": "Wikimedia Commons",
            "author": "Chiemsee2024",
            "sha256": sample_sha256,
        }
        if sample_path.exists()
        else None,
        "input": {
            "shape": [1, 3, 224, 224],
            "preprocessing": {
                "resize": 256,
                "center_crop": 224,
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        },
    }
    spec_path.write_text(json.dumps(spec, indent=2) + "\n")

    print(f"Wrote {weights_path}")
    print("Weights variant: IMAGENET1K_V1")
    print(f"Wrote {labels_path}")
    if sample_path.exists():
        print(f"Wrote {sample_path}")
        print(f"Sample SHA256: {sample_sha256}")
    print(f"Wrote {spec_path}")


def download_sample(url: str, path: Path, expected_sha256: str) -> str:
    if path.exists():
        current_sha256 = file_sha256(path)
        if current_sha256 == expected_sha256:
            return current_sha256

    request = Request(url, headers={"User-Agent": "executorch-on-arm-course/1.0"})
    with urlopen(request, timeout=30) as response:
        data = response.read()

    downloaded_sha256 = hashlib.sha256(data).hexdigest()
    if downloaded_sha256 != expected_sha256:
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


def configure_quiet_output() -> None:
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    warnings.filterwarnings("ignore", category=FutureWarning)
    logging.getLogger().setLevel(logging.ERROR)
    for name in ("torch", "torchvision"):
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


if __name__ == "__main__":
    main()
