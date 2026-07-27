#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a raw MobileNetV2 input tensor for the FVP runner.")
    parser.add_argument("--spec", type=Path, default=Path("module-4-ethos-npu-accelerated-inference/artifacts/mobilenetv2/model_spec.json"))
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("module-4-ethos-npu-accelerated-inference/artifacts/fvp_runs/mobilenetv2_int8/i0.bin"))
    parser.add_argument("--metadata", type=Path, default=None)
    args = parser.parse_args()

    with args.spec.open() as f:
        spec = json.load(f)

    image_path = args.image
    if image_path is None:
        sample = spec.get("sample_image")
        if not sample:
            raise FileNotFoundError("Pass --image or prepare a sample_image in model_spec.json.")
        image_path = args.spec.parent / sample
    if not image_path.exists():
        raise FileNotFoundError(f"Could not find input image: {image_path}")

    tensor = preprocess_image(image_path, spec)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tensor.contiguous().numpy().astype("float32").tofile(args.out)

    metadata_path = args.metadata or args.out.with_suffix(".json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model": spec["name"],
        "image": str(image_path),
        "input": str(args.out),
        "shape": list(tensor.shape),
        "dtype": "float32",
        "nbytes": args.out.stat().st_size,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"Wrote {args.out}")
    print(f"Input shape: {metadata['shape']}")
    print(f"Input size:  {metadata['nbytes']} bytes")
    print(f"Wrote {metadata_path}")


def preprocess_image(path: Path, spec: dict[str, Any]) -> Any:
    from PIL import Image
    import torchvision.transforms as transforms

    prep = spec["input"]["preprocessing"]
    image = Image.open(path).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize(prep["resize"]),
            transforms.CenterCrop(prep["center_crop"]),
            transforms.ToTensor(),
            transforms.Normalize(mean=prep["mean"], std=prep["std"]),
        ]
    )
    return transform(image).unsqueeze(0)


if __name__ == "__main__":
    main()
