#!/usr/bin/env python3
# Run a MobileNetV2 PTE and write its ExecuTorch runtime trace.

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    # Load the model with event tracing enabled and record repeated inference.
    parser = argparse.ArgumentParser(description="Capture an ETDump for MobileNetV2.")
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--pte", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.runs < 1:
        raise ValueError("--runs must be at least 1")

    import torch
    from executorch.extension.pybindings.portable_lib import _load_for_executorch

    spec = json.loads(args.spec.read_text())
    input_tensor = preprocess_image(args.image, spec)
    module = _load_for_executorch(str(args.pte), enable_etdump=True)
    if not module.has_etdump():
        raise RuntimeError("ExecuTorch was not built with ETDump support. Run build_etdump_runtime.sh first.")

    with torch.inference_mode():
        for _ in range(args.runs):
            module.forward((input_tensor,))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    module.write_etdump_result_to_file(str(args.out))
    if not args.out.exists() or args.out.stat().st_size == 0:
        raise RuntimeError("ExecuTorch did not write ETDump data. Rebuild the runtime with event tracing enabled.")
    print(f"Wrote {args.out} ({args.out.stat().st_size} bytes, {args.runs} runs)")


def preprocess_image(path: Path, spec: dict[str, Any]) -> Any:
    # Apply the image transform recorded in the shared model specification.
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
    with Image.open(path) as image:
        return transform(image.convert("RGB")).unsqueeze(0)


if __name__ == "__main__":
    main()
