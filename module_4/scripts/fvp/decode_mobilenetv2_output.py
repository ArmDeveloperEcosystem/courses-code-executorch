#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable


DTYPE_BYTES = {
    "float32": 4,
    "int8": 1,
    "uint8": 1,
    "int32": 4,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode a raw MobileNetV2 output tensor from the FVP runner.")
    parser.add_argument("--output", type=Path, default=Path("module_4/artifacts/fvp_runs/mobilenetv2_int8/out-0.bin"))
    parser.add_argument("--labels", type=Path, default=Path("module_4/artifacts/mobilenetv2/imagenet_labels.json"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--dtype",
        choices=["auto", *DTYPE_BYTES.keys()],
        default="auto",
        help="Output tensor dtype. Use auto for the standard float32 MobileNetV2 output.",
    )
    args = parser.parse_args()

    if not args.output.exists():
        raise FileNotFoundError(f"Could not find output tensor: {args.output}")
    if not args.labels.exists():
        raise FileNotFoundError(f"Could not find labels: {args.labels}")

    labels = json.loads(args.labels.read_text())
    if not isinstance(labels, list) or not labels or not all(isinstance(label, str) for label in labels):
        raise ValueError("Labels file must contain a non-empty JSON list of strings")
    if not 1 <= args.top_k <= len(labels):
        raise ValueError(f"--top-k must be between 1 and {len(labels)}")
    dtype = infer_dtype(args.output, len(labels), args.dtype)
    scores = read_tensor(args.output, dtype)
    if len(scores) != len(labels):
        raise ValueError(
            f"Output has {len(scores)} values, but labels file has {len(labels)} entries. "
            "Pass --dtype if the automatic dtype choice is wrong."
        )

    heading = f"Top-{args.top_k} predictions"
    print(heading)
    print("=" * len(heading))
    if dtype == "float32":
        probabilities = softmax(scores)
        for rank, index in enumerate(topk_indices(probabilities, args.top_k), start=1):
            print(f"{rank}. {labels[index]}: {probabilities[index] * 100:.2f}%")
    else:
        print(f"Output dtype: {dtype}. Printing raw scores because quantization parameters are not available.")
        for rank, index in enumerate(topk_indices(scores, args.top_k), start=1):
            print(f"{rank}. {labels[index]}: {scores[index]}")

    print()
    print(f"Read {args.output}")


def infer_dtype(path: Path, num_labels: int, requested: str) -> str:
    if requested != "auto":
        return requested

    nbytes = path.stat().st_size
    if nbytes == num_labels * DTYPE_BYTES["float32"]:
        return "float32"
    if nbytes == num_labels * DTYPE_BYTES["int8"]:
        return "int8"
    if nbytes == num_labels * DTYPE_BYTES["int32"]:
        return "int32"
    raise ValueError(
        f"Could not infer dtype from {nbytes} bytes and {num_labels} labels. "
        "Pass --dtype explicitly."
    )


def read_tensor(path: Path, dtype: str) -> list[float]:
    import struct

    data = path.read_bytes()
    if dtype == "float32":
        return list(struct.unpack(f"<{len(data) // 4}f", data))
    if dtype == "int32":
        return [float(value) for value in struct.unpack(f"<{len(data) // 4}i", data)]
    if dtype == "int8":
        return [float(value) for value in struct.unpack(f"<{len(data)}b", data)]
    if dtype == "uint8":
        return [float(value) for value in data]
    raise ValueError(f"Unsupported dtype: {dtype}")


def softmax(values: list[float]) -> list[float]:
    max_value = max(values)
    exps = [math.exp(value - max_value) for value in values]
    total = sum(exps)
    return [value / total for value in exps]


def topk_indices(values: Iterable[float], k: int) -> list[int]:
    indexed = list(enumerate(values))
    indexed.sort(key=lambda item: item[1], reverse=True)
    return [index for index, _ in indexed[:k]]


if __name__ == "__main__":
    main()
