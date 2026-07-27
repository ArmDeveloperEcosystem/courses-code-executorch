#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a C++ ImageNet label table.")
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    labels = json.loads(args.labels.read_text())
    if (
        not isinstance(labels, list)
        or len(labels) != 1000
        or not all(isinstance(label, str) for label in labels)
    ):
        raise ValueError("Expected a JSON list containing 1000 ImageNet label strings")

    lines = [
        "#pragma once",
        "",
        f"static constexpr int kImageNetLabelCount = {len(labels)};",
        "static constexpr const char* kImageNetLabels[kImageNetLabelCount] = {",
    ]
    lines.extend(f"    {json.dumps(label)}," for label in labels)
    lines.extend(["};", ""])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
