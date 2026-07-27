#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a C++ string table from JSON labels.")
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    labels = json.loads(args.labels.read_text())
    if not isinstance(labels, list) or not labels or not all(isinstance(label, str) for label in labels):
        raise ValueError("Expected a non-empty JSON list of label strings")

    lines = [
        "#pragma once",
        "",
        "#include <cstddef>",
        "",
        f"inline constexpr std::size_t kImageNetLabelCount = {len(labels)};",
        "inline constexpr const char* kImageNetLabels[kImageNetLabelCount] = {",
    ]
    lines.extend(f"    {json.dumps(label)}," for label in labels)
    lines.extend(["};", ""])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
