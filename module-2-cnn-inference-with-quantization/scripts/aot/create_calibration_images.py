#!/usr/bin/env python3
# Create a deterministic Open Images subset for PTQ calibration.
# Open Images records the selected files as CC BY 2.0. See the dataset notice
# in the course before downloading them.

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from course_support.open_images_calibration import prepare_calibration_subset


def main() -> None:
    # Download and verify the pinned Open Images files used for calibration.
    parser = argparse.ArgumentParser(description="Create a balanced Open Images calibration subset.")
    parser.add_argument("--out", type=Path, default=Path("module-2-cnn-inference-with-quantization/artifacts/mobilenetv2/calibration_images"))
    parser.add_argument("--data-dir", type=Path, default=Path("datasets"))
    parser.add_argument("--count", type=int, default=100)
    args = parser.parse_args()

    prepare_calibration_subset(
        data_dir=args.data_dir,
        out_dir=args.out,
        count=args.count,
    )


if __name__ == "__main__":
    main()
