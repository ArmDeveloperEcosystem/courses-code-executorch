#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy Module 3 ExecuTorch artifacts into Android assets.")
    parser.add_argument("--artifact-dir", type=Path, default=Path("module_3/artifacts/lraspp_mobilenetv3"))
    parser.add_argument("--android-assets", type=Path, default=Path("module_3/android/ObjectCutout/app/src/main/assets"))
    parser.add_argument(
        "--pte-name",
        default="lraspp_mobilenetv3_xnnpack.pte",
        help="Source .pte to package. Use lraspp_mobilenetv3_int8_xnnpack.pte for INT8.",
    )
    parser.add_argument(
        "--asset-pte-name",
        default="lraspp_mobilenetv3.pte",
        help="Destination asset name loaded by the Android app.",
    )
    args = parser.parse_args()

    args.android_assets.mkdir(parents=True, exist_ok=True)
    copies = [
        (args.artifact_dir / args.pte_name, args.android_assets / args.asset_pte_name),
        (args.artifact_dir / "model_spec.json", args.android_assets / "model_spec.json"),
        (args.artifact_dir / "voc_labels.json", args.android_assets / "voc_labels.json"),
    ]
    for source, _dest in copies:
        if not source.exists():
            raise FileNotFoundError(f"Missing {source}. Run the AOT scripts first.")

    for source, dest in copies:
        shutil.copy2(source, dest)
        print(f"Copied {source} -> {dest}")


if __name__ == "__main__":
    main()
