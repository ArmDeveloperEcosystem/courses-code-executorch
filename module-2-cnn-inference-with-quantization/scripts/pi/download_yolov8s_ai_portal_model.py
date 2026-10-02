#!/usr/bin/env python3
# Download the pinned YOLOv8 package and normalize its application contract.

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any


DEFAULT_REPO_ID = "Arm/yolov8s-int8-xnnpack-executorch"
DEFAULT_REVISION = "1a77e7aa86df18d1db1e0c74e40000c665fbd5d9"
COCO_LABELS = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def main() -> None:
    # Fetch the model snapshot and prepare the files consumed by the web demo.
    parser = argparse.ArgumentParser(
        description="Download the AI Portal YOLOv8s INT8 ExecuTorch with XNNPACK model package."
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "module-2-cnn-inference-with-quantization/artifacts/yolov8s_int8_xnnpack"
        ),
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Model repository revision. The default AI Portal model uses the course-pinned revision.",
    )
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    revision = args.revision
    if revision is None and args.repo_id == DEFAULT_REPO_ID:
        revision = DEFAULT_REVISION

    args.out.mkdir(parents=True, exist_ok=True)
    snapshot_path = Path(
        snapshot_download(
            repo_id=args.repo_id,
            revision=revision,
            allow_patterns=["*.pte", "*.json", "*.txt", "*.md", "*.yaml", "*.yml"],
        )
    )

    copied = copy_model_files(snapshot_path, args.out)
    generated = ensure_app_contract(args.out, args.repo_id, snapshot_path.name)

    print(f"Downloaded {args.repo_id}")
    for path in [*copied, *generated]:
        print(f"Wrote {path}")
    print(f"Model folder: {args.out}")


def copy_model_files(source: Path, out: Path) -> list[Path]:
    # Flatten supported files from the snapshot into the course artifact folder.
    copied: list[Path] = []
    destinations: set[Path] = set()
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".pte", ".json", ".txt", ".md", ".yaml", ".yml"}:
            continue
        destination = out / path.name
        if destination in destinations:
            continue
        destinations.add(destination)
        if path.resolve() == destination.resolve():
            continue
        shutil.copy2(path, destination)
        copied.append(destination)
    return copied


def ensure_app_contract(out: Path, repo_id: str, revision: str) -> list[Path]:
    # Ensure labels and model_spec.json contain everything the runner expects.
    written: list[Path] = []

    labels_path = out / "coco_labels.json"
    if not labels_path.exists():
        labels_path.write_text(json.dumps(COCO_LABELS, indent=2) + "\n")
        written.append(labels_path)

    pte_name = select_pte(out).name
    spec_path = out / "model_spec.json"
    if spec_path.exists():
        spec = json.loads(spec_path.read_text())
        changed = normalize_spec(spec, repo_id, revision, pte_name)
        if changed:
            spec_path.write_text(json.dumps(spec, indent=2) + "\n")
            written.append(spec_path)
    else:
        write_default_spec(spec_path, repo_id, revision, pte_name)
        written.append(spec_path)

    return written


def select_pte(out: Path) -> Path:
    # Prefer the optimized INT8 artifact when a package contains several .pte files.
    pte_files = sorted(out.glob("*.pte"))
    if not pte_files:
        raise FileNotFoundError(f"No .pte artifact found in {out}")
    for path in pte_files:
        if "optimized" in path.name.lower():
            return path
    for path in pte_files:
        if "int8" in path.name.lower():
            return path
    return pte_files[0]


def normalize_spec(
    spec: dict[str, Any], repo_id: str, revision: str, pte_name: str
) -> bool:
    # Add missing contract fields while preserving metadata supplied by the package.
    before = json.dumps(spec, sort_keys=True)
    spec.setdefault("adapter", "yolov8_detection")
    spec.setdefault("name", "yolov8s_int8_xnnpack")
    spec.setdefault("model_family", "yolov8")
    spec.setdefault("model_variant", "yolov8s")
    spec.setdefault("task", "object_detection")
    spec.setdefault("runtime", "executorch")
    spec.setdefault("backend", "xnnpack")
    spec.setdefault("precision", "int8")
    spec.setdefault("optimizations", ["xnnpack", "kleidiai"])
    spec.setdefault("license", "AGPL-3.0-or-later")
    spec.setdefault("license_url", f"https://huggingface.co/{repo_id}")
    spec["pte"] = pte_name
    spec["labels"] = "coco_labels.json"
    source = spec.setdefault("source", {})
    source.update(
        {
            "ai_portal": "Arm AI Portal model garden",
            "huggingface_repo": repo_id,
            "huggingface_url": f"https://huggingface.co/{repo_id}",
            "revision": revision,
        }
    )
    spec.setdefault(
        "inputs",
        [
            {
                "name": "images",
                "shape": [1, 3, 640, 640],
                "dtype": "float32",
                "layout": "nchw",
                "color": "rgb",
                "value_range": [0.0, 1.0],
            }
        ],
    )
    spec.setdefault(
        "outputs",
        [
            {
                "name": "predictions",
                "shape": [1, 84, "N"],
                "dtype": "float32",
                "description": "YOLOv8 box and class predictions.",
            }
        ],
    )
    spec.setdefault(
        "preprocessing",
        {
            "resize": "letterbox_to_640",
            "normalize": "divide_by_255",
            "channel_order": "rgb",
        },
    )
    spec.setdefault(
        "postprocessing",
        {
            "box_format": "xywh",
            "confidence_threshold": 0.25,
            "iou_threshold": 0.7,
            "max_detections": 50,
        },
    )
    spec["postprocessing"]["iou_threshold"] = 0.7
    return before != json.dumps(spec, sort_keys=True)


def write_default_spec(
    path: Path, repo_id: str, revision: str, pte_name: str
) -> None:
    # Write the known YOLOv8 contract when the downloaded package has no spec.
    spec: dict[str, Any] = {
        "adapter": "yolov8_detection",
        "name": "yolov8s_int8_xnnpack",
        "model_family": "yolov8",
        "model_variant": "yolov8s",
        "task": "object_detection",
        "runtime": "executorch",
        "backend": "xnnpack",
        "precision": "int8",
        "optimizations": ["xnnpack", "kleidiai"],
        "license": "AGPL-3.0-or-later",
        "license_url": f"https://huggingface.co/{repo_id}",
        "pte": pte_name,
        "source": {
            "ai_portal": "Arm AI Portal model garden",
            "huggingface_repo": repo_id,
            "huggingface_url": f"https://huggingface.co/{repo_id}",
            "revision": revision,
        },
        "inputs": [
            {
                "name": "images",
                "shape": [1, 3, 640, 640],
                "dtype": "float32",
                "layout": "nchw",
                "color": "rgb",
                "value_range": [0.0, 1.0],
            }
        ],
        "outputs": [
            {
                "name": "predictions",
                "shape": [1, 84, "N"],
                "dtype": "float32",
                "description": "YOLOv8 box and class predictions.",
            }
        ],
        "labels": "coco_labels.json",
        "preprocessing": {
            "resize": "letterbox_to_640",
            "normalize": "divide_by_255",
            "channel_order": "rgb",
        },
        "postprocessing": {
            "box_format": "xywh",
            "confidence_threshold": 0.25,
            "iou_threshold": 0.7,
            "max_detections": 50,
        },
    }
    path.write_text(json.dumps(spec, indent=2) + "\n")


if __name__ == "__main__":
    main()
