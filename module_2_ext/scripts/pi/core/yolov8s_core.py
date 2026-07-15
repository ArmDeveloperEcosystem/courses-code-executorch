from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import time
from typing import Any
import warnings


@dataclass
class Detection:
    label: str
    confidence: float
    box: tuple[int, int, int, int]


@dataclass
class DetectionResult:
    frame_jpeg: bytes
    detections: list[Detection]
    inference_ms: float
    total_ms: float


class YoloV8Runner:
    def __init__(self, spec_path: Path):
        configure_quiet_output()
        self.spec_path = spec_path
        self.spec = json.loads(spec_path.read_text())
        self.labels = json.loads((spec_path.parent / self.spec["labels"]).read_text())
        self.input_size = int(self.spec["inputs"][0]["shape"][-1])
        self.confidence_threshold = float(self.spec["postprocessing"].get("confidence_threshold", 0.25))
        self.iou_threshold = float(self.spec["postprocessing"].get("iou_threshold", 0.45))
        self.max_detections = int(self.spec["postprocessing"].get("max_detections", 50))

        with suppress_stderr():
            from executorch.extension.pybindings.portable_lib import _load_for_executorch

            model_path = spec_path.parent / self.spec["pte"]
            if not model_path.exists():
                available = ", ".join(path.name for path in sorted(spec_path.parent.glob("*.pte"))) or "none"
                raise FileNotFoundError(
                    f"Could not find .pte artifact: {model_path}. "
                    f"Available .pte files in {spec_path.parent}: {available}. "
                    "Rerun download_yolov8s_ai_portal_model.py to refresh model_spec.json."
                )
            self.model = _load_for_executorch(str(model_path))

    def detect_frame(
        self,
        frame_bgr: Any,
        *,
        confidence_threshold: float | None = None,
        iou_threshold: float | None = None,
    ) -> DetectionResult:
        import cv2
        with suppress_stderr():
            import torch

        total_start = time.perf_counter()
        input_tensor, scale, pad = self._preprocess(frame_bgr)

        start = time.perf_counter()
        with torch.inference_mode(), suppress_stderr():
            raw_output = self.model.forward((input_tensor,))[0]
        inference_ms = (time.perf_counter() - start) * 1000.0

        detections = self._postprocess(
            raw_output,
            frame_bgr.shape[:2],
            scale,
            pad,
            self.confidence_threshold if confidence_threshold is None else confidence_threshold,
            self.iou_threshold if iou_threshold is None else iou_threshold,
        )
        annotated = draw_detections(frame_bgr.copy(), detections)
        ok, encoded = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 86])
        if not ok:
            raise RuntimeError("Could not encode annotated frame")

        return DetectionResult(
            frame_jpeg=encoded.tobytes(),
            detections=detections,
            inference_ms=inference_ms,
            total_ms=(time.perf_counter() - total_start) * 1000.0,
        )

    def _preprocess(self, frame_bgr: Any) -> tuple[Any, float, tuple[int, int]]:
        import cv2
        import numpy as np
        with suppress_stderr():
            import torch

        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        height, width = image_rgb.shape[:2]
        scale = min(self.input_size / height, self.input_size / width)
        resized_w = int(round(width * scale))
        resized_h = int(round(height * scale))
        resized = cv2.resize(image_rgb, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

        canvas = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        pad_x = (self.input_size - resized_w) // 2
        pad_y = (self.input_size - resized_h) // 2
        canvas[pad_y : pad_y + resized_h, pad_x : pad_x + resized_w] = resized

        tensor = torch.from_numpy(canvas).permute(2, 0, 1).contiguous().float().unsqueeze(0) / 255.0
        return tensor, scale, (pad_x, pad_y)

    def _postprocess(
        self,
        output: Any,
        frame_shape: tuple[int, int],
        scale: float,
        pad: tuple[int, int],
        conf_thres: float,
        iou_thres: float,
    ) -> list[Detection]:
        import numpy as np

        predictions = output.detach().cpu().numpy()
        if predictions.ndim == 3:
            predictions = predictions[0]
        if predictions.shape[0] in (84, 85):
            predictions = predictions.T
        if predictions.shape[1] < 84:
            raise RuntimeError(f"Unexpected YOLOv8 output shape: {output.shape}")

        boxes_xywh = predictions[:, :4]
        class_scores = predictions[:, 4:84]
        if class_scores.min() < 0.0 or class_scores.max() > 1.0:
            class_scores = sigmoid(class_scores)

        class_ids = class_scores.argmax(axis=1)
        scores = class_scores[np.arange(class_scores.shape[0]), class_ids]
        keep = scores >= conf_thres
        if not np.any(keep):
            return []

        boxes = xywh_to_xyxy(boxes_xywh[keep])
        scores = scores[keep]
        class_ids = class_ids[keep]
        selected = nms_by_class(boxes, scores, class_ids, iou_thres)

        frame_h, frame_w = frame_shape
        pad_x, pad_y = pad
        detections: list[Detection] = []
        for index in selected[: self.max_detections]:
            x1, y1, x2, y2 = boxes[index]
            x1 = int(np.clip((x1 - pad_x) / scale, 0, frame_w - 1))
            y1 = int(np.clip((y1 - pad_y) / scale, 0, frame_h - 1))
            x2 = int(np.clip((x2 - pad_x) / scale, 0, frame_w - 1))
            y2 = int(np.clip((y2 - pad_y) / scale, 0, frame_h - 1))
            if x2 <= x1 or y2 <= y1:
                continue
            class_id = int(class_ids[index])
            detections.append(
                Detection(
                    label=self.labels[class_id] if class_id < len(self.labels) else str(class_id),
                    confidence=float(scores[index]),
                    box=(x1, y1, x2, y2),
                )
            )
        return detections


def sigmoid(values: Any) -> Any:
    import numpy as np

    return 1.0 / (1.0 + np.exp(-values))


def xywh_to_xyxy(boxes: Any) -> Any:
    result = boxes.copy()
    result[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    result[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    result[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    result[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return result


def nms_by_class(boxes: Any, scores: Any, class_ids: Any, iou_threshold: float) -> list[int]:
    import numpy as np

    keep: list[int] = []
    for class_id in np.unique(class_ids):
        indexes = np.where(class_ids == class_id)[0]
        selected = nms(boxes[indexes], scores[indexes], iou_threshold)
        keep.extend(indexes[selected].tolist())
    keep.sort(key=lambda index: float(scores[index]), reverse=True)
    return keep


def nms(boxes: Any, scores: Any, iou_threshold: float) -> Any:
    import numpy as np

    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        ious = box_iou(boxes[i], boxes[order[1:]])
        order = order[1:][ious <= iou_threshold]
    return np.array(keep, dtype=np.int64)


def box_iou(box: Any, boxes: Any) -> Any:
    import numpy as np

    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    box_area = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
    boxes_area = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    return intersection / np.maximum(box_area + boxes_area - intersection, 1e-6)


def draw_detections(frame_bgr: Any, detections: list[Detection]) -> Any:
    import cv2

    for detection in detections:
        x1, y1, x2, y2 = detection.box
        color = color_for_label(detection.label)
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
        text = f"{detection.label} {detection.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        y_text = max(0, y1 - th - 8)
        cv2.rectangle(frame_bgr, (x1, y_text), (x1 + tw + 8, y_text + th + 8), color, -1)
        cv2.putText(
            frame_bgr,
            text,
            (x1 + 4, y_text + th + 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return frame_bgr


def color_for_label(label: str) -> tuple[int, int, int]:
    palette = [(38, 95, 174), (42, 157, 143), (190, 75, 48), (116, 87, 174), (206, 146, 38)]
    return palette[sum(ord(char) for char in label) % len(palette)]


def configure_quiet_output() -> None:
    os.environ.setdefault("GLOG_minloglevel", "2")
    os.environ.setdefault("KMP_WARNINGS", "0")
    os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning, message="The given buffer is not writable.*")
    logging.getLogger().setLevel(logging.ERROR)


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
