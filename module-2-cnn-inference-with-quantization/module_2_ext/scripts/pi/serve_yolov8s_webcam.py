#!/usr/bin/env python3
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading


MODULE2_EXT_SCRIPTS = Path(__file__).resolve().parents[1]
if str(MODULE2_EXT_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(MODULE2_EXT_SCRIPTS))

from _support.web.yolov8s_webcam_page import render_page
from core.yolov8s_core import YoloV8Runner, configure_quiet_output


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a YOLOv8s INT8 ExecuTorch with XNNPACK webcam demo.")
    parser.add_argument("--spec", type=Path, default=Path("module-2-cnn-inference-with-quantization/module_2_ext/artifacts/yolov8s_int8_xnnpack/model_spec.json"))
    parser.add_argument("--camera", default="0", help="OpenCV camera index or video path.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8094)
    parser.add_argument("--confidence", type=float, default=None)
    parser.add_argument("--iou", type=float, default=None)
    args = parser.parse_args()

    configure_quiet_output()

    import cv2

    camera_source: int | str = int(args.camera) if args.camera.isdigit() else args.camera
    cap = cv2.VideoCapture(camera_source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera/video source: {args.camera}")

    runner = YoloV8Runner(args.spec)
    lock = threading.Lock()
    latest: dict[str, object] = {
        "status": "Ready",
        "detections": 0,
        "inference_ms": 0.0,
        "total_ms": 0.0,
    }

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path.startswith("/frame.jpg"):
                return self._send_frame()
            if self.path.startswith("/stats"):
                return self._send_json(latest)
            self._send_html(render_page(model_name=runner.spec["name"], camera=args.camera, port=args.port))

        def _send_frame(self) -> None:
            with lock:
                ok, frame = cap.read()
                if not ok:
                    latest.update({"status": "No frame"})
                    self.send_error(503, "No camera frame available")
                    return
                result = runner.detect_frame(frame, confidence_threshold=args.confidence, iou_threshold=args.iou)
                latest.update(
                    {
                        "status": "Running",
                        "detections": len(result.detections),
                        "inference_ms": result.inference_ms,
                        "total_ms": result.total_ms,
                    }
                )
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(result.frame_jpeg)))
            self.end_headers()
            self.wfile.write(result.frame_jpeg)

        def _send_html(self, body: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(body.encode("utf-8"))

        def _send_json(self, payload: dict[str, object]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    try:
        server = ThreadingHTTPServer((args.host, args.port), Handler)
        print(f"Serving YOLOv8s webcam detection at http://{args.host}:{args.port}")
        server.serve_forever()
    finally:
        cap.release()


if __name__ == "__main__":
    main()
