#!/usr/bin/env python3
from __future__ import annotations

import argparse
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys

MODULE2_SCRIPTS = Path(__file__).resolve().parents[1]
if str(MODULE2_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(MODULE2_SCRIPTS))

from core.mobilenetv2_core import MobileNetV2Runner
from _support.web.image_classifier_page import render_page


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a MobileNetV2 image classification page.")
    parser.add_argument("--spec", type=Path, default=Path("module-2-cnn-inference-with-quantization/artifacts/mobilenetv2/model_spec.json"))
    parser.add_argument("--runtime", choices=["pytorch", "executorch"], default="executorch")
    parser.add_argument("--pte", type=Path, default=Path("module-2-cnn-inference-with-quantization/artifacts/mobilenetv2/mobilenetv2_int8_xnnpack.pte"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8092)
    args = parser.parse_args()

    runner = MobileNetV2Runner(args.spec, pte_path=args.pte, runtime=args.runtime)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self._send_page()

        def do_POST(self) -> None:
            image_bytes, image_mime = parse_image_upload(self.headers, self.rfile)
            if not image_bytes:
                if self.path == "/classify":
                    self._send_json({"error": "No image uploaded"}, status=400)
                    return
                self._send_page()
                return
            try:
                result = runner.classify(image_bytes)
            except Exception as exc:
                if self.path == "/classify":
                    self._send_json({"error": str(exc)}, status=500)
                    return
                raise
            if self.path == "/classify":
                self._send_json(
                    {
                        "predictions": result.predictions,
                        "latency_ms": result.latency_ms,
                        "image_mime": image_mime,
                    }
                )
                return
            self._send_page(
                predictions=result.predictions,
                latency_ms=result.latency_ms,
                image_bytes=image_bytes,
                image_mime=image_mime,
            )

        def _send_page(
            self,
            *,
            predictions: list[dict[str, float | str]] | None = None,
            latency_ms: float | None = None,
            image_bytes: bytes | None = None,
            image_mime: str = "image/jpeg",
        ) -> None:
            body = render_page(
                runtime=args.runtime,
                model_name=runner.spec["name"],
                predictions=predictions,
                latency_ms=latency_ms,
                image_bytes=image_bytes,
                image_mime=image_mime,
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(body.encode("utf-8"))

        def _send_json(self, payload: dict[str, object], *, status: int = 200) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Serving MobileNetV2 classifier at http://{args.host}:{args.port}")
    server.serve_forever()


def parse_image_upload(headers: object, stream: object) -> tuple[bytes | None, str]:
    content_length = int(headers.get("Content-Length", "0"))  # type: ignore[attr-defined]
    content_type = headers.get("Content-Type", "")  # type: ignore[attr-defined]
    if content_length <= 0 or "multipart/form-data" not in content_type:
        return None, "image/jpeg"

    body = stream.read(content_length)  # type: ignore[attr-defined]
    message = BytesParser(policy=policy.default).parsebytes(
        f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8") + body
    )

    for part in message.iter_parts():
        if part.get_param("name", header="content-disposition") != "image":
            continue
        payload = part.get_payload(decode=True)
        if payload:
            return payload, part.get_content_type() or "image/jpeg"

    return None, "image/jpeg"


if __name__ == "__main__":
    main()
