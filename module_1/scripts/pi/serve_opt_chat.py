#!/usr/bin/env python3
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
from urllib.parse import parse_qs


MODULE1_SCRIPTS = Path(__file__).resolve().parents[1]
if str(MODULE1_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(MODULE1_SCRIPTS))

from core.opt_chat_core import load_opt_chat_runner
from _support.web.opt_chat_page import render_page


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a tiny OPT ExecuTorch chat page.")
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--pte", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    runner = load_opt_chat_runner(args.spec, args.pte)
    startup = dict(runner.startup)
    startup["max_new_tokens"] = args.max_new_tokens

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self._send_page("", "", None)

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            values = parse_qs(self.rfile.read(length).decode("utf-8"))
            prompt = values.get("prompt", [""])[0]
            result = (
                runner.generate(
                    prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                )
                if prompt
                else None
            )
            self._send_page(
                prompt,
                result.output if result else "",
                result.metrics.as_dict() if result else None,
            )

        def _send_page(self, prompt: str, output: str, metrics: dict[str, float | int] | None) -> None:
            body = render_page(prompt, output, metrics, startup)
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(body.encode("utf-8"))

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Serving OPT demo at http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
