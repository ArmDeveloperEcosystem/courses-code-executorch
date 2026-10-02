#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
from urllib.parse import parse_qs, urlparse


MODULE1_SCRIPTS = Path(__file__).resolve().parents[1]
if str(MODULE1_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(MODULE1_SCRIPTS))

from core.distilbert_autocomplete_core import load_distilbert_autocomplete_runner
from _support.web.distilbert_autocomplete_page import render_page


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a live DistilBERT autocomplete page.")
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--pte", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--topk", type=int, default=10)
    args = parser.parse_args()

    runner = load_distilbert_autocomplete_runner(args.spec, args.pte)
    startup = dict(runner.startup)
    startup["topk"] = args.topk
    default_prompt = "The Raspberry Pi is a"

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/suggest":
                return self._send_suggestions(parsed.query)
            self._send_html(render_page(default_prompt, startup))

        def _send_suggestions(self, query: str) -> None:
            values = parse_qs(query)
            prompt = values.get("prompt", [""])[0]
            try:
                topk = int(values.get("topk", [args.topk])[0])
            except ValueError:
                topk = args.topk
            result = runner.suggest(prompt, topk=topk)
            self._send_json(result.as_dict())

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

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Serving DistilBERT autocomplete at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
