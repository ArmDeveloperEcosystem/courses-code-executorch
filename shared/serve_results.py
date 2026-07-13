#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a small ExecuTorch benchmark dashboard.")
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path.startswith("/plots/"):
                return self._serve_plot(args.results / self.path.lstrip("/"))
            self._send_html(render_dashboard(args.results))

        def _serve_plot(self, path: Path) -> None:
            if not path.exists() or path.suffix != ".png":
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.end_headers()
            self.wfile.write(path.read_bytes())

        def _send_html(self, body: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(body.encode("utf-8"))

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Serving results from {args.results} at http://{args.host}:{args.port}")
    server.serve_forever()


def render_dashboard(results_dir: Path) -> str:
    rows = []
    images = []
    for path in sorted(results_dir.glob("*.json")):
        payload = json.loads(path.read_text())
        stats = payload.get("stats")
        if not isinstance(stats, dict):
            continue
        extra = payload.get("extra", {})
        if not isinstance(extra, dict):
            extra = {}
        model = payload.get("model") or extra.get("model") or path.stem
        runtime = payload.get("runtime") or extra.get("runtime") or "unknown"
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(model))}</td>"
            f"<td>{html.escape(str(runtime))}</td>"
            f"<td>{format_ms(stats, 'mean_ms')}</td>"
            f"<td>{format_ms(stats, 'median_ms')}</td>"
            f"<td>{format_ms(stats, 'min_ms')}</td>"
            f"<td>{format_ms(stats, 'max_ms')}</td>"
            "</tr>"
        )
        for plot in extra.get("plots", []):
            plot_path = Path(plot)
            try:
                rel = plot_path.relative_to(results_dir)
            except ValueError:
                continue
            images.append(f'<img src="/{html.escape(str(rel))}" alt="{html.escape(plot_path.name)}">')

    table_body = "\n".join(rows) or '<tr><td colspan="6">No result JSON files found.</td></tr>'
    image_body = "\n".join(images)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>ExecuTorch Results</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #1f2933; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 1.5rem; }}
    th, td {{ border-bottom: 1px solid #d9e2ec; padding: 0.6rem; text-align: left; }}
    th {{ background: #f0f4f8; }}
    img {{ max-width: 48%; min-width: 320px; margin: 0.5rem 1rem 1rem 0; border: 1px solid #d9e2ec; }}
  </style>
</head>
<body>
  <h1>ExecuTorch Results</h1>
  <table>
    <thead><tr><th>Model</th><th>Runtime</th><th>Mean ms</th><th>Median ms</th><th>Min ms</th><th>Max ms</th></tr></thead>
    <tbody>{table_body}</tbody>
  </table>
  {image_body}
</body>
</html>"""


def format_ms(stats: dict[str, object], key: str) -> str:
    value = stats.get(key)
    return f"{float(value):.2f}" if isinstance(value, int | float) else "-"


if __name__ == "__main__":
    main()
