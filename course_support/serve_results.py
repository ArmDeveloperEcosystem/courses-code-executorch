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
    results_dir = args.results.expanduser().resolve()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path.startswith("/plots/"):
                return self._serve_plot(results_dir / self.path.lstrip("/"))
            self._send_html(render_dashboard(results_dir))

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
    print(f"Serving results from {results_dir} at http://{args.host}:{args.port}")
    server.serve_forever()


def render_dashboard(results_dir: Path) -> str:
    results_dir = results_dir.expanduser().resolve()
    entries = []
    for path in sorted(results_dir.glob("*.json")):
        payload = json.loads(path.read_text())
        entries.append(read_result(path))

    entries.sort(key=entry_sort_key)

    rows = []
    figures = []
    for entry in entries:
        stats = entry["stats"]
        extra = entry["extra"]
        model = entry["model"]
        runtime = extra["runtime_label"]
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(model))}</td>"
            f"<td>{html.escape(str(runtime))}</td>"
            f"<td>{html.escape(str(entry['warmups']))}</td>"
            f"<td>{html.escape(str(stats['count']))}</td>"
            f"<td>{format_ms(stats, 'mean_ms')}</td>"
            f"<td>{format_ms(stats, 'median_ms')}</td>"
            f"<td>{format_ms(stats, 'min_ms')}</td>"
            f"<td>{format_ms(stats, 'max_ms')}</td>"
            "</tr>"
        )
        if extra["backend"] == "portable":
            continue
        for plot in extra["plots"]:
            plot_path = Path(plot)
            try:
                rel = plot_path.resolve().relative_to(results_dir)
            except ValueError:
                continue
            figures.append(
                "<figure>"
                f'<img src="/{html.escape(str(rel))}" alt="{html.escape(plot_path.name)}">'
                f"<figcaption>{html.escape(str(model))} - {html.escape(str(runtime))} - {html.escape(plot_path.name)}</figcaption>"
                "</figure>"
            )

    table_body = "\n".join(rows) or '<tr><td colspan="8">No result JSON files found.</td></tr>'
    figure_body = "\n".join(figures)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Inference Results</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #1f2933; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 1.5rem; }}
    th, td {{ border-bottom: 1px solid #d9e2ec; padding: 0.6rem; text-align: left; }}
    th {{ background: #f0f4f8; }}
    figure {{ display: inline-block; max-width: 48%; min-width: 320px; margin: 0.5rem 1rem 1rem 0; vertical-align: top; }}
    img {{ width: 100%; border: 1px solid #d9e2ec; }}
    figcaption {{ color: #52606d; font-size: 0.9rem; margin-top: 0.35rem; }}
  </style>
</head>
<body>
  <h1>Inference Results</h1>
  <table>
    <thead><tr><th>Model</th><th>Runtime</th><th>Warmups</th><th>Runs</th><th>Mean ms</th><th>Median ms</th><th>Min ms</th><th>Max ms</th></tr></thead>
    <tbody>{table_body}</tbody>
  </table>
  {figure_body}
</body>
</html>"""


def format_ms(stats: dict[str, object], key: str) -> str:
    value = stats.get(key)
    return f"{float(value):.2f}" if isinstance(value, int | float) else "-"


def read_result(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    extra = payload.get("extra")
    if not isinstance(extra, dict) or "runtime_label" not in extra or "backend" not in extra:
        raise RuntimeError(f"{path} uses an old result format. Regenerate it with module-1-transformer-inference-with-xnnpack/scripts/pi/benchmark_model.py.")
    return payload


def entry_sort_key(entry: dict[str, object]) -> tuple[str, int, str]:
    extra = entry["extra"]
    backend = extra["backend"] if isinstance(extra, dict) else ""
    backend_order = {"eager": 0, "portable": 1, "xnnpack": 2}
    return (str(entry["model"]), backend_order.get(str(backend), 99), str(extra.get("runtime_label", "")))


if __name__ == "__main__":
    main()
