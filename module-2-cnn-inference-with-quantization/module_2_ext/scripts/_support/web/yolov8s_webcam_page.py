from __future__ import annotations


def render_page(*, model_name: str, camera: str, port: int) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>YOLOv8s ExecuTorch Webcam</title>
  <style>
    :root {{
      --paper: #f7fbff;
      --ink: #172033;
      --muted: #58677f;
      --line: #c8d8f0;
      --accent: #265fae;
      --accent-strong: #153b73;
      --warn: #b55336;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Avenir Next", "Segoe UI", sans-serif;
      color: var(--ink);
      background: #eef5fb;
    }}
    .shell {{
      width: min(1180px, calc(100vw - 28px));
      margin: 1rem auto 2rem;
      display: grid;
      gap: 1rem;
    }}
    .hero {{
      display: flex;
      justify-content: space-between;
      align-items: end;
      gap: 1rem;
      padding: 1.1rem 1.25rem;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.92);
    }}
    h1 {{
      margin: 0;
      font-size: 2.2rem;
      line-height: 1;
      letter-spacing: 0;
    }}
    .meta {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      justify-content: flex-end;
      color: var(--muted);
      font-size: 0.92rem;
    }}
    .pill {{
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 6px 10px;
      background: #fff;
    }}
    .stage {{
      display: grid;
      grid-template-columns: minmax(320px, 1fr) 280px;
      gap: 18px;
      align-items: start;
    }}
    .video {{
      width: 100%;
      aspect-ratio: 16 / 9;
      object-fit: contain;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
    }}
    .stats {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--paper);
      padding: 16px;
      display: grid;
      gap: 12px;
    }}
    .metric {{
      display: flex;
      justify-content: space-between;
      border-bottom: 1px solid var(--line);
      padding-bottom: 10px;
      gap: 12px;
    }}
    .metric strong {{ color: var(--accent); }}
    .metric.alert strong {{ color: var(--warn); }}
    @media (max-width: 820px) {{
      .hero {{ align-items: start; flex-direction: column; }}
      .meta {{ justify-content: flex-start; }}
      .stage {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main class="shell">
    <section class="hero">
      <h1>YOLOv8s Webcam Detection</h1>
      <div class="meta">
        <span class="pill">{model_name}</span>
        <span class="pill">ExecuTorch with XNNPACK</span>
        <span class="pill">camera {camera}</span>
        <span class="pill">:{port}</span>
      </div>
    </section>
    <section class="stage">
      <img id="frame" class="video" alt="Webcam detection stream">
      <aside class="stats">
        <div class="metric"><span>Detections</span><strong id="detections">-</strong></div>
        <div class="metric"><span>Inference</span><strong id="inference">-</strong></div>
        <div class="metric"><span>Pipeline</span><strong id="total">-</strong></div>
        <div class="metric"><span>Browser round trip</span><strong id="roundtrip">-</strong></div>
        <div class="metric alert"><span>Status</span><strong id="status">Starting</strong></div>
      </aside>
    </section>
  </main>
  <script>
    const frame = document.getElementById('frame');
    const statusEl = document.getElementById('status');
    const roundTripEl = document.getElementById('roundtrip');
    function loadFrame() {{
      return new Promise((resolve, reject) => {{
        frame.onload = resolve;
        frame.onerror = reject;
        frame.src = '/frame.jpg?t=' + Date.now();
      }});
    }}
    async function refresh() {{
      const started = performance.now();
      try {{
        statusEl.textContent = 'Running';
        await loadFrame();
        const response = await fetch('/stats?t=' + Date.now());
        const stats = await response.json();
        document.getElementById('detections').textContent = stats.detections;
        document.getElementById('inference').textContent = stats.inference_ms.toFixed(1) + ' ms';
        document.getElementById('total').textContent = stats.total_ms.toFixed(1) + ' ms';
        statusEl.textContent = stats.status;
        roundTripEl.textContent = (performance.now() - started).toFixed(1) + ' ms';
      }} catch (error) {{
        statusEl.textContent = 'Waiting';
      }}
      const elapsed = performance.now() - started;
      setTimeout(refresh, Math.max(150, 500 - elapsed));
    }}
    refresh();
  </script>
</body>
</html>"""
