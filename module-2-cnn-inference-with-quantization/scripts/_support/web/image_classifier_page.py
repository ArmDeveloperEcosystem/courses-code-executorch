from __future__ import annotations

import base64
from html import escape
import json


def render_page(
    *,
    runtime: str,
    model_name: str,
    predictions: list[dict[str, float | str]] | None = None,
    latency_ms: float | None = None,
    image_bytes: bytes | None = None,
    image_mime: str = "image/jpeg",
) -> str:
    image_src = ""
    if image_bytes:
        encoded = base64.b64encode(image_bytes).decode("ascii")
        image_src = f"data:{image_mime};base64,{encoded}"

    initial_state = json.dumps(
        {
            "predictions": predictions or [],
            "latency_ms": latency_ms,
            "image_src": image_src,
        }
    )
    latency = f"{latency_ms:.2f} ms" if latency_ms is not None else "Ready"

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MobileNetV2 Image Classifier</title>
  <style>
    :root {{
      color-scheme: light;
      --paper: rgba(248, 251, 255, 0.94);
      --ink: #172033;
      --muted: #5e6c86;
      --line: #c4d3e8;
      --panel: #f4f8fd;
      --accent: #2f5f9f;
      --accent-strong: #1c3f72;
      --warm: #b65333;
      --blue: #2f78c4;
      --shadow: 0 18px 42px rgba(32, 69, 120, 0.12);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Avenir Next", "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        linear-gradient(120deg, rgba(240, 247, 255, 0.92), rgba(232, 244, 250, 0.78)),
        repeating-linear-gradient(135deg, rgba(47, 95, 159, 0.055) 0 1px, transparent 1px 18px);
    }}
    .shell {{
      width: min(1120px, calc(100vw - 2rem));
      margin: 1rem auto 2rem;
    }}
    .hero {{
      display: flex;
      justify-content: space-between;
      align-items: end;
      gap: 1rem;
      padding: 1.2rem 1.35rem;
      border: 1px solid rgba(47, 95, 159, 0.18);
      border-radius: 8px;
      background: linear-gradient(135deg, rgba(255,255,255,0.94), rgba(238,246,255,0.88));
      box-shadow: var(--shadow);
    }}
    h1 {{
      margin: 0;
      font-family: "Iowan Old Style", "Palatino Linotype", serif;
      font-size: clamp(2rem, 3vw, 2.65rem);
      line-height: 1;
      letter-spacing: 0;
    }}
    .meta {{
      display: flex;
      flex-wrap: wrap;
      justify-content: flex-end;
      gap: 0.55rem;
      color: var(--muted);
      font-size: 0.92rem;
    }}
    .pill {{
      min-height: 2rem;
      padding: 0.36rem 0.7rem;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: rgba(255,255,255,0.82);
    }}
    .grid {{
      display: grid;
      grid-template-columns: minmax(300px, 0.96fr) minmax(320px, 1.04fr);
      gap: 1rem;
      margin-top: 1rem;
      align-items: start;
    }}
    .panel {{
      border: 1px solid rgba(47, 95, 159, 0.18);
      border-radius: 8px;
      background: var(--paper);
      box-shadow: var(--shadow);
    }}
    .control {{
      padding: 1rem;
      display: grid;
      gap: 0.85rem;
    }}
    .label {{
      display: block;
      color: var(--muted);
      font-size: 0.92rem;
      margin-bottom: 0.45rem;
    }}
    .file-row {{
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 0.7rem;
      align-items: center;
    }}
    input[type="file"] {{
      width: 100%;
      border: 1px dashed #aeb8c6;
      border-radius: 8px;
      padding: 0.8rem;
      background: rgba(255,255,255,0.94);
      font: inherit;
      color: var(--muted);
    }}
    button {{
      min-height: 2.75rem;
      border: 1px solid transparent;
      border-radius: 8px;
      padding: 0.68rem 1rem;
      background: linear-gradient(180deg, var(--accent), var(--accent-strong));
      color: white;
      font: inherit;
      font-weight: 760;
      cursor: pointer;
      white-space: nowrap;
    }}
    button:disabled {{
      opacity: 0.7;
      cursor: progress;
    }}
    .stage {{
      position: relative;
      overflow: hidden;
      min-height: 390px;
      display: grid;
      place-items: center;
      background: linear-gradient(135deg, #edf5ff, #f8fbfe);
      border-top: 1px solid var(--line);
      border-radius: 0 0 8px 8px;
    }}
    .preview {{
      width: 100%;
      height: min(58vh, 560px);
      object-fit: contain;
      display: none;
    }}
    .empty {{
      color: var(--muted);
      font-weight: 650;
    }}
    .busy {{
      position: absolute;
      inset: 0;
      display: none;
      place-items: center;
      background: rgba(247, 250, 252, 0.72);
      backdrop-filter: blur(2px);
    }}
    .busy.active {{ display: grid; }}
    .spinner {{
      width: 2.2rem;
      height: 2.2rem;
      border: 3px solid rgba(18, 107, 100, 0.22);
      border-top-color: var(--accent);
      border-radius: 999px;
      animation: spin 0.82s linear infinite;
    }}
    @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
    .results {{
      padding: 1rem;
    }}
    .status-row {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 1rem;
      margin-bottom: 0.8rem;
      color: var(--muted);
    }}
    .latency {{
      color: var(--warm);
      font-weight: 800;
      font-size: 1.1rem;
    }}
    .predictions {{
      display: grid;
      gap: 0.68rem;
    }}
    .prediction {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 0.7rem;
      align-items: center;
      padding: 0.78rem;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
    }}
    .prediction span {{
      overflow-wrap: anywhere;
      font-weight: 680;
    }}
    .prediction strong {{
      color: var(--accent);
      font-variant-numeric: tabular-nums;
    }}
    .bar {{
      grid-column: 1 / -1;
      height: 0.42rem;
      overflow: hidden;
      border-radius: 999px;
      background: #dce4ed;
    }}
    .fill {{
      height: 100%;
      width: 0%;
      border-radius: inherit;
      background: linear-gradient(90deg, var(--blue), var(--accent));
      transition: width 180ms ease;
    }}
    .placeholder {{
      color: var(--muted);
      border: 1px dashed var(--line);
      border-radius: 8px;
      padding: 1rem;
      background: var(--panel);
    }}
    @media (max-width: 820px) {{
      .hero {{ align-items: start; flex-direction: column; }}
      .meta {{ justify-content: flex-start; }}
      .grid {{ grid-template-columns: 1fr; }}
      .file-row {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main class="shell">
    <section class="hero">
      <h1>MobileNetV2 Classifier</h1>
      <div class="meta">
        <span class="pill">{escape(model_name)}</span>
        <span class="pill">{escape(runtime)}</span>
        <span class="pill" id="header-latency">Model: {latency}</span>
      </div>
    </section>
    <section class="grid">
      <section class="panel">
        <form class="control" id="classifier-form" method="post" enctype="multipart/form-data">
          <div>
            <label class="label" for="image">Image</label>
            <div class="file-row">
              <input id="image" name="image" type="file" accept="image/*" required>
              <button id="classify" type="submit">Classify</button>
            </div>
          </div>
        </form>
        <div class="stage">
          <img class="preview" id="preview" src="" alt="Selected image">
          <div class="empty" id="empty">No image selected</div>
          <div class="busy" id="busy"><div class="spinner" aria-label="Classifying"></div></div>
        </div>
      </section>
      <section class="panel results">
        <div class="status-row">
          <span id="status">Ready</span>
          <span class="latency" id="latency">Model: {latency}</span>
        </div>
        <div class="predictions" id="predictions"></div>
      </section>
    </section>
  </main>
  <script>
    const initial = {initial_state};
    const form = document.getElementById("classifier-form");
    const fileInput = document.getElementById("image");
    const button = document.getElementById("classify");
    const busy = document.getElementById("busy");
    const preview = document.getElementById("preview");
    const empty = document.getElementById("empty");
    const status = document.getElementById("status");
    const latency = document.getElementById("latency");
    const headerLatency = document.getElementById("header-latency");
    const predictions = document.getElementById("predictions");
    let objectUrl = null;

    function setPreview(src) {{
      if (objectUrl) URL.revokeObjectURL(objectUrl);
      objectUrl = src && src.startsWith("blob:") ? src : null;
      preview.src = src || "";
      preview.style.display = src ? "block" : "none";
      empty.style.display = src ? "none" : "block";
    }}

    function setBusy(isBusy) {{
      busy.classList.toggle("active", isBusy);
      button.disabled = isBusy;
      button.textContent = isBusy ? "Classifying..." : "Classify";
    }}

    function setLatency(modelValue, roundTripValue = null) {{
      const text = roundTripValue ? `Model: ${{modelValue}} · Round trip: ${{roundTripValue}}` : `Model: ${{modelValue}}`;
      latency.textContent = text;
      headerLatency.textContent = text;
    }}

    function renderPredictions(items) {{
      if (!items || !items.length) {{
        predictions.innerHTML = '<div class="placeholder">No predictions yet</div>';
        return;
      }}
      predictions.innerHTML = items.map((item) => {{
        const score = Number(item.score || 0);
        const percent = Math.max(0, Math.min(100, score * 100));
        const label = String(item.label || "").replace(/[&<>"']/g, (c) => ({{
          "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
        }}[c]));
        return `
          <div class="prediction">
            <span>${{label}}</span>
            <strong>${{percent.toFixed(1)}}%</strong>
            <div class="bar"><div class="fill" style="width: ${{percent.toFixed(1)}}%"></div></div>
          </div>
        `;
      }}).join("");
    }}

    fileInput.addEventListener("change", () => {{
      const file = fileInput.files && fileInput.files[0];
      if (!file) {{
        setPreview("");
        return;
      }}
      setPreview(URL.createObjectURL(file));
      status.textContent = "Ready";
    }});

    form.addEventListener("submit", async (event) => {{
      event.preventDefault();
      const file = fileInput.files && fileInput.files[0];
      if (!file) {{
        status.textContent = "Select an image first";
        return;
      }}
      setBusy(true);
      status.textContent = "Running inference on the Pi";
      setLatency("Working...");
      const requestStart = performance.now();
      try {{
        const response = await fetch("/classify", {{
          method: "POST",
          body: new FormData(form),
        }});
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || "Classification failed");
        renderPredictions(payload.predictions || []);
        const roundTripMs = performance.now() - requestStart;
        setLatency(
          `${{Number(payload.latency_ms || 0).toFixed(2)}} ms`,
          `${{roundTripMs.toFixed(2)}} ms`
        );
        status.textContent = "Complete";
      }} catch (error) {{
        status.textContent = error.message || "Classification failed";
        setLatency("Error");
      }} finally {{
        setBusy(false);
      }}
    }});

    setPreview(initial.image_src || "");
    renderPredictions(initial.predictions || []);
    if (initial.latency_ms !== null && initial.latency_ms !== undefined) {{
      setLatency(`${{Number(initial.latency_ms).toFixed(2)}} ms`);
    }}
  </script>
</body>
</html>"""
