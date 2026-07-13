from __future__ import annotations

import html
import json
from pathlib import Path
import sys


MODULE1_SCRIPTS = Path(__file__).resolve().parents[2]
if str(MODULE1_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(MODULE1_SCRIPTS))

from _support.common import format_ms


def render_page(default_prompt: str, startup: dict[str, str | float | int]) -> str:
    payload = {
        "defaultPrompt": default_prompt,
        "topk": int(startup["topk"]),
    }
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>DistilBERT ExecuTorch Demo</title>
  <style>
    :root {{
      --paper: rgba(250, 253, 250, 0.94);
      --ink: #17201b;
      --muted: #5f6f65;
      --line: #c6d7cb;
      --accent: #267255;
      --accent-strong: #174934;
      --highlight: #e3f3df;
      --shadow: 0 18px 42px rgba(25, 64, 45, 0.12);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      font-family: "Avenir Next", "Segoe UI", sans-serif;
      background:
        linear-gradient(120deg, rgba(242, 249, 238, 0.9), rgba(236, 247, 244, 0.78)),
        repeating-linear-gradient(135deg, rgba(25, 89, 60, 0.055) 0 1px, transparent 1px 18px);
    }}
    .shell {{
      width: min(980px, calc(100vw - 2rem));
      margin: 1rem auto;
      padding-bottom: 2rem;
    }}
    .hero {{
      padding: 1.2rem 1.35rem;
      border: 1px solid rgba(38, 114, 85, 0.18);
      border-radius: 8px;
      background: linear-gradient(135deg, rgba(255,255,255,0.94), rgba(239,249,241,0.88));
      box-shadow: var(--shadow);
    }}
    h1 {{
      margin: 0;
      font-family: "Iowan Old Style", "Palatino Linotype", serif;
      font-size: 2.35rem;
      line-height: 1;
    }}
    .lede {{
      margin: 0.7rem 0 0;
      color: var(--muted);
      line-height: 1.5;
      max-width: 68ch;
    }}
    .meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.55rem;
      margin-top: 0.9rem;
    }}
    .pill {{
      min-height: 2rem;
      padding: 0.35rem 0.7rem;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: rgba(255,255,255,0.78);
      font-size: 0.92rem;
    }}
    .panel {{
      margin-top: 1rem;
      padding: 1rem;
      border: 1px solid rgba(38, 114, 85, 0.18);
      border-radius: 8px;
      background: var(--paper);
      box-shadow: var(--shadow);
    }}
    .label {{
      display: block;
      margin-bottom: 0.45rem;
      color: var(--muted);
      font-size: 0.92rem;
    }}
    textarea {{
      width: 100%;
      min-height: 7.5rem;
      padding: 0.9rem 1rem;
      border: 1px solid var(--line);
      border-radius: 8px;
      color: var(--ink);
      font: inherit;
      line-height: 1.5;
      resize: vertical;
      background: rgba(255,255,255,0.94);
    }}
    .toolbar {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.7rem;
      align-items: center;
      margin-top: 0.8rem;
    }}
    button {{
      min-height: 2.4rem;
      padding: 0.55rem 0.8rem;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: rgba(255,255,255,0.78);
      color: var(--ink);
      font: inherit;
      cursor: pointer;
    }}
    .status {{
      color: var(--muted);
      font-size: 0.92rem;
    }}
    .suggestions {{
      display: grid;
      gap: 0.7rem;
      margin-top: 1rem;
    }}
    .suggestion {{
      display: grid;
      grid-template-columns: minmax(8rem, 0.24fr) minmax(0, 1fr);
      gap: 0.8rem;
      align-items: center;
      padding: 0.85rem;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: rgba(255,255,255,0.78);
      text-align: left;
    }}
    .token {{
      font-weight: 700;
      color: var(--accent-strong);
    }}
    .sentence {{
      line-height: 1.45;
    }}
    mark {{
      padding: 0.08rem 0.25rem;
      border-radius: 6px;
      background: var(--highlight);
      color: var(--accent-strong);
    }}
    @media (max-width: 760px) {{
      h1 {{ font-size: 2rem; }}
      .suggestion {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main class="shell">
    <section class="hero">
      <h1>DistilBERT Autocomplete</h1>
      <p class="lede">A Pi-hosted autocomplete demo using the lowered XNNPACK model. Suggestions refresh as the sentence changes.</p>
      <div class="meta">
        <div class="pill">Model: {html.escape(str(startup["model"]))}</div>
        <div class="pill">Context: {startup["seq_len"]} tokens</div>
        <div class="pill">Top suggestions: {startup["topk"]}</div>
        <div class="pill">Server load: {format_ms(float(startup["load_ms"]))}</div>
      </div>
    </section>
    <section class="panel">
      <label class="label" for="prompt">Sentence</label>
      <textarea id="prompt">{html.escape(default_prompt)}</textarea>
      <div class="toolbar">
        <button type="button" id="reset">Reset</button>
        <span class="status" id="status">Ready</span>
      </div>
    </section>
    <section class="panel">
      <div class="label">Suggestions</div>
      <div class="suggestions" id="suggestions"></div>
    </section>
  </main>
  <script>
    const config = {json.dumps(payload)};
    const prompt = document.getElementById("prompt");
    const reset = document.getElementById("reset");
    const status = document.getElementById("status");
    const suggestions = document.getElementById("suggestions");
    let controller = null;
    let timer = null;

    function escapeHtml(value) {{
      const node = document.createElement("span");
      node.textContent = value;
      return node.innerHTML;
    }}

    function highlight(sentence, token) {{
      const escaped = escapeHtml(sentence);
      const escapedToken = escapeHtml(token);
      if (!escapedToken) return escaped;
      return escaped.replace(escapedToken, `<mark>${{escapedToken}}</mark>`);
    }}

    function render(items) {{
      if (!items.length) {{
        suggestions.innerHTML = "<div class='status'>Start typing to see suggestions.</div>";
        return;
      }}
      suggestions.innerHTML = items.map((item, index) => `
        <button type="button" class="suggestion" data-index="${{index}}">
          <span class="token">${{escapeHtml(item.token)}}</span>
          <span class="sentence">${{highlight(item.sentence, item.token)}}</span>
        </button>
      `).join("");
      for (const button of document.querySelectorAll(".suggestion")) {{
        button.addEventListener("click", () => {{
          const item = items[Number(button.dataset.index)];
          prompt.value = item ? item.sentence : prompt.value;
          schedule();
        }});
      }}
    }}

    async function suggest() {{
      const value = prompt.value.trim();
      if (controller) controller.abort();
      controller = new AbortController();
      status.textContent = "Running...";
      try {{
        const params = new URLSearchParams({{ prompt: value, topk: String(config.topk) }});
        const response = await fetch(`/suggest?${{params.toString()}}`, {{ signal: controller.signal }});
        const data = await response.json();
        render(data.suggestions || []);
        status.textContent = data.latency_ms ? `Updated in ${{Math.round(data.latency_ms)}} ms` : "Waiting for text";
      }} catch (error) {{
        if (error.name !== "AbortError") status.textContent = "Suggestion request failed";
      }}
    }}

    function schedule() {{
      clearTimeout(timer);
      timer = setTimeout(suggest, 220);
    }}

    reset.addEventListener("click", () => {{
      prompt.value = config.defaultPrompt;
      schedule();
    }});
    prompt.addEventListener("input", schedule);
    suggest();
  </script>
</body>
</html>"""
