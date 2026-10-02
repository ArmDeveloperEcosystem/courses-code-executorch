from __future__ import annotations

import html
from pathlib import Path
import sys


MODULE1_SCRIPTS = Path(__file__).resolve().parents[2]
if str(MODULE1_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(MODULE1_SCRIPTS))

from _support.common import format_ms


def render_page(
    prompt: str,
    output: str,
    metrics: dict[str, float | int] | None,
    startup: dict[str, str | float | int],
) -> str:
    examples = [
        "The sun was setting over the mountains when",
        "At the edge of the forest stood a small cottage. Inside,",
        "In a distant future, humans and robots",
    ]
    example_buttons = "".join(
        f'<button type="button" class="example" data-prompt="{html.escape(text)}">{html.escape(text)}</button>'
        for text in examples
    )
    summary = (
        metrics_panel(metrics)
        if metrics
        else "<div class='hint'>The benchmark measured one forward pass. This demo generates multiple new tokens, so the full response takes several forward passes plus tokenization and sampling.</div>"
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>GPT-2 ExecuTorch Demo</title>
  <style>
    :root {{
      --bg: #f5ecdd;
      --paper: rgba(255, 252, 247, 0.92);
      --ink: #1d232d;
      --muted: #616d7d;
      --line: #d7c6ab;
      --accent: #b45b2e;
      --accent-strong: #8f3d1d;
      --shadow: 0 18px 42px rgba(69, 47, 24, 0.12);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      font-family: "Avenir Next", "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at top left, #fff8ef 0, #fff8ef 12%, transparent 36%),
        linear-gradient(180deg, #f8f1e5 0%, #eee1cd 100%);
    }}
    .shell {{
      width: min(1040px, calc(100vw - 2rem));
      margin: 1rem auto;
      padding-bottom: 2rem;
    }}
    .hero {{
      padding: 1.2rem 1.35rem;
      border: 1px solid rgba(113, 80, 45, 0.18);
      border-radius: 8px;
      background: linear-gradient(135deg, rgba(255,255,255,0.9), rgba(255,247,234,0.82));
      box-shadow: var(--shadow);
    }}
    h1 {{
      margin: 0;
      font-family: "Iowan Old Style", "Palatino Linotype", serif;
      font-size: clamp(2rem, 3vw, 3rem);
      line-height: 1;
    }}
    .lede {{
      margin: 0.7rem 0 0;
      max-width: 68ch;
      color: var(--muted);
      line-height: 1.5;
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
    .grid {{
      display: grid;
      grid-template-columns: minmax(0, 1.08fr) minmax(310px, 0.92fr);
      gap: 1rem;
      margin-top: 1rem;
    }}
    .panel {{
      border: 1px solid rgba(113, 80, 45, 0.18);
      border-radius: 8px;
      background: var(--paper);
      box-shadow: var(--shadow);
    }}
    .composer, .metrics, .output {{
      padding: 1rem;
    }}
    .label {{
      display: block;
      margin-bottom: 0.45rem;
      color: var(--muted);
      font-size: 0.92rem;
    }}
    textarea {{
      width: 100%;
      min-height: 10rem;
      padding: 0.9rem 1rem;
      border: 1px solid var(--line);
      border-radius: 8px;
      font: inherit;
      line-height: 1.5;
      resize: vertical;
      background: rgba(255,255,255,0.92);
    }}
    .actions {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.8rem;
      align-items: center;
      margin-top: 0.85rem;
    }}
    button {{
      font: inherit;
      border-radius: 8px;
      border: 1px solid transparent;
      cursor: pointer;
    }}
    .submit {{
      min-height: 2.7rem;
      padding: 0.65rem 1rem;
      background: linear-gradient(180deg, var(--accent), var(--accent-strong));
      color: white;
    }}
    .submit:disabled {{
      opacity: 0.72;
      cursor: progress;
    }}
    .status {{
      color: var(--muted);
      font-size: 0.92rem;
    }}
    .examples {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.55rem;
      margin-top: 1rem;
    }}
    .example {{
      padding: 0.55rem 0.8rem;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.75);
      text-align: left;
    }}
    .side {{
      display: grid;
      gap: 1rem;
    }}
    .metrics-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 0.7rem;
      margin-top: 0.4rem;
    }}
    .metric {{
      padding: 0.8rem;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: rgba(255,255,255,0.75);
    }}
    .metric strong {{
      display: block;
      font-size: 1.14rem;
      line-height: 1.15;
    }}
    .metric span {{
      display: block;
      margin-top: 0.2rem;
      color: var(--muted);
      font-size: 0.86rem;
    }}
    .hint {{
      color: var(--muted);
      line-height: 1.5;
    }}
    pre {{
      margin: 0;
      white-space: pre-wrap;
      line-height: 1.6;
      font-family: "SFMono-Regular", Consolas, monospace;
    }}
    @media (max-width: 860px) {{
      .grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main class="shell">
    <section class="hero">
      <h1>GPT-2 on ExecuTorch</h1>
      <p class="lede">A Pi-hosted text generation demo using the lowered XNNPACK model.</p>
      <div class="meta">
        <div class="pill">Model: {html.escape(str(startup["model"]))}</div>
        <div class="pill">Context: {startup["seq_len"]} tokens</div>
        <div class="pill">Max new tokens: {startup["max_new_tokens"]}</div>
        <div class="pill">Server load: {format_ms(float(startup["load_ms"]))}</div>
      </div>
    </section>
    <section class="grid">
      <section class="panel composer">
        <form method="post" id="chat-form">
          <label class="label" for="prompt">Prompt</label>
          <textarea id="prompt" name="prompt">{html.escape(prompt)}</textarea>
          <div class="actions">
            <button class="submit" type="submit" id="submit-btn">Generate text</button>
            <div class="status" id="status">Generation runs autoregressively, one token step at a time.</div>
          </div>
        </form>
        <div class="examples">{example_buttons}</div>
      </section>
      <section class="side">
        <section class="panel metrics">
          <div class="label">Timing</div>
          {summary}
        </section>
        <section class="panel output">
          <div class="label">Response</div>
          <pre>{html.escape(output)}</pre>
        </section>
      </section>
    </section>
  </main>
  <script>
    const form = document.getElementById("chat-form");
    const prompt = document.getElementById("prompt");
    const submit = document.getElementById("submit-btn");
    const status = document.getElementById("status");
    for (const button of document.querySelectorAll(".example")) {{
      button.addEventListener("click", () => {{
        prompt.value = button.dataset.prompt || "";
        prompt.focus();
      }});
    }}
    form.addEventListener("submit", () => {{
      submit.disabled = true;
      submit.textContent = "Generating...";
      status.textContent = "Running repeated forward passes on the Pi.";
    }});
  </script>
</body>
</html>"""


def metrics_panel(metrics: dict[str, float | int]) -> str:
    values = [
        ("End-to-end", format_ms(float(metrics["total_ms"]))),
        ("First token", format_ms(float(metrics["first_token_ms"]))),
        ("Avg step", format_ms(float(metrics["avg_step_ms"]))),
        ("Token rate", f"{float(metrics['tokens_per_second']):.2f} tok/s"),
        ("Generated", f"{int(metrics['generated_tokens'])} tokens"),
        ("Forward total", format_ms(float(metrics["forward_ms_total"]))),
    ]
    cards = "".join(
        f"<div class='metric'><strong>{html.escape(value)}</strong><span>{html.escape(label)}</span></div>"
        for label, value in values
    )
    return f"<div class='metrics-grid'>{cards}</div>"
