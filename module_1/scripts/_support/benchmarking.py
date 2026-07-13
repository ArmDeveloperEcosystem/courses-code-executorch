from __future__ import annotations

import csv
import json
import platform
import time
from pathlib import Path
from typing import Callable

from _support.common import suppress_stderr


def benchmark(fn: Callable[[], object], *, warmups: int, runs: int) -> tuple[list[float], dict[str, float | int]]:
    import numpy as np

    for _ in range(warmups):
        with suppress_stderr():
            fn()

    times = np.empty(runs, dtype=np.float64)
    for i in range(runs):
        start = time.perf_counter()
        with suppress_stderr():
            fn()
        times[i] = (time.perf_counter() - start) * 1000.0

    return times.tolist(), {
        "count": int(runs),
        "mean_ms": float(times.mean()),
        "median_ms": float(np.median(times)),
        "std_ms": float(times.std(ddof=0)),
        "min_ms": float(times.min()),
        "max_ms": float(times.max()),
        "p50_ms": float(np.percentile(times, 50)),
        "p90_ms": float(np.percentile(times, 90)),
        "p95_ms": float(np.percentile(times, 95)),
        "p99_ms": float(np.percentile(times, 99)),
    }


def save_plots(times_ms: list[float], *, title: str, out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import numpy as np

    out_dir.mkdir(parents=True, exist_ok=True)
    values = np.asarray(times_ms, dtype=np.float64)
    mean_ms = float(values.mean())
    slug = "".join(c.lower() if c.isalnum() else "_" for c in title).strip("_")

    hist_path = out_dir / f"{slug}_histogram.png"
    plt.figure(figsize=(7, 4))
    plt.hist(values, bins=min(30, max(5, len(values) // 2)), edgecolor="black", alpha=0.75)
    plt.axvline(mean_ms, color="red", linestyle="--", label=f"Mean: {mean_ms:.2f} ms")
    plt.xlabel("Inference time (ms)")
    plt.ylabel("Count")
    plt.title(f"{title}: latency distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(hist_path, dpi=150)
    plt.close()

    series_path = out_dir / f"{slug}_timeseries.png"
    plt.figure(figsize=(7, 4))
    plt.plot(values, marker="o", markersize=3, linewidth=1)
    plt.axhline(mean_ms, color="red", linestyle="--", label=f"Mean: {mean_ms:.2f} ms")
    plt.xlabel("Run")
    plt.ylabel("Inference time (ms)")
    plt.title(f"{title}: latency over runs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(series_path, dpi=150)
    plt.close()

    return [hist_path, series_path]


def write_results(
    *,
    out: Path,
    runtime: str,
    model: str,
    warmups: int,
    times_ms: list[float],
    stats: dict[str, float | int],
    extra: dict[str, object],
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "runtime": runtime,
        "model": model,
        "warmups": warmups,
        "stats": stats,
        "machine": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "extra": extra,
    }
    out.write_text(json.dumps(payload, indent=2) + "\n")

    with out.with_suffix(".csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "latency_ms"])
        for i, value in enumerate(times_ms, start=1):
            writer.writerow([i, f"{value:.6f}"])


def print_stats(name: str, stats: dict[str, float | int], *, warmups: int) -> None:
    print(f"\n{name}")
    print("=" * len(name))
    print(f"Warmups: {warmups}")
    print(f"Runs:    {stats['count']}")
    print(f"Mean:    {stats['mean_ms']:.2f} ms")
    print(f"Median:  {stats['median_ms']:.2f} ms")
    print(f"Min:     {stats['min_ms']:.2f} ms")
    print(f"Max:     {stats['max_ms']:.2f} ms")
