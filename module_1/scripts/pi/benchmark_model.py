#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


MODULE1_SCRIPTS = Path(__file__).resolve().parents[1]
if str(MODULE1_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(MODULE1_SCRIPTS))

from _support.benchmarking import benchmark, print_stats, save_plots, write_results
from _support.common import (
    LastTokenLogits,
    MaskLogits,
    configure_quiet_output,
    example_inputs,
    load_executorch_program,
    load_hf_model,
    quiet_transformers_progress,
    suppress_stderr,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark PyTorch or ExecuTorch on the Pi.")
    parser.add_argument("--runtime", choices=["pytorch", "executorch"], required=True)
    parser.add_argument("--backend", choices=["portable", "xnnpack"], default=None)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--pte", type=Path, default=None)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--plots",
        type=Path,
        default=None,
        help="Plot output directory. Defaults to a plots folder next to --out.",
    )
    args = parser.parse_args()

    configure_quiet_output(include_transformers=True)

    with suppress_stderr():
        import torch
        quiet_transformers_progress()

    spec = json.loads(args.spec.read_text())
    model_name = spec["name"]
    if args.runtime == "pytorch":
        if args.backend is not None:
            raise ValueError("--backend is only valid for --runtime executorch")
        backend = "eager"
    else:
        backend = args.backend or infer_backend(args.runtime, args.pte, args.out)
    runtime_label = format_runtime_label(args.runtime, backend)
    result_name = format_result_name(model_name, args.runtime, backend)

    if args.runtime == "pytorch":
        with suppress_stderr():
            module, inputs = prepare_model(spec)

        def forward() -> object:
            with torch.inference_mode():
                return module(*inputs)

    else:
        if args.pte is None:
            raise ValueError("--pte is required for --runtime executorch")
        with suppress_stderr():
            model = load_executorch_program(args.pte)
            inputs = example_inputs(spec)

        def forward() -> object:
            with torch.inference_mode():
                return model.forward(inputs)

    times_ms, stats = benchmark(forward, warmups=args.warmups, runs=args.runs)
    plot_dir = args.plots or args.out.parent / "plots"
    plot_paths = []
    if should_plot_result(backend):
        plot_paths = save_plots(times_ms, title=result_name, out_dir=plot_dir)
    write_results(
        out=args.out,
        runtime=args.runtime,
        model=model_name,
        warmups=args.warmups,
        times_ms=times_ms,
        stats=stats,
        extra={
            "backend": backend,
            "runtime_label": runtime_label,
            "result_name": result_name,
            "plots": [str(p) for p in plot_paths],
            "pte": str(args.pte) if args.pte else None,
        },
    )
    print_stats(result_name, stats, warmups=args.warmups)
    print(f"Wrote {args.out}")


def prepare_model(spec: dict[str, Any]) -> tuple[Any, tuple[Any, ...]]:
    if spec["adapter"] == "opt125m":
        return prepare_opt(spec)
    if spec["adapter"] == "distilbert_fillmask":
        return prepare_distilbert(spec)
    raise ValueError(f"Unknown adapter: {spec['adapter']}")


def prepare_opt(spec: dict[str, Any]) -> tuple[Any, tuple[Any, ...]]:
    from transformers import AutoModelForCausalLM

    model = load_hf_model(AutoModelForCausalLM, spec["model_id"]).eval()
    return LastTokenLogits(model).eval(), example_inputs(spec)


def prepare_distilbert(spec: dict[str, Any]) -> tuple[Any, tuple[Any, ...]]:
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(spec["model_id"], use_fast=True)
    model = load_hf_model(AutoModelForMaskedLM, spec["model_id"]).eval()
    return MaskLogits(model, tokenizer.mask_token_id).eval(), example_inputs(spec)


def infer_backend(runtime: str, pte: Path | None, out: Path) -> str:
    if runtime == "pytorch":
        return "eager"

    for path in (pte, out):
        if path is None:
            continue
        name = path.stem.lower()
        if name.endswith("_portable") or "_portable" in name:
            return "portable"
        if name.endswith("_xnnpack") or "_xnnpack" in name:
            return "xnnpack"

    return "executorch"


def format_runtime_label(runtime: str, backend: str) -> str:
    if runtime == "pytorch":
        return "PyTorch eager"
    if backend == "portable":
        return "ExecuTorch portable"
    if backend == "xnnpack":
        return "ExecuTorch with XNNPACK"
    return "ExecuTorch"


def format_result_name(model_name: str, runtime: str, backend: str) -> str:
    if runtime == "pytorch":
        return f"{model_name}_pytorch"
    return f"{model_name}_{backend}"


def should_plot_result(backend: str) -> bool:
    return backend != "portable"


if __name__ == "__main__":
    main()
