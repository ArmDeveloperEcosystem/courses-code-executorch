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
    load_hf_model,
    quiet_transformers_progress,
    suppress_stderr,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark PyTorch or ExecuTorch on the Pi.")
    parser.add_argument("--runtime", choices=["pytorch", "executorch"], required=True)
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
        with suppress_stderr():
            module, inputs = prepare_model(spec)

        def forward() -> object:
            with torch.inference_mode():
                return module(*inputs)

    else:
        if args.pte is None:
            raise ValueError("--pte is required for --runtime executorch")
        with suppress_stderr():
            from executorch.extension.pybindings.portable_lib import _load_for_executorch

            model = _load_for_executorch(str(args.pte))
            inputs = example_inputs(spec)

        def forward() -> object:
            with torch.inference_mode():
                return model.forward(inputs)

    times_ms, stats = benchmark(forward, warmups=args.warmups, runs=args.runs)
    plot_dir = args.plots or args.out.parent / "plots"
    plot_paths = save_plots(times_ms, title=f"{model_name}_{args.runtime}", out_dir=plot_dir)
    write_results(
        out=args.out,
        runtime=args.runtime,
        model=model_name,
        warmups=args.warmups,
        times_ms=times_ms,
        stats=stats,
        extra={"plots": [str(p) for p in plot_paths], "pte": str(args.pte) if args.pte else None},
    )
    print_stats(f"{model_name}_{args.runtime}", stats, warmups=args.warmups)
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


if __name__ == "__main__":
    main()
