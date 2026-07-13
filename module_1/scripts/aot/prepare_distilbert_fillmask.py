#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


MODULE1_SCRIPTS = Path(__file__).resolve().parents[1]
if str(MODULE1_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(MODULE1_SCRIPTS))

from _support.common import MaskLogits, configure_quiet_output, load_hf_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare DistilBERT fill-mask by exporting it and writing a model spec.")
    parser.add_argument("--model-id", default="distilbert-base-uncased")
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--out", type=Path, default=Path("module_1/artifacts/distilbert"))
    parser.add_argument(
        "--example-prompt",
        default=None,
        help="Optional export-only prompt; must contain the tokenizer mask token.",
    )
    parser.add_argument("--exported-name", default="distilbert_fillmask_exported.pt2")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    configure_quiet_output()

    import torch
    from torch.export import export, save
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    args.out.mkdir(parents=True, exist_ok=True)
    exported_path = args.out / args.exported_name

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    prompt = args.example_prompt or f"The Raspberry Pi is a {tokenizer.mask_token} computer."
    if tokenizer.mask_token not in prompt:
        raise ValueError(f"Example prompt must contain {tokenizer.mask_token}")

    model = load_hf_model(AutoModelForMaskedLM, args.model_id).to(args.device).eval()
    wrapped = MaskLogits(model, tokenizer.mask_token_id).eval()
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=args.seq_len,
    )
    example_inputs = (
        encoded["input_ids"].to(args.device),
        encoded["attention_mask"].to(args.device),
    )

    print(f"Exporting {args.model_id} to {exported_path}...")
    with torch.inference_mode():
        exported_program = export(wrapped, example_inputs)
    save(exported_program, exported_path)

    spec = {
        "adapter": "distilbert_fillmask",
        "name": "distilbert_fillmask",
        "model_id": args.model_id,
        "seq_len": args.seq_len,
        "example_prompt": prompt,
        "exported_program": args.exported_name,
        "inputs": [
            {"name": "input_ids", "shape": [1, args.seq_len], "dtype": "int64"},
            {"name": "attention_mask", "shape": [1, args.seq_len], "dtype": "int64"},
        ],
        "task": "fill_mask",
    }

    spec_path = args.out / "model_spec.json"
    spec_path.write_text(json.dumps(spec, indent=2) + "\n")
    print(f"Wrote {exported_path}")
    print(f"Wrote {spec_path}")
    print("Runtime prompts can change; keep one mask token, batch size 1, and this seq-len.")


if __name__ == "__main__":
    main()
