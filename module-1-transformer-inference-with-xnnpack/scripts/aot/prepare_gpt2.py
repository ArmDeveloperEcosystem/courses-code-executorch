#!/usr/bin/env python3
# Export fixed-shape GPT-2 next-token logits and write its model contract.

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


MODULE1_SCRIPTS = Path(__file__).resolve().parents[1]
if str(MODULE1_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(MODULE1_SCRIPTS))

from _support.common import LastTokenLogits, configure_quiet_output, load_hf_model


def main() -> None:
    # Script setup: command-line options and output paths.
    parser = argparse.ArgumentParser(description="Prepare GPT-2 by exporting it and writing a model spec.")
    parser.add_argument("--model-id", default="openai-community/gpt2")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--out", type=Path, default=Path("module-1-transformer-inference-with-xnnpack/artifacts/gpt2"))
    parser.add_argument(
        "--example-prompt",
        default="Write a short sentence about fast inference.",
        help="Used only to create fixed-shape example tensors for export.",
    )
    parser.add_argument("--exported-name", default="gpt2_exported.pt2")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    configure_quiet_output()

    import torch
    from torch.export import export, save
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args.out.mkdir(parents=True, exist_ok=True)
    exported_path = args.out / args.exported_name

    # 1. Hugging Face and PyTorch: load the tokenizer and model.
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = load_hf_model(AutoModelForCausalLM, args.model_id).to(args.device).eval()
    # Export only the logits needed for the next generation step instead of
    # the full [batch, sequence, vocabulary] tensor.
    wrapped = LastTokenLogits(model).eval()
    encoded = tokenizer(
        args.example_prompt,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=args.seq_len,
    )
    example_inputs = (
        encoded["input_ids"].to(args.device),
        encoded["attention_mask"].to(args.device),
    )

    # 2. PyTorch export API: capture the fixed-shape forward graph.
    print(f"Exporting {args.model_id} to {exported_path}...")
    with torch.inference_mode():
        exported_program = export(wrapped, example_inputs)
    save(exported_program, exported_path)

    model_source = {"repository": f"https://huggingface.co/{args.model_id}"}
    if args.model_id in {"gpt2", "openai-community/gpt2"}:
        model_source.update(
            {
                "license": "Modified MIT License",
                "license_url": "https://github.com/openai/gpt-2/blob/master/LICENSE",
            }
        )

    # Supporting output: record the tensor shapes required at runtime.
    spec = {
        "adapter": "causal_lm",
        "name": "gpt2",
        "model_id": args.model_id,
        "model_source": model_source,
        "seq_len": args.seq_len,
        "example_prompt": args.example_prompt,
        "exported_program": args.exported_name,
        "inputs": [
            {"name": "input_ids", "shape": [1, args.seq_len], "dtype": "int64"},
            {"name": "attention_mask", "shape": [1, args.seq_len], "dtype": "int64"},
        ],
        "task": "causal_lm_next_token",
    }

    spec_path = args.out / "model_spec.json"
    spec_path.write_text(json.dumps(spec, indent=2) + "\n")
    print(f"Wrote {exported_path}")
    print(f"Wrote {spec_path}")
    print("Runtime prompts can change; keep batch size 1 and pad/truncate to this seq-len.")


if __name__ == "__main__":
    main()
