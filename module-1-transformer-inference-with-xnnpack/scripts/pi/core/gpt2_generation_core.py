# ExecuTorch next-token generation used by the Module 1 GPT-2 web demo.

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time
from typing import Any


MODULE1_SCRIPTS = Path(__file__).resolve().parents[2]
if str(MODULE1_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(MODULE1_SCRIPTS))

from _support.common import configure_quiet_output, load_executorch_program, suppress_stderr


@dataclass(frozen=True)
class Gpt2GenerationMetrics:
    prompt_tokens: int
    generated_tokens: int
    first_token_ms: float
    forward_ms_total: float
    avg_step_ms: float
    total_ms: float
    tokens_per_second: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "first_token_ms": self.first_token_ms,
            "forward_ms_total": self.forward_ms_total,
            "avg_step_ms": self.avg_step_ms,
            "total_ms": self.total_ms,
            "tokens_per_second": self.tokens_per_second,
        }


@dataclass(frozen=True)
class Gpt2GenerationResult:
    output: str
    metrics: Gpt2GenerationMetrics


@dataclass
class Gpt2GenerationRunner:
    # Run a fixed-context GPT-2 .pte repeatedly to generate text.
    spec: dict[str, Any]
    tokenizer: Any
    model: Any
    startup: dict[str, str | float | int]

    def generate(self, prompt: str, *, max_new_tokens: int, temperature: float = 0.5) -> Gpt2GenerationResult:
        # Generate tokens and record forward-pass and end-to-end timing.
        import torch

        started = time.perf_counter()
        input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
        attention_mask = torch.ones_like(input_ids)
        ctx = int(self.spec["seq_len"])
        prompt_tokens = int(input_ids.shape[1])
        generated_tokens = 0
        first_token_ms = 0.0
        forward_ms_total = 0.0

        # ExecuTorch is running next-token generation without a KV cache here,
        # so each new token requires another forward pass over the current context.
        for _ in range(max_new_tokens):
            in_ids = input_ids[:, -ctx:]
            in_mask = attention_mask[:, -ctx:]
            if in_ids.shape[1] < ctx:
                pad = ctx - in_ids.shape[1]
                pad_ids = torch.full((1, pad), int(self.tokenizer.pad_token_id), dtype=in_ids.dtype)
                in_ids = torch.cat([in_ids, pad_ids], dim=1)
                in_mask = torch.cat([in_mask, torch.zeros((1, pad), dtype=in_mask.dtype)], dim=1)

            step_started = time.perf_counter()
            # ExecuTorch Python runtime: invoke the exported forward method.
            with suppress_stderr():
                logits = self.model.forward((in_ids, in_mask))[0]
            forward_ms_total += (time.perf_counter() - step_started) * 1000.0

            next_token = torch.multinomial(torch.softmax(logits / temperature, dim=-1), num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=1)
            generated_tokens += 1

            if first_token_ms == 0.0:
                first_token_ms = (time.perf_counter() - started) * 1000.0
            if int(next_token.item()) == int(self.tokenizer.eos_token_id):
                break

        total_ms = (time.perf_counter() - started) * 1000.0
        return Gpt2GenerationResult(
            output=self.tokenizer.decode(input_ids[0], skip_special_tokens=True),
            metrics=Gpt2GenerationMetrics(
                prompt_tokens=prompt_tokens,
                generated_tokens=generated_tokens,
                first_token_ms=first_token_ms,
                forward_ms_total=forward_ms_total,
                avg_step_ms=forward_ms_total / max(generated_tokens, 1),
                total_ms=total_ms,
                tokens_per_second=(generated_tokens * 1000.0 / total_ms) if total_ms > 0 else 0.0,
            ),
        )


def load_gpt2_generation_runner(spec_path: Path, pte_path: Path) -> Gpt2GenerationRunner:
    # Load the tokenizer, model settings, and .pte program once at startup.
    configure_quiet_output(include_transformers=True)

    from transformers import AutoTokenizer

    load_started = time.perf_counter()
    spec = json.loads(spec_path.read_text())
    tokenizer = AutoTokenizer.from_pretrained(spec["model_id"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = load_executorch_program(pte_path)

    startup = {
        "model": spec["model_id"],
        "backend": pte_path.stem,
        "seq_len": int(spec["seq_len"]),
        "load_ms": (time.perf_counter() - load_started) * 1000.0,
    }
    return Gpt2GenerationRunner(spec=spec, tokenizer=tokenizer, model=model, startup=startup)
