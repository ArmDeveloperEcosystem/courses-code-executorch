from __future__ import annotations

from contextlib import contextmanager
import logging
import os
from pathlib import Path
import sys
from typing import Any, Iterator
import warnings


def configure_quiet_output(
    *,
    include_transformers: bool = False,
    include_interpreter_bin: bool = False,
) -> None:
    """Hide dependency warnings that obscure the useful lab output."""
    if include_interpreter_bin:
        ensure_interpreter_bin_on_path()

    if include_transformers:
        os.environ.setdefault("GLOG_minloglevel", "2")
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message="The given buffer is not writable.*", category=UserWarning)

    loggers = [
        "torch",
        "torch.distributed",
        "torch.distributed.elastic",
        "torch.utils._pytree",
        "executorch",
    ]
    if include_transformers:
        loggers.append("transformers")

    logging.getLogger().setLevel(logging.ERROR)
    for name in loggers:
        logging.getLogger(name).setLevel(logging.ERROR)


def ensure_interpreter_bin_on_path() -> None:
    """Make venv-provided tools such as flatc visible when using venv/bin/python directly."""
    bin_path = Path(sys.executable).parent
    flatc_path = bin_path / "flatc"
    if flatc_path.exists():
        os.environ.setdefault("FLATC_EXECUTABLE", str(flatc_path))
    bin_dir = str(bin_path)
    path_parts = [part for part in os.environ.get("PATH", "").split(os.pathsep) if part and part != bin_dir]
    os.environ["PATH"] = os.pathsep.join([bin_dir, *path_parts])


def quiet_transformers_progress() -> None:
    try:
        from transformers.utils import logging as transformers_logging

        transformers_logging.set_verbosity_error()
        transformers_logging.disable_progress_bar()
    except Exception:
        pass


@contextmanager
def suppress_stderr() -> Iterator[None]:
    """Temporarily hide noisy native dependency logs written directly to stderr."""
    saved_stderr = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved_stderr, 2)
        os.close(saved_stderr)
        os.close(devnull)


def load_hf_model(model_cls: Any, model_id: str) -> Any:
    import torch

    try:
        return model_cls.from_pretrained(model_id, dtype=torch.float32)
    except TypeError:
        return model_cls.from_pretrained(model_id, torch_dtype=torch.float32)


class LastTokenLogits:
    """Wrap OPT so export/runtime returns only the logits needed for next-token generation."""

    def __new__(cls, model: Any) -> Any:
        import torch

        class _LastTokenLogits(torch.nn.Module):
            def __init__(self, wrapped: Any):
                super().__init__()
                self.wrapped = wrapped

            def forward(self, input_ids: Any, attention_mask: Any) -> Any:
                out = self.wrapped(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
                return out.logits[:, -1, :]

        return _LastTokenLogits(model)


class MaskLogits:
    """Wrap DistilBERT so export/runtime returns only logits at the single [MASK] token."""

    def __new__(cls, model: Any, mask_token_id: int) -> Any:
        import torch

        class _MaskLogits(torch.nn.Module):
            def __init__(self, wrapped: Any, mask_id: int):
                super().__init__()
                self.wrapped = wrapped
                self.mask_id = mask_id

            def forward(self, input_ids: Any, attention_mask: Any) -> Any:
                logits = self.wrapped(input_ids=input_ids, attention_mask=attention_mask).logits
                mask_pos = (input_ids == self.mask_id).to(torch.int64).argmax(dim=1)
                batch = torch.arange(input_ids.shape[0], device=input_ids.device)
                return logits[batch, mask_pos, :]

        return _MaskLogits(model, mask_token_id)


def example_inputs(spec: dict[str, Any]) -> tuple[Any, ...]:
    from transformers import AutoTokenizer

    if spec["adapter"] == "opt125m":
        tokenizer = AutoTokenizer.from_pretrained(spec["model_id"])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        prompt = spec.get("example_prompt") or "Write a short sentence about fast inference."
    elif spec["adapter"] == "distilbert_fillmask":
        tokenizer = AutoTokenizer.from_pretrained(spec["model_id"], use_fast=True)
        prompt = spec.get("example_prompt") or f"The Raspberry Pi is a {tokenizer.mask_token} computer."
    else:
        raise ValueError(f"Unknown adapter: {spec['adapter']}")

    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=int(spec["seq_len"]),
    )
    return encoded["input_ids"], encoded["attention_mask"]


def format_ms(value: float) -> str:
    if value >= 1000.0:
        return f"{value / 1000.0:.2f} s"
    return f"{value:.0f} ms"
