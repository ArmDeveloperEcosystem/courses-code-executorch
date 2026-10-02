# Shared model adapters, example inputs, and runtime helpers for Module 1.

from __future__ import annotations

from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
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
    # Filter known dependency warnings and progress output; exceptions remain visible.
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
    # Make venv-provided tools such as flatc visible when using venv/bin/python directly.
    bin_path = Path(sys.executable).parent
    flatc_path = bin_path / "flatc"
    if flatc_path.exists():
        os.environ.setdefault("FLATC_EXECUTABLE", str(flatc_path))
    bin_dir = str(bin_path)
    path_parts = [part for part in os.environ.get("PATH", "").split(os.pathsep) if part and part != bin_dir]
    os.environ["PATH"] = os.pathsep.join([bin_dir, *path_parts])


def quiet_transformers_progress() -> None:
    # Disable Hugging Face progress output when the installed version supports it.
    try:
        from transformers.utils import logging as transformers_logging

        transformers_logging.set_verbosity_error()
        transformers_logging.disable_progress_bar()
    except Exception:
        pass


@contextmanager
def suppress_stderr() -> Iterator[None]:
    # Use only around repeated runtime calls that print the same native message.
    # Set COURSE_DEBUG=1 to keep native diagnostics visible.
    if os.environ.get("COURSE_DEBUG") == "1":
        yield
        return
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
    # Load a Hugging Face model in FP32 across old and new Transformers APIs.
    import torch

    try:
        return model_cls.from_pretrained(model_id, dtype=torch.float32)
    except TypeError:
        return model_cls.from_pretrained(model_id, torch_dtype=torch.float32)


def load_executorch_program(pte_path: Path | str) -> Any:
    # Course wrapper around ExecuTorch's _load_for_executorch() runtime binding.
    try:
        from executorch.extension.pybindings.portable_lib import _load_for_executorch
    except ImportError as exc:
        raise RuntimeError(
            "Could not import ExecuTorch portable pybindings. This usually means "
            "the Python, torch, and executorch wheels in this virtual environment "
            "are not ABI-compatible. Module 1 is pinned for Python 3.12. If this "
            "venv was created with another Python version, remove it, select "
            "Python with `pyenv shell 3.12.14`, recreate it with `python -m venv "
            "~/executorch_on_arm/.cpu_lab_venv`, activate it, then reinstall "
            "`module-1-transformer-inference-with-xnnpack/requirements.txt`.\n\n"
            f"Python: {sys.version.split()[0]}\n"
            f"torch: {installed_version('torch')}\n"
            f"executorch: {installed_version('executorch')}\n"
            f"Original import error: {exc}"
        ) from exc

    return _load_for_executorch(str(pte_path))


def installed_version(package_name: str) -> str:
    # Return a package version for diagnostic error messages.
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "not installed"


class LastTokenLogits:
    # Return only the logits needed for the next causal-LM generation step.

    def __new__(cls, model: Any) -> Any:
        import torch

        class _LastTokenLogits(torch.nn.Module):
            def __init__(self, model: Any):
                super().__init__()
                self.model = model

            def forward(self, input_ids: Any, attention_mask: Any) -> Any:
                output = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
                token_positions = torch.arange(output.logits.shape[1], device=output.logits.device)
                # The mask is 1 for real tokens, so the row maximum is the last
                # non-padding position for each item in the batch.
                last_token_positions = (attention_mask.to(torch.int64) * token_positions).amax(dim=1)
                batch_indices = torch.arange(input_ids.shape[0], device=input_ids.device)
                return output.logits[batch_indices, last_token_positions, :]

        return _LastTokenLogits(model)


class MaskLogits:
    # Wrap DistilBERT so export/runtime returns only logits at the single [MASK] token.

    def __new__(cls, model: Any, mask_token_id: int) -> Any:
        import torch

        class _MaskLogits(torch.nn.Module):
            def __init__(self, model: Any, mask_token_id: int):
                super().__init__()
                self.model = model
                self.mask_token_id = mask_token_id

            def forward(self, input_ids: Any, attention_mask: Any) -> Any:
                # DistilBERT returns [batch, sequence, vocabulary] logits.
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
                mask_positions = (input_ids == self.mask_token_id).to(torch.int64).argmax(dim=1)
                batch_indices = torch.arange(input_ids.shape[0], device=input_ids.device)
                # Pair each batch item with its own mask position to return
                # [batch, vocabulary], not every mask position for every item.
                return logits[batch_indices, mask_positions, :]

        return _MaskLogits(model, mask_token_id)


def example_inputs(spec: dict[str, Any]) -> tuple[Any, ...]:
    # Tokenize the spec prompt into the fixed-shape inputs used by export and benchmarks.
    from transformers import AutoTokenizer

    if spec["adapter"] == "causal_lm":
        tokenizer = AutoTokenizer.from_pretrained(spec["model_id"])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
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
    # Format millisecond measurements for learner-facing output.
    if value >= 1000.0:
        return f"{value / 1000.0:.2f} s"
    return f"{value:.0f} ms"
