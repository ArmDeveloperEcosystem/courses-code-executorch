# Shared output control and ExecuTorch runtime loading for Module 2.

from __future__ import annotations

from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
import logging
import os
from pathlib import Path
import sys
from typing import Iterator
import warnings


def configure_quiet_output() -> None:
    # Filter known dependency warnings and progress output; exceptions remain visible.
    os.environ.setdefault("GLOG_minloglevel", "2")
    os.environ.setdefault("KMP_WARNINGS", "0")
    os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message="The given buffer is not writable.*", category=UserWarning)

    logging.getLogger().setLevel(logging.ERROR)
    for name in (
        "torch",
        "torch.distributed",
        "torch.distributed.elastic",
        "torch.utils._pytree",
        "torchvision",
        "executorch",
        "matplotlib",
    ):
        logging.getLogger(name).setLevel(logging.ERROR)


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


def load_executorch_program(pte_path: Path | str) -> object:
    # Course wrapper around ExecuTorch's _load_for_executorch() runtime binding.
    try:
        from executorch.extension.pybindings.portable_lib import _load_for_executorch
    except ImportError as exc:
        raise RuntimeError(
            "Could not import ExecuTorch portable pybindings. This usually means "
            "the Python, torch, and executorch wheels in this virtual environment "
            "are not ABI-compatible. Module 2 is pinned for Python 3.12. If this "
            "venv was created with another Python version, remove it, select "
            "Python with `pyenv shell 3.12.14`, recreate it with `python -m venv "
            "~/executorch_on_arm/.cpu_lab_venv`, activate it, then reinstall "
            "`module-1-transformer-inference-with-xnnpack/requirements.txt` and `module-2-cnn-inference-with-quantization/requirements.txt`.\n\n"
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
