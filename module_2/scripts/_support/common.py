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
    """Hide dependency warnings/progress bars so lab output stays readable."""
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
    try:
        from executorch.extension.pybindings.portable_lib import _load_for_executorch
    except ImportError as exc:
        raise RuntimeError(
            "Could not import ExecuTorch portable pybindings. This usually means "
            "the Python, torch, and executorch wheels in this virtual environment "
            "are not ABI-compatible. Module 2 is pinned for Python 3.11. If this "
            "venv was created with another Python version, remove it and recreate "
            "it with `PYENV_VERSION=3.11.14 python -m venv "
            "~/executorch_on_arm/.cpu_lab_venv`, then reinstall "
            "`module_1/requirements.txt` and `module_2/requirements.txt`.\n\n"
            f"Python: {sys.version.split()[0]}\n"
            f"torch: {installed_version('torch')}\n"
            f"executorch: {installed_version('executorch')}\n"
            f"Original import error: {exc}"
        ) from exc

    return _load_for_executorch(str(pte_path))


def installed_version(package_name: str) -> str:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "not installed"
