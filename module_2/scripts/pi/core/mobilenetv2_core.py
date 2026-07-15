from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from io import BytesIO
import json
import logging
import os
from pathlib import Path
import time
from typing import Any
import warnings


@dataclass
class ClassificationResult:
    predictions: list[dict[str, float | str]]
    latency_ms: float


class MobileNetV2Runner:
    def __init__(self, spec_path: Path, *, pte_path: Path | None = None, runtime: str = "executorch"):
        configure_quiet_output()
        self.spec_path = spec_path
        self.spec = json.loads(spec_path.read_text())
        self.labels = json.loads((spec_path.parent / self.spec["labels"]).read_text())
        self.runtime = runtime
        self.pte_path = pte_path
        self.model = self._load_model()

    def classify(self, image_bytes: bytes, *, topk: int = 5) -> ClassificationResult:
        with suppress_stderr():
            import torch

        input_tensor = self._preprocess(image_bytes)
        start = time.perf_counter()
        with torch.inference_mode(), suppress_stderr():
            if self.runtime == "executorch":
                output = self.model.forward((input_tensor,))[0]
            else:
                output = self.model(input_tensor)
        latency_ms = (time.perf_counter() - start) * 1000.0

        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        scores, indices = torch.topk(probabilities, topk)
        predictions = [
            {"label": self.labels[int(index)], "score": float(score)}
            for score, index in zip(scores, indices)
        ]
        return ClassificationResult(predictions=predictions, latency_ms=latency_ms)

    def _load_model(self) -> Any:
        if self.runtime == "executorch":
            if self.pte_path is None:
                raise ValueError("--pte is required for ExecuTorch runtime")
            with suppress_stderr():
                from executorch.extension.pybindings.portable_lib import _load_for_executorch

                return _load_for_executorch(str(self.pte_path))

        with suppress_stderr():
            import torch
            from torchvision import models

            model = models.mobilenet_v2(weights=None)
            model.load_state_dict(torch.load(self.spec_path.parent / self.spec["weights"], map_location="cpu"))
        return model.eval()

    def _preprocess(self, image_bytes: bytes) -> Any:
        with suppress_stderr():
            from PIL import Image
            import torchvision.transforms as transforms

        prep = self.spec["input"]["preprocessing"]
        transform = transforms.Compose(
            [
                transforms.Resize(prep["resize"]),
                transforms.CenterCrop(prep["center_crop"]),
                transforms.ToTensor(),
                transforms.Normalize(mean=prep["mean"], std=prep["std"]),
            ]
        )
        return transform(Image.open(BytesIO(image_bytes)).convert("RGB")).unsqueeze(0)


def configure_quiet_output() -> None:
    """Hide dependency warnings so demos show model behavior, not library noise."""
    os.environ.setdefault("GLOG_minloglevel", "2")
    os.environ.setdefault("KMP_WARNINGS", "0")
    os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")
    os.environ.setdefault("PYTHONWARNINGS", "ignore")

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning, message="The given buffer is not writable.*")

    logging.getLogger().setLevel(logging.ERROR)
    for name in (
        "torch",
        "torch.distributed",
        "torch.distributed.elastic",
        "torch.utils._pytree",
        "torchvision",
        "executorch",
    ):
        logging.getLogger(name).setLevel(logging.ERROR)


@contextmanager
def suppress_stderr() -> Any:
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
