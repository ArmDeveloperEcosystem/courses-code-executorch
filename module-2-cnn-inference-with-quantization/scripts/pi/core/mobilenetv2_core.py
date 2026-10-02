# MobileNetV2 preprocessing, runtime invocation, and classification postprocessing.

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import json
from pathlib import Path
import time
from typing import Any

from _support.common import configure_quiet_output, suppress_stderr


@dataclass
class ClassificationResult:
    predictions: list[dict[str, float | str]]
    latency_ms: float


class MobileNetV2Runner:
    # Run the same image-classification contract with ExecuTorch or PyTorch.
    def __init__(self, spec_path: Path, *, pte_path: Path | None = None, runtime: str = "executorch"):
        configure_quiet_output()
        self.spec_path = spec_path
        self.spec = json.loads(spec_path.read_text())
        self.labels = json.loads((spec_path.parent / self.spec["labels"]).read_text())
        self.runtime = runtime
        self.pte_path = pte_path
        self.model = self._load_model()

    def classify(self, image_bytes: bytes, *, topk: int = 5) -> ClassificationResult:
        # Keep preprocessing and postprocessing outside the measured forward call.
        import torch

        input_tensor = self._preprocess(image_bytes)
        start = time.perf_counter()
        with torch.inference_mode(), suppress_stderr():
            if self.runtime == "executorch":
                # ExecuTorch Python runtime: invoke the exported forward method.
                output = self.model.forward((input_tensor,))[0]
            else:
                # PyTorch eager baseline: call the nn.Module directly.
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
        # Load either the .pte runtime module or the eager reference model.
        if self.runtime == "executorch":
            if self.pte_path is None:
                raise ValueError("--pte is required for ExecuTorch runtime")
            from executorch.extension.pybindings.portable_lib import _load_for_executorch

            # ExecuTorch Python runtime binding: load the packaged .pte.
            return _load_for_executorch(str(self.pte_path))

        import torch
        from torchvision import models

        model = models.mobilenet_v2(weights=None)
        model.load_state_dict(torch.load(self.spec_path.parent / self.spec["weights"], map_location="cpu"))
        return model.eval()

    def _preprocess(self, image_bytes: bytes) -> Any:
        # Turn encoded image bytes into the NCHW tensor described by the spec.
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
