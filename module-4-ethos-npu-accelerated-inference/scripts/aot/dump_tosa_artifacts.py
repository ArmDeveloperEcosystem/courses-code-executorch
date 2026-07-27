#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import logging
import os
from pathlib import Path
import shutil
from typing import Any
import warnings


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump TOSA artifacts for MobileNetV2 variants.")
    parser.add_argument("--spec", type=Path, default=Path("module-4-ethos-npu-accelerated-inference/artifacts/mobilenetv2/model_spec.json"))
    parser.add_argument("--calibration-images", type=Path, required=True)
    parser.add_argument("--num-calib", type=int, default=16)
    parser.add_argument("--out", type=Path, default=Path("module-4-ethos-npu-accelerated-inference/artifacts/tosa"))
    args = parser.parse_args()

    configure_quiet_output()
    if args.num_calib <= 0:
        raise ValueError("--num-calib must be greater than zero")
    if not args.calibration_images.exists():
        raise FileNotFoundError(f"Calibration images not found: {args.calibration_images}")

    with suppress_stderr():
        import torch
        import torchvision.datasets as datasets
        from torchvision.models import mobilenet_v2

    spec = json.loads(args.spec.read_text())
    artifact_dir = args.spec.parent
    with suppress_stderr():
        base_model = mobilenet_v2(weights=None).eval()
        base_model.load_state_dict(torch.load(artifact_dir / spec["weights"], map_location="cpu"))

    example_inputs = (torch.randn(*spec["input"]["shape"]),)

    with torch.inference_mode(), suppress_stderr():
        fp32_ep = export_no_guards(base_model, example_inputs)
    dump_tosa(fp32_ep, "TOSA-1.0+FP", args.out / "mobilenetv2_fp32", "MobileNetV2 FP32")

    int8_ep = quantized_export(base_model, spec, args, example_inputs, datasets)
    dump_tosa(int8_ep, "TOSA-1.0+INT", args.out / "mobilenetv2_int8", "MobileNetV2 INT8")

    lrn_model = MobileNetV2WithLRN(base_model).eval()
    lrn_int8_ep = quantized_export(lrn_model, spec, args, example_inputs, datasets)
    dump_tosa(lrn_int8_ep, "TOSA-1.0+INT", args.out / "mobilenetv2_lrn_int8", "MobileNetV2 LRN INT8")


class MobileNetV2WithLRN:
    def __init__(self, base_model: Any):
        import torch
        import torch.nn.functional as F

        class Wrapper(torch.nn.Module):
            def __init__(self, inner: Any):
                super().__init__()
                self.features = inner.features
                self.classifier = inner.classifier
                self.lrn = torch.nn.LocalResponseNorm(5)

            def forward(self, image: Any) -> Any:
                x = self.features(image)
                x = self.lrn(x)
                x = F.adaptive_avg_pool2d(x, (1, 1))
                x = torch.flatten(x, 1)
                return self.classifier(x)

        self.module = Wrapper(base_model)

    def eval(self) -> Any:
        return self.module.eval()


def quantized_export(model: Any, spec: dict[str, Any], args: argparse.Namespace, example_inputs: tuple[Any, ...], datasets: Any) -> Any:
    import torch
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    quantizer = make_quantizer()
    with torch.inference_mode(), suppress_stderr():
        exported = torch.export.export(model.eval(), example_inputs)
        prepared = prepare_pt2e(exported.module(check_guards=False), quantizer)

    loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.calibration_images, transform=image_transform(spec)),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    with torch.inference_mode():
        for index, (image, _) in enumerate(loader):
            if index >= args.num_calib:
                break
            prepared(image)

    quantized_model = convert_pt2e(prepared)
    return export_no_guards(quantized_model, example_inputs)


def make_quantizer() -> Any:
    from executorch.backends.arm.ethosu import EthosUCompileSpec
    from executorch.backends.arm.quantizer import EthosUQuantizer, get_symmetric_quantization_config

    compile_spec = EthosUCompileSpec(
        target="ethos-u85-256",
        system_config="Ethos_U85_SYS_DRAM_Mid",
        memory_mode="Shared_Sram",
        extra_flags=["--output-format=raw"],
    )
    quantizer = EthosUQuantizer(compile_spec)
    quantizer.set_global(get_symmetric_quantization_config())
    return quantizer


def dump_tosa(exported: Any, profile: str, out_dir: Path, label: str) -> None:
    from executorch.backends.arm.tosa import TosaSpecification
    from executorch.backends.arm.tosa.compile_spec import TosaCompileSpec
    from executorch.backends.arm.util._factory import create_partitioner
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tosa_spec = TosaSpecification.create_from_string(profile)
    compile_spec = TosaCompileSpec(tosa_spec)
    compile_spec.dump_intermediate_artifacts_to(str(out_dir))
    partitioner = create_partitioner(compile_spec)

    print(f"Dumping {label}: {profile}")
    with suppress_stderr():
        to_edge_transform_and_lower(
            exported,
            partitioner=[partitioner],
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        )
    tosa_files = sorted(out_dir.rglob("*.tosa"))
    file_label = "file" if len(tosa_files) == 1 else "files"
    print(f"  {len(tosa_files)} .tosa {file_label} in {out_dir}")


def export_no_guards(model: Any, example_inputs: tuple[Any, ...]) -> Any:
    import torch

    first = torch.export.export(model, example_inputs)
    graph_module = first.module(check_guards=False)
    return torch.export.export(graph_module, example_inputs)


def image_transform(spec: dict[str, Any]) -> Any:
    import torchvision.transforms as transforms

    prep = spec["input"]["preprocessing"]
    return transforms.Compose(
        [
            transforms.Resize(prep["resize"]),
            transforms.CenterCrop(prep["center_crop"]),
            transforms.ToTensor(),
            transforms.Normalize(mean=prep["mean"], std=prep["std"]),
        ]
    )


def configure_quiet_output() -> None:
    os.environ.setdefault("GLOG_minloglevel", "2")
    os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    warnings.filterwarnings("ignore", category=FutureWarning)
    logging.getLogger().setLevel(logging.ERROR)


@contextmanager
def suppress_stderr() -> Any:
    saved_stderr = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved_stderr, 2)
        os.close(saved_stderr)
        os.close(devnull)


if __name__ == "__main__":
    main()
