# FVP Image Classification Runner

This folder contains a small patch for the upstream ExecuTorch Arm executor
runner. The patch keeps the generic runner behavior, then adds MobileNetV2
top-5 ImageNet label and class-ID printing for the first output tensor.

Generate the label header before applying the patch:

```bash
python ~/executorch_on_arm/module-4-ethos-npu-accelerated-inference/apps/fvp_image_classifier/generate_imagenet_labels_header.py \
  --labels ~/executorch_on_arm/module-4-ethos-npu-accelerated-inference/artifacts/mobilenetv2/imagenet_labels.json \
  --out ~/executorch/examples/arm/executor_runner/imagenet_labels.h
```

Apply it from the ExecuTorch repository root as described in the live course when available.

```bash
cd ~/executorch
git apply --directory=examples/arm/executor_runner \
  ~/executorch_on_arm/module-4-ethos-npu-accelerated-inference/apps/fvp_image_classifier/arm_executor_runner_topk.patch
```

After building the classification runner, restore the upstream source with the
reverse patch:

```bash
git apply -R --directory=examples/arm/executor_runner \
  ~/executorch_on_arm/module-4-ethos-npu-accelerated-inference/apps/fvp_image_classifier/arm_executor_runner_topk.patch

rm examples/arm/executor_runner/imagenet_labels.h
```
