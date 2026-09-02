# ExecuTorch Course Context

Use this file with the explain-mode and build-mode skills. It identifies the
parts of the repository that matter most and prevents generic course plumbing
from obscuring the ExecuTorch workflow.

## Course scope

Treat Modules 0-4 as the course:

- Module 0 introduces the course and ExecuTorch deployment concepts.
- Module 1 covers transformer export, portable execution, XNNPACK delegation,
  Python runtime use, and Edge Linux applications.
- Module 2 covers CNN classification, XNNPACK, post-training quantization,
  benchmarking, and a Python application.
- Module 3 moves the workflow to Android and uses the ExecuTorch Java API for
  semantic segmentation.
- Module 4 covers the Arm backend, Ethos-U, TOSA, Vela, the Corstone-320 FVP,
  ETRecord, ETDump, and an embedded C++ application.

## Investigation priority

Prioritize these subjects in order.

### 1. ExecuTorch ahead-of-time flow

Trace:

```text
PyTorch model
  -> example inputs and model contract
  -> torch.export
  -> optional backend-aware quantization
  -> ExecuTorch edge lowering and partitioning
  -> ExecuTorch program packaging
  -> .pte artifact
```

Identify the object entering and leaving each stage. Explain which graph regions
are delegated and which operations remain on the default runtime path.

### 2. ExecuTorch application integration

Trace:

```text
.pte program
  -> runtime load
  -> input tensor or value creation
  -> exported method invocation
  -> output tensor
  -> application postprocessing
  -> runtime cleanup
```

Focus on how Python, Java, and C++ applications call ExecuTorch. Explain
platform code when it affects model loading, tensor ownership, threading,
lifecycle, memory, or output correctness.

### 3. Model contract

Track the input and output shape, dtype, layout, preprocessing, labels, and
postprocessing from development-machine validation into the target
application. `model_spec.json` is the main contract shared between stages.

### 4. Deployment decisions

Explain the tradeoffs between:

- portable kernels and optimized delegation
- FP32 validation and INT8 deployment
- CPU and NPU execution
- delegated and fallback operations
- host validation, emulation, simulation, and physical-device execution

### 5. Supporting code

Spend less time on generic `argparse`, filesystem helpers, HTTP handling, HTML,
UI styling, output formatting, and ordinary language boilerplate. Cover these
only when the user asks or when they affect the ExecuTorch integration.

## API taxonomy

Do not describe every API in the workflow as an ExecuTorch API.

### PyTorch and PT2E APIs

- `torch.export.export()`
- `prepare_pt2e()`
- `convert_pt2e()`

These prepare the graph consumed by ExecuTorch and its backend integrations.

### ExecuTorch AOT and backend APIs

- `to_edge_transform_and_lower()`
- `XnnpackPartitioner`
- `XNNPACKQuantizer`
- `EthosUCompileSpec`
- `EthosUPartitioner`
- `EthosUQuantizer`
- `edge.transform()`
- `edge.to_executorch()`
- `ExecutorchBackendConfig`
- `generate_etrecord()`

### Python runtime binding

- `_load_for_executorch()`
- the loaded module's `forward()` method

Course helpers such as `load_executorch_program()` wrap the portable Python
binding. Follow the helper before describing the underlying API. Note that
`_load_for_executorch()` has a leading underscore and is exposed through the
portable Python bindings used by the course.

### Android runtime API

- `Module.load()`
- `Tensor.fromBlob()`
- `EValue.from()`
- `Module.forward()`
- `EValue.toTensor()`
- `Module.close()`

### Embedded C++ runtime API

- `runtime_init()`
- `Program::load()`
- `Program::load_method()`
- `Method::execute()`

## Repository conventions

- `scripts/aot/` contains development-machine preparation, export,
  quantization, lowering, validation, and artifact creation.
- `scripts/pi/` contains Raspberry Pi runtime, benchmark, and Python
  application code.
- `scripts/android/` contains Android packaging support.
- `android/` contains the Android application.
- `scripts/fvp/` contains FVP input and output conversion.
- `apps/` contains target application adaptations.
- `_support/` contains shared helpers and can hide the underlying runtime API.
- `artifacts/` contains downloaded or generated model inputs and deployable
  files. Do not treat generated artifacts as source code.
- `results/` contains benchmark and analysis output.

The Markdown explains the intended task. The invoked script is the source of
truth for the implemented API calls. Follow imports and helpers when the
Markdown shows only a wrapper or shortened excerpt.

## Official documentation for prototypes

Course scripts are intentionally pinned and may contain compatibility code or
workarounds for the version used in the module. Before adapting them into a new
prototype, check the current official documentation.

Start with:

- ExecuTorch getting started:
  `https://docs.pytorch.org/executorch/stable/getting-started.html`
- Model export and lowering:
  `https://docs.pytorch.org/executorch/stable/using-executorch-export.html`
- Export API reference:
  `https://docs.pytorch.org/executorch/stable/export-to-executorch-api-reference.html`
- API lifecycle and deprecation policy:
  `https://docs.pytorch.org/executorch/stable/api-life-cycle.html`
- Backend selection and requirements:
  `https://docs.pytorch.org/executorch/stable/backends-overview.html`
- Runtime overview and API reference:
  `https://docs.pytorch.org/executorch/stable/runtime-overview.html`
  and
  `https://docs.pytorch.org/executorch/stable/executorch-runtime-api-reference.html`
- Android integration:
  `https://docs.pytorch.org/executorch/stable/using-executorch-android.html`
- iOS integration:
  `https://docs.pytorch.org/executorch/stable/using-executorch-ios.html`
- C++ integration:
  `https://docs.pytorch.org/executorch/stable/using-executorch-cpp.html`
- PT2E quantization:
  `https://docs.pytorch.org/ao/stable/pt2e_quantization/index.html`
- XNNPACK backend:
  `https://docs.pytorch.org/executorch/stable/ios-xnnpack.html`
- Arm Ethos-U backend:
  `https://docs.pytorch.org/executorch/stable/embedded-arm-ethos-u.html`

For a prototype:

- confirm documentation matches the selected release rather than assuming the
  course's pinned APIs are current
- check whether an API is stable, experimental, deprecated, or private
- verify backend hardware, operator, quantization, build, and linking support
- confirm current Android dependencies, native ABIs, iOS frameworks, compiler
  requirements, and C++ runtime integration
- use official backend or hardware-vendor documentation for target-specific
  requirements such as Vela, memory modes, accelerators, and deployment tools
- record the official sources that justify important implementation decisions

When official guidance differs from the course, preserve the course behavior
only when reproducing its pinned environment. For a new prototype, use the
current supported path and explain the migration.

## Key entry points

Use this map to start an investigation, then confirm current paths with file
search rather than assuming the list is exhaustive.

### Module 1

- `module-1-transformer-inference-with-xnnpack/scripts/aot/prepare_opt125m.py`
- `module-1-transformer-inference-with-xnnpack/scripts/aot/prepare_distilbert_fillmask.py`
- `module-1-transformer-inference-with-xnnpack/scripts/aot/lower_to_executorch.py`
- `module-1-transformer-inference-with-xnnpack/scripts/pi/benchmark_model.py`
- `module-1-transformer-inference-with-xnnpack/scripts/pi/core/opt_chat_core.py`
- `module-1-transformer-inference-with-xnnpack/scripts/pi/core/distilbert_autocomplete_core.py`
- `module-1-transformer-inference-with-xnnpack/scripts/_support/common.py`

The web-serving files are secondary unless the user asks about application or
HTTP integration.

### Module 2

- `module-2-cnn-inference-with-quantization/scripts/aot/prepare_mobilenetv2.py`
- `module-2-cnn-inference-with-quantization/scripts/aot/lower_mobilenetv2_xnnpack.py`
- `module-2-cnn-inference-with-quantization/scripts/aot/quantize_mobilenetv2_xnnpack.py`
- `module-2-cnn-inference-with-quantization/scripts/pi/benchmark_mobilenetv2.py`
- `module-2-cnn-inference-with-quantization/scripts/pi/core/mobilenetv2_core.py`
- `module-2-cnn-inference-with-quantization/scripts/_support/common.py`

The optional object-detection extension is under
`module-2-cnn-inference-with-quantization/module_2_ext/`. Its main entry points
are `scripts/pi/download_yolov8s_ai_portal_model.py` and
`scripts/pi/serve_yolov8s_webcam.py` within that directory.

`create_calibration_images.py` explains how calibration inputs are produced,
but its image-generation code is not the main ExecuTorch API path.

### Module 3

- `module-3-executorch-on-android-semantic-segmentation/scripts/aot/prepare_lraspp_mobilenetv3.py`
- `module-3-executorch-on-android-semantic-segmentation/scripts/aot/lower_lraspp_xnnpack.py`
- `module-3-executorch-on-android-semantic-segmentation/scripts/aot/quantize_lraspp_xnnpack.py`
- `module-3-executorch-on-android-semantic-segmentation/scripts/aot/validate_cutout.py`
- `module-3-executorch-on-android-semantic-segmentation/scripts/aot/segmentation_preprocessing.py`
- `module-3-executorch-on-android-semantic-segmentation/scripts/android/sync_android_assets.py`
- `module-3-executorch-on-android-semantic-segmentation/android/ObjectCutout/app/src/main/java/org/arm/education/executorch/objectcutout/ExecuTorchSegmenter.java`

For the application contract and lifecycle, also inspect `ImagePreprocessor.java`,
`MaskPostprocessor.java`, `ModelSpec.java`, and `MainActivity.java` in the same
Java package. Start with `ExecuTorchSegmenter.java` for the runtime API.

### Module 4

- `module-4-ethos-npu-accelerated-inference/scripts/aot/lower_fp32_ethosu.py`
- `module-4-ethos-npu-accelerated-inference/scripts/aot/quantize_lower_ethosu.py`
- `module-4-ethos-npu-accelerated-inference/scripts/aot/dump_tosa_artifacts.py`
- `module-4-ethos-npu-accelerated-inference/scripts/fvp/create_mobilenetv2_input.py`
- `module-4-ethos-npu-accelerated-inference/scripts/fvp/decode_mobilenetv2_output.py`
- `module-4-ethos-npu-accelerated-inference/apps/fvp_image_classifier/arm_executor_runner_topk.patch`
- `module-4-ethos-npu-accelerated-inference/apps/ethosu_cpp_reference/model_runner.cpp`
- `module-4-ethos-npu-accelerated-inference/apps/ethosu_cpp_reference/classification_postprocess.cpp`

When the ExecuTorch source checkout is available, inspect the upstream Arm
executor runner to see the complete C++ runtime load and execute path.

## Course presentation style

The modules generally follow this pattern:

```text
explain the task
  -> run a command
  -> inspect relevant script excerpts
  -> interpret selected output
  -> validate the artifact or model behavior
  -> integrate it into an application
  -> compare deployment tradeoffs
```

Use these rules when explaining or extending the material:

- Later modules assume concepts introduced in earlier modules. Do not repeat a
  full introductory explanation unless the learner needs it.
- Code snippets select the relevant part of a larger script. Open the script
  before giving a detailed API explanation.
- Console examples show meaningful output rather than every warning or log
  line.
- FP32 establishes a correctness baseline before an INT8 program is selected
  for deployment.
- One sample input checks the workflow; it does not establish production model
  accuracy.
- A delegation count does not by itself establish correctness or performance.
- Host, emulator, and FVP results are not substitutes for physical-target
  performance measurements.
- Preprocessing and application postprocessing are not ExecuTorch APIs, but
  they remain part of end-to-end correctness.
- Prefer the model pipeline and runtime integration over course-specific prose,
  formatting, and supporting scripts.
