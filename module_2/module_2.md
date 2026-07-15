# Module 2: CNN Inference on Arm Cortex-A and and Optimize with INT8 Quantization

## Introduction

In Module 1, you compared PyTorch eager, portable ExecuTorch, and ExecuTorch
with XNNPACK for two transformer workloads. Portable ExecuTorch provided a
deployable baseline, while XNNPACK reduced latency by capturing expensive work
in optimized backend regions. The transformer graphs still contained many
delegation boundaries and fallback operations.

In this module, you will apply the same deployment workflow to MobileNetV2, a
lightweight convolutional neural network (CNN) for image classification. You
will first record the PyTorch result, then export the model to ExecuTorch
with XNNPACK. Finally, you will quantize the model to INT8 and measure the
effect on artifact size, latency, and classification output.

You will build and compare three runtime paths:

| Model and runtime path | Role in the comparison |
|---|---|
| PyTorch eager MobileNetV2 | Provides the full-stack latency and classification reference |
| FP32 ExecuTorch with XNNPACK | Shows the effect of deploying the CNN through optimized CPU kernels |
| INT8 ExecuTorch with XNNPACK | Shows the effect of quantization on artifact size, latency, and classification output |

Portable ExecuTorch is not repeated in this module. Module 1 already
used it as a lightweight, broadly compatible deployment path
rather than the main CPU performance path for these workloads. Keeping XNNPACK
as the backend for both `.pte` artifacts lets this module focus on two new
variables: a convolution-heavy model and INT8 precision.

The completed comparison shows the practical tradeoffs for running MobileNetV2
on an edge CPU: runtime latency, `.pte` artifact size, delegation, and
classification output. Producing the correct top-1 label for one image is not
enough on its own.

## Before You Begin

### Why MobileNetV2 Is an Appropriate Model for Edge CPU Inference

Running inference on an edge device can reduce latency, keep data local, and
support applications without reliable cloud connectivity. The model still has
to fit within the device's compute, memory, storage, and power limits.

MobileNetV2 was designed for efficient inference. It uses depthwise separable
convolutions and pointwise convolutions, which commonly map to optimized CPU
libraries. XNNPACK provides optimized kernels for supported operations, so a
convolution-heavy model such as MobileNetV2 can often be lowered into a small
number of large delegated regions. Compared with the transformer graphs in
Module 1, this can mean:

- fewer backend boundaries
- less work on the fallback path
- larger portions of the model handled by XNNPACK
- a greater opportunity to reduce latency

INT8 quantization changes model weights and, where supported, activations, from
FP32 to a lower-precision representation. This reduces storage and memory
bandwidth requirements and can make optimized integer kernels available.
Quantization also introduces numerical approximation, so confidence scores and
occasionally prediction rankings can change. The relevant check is whether the
model still gives useful results on images like the ones the application will
see.

### Learning Outcomes

By the end of this module, you will be able to:

- prepare pretrained MobileNetV2 files and record the required image size and
  preprocessing
- export MobileNetV2 to ExecuTorch and lower supported regions to XNNPACK
- read a delegation summary and identify which operations did not run through
  XNNPACK
- benchmark PyTorch eager and ExecuTorch with XNNPACK on a Raspberry Pi
- prepare calibration images for post-training quantization
- create an INT8 MobileNetV2 `.pte` artifact with the XNNPACK quantization flow
- compare FP32 and INT8 artifact size, latency, and top-5 classification output
- explain how model structure, XNNPACK delegation, calibration data, and
  classification results affect the deployment tradeoff

### Module Workflow and Script Layout

Module 2 follows the script workflow used in Module 1:

- `module_2/scripts/aot`: development-machine scripts for model preparation,
  export, lowering, calibration, and quantization
- `module_2/scripts/pi`: Raspberry Pi benchmark and demo entry points
- `module_2/scripts/pi/core`: Raspberry Pi runtime helpers
- `module_2/scripts/_support`: shared helpers

You will prepare and lower the model on the development machine, copy the
deployable `.pte` files to the Raspberry Pi, and measure them on the target.
The model family changes from transformers to CNNs, and we introduce the new quantization step, but the split between
ahead-of-time preparation and target-side execution remains the same.

## Development Machine Setup

Run the setup commands on your development machine from the repository root.

If you completed Module 1, reuse `.cpu_aot_venv` and install the Module 2
add-on requirements:

```bash
cd ~/executorch_on_arm
source .cpu_aot_venv/bin/activate
python -m pip install -r module_2/requirements.txt
```

For a clean clone, create the Python 3.11.14 environment, install the Module 1
base requirements, and then install the Module 2 add-ons:

```bash
cd ~/executorch_on_arm
PYENV_VERSION=3.11.14 python -m venv .cpu_aot_venv
source .cpu_aot_venv/bin/activate
python -m pip install -r module_1/requirements.txt
python -m pip install -r module_2/requirements.txt
```

Use Python 3.11.14 through pyenv for this module. The ExecuTorch Python package
contains native pybindings that must match the installed PyTorch build. An
environment created with another Python version can install incompatible wheels
and fail later when it loads a `.pte` artifact.

This is the same version requirement as Module 1. Reusing `.cpu_aot_venv` keeps
the ExecuTorch, PyTorch, and pybinding versions aligned while adding the additional dependencies required by this module.

## Task 1: Prepare MobileNetV2 and Its Input Contract

In this task, we will prepare one set of model weights, labels, image, tensor
shape, and preprocessing settings for every runtime path. This gives the rest
of the module the same inputs to work from, so later differences come from the
runtime path or precision rather than from a different input tensor.

### Prepare the Model Assets

Run on the development machine:

```bash
module_2/scripts/aot/prepare_mobilenetv2.py \
  --out module_2/artifacts/mobilenetv2
```

The script downloads the MobileNetV2 weights through TorchVision, saves the
state dictionary, writes the ImageNet labels and `model_spec.json`, and attempts
to download a sample dog image for benchmarking. TorchVision is the PyTorch
library that provides common computer vision datasets, image transforms, and
pretrained vision models.

You should expect to see output similar to this:

```text
Wrote module_2/artifacts/mobilenetv2/mobilenet_v2.pth
Wrote module_2/artifacts/mobilenetv2/imagenet_labels.json
Wrote module_2/artifacts/mobilenetv2/sample.jpg
Wrote module_2/artifacts/mobilenetv2/model_spec.json
```

The four files have distinct roles:

- `mobilenet_v2.pth`: the local TorchVision MobileNetV2 state dictionary
- `imagenet_labels.json`: the ImageNet class names used to interpret the output
  logits
- `sample.jpg`: the image used by the benchmark and demo commands
- `model_spec.json`: the image size, layout, and preprocessing settings used by
  export, benchmarking, quantization, and the web demo

This task creates the PyTorch-side inputs for later export and lowering. It does
not create an ExecuTorch program.

### Understand the Saved Model Data

The preparation script loads the pretrained model and saves its weights and
labels locally:

```python
weights = MobileNet_V2_Weights.IMAGENET1K_V1
model = mobilenet_v2(weights=weights).eval()
torch.save(model.state_dict(), weights_path)

labels = weights.meta["categories"]
labels_path.write_text(json.dumps(labels, indent=2) + "\n")
```

TorchVision supplies both the pretrained weights and the ImageNet categories.
Saving them in the module artifact directory ensures that export, benchmarking,
and the demo use the same model data.

The script records the image size and preprocessing in `model_spec.json`:

```python
"input": {
    "shape": [1, 3, 224, 224],
    "dtype": "float32",
    "layout": "nchw",
    "preprocessing": {
        "resize": 256,
        "center_crop": 224,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    },
}
```

The shape `[1, 3, 224, 224]` represents one RGB image in NCHW order: batch,
channels, height, and width. The resize, crop, mean, and standard deviation are
the ImageNet preprocessing values used by the pretrained model. The PyTorch
reference, ExecuTorch benchmarks, calibration flow, and web demo all read these
settings so they do not accidentally resize or normalize the image differently.

Module 1 used `model_spec.json` for token inputs such as `input_ids` and
`attention_mask`. Module 2 uses it for image tensor shape and preprocessing. In
both modules, the file keeps the development-machine and Raspberry Pi code
using the same model inputs.

### Task 1 Takeaway

You now have the shared model files, ImageNet labels, sample image, and
preprocessing settings that every later command will use. The important point
is that a fair runtime comparison starts before benchmarking: PyTorch,
ExecuTorch with XNNPACK, and INT8 ExecuTorch with XNNPACK all need to receive
the same resized, cropped, normalized image tensor.

## Task 2: Export MobileNetV2 and Lower It to XNNPACK

As in Module 1, ExecuTorch deploys a prepared program rather than a live PyTorch model. In this
task, we will capture the fixed-shape MobileNetV2 graph, delegate supported
work to XNNPACK, and package the result as an FP32 `.pte` artifact that can be
copied to the Raspberry Pi.

### Create the FP32 ExecuTorch Artifact

Run on the development machine:

```bash
module_2/scripts/aot/lower_mobilenetv2_xnnpack.py \
  --spec module_2/artifacts/mobilenetv2/model_spec.json
```

The script exports MobileNetV2, lowers supported regions to XNNPACK, prints a
delegation summary, and writes the FP32 `.pte` artifact and its size metadata.

You should expect to see output similar to this:

```text
Loading MobileNetV2 weights from module_2/artifacts/mobilenetv2/mobilenet_v2.pth...
Exporting MobileNetV2...
Lowering with XNNPACK...
Wrote module_2/artifacts/mobilenetv2/mobilenetv2_xnnpack.pte
PTE size: 13.99 MB
Wrote module_2/artifacts/mobilenetv2/mobilenetv2_xnnpack.json

Delegation
==========
XNNPACK delegate call sites: 1
Fallback call_function ops:  1
  <built-in function getitem> (1)
```

The generated `mobilenetv2_xnnpack.pte` is the FP32 artifact that will run on
the Raspberry Pi.

### Understand Export and Lowering

The central export and lowering sequence is:

```python
example_inputs = (torch.randn(*spec["input"]["shape"]),)

with torch.inference_mode():
    exported = export(model, example_inputs)

edge = to_edge_transform_and_lower(
    exported,
    partitioner=[XnnpackPartitioner()],
)
program = edge.to_executorch()
```

`torch.export` captures a fixed-shape PyTorch program using the input shape in
`model_spec.json`. `to_edge_transform_and_lower()` converts the exported
program to the ExecuTorch edge representation and uses `XnnpackPartitioner` to
identify subgraphs that XNNPACK supports. `to_executorch()` then packages the
lowered program as a deployable `.pte` file.

This is the same export-and-lower sequence used for OPT-125M and DistilBERT in
Module 1. The difference lies in the graph. Transformer graphs contain
attention, masks, reshapes, indexing, and normalization around their main
compute. MobileNetV2 is dominated by convolution patterns that can form a more
compact XNNPACK region.

### Read the Delegation Summary

For the example run, the FP32 artifact data was:

| Artifact | Backend | Precision | `.pte` size | XNNPACK delegate call sites | Fallback `call_function` operations |
|---|---|---|---:|---:|---:|
| `mobilenetv2_xnnpack.pte` | XNNPACK | FP32 | 13.99 MB | 1 | 1 |

One XNNPACK delegate call site means that ExecuTorch will hand one region to
XNNPACK at runtime. In this graph, that region contains the expensive CNN work.
The only remaining `call_function` operation is `<built-in function getitem>`.
This operation selects a value from the graph output. It is not part of the main
convolutional workload.

The script obtains this summary by walking the lowered graph:

```python
graph = edge.exported_program().graph_module.graph
delegate_op = torch.ops.higher_order.executorch_call_delegate

delegate_calls = 0
fallback_targets = []
for node in graph.nodes:
    if node.op != "call_function":
        continue
    if node.target == delegate_op:
        delegate_calls += 1
    else:
        fallback_targets.append(str(node.target))
```

A delegate call marks a region handled by XNNPACK. A fallback target remains on
the default ExecuTorch path. Counts alone do not describe the amount of work in
either path: one delegate call can contain most of a model, and one fallback
operation can be a small helper operation such as selecting an output value.
Read the delegate regions and fallback list together, then confirm their
practical effect with target-side timing.

The example is a useful contrast with Module 1. XNNPACK captured useful work in
the transformer models, but their graph structure produced more boundaries.
MobileNetV2's convolution-heavy structure allows its main compute to sit in one
delegated region.

Compare the delegation counts from Module 1 with the MobileNetV2 count in this module:

| Model | Workload | XNNPACK delegate call sites | Fallback `call_function` operations |
|---|---|---:|---:|
| OPT-125M | Autoregressive transformer decoder | 50 | 632 |
| DistilBERT | Transformer encoder | 27 | 354 |
| MobileNetV2 | Convolutional image classifier | 1 | 1 |

This table should not be read as a percentage comparison. The models have
different graph sizes and different operation costs, so the counts are not
directly comparable as coverage metrics. The table shows graph fragmentation:
OPT-125M and DistilBERT are split across many delegated and fallback
operations, while MobileNetV2 is captured as one large XNNPACK region with one
small helper operation left on the default ExecuTorch path.

### Task 2 Takeaway

You have converted the PyTorch model into a deployable FP32 ExecuTorch program
and checked how much of the graph XNNPACK will run. When you read the
delegation summary, remember that one delegate call can contain most of the
model's work, while a fallback operation may be a small helper such as selecting
an output value. The next task tests whether that graph placement reduces
latency on the target CPU.

## Task 3: Benchmark PyTorch and FP32 ExecuTorch on the Raspberry Pi

Delegation shows where ExecuTorch plans to run parts of the graph; it does not
tell you how fast the model will run on the Raspberry Pi. In this task, we will
run PyTorch eager and FP32 ExecuTorch with XNNPACK on the same Raspberry Pi,
using the same weights, image, preprocessing, warm-up count, and run count.
That lets us measure the effect of changing the runtime while keeping the input
and model fixed.

Module 1 used two warm-ups and 20 measured runs for its slower transformer
paths, and only one measured run for portable ExecuTorch. MobileNetV2 completes
each forward pass much more quickly, so ten warm-ups and 50 measured runs remain
practical and provide a larger latency sample.

### Prepare the Raspberry Pi Runtime

If you connect over SSH, forward the benchmark dashboard and web application
ports:

```bash
ssh -L 8080:localhost:8080 -L 8092:localhost:8092 <user>@<pi>
```

| Port | Use |
|---:|---|
| `8080` | benchmark results dashboard |
| `8092` | MobileNetV2 image classifier demo |

Clone or copy the course repository to the Pi, then create the runtime
environment. If you previously performed Module 1 setup on this Pi, these commands will just add additional installs:

```bash
cd ~/executorch_on_arm
PYENV_VERSION=3.11.14 python -m venv .cpu_lab_venv
source .cpu_lab_venv/bin/activate
python -m pip install -r module_1/requirements.txt
python -m pip install -r module_2/requirements.txt
```

As in Module 1, you keep the Pi runtime environment separate from the development
machine's AOT environment. Both use the same Python version and pinned base
dependencies, but they serve different parts of the workflow.

Copy the MobileNetV2 artifacts from the development machine:

```bash
ssh <user>@<pi> "mkdir -p ~/executorch_on_arm/module_2/artifacts"
scp -r module_2/artifacts/mobilenetv2 \
  <user>@<pi>:~/executorch_on_arm/module_2/artifacts/
```

### Run the PyTorch Reference

```bash
cd ~/executorch_on_arm
source ~/executorch_on_arm/.cpu_lab_venv/bin/activate

module_2/scripts/pi/benchmark_mobilenetv2.py \
  --runtime pytorch \
  --spec ~/executorch_on_arm/module_2/artifacts/mobilenetv2/model_spec.json \
  --image ~/executorch_on_arm/module_2/artifacts/mobilenetv2/sample.jpg \
  --runs 50 \
  --warmups 10 \
  --out ~/executorch_on_arm/module_2/results/mobilenetv2_pytorch.json
```

This run records the classification output and latency for the full PyTorch
stack.

### Run FP32 ExecuTorch with XNNPACK

```bash
module_2/scripts/pi/benchmark_mobilenetv2.py \
  --runtime executorch \
  --backend xnnpack \
  --spec ~/executorch_on_arm/module_2/artifacts/mobilenetv2/model_spec.json \
  --pte ~/executorch_on_arm/module_2/artifacts/mobilenetv2/mobilenetv2_xnnpack.pte \
  --image ~/executorch_on_arm/module_2/artifacts/mobilenetv2/sample.jpg \
  --runs 50 \
  --warmups 10 \
  --out ~/executorch_on_arm/module_2/results/mobilenetv2_xnnpack.json
```

Both benchmark paths use the same preprocessed tensor. The script changes only
the runtime call:

```python
if args.runtime == "pytorch":
    def forward():
        return module(input_tensor)
else:
    def forward():
        return module.forward((input_tensor,))[0]
```

The PyTorch branch calls the `torch.nn.Module`. The ExecuTorch branch loads the
`.pte` artifact and calls it through the ExecuTorch runtime. The source image,
preprocessing, weights, warm-ups, and measured run count remain fixed.

### Interpret the Example Results

On the Raspberry Pi, you should expect to see top-5 predictions similar
to this:

| Rank | Label | Score |
|---:|---|---:|
| 1 | Staffordshire bullterrier | 71.62% |
| 2 | bull mastiff | 13.37% |
| 3 | French bulldog | 8.97% |
| 4 | American Staffordshire terrier | 5.01% |
| 5 | boxer | 0.82% |

Example latency measurements are:

| Runtime path | Warm-ups | Runs | Mean | Median | Min | Max |
|---|---:|---:|---:|---:|---:|---:|
| PyTorch eager | 10 | 50 | 72.55 ms | 72.15 ms | 71.77 ms | 76.96 ms |
| ExecuTorch with XNNPACK | 10 | 50 | 10.89 ms | 10.88 ms | 10.73 ms | 11.63 ms |

Your own results will differ slightly, but should be similar.

The matching predictions confirm that the runtime comparison is performing the
same classification for this input. Mean latency fell from 72.55 ms to 10.89
ms, a speedup of approximately 6.7 times for this run. This timing result is
consistent with the delegation report: XNNPACK received the model's main
convolutional workload as one region.

### Task 3 Takeaway

You have measured the runtime change on the target device while keeping the
model input and classification result fixed. In the example measurements,
ExecuTorch with XNNPACK reduced mean latency by about 6.7 times. Delegation and
benchmarking answer different questions: delegation shows where the graph is
expected to run, while the Raspberry Pi benchmark shows the actual latency
improvement. MobileNetV2 delegated cleanly to XNNPACK in one single delegate region where the heavy convolution compute is optimized. This caused the substantial improvements in latency.

## Task 4: Quantize MobileNetV2 to INT8

The FP32 ExecuTorch artifact improves latency, but edge deployment also places
limits on storage, memory bandwidth, and response time. Post-training
quantization produces a second artifact that uses INT8 values for supported
parts of the graph. In this task, we will calibrate the model, convert it,
lower it to XNNPACK, and compare the new INT8 `.pte` artifact with the FP32
artifact.

Quantization happens ahead of deployment. ExecuTorch does not take the FP32
`.pte` file and quantize it at runtime. The workflow starts from the FP32
PyTorch program, applies quantization preparation and conversion, and then
packages a new INT8 `.pte` artifact.

If you completed Module 1's optional LLM deployment section, you saw
quantization selected through LLM-specific export flags such as `--qlinear` and
`--qembedding`. This task exposes a different path directly: PT2E
post-training quantization with observers, a calibration dataset, conversion,
and XNNPACK lowering.

### Understand Post-Training Quantization

Post-training quantization (PTQ) starts with a trained FP32 model. Preparation
adds observers to the graph. A calibration pass then runs example images
through the prepared model so the observers can record activation ranges.
Conversion uses those ranges to select parameters such as scales and
zero-points and replaces supported FP32 computation with quantized forms.

This module uses `XNNPACKQuantizer`, so the quantization configuration is
designed for the backend that will run the model. This is preferable to applying
a generic INT8 transformation and expecting the backend to accept every
resulting pattern. The converted graph must still contain forms that XNNPACK can
lower and execute efficiently.

The three expected benefits are:

- a smaller model artifact
- lower memory bandwidth requirements
- lower latency where optimized integer kernels are available

The cost is numerical approximation. We will evaluate that cost by checking the
classification results, not by requiring bit-for-bit agreement with FP32.

### Create the Calibration Image Set

Run on the development machine:

```bash
module_2/scripts/aot/create_calibration_images.py \
  --out module_2/artifacts/mobilenetv2/calibration_images \
  --images-per-class 10
```

On its first run, the script downloads the CIFAR-10 archive. This can take
several minutes. TorchVision or NumPy may also print a deprecation warning while
the archive is unpacked.

You should expect to see output similar to this:

```text
Saved calibration images:
  airplane: 10 images
  automobile: 10 images
  bird: 10 images
  cat: 10 images
  deer: 10 images
  dog: 10 images
  frog: 10 images
  horse: 10 images
  ship: 10 images
  truck: 10 images
```

The result is a small ImageFolder-style subset containing 100 images across ten
classes. It is suitable for demonstrating PTQ in this lab. A production
calibration image set should be versioned and should match the images the
deployed application will receive, including relevant lighting, composition,
camera characteristics, and content.

The script downloads CIFAR-10 once and writes the selected images into class
folders:

```python
dataset = datasets.CIFAR10(
    root=args.data_root,
    train=True,
    download=True,
    transform=transforms.ToTensor(),
)

for image, label in dataset:
    class_dir = args.out / class_name
    save_image(image, class_dir / f"img_{index:04d}.jpg")
```

The directory layout allows the quantization script to load the data with
`torchvision.datasets.ImageFolder`. Calibration does not train the model and
does not use the labels as targets. It only supplies realistic input values for
the observers.

![CIFAR-10 calibration images used for MobileNetV2 post-training quantization.](images/mobilenetv2-calibration-image-grid.png)

### Quantize and Lower the Model

```bash
module_2/scripts/aot/quantize_mobilenetv2_xnnpack.py \
  --spec module_2/artifacts/mobilenetv2/model_spec.json \
  --calibration-images module_2/artifacts/mobilenetv2/calibration_images \
  --num-calib 100
```

The script runs PT2E post-training quantization with the XNNPACK quantizer,
calibrates the prepared graph, converts it, lowers it to ExecuTorch with
XNNPACK, and writes the INT8 `.pte` artifact.

You should expect to see output similar to this:

```text
Calibrating on up to 100 images from module_2/artifacts/mobilenetv2/calibration_images...
Wrote module_2/artifacts/mobilenetv2/mobilenetv2_int8_xnnpack.pte
INT8 PTE size: 7.59 MB
FP32 PTE size: 13.99 MB
Wrote module_2/artifacts/mobilenetv2/mobilenetv2_int8_xnnpack.json
```

In that run, the INT8 artifact was 7.59 MB rather than 13.99 MB, a reduction of
approximately 46%.

Module 1 described the theoretical fourfold storage reduction when a value
moves from 32 bits to 8 bits. That ratio does not apply automatically to the
complete `.pte` artifact. The file also contains graph and backend metadata,
not every component is quantized, and this lab leaves the classifier head in
FP32 by default. Measure the finished artifact instead of deriving its size
from the precision ratio.

PT2E means PyTorch 2 Export quantization. The complete preparation path is:

```text
FP32 PyTorch model
  -> torch.export
  -> prepare_pt2e with XNNPACKQuantizer
  -> calibration images
  -> convert_pt2e
  -> torch.export quantized model
  -> lower to ExecuTorch with XNNPACK
  -> INT8 .pte artifact
```

The script configures the quantizer as follows:

```python
quantizer = XNNPACKQuantizer()
quantizer.set_global(get_symmetric_quantization_config(is_per_channel=True))

if not args.quantize_classifier_head:
    quantizer.set_filter_function(skip_classifier_head)
```

The global configuration requests symmetric quantization with per-channel
weight quantization where supported. Per-channel quantization gives each output
channel its own scale, which is often useful for convolution weights. By
default, the script leaves the final classifier head in FP32. The classifier
head is the last layer that maps MobileNetV2 features to ImageNet class scores.
This is a conservative compromise between optimization and classification
stability: most of the model's compute is in the convolutional feature
extractor, while the classifier head directly produces the final ImageNet
scores. You can optionally pass `--quantize-classifier-head` as a follow-up
experiment; it may reduce size or latency a little more, but rerun the
delegation, latency, and top-5 checks because the final scores may shift.

Next, the script prepares the exported model for calibration by adding observer
modules:

```python
exported = export(model, example_inputs)
prepared = prepare_pt2e(exported.module(), quantizer)
move_exported_model_to_eval(prepared)
```

The observers collect values during calibration; their presence alone does not
make inference faster. The calibration loop supplies the observed ranges:

```python
for index, (image, _) in enumerate(calibration_loader):
    if index >= args.num_calib:
        break
    prepared(image)
```

Calibration quality depends on input quality. A set with different lighting,
composition, preprocessing, or subject matter from the deployed workload can
produce activation ranges that do not represent real use. Module 1 did not
include this step because its transformer comparisons remained in FP32.

After calibration, conversion creates the quantized model. The script exports
and lowers that converted graph:

```python
quantized_model = convert_pt2e(prepared)
exported_quantized = export(quantized_model, example_inputs)
edge = to_edge_transform_and_lower(
    exported_quantized,
    partitioner=[XnnpackPartitioner()],
    compile_config=EdgeCompileConfig(_check_ir_validity=False),
)
program = edge.to_executorch()
```

The resulting graph is no longer the original FP32 program. It contains the
quantized operations and delegated regions that ExecuTorch will execute. The
metadata JSON records the artifact path and size, calibration count, and
delegation summary.

### (Optional) Inspect the FP32 and INT8 Graphs with Model Explorer

Use Model Explorer if you want a visual comparison with the transformer graphs
from Module 1 and with the FP32 MobileNetV2 graph from earlier in this module.

Install the optional tooling used in Module 1:

```bash
.cpu_aot_venv/bin/python -m pip install -r module_1/requirements-model-explorer.txt
```

Start Model Explorer with the Arm `.pte` adapter:

```bash
.cpu_aot_venv/bin/model-explorer \
  --port 8081 \
  --extensions=pte_adapter_model_explorer
```

Open the FP32 and INT8 artifacts:

```text
module_2/artifacts/mobilenetv2/mobilenetv2_xnnpack.pte
module_2/artifacts/mobilenetv2/mobilenetv2_int8_xnnpack.pte
```

In both graphs, look for the large `LoweredBackendModule` region. The FP32 and
INT8 graphs will not look identical because quantization changes the operations
that are lowered, but the main question is the same: does XNNPACK contain the
expensive MobileNetV2 work? Compare both MobileNetV2 artifacts with the more
fragmented transformer graphs from Module 1. Press `Ctrl+C` in the terminal
when you have finished.

### Copy and Benchmark the INT8 Artifact

Copy the new artifact to the Raspberry Pi:

```bash
scp module_2/artifacts/mobilenetv2/mobilenetv2_int8_xnnpack.pte \
  <user>@<pi>:~/executorch_on_arm/module_2/artifacts/mobilenetv2/
```

Run the benchmark:

```bash
module_2/scripts/pi/benchmark_mobilenetv2.py \
  --runtime executorch \
  --backend int8_xnnpack \
  --spec ~/executorch_on_arm/module_2/artifacts/mobilenetv2/model_spec.json \
  --pte ~/executorch_on_arm/module_2/artifacts/mobilenetv2/mobilenetv2_int8_xnnpack.pte \
  --image ~/executorch_on_arm/module_2/artifacts/mobilenetv2/sample.jpg \
  --runs 50 \
  --warmups 10 \
  --out ~/executorch_on_arm/module_2/results/mobilenetv2_int8_xnnpack.json
```

On the Raspberry Pi, you should expect to see output similar to this:

| Rank | Label | Score |
|---:|---|---:|
| 1 | Staffordshire bullterrier | 88.9% |
| 2 | French bulldog | 4.2% |
| 3 | American Staffordshire terrier | 4.1% |
| 4 | bull mastiff | 2.7% |
| 5 | boxer | 0.1% |

The latency result was:

| Runtime path | Warm-ups | Runs | Mean | Median | Min | Max |
|---|---:|---:|---:|---:|---:|---:|
| INT8 ExecuTorch with XNNPACK | 10 | 50 | 4.01 ms | 3.98 ms | 3.95 ms | 4.56 ms |

The top prediction remained `Staffordshire bullterrier`, and the same five dog
breeds appeared in the top five, although their scores and order changed. Such
numerical differences are expected after quantization. For this sample, the
classification output remained useful.

Mean latency fell from 10.89 ms for FP32 ExecuTorch with XNNPACK to 4.01 ms for
the INT8 path. Relative to the 72.55 ms PyTorch eager reference, the INT8 path
was approximately 18.1 times faster in this run.

### Is Quantization Always Straightforward?

CNN classifiers are often a practical starting point for quantization because
convolution, activation, and pooling patterns are common in backend
quantization flows; their input shape is usually fixed; and top-5
classification often tolerates small score changes.

Other model types can be more sensitive. Detection models may include heads and
postprocessing thresholds affected by small numerical changes. Transformer
graphs can contain masks, normalization, indexing, shape operations, and
dynamic sequence behavior that fragment the graph or increase numerical
sensitivity. A successful export therefore does not guarantee a satisfactory
quantized model.

Possible remedies include a larger calibration set that better matches the
application's images,
selective quantization, mixed precision, or quantization-aware training (QAT).
QAT simulates quantization effects during training or fine-tuning so that the
model can adapt before final INT8 conversion.

To explore techniques such as QAT, begin with the same MobileNetV2 workflow: train or
fine-tune with QAT enabled, convert the prepared model, lower it to ExecuTorch
with XNNPACK, and repeat the artifact size, delegation, latency, and top-5
checks.

For initial resources to get started, see the
[PyTorch 2 Export QAT documentation](https://docs.pytorch.org/ao/stable/pt2e_quantization/pt2e_quant_qat.html),
which matches this module's PT2E and `XNNPACKQuantizer` approach. The broader
[torchao QAT workflow](https://docs.pytorch.org/ao/stable/workflows/qat.html)
explains fake quantization and the prepare, train, and convert sequence.

### Task 4 Takeaway

You have created a second `.pte` artifact from the same FP32 PyTorch model,
this time using PT2E post-training quantization for ExecuTorch with XNNPACK.
The important pieces are calibration, conversion, and measurement: calibration
supplies the activation ranges, conversion creates the quantized graph, and the
Raspberry Pi benchmark checks whether the result is still acceptable. In the
example run, INT8 reduced artifact size by about 46% and mean latency from
10.89 ms to 4.01 ms while keeping a useful top-5 result for the sample image.

## Task 5: Review the Benchmark Results Together

After the individual benchmark runs, we will view the results together in the
dashboard. This makes it easier to compare average latency, run-to-run
variation, artifact size, classification output, and delegation instead of
looking only at one number.

Run on the Raspberry Pi:

```bash
shared/serve_results.py \
  --results ~/executorch_on_arm/module_2/results \
  --host 127.0.0.1 \
  --port 8080
```

Open on the development machine:

```text
http://localhost:8080
```

The summary shows the three MobileNetV2 runs side by side:

![MobileNetV2 benchmark summary table.](images/mobilenetv2-benchmark-summary.png)

The PyTorch eager and FP32 ExecuTorch with XNNPACK plots show the latency
distribution and run-to-run variation for the first two paths:

![MobileNetV2 PyTorch and FP32 XNNPACK latency plots.](images/mobilenetv2-pytorch-xnnpack-latency-plots.png)

The INT8 plots show the quantized path:

![MobileNetV2 INT8 XNNPACK latency plots.](images/mobilenetv2-int8-xnnpack-latency-plots.png)

Use the summary table for the average latency comparison. Use the histograms
and run-by-run plots to verify that the result is consistent across the
measured runs.

Complete the following table with your results:

| Comparison point | PyTorch eager | FP32 ExecuTorch with XNNPACK | INT8 ExecuTorch with XNNPACK | What to compare |
|---|---|---|---|---|
| Latency |  |  |  | Response time on the target device |
| `.pte` artifact size | N/A |  |  | Deployable artifact size |
| Top-5 output |  |  |  | Classification output for the sample image |
| Delegation | N/A |  |  | How much work XNNPACK runs |

### Task 5 Takeaway

You have brought the three runtime paths into one comparison. The dashboard
helps you check both the average latency and the spread across runs, while the
results table keeps size, classification output, and delegation in the same
view.
A deployment path needs acceptable latency, artifact size, and classification
output, not just a good mean time.

## Task 6: Run the Image Classifier Demo

The benchmark uses one fixed image to make the runtime comparison controlled.
In this task, we will place the INT8 `.pte` artifact inside a small application
so that you can check the complete input-to-output path with additional images.

Run on the Raspberry Pi:

```bash
module_2/scripts/pi/serve_image_classifier.py \
  --runtime executorch \
  --spec ~/executorch_on_arm/module_2/artifacts/mobilenetv2/model_spec.json \
  --pte ~/executorch_on_arm/module_2/artifacts/mobilenetv2/mobilenetv2_int8_xnnpack.pte \
  --host 127.0.0.1 \
  --port 8092
```

Open on the development machine:

```text
http://localhost:8092
```

Upload JPEG or PNG images and review the top-5 predictions and model latency.
This demo is an integration check rather than a complete image-classification
product.

The application uses the same inference sequence as the benchmark:

```python
input_tensor = self._preprocess(image_bytes)
output = self.model.forward((input_tensor,))[0]

probabilities = torch.nn.functional.softmax(output[0], dim=0)
scores, indices = torch.topk(probabilities, topk)
```

The uploaded image is converted to the normalized tensor described by
`model_spec.json`. The ExecuTorch program returns logits; softmax and top-k
selection convert them to labels and confidence scores.

This follows the same application pattern as the Module 1 demos: load a `.pte`
artifact, convert user input to the expected tensor representation, run
inference, and translate the raw output into application behavior.

The demo displays the uploaded image, top-5 predictions, and latency for the
INT8 ExecuTorch with XNNPACK artifact:

![MobileNetV2 classifier demo showing uploaded image and top-5 predictions.](images/mobilenetv2-classifier-demo.png)

### Task 6 Takeaway

You have moved from a benchmark command to a small application that accepts
user-supplied images. This shows that the `.pte` artifact is only one part of a
working deployment: the application must turn image bytes into the expected
tensor, call the ExecuTorch runtime, convert logits to probabilities, and map
the top results back to labels. The same image size, normalization, and label
mapping used during preparation and benchmarking still matter in the demo.

## Compare the Deployment Tradeoffs

You now have four comparison points:

- **Model architecture:** MobileNetV2 uses convolution patterns that align well
  with optimized CPU kernels.
- **Backend delegation:** the example graph places its expensive CNN work in one
  XNNPACK region and leaves only a small helper operation on the default
  ExecuTorch path.
- **Quantization:** the example INT8 artifact is smaller and faster than the
  FP32 artifact.
- **Classification results:** the top prediction and set of top-five labels
  remain useful for the sample image, despite changed confidence values and
  ranking.

You can use your own measurements and images like the ones the application will
see to decide whether the deployment fits the application:

- latency is low enough for the intended interaction
- the artifact fits the device's storage and memory constraints
- classification quality is acceptable across the expected image types
- preprocessing and application integration remain consistent with the model
  input settings

The comparison with Module 1 demonstrates a broader deployment principle. The
framework and hardware do not determine performance on their own. Model graph
structure affects both delegation and quantization, so model choice is part of
system optimization.

## Check Your Understanding

Question 1. Which recorded result best shows that XNNPACK improved the CNN
deployment?

A. A significant target-side latency reduction after delegation

B. A longer export time on the development machine

C. A larger exported `.pte` artifact

D. Additional image preprocessing operations

<details>
<summary>Answer</summary>

**A. A significant target-side latency reduction after delegation**

Delegation describes the graph mapping; the target-side measurement confirms
the practical performance benefit.
</details>

Question 2. The INT8 model produces different confidence values but retains
useful classification output. Why can this be acceptable for deployment?

A. Quantization always improves model accuracy

B. Small numerical changes are acceptable when the classifier still gives useful
results and inference becomes more efficient

C. Confidence values never matter in deployed classifiers

D. Quantization guarantees identical rankings on all inputs

<details>
<summary>Answer</summary>

**B. Small numerical changes are acceptable when the classifier still gives
useful results and inference becomes more efficient**
</details>

Question 3. Which two results are required to support an upload-based or live
camera application?

A. The exported artifact exists

B. Top-5 results remain useful on images like the application will receive

C. Latency meets the application's responsiveness requirement

D. Calibration completed without an error

<details>
<summary>Answer</summary>

**B and C.** The application needs both useful predictions and acceptable
response time. Export and calibration are preparation steps; they do not show
whether the deployed application is fast enough or accurate enough for its use
case.
</details>

## Module 2 Summary

In this module, you moved a vision model through the full edge CPU deployment
path. You prepared MobileNetV2 with a fixed image size, preprocessing recipe,
labels, and sample image; exported it to an FP32 `.pte` artifact; lowered
supported work to XNNPACK; benchmarked it on a Raspberry Pi; then created and
measured an INT8 `.pte` artifact.

You also compared the practical tradeoffs between the three runtime paths:

- The same weights, labels, image size, and preprocessing must be used across
  PyTorch, FP32 ExecuTorch with XNNPACK, and INT8 ExecuTorch with XNNPACK.
- A delegation report tells you which parts of the graph are handed to
  XNNPACK, but the Raspberry Pi benchmark tells you whether that actually
  improves latency.
- MobileNetV2 is a good fit for this exercise because its convolution-heavy
  graph can be captured as one large XNNPACK region with only a small fallback
  operation.
- PTQ is an ahead-of-time conversion step. Calibration images give the
  quantizer the activation ranges it needs before the INT8 `.pte` artifact is
  created.
- INT8 quantization can change confidence scores and ranking details, so you
  compare top-5 output as well as artifact size and latency.
- The web demo checks the application path: image upload, preprocessing,
  ExecuTorch runtime call, softmax, top-k selection, and label lookup.

For the example measurements in this module, FP32 ExecuTorch with XNNPACK was
much faster than PyTorch eager, and the INT8 artifact was smaller and faster
again while preserving a useful top-5 result for the sample image. The same
workflow can be applied to another model, but you need to rerun the same
comparisons. Export success alone does not prove that delegation is good, that
quantization is acceptable, or that the target device meets the application's
latency and quality requirements.
