# Module 0: Overview of the Course, and of ExecuTorch

Welcome to Optimize and Deploy AI Inference with ExecuTorch on Arm. In
these hands-on sessions, you will explore how PyTorch models can be
exported and lowered to the ExecuTorch runtime for efficient,
lightweight inference at the edge.

You will work across multiple deployment targets, including: 

- CPU-based inference using XNNPACK and Arm KleidiAI

- deployment to Arm Ethos-U Neural Processing Units (NPUs)

## Course Overview

### **What will you learn?**

Upon completion of this course, you will be able to:

- export and lower PyTorch transformer, classification, and semantic
  segmentation models to ExecuTorch `.pte` programs

- compare PyTorch eager execution with ExecuTorch portable and delegated
  execution on an Arm CPU

- use XNNPACK for optimized CPU execution and explain how delegation changes
  the exported graph

- apply post-training quantization, calibrate an INT8 model, and check the
  resulting size, latency, delegation, and model output

- move an ExecuTorch application from Python on Edge Linux to Java or Kotlin
  on Android

- prepare a model for the Arm Ethos-U backend using TOSA and Vela, then run it
  on the Corstone-320 Fixed Virtual Platform (FVP)

- inspect exported and delegated programs with Model Explorer, ETRecord,
  ETDump, and ExecuTorch Inspector

### **Prerequisites**

To run every exercise, you will need:

- a Raspberry Pi 5 for Modules 1 and 2

- an Android smartphone or an Android Studio emulator for Module 3

- a Linux host or cloud instance for Module 4

- basic experience with Python and a working understanding of AI and
  machine learning concepts

Module 3 introduces Java and Android development, but prior Java or Kotlin
experience is not required. If you do not have every target device, you can
still run the ahead-of-time model preparation steps and review the target-side
commands and application code.

### **Course modules**

**Module 0: Introduction and Setup**

Introduces the course, ExecuTorch, and how you should approach this resource.

**Estimated time to complete: 15 minutes**

**Module 1: Transformer Inference and Delegation on Arm CPUs**

Prepare OPT-125M and DistilBERT, lower them to the ExecuTorch Portable Kernel
Library and XNNPACK, inspect their graphs, benchmark them on a Raspberry Pi 5,
and run chatbot and autocomplete applications.

**Estimated time to complete: 120-150 minutes**

**Module 2: CNN Inference and INT8 Quantization on Arm CPUs**

Prepare MobileNetV2 for image classification, delegate it to XNNPACK, compare
PyTorch and FP32 ExecuTorch performance, then use post-training quantization to
create and validate an INT8 image-classification application.

**Estimated time to complete: 90-120 minutes**

**Module 3: ExecuTorch on Android for Semantic Segmentation**

Prepare FP32 and INT8 LR-ASPP MobileNetV3 programs with XNNPACK, validate their
semantic-segmentation output, and package the INT8 model in an Android object-
cutout application. Use the ExecuTorch Java API to load the model, create
tensors, run inference, and manage native runtime resources.

**Estimated time to complete: 90-120 minutes**

**Module 4: Deploy ExecuTorch Models with Arm Ethos-U Acceleration**

Quantize MobileNetV2 for Ethos-U, lower supported regions through TOSA and
Vela, examine unsupported operations and delegation boundaries, and run the
program on the Corstone-320 FVP. Inspect ETRecord and ETDump data and adapt the
generic C++ runner into an image-classification application.

**Estimated time to complete: 120-180 minutes**

These estimates assume that commands complete successfully. Large downloads,
dependency compilation, and environment-specific troubleshooting can add time,
particularly in Module 4.

## ExecuTorch: How do I move a PyTorch model from development to efficient edge deployment?**

Edge AI refers to deploying machine learning models on small, low-power
devices with stricter constraints on memory, compute, and energy. Due to
the ubiquitous nature of CPUs, as every general-purpose, compute-enabled
device contains a CPU, inference can be performed on a wide range of
devices.

Edge AI is not only about performance. It also enables:

- processing sensitive data locally rather than sending it to the cloud

- operating without network connectivity

- deploying AI on lower-cost, lower-power devices

**PyTorch** is a popular open-source machine learning framework for
building and training deep learning models. Originally developed by
Meta’s AI Research lab (FAIR) in 2016, PyTorch has become a cornerstone
of AI research and production thanks to its flexible, Pythonic design.
It offers dynamic computation graphs, known as eager execution, an
intuitive API, and a vast ecosystem, including TorchVision, Hugging Face
Transformers, and others that make it easy for engineers to prototype
and train neural networks. In essence, PyTorch is widely used to develop
models on powerful hardware, such as GPUs and CPUs, accelerating the
path from research prototyping to production deployments.

**ExecuTorch** is part of the PyTorch workflow and is described as
"PyTorch’s unified solution for deploying AI models on-device, from
smartphones to microcontrollers, built for privacy, performance, and
portability.” It is an inference runtime and toolchain that takes a
PyTorch model and prepares it for efficient on-device execution.
Importantly, ExecuTorch is PyTorch-native. It uses the same APIs and
requires no translation to another format, unlike exporting to ONNX or
TFLite. The core ExecuTorch runtime is extremely lightweight: in the
order of tens of kilobytes. In contrast, the full PyTorch library is
hundreds of megabytes. This difference enables deployment in
environments where:

- memory is limited

- startup time matters

- dependencies must be minimal

In many edge scenarios, this determines whether deployment is possible
at all.

### Framework Comparison: ExecuTorch vs TensorFlow Lite

TensorFlow Lite is another framework and runtime you might know. Both
frameworks support edge deployment, but they fit different training and
deployment workflows.

| Feature | ExecuTorch | TensorFlow Lite |
|---|---|---|
| Training framework | PyTorch | TensorFlow/Keras |
| Model format | `.pte` | `.tflite` |
| Quantization flow | PyTorch/torchao oriented | TensorFlow tooling oriented |
| Runtime direction | PyTorch-native edge deployment | Mature mobile/embedded deployment |
| Best fit | PyTorch teams that want a direct edge path | TensorFlow teams with existing TFLite workflows |

The choice often comes down to the training framework, team expertise,
target hardware, and maturity requirements.

**VIDEO** *Course trailer-style polished video/animation on PyTorch model (30-45
seconds): show overall deployment workflow that the course follows.
Visually shows the journey from PyTorch model → export/lowering → .pte
artifact → lightweight ExecuTorch runtime → Arm CPU/NPU device.*

## How to use this resource - including via AI Agents

There are two ways to work through this course. The first is the standard route. The second uses an AI coding agent to help inspect the repository and provide additional explanation.

### Standard route

Complete Modules 0-4 in order. Follow the commands, run the scripts, read the
explanations, and consider the deployment decisions and tradeoffs presented in
each task.

Do not treat the scripts as black boxes. Open the main preparation, export,
quantization, lowering, and runtime scripts and identify the ExecuTorch APIs
used at each stage of the ahead-of-time flow. For the example applications,
inspect the Python, Java, and C++ code to see how each application loads a
`.pte` program, creates its inputs, invokes the runtime, and turns model output
into application behavior.

### AI-assisted route

The repository is structured so that an AI coding agent can trace commands
from the module text into the scripts and application code. Use an agent to
probe the implementation, ask follow-up questions, compare deployment choices,
and test your understanding. The agent should supplement the practical work,
not replace running the steps and reading the code yourself.

Two portable Agent Skills are provided under `.agents/skills`. Each skill uses
the `SKILL.md` format: plain Markdown instructions with a small YAML header.
Agents that support Agent Skills can load these files directly. The repository
also includes `AGENTS.md` so that agents which read repository guidance can
discover the same modes.

Both skills use `.agents/course-context.md`, which maps the important scripts,
identifies the ExecuTorch APIs, and explains the repeated course structure. It
directs the agent to focus on ahead-of-time preparation and model integration
rather than generic command-line, web-server, or UI code.

The two modes are:

- **`$explain-mode`** turns the agent into an interactive ExecuTorch tutor. It
  reads the relevant module and code, traces the real APIs, explains command
  outputs and artifacts, and asks short questions to check your understanding.
  This is the recommended AI-assisted route.
- **`$build-mode`** helps adapt the course patterns into a new model pipeline
  or application. It is useful when you want to use the repository as a
  starting point for a prototype. It checks current official ExecuTorch,
  PyTorch, platform, and backend documentation before carrying course APIs,
  dependencies, or workarounds into the new application.

Skill discovery and invocation vary between agent clients. If your agent
supports named skill invocation, prompts can use `$explain-mode` or
`$build-mode`. Otherwise, point the agent directly at the relevant file:

```text
Read .agents/skills/explain-mode/SKILL.md and follow it while teaching me
Module 2.
```

For example, with named skill invocation:

```text
Use $explain-mode to walk me through Module 2's INT8 flow. Show me the relevant
code, explain which calls are PyTorch and which are ExecuTorch, and ask me
questions as we go.
```

```text
Use $build-mode to adapt the Module 3 Android application for my own image
model. Preserve the FP32 validation step and explain each ExecuTorch API used.
```

Build mode can produce a working application quickly, but code generation by
itself does not provide the same understanding as tracing the implementation.
Spend time with `$explain-mode` before or after a build so that you understand
the model contract, ahead-of-time pipeline, backend choice, runtime calls, and
application integration produced by the agent.

Remember that AI Agents can, and do, sometimes make mistakes. Verify everything yourself - ensure you understand, and push back if something seems incorrect.

### Recommendation

Ultimately, if you have access to an Agent, the recommendation is to combine all approaches. Take the course as a normal user, but after reading and interpreting the scripts yourself, use your agent in explain mode as your custom 1-2-1 personal ExecuTorch tutor. Interrogate your agent, going as deep as you feel is necessary to ensure you understand. When you are ready, use build mode to start prototyping your own applications. Remember to always refer to the official documentation for the latest best practice - build mode will encourage your agent to lean upon the documentation.
