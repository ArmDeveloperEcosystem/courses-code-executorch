![Laptop, smartphone, smartwatch, controllers, and other edge devices](docs/images/course-banner.jpg)

# Optimize and Deploy AI Inference with ExecuTorch on Arm

[![Repository integrity](https://github.com/arm-education/executorch_on_arm_course/actions/workflows/repo-integrity.yml/badge.svg?branch=main)](https://github.com/arm-education/executorch_on_arm_course/actions/workflows/repo-integrity.yml)
[![Module 3 Android](https://github.com/arm-education/executorch_on_arm_course/actions/workflows/module-3-android.yml/badge.svg?branch=main)](https://github.com/arm-education/executorch_on_arm_course/actions/workflows/module-3-android.yml)

[Course content](#course-content) · [Course Overview](#course-overview) · [Who This Is For](#who-this-is-for) · [Hardware and Environment](#hardware-and-environment) · [Course Modules](#course-modules)

This repository accompanies the hands-on course for deploying PyTorch models to Arm CPUs and NPUs with
ExecuTorch available at [URL TO FOLLOW].
The course walks through ahead-of-time model preparation,
backend delegation and graph inspection, post-training quantization, and runtime benchmarking and example applications across a range of hardware targets.

You will start with CPU inference on Raspberry Pi using XNNPACK, covering examples including both transformer and CNN-based models. You will then explore Mobile deployment, and understand what changes when moving from Cortex-A on Edge Linux, to a Mobile OS like Android. Finally, you will get hands-on with Cortex-M Microcontroller + Ethos-U NPU acceleration. Each Module should extend your knowledge, and provide practical examples that allow you to quickly begin trying the workflow. Each Module links to supplementary resources that act as useful next steps, either to reinforce what you have done, or to extend to more advanced topics.

This course is best used in parallel with the ExecuTorch Documentation.

## Course content

This repository contains the necessary scripts and information to accompany the course.

The learner-facing Modules 1–4 are published separately at [URL TO FOLLOW].
This repository contains the supporting scripts, applications, images, and
configuration used by those modules.

## Course Overview

**Skill level:** Introductory

**Course length:** 8-10 hours

**Author:** Matt Cossins, with contributions from Marcelo Rovai

**Main technologies:** PyTorch, ExecuTorch, XNNPACK, KleidiAI, Model Explorer, Cortex-A, Cortex-M, Ethos-U, TOSA,
Vela, Corstone-320, Transformers, LLMs, CNNs

**License:** [BSD-3-Clause-Clear](https://spdx.org/licenses/BSD-3-Clause-Clear.html)

## Who This Is For

This course is for developers who want to learn how to deploy PyTorch
models efficiently on Arm-based edge devices, using the ExecuTorch Runtime.

You should be comfortable with basic Python and familiar with core
machine learning concepts such as models, tensors, inference, and
quantization. You do not need prior ExecuTorch experience - this is ideal for beginners with ExecuTorch.

You are encouraged to use AI Agents when taking this course, and skills are provided to aid your understanding, and next steps. See Module 0 for more information.

## Hardware and Environment

The full course uses:

- macOS or Ubuntu, including Ubuntu under WSL, for development-machine steps
- Raspberry Pi 5 running 64-bit Raspberry Pi OS Trixie for Modules 1 and 2
- Android smartphone or Emulation via Android Studio for Module 3
- Ubuntu LTS host or cloud instance for Module 4

If you do not have all target devices, you can still follow the
ahead-of-time preparation scripts and either read the content without running the benchmarks on hardware, or try running them on your local development machine instead.

## Course Modules

### Module 0: Course and ExecuTorch Overview

Introduces ExecuTorch, the ahead-of-time deployment workflow, the course
hardware targets, and the optional AI-agent learning modes.

### Module 1: Get started with ExecuTorch and Understand Delegation - Transformer Inference on Arm Cortex-A CPU

In this Module, we introduce ExecuTorch, explaining its role in the PyTorch Ecosystem, and in Edge AI. You will install ExecuTorch to your development machine in a Python environment. In an `ahead-of-time` flow, you will obtain two different Transformer-based PyTorch models from Hugging Face (GPT-2 and DistilBERT), and perform export and lowering to convert them to an ExecuTorch format. You will use both the ExecuTorch Portable Kernel Library, and also optimized delegation to XNNPACK.

After comparing your original PyTorch models, your ExecuTorch Portable models, and your optimized XNNPACK delegate ExecuTorch models - including via the use of Model Explorer - you will benchmark each on your target device: a Raspberry Pi 5.

Finally, you will run example applications for each: a simple text-generation application and an autocomplete application.

By the end, you will have completed a typical `ahead-of-time` preparation flow, and deployed your prepared models in applications on target hardware. You will be able to discuss when and why to delegate to optimized backends like XNNPACK, and how you can use tools like Model Explorer to visualize graph changes after delegation has occurred. You will be ready to move onto Module 2, where you will cover image classification, and will also explore Quantization in ExecuTorch.

### Module 2: CNN Inference on Arm Cortex-A CPU and Optimize with INT8 Quantization

In this Module, you move from transformer inference to image classification.
You will prepare a pretrained MobileNetV2 model from TorchVision, and export the model to an ExecuTorch `.pte` artifact with XNNPACK delegation. You will benchmark two runtime paths on the Raspberry Pi: PyTorch eager and
FP32 ExecuTorch with XNNPACK.

You will then add post-training quantization
(PTQ), using calibration images to create an INT8 ExecuTorch with XNNPACK
`.pte` artifact. This lets you compare FP32 and INT8 deployment tradeoffs:
artifact size, target-side latency, delegation, and top-5 classification
output. You will again run an example application, a simple image classifier web app.

An optional profiling section builds an ETDump-enabled runtime on the Raspberry
Pi, captures ETDump and ETRecord data, examines it with the ExecuTorch Inspector,
and opens a timing overlay in Model Explorer.

By the end, you will be confident quantizing models with post-training
quantization and explaining the benefits and tradeoffs of an INT8 `.pte`
artifact. You will also understand how CNN model architectures delegate
differently to XNNPACK when compared with transformer-based models, and you
will be able to compare PyTorch eager, FP32 ExecuTorch with XNNPACK, and INT8
ExecuTorch with XNNPACK on the same target device.

### Module 3: ExecuTorch on Mobile - Using the XNNPACK Backend on Android for Semantic Segmentation

In this module, you move from Edge Linux deployment on a Raspberry Pi to an
Android application. You will prepare an LR-ASPP MobileNetV3 semantic
segmentation model, lower an FP32 version with XNNPACK, and validate its object
cutout output on the development machine. You will then apply post-training
quantization, compare the FP32 and INT8 results, and select the INT8 `.pte`
artifact for the mobile application.

You will package the model, labels, and preprocessing settings as Android
assets and run the application on a physical Arm phone or an Android Studio
emulator. The Java application uses the ExecuTorch Android API to load the
model, create input tensors, run `forward()`, interpret the semantic
segmentation logits, and produce an image with a transparent background.

By the end, you will understand what changes when moving an ExecuTorch
application from Python on Edge Linux to Java or Kotlin on Android, including
application assets, native ABIs, lifecycle management, background inference,
and native runtime cleanup. You will also understand where XNNPACK CPU
execution fits alongside other mobile options such as iOS, device NPUs, SME2,
and the Arm VGF backend for neural graphics workloads.

### Module 4: Deploy ExecuTorch Models with Arm Ethos-U Acceleration

In this module, you return to the MobileNetV2 model, and explore how to optimize and accelerate its deployment via use of an Ethos-U NPU. You will explore FP32 and INT8 Post-Training Quantization routes, and understand how the Ethos-U backend differs from the XNNPACK backend. You will become familiar with the Vela compiler and the TOSA intermediate representation.

You then add an unsupported operation to split the model into two delegated
regions and compare the fallback path. After returning to the clean INT8 model,
you run it on a Corstone-320 FVP simulation of the Cortex-M + Ethos-U and inspect
the generated `.pte` graph and FVP output. The module finishes by adapting the
generic C++ runner into a MobileNetV2 classification application that prints
ImageNet labels over UART.
