title: Optimize and deploy AI inference with ExecuTorch on Arm

### **About this course**

Skill level: Introductory

Course length: 8-10 hours

Author: Matt Cossins, Marcelo Rovai

Arm IP: Ethos U

Tags: Linux, KleidiAI, PyTorch, Edge AI, XNNPACK, TOSA, ExecuTorch,
Model Explorer

Last updated: June 2026

License: <https://creativecommons.org/licenses/by-sa/4.0/>

### **Who is this for?**

This is a course for developers who want to learn how to deploy ExecuTorch on Arm CPUs and NPUs for effective and lightweight inference at the edge.

### **What will you learn?**

Upon completion of this course, you will be able to:

- deploy transformer-based and convolutional neural network models using
  ExecuTorch

- compare PyTorch and ExecuTorch performance at the edge

- understand how backend delegate choices affect inference latency

- explore quantization techniques for reducing model size (and
  consequently computation) at deployment

- simulate deployment to embedded hardware using Fixed Virtual Platforms
  (FVPs)

- analyze models using tools such as Google's Model Explorer with Arm
  adapters

### **Prerequisites**

Before starting, you will ideally need the following:

- a Raspberry Pi 4 or 5 (for Modules 1 and 2)

- An Android smartphone (for Module 3)

- access to a cloud-based Arm/x86 instance (for Module 4)

- basic experience with Python and a working understanding of AI and
  machine learning concepts

If you do not have access to all the above hardware, you may not be able to run every example. However you will still be able to see the scripts and run `ahead-of-time` model preparation. You should therefore still receive a lot of value from the course.

### **Course modules**

Module 0: Introduction and Setup

Introduces ExecuTorch, edge AI, the course workflow, hardware
requirements, and environment setup across CPU and NPU labs.

**Estimated time to complete: 45-60 minutes**

Module 1: Compare Transformer Inference on Arm CPUs

Compare PyTorch eager, ExecuTorch portable, and ExecuTorch with XNNPACK
for transformer inference, using benchmark results to understand
portability, delegation, and backend optimization trade-offs.

**Estimated time to complete: 75-100 minutes**

Module 2: Optimize Image Classification on Arm CPUs

Deploy MobileNetV2 for image classification on Arm CPUs, then
investigate XNNPACK delegation, INT8 quantization, latency reduction,
and task-level behavior after optimization.

**Estimated time to complete: 75-100 minutes**

Module 3: Deploy ExecuTorch on Mobile Devices

Deploy ExecuTorch models on mobile devices, using an Android smartphone. Understand what changes between Edge IoT / Embedded AI, and Mobile AI.

**Estimated time to complete: 45-60 minutes**

Module 4: Evaluate ExecuTorch Models for Arm Ethos-U Acceleration

Move from CPU inference to Ethos-U NPU acceleration, exploring
quantization, Vela compilation, TOSA artifacts, Model Explorer,
delegation coverage, fallback execution, and FVP simulation.

**Estimated time to complete: 100-125 minutes**

Next Steps

Points learners toward further ExecuTorch, quantization, Ethos-U, and
community resources for extending the practical workflows.

**Estimated time to complete: 15-20 minutes**

# **Module 0: How do I move a PyTorch model from development to efficient edge deployment?**

## **Introduction**

### **About this module**

Welcome to Optimize and Deploy AI Inference with ExecuTorch on Arm. In
these hands-on sessions, you will explore how PyTorch models can be
exported and lowered to the ExecuTorch runtime for efficient,
lightweight inference at the edge.

You will work across multiple deployment targets, including: 

- CPU-based inference using XNNPACK and Arm KleidiAI

- deployment to Arm Ethos-U Neural Processing Units (NPUs)

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

ExecuTorch is part of the PyTorch workflow and is described as
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

## Framework Comparison: ExecuTorch vs TensorFlow Lite

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
