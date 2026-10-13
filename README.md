# Course code for 'Optimize and deploy AI inference with ExecuTorch on Arm'

This repository contains the supporting scripts and example applications for
[Optimize and deploy AI inference with ExecuTorch on Arm](https://learn.arm.com/courses/executorch-on-arm/).
Use the course for prerequisites, setup, and exercise instructions. The links below
take you to the code for the module you are following.

## Find your module

| Module | Code |
| --- | --- |
| 1 | [Transformer inference on Raspberry Pi](module-1-transformer-inference-with-xnnpack/) |
| 2 | [Image classification and INT8 quantization on Raspberry Pi](module-2-cnn-inference-with-quantization/) |
| 3 | [Semantic segmentation on Android](module-3-executorch-on-android-semantic-segmentation/) |
| 4 | [Ethos-U inference with Corstone-320](module-4-ethos-npu-accelerated-inference/) |

## Using this repository

- Follow your lesson's environment, working-directory, and command instructions.
  Requirements differ between modules, so use the versions and installation steps
  specified for that exercise.
- Clone and keep the complete repository together. Module scripts import shared
  helpers from [course_support](course_support/). These helpers support tasks such
  as calibration and benchmarking across the course modules. Copying a single
  module directory can leave those imports unavailable.
- Open the script or application named in the lesson to connect each command with
  its implementation. Continue using the course to interpret outputs and decide
  the next step.

Optional [AI-agent skills](.agents/skills/) support working with the code; Module 0
of the course explains how to use them.

For proposed fixes or improvements, see [Contributing](CONTRIBUTING.md).
Code is provided under the [Clear BSD License](LICENSE).
