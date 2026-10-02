# Ethos-U C++ Reference Application

This is a small application scaffold for building on the Module 4
MobileNetV2 example. It keeps the Corstone-320 startup code, Ethos-U driver,
delegate, kernels, and linker script in the ExecuTorch source tree while making
the application-owned code easy to inspect and replace.

The files are split by responsibility:

- `app_main.cpp` initializes ExecuTorch and connects inference to the
  application postprocessing.
- `model_runner.cpp` loads the `.pte`, creates the runtime allocators, loads the
  first method, constructs the input tensor from `MethodMeta`, calls
  `Method::set_input()` and `Method::execute()`, and reads the outputs.
- `classification_postprocess.cpp` converts the output tensor into top-k class
  IDs, labels, and logits.
- `app_config.h` contains the model-specific input count, output index, top-k
  value, and allocator sizes.
- `CMakeLists.txt` links the application against the installed ExecuTorch
  runtime, Ethos-U delegate, Arm kernels, and Corstone-320 platform support.

The scaffold embeds the `.pte`, prepared input tensor, and labels in the ELF.
This removes semihosting file access from the application and makes the same
structure easier to adapt for a physical board.

## Build

Complete the Module 4 AOT, input-preparation, and **Build the runner** steps
first. You do not need to run that runner on the FVP before
building this application. The earlier build step is required because this
application reuses the target-specific ExecuTorch libraries it produces.

You need:

- `mobilenetv2_int8_ethosu.pte`
- `i0.bin`
- `imagenet_labels.json`
- the Module 4 runner build under
  `module-4-ethos-npu-accelerated-inference/artifacts/fvp_runner/mobilenetv2_int8`

The runner build directory contains the ExecuTorch runtime, Ethos-U delegate,
kernel, and CMSIS libraries compiled for the same U85 target. Reusing them
avoids compiling a second copy and keeps the
application ABI consistent with the runner build.

In each new shell, activate the ExecuTorch environment and add the downloaded
Arm toolchain to `PATH`:

```bash
cd ~/executorch
source .venv/bin/activate
source examples/arm/arm-scratch/setup_path.sh
command -v arm-none-eabi-gcc
cd ~/executorch_on_arm
```

The `command -v` check must print the full path to `arm-none-eabi-gcc`. If it
prints nothing, rerun this command before continuing:

```bash
~/executorch/examples/arm/setup.sh --i-agree-to-the-contained-eula
```

Configure the application from the course repository root:

```bash
cmake --fresh \
  -S module-4-ethos-npu-accelerated-inference/apps/ethosu_cpp_reference \
  -B module-4-ethos-npu-accelerated-inference/artifacts/fvp_runner/ethosu_cpp_reference \
  -DCMAKE_BUILD_TYPE=Release \
  -DTARGET_CPU=cortex-m85 \
  -DCMAKE_TOOLCHAIN_FILE="$HOME/executorch/examples/arm/ethos-u-setup/arm-none-eabi-gcc.cmake" \
  -DET_DIR_PATH="$HOME/executorch" \
  -DET_PTE_FILE_PATH="$PWD/module-4-ethos-npu-accelerated-inference/artifacts/mobilenetv2/mobilenetv2_int8_ethosu.pte" \
  -DET_INPUT_FILE_PATH="$PWD/module-4-ethos-npu-accelerated-inference/artifacts/fvp_runs/mobilenetv2_int8/i0.bin" \
  -DET_LABELS_FILE_PATH="$PWD/module-4-ethos-npu-accelerated-inference/artifacts/mobilenetv2/imagenet_labels.json"
```

`TARGET_CPU=cortex-m85` selects the Corstone-320 startup files and CMSIS device
definitions.

Build the ELF:

```bash
cmake --build \
  module-4-ethos-npu-accelerated-inference/artifacts/fvp_runner/ethosu_cpp_reference \
  --target ethosu_cpp_reference -j2
```

The Arm linker may report `_gettimeofday`, RWX segment, `.ddr`, or `.sram`
warnings while producing the bare-metal ELF. The upstream Module 4 runner uses
the same platform linker script and reports the same warnings. Continue if the
build finishes with `Built target ethosu_cpp_reference`.

Run it on the Corstone-320 FVP:

```bash
FVP_Corstone_SSE-320 \
  -C mps4_board.subsystem.ethosu.num_macs=256 \
  -C mps4_board.visualisation.disable-visualisation=1 \
  -C vis_hdlcd.disable_visualisation=1 \
  -C mps4_board.telnetterminal0.start_telnet=0 \
  -C mps4_board.uart0.out_file=- \
  -C mps4_board.uart0.shutdown_on_eot=1 \
  -C mps4_board.subsystem.ethosu.extra_args=--fast \
  -a module-4-ethos-npu-accelerated-inference/artifacts/fvp_runner/ethosu_cpp_reference/ethosu_cpp_reference
```

For the Labrador sample image, the output should end with a result similar to:

```text
Top-5 predictions:
  1. Labrador retriever (class 208): 14.136751
  2. golden retriever (class 207): 12.889391
  3. tennis ball (class 852): 8.211789
Application inference completed
Application exit code: 0.
```

The exact logits can change with model or toolchain versions. The top labels
show that the embedded input reached the model, while exit code `0` confirms
that the `.pte` loaded, delegated execution completed, and outputs were read.

## Adapt the Scaffold

For another model or application:

1. Change the model and input paths passed to CMake.
2. Update the input count, output index, allocator sizes, and top-k value in
   `app_config.h`.
3. Replace `classification_postprocess.cpp` with postprocessing for the new
   task.
4. Update `handle_output()` in `app_main.cpp` to pass outputs into the new
   application code.
5. Keep the input preprocessing identical to the preprocessing used when the
   model was exported and validated.

The allocator sizes are intentionally visible. Reduce them for a physical
target after checking the memory plan and the target memory map.

## Further Reading

- [Run an ExecuTorch model using the C++ runtime](https://docs.pytorch.org/executorch/stable/running-a-model-cpp-tutorial.html)
- [Arm Ethos-U backend overview](https://docs.pytorch.org/executorch/stable/backends/arm-ethos-u/arm-ethos-u-overview.html)
