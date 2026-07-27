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

Complete the Module 4 AOT and input-preparation tasks first. You need:

- `mobilenetv2_int8_ethosu.pte`
- `i0.bin`
- `imagenet_labels.json`
- an installed bare-metal ExecuTorch build under `~/executorch/cmake-out-arm`

Configure the application from the course repository root:

```bash
cmake \
  -S module-4-ethos-npu-accelerated-inference/apps/ethosu_cpp_reference \
  -B module-4-ethos-npu-accelerated-inference/artifacts/fvp_runner/ethosu_cpp_reference \
  -DCMAKE_TOOLCHAIN_FILE="$HOME/executorch/examples/arm/ethos-u-setup/arm-none-eabi-gcc.cmake" \
  -DET_DIR_PATH="$HOME/executorch" \
  -DET_PTE_FILE_PATH="$PWD/module-4-ethos-npu-accelerated-inference/artifacts/mobilenetv2/mobilenetv2_int8_ethosu.pte" \
  -DET_INPUT_FILE_PATH="$PWD/module-4-ethos-npu-accelerated-inference/artifacts/fvp_runs/mobilenetv2_int8/i0.bin" \
  -DET_LABELS_FILE_PATH="$PWD/module-4-ethos-npu-accelerated-inference/artifacts/mobilenetv2/imagenet_labels.json"
```

Build the ELF:

```bash
cmake --build \
  module-4-ethos-npu-accelerated-inference/artifacts/fvp_runner/ethosu_cpp_reference \
  --target ethosu_cpp_reference -j2
```

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

- [Run an ExecuTorch model using the C++ runtime](https://docs.pytorch.org/executorch/stable/runtime-how-to.html)
- [Arm Ethos-U backend overview](https://docs.pytorch.org/executorch/stable/backends/arm-ethos-u/arm-ethos-u-overview.html)
