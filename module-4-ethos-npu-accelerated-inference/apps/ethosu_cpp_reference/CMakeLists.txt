cmake_minimum_required(VERSION 3.20)
project(ethosu_cpp_reference LANGUAGES C CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(NOT CMAKE_SYSTEM_PROCESSOR STREQUAL "cortex-m85")
  message(
    FATAL_ERROR
    "Corstone-320 requires TARGET_CPU=cortex-m85; configure with -DTARGET_CPU=cortex-m85"
  )
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ET_DIR_PATH "$ENV{HOME}/executorch" CACHE PATH "Path to the ExecuTorch source tree")
set(
  ET_BUILD_DIR_PATH
  "${CMAKE_CURRENT_SOURCE_DIR}/../../artifacts/fvp_runner/mobilenetv2_int8"
  CACHE PATH
  "Path to the Module 4 runner build for this PTE"
)
set(ET_PTE_FILE_PATH "" CACHE FILEPATH "Path to the lowered ExecuTorch program")
set(ET_INPUT_FILE_PATH "" CACHE FILEPATH "Path to the prepared input tensor bytes")
set(ET_LABELS_FILE_PATH "" CACHE FILEPATH "Path to the JSON label list")
set(
  ETHOS_SDK_PATH
  "${ET_DIR_PATH}/examples/arm/arm-scratch/ethos-u"
  CACHE PATH
  "Path to the Ethos-U core platform and driver sources"
)

foreach(required_path ET_DIR_PATH ET_BUILD_DIR_PATH ET_PTE_FILE_PATH ET_INPUT_FILE_PATH ET_LABELS_FILE_PATH)
  if(NOT EXISTS "${${required_path}}")
    message(FATAL_ERROR "${required_path} does not exist: ${${required_path}}")
  endif()
endforeach()

find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(ET_BUILD_LIB_DIR "${ET_BUILD_DIR_PATH}/executorch")
set(ET_RUNNER_LIB_DIR "${ET_BUILD_DIR_PATH}/examples/arm/executor_runner")
set(
  ET_REQUIRED_LIBRARIES
  "${ET_BUILD_LIB_DIR}/libexecutorch_core.a"
  "${ET_BUILD_LIB_DIR}/backends/arm/libexecutorch_delegate_ethos_u.a"
  "${ET_BUILD_LIB_DIR}/backends/cortex_m/libcortex_m_kernels.a"
  "${ET_BUILD_LIB_DIR}/backends/cortex_m/cmsis_nn-build/libcmsis-nn.a"
  "${ET_BUILD_LIB_DIR}/kernels/portable/libportable_kernels.a"
  "${ET_BUILD_LIB_DIR}/kernels/portable/cpu/util/libkernels_util_all_deps.a"
  "${ET_BUILD_LIB_DIR}/kernels/quantized/libquantized_kernels.a"
  "${ET_RUNNER_LIB_DIR}/libarm_portable_ops_lib.a"
  "${ET_RUNNER_LIB_DIR}/libarm_quantized_ops_lib.a"
  "${ET_RUNNER_LIB_DIR}/libarm_cortex_m_ops_lib.a"
)
foreach(required_library IN LISTS ET_REQUIRED_LIBRARIES)
  if(NOT EXISTS "${required_library}")
    message(FATAL_ERROR "Required Module 4 library does not exist: ${required_library}")
  endif()
endforeach()

include(${ET_DIR_PATH}/backends/arm/scripts/corstone_utils.cmake)
fetch_ethos_u_content(${ETHOS_SDK_PATH} ${ET_DIR_PATH})
add_subdirectory(${ETHOS_SDK_PATH}/core_platform/targets/corstone-320 target)

set(GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${GENERATED_DIR}")

add_custom_command(
  OUTPUT "${GENERATED_DIR}/model_pte.h"
  COMMAND
    ${Python3_EXECUTABLE}
    ${ET_DIR_PATH}/examples/arm/common/pte_to_header.py
    --pte ${ET_PTE_FILE_PATH}
    --outdir ${GENERATED_DIR}
  DEPENDS ${ET_PTE_FILE_PATH}
  VERBATIM
)

add_custom_command(
  OUTPUT "${GENERATED_DIR}/input_data.h"
  COMMAND
    ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/binary_to_header.py
    --input ${ET_INPUT_FILE_PATH}
    --output ${GENERATED_DIR}/input_data.h
    --symbol input_data
  DEPENDS ${ET_INPUT_FILE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/binary_to_header.py
  VERBATIM
)

add_custom_command(
  OUTPUT "${GENERATED_DIR}/imagenet_labels.h"
  COMMAND
    ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/generate_labels_header.py
    --labels ${ET_LABELS_FILE_PATH}
    --output ${GENERATED_DIR}/imagenet_labels.h
  DEPENDS ${ET_LABELS_FILE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/generate_labels_header.py
  VERBATIM
)

add_custom_target(
  ethosu_cpp_reference_generated
  DEPENDS
    "${GENERATED_DIR}/model_pte.h"
    "${GENERATED_DIR}/input_data.h"
    "${GENERATED_DIR}/imagenet_labels.h"
)

add_executable(
  ethosu_cpp_reference
  app_main.cpp
  classification_postprocess.cpp
  model_runner.cpp
)
add_dependencies(ethosu_cpp_reference ethosu_cpp_reference_generated)

target_include_directories(
  ethosu_cpp_reference
  PRIVATE
    ${ET_DIR_PATH}/..
    ${ET_DIR_PATH}/runtime/core/portable_type/c10
    ${ET_BUILD_LIB_DIR}/include
    ${GENERATED_DIR}
)
target_compile_definitions(
  ethosu_cpp_reference
  PRIVATE
    C10_USING_CUSTOM_GENERATED_MACROS
)
target_link_libraries(
  ethosu_cpp_reference
  PUBLIC
    ethosu_target_init
    "-Wl,--start-group"
    "-Wl,--whole-archive"
    ${ET_BUILD_LIB_DIR}/backends/arm/libexecutorch_delegate_ethos_u.a
    ${ET_RUNNER_LIB_DIR}/libarm_portable_ops_lib.a
    ${ET_RUNNER_LIB_DIR}/libarm_quantized_ops_lib.a
    ${ET_RUNNER_LIB_DIR}/libarm_cortex_m_ops_lib.a
    ${ET_BUILD_LIB_DIR}/kernels/quantized/libquantized_kernels.a
    ${ET_BUILD_LIB_DIR}/backends/cortex_m/libcortex_m_kernels.a
    ${ET_BUILD_LIB_DIR}/kernels/portable/libportable_kernels.a
    ${ET_BUILD_LIB_DIR}/libexecutorch_core.a
    ${ET_BUILD_LIB_DIR}/backends/cortex_m/cmsis_nn-build/libcmsis-nn.a
    "-Wl,--no-whole-archive"
    ${ET_BUILD_LIB_DIR}/kernels/portable/cpu/util/libkernels_util_all_deps.a
    "-Wl,--end-group"
)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set(LINK_FILE_EXT ld)
  set(COMPILER_PREPROCESSOR_OPTIONS -E -x c -P)
else()
  message(FATAL_ERROR "This reference CMake project currently expects arm-none-eabi-gcc")
endif()

set(LINK_FILE_OUT "${CMAKE_CURRENT_BINARY_DIR}/Corstone-320.${LINK_FILE_EXT}")
execute_process(
  COMMAND
    ${CMAKE_C_COMPILER}
    ${COMPILER_PREPROCESSOR_OPTIONS}
    -DETHOSU_ARENA=1
    -o ${LINK_FILE_OUT}
    ${ET_DIR_PATH}/backends/arm/cmake/linker_scripts/Corstone-320.ld
  COMMAND_ERROR_IS_FATAL ANY
)
target_link_options(
  ethosu_cpp_reference
  PRIVATE
    "-T"
    "${LINK_FILE_OUT}"
    "LINKER:-Map=ethosu_cpp_reference.map"
)
