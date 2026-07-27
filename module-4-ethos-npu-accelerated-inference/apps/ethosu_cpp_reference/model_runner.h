#pragma once

#include <cstddef>
#include <cstdint>

#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>

namespace app {

using OutputHandler = void (*)(
    const executorch::aten::Tensor& output,
    std::size_t output_index);

executorch::runtime::Error run_model(
    const std::uint8_t* program_data,
    std::size_t program_size,
    const std::uint8_t* input_data,
    std::size_t input_size,
    OutputHandler output_handler);

} // namespace app
