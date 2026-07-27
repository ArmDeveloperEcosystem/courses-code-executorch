#pragma once

#include <cstddef>

#include <executorch/runtime/core/exec_aten/exec_aten.h>

namespace app {

void print_classification_top_k(
    const executorch::aten::Tensor& output,
    const char* const* labels,
    std::size_t label_count,
    std::size_t top_k);

} // namespace app
