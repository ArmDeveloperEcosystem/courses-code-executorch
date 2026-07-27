#include "classification_postprocess.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#include <executorch/runtime/platform/log.h>

namespace app {
namespace {

float value_at(
    const executorch::aten::Tensor& output,
    std::size_t index,
    bool& supported) {
  using executorch::aten::ScalarType;

  supported = true;
  switch (output.scalar_type()) {
    case ScalarType::Float:
      return output.const_data_ptr<float>()[index];
    case ScalarType::Int:
      return static_cast<float>(output.const_data_ptr<std::int32_t>()[index]);
    case ScalarType::Char:
      return static_cast<float>(output.const_data_ptr<std::int8_t>()[index]);
    case ScalarType::Byte:
      return static_cast<float>(output.const_data_ptr<std::uint8_t>()[index]);
    default:
      supported = false;
      return 0.0f;
  }
}

} // namespace

void print_classification_top_k(
    const executorch::aten::Tensor& output,
    const char* const* labels,
    std::size_t label_count,
    std::size_t top_k) {
  if (top_k == 0 || output.numel() == 0) {
    return;
  }

  const std::size_t element_count =
      static_cast<std::size_t>(output.numel());
  const std::size_t result_count = std::min(top_k, element_count);
  std::vector<float> top_values(
      result_count, std::numeric_limits<float>::lowest());
  std::vector<std::size_t> top_indices(result_count, 0);

  for (std::size_t index = 0; index < element_count; ++index) {
    bool supported = false;
    const float value = value_at(output, index, supported);
    if (!supported) {
      ET_LOG(
          Error,
          "Unsupported classification output scalar type: %d",
          static_cast<int>(output.scalar_type()));
      return;
    }

    for (std::size_t rank = 0; rank < result_count; ++rank) {
      if (value <= top_values[rank]) {
        continue;
      }
      for (std::size_t move = result_count - 1; move > rank; --move) {
        top_values[move] = top_values[move - 1];
        top_indices[move] = top_indices[move - 1];
      }
      top_values[rank] = value;
      top_indices[rank] = index;
      break;
    }
  }

  ET_LOG(Info, "Top-%zu predictions:", result_count);
  for (std::size_t rank = 0; rank < result_count; ++rank) {
    const std::size_t class_index = top_indices[rank];
    const char* label =
        class_index < label_count ? labels[class_index] : "unknown";
    ET_LOG(
        Info,
        "  %zu. %s (class %zu): %.6f",
        rank + 1,
        label,
        class_index,
        static_cast<double>(top_values[rank]));
  }
}

} // namespace app
