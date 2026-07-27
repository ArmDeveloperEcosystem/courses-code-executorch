#pragma once

#include <cstddef>

namespace app {

inline constexpr std::size_t kMethodAllocatorPoolSize = 60 * 1024 * 1024;
inline constexpr std::size_t kTemporaryAllocatorPoolSize = 2 * 1024 * 1024;
inline constexpr std::size_t kExpectedInputCount = 1;
inline constexpr std::size_t kClassificationOutputIndex = 0;
inline constexpr std::size_t kTopK = 5;

} // namespace app
