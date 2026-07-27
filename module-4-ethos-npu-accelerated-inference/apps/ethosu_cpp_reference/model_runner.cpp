#include "model_runner.h"

#include <cinttypes>
#include <vector>

#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/log.h>

#include "app_config.h"

namespace app {
namespace {

#if defined(__APPLE__)
#define APP_SECTION(name)
#else
#define APP_SECTION(name) __attribute__((section(name)))
#endif

alignas(16) unsigned char APP_SECTION("input_data_sec")
    method_allocation_pool[kMethodAllocatorPoolSize];

alignas(16) unsigned char APP_SECTION(".bss.tensor_arena")
    temporary_allocation_pool[kTemporaryAllocatorPoolSize];

} // namespace

executorch::runtime::Error run_model(
    const std::uint8_t* program_data,
    std::size_t program_size,
    const std::uint8_t* input_data,
    std::size_t input_size,
    OutputHandler output_handler) {
  using executorch::aten::Tensor;
  using executorch::extension::BufferDataLoader;
  using executorch::runtime::Error;
  using executorch::runtime::EValue;
  using executorch::runtime::HierarchicalAllocator;
  using executorch::runtime::MemoryAllocator;
  using executorch::runtime::MemoryManager;
  using executorch::runtime::Method;
  using executorch::runtime::MethodMeta;
  using executorch::runtime::Program;
  using executorch::runtime::Result;
  using executorch::runtime::Span;

  BufferDataLoader loader(program_data, program_size);
  Result<Program> program = Program::load(&loader);
  if (!program.ok()) {
    ET_LOG(
        Error,
        "Program::load failed: 0x%" PRIx32,
        static_cast<std::uint32_t>(program.error()));
    return program.error();
  }

  auto method_name = program->get_method_name(0);
  if (!method_name.ok()) {
    ET_LOG(Error, "The program does not contain an executable method");
    return method_name.error();
  }

  Result<MethodMeta> method_meta = program->method_meta(*method_name);
  if (!method_meta.ok()) {
    ET_LOG(
        Error,
        "method_meta failed: 0x%" PRIx32,
        static_cast<std::uint32_t>(method_meta.error()));
    return method_meta.error();
  }

  MemoryAllocator method_allocator(
      kMethodAllocatorPoolSize, method_allocation_pool);
  MemoryAllocator temporary_allocator(
      kTemporaryAllocatorPoolSize, temporary_allocation_pool);

  std::vector<Span<std::uint8_t>> planned_spans;
  planned_spans.reserve(method_meta->num_memory_planned_buffers());
  for (std::size_t index = 0;
       index < method_meta->num_memory_planned_buffers();
       ++index) {
    auto buffer_size = method_meta->memory_planned_buffer_size(index);
    if (!buffer_size.ok()) {
      return buffer_size.error();
    }

    const std::size_t planned_size = static_cast<std::size_t>(*buffer_size);
    void* buffer = method_allocator.allocate(planned_size, 16);
    if (buffer == nullptr) {
      ET_LOG(
          Error,
          "Could not allocate planned buffer %zu (%zu bytes)",
          index,
          planned_size);
      return Error::MemoryAllocationFailed;
    }
    planned_spans.emplace_back(
        static_cast<std::uint8_t*>(buffer), planned_size);
  }

  HierarchicalAllocator planned_memory(
      {planned_spans.data(), planned_spans.size()});
  MemoryManager memory_manager(
      &method_allocator, &planned_memory, &temporary_allocator);

  Result<Method> method =
      program->load_method(*method_name, &memory_manager);
  if (!method.ok()) {
    ET_LOG(
        Error,
        "load_method failed: 0x%" PRIx32,
        static_cast<std::uint32_t>(method.error()));
    return method.error();
  }

  if (method_meta->num_inputs() != kExpectedInputCount) {
    ET_LOG(
        Error,
        "Expected %zu input, but the method has %zu",
        kExpectedInputCount,
        method_meta->num_inputs());
    return Error::InvalidProgram;
  }

  auto input_meta = method_meta->input_tensor_meta(0);
  if (!input_meta.ok()) {
    ET_LOG(Error, "Input 0 is not a tensor");
    return input_meta.error();
  }
  if (input_meta->nbytes() != input_size) {
    ET_LOG(
        Error,
        "Input byte count mismatch: method expects %zu, input has %zu",
        input_meta->nbytes(),
        input_size);
    return Error::InvalidArgument;
  }

  executorch::aten::TensorImpl input_impl(
      input_meta->scalar_type(),
      input_meta->sizes().size(),
      const_cast<executorch::aten::TensorImpl::SizesType*>(
          input_meta->sizes().data()),
      const_cast<std::uint8_t*>(input_data),
      const_cast<executorch::aten::TensorImpl::DimOrderType*>(
          input_meta->dim_order().data()));
  Tensor input_tensor(&input_impl);
  Error status = method->set_input(input_tensor, 0);
  if (status != Error::Ok) {
    ET_LOG(
        Error,
        "Method::set_input failed: 0x%" PRIx32,
        static_cast<std::uint32_t>(status));
    return status;
  }

  status = method->execute();
  if (status != Error::Ok) {
    ET_LOG(
        Error,
        "Method::execute failed: 0x%" PRIx32,
        static_cast<std::uint32_t>(status));
    return status;
  }

  std::vector<EValue> outputs(method->outputs_size());
  status = method->get_outputs(outputs.data(), outputs.size());
  if (status != Error::Ok) {
    return status;
  }

  for (std::size_t index = 0; index < outputs.size(); ++index) {
    if (outputs[index].isTensor() && output_handler != nullptr) {
      output_handler(outputs[index].toTensor(), index);
    }
  }

  return Error::Ok;
}

} // namespace app
