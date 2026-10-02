#include <cstddef>
#include <cstdint>

#include <executorch/runtime/core/error.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/runtime.h>

#include "app_config.h"
#include "classification_postprocess.h"
#include "imagenet_labels.h"
#include "input_data.h"
#include "model_pte.h"
#include "model_runner.h"

namespace {

void handle_output(
    const executorch::aten::Tensor& output,
    std::size_t output_index) {
  if (output_index != app::kClassificationOutputIndex) {
    return;
  }
  app::print_classification_top_k(
      output, kImageNetLabels, kImageNetLabelCount, app::kTopK);
}

} // namespace

int main() {
  // ExecuTorch C++ API: initialize platform services before loading a program.
  executorch::runtime::runtime_init();
  ET_LOG(Info, "ExecuTorch runtime initialized");

  // The runner owns the Program and Method lifecycle; this file supplies the
  // embedded .pte, input bytes, and application postprocessing callback.
  const executorch::runtime::Error status = app::run_model(
      model_pte,
      sizeof(model_pte),
      input_data,
      sizeof(input_data),
      handle_output);
  if (status != executorch::runtime::Error::Ok) {
    ET_LOG(Error, "Application inference failed");
    return 1;
  }

  ET_LOG(Info, "Application inference completed");
  return 0;
}
