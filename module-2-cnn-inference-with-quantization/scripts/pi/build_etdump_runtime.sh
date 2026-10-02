#!/usr/bin/env bash
set -euo pipefail

EXECUTORCH_VERSION="1.3.1"
EXECUTORCH_TAG="v${EXECUTORCH_VERSION}"
SOURCE_DIR="${EXECUTORCH_ETDUMP_SOURCE_DIR:-${HOME}/executorch-etdump/executorch}"

if [[ "$(basename "${SOURCE_DIR}")" != "executorch" ]]; then
  echo "ExecuTorch must be checked out to a directory named executorch: ${SOURCE_DIR}" >&2
  exit 1
fi

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "Activate ~/executorch_on_arm/.cpu_lab_venv before running this script." >&2
  exit 1
fi

python - <<'PY'
from importlib.metadata import version
from packaging.version import Version

expected = "1.3.1"
installed = version("executorch")
if Version(installed).base_version != expected:
    raise SystemExit(f"Expected executorch {expected}, found {installed}")
PY

if [[ ! -d "${SOURCE_DIR}/.git" ]]; then
  mkdir -p "$(dirname "${SOURCE_DIR}")"
  git clone \
    --branch "${EXECUTORCH_TAG}" \
    --depth 1 \
    --recurse-submodules \
    --shallow-submodules \
    https://github.com/pytorch/executorch.git \
    "${SOURCE_DIR}"
else
  EXPECTED_COMMIT="$(git -C "${SOURCE_DIR}" rev-list -n 1 "${EXECUTORCH_TAG}")"
  CURRENT_COMMIT="$(git -C "${SOURCE_DIR}" rev-parse HEAD)"
  if [[ "${CURRENT_COMMIT}" != "${EXPECTED_COMMIT}" ]]; then
    echo "${SOURCE_DIR} is not at ${EXECUTORCH_TAG}; move it or set EXECUTORCH_ETDUMP_SOURCE_DIR." >&2
    exit 1
  fi
fi

git -C "${SOURCE_DIR}" submodule update --init --recursive --depth 1

# The default Linux AArch64 index can select CUDA-enabled PyTorch wheels. The
# ExecuTorch source build needs the CPU CMake package on a Raspberry Pi.
python -m pip install \
  --force-reinstall \
  --no-deps \
  --index-url https://download.pytorch.org/whl/cpu \
  torch==2.12.1+cpu \
  torchvision==0.27.1+cpu

export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-2}"
export CMAKE_ARGS="-DEXECUTORCH_BUILD_XNNPACK=ON -DEXECUTORCH_BUILD_DEVTOOLS=ON -DEXECUTORCH_ENABLE_EVENT_TRACER=ON -DEXECUTORCH_BUILD_OPENVINO=OFF -DEXECUTORCH_BUILD_COREML=OFF -DEXECUTORCH_BUILD_EXTENSION_TRAINING=OFF -DEXECUTORCH_BUILD_EXTENSION_LLM_RUNNER=OFF -DEXECUTORCH_BUILD_EXTENSION_LLM=OFF -DEXECUTORCH_BUILD_KERNELS_LLM=OFF -DEXECUTORCH_BUILD_KERNELS_LLM_AOT=OFF"

python -m pip install \
  --force-reinstall \
  --no-build-isolation \
  --no-deps \
  "${SOURCE_DIR}"

python - <<'PY'
from executorch.extension.pybindings.portable_lib import _load_for_executorch

if "enable_etdump" not in (_load_for_executorch.__doc__ or ""):
    raise SystemExit("The rebuilt ExecuTorch binding does not expose ETDump capture.")
print("Built ExecuTorch with XNNPACK and ETDump support.")
PY
