#!/usr/bin/env bash
set -euo pipefail

SESSION_DIR="${1:?usage: $0 SESSION_DIR}"
ENV_DIR="${2:-${SESSION_DIR}/env-v0517-cu129}"
PIP_LOG="${SESSION_DIR}/pip-check.txt"
VERSIONS_JSON="${SESSION_DIR}/versions.json"

if [[ -e "${ENV_DIR}" ]]; then
  echo "refusing to overwrite existing environment: ${ENV_DIR}" >&2
  exit 90
fi

python3 -m venv --copies "${ENV_DIR}"
PYTHON="${ENV_DIR}/bin/python"
PIP="${ENV_DIR}/bin/pip"

"${PIP}" install --upgrade pip setuptools wheel

# SGLang's official v0.5.17 CUDA-12 path: install the release first, then
# replace PyTorch and SGLang compiled components with their CUDA-12.9 wheels.
"${PIP}" install "sglang==0.5.17"
"${PIP}" install --force-reinstall \
  "torch==2.11.0" "torchvision==0.26.0" "torchaudio==2.11.0" \
  --index-url https://download.pytorch.org/whl/cu129
"${PIP}" install --force-reinstall "sglang-kernel==0.4.5" \
  --index-url https://docs.sglang.ai/whl/cu129/
"${PIP}" install --force-reinstall --no-deps "sgl-deep-gemm==0.1.5.post1" \
  --index-url https://docs.sglang.ai/whl/cu129/

set +e
"${PIP}" check >"${PIP_LOG}" 2>&1
PIP_CHECK_STATUS=$?
set -e

"${PYTHON}" - <<'PY' >"${VERSIONS_JSON}"
import importlib.metadata
import json
import sys

import sglang
import torch
import torchaudio
import torchvision

packages = {}
for name in (
    "sglang",
    "sglang-kernel",
    "sgl-deep-gemm",
    "torch",
    "torchvision",
    "torchaudio",
    "cuda-python",
    "cuda-bindings",
    "flashinfer-python",
    "humming-kernels",
):
    try:
        packages[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        packages[name] = None

record = {
    "python": sys.version,
    "packages": packages,
    "torch_cuda": torch.version.cuda,
    "torch_cuda_available": torch.cuda.is_available(),
    "torch_device_count": torch.cuda.device_count(),
}
print(json.dumps(record, sort_keys=True, indent=2))

assert packages["sglang"] == "0.5.17"
assert packages["torch"] == "2.11.0+cu129"
assert packages["torchvision"] == "0.26.0+cu129"
assert packages["torchaudio"] == "2.11.0+cu129"
assert packages["sglang-kernel"] == "0.4.5+cu129"
assert packages["sgl-deep-gemm"] == "0.1.5.post1+cu129"
assert torch.version.cuda == "12.9"
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 8
PY

"${PYTHON}" -m sglang.launch_server --help >/dev/null
printf '%s\n' "${PIP_CHECK_STATUS}" >"${SESSION_DIR}/pip-check.exit"
touch "${SESSION_DIR}/INSTALL_COMPLETE"
