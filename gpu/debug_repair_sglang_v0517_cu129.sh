#!/usr/bin/env bash
set -euo pipefail

SESSION_DIR="${1:?usage: $0 SESSION_DIR ENV_DIR}"
ENV_DIR="${2:?usage: $0 SESSION_DIR ENV_DIR}"
DEEP_GEMM_WHEEL="${3:-}"
PYTHON="${ENV_DIR}/bin/python"
PIP="${ENV_DIR}/bin/pip"

test -x "${PYTHON}"
"${PYTHON}" -c 'import importlib.metadata; assert importlib.metadata.version("sglang") == "0.5.17"'

# The authenticated internal mirror exposes the same pinned CUDA-12.9 wheels
# and is used here to avoid re-downloading the 1.16 GB PyTorch wheel.
if ! "${PYTHON}" - <<'PY'
import importlib.metadata
expected = {
    "torch": "2.11.0+cu129",
    "torchvision": "0.26.0+cu129",
    "torchaudio": "2.11.0+cu129",
}
assert {name: importlib.metadata.version(name) for name in expected} == expected
PY
then
  "${PIP}" install --force-reinstall \
    "torch==2.11.0+cu129" "torchvision==0.26.0+cu129" \
    "torchaudio==2.11.0+cu129"
fi
if [[ "$("${PYTHON}" -c 'import importlib.metadata as m; print(m.version("sglang-kernel"))')" != "0.4.5+cu129" ]]; then
  "${PIP}" install --force-reinstall "sglang-kernel==0.4.5" \
    --index-url https://docs.sglang.io/whl/cu129/
fi
if [[ -n "${DEEP_GEMM_WHEEL}" ]]; then
  test -f "${DEEP_GEMM_WHEEL}"
  "${PIP}" install --force-reinstall --no-deps "${DEEP_GEMM_WHEEL}"
else
  "${PIP}" install --force-reinstall --no-deps "sgl-deep-gemm==0.1.5.post1" \
    --index-url https://docs.sglang.io/whl/cu129/
fi
"${PIP}" install --force-reinstall "numpy==2.3.5" "fsspec==2026.6.0"

set +e
"${PIP}" check >"${SESSION_DIR}/pip-check.txt" 2>&1
PIP_CHECK_STATUS=$?
set -e

"${PYTHON}" - <<'PY' >"${SESSION_DIR}/versions.json"
import importlib.metadata
import json
import sys

import sglang
import torch
import torchaudio
import torchvision

packages = {}
for name in (
    "sglang", "sglang-kernel", "sgl-deep-gemm", "torch",
    "torchvision", "torchaudio", "cuda-python", "cuda-bindings",
    "flashinfer-python", "humming-kernels",
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

"${PYTHON}" -m sglang.launch_server --help >"${SESSION_DIR}/sglang-launch-help.txt"
printf '%s\n' "${PIP_CHECK_STATUS}" >"${SESSION_DIR}/pip-check.exit"
"${PIP}" freeze --all | sort >"${SESSION_DIR}/environment-freeze.txt"
touch "${SESSION_DIR}/INSTALL_COMPLETE"
