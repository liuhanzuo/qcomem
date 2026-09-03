#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 NEW_OUTPUT_DIRECTORY" >&2
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
package_root="$(cd -- "${script_dir}/.." && pwd)"
output_path="$1"

if [[ -e "${output_path}" ]]; then
  echo "refusing to overwrite existing output: ${output_path}" >&2
  exit 3
fi

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${package_root}"
exec python3 "${package_root}/scripts/run_formal.py" \
  --root "${package_root}" \
  --output "${output_path}"

