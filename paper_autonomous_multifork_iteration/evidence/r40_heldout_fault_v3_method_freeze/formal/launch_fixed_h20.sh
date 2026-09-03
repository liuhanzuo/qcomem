#!/usr/bin/env bash
set -euo pipefail

if [[ "${R40_V3_H20_AUTHORIZED:-}" != "yes" ]]; then
  echo "HOLD: R40_V3_H20_AUTHORIZED=yes is required after independent audit" >&2
  exit 64
fi

if [[ "$#" -ne 0 ]]; then
  echo "usage: $0  (no caller config or output-root arguments)" >&2
  exit 64
fi

script_dir="$(cd "$(dirname "$0")" && pwd)"
exec python3 "$script_dir/../executed_source/v3_executor.py"

