#!/usr/bin/env bash
set -euo pipefail

if [[ "${R40_H20_EXECUTION_AUTHORIZED:-}" != "yes" ]]; then
  echo "HOLD: set R40_H20_EXECUTION_AUTHORIZED=yes only after independent audit" >&2
  exit 64
fi

if [[ "$#" -ne 2 ]]; then
  echo "usage: $0 /absolute/formal-execution.json /absolute/new-output-root" >&2
  exit 64
fi

script_dir="$(cd "$(dirname "$0")" && pwd)"
exec python3 "$script_dir/v2_one_shot_driver.py" \
  --formal-config "$1" --output-root "$2" --execute

