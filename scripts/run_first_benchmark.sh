#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ ! -x .venv/bin/macllm-run-mlx ]]; then
  bash scripts/bootstrap.sh
fi

.venv/bin/macllm-system-info
.venv/bin/macllm-run-mlx --config configs/smoke.json

echo
echo "Benchmark complete: $repo_root/results/summary.json"
