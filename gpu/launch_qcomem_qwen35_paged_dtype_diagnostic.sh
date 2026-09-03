#!/usr/bin/env bash
set -euo pipefail

CODE_DIR=${CODE_DIR:?set CODE_DIR}
MODEL_DIR=${MODEL_DIR:?set MODEL_DIR}
RUN_DIR=${RUN_DIR:?set RUN_DIR}
ENV_DIR=${ENV_DIR:?set ENV_DIR}

if [[ -e "$RUN_DIR" ]] && [[ -n "$(find "$RUN_DIR" -mindepth 1 -print -quit)" ]]; then
  echo "RUN_DIR must be absent or empty: $RUN_DIR" >&2
  exit 2
fi
GPU_COUNT=$(nvidia-smi -L | wc -l | tr -d ' ')
if [[ "$GPU_COUNT" -ne 1 ]]; then
  echo "single-rank diagnostic requires exactly one GPU, found $GPU_COUNT" >&2
  exit 2
fi

mkdir -p "$RUN_DIR/logs" "$RUN_DIR/stages"
trap 'date -u +%FT%TZ > "$RUN_DIR/stages/FAILED"' ERR
date -u +%FT%TZ > "$RUN_DIR/stages/00_start"
nvidia-smi --query-gpu=index,uuid,name,driver_version,memory.total,pstate,power.limit \
  --format=csv > "$RUN_DIR/gpu-before.csv"
LC_ALL=C find "$CODE_DIR" -maxdepth 1 -type f \( -name '*.py' -o -name '*.sh' \) \
  -print0 | LC_ALL=C sort -z | xargs -0 sha256sum > "$RUN_DIR/code.sha256"

"$ENV_DIR/bin/python" -m py_compile \
  "$CODE_DIR/diagnose_qcomem_qwen35_paged_dtype.py" \
  "$CODE_DIR/qcomem_paged_attention.py" \
  "$CODE_DIR/qcomem_qwen35_paged_integration.py"
PYTHONPATH="$CODE_DIR" "$ENV_DIR/bin/python" -m unittest -v \
  test_qcomem_paged_attention \
  test_qcomem_qwen35_paged_integration \
  > "$RUN_DIR/logs/preflight-tests.log" 2>&1
date -u +%FT%TZ > "$RUN_DIR/stages/01_preflight_ok"

CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$CODE_DIR" "$ENV_DIR/bin/python" \
  "$CODE_DIR/diagnose_qcomem_qwen35_paged_dtype.py" \
  --model "$MODEL_DIR" \
  --output "$RUN_DIR/paged-dtype-diagnostic.json" \
  > "$RUN_DIR/logs/diagnostic.log" 2>&1
test -s "$RUN_DIR/paged-dtype-diagnostic.json"
date -u +%FT%TZ > "$RUN_DIR/stages/99_done"
