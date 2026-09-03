#!/usr/bin/env bash
set -u -o pipefail

CODE_DIR=${CODE_DIR:?set CODE_DIR}
MODEL_DIR=${MODEL_DIR:?set MODEL_DIR}
DATA_FILE=${DATA_FILE:?set DATA_FILE}
RUN_DIR=${RUN_DIR:?set RUN_DIR}
ENV_DIR=${ENV_DIR:?set ENV_DIR}

mkdir -p "$RUN_DIR"
nvidia-smi --query-gpu=index,name,memory.total --format=csv > "$RUN_DIR/gpus.csv"
"$ENV_DIR/bin/python" "$CODE_DIR/run_cached_smoke.py" \
  --model "$MODEL_DIR" \
  --data "$DATA_FILE" \
  --output "$RUN_DIR/cached_smoke.json" \
  --depth 7 \
  --max-input-tokens 512 \
  --max-new-tokens 3 \
  > "$RUN_DIR/cached_smoke.log" 2>&1
echo "Q-CoMem cached smoke complete: $RUN_DIR"
