#!/usr/bin/env bash
set -euo pipefail

CODE_DIR=${CODE_DIR:?set CODE_DIR}
MODEL_DIR=${MODEL_DIR:?set MODEL_DIR}
DATA_FILE=${DATA_FILE:?set DATA_FILE}
RUN_DIR=${RUN_DIR:?set RUN_DIR}
ENV_DIR=${ENV_DIR:?set ENV_DIR}
LIMIT_PER_DATASET=${LIMIT_PER_DATASET:-2}
MAX_INPUT_TOKENS=${MAX_INPUT_TOKENS:-2048}

mkdir -p "$RUN_DIR/logs"
nvidia-smi --query-gpu=index,name,memory.total --format=csv > "$RUN_DIR/gpus.csv"

CUDA_VISIBLE_DEVICES=0 "$ENV_DIR/bin/python" "$CODE_DIR/run_cached_smoke.py" \
  --model "$MODEL_DIR" \
  --data "$DATA_FILE" \
  --output "$RUN_DIR/cached_smoke.json" \
  --depth 7 \
  --max-input-tokens 512 \
  --max-new-tokens 3 \
  > "$RUN_DIR/logs/cached-smoke.log" 2>&1

PIDS=()
for RANK in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$RANK "$ENV_DIR/bin/python" "$CODE_DIR/run_layer_sensitivity.py" \
    --model "$MODEL_DIR" \
    --data "$DATA_FILE" \
    --run-dir "$RUN_DIR" \
    --rank "$RANK" \
    --world-size 8 \
    --depth 7 \
    --limit-per-dataset "$LIMIT_PER_DATASET" \
    --max-input-tokens "$MAX_INPUT_TOKENS" \
    > "$RUN_DIR/logs/sensitivity-${RANK}.log" 2>&1 &
  PIDS+=("$!")
  sleep 5
done

FAILED=0
for INDEX in 0 1 2 3 4 5 6 7; do
  if ! wait "${PIDS[$INDEX]}"; then
    echo "rank $INDEX failed; see logs/sensitivity-${INDEX}.log" >&2
    FAILED=1
  fi
done
if [ "$FAILED" -ne 0 ]; then
  exit 1
fi

"$ENV_DIR/bin/python" "$CODE_DIR/aggregate_layer_sensitivity.py" "$RUN_DIR" \
  > "$RUN_DIR/layer-policy.log" 2>&1
echo "Q-CoMem layer sensitivity complete: $RUN_DIR"
