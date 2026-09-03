#!/usr/bin/env bash
set -u -o pipefail

CODE_DIR=${CODE_DIR:?set CODE_DIR}
MODEL_DIR=${MODEL_DIR:?set MODEL_DIR}
DATA_FILE=${DATA_FILE:?set DATA_FILE}
RUN_DIR=${RUN_DIR:?set RUN_DIR}
ENV_DIR=${ENV_DIR:?set ENV_DIR}
LIMIT_PER_DATASET=${LIMIT_PER_DATASET:-4}
MAX_INPUT_TOKENS=${MAX_INPUT_TOKENS:-4096}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-32}

mkdir -p "$RUN_DIR/logs" "$RUN_DIR/stages"
date -u +%FT%TZ > "$RUN_DIR/stages/00_start"

GPU_COUNT=$(nvidia-smi -L | wc -l | tr -d ' ')
if [ "$GPU_COUNT" -ne 8 ]; then
  echo "expected 8 GPUs, found $GPU_COUNT" >&2
  exit 1
fi
nvidia-smi --query-gpu=index,name,memory.total --format=csv > "$RUN_DIR/gpus.csv"
"$ENV_DIR/bin/python" - <<'PY'
import torch
assert torch.cuda.is_available()
x=torch.arange(1024,device="cuda",dtype=torch.float32)
assert float((x*x).sum()) > 0
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
PY
date -u +%FT%TZ > "$RUN_DIR/stages/01_cuda_ok"

PIDS=()
for RANK in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$RANK "$ENV_DIR/bin/python" "$CODE_DIR/run_downstream.py" \
    --model "$MODEL_DIR" \
    --data "$DATA_FILE" \
    --run-dir "$RUN_DIR" \
    --rank "$RANK" \
    --world-size 8 \
    --limit-per-dataset "$LIMIT_PER_DATASET" \
    --max-input-tokens "$MAX_INPUT_TOKENS" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    > "$RUN_DIR/logs/rank-${RANK}.log" 2>&1 &
  PIDS+=("$!")
  sleep 5
done

FAILED=0
for INDEX in 0 1 2 3 4 5 6 7; do
  if ! wait "${PIDS[$INDEX]}"; then
    echo "rank $INDEX failed; see logs/rank-${INDEX}.log" >&2
    FAILED=1
  fi
done
if [ "$FAILED" -ne 0 ]; then
  date -u +%FT%TZ > "$RUN_DIR/stages/FAILED"
  exit 1
fi

date -u +%FT%TZ > "$RUN_DIR/stages/02_configs_done"
"$ENV_DIR/bin/python" "$CODE_DIR/aggregate_downstream.py" "$RUN_DIR" \
  > "$RUN_DIR/aggregate.log" 2>&1
date -u +%FT%TZ > "$RUN_DIR/stages/99_done"
echo "Q-CoMem downstream pilot complete: $RUN_DIR"
