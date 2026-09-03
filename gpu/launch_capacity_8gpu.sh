#!/usr/bin/env bash
set -u -o pipefail

CODE_DIR=${CODE_DIR:?set CODE_DIR}
MODEL_DIR=${MODEL_DIR:?set MODEL_DIR}
RUN_DIR=${RUN_DIR:?set RUN_DIR}
ENV_DIR=${ENV_DIR:?set ENV_DIR}
CAPACITY_SUITE=${CAPACITY_SUITE:-extreme}

mkdir -p "$RUN_DIR/logs" "$RUN_DIR/stages"
date -u +%FT%TZ > "$RUN_DIR/stages/00_start"
GPU_COUNT=$(nvidia-smi -L | wc -l | tr -d ' ')
if [ "$GPU_COUNT" -ne 8 ]; then
  echo "expected 8 GPUs, found $GPU_COUNT" >&2
  exit 1
fi
date -u +%FT%TZ > "$RUN_DIR/stages/01_cuda_ok"

PIDS=()
for RANK in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$RANK "$ENV_DIR/bin/python" "$CODE_DIR/run_capacity_scaling.py" \
    --model "$MODEL_DIR" --run-dir "$RUN_DIR" --rank "$RANK" \
    --suite "$CAPACITY_SUITE" \
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

date -u +%FT%TZ > "$RUN_DIR/stages/02_shards_done"
"$ENV_DIR/bin/python" "$CODE_DIR/aggregate_capacity.py" "$RUN_DIR" \
  > "$RUN_DIR/aggregate.log" 2>&1
date -u +%FT%TZ > "$RUN_DIR/stages/99_done"
