#!/usr/bin/env bash
set -u -o pipefail
CODE_DIR=${CODE_DIR:?}; MODEL_DIR=${MODEL_DIR:?}; RUN_DIR=${RUN_DIR:?}; ENV_DIR=${ENV_DIR:?}
DEPTHS=${DEPTHS:-7,13,20,26}
LENGTHS=${LENGTHS:-4096,8192,16384,32768}
BITS=${BITS:-r4-a8-l8}
REPEATS=${REPEATS:-3}
mkdir -p "$RUN_DIR/logs"
G=$(nvidia-smi -L | wc -l | tr -d ' '); [ "$G" -eq 8 ] || { echo "expected 8 GPUs, got $G" >&2; exit 1; }
PIDS=()
for RANK in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$RANK PYTHONPATH="$CODE_DIR" "$ENV_DIR/bin/python" \
    "$CODE_DIR/run_capacity_scaling_r45.py" --model "$MODEL_DIR" --run-dir "$RUN_DIR" \
    --rank "$RANK" --depths "$DEPTHS" --lengths "$LENGTHS" --bits "$BITS" --repeats "$REPEATS" \
    > "$RUN_DIR/logs/rank-${RANK}.log" 2>&1 &
  PIDS+=("$!"); sleep 4
done
F=0; for i in 0 1 2 3 4 5 6 7; do wait "${PIDS[$i]}" || { echo "rank $i FAILED"; tail -20 "$RUN_DIR/logs/rank-${i}.log"; F=1; }; done
exit $F
