#!/usr/bin/env bash
set -euo pipefail

CODE_DIR=${CODE_DIR:?set CODE_DIR}
MODEL_DIR=${MODEL_DIR:?set MODEL_DIR}
DATA_FILE=${DATA_FILE:?set DATA_FILE}
RUN_DIR=${RUN_DIR:?set RUN_DIR}
ENV_DIR=${ENV_DIR:?set ENV_DIR}
LIMIT_PER_DATASET=${LIMIT_PER_DATASET:-30}
MAX_INPUT_TOKENS=${MAX_INPUT_TOKENS:-4096}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-128}
SOURCE_INDEX_START=${SOURCE_INDEX_START:-6}
SOURCE_INDEX_END=${SOURCE_INDEX_END:-35}
EXCLUDE_SOURCE_INDICES=${EXCLUDE_SOURCE_INDICES:-4,5}
OVERALL_MARGIN=${OVERALL_MARGIN:--0.02}
DATASET_MARGIN=${DATASET_MARGIN:--0.03}
LORA_CHECKPOINT=${LORA_CHECKPOINT:-}
LORA_APPLY_TO_CONFIGS=${LORA_APPLY_TO_CONFIGS:-}

if { [ -n "$LORA_CHECKPOINT" ] && [ -z "$LORA_APPLY_TO_CONFIGS" ]; } || \
   { [ -z "$LORA_CHECKPOINT" ] && [ -n "$LORA_APPLY_TO_CONFIGS" ]; }; then
  echo "LORA_CHECKPOINT and LORA_APPLY_TO_CONFIGS must be set together" >&2
  exit 1
fi

LORA_ARGS=()
if [ -n "$LORA_CHECKPOINT" ]; then
  IFS=',' read -r -a LORA_TARGETS <<< "$LORA_APPLY_TO_CONFIGS"
  LORA_ARGS+=(--lora-checkpoint "$LORA_CHECKPOINT")
  LORA_ARGS+=(--lora-apply-to-configs "${LORA_TARGETS[@]}")
fi

mkdir -p "$RUN_DIR/logs" "$RUN_DIR/stages"
date -u +%FT%TZ > "$RUN_DIR/stages/00_start"
nvidia-smi --query-gpu=index,name,memory.total --format=csv > "$RUN_DIR/gpus.csv"

GPU_COUNT=$(nvidia-smi -L | wc -l | tr -d ' ')
if [ "$GPU_COUNT" -ne 8 ]; then
  echo "expected 8 GPUs, found $GPU_COUNT" >&2
  exit 1
fi

"$ENV_DIR/bin/python" -m py_compile \
  "$CODE_DIR/qcomem_torch.py" \
  "$CODE_DIR/run_downstream.py" \
  "$CODE_DIR/run_replay_diagnostic.py" \
  "$CODE_DIR/aggregate_replay.py"

PYTHONPATH="$CODE_DIR" \
DATA_FILE="$DATA_FILE" \
LIMIT_PER_DATASET="$LIMIT_PER_DATASET" \
SOURCE_INDEX_START="$SOURCE_INDEX_START" \
SOURCE_INDEX_END="$SOURCE_INDEX_END" \
EXCLUDE_SOURCE_INDICES="$EXCLUDE_SOURCE_INDICES" \
"$ENV_DIR/bin/python" - <<'PY'
import os
from collections import Counter
from pathlib import Path

from run_downstream import load_samples
from run_replay_diagnostic import parse_excluded_indices

limit = int(os.environ["LIMIT_PER_DATASET"])
start = int(os.environ["SOURCE_INDEX_START"])
end = int(os.environ["SOURCE_INDEX_END"])
excluded = parse_excluded_indices(os.environ["EXCLUDE_SOURCE_INDICES"])
rows = load_samples(
    Path(os.environ["DATA_FILE"]),
    limit,
    source_index_start=start,
    source_index_end=end,
    exclude_source_indices=excluded,
)
counts = Counter(row["dataset"] for row in rows)
if not counts or any(count != limit for count in counts.values()):
    raise SystemExit(f"incomplete validation slice: {dict(counts)}")
if any(int(row["_source_index"]) in excluded for row in rows):
    raise SystemExit("calibration sample leaked into validation")
print({"samples": len(rows), "per_dataset": dict(counts)})
PY
date -u +%FT%TZ > "$RUN_DIR/stages/01_protocol_ok"

# The job stops here if manual split replay, full-prefix cache, fixed-order
# multidocument replay, or the per-layer Q16 path differs from the oracle.
CUDA_VISIBLE_DEVICES=0 "$ENV_DIR/bin/python" "$CODE_DIR/run_cached_smoke.py" \
  --model "$MODEL_DIR" \
  --data "$DATA_FILE" \
  --output "$RUN_DIR/cached_smoke.json" \
  --depth 7 \
  --max-input-tokens 512 \
  --max-new-tokens 3 \
  > "$RUN_DIR/logs/cached-smoke.log" 2>&1
date -u +%FT%TZ > "$RUN_DIR/stages/02_exactness_ok"

PIDS=()
for RANK in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$RANK "$ENV_DIR/bin/python" "$CODE_DIR/run_replay_diagnostic.py" \
    --model "$MODEL_DIR" \
    --data "$DATA_FILE" \
    --run-dir "$RUN_DIR" \
    --rank "$RANK" \
    --world-size 8 \
    --limit-per-dataset "$LIMIT_PER_DATASET" \
    --max-input-tokens "$MAX_INPUT_TOKENS" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --source-index-start "$SOURCE_INDEX_START" \
    --source-index-end "$SOURCE_INDEX_END" \
    --exclude-source-indices "$EXCLUDE_SOURCE_INDICES" \
    --suite layer-validation \
    "${LORA_ARGS[@]}" \
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

date -u +%FT%TZ > "$RUN_DIR/stages/03_shards_done"
"$ENV_DIR/bin/python" "$CODE_DIR/aggregate_replay.py" "$RUN_DIR" \
  --suite layer-validation \
  --expected-world-size 8 \
  --overall-margin "$OVERALL_MARGIN" \
  --dataset-margin "$DATASET_MARGIN" \
  > "$RUN_DIR/aggregate.log" 2>&1
test -s "$RUN_DIR/replay_analysis.json"
date -u +%FT%TZ > "$RUN_DIR/stages/99_done"
echo "Q-CoMem mixed-bit downstream validation complete: $RUN_DIR"
