#!/usr/bin/env bash
set -euo pipefail

CODE_DIR=${CODE_DIR:?set CODE_DIR}
MODEL_DIR=${MODEL_DIR:?set MODEL_DIR}
DATA_FILE=${DATA_FILE:?set DATA_FILE}
RUN_DIR=${RUN_DIR:?set RUN_DIR}
ENV_DIR=${ENV_DIR:?set ENV_DIR}
LIMIT_PER_DATASET=${LIMIT_PER_DATASET:-4}
SOURCE_INDEX_START=${SOURCE_INDEX_START:-6}
SOURCE_INDEX_END=${SOURCE_INDEX_END:-35}
EXCLUDE_SOURCE_INDICES=${EXCLUDE_SOURCE_INDICES:-4,5}
MAX_INPUT_TOKENS=${MAX_INPUT_TOKENS:-4096}
SEGMENT_COUNT=${SEGMENT_COUNT:-4}
RESIDUAL_BITS=${RESIDUAL_BITS:-4}
ATTENTION_BITS=${ATTENTION_BITS:-4}
LINEAR_BITS=${LINEAR_BITS:-8}
CACHE_LAYER_BITS=${CACHE_LAYER_BITS:-8,8,8,4,8,8,8}
WARMUPS=${WARMUPS:-1}
REPEATS=${REPEATS:-3}
CONFIGS=${CONFIGS:-hypic-lite-naive-w0,hypic-lite-naive-w8,hypic-lite-transition-w0,hypic-lite-transition-w8}

mkdir -p "$RUN_DIR/logs" "$RUN_DIR/stages"
date -u +%FT%TZ > "$RUN_DIR/stages/00_start"
GPU_COUNT=$(nvidia-smi -L | wc -l | tr -d ' ')
if [ "$GPU_COUNT" -ne 8 ]; then
  echo "expected 8 H20 GPUs, found $GPU_COUNT" >&2
  exit 1
fi
nvidia-smi \
  --query-gpu=index,uuid,name,memory.total,driver_version,pstate,power.limit \
  --format=csv > "$RUN_DIR/gpus-before.csv"

"$ENV_DIR/bin/python" -m py_compile \
  "$CODE_DIR/hypic_lite.py" \
  "$CODE_DIR/run_hypic_lite_bench.py" \
  "$CODE_DIR/aggregate_hypic_lite.py" \
  "$CODE_DIR/qcomem_torch.py" \
  "$CODE_DIR/qcomem_deployment.py"
PYTHONPATH="$CODE_DIR" "$ENV_DIR/bin/python" -m unittest discover \
  -s "$CODE_DIR" -p 'test_hypic_lite.py' -v \
  > "$RUN_DIR/hypic-lite-tests.log" 2>&1

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

excluded = tuple(int(value) for value in os.environ["EXCLUDE_SOURCE_INDICES"].split(","))
rows = load_samples(
    Path(os.environ["DATA_FILE"]),
    int(os.environ["LIMIT_PER_DATASET"]),
    source_index_start=int(os.environ["SOURCE_INDEX_START"]),
    source_index_end=int(os.environ["SOURCE_INDEX_END"]),
    exclude_source_indices=excluded,
)
counts = Counter(row["dataset"] for row in rows)
expected = int(os.environ["LIMIT_PER_DATASET"])
if set(counts) != {"qasper", "2wikimqa"} or any(value != expected for value in counts.values()):
    raise SystemExit(f"incomplete validation slice: {dict(counts)}")
if any(int(row["_source_index"]) in excluded for row in rows):
    raise SystemExit("calibration indices leaked into validation")
if any(int(row["_source_index"]) >= 68 for row in rows):
    raise SystemExit("test-v2 is frozen and cannot be consumed")
print({"rows": len(rows), "per_dataset": dict(counts)})
PY
date -u +%FT%TZ > "$RUN_DIR/stages/01_preflight_ok"

IFS=',' read -r -a CONFIG_LIST <<< "$CONFIGS"
if [ "${#CONFIG_LIST[@]}" -eq 0 ]; then
  echo "CONFIGS must be non-empty" >&2
  exit 1
fi
COMMON_ARGS=(
  --model "$MODEL_DIR"
  --data "$DATA_FILE"
  --run-dir "$RUN_DIR"
  --world-size 8
  --depth 7
  --segment-count "$SEGMENT_COUNT"
  --limit-per-dataset "$LIMIT_PER_DATASET"
  --source-index-start "$SOURCE_INDEX_START"
  --source-index-end "$SOURCE_INDEX_END"
  --exclude-source-indices "$EXCLUDE_SOURCE_INDICES"
  --max-input-tokens "$MAX_INPUT_TOKENS"
  --residual-bits "$RESIDUAL_BITS"
  --attention-bits "$ATTENTION_BITS"
  --linear-bits "$LINEAR_BITS"
  --cache-layer-bits "$CACHE_LAYER_BITS"
  --warmups "$WARMUPS"
  --repeats "$REPEATS"
  --configs "${CONFIG_LIST[@]}"
)

PIDS=()
for RANK in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$RANK "$ENV_DIR/bin/python" \
    "$CODE_DIR/run_hypic_lite_bench.py" \
    "${COMMON_ARGS[@]}" --rank "$RANK" \
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

PYTHONPATH="$CODE_DIR" "$ENV_DIR/bin/python" \
  "$CODE_DIR/aggregate_hypic_lite.py" "$RUN_DIR" --expected-shards 8 \
  > "$RUN_DIR/aggregate.log" 2>&1
test -s "$RUN_DIR/hypic-lite-analysis.json"
nvidia-smi \
  --query-gpu=index,uuid,name,memory.used,pstate,power.draw,temperature.gpu \
  --format=csv > "$RUN_DIR/gpus-after.csv"
date -u +%FT%TZ > "$RUN_DIR/stages/99_done"
echo "HYPIC-lite reference benchmark complete: $RUN_DIR"
