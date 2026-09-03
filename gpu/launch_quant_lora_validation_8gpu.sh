#!/usr/bin/env bash
set -euo pipefail

CODE_DIR=${CODE_DIR:?set CODE_DIR}
MODEL_DIR=${MODEL_DIR:?set MODEL_DIR}
TRAIN_DATA_FILE=${TRAIN_DATA_FILE:?set TRAIN_DATA_FILE}
VALIDATION_DATA_FILE=${VALIDATION_DATA_FILE:?set VALIDATION_DATA_FILE}
LORA_CHECKPOINT=${LORA_CHECKPOINT:?set LORA_CHECKPOINT}
RUN_DIR=${RUN_DIR:?set RUN_DIR}
ENV_DIR=${ENV_DIR:?set ENV_DIR}
EXPECTED_TRAIN_SHA256=${EXPECTED_TRAIN_SHA256:?set EXPECTED_TRAIN_SHA256}
EXPECTED_VALIDATION_SHA256=${EXPECTED_VALIDATION_SHA256:?set EXPECTED_VALIDATION_SHA256}
EXPECTED_LORA_CHECKPOINT_SHA256=${EXPECTED_LORA_CHECKPOINT_SHA256:?set EXPECTED_LORA_CHECKPOINT_SHA256}
LIMIT_PER_DATASET=${LIMIT_PER_DATASET:-30}
MAX_INPUT_TOKENS=${MAX_INPUT_TOKENS:-4096}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-128}
SEMANTIC_SAMPLES=${SEMANTIC_SAMPLES:-16}

FROZEN_TEST_V2_SHA256=fe046477bf5b39629e9f66fd4def7a55c2d5d1f073c8bb601ee3833f08eaaa5f
LORA_TARGET_CONFIG=replay-d7-frozen-static-lora

# The semantic gate is a fixed 16-window diagnostic on PG-19 train-only data,
# not a quality evaluation and never a LongBench/test-v2 read.  Its SHA is
# independently pinned from the validation corpus below.
if [ "$SEMANTIC_SAMPLES" -ne 16 ]; then
  echo "SEMANTIC_SAMPLES is frozen to 16 PG-19 train-only windows" >&2
  exit 2
fi
NORMALIZED_TRAIN_DATA=$(printf '%s' "$TRAIN_DATA_FILE" | tr '[:upper:]_' '[:lower:]-')
case "$NORMALIZED_TRAIN_DATA" in
  *pg19*|*pg-19*) ;;
  *)
    echo "TRAIN_DATA_FILE must be the pinned PG-19 train-only JSONL" >&2
    exit 2
    ;;
esac
case "$NORMALIZED_TRAIN_DATA" in
  *longbench*|*qasper*|*2wiki*|*test-v2*)
    echo "semantic diagnostic refuses evaluation/test data" >&2
    exit 2
    ;;
esac

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

ACTUAL_TRAIN_SHA256=$(sha256sum "$TRAIN_DATA_FILE" | awk '{print $1}')
ACTUAL_VALIDATION_SHA256=$(sha256sum "$VALIDATION_DATA_FILE" | awk '{print $1}')
ACTUAL_CHECKPOINT_SHA256=$(sha256sum "$LORA_CHECKPOINT" | awk '{print $1}')
if [ "$ACTUAL_TRAIN_SHA256" != "$EXPECTED_TRAIN_SHA256" ]; then
  echo "PG-19 train SHA256 mismatch" >&2
  exit 2
fi
if [ "$ACTUAL_VALIDATION_SHA256" != "$EXPECTED_VALIDATION_SHA256" ]; then
  echo "LongBench validation SHA256 mismatch" >&2
  exit 2
fi
if [ "$ACTUAL_VALIDATION_SHA256" = "$FROZEN_TEST_V2_SHA256" ]; then
  echo "refusing frozen LongBench test-v2" >&2
  exit 2
fi
if [ "$ACTUAL_CHECKPOINT_SHA256" != "$EXPECTED_LORA_CHECKPOINT_SHA256" ]; then
  echo "quant LoRA checkpoint SHA256 mismatch" >&2
  exit 2
fi
sha256sum "$TRAIN_DATA_FILE" > "$RUN_DIR/train-data.sha256"
sha256sum "$VALIDATION_DATA_FILE" > "$RUN_DIR/validation-data.sha256"
sha256sum "$LORA_CHECKPOINT" > "$RUN_DIR/lora-checkpoint.sha256"

"$ENV_DIR/bin/python" -m py_compile \
  "$CODE_DIR/qcomem_torch.py" \
  "$CODE_DIR/qcomem_lora.py" \
  "$CODE_DIR/run_lora_deployment_semantic_gate.py" \
  "$CODE_DIR/aggregate_lora_deployment_semantic_gate.py" \
  "$CODE_DIR/run_replay_diagnostic.py" \
  "$CODE_DIR/aggregate_replay.py"
PYTHONPATH="$CODE_DIR" "$ENV_DIR/bin/python" -m unittest discover \
  -s "$CODE_DIR" -p 'test_lora_deployment_semantic_gate.py' -v \
  > "$RUN_DIR/logs/preflight-tests.log" 2>&1
PYTHONPATH="$CODE_DIR" "$ENV_DIR/bin/python" -m unittest discover \
  -s "$CODE_DIR" -p 'test_mixed_validation.py' -v \
  >> "$RUN_DIR/logs/preflight-tests.log" 2>&1

PYTHONPATH="$CODE_DIR" \
VALIDATION_DATA_FILE="$VALIDATION_DATA_FILE" \
EXPECTED_VALIDATION_SHA256="$EXPECTED_VALIDATION_SHA256" \
LIMIT_PER_DATASET="$LIMIT_PER_DATASET" \
"$ENV_DIR/bin/python" - <<'PY' > "$RUN_DIR/logs/protocol-preflight.log"
import hashlib
import os
from collections import Counter, defaultdict
from pathlib import Path

from run_downstream import load_samples

path = Path(os.environ["VALIDATION_DATA_FILE"])
digest = hashlib.sha256(path.read_bytes()).hexdigest()
if digest != os.environ["EXPECTED_VALIDATION_SHA256"]:
    raise SystemExit("validation SHA changed during preflight")
limit = int(os.environ["LIMIT_PER_DATASET"])
rows = load_samples(
    path,
    limit,
    source_index_start=6,
    source_index_end=35,
    exclude_source_indices=(4, 5),
)
counts = Counter(row["dataset"] for row in rows)
if counts != {"qasper": limit, "2wikimqa": limit}:
    raise SystemExit(f"incomplete validation slice: {dict(counts)}")
indices = defaultdict(set)
keys = set()
for row in rows:
    index = int(row["_source_index"])
    indices[row["dataset"]].add(index)
    keys.add((row["dataset"], row.get("_id"), index))
if limit == 30:
    expected = set(range(6, 36))
    if any(value != expected for value in indices.values()):
        raise SystemExit(f"source-index coverage mismatch: {dict(indices)}")
if len(keys) != len(rows):
    raise SystemExit("validation rows are not unique")
print({"sha256": digest, "samples": len(rows), "per_dataset": dict(counts),
       "source_index_start": 6, "source_index_end": 35,
       "excluded": [4, 5], "test_v2_used": False})
PY
date -u +%FT%TZ > "$RUN_DIR/stages/01_preflight_ok"

# Reconfirm the underlying split/cache implementation before evaluating the
# adapter.  This gate keeps LoRA conclusions separate from replay regressions.
CUDA_VISIBLE_DEVICES=0 "$ENV_DIR/bin/python" "$CODE_DIR/run_cached_smoke.py" \
  --model "$MODEL_DIR" \
  --data "$VALIDATION_DATA_FILE" \
  --output "$RUN_DIR/cached-smoke.json" \
  --depth 7 --max-input-tokens 512 --max-new-tokens 3 \
  > "$RUN_DIR/logs/cached-smoke.log" 2>&1
"$ENV_DIR/bin/python" - "$RUN_DIR/cached-smoke.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit("cached smoke did not write its JSON result")
result = json.loads(path.read_text())
if result.get("status") != "passed":
    raise SystemExit(f"cached smoke JSON is not passed: {result.get('status')!r}")
matches = result.get("matches_oracle", {})
if not matches or not all(matches.values()):
    raise SystemExit(f"cached smoke oracle checks are incomplete/failed: {matches}")
print({"status": result["status"], "matches_oracle": matches})
PY
date -u +%FT%TZ > "$RUN_DIR/stages/02_exactness_ok"

# Hard training-vs-deployment suffix gate.  All 8 H20s share the 16 windows;
# the aggregate owns the strict global decision so a failed local shard is
# retained as evidence rather than dropped.
SEMANTIC_PIDS=()
for RANK in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$RANK "$ENV_DIR/bin/python" \
    "$CODE_DIR/run_lora_deployment_semantic_gate.py" \
    --model "$MODEL_DIR" \
    --data "$TRAIN_DATA_FILE" \
    --checkpoint "$LORA_CHECKPOINT" \
    --output "$RUN_DIR/deployment-semantic-shard-${RANK}.json" \
    --depth 7 \
    --residual-bits 4 --attention-bits 4 --linear-bits 8 \
    --cache-layer-bits 8,8,8,4,8,8,8 \
    --context-tokens 1792 --query-tokens 256 \
    --samples "$SEMANTIC_SAMPLES" --projection-block-size 16 \
    --min-top1-match 1.0 --max-mean-kl 0.001 \
    --rank "$RANK" --world-size 8 \
    > "$RUN_DIR/logs/deployment-semantic-rank-${RANK}.log" 2>&1 &
  SEMANTIC_PIDS+=("$!")
  sleep 5
done
SEMANTIC_FAILED=0
for INDEX in 0 1 2 3 4 5 6 7; do
  if ! wait "${SEMANTIC_PIDS[$INDEX]}"; then
    echo "semantic rank $INDEX failed" >&2
    SEMANTIC_FAILED=1
  fi
done
if [ "$SEMANTIC_FAILED" -ne 0 ]; then
  date -u +%FT%TZ > "$RUN_DIR/stages/FAILED"
  exit 1
fi
"$ENV_DIR/bin/python" "$CODE_DIR/aggregate_lora_deployment_semantic_gate.py" \
  "$RUN_DIR" --expected-world-size 8 \
  --expected-data-sha256 "$EXPECTED_TRAIN_SHA256" \
  --expected-checkpoint-sha256 "$EXPECTED_LORA_CHECKPOINT_SHA256" \
  > "$RUN_DIR/logs/deployment-semantic-gate.log" 2>&1
test -s "$RUN_DIR/deployment-semantic-gate.json"
date -u +%FT%TZ > "$RUN_DIR/stages/03_semantic_gate_ok"

PIDS=()
for RANK in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$RANK "$ENV_DIR/bin/python" \
    "$CODE_DIR/run_replay_diagnostic.py" \
    --model "$MODEL_DIR" \
    --data "$VALIDATION_DATA_FILE" \
    --run-dir "$RUN_DIR" \
    --rank "$RANK" --world-size 8 \
    --suite quant-lora-validation \
    --source-index-start 6 --source-index-end 35 \
    --exclude-source-indices 4,5 \
    --limit-per-dataset "$LIMIT_PER_DATASET" \
    --max-input-tokens "$MAX_INPUT_TOKENS" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --lora-checkpoint "$LORA_CHECKPOINT" \
    --lora-apply-to-configs "$LORA_TARGET_CONFIG" \
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
date -u +%FT%TZ > "$RUN_DIR/stages/04_shards_done"

"$ENV_DIR/bin/python" "$CODE_DIR/aggregate_replay.py" "$RUN_DIR" \
  --suite quant-lora-validation \
  --expected-world-size 8 \
  --overall-margin -0.02 --dataset-margin -0.03 \
  --catastrophic-delta -0.5 \
  --expected-data-sha256 "$EXPECTED_VALIDATION_SHA256" \
  --expected-checkpoint-sha256 "$EXPECTED_LORA_CHECKPOINT_SHA256" \
  > "$RUN_DIR/aggregate.log" 2>&1
test -s "$RUN_DIR/replay_analysis.json"
nvidia-smi \
  --query-gpu=index,uuid,name,memory.used,pstate,power.draw,temperature.gpu \
  --format=csv > "$RUN_DIR/gpus-after.csv"
date -u +%FT%TZ > "$RUN_DIR/stages/99_done"
echo "Quant LoRA paired validation complete: $RUN_DIR"
