#!/usr/bin/env bash
set -euo pipefail

CODE_DIR=${CODE_DIR:?set CODE_DIR}
MODEL_DIR=${MODEL_DIR:?set MODEL_DIR}
DATA_FILE=${DATA_FILE:?set DATA_FILE}
RUN_DIR=${RUN_DIR:?set RUN_DIR}
ENV_DIR=${ENV_DIR:?set ENV_DIR}
LORA_CHECKPOINT=${LORA_CHECKPOINT:?set LORA_CHECKPOINT}
EXPECTED_VALIDATION_SHA256=${EXPECTED_VALIDATION_SHA256:?set EXPECTED_VALIDATION_SHA256}
EXPECTED_LORA_CHECKPOINT_SHA256=${EXPECTED_LORA_CHECKPOINT_SHA256:?set EXPECTED_LORA_CHECKPOINT_SHA256}

FROZEN_TEST_V2_SHA256=fe046477bf5b39629e9f66fd4def7a55c2d5d1f073c8bb601ee3833f08eaaa5f
NORMALIZED_DATA_FILE=$(printf '%s' "$DATA_FILE" | tr '[:upper:]_' '[:lower:]-')
case "$NORMALIZED_DATA_FILE" in
  *qcomem-longbench-test-v2*|*longbench-test-v2*)
    echo "refusing frozen LongBench test-v2 by path" >&2
    exit 2
    ;;
esac

mkdir -p "$RUN_DIR/logs" "$RUN_DIR/stages"
date -u +%FT%TZ > "$RUN_DIR/stages/00_start"
test "$(nvidia-smi -L | wc -l | tr -d ' ')" -eq 8
nvidia-smi --query-gpu=index,name,memory.total --format=csv > "$RUN_DIR/gpus.csv"
ACTUAL_VALIDATION_SHA256=$(sha256sum "$DATA_FILE" | awk '{print $1}')
ACTUAL_LORA_CHECKPOINT_SHA256=$(sha256sum "$LORA_CHECKPOINT" | awk '{print $1}')
if [ "$ACTUAL_VALIDATION_SHA256" = "$FROZEN_TEST_V2_SHA256" ]; then
  echo "refusing frozen LongBench test-v2 by SHA256" >&2
  exit 2
fi
if [ "$ACTUAL_VALIDATION_SHA256" != "$EXPECTED_VALIDATION_SHA256" ]; then
  echo "validation SHA256 mismatch" >&2
  exit 2
fi
if [ "$ACTUAL_LORA_CHECKPOINT_SHA256" != "$EXPECTED_LORA_CHECKPOINT_SHA256" ]; then
  echo "Interface LoRA checkpoint SHA256 mismatch" >&2
  exit 2
fi
sha256sum "$DATA_FILE" > "$RUN_DIR/validation-data.sha256"
sha256sum "$LORA_CHECKPOINT" > "$RUN_DIR/lora-checkpoint.sha256"

"$ENV_DIR/bin/python" -m py_compile \
  "$CODE_DIR/qcomem_lora.py" \
  "$CODE_DIR/run_interface_diagnostic.py" \
  "$CODE_DIR/aggregate_interface_lora.py"
PYTHONPATH="$CODE_DIR" \
DATA_FILE="$DATA_FILE" \
EXPECTED_VALIDATION_SHA256="$EXPECTED_VALIDATION_SHA256" \
"$ENV_DIR/bin/python" - <<'PY' > "$RUN_DIR/logs/protocol-preflight.log"
import hashlib
import os
from collections import Counter, defaultdict
from pathlib import Path

from run_downstream import load_samples

path = Path(os.environ["DATA_FILE"])
digest = hashlib.sha256(path.read_bytes()).hexdigest()
if digest != os.environ["EXPECTED_VALIDATION_SHA256"]:
    raise SystemExit("validation SHA changed between shell and Python preflight")
rows = load_samples(
    path,
    30,
    source_index_start=6,
    source_index_end=35,
    exclude_source_indices=(4, 5),
)
counts = Counter(row["dataset"] for row in rows)
if counts != {"qasper": 30, "2wikimqa": 30}:
    raise SystemExit(f"expected 30 Qasper + 30 2Wiki validation rows: {dict(counts)}")
indices = defaultdict(set)
keys = set()
for row in rows:
    source_index = int(row["_source_index"])
    indices[row["dataset"]].add(source_index)
    keys.add((row["dataset"], row.get("_id"), source_index))
expected_indices = set(range(6, 36))
if any(value != expected_indices for value in indices.values()):
    raise SystemExit(f"validation source-index coverage mismatch: {dict(indices)}")
if len(rows) != 60 or len(keys) != 60:
    raise SystemExit(f"expected 60 unique validation rows, got {len(rows)}/{len(keys)}")
print({"sha256": digest, "samples": len(rows), "per_dataset": dict(counts),
       "source_index_start": 6, "source_index_end": 35, "test_v2_used": False})
PY
date -u +%FT%TZ > "$RUN_DIR/stages/01_preflight_ok"

PIDS=()
for RANK in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$RANK "$ENV_DIR/bin/python" "$CODE_DIR/run_interface_diagnostic.py" \
    --model "$MODEL_DIR" --data "$DATA_FILE" --run-dir "$RUN_DIR" \
    --rank "$RANK" --world-size 8 --suite lora-validation \
    --source-index-start 6 --source-index-end 35 --limit-per-dataset 30 \
    --max-input-tokens 4096 --max-new-tokens 128 --chunk-size 512 --overlap 0 \
    --expected-data-sha256 "$EXPECTED_VALIDATION_SHA256" \
    --lora-checkpoint "$LORA_CHECKPOINT" \
    > "$RUN_DIR/logs/rank-${RANK}.log" 2>&1 &
  PIDS+=("$!")
  sleep 5
done
FAILED=0
for INDEX in 0 1 2 3 4 5 6 7; do
  wait "${PIDS[$INDEX]}" || FAILED=1
done
test "$FAILED" -eq 0
date -u +%FT%TZ > "$RUN_DIR/stages/02_shards_done"
"$ENV_DIR/bin/python" "$CODE_DIR/aggregate_interface_lora.py" "$RUN_DIR" \
  --expected-data-sha256 "$EXPECTED_VALIDATION_SHA256" \
  --expected-checkpoint-sha256 "$EXPECTED_LORA_CHECKPOINT_SHA256" \
  > "$RUN_DIR/aggregate.log" 2>&1
date -u +%FT%TZ > "$RUN_DIR/stages/99_done"
