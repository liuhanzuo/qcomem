#!/usr/bin/env bash
set -euo pipefail

CODE_DIR=${CODE_DIR:?set CODE_DIR}
MODEL_DIR=${MODEL_DIR:?set MODEL_DIR}
CHECKPOINT_DIR=${CHECKPOINT_DIR:?set CHECKPOINT_DIR}
EXPECTED_CHECKPOINT_MANIFEST_SHA256=${EXPECTED_CHECKPOINT_MANIFEST_SHA256:?set EXPECTED_CHECKPOINT_MANIFEST_SHA256}
DATA_FILE=${DATA_FILE:?set DATA_FILE}
EXPECTED_DATA_SHA256=${EXPECTED_DATA_SHA256:?set EXPECTED_DATA_SHA256}
INTERFACE_LORA_RUN_DIR=${INTERFACE_LORA_RUN_DIR:?set INTERFACE_LORA_RUN_DIR}
EXPECTED_LORA_CHECKPOINT_SHA256=${EXPECTED_LORA_CHECKPOINT_SHA256:?set EXPECTED_LORA_CHECKPOINT_SHA256}
RUN_DIR=${RUN_DIR:?set RUN_DIR}
ENV_DIR=${ENV_DIR:?set ENV_DIR}

for VALUE in \
  "$EXPECTED_CHECKPOINT_MANIFEST_SHA256" \
  "$EXPECTED_DATA_SHA256" \
  "$EXPECTED_LORA_CHECKPOINT_SHA256"; do
  if [[ ! "$VALUE" =~ ^[0-9a-f]{64}$ ]]; then
    echo "expected hashes must be lowercase SHA256 values" >&2
    exit 2
  fi
done
if [ -e "$RUN_DIR" ] && [ -n "$(find "$RUN_DIR" -mindepth 1 -print -quit 2>/dev/null)" ]; then
  echo "RUN_DIR must be absent or empty: $RUN_DIR" >&2
  exit 2
fi
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/stages"
date -u +%FT%TZ > "$RUN_DIR/stages/00_start"

FILES=(
  "$CODE_DIR/run_sft_dcp_downstream.py"
  "$CODE_DIR/aggregate_sft_dcp_downstream.py"
  "$CODE_DIR/dcp_replicated_load_preflight.py"
  "$CODE_DIR/sft_dcp_checkpoint.py"
  "$CODE_DIR/sft_quality_validation.py"
  "$CODE_DIR/supervised_sft.py"
  "$CODE_DIR/run_downstream.py"
  "$CODE_DIR/qcomem_torch.py"
  "$CODE_DIR/analyze_validation.py"
  "$CODE_DIR/launch_sft_dcp_downstream_8gpu.sh"
  "$CHECKPOINT_DIR/checkpoint-manifest.json"
  "$CHECKPOINT_DIR/_SUCCESS"
  "$DATA_FILE"
)
for FILE in "${FILES[@]}"; do
  test -f "$FILE" || { echo "missing required artifact: $FILE" >&2; exit 2; }
done
test "$(nvidia-smi -L | wc -l | tr -d ' ')" -eq 8
if [ "$(sha256sum "$CHECKPOINT_DIR/checkpoint-manifest.json" | awk '{print $1}')" != "$EXPECTED_CHECKPOINT_MANIFEST_SHA256" ]; then
  echo "selected SFT checkpoint manifest SHA256 mismatch" >&2
  exit 2
fi
if [ "$(sha256sum "$DATA_FILE" | awk '{print $1}')" != "$EXPECTED_DATA_SHA256" ]; then
  echo "frozen LongBench validation SHA256 mismatch" >&2
  exit 2
fi
if [ "$EXPECTED_DATA_SHA256" = "fe046477bf5b39629e9f66fd4def7a55c2d5d1f073c8bb601ee3833f08eaaa5f" ]; then
  echo "refusing frozen LongBench test-v2" >&2
  exit 2
fi
NORMALIZED_DATA=$(printf '%s' "$DATA_FILE" | tr '[:upper:]_' '[:lower:]-')
case "$NORMALIZED_DATA" in
  *test-v2*|*testv2*) echo "refusing test-v2 path" >&2; exit 2 ;;
esac

sha256sum "${FILES[@]:0:10}" > "$RUN_DIR/code.sha256"
sha256sum "$DATA_FILE" > "$RUN_DIR/validation-data.sha256"
sha256sum "$CHECKPOINT_DIR/checkpoint-manifest.json" \
  > "$RUN_DIR/checkpoint-manifest.sha256"
nvidia-smi --query-gpu=index,uuid,name,memory.total,driver_version,pstate,power.limit \
  --format=csv > "$RUN_DIR/gpus-before.csv"

"$ENV_DIR/bin/python" -m py_compile \
  "$CODE_DIR/run_sft_dcp_downstream.py" \
  "$CODE_DIR/aggregate_sft_dcp_downstream.py" \
  "$CODE_DIR/dcp_replicated_load_preflight.py"
bash -n "$CODE_DIR/launch_sft_dcp_downstream_8gpu.sh"
DATA_FILE="$DATA_FILE" EXPECTED_DATA_SHA256="$EXPECTED_DATA_SHA256" \
PYTHONPATH="$CODE_DIR" "$ENV_DIR/bin/python" - <<'PY' \
  > "$RUN_DIR/logs/validation-contract.json"
import os
from pathlib import Path
from sft_quality_validation import validate_longbench_validation_rows

rows, audit = validate_longbench_validation_rows(
    Path(os.environ["DATA_FILE"]), expected_sha256=os.environ["EXPECTED_DATA_SHA256"]
)
assert len(rows) == 60 and audit["raw_test_v2_read"] is False
print(audit)
PY
PYTHONPATH="$CODE_DIR" "$ENV_DIR/bin/torchrun" \
  --standalone --nproc-per-node=2 \
  "$CODE_DIR/dcp_replicated_load_preflight.py" \
  --output-root "$RUN_DIR/dcp-replicated-tiny-preflight" \
  > "$RUN_DIR/logs/dcp-replicated-tiny-preflight.log" 2>&1
date -u +%FT%TZ > "$RUN_DIR/stages/01_preflight_ok"

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ENABLE_MONITORING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200
PYTHONPATH="$CODE_DIR" "$ENV_DIR/bin/torchrun" \
  --standalone --nproc-per-node=8 \
  "$CODE_DIR/run_sft_dcp_downstream.py" \
  --model "$MODEL_DIR" \
  --checkpoint "$CHECKPOINT_DIR" \
  --expected-checkpoint-manifest-sha256 \
    "$EXPECTED_CHECKPOINT_MANIFEST_SHA256" \
  --data "$DATA_FILE" \
  --expected-data-sha256 "$EXPECTED_DATA_SHA256" \
  --run-dir "$RUN_DIR" \
  --max-input-tokens 4096 --max-new-tokens 128 \
  --chunk-size 512 --overlap 0 --group-size 64 \
  > "$RUN_DIR/logs/eval.log" 2>&1
date -u +%FT%TZ > "$RUN_DIR/stages/02_eval_done"

PYTHONPATH="$CODE_DIR" "$ENV_DIR/bin/python" \
  "$CODE_DIR/aggregate_sft_dcp_downstream.py" \
  --sft-run-dir "$RUN_DIR" \
  --interface-lora-run-dir "$INTERFACE_LORA_RUN_DIR" \
  --expected-data-sha256 "$EXPECTED_DATA_SHA256" \
  --expected-checkpoint-manifest-sha256 \
    "$EXPECTED_CHECKPOINT_MANIFEST_SHA256" \
  --expected-lora-checkpoint-sha256 "$EXPECTED_LORA_CHECKPOINT_SHA256" \
  --output "$RUN_DIR/unified-analysis.json" \
  > "$RUN_DIR/logs/aggregate.log" 2>&1
test -s "$RUN_DIR/unified-analysis.json"
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,pstate,power.draw \
  --format=csv > "$RUN_DIR/gpus-after.csv"
date -u +%FT%TZ > "$RUN_DIR/stages/99_done"
