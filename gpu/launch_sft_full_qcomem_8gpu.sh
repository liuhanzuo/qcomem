#!/usr/bin/env bash
set -euo pipefail

CODE_DIR=${CODE_DIR:?set CODE_DIR}
MODEL_DIR=${MODEL_DIR:?set MODEL_DIR}
MODEL_WEIGHT_LEDGER_FILE=${MODEL_WEIGHT_LEDGER_FILE:?set MODEL_WEIGHT_LEDGER_FILE}
EXPECTED_MODEL_WEIGHT_LEDGER_SHA256=${EXPECTED_MODEL_WEIGHT_LEDGER_SHA256:?set EXPECTED_MODEL_WEIGHT_LEDGER_SHA256}
EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256=${EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256:?set EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256}
EXPECTED_CODE_LEDGER_SHA256=${EXPECTED_CODE_LEDGER_SHA256:?set EXPECTED_CODE_LEDGER_SHA256}
CHECKPOINT_DIR=${CHECKPOINT_DIR:?set CHECKPOINT_DIR}
EXPECTED_CHECKPOINT_MANIFEST_SHA256=${EXPECTED_CHECKPOINT_MANIFEST_SHA256:?set EXPECTED_CHECKPOINT_MANIFEST_SHA256}
DATA_FILE=${DATA_FILE:?set DATA_FILE}
EXPECTED_DATA_SHA256=${EXPECTED_DATA_SHA256:?set EXPECTED_DATA_SHA256}
RUN_DIR=${RUN_DIR:?set RUN_DIR}
ENV_DIR=${ENV_DIR:?set ENV_DIR}

for VALUE in \
  "$EXPECTED_MODEL_WEIGHT_LEDGER_SHA256" \
  "$EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256" \
  "$EXPECTED_CODE_LEDGER_SHA256" \
  "$EXPECTED_CHECKPOINT_MANIFEST_SHA256" \
  "$EXPECTED_DATA_SHA256"; do
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

CODE_FILES=(
  "$CODE_DIR/run_sft_full_qcomem_downstream.py"
  "$CODE_DIR/aggregate_sft_full_qcomem.py"
  "$CODE_DIR/test_sft_full_qcomem_downstream.py"
  "$CODE_DIR/launch_sft_full_qcomem_8gpu.sh"
  "$CODE_DIR/dcp_replicated_load_preflight.py"
  "$CODE_DIR/sft_dcp_checkpoint.py"
  "$CODE_DIR/sft_quality_validation.py"
  "$CODE_DIR/supervised_sft.py"
  "$CODE_DIR/run_replay_diagnostic.py"
  "$CODE_DIR/run_downstream.py"
  "$CODE_DIR/qcomem_torch.py"
  "$CODE_DIR/analyze_validation.py"
)
MODEL_ARTIFACT_FILES=(
  "$MODEL_DIR/config.json"
  "$MODEL_DIR/model.safetensors.index.json"
  "$MODEL_DIR/tokenizer_config.json"
  "$MODEL_DIR/vocab.json"
  "$MODEL_DIR/merges.txt"
  "$MODEL_DIR/chat_template.jinja"
)
REQUIRED_FILES=(
  "${CODE_FILES[@]}"
  "${MODEL_ARTIFACT_FILES[@]}"
  "$MODEL_WEIGHT_LEDGER_FILE"
  "$CHECKPOINT_DIR/checkpoint-manifest.json"
  "$CHECKPOINT_DIR/_SUCCESS"
  "$DATA_FILE"
)
for FILE in "${REQUIRED_FILES[@]}"; do
  test -f "$FILE" || { echo "missing required artifact: $FILE" >&2; exit 2; }
done
test "$(nvidia-smi -L | wc -l | tr -d ' ')" -eq 8

sha256sum "${CODE_FILES[@]}" > "$RUN_DIR/code.sha256"
sha256sum "${MODEL_ARTIFACT_FILES[@]}" > "$RUN_DIR/model-artifacts.sha256"
cp "$MODEL_WEIGHT_LEDGER_FILE" "$RUN_DIR/model-weights.sha256"
if [ "$(sha256sum "$RUN_DIR/code.sha256" | awk '{print $1}')" != "$EXPECTED_CODE_LEDGER_SHA256" ]; then
  echo "code ledger SHA256 mismatch" >&2
  exit 2
fi
if [ "$(sha256sum "$RUN_DIR/model-artifacts.sha256" | awk '{print $1}')" != "$EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256" ]; then
  echo "model artifact ledger SHA256 mismatch" >&2
  exit 2
fi
if [ "$(sha256sum "$RUN_DIR/model-weights.sha256" | awk '{print $1}')" != "$EXPECTED_MODEL_WEIGHT_LEDGER_SHA256" ]; then
  echo "model weight ledger SHA256 mismatch" >&2
  exit 2
fi
sha256sum -c "$RUN_DIR/code.sha256" > "$RUN_DIR/logs/code-integrity.log"
sha256sum -c "$RUN_DIR/model-artifacts.sha256" > "$RUN_DIR/logs/model-artifact-integrity.log"
sha256sum -c "$RUN_DIR/model-weights.sha256" > "$RUN_DIR/logs/model-weight-integrity.log"
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
sha256sum "$DATA_FILE" > "$RUN_DIR/validation-data.sha256"
sha256sum "$CHECKPOINT_DIR/checkpoint-manifest.json" > "$RUN_DIR/checkpoint-manifest.sha256"
nvidia-smi --query-gpu=index,uuid,name,memory.total,driver_version,pstate,power.limit \
  --format=csv > "$RUN_DIR/gpus-before.csv"

"$ENV_DIR/bin/python" -m py_compile "${CODE_FILES[@]:0:3}"
bash -n "$CODE_DIR/launch_sft_full_qcomem_8gpu.sh"
PYTHONPATH="$CODE_DIR" "$ENV_DIR/bin/python" -m unittest -v \
  test_sft_full_qcomem_downstream \
  > "$RUN_DIR/logs/unit-tests.log" 2>&1
DATA_FILE="$DATA_FILE" EXPECTED_DATA_SHA256="$EXPECTED_DATA_SHA256" \
PYTHONPATH="$CODE_DIR" "$ENV_DIR/bin/python" - <<'PY' \
  > "$RUN_DIR/logs/validation-contract.json"
import json
import os
from pathlib import Path
from run_sft_full_qcomem_downstream import load_frozen_validation_slice

rows, audit = load_frozen_validation_slice(
    Path(os.environ["DATA_FILE"]), expected_sha256=os.environ["EXPECTED_DATA_SHA256"]
)
assert len(rows) == 60
assert audit["raw_test_v2_read"] is False
print(json.dumps(audit, sort_keys=True))
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
  "$CODE_DIR/run_sft_full_qcomem_downstream.py" \
  --model "$MODEL_DIR" \
  --checkpoint "$CHECKPOINT_DIR" \
  --expected-checkpoint-manifest-sha256 "$EXPECTED_CHECKPOINT_MANIFEST_SHA256" \
  --data "$DATA_FILE" \
  --expected-data-sha256 "$EXPECTED_DATA_SHA256" \
  --run-dir "$RUN_DIR" \
  --max-input-tokens 4096 --max-new-tokens 128 --group-size 64 \
  > "$RUN_DIR/logs/eval.log" 2>&1
date -u +%FT%TZ > "$RUN_DIR/stages/02_eval_done"

PYTHONPATH="$CODE_DIR" "$ENV_DIR/bin/python" \
  "$CODE_DIR/aggregate_sft_full_qcomem.py" \
  --run-dir "$RUN_DIR" \
  --expected-data-sha256 "$EXPECTED_DATA_SHA256" \
  --expected-checkpoint-manifest-sha256 "$EXPECTED_CHECKPOINT_MANIFEST_SHA256" \
  --output "$RUN_DIR/analysis.json" \
  > "$RUN_DIR/logs/aggregate.log" 2>&1
test -s "$RUN_DIR/analysis.json"
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,pstate,power.draw \
  --format=csv > "$RUN_DIR/gpus-after.csv"
date -u +%FT%TZ > "$RUN_DIR/stages/99_done"
echo "SFT + full-state Q-CoMem downstream evaluation complete: $RUN_DIR"
