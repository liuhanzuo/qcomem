#!/usr/bin/env bash
set -euo pipefail

CODE_DIR=${CODE_DIR:?set CODE_DIR}
MODEL_DIR=${MODEL_DIR:?set MODEL_DIR}
VALIDATION_DATA_FILE=${VALIDATION_DATA_FILE:?set VALIDATION_DATA_FILE}
SOURCE_RUN_DIR=${SOURCE_RUN_DIR:?set SOURCE_RUN_DIR}
RUN_DIR=${RUN_DIR:?set RUN_DIR}
ENV_DIR=${ENV_DIR:?set ENV_DIR}
EXPECTED_VALIDATION_SHA256=${EXPECTED_VALIDATION_SHA256:?set EXPECTED_VALIDATION_SHA256}
EXPECTED_STEP0_SHA256=${EXPECTED_STEP0_SHA256:?set EXPECTED_STEP0_SHA256}
EXPECTED_STEP64_SHA256=${EXPECTED_STEP64_SHA256:?set EXPECTED_STEP64_SHA256}
EXPECTED_STEP128_SHA256=${EXPECTED_STEP128_SHA256:?set EXPECTED_STEP128_SHA256}
EXPECTED_CODE_LEDGER_SHA256=${EXPECTED_CODE_LEDGER_SHA256:?set EXPECTED_CODE_LEDGER_SHA256}
SOURCE_JOB_ID=${SOURCE_JOB_ID:-235749}
SOURCE_TRIAL_ID=${SOURCE_TRIAL_ID:-1834056}

if [[ "$SOURCE_JOB_ID" != 235749 || "$SOURCE_TRIAL_ID" != 1834056 ]]; then
  echo "experiment A is bound to source Job/Trial 235749/1834056" >&2
  exit 2
fi
if [[ -e "$RUN_DIR" && -n "$(find "$RUN_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "refusing non-empty attribution run directory: $RUN_DIR" >&2
  exit 2
fi
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/stages"
trap 'date -u +%FT%TZ > "$RUN_DIR/stages/FAILED"' ERR
date -u +%FT%TZ > "$RUN_DIR/stages/00_start"

GPU_COUNT=$(nvidia-smi -L | wc -l | tr -d ' ')
if [[ "$GPU_COUNT" -ne 8 ]]; then
  echo "formal attribution requires exactly 8 GPUs, found $GPU_COUNT" >&2
  exit 2
fi
nvidia-smi \
  --query-gpu=index,uuid,name,memory.total,driver_version,pstate,power.limit \
  --format=csv > "$RUN_DIR/gpus-before.csv"

STEP0_FILE="$SOURCE_RUN_DIR/checkpoint-000000.pt"
STEP64_FILE="$SOURCE_RUN_DIR/checkpoint-000064.pt"
STEP128_FILE="$SOURCE_RUN_DIR/checkpoint-000128.pt"
test -s "$SOURCE_RUN_DIR/stages/99_done"

verify_sha() {
  local path=$1
  local expected=$2
  local label=$3
  local actual
  actual=$(sha256sum "$path" | awk '{print $1}')
  if [[ "$actual" != "$expected" ]]; then
    echo "$label SHA256 mismatch: expected=$expected actual=$actual" >&2
    exit 2
  fi
}
verify_sha "$VALIDATION_DATA_FILE" "$EXPECTED_VALIDATION_SHA256" validation
verify_sha "$STEP0_FILE" "$EXPECTED_STEP0_SHA256" step0
verify_sha "$STEP64_FILE" "$EXPECTED_STEP64_SHA256" step64
verify_sha "$STEP128_FILE" "$EXPECTED_STEP128_SHA256" step128
if [[ "$EXPECTED_VALIDATION_SHA256" == "fe046477bf5b39629e9f66fd4def7a55c2d5d1f073c8bb601ee3833f08eaaa5f" ]]; then
  echo "refusing frozen LongBench test-v2" >&2
  exit 2
fi
sha256sum "$VALIDATION_DATA_FILE" "$STEP0_FILE" "$STEP64_FILE" "$STEP128_FILE" \
  > "$RUN_DIR/input-artifacts.sha256"
printf '%s\t%s\n' "$SOURCE_JOB_ID" "$SOURCE_TRIAL_ID" \
  > "$RUN_DIR/source-job-trial.tsv"

CODE_FILES=(
  qcomem_torch.py qcomem_lora.py run_downstream.py run_replay_diagnostic.py
  analyze_validation.py aggregate_replay.py
  run_native_lora_checkpoint_attribution.py
  aggregate_native_lora_checkpoint_attribution.py
  test_native_lora_checkpoint_attribution.py
  launch_native_lora_checkpoint_attribution_8gpu.sh
)
PYTHON_FILES=(
  qcomem_torch.py qcomem_lora.py run_downstream.py run_replay_diagnostic.py
  analyze_validation.py aggregate_replay.py
  run_native_lora_checkpoint_attribution.py
  aggregate_native_lora_checkpoint_attribution.py
  test_native_lora_checkpoint_attribution.py
)
for file in "${CODE_FILES[@]}"; do
  test -s "$CODE_DIR/$file"
done
sha256sum "${CODE_FILES[@]/#/$CODE_DIR/}" > "$RUN_DIR/code.sha256"
CODE_LEDGER_SHA256=$(sha256sum "$RUN_DIR/code.sha256" | awk '{print $1}')
if [[ "$CODE_LEDGER_SHA256" != "$EXPECTED_CODE_LEDGER_SHA256" ]]; then
  echo "code ledger SHA256 mismatch: expected=$EXPECTED_CODE_LEDGER_SHA256 actual=$CODE_LEDGER_SHA256" >&2
  exit 2
fi
printf '%s  %s\n' "$CODE_LEDGER_SHA256" "$RUN_DIR/code.sha256" \
  > "$RUN_DIR/code-ledger.sha256"

"$ENV_DIR/bin/python" -m py_compile "${PYTHON_FILES[@]/#/$CODE_DIR/}"
PYTHONPATH="$CODE_DIR" "$ENV_DIR/bin/python" -m unittest \
  test_native_lora_checkpoint_attribution -v \
  > "$RUN_DIR/logs/preflight-tests.log" 2>&1
PYTHONPATH="$CODE_DIR" "$ENV_DIR/bin/python" \
  "$CODE_DIR/run_native_lora_checkpoint_attribution.py" \
  --model "$MODEL_DIR" --data "$VALIDATION_DATA_FILE" \
  --expected-data-sha256 "$EXPECTED_VALIDATION_SHA256" \
  --checkpoint "0=$STEP0_FILE=$EXPECTED_STEP0_SHA256" \
  --checkpoint "64=$STEP64_FILE=$EXPECTED_STEP64_SHA256" \
  --checkpoint "128=$STEP128_FILE=$EXPECTED_STEP128_SHA256" \
  --run-dir "$RUN_DIR" --preflight-only \
  > "$RUN_DIR/logs/protocol-preflight.log" 2>&1
date -u +%FT%TZ > "$RUN_DIR/stages/01_preflight_ok"

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
PIDS=()
for RANK in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$RANK PYTHONPATH="$CODE_DIR" "$ENV_DIR/bin/python" \
    "$CODE_DIR/run_native_lora_checkpoint_attribution.py" \
    --model "$MODEL_DIR" --data "$VALIDATION_DATA_FILE" \
    --expected-data-sha256 "$EXPECTED_VALIDATION_SHA256" \
    --checkpoint "0=$STEP0_FILE=$EXPECTED_STEP0_SHA256" \
    --checkpoint "64=$STEP64_FILE=$EXPECTED_STEP64_SHA256" \
    --checkpoint "128=$STEP128_FILE=$EXPECTED_STEP128_SHA256" \
    --run-dir "$RUN_DIR" --rank "$RANK" --world-size 8 \
    --max-input-tokens 4096 --max-new-tokens 128 --group-size 64 \
    > "$RUN_DIR/logs/attribution-rank-${RANK}.log" 2>&1 &
  PIDS+=("$!")
  sleep 3
done
FAILED=0
for INDEX in 0 1 2 3 4 5 6 7; do
  if ! wait "${PIDS[$INDEX]}"; then
    echo "attribution rank $INDEX failed" >&2
    FAILED=1
  fi
done
[[ "$FAILED" -eq 0 ]]
date -u +%FT%TZ > "$RUN_DIR/stages/02_shards_ok"

PYTHONPATH="$CODE_DIR" "$ENV_DIR/bin/python" \
  "$CODE_DIR/aggregate_native_lora_checkpoint_attribution.py" \
  --run-dir "$RUN_DIR" \
  --expected-data-sha256 "$EXPECTED_VALIDATION_SHA256" \
  --expected-step0-sha256 "$EXPECTED_STEP0_SHA256" \
  --expected-step64-sha256 "$EXPECTED_STEP64_SHA256" \
  --expected-step128-sha256 "$EXPECTED_STEP128_SHA256" \
  --source-job-id "$SOURCE_JOB_ID" --source-trial-id "$SOURCE_TRIAL_ID" \
  --bootstrap-seed 20260831 \
  > "$RUN_DIR/logs/aggregate.log" 2>&1
test -s "$RUN_DIR/checkpoint-attribution-analysis.json"

nvidia-smi \
  --query-gpu=index,uuid,name,memory.used,pstate,power.draw,temperature.gpu \
  --format=csv > "$RUN_DIR/gpus-after.csv"
date -u +%FT%TZ > "$RUN_DIR/stages/99_done"
echo "Native LoRA checkpoint attribution complete: $RUN_DIR"
