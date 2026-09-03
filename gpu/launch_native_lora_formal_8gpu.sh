#!/usr/bin/env bash
set -euo pipefail

CODE_DIR=${CODE_DIR:?set CODE_DIR}
MODEL_DIR=${MODEL_DIR:?set MODEL_DIR}
TRAIN_VIEW=${TRAIN_VIEW:?set TRAIN_VIEW}
HELDOUT_VIEW=${HELDOUT_VIEW:?set HELDOUT_VIEW}
VIEW_MANIFEST=${VIEW_MANIFEST:?set VIEW_MANIFEST}
PARENT_TRAIN=${PARENT_TRAIN:?set PARENT_TRAIN}
PARENT_HELDOUT=${PARENT_HELDOUT:?set PARENT_HELDOUT}
PARENT_MANIFEST=${PARENT_MANIFEST:?set PARENT_MANIFEST}
PARENT_AUDIT=${PARENT_AUDIT:?set PARENT_AUDIT}
INIT_ADAPTER_FILE=${INIT_ADAPTER_FILE:?set INIT_ADAPTER_FILE}
VALIDATION_DATA_FILE=${VALIDATION_DATA_FILE:?set VALIDATION_DATA_FILE}
RUN_DIR=${RUN_DIR:?set RUN_DIR}
ENV_DIR=${ENV_DIR:?set ENV_DIR}
CONFIG_FILE=${CONFIG_FILE:?set CONFIG_FILE}
EXPECTED_TRAIN_VIEW_SHA256=${EXPECTED_TRAIN_VIEW_SHA256:?set EXPECTED_TRAIN_VIEW_SHA256}
EXPECTED_HELDOUT_VIEW_SHA256=${EXPECTED_HELDOUT_VIEW_SHA256:?set EXPECTED_HELDOUT_VIEW_SHA256}
EXPECTED_VIEW_MANIFEST_SHA256=${EXPECTED_VIEW_MANIFEST_SHA256:?set EXPECTED_VIEW_MANIFEST_SHA256}
EXPECTED_PARENT_TRAIN_SHA256=${EXPECTED_PARENT_TRAIN_SHA256:?set EXPECTED_PARENT_TRAIN_SHA256}
EXPECTED_PARENT_HELDOUT_SHA256=${EXPECTED_PARENT_HELDOUT_SHA256:?set EXPECTED_PARENT_HELDOUT_SHA256}
EXPECTED_PARENT_MANIFEST_SHA256=${EXPECTED_PARENT_MANIFEST_SHA256:?set EXPECTED_PARENT_MANIFEST_SHA256}
EXPECTED_PARENT_AUDIT_SHA256=${EXPECTED_PARENT_AUDIT_SHA256:?set EXPECTED_PARENT_AUDIT_SHA256}
EXPECTED_INIT_ADAPTER_SHA256=${EXPECTED_INIT_ADAPTER_SHA256:?set EXPECTED_INIT_ADAPTER_SHA256}
EXPECTED_VALIDATION_SHA256=${EXPECTED_VALIDATION_SHA256:?set EXPECTED_VALIDATION_SHA256}
EXPECTED_CODE_LEDGER_SHA256=${EXPECTED_CODE_LEDGER_SHA256:?set EXPECTED_CODE_LEDGER_SHA256}
LIMIT_PER_DATASET=${LIMIT_PER_DATASET:-30}
MAX_INPUT_TOKENS=${MAX_INPUT_TOKENS:-4096}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-128}

if [[ -e "$RUN_DIR" && -n "$(find "$RUN_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "refusing non-empty native-LoRA run directory: $RUN_DIR" >&2
  exit 2
fi
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/stages"
trap 'date -u +%FT%TZ > "$RUN_DIR/stages/FAILED"' ERR
date -u +%FT%TZ > "$RUN_DIR/stages/00_start"

GPU_COUNT=$(nvidia-smi -L | wc -l | tr -d ' ')
if [[ "$GPU_COUNT" -ne 8 ]]; then
  echo "formal native LoRA requires exactly 8 GPUs, found $GPU_COUNT" >&2
  exit 2
fi
nvidia-smi \
  --query-gpu=index,uuid,name,memory.total,driver_version,pstate,power.limit \
  --format=csv > "$RUN_DIR/gpus-before.csv"

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
verify_sha "$TRAIN_VIEW" "$EXPECTED_TRAIN_VIEW_SHA256" train-view
verify_sha "$HELDOUT_VIEW" "$EXPECTED_HELDOUT_VIEW_SHA256" heldout-view
verify_sha "$VIEW_MANIFEST" "$EXPECTED_VIEW_MANIFEST_SHA256" view-manifest
verify_sha "$PARENT_TRAIN" "$EXPECTED_PARENT_TRAIN_SHA256" parent-train
verify_sha "$PARENT_HELDOUT" "$EXPECTED_PARENT_HELDOUT_SHA256" parent-heldout
verify_sha "$PARENT_MANIFEST" "$EXPECTED_PARENT_MANIFEST_SHA256" parent-manifest
verify_sha "$PARENT_AUDIT" "$EXPECTED_PARENT_AUDIT_SHA256" parent-audit
verify_sha "$INIT_ADAPTER_FILE" "$EXPECTED_INIT_ADAPTER_SHA256" init-adapter
verify_sha "$VALIDATION_DATA_FILE" "$EXPECTED_VALIDATION_SHA256" validation
if [[ "$EXPECTED_VALIDATION_SHA256" == "fe046477bf5b39629e9f66fd4def7a55c2d5d1f073c8bb601ee3833f08eaaa5f" ]]; then
  echo "refusing frozen LongBench test-v2" >&2
  exit 2
fi
sha256sum "$TRAIN_VIEW" "$HELDOUT_VIEW" "$VIEW_MANIFEST" \
  "$PARENT_TRAIN" "$PARENT_HELDOUT" "$PARENT_MANIFEST" "$PARENT_AUDIT" \
  "$INIT_ADAPTER_FILE" "$VALIDATION_DATA_FILE" > "$RUN_DIR/input-artifacts.sha256"

CODE_FILES=(
  qcomem_torch.py qcomem_lora.py qcomem_qwen35_native_cache.py
  train_qcomem_lora.py train_native_lora_formal.py
  qcomem_native_lora_protocol.py build_native_lora_domain_view.py
  run_native_lora_heldout.py aggregate_native_lora_heldout.py
  run_native_lora_semantic_gate.py aggregate_native_lora_semantic_gate.py
  run_replay_diagnostic.py aggregate_replay.py run_downstream.py
  analyze_validation.py test_qcomem_lora.py
  test_qcomem_qwen35_native_cache.py test_qcomem_native_lora_protocol.py
  launch_native_lora_formal_8gpu.sh lora_quant_native_domain_128.json
)
PYTHON_FILES=(
  qcomem_torch.py qcomem_lora.py qcomem_qwen35_native_cache.py
  train_qcomem_lora.py train_native_lora_formal.py
  qcomem_native_lora_protocol.py build_native_lora_domain_view.py
  run_native_lora_heldout.py aggregate_native_lora_heldout.py
  run_native_lora_semantic_gate.py aggregate_native_lora_semantic_gate.py
  run_replay_diagnostic.py aggregate_replay.py run_downstream.py
  analyze_validation.py test_qcomem_lora.py
  test_qcomem_qwen35_native_cache.py test_qcomem_native_lora_protocol.py
)
for file in "${CODE_FILES[@]}"; do
  test -s "$CODE_DIR/$file"
done
sha256sum "${CODE_FILES[@]/#/$CODE_DIR/}" > "$RUN_DIR/code.sha256"
CODE_LEDGER_SHA256=$(sha256sum "$RUN_DIR/code.sha256" | awk '{print $1}')
if [[ "$CODE_LEDGER_SHA256" != "$EXPECTED_CODE_LEDGER_SHA256" ]]; then
  echo "code ledger SHA256 mismatch" >&2
  exit 2
fi
printf '%s  %s\n' "$CODE_LEDGER_SHA256" "$RUN_DIR/code.sha256" \
  > "$RUN_DIR/code-ledger.sha256"

"$ENV_DIR/bin/python" -m py_compile "${PYTHON_FILES[@]/#/$CODE_DIR/}"
PYTHONPATH="$CODE_DIR" "$ENV_DIR/bin/python" -m unittest \
  test_qcomem_native_lora_protocol test_qcomem_qwen35_native_cache \
  test_qcomem_lora -v > "$RUN_DIR/logs/preflight-tests.log" 2>&1
PYTHONPATH="$CODE_DIR" "$ENV_DIR/bin/python" - "$CONFIG_FILE" \
  "$VIEW_MANIFEST" <<'PY' > "$RUN_DIR/logs/protocol-preflight.log"
import json
import sys

config = json.load(open(sys.argv[1]))
manifest = json.load(open(sys.argv[2]))
assert config["mode"] == "quant"
assert config["depth"] == 7
assert config["teacher_kind"] == "q16_replay"
assert config["student_suffix_execution"] == "native-functional-cache"
assert config["context_tokens"] == 1536
assert config["query_tokens"] == 512
assert config["dataset_limit"] == 410
assert config["residual_bits"] == 4
assert config["attention_bits"] == 4
assert config["linear_bits"] == 8
assert config["cache_layer_bits"] == [8, 8, 8, 4, 8, 8, 8]
assert config["steps"] == 128
assert config["learning_rate"] == 2e-5
assert config["warmup_steps"] == 8
assert config["save_every"] == 64
assert manifest["status"] == "passed"
assert manifest["outputs"]["train"]["summary"]["rows"] == 410
assert manifest["outputs"]["heldout"]["summary"]["rows"] == 26
assert manifest["view_contract"]["included_strata"] == ["domain"]
assert manifest["view_contract"]["query_truncation"] == "forbidden_fail_closed"
assert manifest["governance"]["test_v2_used"] is False
print({"status": "passed", "single_token_autograd_claimed": False})
PY
date -u +%FT%TZ > "$RUN_DIR/stages/01_preflight_ok"

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export TOKENIZERS_PARALLELISM=false
export NATIVE_LORA_STEP1_GATE_FILE="$RUN_DIR/native-step1-hard-gate.json"
export NATIVE_LORA_STEP0_CHECKPOINT="$RUN_DIR/checkpoint-000000.pt"
export NATIVE_LORA_EXPECTED_MODULES=36
export NATIVE_LORA_EXPECTED_PARAMETER_TENSORS=72
export NATIVE_LORA_MINIMUM_HEADROOM_BYTES=4294967296
export EXPECTED_INIT_ADAPTER_SHA256
"$ENV_DIR/bin/torchrun" --standalone --nproc-per-node=8 \
  "$CODE_DIR/train_native_lora_formal.py" \
  --config "$CONFIG_FILE" \
  --model "$MODEL_DIR" \
  --data "$TRAIN_VIEW" \
  --output-dir "$RUN_DIR" \
  --init-adapter "$INIT_ADAPTER_FILE" \
  > "$RUN_DIR/logs/train.log" 2>&1

test -s "$RUN_DIR/native-step1-hard-gate.json"
test -s "$RUN_DIR/checkpoint-000000.pt"
test -s "$RUN_DIR/checkpoint-000064.pt"
test -s "$RUN_DIR/checkpoint-000128.pt"
"$ENV_DIR/bin/python" - "$RUN_DIR/native-step1-hard-gate.json" <<'PY'
import json, sys
gate = json.load(open(sys.argv[1]))
assert gate["status"] == "passed"
assert gate["checks"]["all_adapter_gradients_finite_nonzero"]
assert gate["checks"]["all_adapter_updates_finite_nonzero"]
assert gate["checks"]["metadata_native_cache_gate"]
assert gate["checks"]["memory_headroom"]
assert gate["single_token_autograd_claimed"] is False
PY
sha256sum "$RUN_DIR"/checkpoint-000{000,064,128}.pt \
  > "$RUN_DIR/checkpoints.sha256"
date -u +%FT%TZ > "$RUN_DIR/stages/02_train_step128_ok"

STEP0_SHA=$(sha256sum "$RUN_DIR/checkpoint-000000.pt" | awk '{print $1}')
STEP64_SHA=$(sha256sum "$RUN_DIR/checkpoint-000064.pt" | awk '{print $1}')
STEP128_SHA=$(sha256sum "$RUN_DIR/checkpoint-000128.pt" | awk '{print $1}')
PIDS=()
for RANK in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$RANK "$ENV_DIR/bin/python" \
    "$CODE_DIR/run_native_lora_heldout.py" \
    --model "$MODEL_DIR" --data "$HELDOUT_VIEW" \
    --expected-data-sha256 "$EXPECTED_HELDOUT_VIEW_SHA256" \
    --checkpoint "0=$RUN_DIR/checkpoint-000000.pt=$STEP0_SHA" \
    --checkpoint "64=$RUN_DIR/checkpoint-000064.pt=$STEP64_SHA" \
    --checkpoint "128=$RUN_DIR/checkpoint-000128.pt=$STEP128_SHA" \
    --output "$RUN_DIR/heldout-rank-${RANK}.json" \
    --rank "$RANK" --world-size 8 \
    > "$RUN_DIR/logs/heldout-rank-${RANK}.log" 2>&1 &
  PIDS+=("$!")
  sleep 5
done
FAILED=0
for INDEX in 0 1 2 3 4 5 6 7; do
  if ! wait "${PIDS[$INDEX]}"; then
    echo "heldout rank $INDEX failed" >&2
    FAILED=1
  fi
done
[[ "$FAILED" -eq 0 ]]
"$ENV_DIR/bin/python" "$CODE_DIR/aggregate_native_lora_heldout.py" \
  "$RUN_DIR" --expected-world-size 8 --expected-examples 26 \
  --expected-data-sha256 "$EXPECTED_HELDOUT_VIEW_SHA256" \
  > "$RUN_DIR/logs/heldout-aggregate.log" 2>&1
test -s "$RUN_DIR/heldout-selection.json"
BEST_CHECKPOINT=$(tr -d '\n' < "$RUN_DIR/best-checkpoint.path")
BEST_CHECKPOINT_SHA256=$(tr -d '\n' < "$RUN_DIR/best-checkpoint.sha256")
verify_sha "$BEST_CHECKPOINT" "$BEST_CHECKPOINT_SHA256" selected-checkpoint
date -u +%FT%TZ > "$RUN_DIR/stages/03_heldout_selected"

PIDS=()
for RANK in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$RANK "$ENV_DIR/bin/python" \
    "$CODE_DIR/run_native_lora_semantic_gate.py" \
    --model "$MODEL_DIR" --data "$HELDOUT_VIEW" \
    --expected-data-sha256 "$EXPECTED_HELDOUT_VIEW_SHA256" \
    --checkpoint "$BEST_CHECKPOINT" \
    --expected-checkpoint-sha256 "$BEST_CHECKPOINT_SHA256" \
    --output "$RUN_DIR/native-semantic-rank-${RANK}.json" \
    --samples 16 --projection-block-size 16 \
    --rank "$RANK" --world-size 8 \
    > "$RUN_DIR/logs/native-semantic-rank-${RANK}.log" 2>&1 &
  PIDS+=("$!")
  sleep 5
done
FAILED=0
for INDEX in 0 1 2 3 4 5 6 7; do
  if ! wait "${PIDS[$INDEX]}"; then
    echo "native semantic rank $INDEX failed" >&2
    FAILED=1
  fi
done
[[ "$FAILED" -eq 0 ]]
"$ENV_DIR/bin/python" "$CODE_DIR/aggregate_native_lora_semantic_gate.py" \
  "$RUN_DIR" --expected-world-size 8 \
  --expected-data-sha256 "$EXPECTED_HELDOUT_VIEW_SHA256" \
  --expected-checkpoint-sha256 "$BEST_CHECKPOINT_SHA256" \
  > "$RUN_DIR/logs/native-semantic-aggregate.log" 2>&1
test -s "$RUN_DIR/native-semantic-gate.json"
date -u +%FT%TZ > "$RUN_DIR/stages/04_semantic_gate_ok"

# This stage is reached only if the target-semantic heldout gate passed.  The
# fixed source-index 6--35 validation is never training data and test-v2 is
# never read.  LoRA is enabled only on the frozen Q4 store; the matching
# untrained Q4, Q16 and dense rows are paired controls.
PIDS=()
for RANK in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$RANK "$ENV_DIR/bin/python" \
    "$CODE_DIR/run_replay_diagnostic.py" \
    --model "$MODEL_DIR" --data "$VALIDATION_DATA_FILE" \
    --run-dir "$RUN_DIR" --rank "$RANK" --world-size 8 \
    --suite quant-lora-validation \
    --source-index-start 6 --source-index-end 35 \
    --exclude-source-indices 4,5 \
    --limit-per-dataset "$LIMIT_PER_DATASET" \
    --max-input-tokens "$MAX_INPUT_TOKENS" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --lora-checkpoint "$BEST_CHECKPOINT" \
    --lora-apply-to-configs replay-d7-frozen-static-lora \
    > "$RUN_DIR/logs/downstream-rank-${RANK}.log" 2>&1 &
  PIDS+=("$!")
  sleep 5
done
FAILED=0
for INDEX in 0 1 2 3 4 5 6 7; do
  if ! wait "${PIDS[$INDEX]}"; then
    echo "downstream rank $INDEX failed" >&2
    FAILED=1
  fi
done
[[ "$FAILED" -eq 0 ]]
date -u +%FT%TZ > "$RUN_DIR/stages/05_downstream_shards_ok"
"$ENV_DIR/bin/python" "$CODE_DIR/aggregate_replay.py" "$RUN_DIR" \
  --suite quant-lora-validation --expected-world-size 8 \
  --overall-margin -0.02 --dataset-margin -0.03 --catastrophic-delta -0.5 \
  --expected-data-sha256 "$EXPECTED_VALIDATION_SHA256" \
  --expected-checkpoint-sha256 "$BEST_CHECKPOINT_SHA256" \
  > "$RUN_DIR/logs/downstream-aggregate.log" 2>&1
test -s "$RUN_DIR/replay_analysis.json"

nvidia-smi \
  --query-gpu=index,uuid,name,memory.used,pstate,power.draw,temperature.gpu \
  --format=csv > "$RUN_DIR/gpus-after.csv"
date -u +%FT%TZ > "$RUN_DIR/stages/99_done"
echo "Native-functional-cache domain LoRA formal run complete: $RUN_DIR"
