#!/usr/bin/env bash
set -euo pipefail

CODE_DIR=${CODE_DIR:?set CODE_DIR}
MODEL_DIR=${MODEL_DIR:?set MODEL_DIR}
TRAIN_DATA_FILE=${TRAIN_DATA_FILE:?set TRAIN_DATA_FILE}
HELDOUT_DATA_FILE=${HELDOUT_DATA_FILE:?set HELDOUT_DATA_FILE}
VALIDATION_DATA_FILE=${VALIDATION_DATA_FILE:?set VALIDATION_DATA_FILE}
DATA_MANIFEST_FILE=${DATA_MANIFEST_FILE:?set DATA_MANIFEST_FILE}
INDEPENDENT_AUDIT_FILE=${INDEPENDENT_AUDIT_FILE:?set INDEPENDENT_AUDIT_FILE}
INIT_ADAPTER_FILE=${INIT_ADAPTER_FILE:?set INIT_ADAPTER_FILE}
MODEL_WEIGHT_LEDGER_FILE=${MODEL_WEIGHT_LEDGER_FILE:?set MODEL_WEIGHT_LEDGER_FILE}
CONFIG_FILE=${CONFIG_FILE:?set CONFIG_FILE}
RUN_DIR=${RUN_DIR:?set RUN_DIR}
ENV_DIR=${ENV_DIR:?set ENV_DIR}
EXPECTED_TRAIN_SHA256=${EXPECTED_TRAIN_SHA256:?set EXPECTED_TRAIN_SHA256}
EXPECTED_HELDOUT_SHA256=${EXPECTED_HELDOUT_SHA256:?set EXPECTED_HELDOUT_SHA256}
EXPECTED_VALIDATION_SHA256=${EXPECTED_VALIDATION_SHA256:?set EXPECTED_VALIDATION_SHA256}
EXPECTED_DATA_MANIFEST_SHA256=${EXPECTED_DATA_MANIFEST_SHA256:?set EXPECTED_DATA_MANIFEST_SHA256}
EXPECTED_INDEPENDENT_AUDIT_SHA256=${EXPECTED_INDEPENDENT_AUDIT_SHA256:?set EXPECTED_INDEPENDENT_AUDIT_SHA256}
EXPECTED_INIT_ADAPTER_SHA256=${EXPECTED_INIT_ADAPTER_SHA256:?set EXPECTED_INIT_ADAPTER_SHA256}
EXPECTED_CODE_LEDGER_SHA256=${EXPECTED_CODE_LEDGER_SHA256:?set EXPECTED_CODE_LEDGER_SHA256}
EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256=${EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256:?set EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256}
EXPECTED_MODEL_WEIGHT_LEDGER_SHA256=${EXPECTED_MODEL_WEIGHT_LEDGER_SHA256:?set EXPECTED_MODEL_WEIGHT_LEDGER_SHA256}

for VALUE in \
  "$EXPECTED_TRAIN_SHA256" "$EXPECTED_HELDOUT_SHA256" \
  "$EXPECTED_VALIDATION_SHA256" \
  "$EXPECTED_DATA_MANIFEST_SHA256" "$EXPECTED_INDEPENDENT_AUDIT_SHA256" \
  "$EXPECTED_INIT_ADAPTER_SHA256" "$EXPECTED_CODE_LEDGER_SHA256" \
  "$EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256" \
  "$EXPECTED_MODEL_WEIGHT_LEDGER_SHA256"; do
  if [[ ! "$VALUE" =~ ^[0-9a-f]{64}$ ]]; then
    echo "every frozen digest must be one lowercase SHA256" >&2
    exit 2
  fi
done
if [[ "$EXPECTED_TRAIN_SHA256" == fe046477bf5b39629e9f66fd4def7a55c2d5d1f073c8bb601ee3833f08eaaa5f ]]; then
  echo "refusing LongBench test-v2" >&2
  exit 2
fi
if [[ "$EXPECTED_VALIDATION_SHA256" == fe046477bf5b39629e9f66fd4def7a55c2d5d1f073c8bb601ee3833f08eaaa5f ]]; then
  echo "refusing LongBench test-v2 as downstream input" >&2
  exit 2
fi
for PATH_VALUE in "$TRAIN_DATA_FILE" "$HELDOUT_DATA_FILE" "$DATA_MANIFEST_FILE"; do
  NORMALIZED=$(printf '%s' "$PATH_VALUE" | tr '[:upper:]_' '[:lower:]-')
  if [[ "$NORMALIZED" == *longbench* || "$NORMALIZED" == *test-v2* ]]; then
    echo "formal training inputs must be official-train derivatives only" >&2
    exit 2
  fi
done
if [[ -e "$RUN_DIR" && -n "$(find "$RUN_DIR" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "RUN_DIR must be absent or empty: $RUN_DIR" >&2
  exit 2
fi
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/stages" "$RUN_DIR/pycache"
export PYTHONPYCACHEPREFIX="$RUN_DIR/pycache"
CURRENT_PHASE=preflight
on_error() {
  date -u +%FT%TZ > "$RUN_DIR/stages/FAILED"
  printf '%s\n' "$CURRENT_PHASE" > "$RUN_DIR/stages/FAILED_PHASE"
  date -u +%FT%TZ > "$RUN_DIR/stages/FAILED_${CURRENT_PHASE}"
}
trap on_error ERR
date -u +%FT%TZ > "$RUN_DIR/stages/00_start"

verify_sha() {
  local path=$1 expected=$2 label=$3 actual
  actual=$(sha256sum "$path" | awk '{print $1}')
  if [[ "$actual" != "$expected" ]]; then
    echo "$label SHA256 mismatch: expected=$expected actual=$actual" >&2
    exit 2
  fi
}
verify_sha "$TRAIN_DATA_FILE" "$EXPECTED_TRAIN_SHA256" train
verify_sha "$HELDOUT_DATA_FILE" "$EXPECTED_HELDOUT_SHA256" heldout
verify_sha "$DATA_MANIFEST_FILE" "$EXPECTED_DATA_MANIFEST_SHA256" data-manifest
verify_sha "$INDEPENDENT_AUDIT_FILE" "$EXPECTED_INDEPENDENT_AUDIT_SHA256" independent-audit
verify_sha "$INIT_ADAPTER_FILE" "$EXPECTED_INIT_ADAPTER_SHA256" init-adapter

CODE_FILES=(
  "$CODE_DIR/qcomem_answer_supervised_lora.py"
  "$CODE_DIR/train_answer_supervised_native_lora.py"
  "$CODE_DIR/test_answer_supervised_native_lora.py"
  "$CODE_DIR/run_answer_lora_full_state_downstream.py"
  "$CODE_DIR/aggregate_answer_lora_full_state_downstream.py"
  "$CODE_DIR/test_answer_lora_full_state_downstream.py"
  "$CODE_DIR/launch_answer_supervised_native_lora_8gpu.sh"
  "$CONFIG_FILE"
  "$CODE_DIR/deployment_aware_sft.py"
  "$CODE_DIR/qcomem_lora.py"
  "$CODE_DIR/qcomem_torch.py"
  "$CODE_DIR/qcomem_qwen35_native_cache.py"
  "$CODE_DIR/supervised_sft.py"
  "$CODE_DIR/run_downstream.py"
  "$CODE_DIR/run_replay_diagnostic.py"
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
for ARTIFACT in "${CODE_FILES[@]}" "${MODEL_ARTIFACT_FILES[@]}" \
  "$MODEL_WEIGHT_LEDGER_FILE"; do
  test -s "$ARTIFACT"
done
sha256sum "${CODE_FILES[@]}" > "$RUN_DIR/code.sha256"
sha256sum "${MODEL_ARTIFACT_FILES[@]}" > "$RUN_DIR/model-artifacts.sha256"
cp "$MODEL_WEIGHT_LEDGER_FILE" "$RUN_DIR/model-weights.sha256"
verify_sha "$RUN_DIR/code.sha256" "$EXPECTED_CODE_LEDGER_SHA256" code-ledger
verify_sha "$RUN_DIR/model-artifacts.sha256" \
  "$EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256" model-artifact-ledger
verify_sha "$RUN_DIR/model-weights.sha256" \
  "$EXPECTED_MODEL_WEIGHT_LEDGER_SHA256" model-weight-ledger
sha256sum -c "$RUN_DIR/code.sha256" > "$RUN_DIR/logs/code-integrity.log"
sha256sum -c "$RUN_DIR/model-artifacts.sha256" \
  > "$RUN_DIR/logs/model-artifact-integrity.log"
# Read every large shard exactly once before torchrun. The trainer pins this
# verified ledger and paths without eight redundant full rehashes.
sha256sum -c "$RUN_DIR/model-weights.sha256" \
  > "$RUN_DIR/logs/model-weight-integrity.log"
sha256sum "$TRAIN_DATA_FILE" "$HELDOUT_DATA_FILE" "$DATA_MANIFEST_FILE" \
  "$INDEPENDENT_AUDIT_FILE" "$INIT_ADAPTER_FILE" \
  > "$RUN_DIR/input-artifacts.sha256"

GPU_COUNT=$(nvidia-smi -L | wc -l | tr -d ' ')
if [[ "$GPU_COUNT" -ne 8 ]]; then
  echo "formal answer LoRA requires exactly eight GPUs, found $GPU_COUNT" >&2
  exit 2
fi
nvidia-smi \
  --query-gpu=index,uuid,name,memory.total,driver_version,pstate,power.limit \
  --format=csv > "$RUN_DIR/gpus-before.csv"

"$ENV_DIR/bin/python" -m py_compile \
  "$CODE_DIR/qcomem_answer_supervised_lora.py" \
  "$CODE_DIR/train_answer_supervised_native_lora.py" \
  "$CODE_DIR/test_answer_supervised_native_lora.py" \
  "$CODE_DIR/run_answer_lora_full_state_downstream.py" \
  "$CODE_DIR/aggregate_answer_lora_full_state_downstream.py" \
  "$CODE_DIR/test_answer_lora_full_state_downstream.py"
bash -n "$CODE_DIR/launch_answer_supervised_native_lora_8gpu.sh"
PYTHONPATH="$CODE_DIR" "$ENV_DIR/bin/python" -m unittest \
  test_answer_supervised_native_lora \
  test_answer_lora_full_state_downstream -v \
  > "$RUN_DIR/logs/focused-tests.log" 2>&1
PYTHONPATH="$CODE_DIR" "$ENV_DIR/bin/python" - \
  "$CONFIG_FILE" "$TRAIN_DATA_FILE" "$HELDOUT_DATA_FILE" <<'PY' \
  > "$RUN_DIR/logs/protocol-preflight.json"
import collections, json, sys
config=json.load(open(sys.argv[1]))
assert config == {
  "steps":128,"depth":7,"learning_rate":0.00002,"warmup_steps":8,
  "weight_decay":0.0,"max_grad_norm":1.0,"seed":20260814,
  "teacher_topk":32,"teacher_projection_chunk_tokens":32,
  "student_projection_chunk_positions":32,
  "hard_weight":0.45,"kl_weight":0.35,"hidden_weight":0.20,
  "lora_rank":32,"lora_alpha":64.0,"lora_dropout":0.0,
  "residual_bits":4,"attention_bits":4,"linear_bits":8,
  "cache_layer_bits":"8,8,8,4,8,8,8","group_size":64,
  "max_adapter_parameters":27000000,
  "minimum_step1_headroom_bytes":4294967296,
}
counts=[]
positions=0
for path in sys.argv[2:]:
 rows=[json.loads(line) for line in open(path) if line.strip()]
 domain=[row for row in rows if row["stratum"]=="domain"]
 counts.append(dict(collections.Counter(row["dataset"] for row in domain)))
 positions += sum(row["token_counts"]["target"] for row in domain)
 assert all(row["deployment_boundary"]["applicable"] is True for row in domain)
 assert all(row["deployment_boundary"]["answer_or_eos_tokens_in_query"] is False for row in domain)
assert counts == [{"qasper":256,"2wikimqa":154},{"qasper":12,"2wikimqa":14}]
assert positions == 5992
print(json.dumps({"status":"passed","domain_counts":counts,"teacher_positions":positions}))
PY
date -u +%FT%TZ > "$RUN_DIR/stages/01_preflight_ok"

# Recheck mutable small artifacts immediately before the expensive formal run.
sha256sum -c "$RUN_DIR/code.sha256" >> "$RUN_DIR/logs/code-integrity.log"
sha256sum -c "$RUN_DIR/model-artifacts.sha256" \
  >> "$RUN_DIR/logs/model-artifact-integrity.log"
verify_sha "$TRAIN_DATA_FILE" "$EXPECTED_TRAIN_SHA256" train
verify_sha "$HELDOUT_DATA_FILE" "$EXPECTED_HELDOUT_SHA256" heldout
verify_sha "$INIT_ADAPTER_FILE" "$EXPECTED_INIT_ADAPTER_SHA256" init-adapter

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ENABLE_MONITORING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200
CURRENT_PHASE=training
PYTHONPATH="$CODE_DIR" "$ENV_DIR/bin/torchrun" \
  --standalone --nproc-per-node=8 \
  "$CODE_DIR/train_answer_supervised_native_lora.py" \
  --config "$CONFIG_FILE" \
  --model "$MODEL_DIR" \
  --train-data "$TRAIN_DATA_FILE" \
  --heldout-data "$HELDOUT_DATA_FILE" \
  --data-manifest "$DATA_MANIFEST_FILE" \
  --independent-audit "$INDEPENDENT_AUDIT_FILE" \
  --init-adapter "$INIT_ADAPTER_FILE" \
  --expected-train-sha256 "$EXPECTED_TRAIN_SHA256" \
  --expected-heldout-sha256 "$EXPECTED_HELDOUT_SHA256" \
  --expected-data-manifest-sha256 "$EXPECTED_DATA_MANIFEST_SHA256" \
  --expected-independent-audit-sha256 "$EXPECTED_INDEPENDENT_AUDIT_SHA256" \
  --expected-init-adapter-sha256 "$EXPECTED_INIT_ADAPTER_SHA256" \
  --code-ledger "$RUN_DIR/code.sha256" \
  --expected-code-ledger-sha256 "$EXPECTED_CODE_LEDGER_SHA256" \
  --model-artifact-ledger "$RUN_DIR/model-artifacts.sha256" \
  --expected-model-artifact-ledger-sha256 \
    "$EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256" \
  --model-weight-ledger "$RUN_DIR/model-weights.sha256" \
  --expected-model-weight-ledger-sha256 "$EXPECTED_MODEL_WEIGHT_LEDGER_SHA256" \
  --output-dir "$RUN_DIR/artifacts" \
  > "$RUN_DIR/logs/train.log" 2>&1
test -s "$RUN_DIR/logs/train.log"

for REQUIRED in \
  "$RUN_DIR/artifacts/metadata.json" \
  "$RUN_DIR/artifacts/best-checkpoint.json" \
  "$RUN_DIR/artifacts/step-1-hard-gate.json" \
  "$RUN_DIR/artifacts/step-2-hard-gate.json" \
  "$RUN_DIR/artifacts/answer-decode-semantic-diagnostic.json" \
  "$RUN_DIR/artifacts/checkpoint-000000.pt" \
  "$RUN_DIR/artifacts/checkpoint-000064.pt" \
  "$RUN_DIR/artifacts/checkpoint-000128.pt" \
  "$RUN_DIR/artifacts/teacher-targets/teacher-manifest.json" \
  "$RUN_DIR/artifacts/teacher-targets/teacher-artifacts.sha256"; do
  test -s "$REQUIRED"
done
sha256sum -c "$RUN_DIR/artifacts/teacher-targets/teacher-artifacts.sha256" \
  > "$RUN_DIR/logs/teacher-integrity.log"
"$ENV_DIR/bin/python" - "$RUN_DIR/artifacts" <<'PY' \
  > "$RUN_DIR/logs/final-audit.json"
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
m=json.load(open(root/"metadata.json"))
g1=json.load(open(root/"step-1-hard-gate.json"))
g2=json.load(open(root/"step-2-hard-gate.json"))
b=json.load(open(root/"best-checkpoint.json"))
d=json.load(open(root/"answer-decode-semantic-diagnostic.json"))
assert m["format"]=="qcomem_answer_supervised_native_lora_v1"
assert m["last_step"]==128
assert m["data"]["train_domain_counts"]=={"qasper":256,"2wikimqa":154}
assert m["data"]["heldout_domain_counts"]=={"qasper":12,"2wikimqa":14}
assert m["data"]["included_strata"]==["domain"]
assert m["adapter_surface"]["module_counts"]=={"full_attention":36,"gdn":120}
assert m["adapter_surface"]["mlp"]["covered"] is False
assert m["adapter_memory"]["trainable_parameters"]==26689536
assert m["adapter_config"]["rank"]==32
assert m["adapter_config"]["alpha"]==64.0
assert m["adapter_config"]["dropout"]==0.0
assert m["adapter_config"]["installed_module_count"]==156
assert m["adapter_config"]["parameter_tensor_count"]==312
assert m["adapter_config"]["trainable_parameters"]==26689536
assert m["adapter_config"]["target_suffixes"]==["q_proj","k_proj","v_proj","o_proj","in_proj_qkv","in_proj_z","in_proj_b","in_proj_a","out_proj"]
assert m["initialization_attribution"]["full_attention_modules_warm_started"]==36
assert m["initialization_attribution"]["gdn_modules_cold_started"]==120
assert m["initialization_attribution"]["pure_cold_start_experiment"] is False
assert m["initialization_attribution"]["step_zero_remains_eligible_for_official_train_heldout_selection"] is True
assert m["loss"]["student_projection_chunk_backward"]=="non_reentrant_activation_checkpoint_recompute"
assert m["loss"]["all_answer_full_vocab_logits_retained_until_backward"] is False
assert g1["status"]==g2["status"]=="passed"
assert g1["checks"]["memory_headroom"]
assert g1["checks"]["cold_gdn_a_expected_zero_first_update"]
assert g2["checks"]["all_adapter_gradients_finite_nonzero"]
assert g2["checks"]["all_adapter_updates_finite_nonzero"]
assert b["selected_step"] in (0,64,128)
assert b["validation_6_35_used_for_selection"] is False
assert b["test_v2_used"] is False
assert d["examples"]==26 and d["equivalence_claimed"] is False
assert d["blocking_gate"] is False
assert d["validation_6_35_used"] is False and d["test_v2_used"] is False
print(json.dumps({"status":"passed","best_step":b["selected_step"]}))
PY
sha256sum "$RUN_DIR/artifacts"/checkpoint-*.pt \
  > "$RUN_DIR/checkpoints.sha256"
date -u +%FT%TZ > "$RUN_DIR/stages/02_training_complete"

# The checkpoint is now irrevocably selected from official-train heldout.
# Only after that freeze do we read the already-consumed validation 6--35
# artifact for attribution-only downstream evaluation.
CURRENT_PHASE=downstream
sha256sum -c "$RUN_DIR/code.sha256" >> "$RUN_DIR/logs/code-integrity.log"
sha256sum -c "$RUN_DIR/model-artifacts.sha256" \
  >> "$RUN_DIR/logs/model-artifact-integrity.log"
verify_sha "$RUN_DIR/model-weights.sha256" \
  "$EXPECTED_MODEL_WEIGHT_LEDGER_SHA256" model-weight-ledger
sha256sum -c "$RUN_DIR/model-weights.sha256" \
  >> "$RUN_DIR/logs/model-weight-integrity.log"
verify_sha "$VALIDATION_DATA_FILE" "$EXPECTED_VALIDATION_SHA256" validation
sha256sum "$VALIDATION_DATA_FILE" >> "$RUN_DIR/input-artifacts.sha256"
DOWNSTREAM_DIR="$RUN_DIR/downstream"
mkdir -p "$DOWNSTREAM_DIR"
PYTHONPATH="$CODE_DIR" "$ENV_DIR/bin/python" \
  "$CODE_DIR/run_answer_lora_full_state_downstream.py" \
  --model "$MODEL_DIR" --data "$VALIDATION_DATA_FILE" \
  --expected-data-sha256 "$EXPECTED_VALIDATION_SHA256" \
  --best-checkpoint "$RUN_DIR/artifacts/best-checkpoint.json" \
  --run-dir "$DOWNSTREAM_DIR" --preflight-only \
  > "$RUN_DIR/logs/downstream-protocol-preflight.log" 2>&1
date -u +%FT%TZ > "$RUN_DIR/stages/03_downstream_preflight_ok"

PIDS=()
for RANK in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$RANK PYTHONPATH="$CODE_DIR" \
    timeout --signal=TERM --kill-after=60s 14400s "$ENV_DIR/bin/python" \
    "$CODE_DIR/run_answer_lora_full_state_downstream.py" \
    --model "$MODEL_DIR" --data "$VALIDATION_DATA_FILE" \
    --expected-data-sha256 "$EXPECTED_VALIDATION_SHA256" \
    --best-checkpoint "$RUN_DIR/artifacts/best-checkpoint.json" \
    --run-dir "$DOWNSTREAM_DIR" --rank "$RANK" --world-size 8 \
    --max-input-tokens 4096 --max-new-tokens 128 --group-size 64 \
    > "$RUN_DIR/logs/downstream-rank-${RANK}.log" 2>&1 &
  PIDS+=("$!")
  sleep 3
done
FAILED=0
for INDEX in 0 1 2 3 4 5 6 7; do
  if ! wait "${PIDS[$INDEX]}"; then
    echo "answer downstream rank $INDEX failed" >&2
    FAILED=1
  fi
done
[[ "$FAILED" -eq 0 ]]
date -u +%FT%TZ > "$RUN_DIR/stages/04_downstream_shards_ok"

PYTHONPATH="$CODE_DIR" "$ENV_DIR/bin/python" \
  "$CODE_DIR/aggregate_answer_lora_full_state_downstream.py" \
  --run-dir "$DOWNSTREAM_DIR" \
  --expected-data-sha256 "$EXPECTED_VALIDATION_SHA256" \
  --bootstrap-seed 20260814 \
  > "$RUN_DIR/logs/downstream-aggregate.log" 2>&1
test -s "$DOWNSTREAM_DIR/answer-full-state-downstream-analysis.json"
"$ENV_DIR/bin/python" - "$DOWNSTREAM_DIR/answer-full-state-downstream-analysis.json" <<'PY' \
  > "$RUN_DIR/logs/downstream-final-audit.json"
import json,sys
r=json.load(open(sys.argv[1]))
assert r["status"]=="completed" and r["samples"]==60
assert r["conditions"]==[
 "dense-adapter-disabled-control","q16-adapter-disabled-control",
 "frozen-static-adapter-disabled","frozen-static-answer-lora-step0",
 "frozen-static-answer-lora-step64","frozen-static-answer-lora-step128"]
assert r["claim_boundaries"]["validation_may_select_checkpoint_or_policy"] is False
assert r["claim_boundaries"]["validation_step_trajectory_may_reselect_heldout_checkpoint"] is False
assert r["claim_boundaries"]["raw_test_v2_read"] is False
assert r["protocol"]["checkpoint_frozen_before_validation_read"] is True
assert r["protocol"]["all_checkpoint_steps_evaluated_regardless_of_validation"] is True
assert r["heldout_selected_alias"]["additional_forward_executed"] is False
for step in (0,64,128):
 key=f"frozen-static-answer-lora-step{step}_vs_frozen-static-adapter-disabled"
 assert r["paired_comparisons"][key]["samples"]==60
for key in (
 "frozen-static-answer-lora-step64_vs_frozen-static-answer-lora-step0",
 "frozen-static-answer-lora-step128_vs_frozen-static-answer-lora-step64"):
 assert len(r["paired_comparisons"][key]["paired_bootstrap_95_ci"])==2
p=r["heldout_selected_vs_disabled"]
assert p["samples"]==60 and len(p["paired_bootstrap_95_ci"])==2
assert set(p["per_dataset"])=={"qasper","2wikimqa"}
print(json.dumps({"status":"passed","selected_step":r["selected_checkpoint_step"]}))
PY
date -u +%FT%TZ > "$RUN_DIR/stages/05_downstream_complete"

nvidia-smi \
  --query-gpu=index,uuid,name,memory.used,pstate,power.draw,temperature.gpu \
  --format=csv > "$RUN_DIR/gpus-after.csv"
date -u +%FT%TZ > "$RUN_DIR/stages/99_done"
echo "Answer-supervised native-cache LoRA B + frozen downstream complete: $RUN_DIR"
