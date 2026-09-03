#!/usr/bin/env bash
set -euo pipefail

# Ledger ordering is protocol state.  Freeze bytewise C order on every host.
export LC_ALL=C

CODE_DIR=${CODE_DIR:?set CODE_DIR}
MODEL_DIR=${MODEL_DIR:?set MODEL_DIR}
MODEL_WEIGHT_LEDGER_FILE=${MODEL_WEIGHT_LEDGER_FILE:?set MODEL_WEIGHT_LEDGER_FILE}
PG19_DATA=${PG19_DATA:?set PG19_DATA}
PG19_MANIFEST=${PG19_MANIFEST:?set PG19_MANIFEST}
PROTOCOL_MANIFEST_FILE=${PROTOCOL_MANIFEST_FILE:?set PROTOCOL_MANIFEST_FILE}
RUN_DIR=${RUN_DIR:?set RUN_DIR}
ENV_DIR=${ENV_DIR:?set ENV_DIR}
EXPECTED_PG19_SHA256=${EXPECTED_PG19_SHA256:?set EXPECTED_PG19_SHA256}
EXPECTED_PG19_MANIFEST_SHA256=${EXPECTED_PG19_MANIFEST_SHA256:?set EXPECTED_PG19_MANIFEST_SHA256}
EXPECTED_PG19_WINDOWS_SHA256=${EXPECTED_PG19_WINDOWS_SHA256:?set EXPECTED_PG19_WINDOWS_SHA256}
EXPECTED_MODEL_MANIFEST_SHA256=${EXPECTED_MODEL_MANIFEST_SHA256:?set EXPECTED_MODEL_MANIFEST_SHA256}
EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256=${EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256:?set EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256}
EXPECTED_MODEL_WEIGHT_LEDGER_SHA256=${EXPECTED_MODEL_WEIGHT_LEDGER_SHA256:?set EXPECTED_MODEL_WEIGHT_LEDGER_SHA256}
EXPECTED_CODE_LEDGER_SHA256=${EXPECTED_CODE_LEDGER_SHA256:?set EXPECTED_CODE_LEDGER_SHA256}
EXPECTED_PROTOCOL_MANIFEST_SHA256=${EXPECTED_PROTOCOL_MANIFEST_SHA256:?set EXPECTED_PROTOCOL_MANIFEST_SHA256}

for VALUE in \
  "$EXPECTED_PG19_SHA256" "$EXPECTED_PG19_MANIFEST_SHA256" \
  "$EXPECTED_PG19_WINDOWS_SHA256" "$EXPECTED_MODEL_MANIFEST_SHA256" \
  "$EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256" \
  "$EXPECTED_MODEL_WEIGHT_LEDGER_SHA256" "$EXPECTED_CODE_LEDGER_SHA256" \
  "$EXPECTED_PROTOCOL_MANIFEST_SHA256"; do
  if [[ ! "$VALUE" =~ ^[0-9a-f]{64}$ ]]; then
    echo "every frozen digest must be one lowercase SHA256" >&2
    exit 2
  fi
done
case "$PG19_DATA:$PG19_MANIFEST" in
  *[Ll][Oo][Nn][Gg][Bb][Ee][Nn][Cc][Hh]*|*[Tt][Ee][Ss][Tt]-[Vv]2*|*68-99*)
    echo "multi-fork resident protocol accepts PG19 train-only inputs" >&2
    exit 2
    ;;
esac
if [[ -e "$RUN_DIR" && -n "$(find "$RUN_DIR" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "RUN_DIR must be absent or empty: $RUN_DIR" >&2
  exit 2
fi

mkdir -p \
  "$RUN_DIR/logs" "$RUN_DIR/stages" "$RUN_DIR/pycache" \
  "$RUN_DIR/resident-shards"
export PYTHONPYCACHEPREFIX="$RUN_DIR/pycache"
CURRENT_PHASE=preflight
PIDS=()
terminate_children() {
  local pid
  for pid in "${PIDS[@]:-}"; do
    kill -TERM "$pid" 2>/dev/null || true
  done
  for pid in "${PIDS[@]:-}"; do
    wait "$pid" 2>/dev/null || true
  done
  PIDS=()
}
on_error() {
  local status=$?
  trap - ERR INT TERM
  terminate_children
  date -u +%FT%TZ > "$RUN_DIR/stages/FAILED"
  printf '%s\n' "$CURRENT_PHASE" > "$RUN_DIR/stages/FAILED_PHASE"
  date -u +%FT%TZ > "$RUN_DIR/stages/FAILED_${CURRENT_PHASE}"
  exit "$status"
}
on_signal() {
  local status=$1
  trap - ERR INT TERM
  terminate_children
  date -u +%FT%TZ > "$RUN_DIR/stages/FAILED"
  printf '%s\n' "$CURRENT_PHASE" > "$RUN_DIR/stages/FAILED_PHASE"
  date -u +%FT%TZ > "$RUN_DIR/stages/FAILED_${CURRENT_PHASE}"
  exit "$status"
}
trap on_error ERR
trap 'on_signal 130' INT
trap 'on_signal 143' TERM
date -u +%FT%TZ > "$RUN_DIR/stages/00_start"

verify_sha() {
  local path=$1 expected=$2 label=$3 actual
  actual=$(sha256sum "$path" | awk '{print $1}')
  if [[ "$actual" != "$expected" ]]; then
    echo "$label SHA256 mismatch: expected=$expected actual=$actual" >&2
    return 2
  fi
}
fail_stage() {
  echo "$1" >&2
  return 2
}

for INPUT in "$PG19_DATA" "$PG19_MANIFEST" "$PROTOCOL_MANIFEST_FILE" \
  "$MODEL_WEIGHT_LEDGER_FILE"; do
  test -s "$INPUT"
done
verify_sha "$PG19_DATA" "$EXPECTED_PG19_SHA256" pg19-train
verify_sha "$PG19_MANIFEST" "$EXPECTED_PG19_MANIFEST_SHA256" pg19-manifest
verify_sha "$PROTOCOL_MANIFEST_FILE" \
  "$EXPECTED_PROTOCOL_MANIFEST_SHA256" runtime-protocol-manifest

if find "$CODE_DIR" -type f -perm /222 -print -quit | grep -q .; then
  fail_stage "frozen code snapshot contains a writable file"
fi
mapfile -d '' PY_FILES < <(
  find "$CODE_DIR" -maxdepth 1 -type f -name '*.py' -print0 | LC_ALL=C sort -z
)
CODE_FILES=(
  "${PY_FILES[@]}"
  "$CODE_DIR/launch_qcomem_qwen35_vllm_paged_multifork_resident_8gpu.sh"
  "$CODE_DIR/MULTIFORK_RESIDENT_PROTOCOL_ZH.md"
)
if [[ "${#CODE_FILES[@]}" -lt 10 ]]; then
  fail_stage "frozen code snapshot is unexpectedly incomplete"
fi
MODEL_ARTIFACT_FILES=(
  "$MODEL_DIR/config.json"
  "$MODEL_DIR/model.safetensors.index.json"
  "$MODEL_DIR/tokenizer_config.json"
  "$MODEL_DIR/vocab.json"
  "$MODEL_DIR/merges.txt"
  "$MODEL_DIR/chat_template.jinja"
)
for ARTIFACT in "${CODE_FILES[@]}" "${MODEL_ARTIFACT_FILES[@]}"; do
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
sha256sum -c "$RUN_DIR/model-weights.sha256" \
  > "$RUN_DIR/logs/model-weight-integrity.log"
sha256sum "$PG19_DATA" "$PG19_MANIFEST" "$PROTOCOL_MANIFEST_FILE" \
  > "$RUN_DIR/input-artifacts.sha256"

timeout --signal=TERM --kill-after=30s 300s \
  "$ENV_DIR/bin/python" -m py_compile "${PY_FILES[@]}"
bash -n "$CODE_DIR/launch_qcomem_qwen35_vllm_paged_multifork_resident_8gpu.sh"
timeout --signal=TERM --kill-after=30s 900s \
  env PYTHONPATH="$CODE_DIR" "$ENV_DIR/bin/python" -m unittest -v \
  test_qcomem_vllm_paged_kernel \
  test_qcomem_qwen35_vllm_paged_integration \
  test_qcomem_vllm_paged_multifork_resident \
  test_run_qcomem_qwen35_vllm_paged_multifork_resident \
  test_launch_qcomem_qwen35_vllm_paged_multifork_resident \
  > "$RUN_DIR/logs/focused-tests.log" 2>&1
if grep -Eq 'skipped=|\.\.\. skipped' "$RUN_DIR/logs/focused-tests.log"; then
  fail_stage "focused test suite contained a skip"
fi
grep -Eq '^test_real_tf514_qwen_call_consumes_and_advances_position_ids .* \.\.\. ok$' \
  "$RUN_DIR/logs/focused-tests.log"

COMMON=(
  --model "$MODEL_DIR"
  --pg19-data "$PG19_DATA"
  --pg19-manifest "$PG19_MANIFEST"
  --expected-pg19-sha256 "$EXPECTED_PG19_SHA256"
  --expected-pg19-manifest-sha256 "$EXPECTED_PG19_MANIFEST_SHA256"
  --expected-pg19-windows-sha256 "$EXPECTED_PG19_WINDOWS_SHA256"
  --expected-model-manifest-sha256 "$EXPECTED_MODEL_MANIFEST_SHA256"
  --run-dir "$RUN_DIR"
  --protocol-manifest "$PROTOCOL_MANIFEST_FILE"
  --expected-protocol-manifest-sha256 "$EXPECTED_PROTOCOL_MANIFEST_SHA256"
  --code-ledger "$RUN_DIR/code.sha256"
  --model-artifact-ledger "$RUN_DIR/model-artifacts.sha256"
  --model-weight-ledger "$RUN_DIR/model-weights.sha256"
  --expected-code-ledger-sha256 "$EXPECTED_CODE_LEDGER_SHA256"
  --expected-model-artifact-ledger-sha256 \
    "$EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256"
  --expected-model-weight-ledger-sha256 "$EXPECTED_MODEL_WEIGHT_LEDGER_SHA256"
  --bits 16
  --page-size 128
  --resident-counts 1 2 4 8 16 32
  --execution-order 1 32 2 16 4 8
  --pg19-books 8
  --pg19-document-tokens 4095
  --pg19-query-tokens 32
  --pg19-window-stride 257
  --pg19-candidate-windows 8
  --pg19-seed 20260814
  --query-bank-stride 64
  --max-new-tokens 8
)

timeout --signal=TERM --kill-after=30s 300s \
  env PYTHONPATH="$CODE_DIR" "$ENV_DIR/bin/python" \
  "$CODE_DIR/run_qcomem_qwen35_vllm_paged_multifork_resident.py" \
  --stage static-dry-run --rank 0 --world-size 8 \
  --output "$RUN_DIR/static-dry-run.json" "${COMMON[@]}" \
  > "$RUN_DIR/logs/static-dry-run.log" 2>&1
"$ENV_DIR/bin/python" - "$RUN_DIR/static-dry-run.json" <<'PY' \
  > "$RUN_DIR/logs/static-final-audit.json"
import json,sys
value=json.load(open(sys.argv[1]))
assert value["status"]=="multifork_resident_static_dry_run_passed"
assert value["gpu_initialized"] is False
assert value["pg19_train_only"] is True
assert value["longbench_consumed"] is False
assert value["source_6_9_consumed"] is False
assert value["source_68_99_consumed"] is False
assert value["test_v2_consumed"] is False
assert len(value["frozen_query_banks"])==8
print(json.dumps({"status":"passed","gpu_initialized":False,"query_banks":8}))
PY
date -u +%FT%TZ > "$RUN_DIR/stages/01_static_preflight_ok"

GPU_COUNT=$(nvidia-smi -L | wc -l | tr -d ' ')
if [[ "$GPU_COUNT" -ne 8 ]]; then
  fail_stage "formal multi-fork run requires exactly eight GPUs, found $GPU_COUNT"
fi
nvidia-smi \
  --query-gpu=index,uuid,name,memory.total,driver_version,pstate,power.limit \
  --format=csv > "$RUN_DIR/gpus-before.csv"

CURRENT_PHASE=pg19_multifork_resident_shards
PIDS=()
for RANK in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$RANK PYTHONPATH="$CODE_DIR" \
    timeout --signal=TERM --kill-after=60s 21600s \
    "$ENV_DIR/bin/python" \
    "$CODE_DIR/run_qcomem_qwen35_vllm_paged_multifork_resident.py" \
    --stage resident-shard --rank "$RANK" --world-size 8 \
    --output "$RUN_DIR/resident-shards/multifork-resident-shard-$RANK.json" \
    "${COMMON[@]}" > "$RUN_DIR/logs/resident-rank-$RANK.log" 2>&1 &
  PIDS+=("$!")
done
for INDEX in 0 1 2 3 4 5 6 7; do
  if ! wait "${PIDS[$INDEX]}"; then
    echo "multi-fork resident rank $INDEX failed" >&2
    terminate_children
    fail_stage "multi-fork resident shard phase failed"
  fi
done
PIDS=()
for RANK in 0 1 2 3 4 5 6 7; do
  test -s "$RUN_DIR/resident-shards/multifork-resident-shard-$RANK.json"
done
SHARD_COUNT=$(find "$RUN_DIR/resident-shards" -maxdepth 1 -type f \
  -name 'multifork-resident-shard-*.json' | wc -l | tr -d ' ')
if [[ "$SHARD_COUNT" -ne 8 ]]; then
  fail_stage "resident shard cardinality differs from eight"
fi
date -u +%FT%TZ > "$RUN_DIR/stages/02_resident_shards_ok"

CURRENT_PHASE=aggregate_and_integrity
sha256sum -c "$RUN_DIR/code.sha256" >> "$RUN_DIR/logs/code-integrity.log"
sha256sum -c "$RUN_DIR/model-artifacts.sha256" \
  >> "$RUN_DIR/logs/model-artifact-integrity.log"
sha256sum -c "$RUN_DIR/model-weights.sha256" \
  >> "$RUN_DIR/logs/model-weight-integrity.log"
verify_sha "$PG19_DATA" "$EXPECTED_PG19_SHA256" pg19-train
verify_sha "$PG19_MANIFEST" "$EXPECTED_PG19_MANIFEST_SHA256" pg19-manifest
verify_sha "$PROTOCOL_MANIFEST_FILE" \
  "$EXPECTED_PROTOCOL_MANIFEST_SHA256" runtime-protocol-manifest

SUMMARY="$RUN_DIR/multifork-resident-summary.json"
timeout --signal=TERM --kill-after=30s 900s \
  env PYTHONPATH="$CODE_DIR" "$ENV_DIR/bin/python" \
  "$CODE_DIR/run_qcomem_qwen35_vllm_paged_multifork_resident.py" \
  --stage aggregate --rank 0 --world-size 8 \
  --output "$SUMMARY" "${COMMON[@]}" \
  > "$RUN_DIR/logs/aggregate.log" 2>&1
timeout --signal=TERM --kill-after=30s 120s \
  "$ENV_DIR/bin/python" - "$SUMMARY" \
  "$EXPECTED_CODE_LEDGER_SHA256" "$EXPECTED_MODEL_MANIFEST_SHA256" \
  "$EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256" \
  "$EXPECTED_MODEL_WEIGHT_LEDGER_SHA256" "$EXPECTED_PG19_SHA256" \
  "$EXPECTED_PG19_MANIFEST_SHA256" "$EXPECTED_PG19_WINDOWS_SHA256" \
  "$EXPECTED_PROTOCOL_MANIFEST_SHA256" <<'PY' \
  > "$RUN_DIR/logs/summary-final-audit.json"
import json,sys
value=json.load(open(sys.argv[1]))
assert value["status"]=="completed_multifork_resident_pg19_summary"
assert value["passed"] is True and value["rank_count"]==8
assert value["resident_counts"]==[1,2,4,8,16,32]
assert value["same_kernel_full_logit_token_logical_kv_gdn_exact_fraction"]==1.0
assert value["cross_n_prefix_isolation_exact"] is True
assert value["primary_capacity_slopes_use_replayed_analytic_q16_pools_only"] is True
assert value["combined_unique_inventory_is_diagnostic_not_fitted_or_claim_authorizing"] is True
assert value["timing_is_raw_validation_instrumented_single_observation_not_aggregated"] is True
assert value["allocator_deltas_are_relative_to_post_pack_request_setup_baseline"] is True
assert value["allocator_absolute_values_are_pytorch_allocator_not_nvml_or_total_model_capacity"] is True
assert value["pg19_train_only"] is True
assert value["longbench_consumed"] is False
assert value["source_6_9_consumed"] is False
assert value["source_68_99_consumed"] is False
assert value["test_v2_consumed"] is False
identity=value["frozen_identity"]
assert identity["code_ledger_sha256"]==sys.argv[2]
assert identity["model_manifest_sha256"]==sys.argv[3]
assert identity["model_artifact_ledger_sha256"]==sys.argv[4]
assert identity["model_weight_ledger_sha256"]==sys.argv[5]
assert identity["pg19_data_sha256"]==sys.argv[6]
assert identity["pg19_manifest_sha256"]==sys.argv[7]
assert identity["pg19_windows_sha256"]==sys.argv[8]
assert identity["protocol_manifest_sha256"]==sys.argv[9]
for rank in value["rank_capacity_curves_and_fits"]:
    fits=rank["fits"]
    assert fits["fresh_full_attention_pool"]["slope_nbytes_per_request"]==94371840.0
    assert fits["reuse_full_attention_pool"]["slope_nbytes_per_request"]==5242880.0
    assert fits["controlled_pool_bytes_saved"]["slope_nbytes_per_request"]==89128960.0
    assert fits["fresh_physical_document_copy"]["slope_nbytes_per_request"]==83886080.0
print(json.dumps({"status":"passed","claim":"Q16-PG19-train-only-multifork-capacity"}))
PY

nvidia-smi \
  --query-gpu=index,uuid,name,memory.used,pstate,power.draw,temperature.gpu \
  --format=csv > "$RUN_DIR/gpus-after.csv"
find "$RUN_DIR" -type f \
  ! -path "$RUN_DIR/pycache/*" \
  ! -name 'scientific-artifacts.sha256' -print0 \
  | LC_ALL=C sort -z \
  | xargs -0 sha256sum > "$RUN_DIR/scientific-artifacts.sha256"
sha256sum -c "$RUN_DIR/scientific-artifacts.sha256" \
  > "$RUN_DIR/logs/scientific-artifact-integrity.log"
date -u +%FT%TZ > "$RUN_DIR/stages/99_done"
echo "Q16 PG19-train multi-fork resident protocol complete: $RUN_DIR"
