#!/usr/bin/env bash
set -euo pipefail

# Ledger order is part of the frozen protocol.  Never inherit the submit host
# or container locale: en_US collation is not byte-for-byte C collation.
export LC_ALL=C

CODE_DIR=${CODE_DIR:?set CODE_DIR}
MODEL_DIR=${MODEL_DIR:?set MODEL_DIR}
MODEL_WEIGHT_LEDGER_FILE=${MODEL_WEIGHT_LEDGER_FILE:?set MODEL_WEIGHT_LEDGER_FILE}
PG19_DATA=${PG19_DATA:?set PG19_DATA}
PG19_MANIFEST=${PG19_MANIFEST:?set PG19_MANIFEST}
VALIDATION_DATA=${VALIDATION_DATA:?set VALIDATION_DATA}
PROTOCOL_MANIFEST_FILE=${PROTOCOL_MANIFEST_FILE:?set PROTOCOL_MANIFEST_FILE}
RUN_DIR=${RUN_DIR:?set RUN_DIR}
ENV_DIR=${ENV_DIR:?set ENV_DIR}
EXPECTED_PG19_SHA256=${EXPECTED_PG19_SHA256:?set EXPECTED_PG19_SHA256}
EXPECTED_PG19_MANIFEST_SHA256=${EXPECTED_PG19_MANIFEST_SHA256:?set EXPECTED_PG19_MANIFEST_SHA256}
EXPECTED_PG19_WINDOWS_SHA256=${EXPECTED_PG19_WINDOWS_SHA256:?set EXPECTED_PG19_WINDOWS_SHA256}
EXPECTED_VALIDATION_SHA256=${EXPECTED_VALIDATION_SHA256:?set EXPECTED_VALIDATION_SHA256}
EXPECTED_SOURCE_REVISION=${EXPECTED_SOURCE_REVISION:?set EXPECTED_SOURCE_REVISION}
EXPECTED_MODEL_MANIFEST_SHA256=${EXPECTED_MODEL_MANIFEST_SHA256:?set EXPECTED_MODEL_MANIFEST_SHA256}
EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256=${EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256:?set EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256}
EXPECTED_MODEL_WEIGHT_LEDGER_SHA256=${EXPECTED_MODEL_WEIGHT_LEDGER_SHA256:?set EXPECTED_MODEL_WEIGHT_LEDGER_SHA256}
EXPECTED_CODE_LEDGER_SHA256=${EXPECTED_CODE_LEDGER_SHA256:?set EXPECTED_CODE_LEDGER_SHA256}
EXPECTED_PROTOCOL_MANIFEST_SHA256=${EXPECTED_PROTOCOL_MANIFEST_SHA256:?set EXPECTED_PROTOCOL_MANIFEST_SHA256}

for VALUE in \
  "$EXPECTED_PG19_SHA256" "$EXPECTED_PG19_MANIFEST_SHA256" \
  "$EXPECTED_PG19_WINDOWS_SHA256" "$EXPECTED_VALIDATION_SHA256" \
  "$EXPECTED_MODEL_MANIFEST_SHA256" \
  "$EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256" \
  "$EXPECTED_MODEL_WEIGHT_LEDGER_SHA256" "$EXPECTED_CODE_LEDGER_SHA256" \
  "$EXPECTED_PROTOCOL_MANIFEST_SHA256"; do
  if [[ ! "$VALUE" =~ ^[0-9a-f]{64}$ ]]; then
    echo "every frozen digest must be one lowercase SHA256" >&2
    exit 2
  fi
done
if [[ ! "$EXPECTED_SOURCE_REVISION" =~ ^[0-9a-f]{40}$ ]]; then
  echo "EXPECTED_SOURCE_REVISION must be one frozen git SHA1" >&2
  exit 2
fi
if [[ "$EXPECTED_VALIDATION_SHA256" == fe046477bf5b39629e9f66fd4def7a55c2d5d1f073c8bb601ee3833f08eaaa5f ]]; then
  echo "refusing LongBench test-v2" >&2
  exit 2
fi
case "$VALIDATION_DATA" in
  *[Tt][Ee][Ss][Tt]-[Vv]2*)
    echo "refusing a test-v2 validation path" >&2
    exit 2
    ;;
esac
case "$PG19_DATA" in
  *[Ll][Oo][Nn][Gg][Bb][Ee][Nn][Cc][Hh]*)
    echo "PG19 correctness gate must use train-only data" >&2
    exit 2
    ;;
esac
if [[ -e "$RUN_DIR" && -n "$(find "$RUN_DIR" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "RUN_DIR must be absent or empty: $RUN_DIR" >&2
  exit 2
fi

mkdir -p \
  "$RUN_DIR/logs" "$RUN_DIR/stages" "$RUN_DIR/pycache" \
  "$RUN_DIR/pg19-gate-shards" "$RUN_DIR/validation-shards"
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
  local message=$1
  echo "$message" >&2
  return 2
}

# Validation is intentionally neither tested for existence nor hashed here.
# The first content access appears only after the PG19 authorization artifact.
for INPUT in "$PG19_DATA" "$PG19_MANIFEST" "$PROTOCOL_MANIFEST_FILE" \
  "$MODEL_WEIGHT_LEDGER_FILE"; do
  test -s "$INPUT"
done
verify_sha "$PG19_DATA" "$EXPECTED_PG19_SHA256" pg19-train
verify_sha "$PG19_MANIFEST" "$EXPECTED_PG19_MANIFEST_SHA256" pg19-manifest
verify_sha "$PROTOCOL_MANIFEST_FILE" \
  "$EXPECTED_PROTOCOL_MANIFEST_SHA256" runtime-protocol-manifest

mapfile -d '' PY_FILES < <(
  find "$CODE_DIR" -maxdepth 1 -type f -name '*.py' -print0 | LC_ALL=C sort -z
)
CODE_FILES=("${PY_FILES[@]}" "$CODE_DIR/launch_qcomem_qwen35_vllm_paged_fair_v2_8gpu.sh")
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
  > "$RUN_DIR/input-artifacts-before-authorization.sha256"

"$ENV_DIR/bin/python" -m py_compile "${PY_FILES[@]}"
bash -n "$CODE_DIR/launch_qcomem_qwen35_vllm_paged_fair_v2_8gpu.sh"
PYTHONPATH="$CODE_DIR" "$ENV_DIR/bin/python" -m unittest -v \
  test_qcomem_vllm_paged_kernel \
  test_qcomem_qwen35_vllm_paged_integration \
  test_qcomem_vllm_paged_fair_control \
  test_run_qcomem_qwen35_vllm_paged_fair_v2 \
  > "$RUN_DIR/logs/focused-tests.log" 2>&1

COMMON=(
  --model "$MODEL_DIR"
  --pg19-data "$PG19_DATA"
  --pg19-manifest "$PG19_MANIFEST"
  --validation-data "$VALIDATION_DATA"
  --expected-pg19-sha256 "$EXPECTED_PG19_SHA256"
  --expected-pg19-manifest-sha256 "$EXPECTED_PG19_MANIFEST_SHA256"
  --expected-pg19-windows-sha256 "$EXPECTED_PG19_WINDOWS_SHA256"
  --expected-validation-sha256 "$EXPECTED_VALIDATION_SHA256"
  --expected-model-manifest-sha256 "$EXPECTED_MODEL_MANIFEST_SHA256"
  --expected-source-revision "$EXPECTED_SOURCE_REVISION"
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
  --max-input-tokens 4096
  --max-query-tokens 64
  --max-new-tokens 8
  --source-index-start 6
  --source-index-end 9
  --limit-per-dataset 4
  --min-input-tokens 1
  --pg19-books 8
  --pg19-document-tokens 1025
  --pg19-query-tokens 32
  --pg19-window-stride 257
  --pg19-candidate-windows 8
  --pg19-seed 20260814
)

PYTHONPATH="$CODE_DIR" "$ENV_DIR/bin/python" \
  "$CODE_DIR/run_qcomem_qwen35_vllm_paged_fair_v2.py" \
  --stage static-dry-run --rank 0 --world-size 8 \
  --output "$RUN_DIR/static-dry-run.json" "${COMMON[@]}" \
  > "$RUN_DIR/logs/static-dry-run.log" 2>&1
date -u +%FT%TZ > "$RUN_DIR/stages/01_static_preflight_ok"

GPU_COUNT=$(nvidia-smi -L | wc -l | tr -d ' ')
if [[ "$GPU_COUNT" -ne 8 ]]; then
  fail_stage "formal fair-v2 run requires exactly eight GPUs, found $GPU_COUNT"
fi
nvidia-smi \
  --query-gpu=index,uuid,name,memory.total,driver_version,pstate,power.limit \
  --format=csv > "$RUN_DIR/gpus-before.csv"

# Phase 1: PG19 train-only correctness. No validation content has been read.
CURRENT_PHASE=pg19_train_only_gate
PIDS=()
for RANK in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$RANK PYTHONPATH="$CODE_DIR" \
    timeout --signal=TERM --kill-after=60s 3600s \
    "$ENV_DIR/bin/python" "$CODE_DIR/run_qcomem_qwen35_vllm_paged_fair_v2.py" \
    --stage pg19-gate --rank "$RANK" --world-size 8 \
    --output "$RUN_DIR/pg19-gate-shards/pg19-fair-v2-shard-$RANK.json" \
    "${COMMON[@]}" > "$RUN_DIR/logs/pg19-rank-$RANK.log" 2>&1 &
  PIDS+=("$!")
done
FAILED=0
for INDEX in 0 1 2 3 4 5 6 7; do
  if ! wait "${PIDS[$INDEX]}"; then
    echo "PG19 fair-v2 rank $INDEX failed" >&2
    FAILED=1
  fi
done
[[ "$FAILED" -eq 0 ]]
PIDS=()
date -u +%FT%TZ > "$RUN_DIR/stages/02_pg19_shards_ok"

AUTHORIZATION="$RUN_DIR/pg19-fair-v2-authorization.json"
PYTHONPATH="$CODE_DIR" "$ENV_DIR/bin/python" \
  "$CODE_DIR/run_qcomem_qwen35_vllm_paged_fair_v2.py" \
  --stage aggregate-pg19 --rank 0 --world-size 8 \
  --output "$AUTHORIZATION" "${COMMON[@]}" \
  > "$RUN_DIR/logs/aggregate-pg19.log" 2>&1
AUTHORIZATION_SHA256=$(sha256sum "$AUTHORIZATION" | awk '{print $1}')
printf '%s  %s\n' "$AUTHORIZATION_SHA256" "$AUTHORIZATION" \
  > "$RUN_DIR/pg19-fair-v2-authorization.sha256"
"$ENV_DIR/bin/python" - "$AUTHORIZATION" <<'PY' \
  > "$RUN_DIR/logs/authorization-final-audit.json"
import json,sys
value=json.load(open(sys.argv[1]))
assert value["status"]=="pg19_fair_v2_authorized" and value["passed"] is True
assert value["same_kernel_layout_gate_passed"] is True
assert value["same_kernel_full_vocab_logit_gate_passed"] is True
assert value["backend_compatibility_is_authorization_gate"] is False
assert value["validation_consumed"] is False
assert value["validation_hashed"] is False
assert value["source_68_99_consumed"] is False
assert value["test_v2_consumed"] is False
assert value["protocol_config"]["expected_source_revision"]
print(json.dumps({"status":"passed","authorization":"same-kernel-only"}))
PY
date -u +%FT%TZ > "$RUN_DIR/stages/03_pg19_authorized"

# Phase boundary: rehash every mutable frozen input, then and only then touch
# LongBench validation content.
CURRENT_PHASE=post_authorization_validation
sha256sum -c "$RUN_DIR/code.sha256" >> "$RUN_DIR/logs/code-integrity.log"
sha256sum -c "$RUN_DIR/model-artifacts.sha256" \
  >> "$RUN_DIR/logs/model-artifact-integrity.log"
sha256sum -c "$RUN_DIR/model-weights.sha256" \
  >> "$RUN_DIR/logs/model-weight-integrity.log"
verify_sha "$PG19_DATA" "$EXPECTED_PG19_SHA256" pg19-train
verify_sha "$PG19_MANIFEST" "$EXPECTED_PG19_MANIFEST_SHA256" pg19-manifest
verify_sha "$PROTOCOL_MANIFEST_FILE" \
  "$EXPECTED_PROTOCOL_MANIFEST_SHA256" runtime-protocol-manifest
test -s "$VALIDATION_DATA"
verify_sha "$VALIDATION_DATA" "$EXPECTED_VALIDATION_SHA256" validation-source6-9
sha256sum "$VALIDATION_DATA" > "$RUN_DIR/validation-after-authorization.sha256"
date -u +%FT%TZ > "$RUN_DIR/stages/04_validation_hash_authorized"

PIDS=()
for RANK in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$RANK PYTHONPATH="$CODE_DIR" \
    timeout --signal=TERM --kill-after=60s 7200s \
    "$ENV_DIR/bin/python" "$CODE_DIR/run_qcomem_qwen35_vllm_paged_fair_v2.py" \
    --stage validation --rank "$RANK" --world-size 8 \
    --authorization "$AUTHORIZATION" \
    --expected-authorization-sha256 "$AUTHORIZATION_SHA256" \
    --output "$RUN_DIR/validation-shards/fair-v2-shard-$RANK.json" \
    "${COMMON[@]}" > "$RUN_DIR/logs/validation-rank-$RANK.log" 2>&1 &
  PIDS+=("$!")
done
FAILED=0
for INDEX in 0 1 2 3 4 5 6 7; do
  if ! wait "${PIDS[$INDEX]}"; then
    echo "fair-v2 validation rank $INDEX failed" >&2
    FAILED=1
  fi
done
[[ "$FAILED" -eq 0 ]]
PIDS=()
date -u +%FT%TZ > "$RUN_DIR/stages/05_validation_shards_ok"

SUMMARY="$RUN_DIR/fair-v2-summary.json"
PYTHONPATH="$CODE_DIR" "$ENV_DIR/bin/python" \
  "$CODE_DIR/run_qcomem_qwen35_vllm_paged_fair_v2.py" \
  --stage aggregate-validation --rank 0 --world-size 8 \
  --authorization "$AUTHORIZATION" \
  --expected-authorization-sha256 "$AUTHORIZATION_SHA256" \
  --output "$SUMMARY" "${COMMON[@]}" \
  > "$RUN_DIR/logs/aggregate-validation.log" 2>&1
"$ENV_DIR/bin/python" - "$SUMMARY" <<'PY' \
  > "$RUN_DIR/logs/summary-final-audit.json"
import json,sys
value=json.load(open(sys.argv[1]))
assert value["status"]=="completed_fair_v2_summary"
assert value["primary_full_logit_parity_fraction"]==1.0
assert value["backend_compatibility_used_for_primary_speedup"] is False
assert value["isolated_kernel_latency_measured"] is False
assert value["single_request_only"] is True
assert value["ragged_batch_claimed"] is False
assert value["multi_query_serving_completed"] is False
assert value["source_68_99_consumed"] is False
assert value["test_v2_consumed"] is False
print(json.dumps({"status":"passed","claim":"same-kernel-single-request"}))
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
echo "Q16 same-kernel paged fair-v2 protocol complete: $RUN_DIR"
