#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

# Affected-path-only debug recapture for Round-22 A1.  This is deliberately a
# single-rank receipt recapture, not a replacement for the formal eight-rank
# launcher.  All inputs below are the preserved W-run inputs and outputs.
OLD_CODE_DIR=${OLD_CODE_DIR:?set OLD_CODE_DIR to the preserved W source tree}
OLD_RUN_DIR=${OLD_RUN_DIR:?set OLD_RUN_DIR to the preserved W result}
INPUT_DIR=${INPUT_DIR:?set INPUT_DIR to the preserved W frozen inputs}
RUN_DIR=${RUN_DIR:?set a new debug RUN_DIR}
ENV_DIR=${ENV_DIR:?set the frozen W Python environment}
SOURCE_MODEL_DIR=${SOURCE_MODEL_DIR:?set the frozen W source model directory}
PG19_DATA=${PG19_DATA:?set the frozen PG19 train jsonl}
PG19_MANIFEST=${PG19_MANIFEST:?set the frozen PG19 train manifest}
IDENTITY_BUILDER=${IDENTITY_BUILDER:?set the reviewed execution identity builder}

if [[ -e "$RUN_DIR" ]]; then
  echo "RUN_DIR must be absent: $RUN_DIR" >&2
  exit 2
fi

SELF=$(realpath "$0")
PYTHON="$ENV_DIR/bin/python"
CODE_DIR="$RUN_DIR/code"
RESULT_DIR="$RUN_DIR/run"
RUNNER="$CODE_DIR/run_qcomem_qwen35_forkaudit_review_revision.py"
MANIFEST_BUILDER="$CODE_DIR/build_qcomem_qwen35_forkaudit_review_manifest.py"
PROTOCOL_MANIFEST="$CODE_DIR/qcomem_qwen35_vllm_paged_multifork_resident_protocol_manifest.json"
MODEL_VIEW="$RUN_DIR/model-view"
MODEL_ARTIFACT_LEDGER="$INPUT_DIR/model-artifacts.formal.sha256"
MODEL_WEIGHT_LEDGER="$INPUT_DIR/model-weights.canonical.sha256"
RR2_INPUT_MANIFEST="$INPUT_DIR/rr2-pg19-input-manifest.json"
PRIOR_CAPACITY_MANIFEST="$PROTOCOL_MANIFEST"
PRIVATE_MODEL_VIEW_MANIFEST="$RESULT_DIR/preregistration/private-model-view-manifest.json"

mkdir -p "$CODE_DIR" "$RESULT_DIR/logs" "$RESULT_DIR/stages" \
  "$RESULT_DIR/preregistration" "$RESULT_DIR/receipts" \
  "$RESULT_DIR/raw/shards" "$RESULT_DIR/runtime-cache/python" \
  "$RESULT_DIR/runtime-cache/triton" \
  "$RESULT_DIR/runtime-cache/torchinductor" \
  "$RESULT_DIR/runtime-cache/cuda"

# Copy into a new tree without deleting or modifying the preserved W source.
(cd "$OLD_CODE_DIR" && tar --exclude='__pycache__' --exclude='*.pyc' -cf - .) | \
  (cd "$CODE_DIR" && tar -xf -)
find "$CODE_DIR" -type f -exec chmod 0444 {} +
find "$CODE_DIR" -type d -exec chmod 0555 {} +

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="$RESULT_DIR/runtime-cache/python"
export TRITON_CACHE_DIR="$RESULT_DIR/runtime-cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$RESULT_DIR/runtime-cache/torchinductor"
export CUDA_CACHE_PATH="$RESULT_DIR/runtime-cache/cuda"

CURRENT_PHASE=source_preflight
record_failure() {
  local status=$1
  trap - ERR INT TERM
  date -u +%FT%TZ > "$RESULT_DIR/stages/FAILED"
  printf '%s\n' "$CURRENT_PHASE" > "$RESULT_DIR/stages/FAILED_PHASE"
  exit "$status"
}
trap 'record_failure $?' ERR
trap 'record_failure 130' INT
trap 'record_failure 143' TERM
date -u +%FT%TZ > "$RESULT_DIR/stages/00_start"

cp "$OLD_RUN_DIR/preregistration/code.sha256" \
  "$RESULT_DIR/preregistration/code.sha256"
cp "$OLD_RUN_DIR/preregistration/static-artifact.json" \
  "$RESULT_DIR/preregistration/static-artifact.json"
(cd "$CODE_DIR" && sha256sum -c "$RESULT_DIR/preregistration/code.sha256") \
  > "$RESULT_DIR/logs/code-preflight.log"

EXPECTED_CODE_LEDGER_RAW_SHA256=$(sha256sum \
  "$RESULT_DIR/preregistration/code.sha256" | awk '{print $1}')
if [[ "$EXPECTED_CODE_LEDGER_RAW_SHA256" != \
  "837f7a488d75cbedbc01e35a236a97f00b85259746e6a368b7aeec873045e94a" ]]; then
  echo "preserved W code-ledger receipt drift" >&2
  exit 2
fi

STATIC_ARTIFACT_SHA256=$(
  env PYTHONPATH="$CODE_DIR" "$PYTHON" -B "$MANIFEST_BUILDER" \
    digest-json --input "$RESULT_DIR/preregistration/static-artifact.json"
)
PROTOCOL_MANIFEST_SHA256=$(sha256sum "$PROTOCOL_MANIFEST" | awk '{print $1}')
RR2_INPUT_MANIFEST_SHA256=$(sha256sum "$RR2_INPUT_MANIFEST" | awk '{print $1}')
MODEL_ARTIFACT_LEDGER_SHA256=$(sha256sum "$MODEL_ARTIFACT_LEDGER" | awk '{print $1}')
MODEL_WEIGHT_LEDGER_SHA256=$(sha256sum "$MODEL_WEIGHT_LEDGER" | awk '{print $1}')

capture_identity() {
  local output=$1
  shift
  "$PYTHON" -I -B "$IDENTITY_BUILDER" capture \
    --source-root "$CODE_DIR" --python "$PYTHON" \
    --cache "python=$PYTHONPYCACHEPREFIX" \
    --cache "triton=$TRITON_CACHE_DIR" \
    --cache "torchinductor=$TORCHINDUCTOR_CACHE_DIR" \
    --cache "cuda=$CUDA_CACHE_PATH" \
    --command-file "rank0_recapture_launcher=$SELF" \
    --command-file "runner=$RUNNER" \
    --command-file "manifest_builder=$MANIFEST_BUILDER" \
    --command-file "execution_identity_builder=$IDENTITY_BUILDER" \
    --command-template \
      'CUDA_VISIBLE_DEVICES=<GPU receipt rank-0 UUID> python -B <W-reviewed runner> --stage shard --rank 0 <receipt-bound arguments>' \
    --output "$output" "$@"
}

capture_identity \
  "$RESULT_DIR/preregistration/execution-identity-preregistration.json" \
  --require-empty-caches
date -u +%FT%TZ > "$RESULT_DIR/stages/01_source_and_execution_identity_ok"

CURRENT_PHASE=private_model_view
env PYTHONPATH="$CODE_DIR" "$PYTHON" -B "$MANIFEST_BUILDER" \
  materialize-private-model-view \
  --source-model-dir "$SOURCE_MODEL_DIR" \
  --private-model-view "$MODEL_VIEW" \
  --model-artifact-ledger "$MODEL_ARTIFACT_LEDGER" \
  --expected-model-artifact-ledger-raw-sha256 \
    "$MODEL_ARTIFACT_LEDGER_SHA256" \
  --model-weight-ledger "$MODEL_WEIGHT_LEDGER" \
  --expected-model-weight-ledger-raw-sha256 "$MODEL_WEIGHT_LEDGER_SHA256" \
  --model-id "Qwen/Qwen3.5-35B-A3B" \
  --model-revision "59d61f3ce65a6d9863b86d2e96597125219dc754" \
  --manifest-output "$PRIVATE_MODEL_VIEW_MANIFEST" \
  > "$RESULT_DIR/logs/private-model-view-materialization.json"
PRIVATE_MODEL_VIEW_MANIFEST_SHA256=$(sha256sum \
  "$PRIVATE_MODEL_VIEW_MANIFEST" | awk '{print $1}')
date -u +%FT%TZ > "$RESULT_DIR/stages/01_private_model_view_ok"

CURRENT_PHASE=receipt_preregistration
RUN_ID=$(
  env PYTHONPATH="$CODE_DIR" "$PYTHON" -B "$MANIFEST_BUILDER" \
    run-id-receipt \
    --static-artifact-sha256 "$STATIC_ARTIFACT_SHA256" \
    --protocol-manifest-sha256 "$PROTOCOL_MANIFEST_SHA256" \
    --output "$RESULT_DIR/receipts/run-id-receipt.json"
)
if [[ ! "$RUN_ID" =~ ^[0-9a-f]{32}$ ]]; then
  echo "new debug run ID is not 128-bit lowercase hex" >&2
  exit 2
fi
RUN_ID_RECEIPT_SHA256=$(
  env PYTHONPATH="$CODE_DIR" "$PYTHON" -B "$MANIFEST_BUILDER" \
    digest-json --input "$RESULT_DIR/receipts/run-id-receipt.json"
)
nvidia-smi \
  --query-gpu=index,uuid,name,memory.total,compute_cap \
  --format=csv,noheader,nounits \
  > "$RESULT_DIR/preregistration/gpu-assignment-inventory.csv"
env PYTHONPATH="$CODE_DIR" "$PYTHON" -B "$MANIFEST_BUILDER" \
  gpu-assignment-receipt \
  --inventory "$RESULT_DIR/preregistration/gpu-assignment-inventory.csv" \
  --output "$RESULT_DIR/receipts/gpu-assignment-receipt.json" \
  > "$RESULT_DIR/logs/gpu-assignment-receipt-build.json"
GPU_ASSIGNMENT_RECEIPT_SHA256=$(sha256sum \
  "$RESULT_DIR/receipts/gpu-assignment-receipt.json" | awk '{print $1}')
GPU_UUID=$(
  "$PYTHON" -I -B - "$RESULT_DIR/receipts/gpu-assignment-receipt.json" <<'PY'
import json,sys
value=json.load(open(sys.argv[1],encoding="utf-8"))
assert len(value["rows"]) == 8
assert [row["rank"] for row in value["rows"]] == list(range(8))
print(value["rows"][0]["uuid"])
PY
)
date -u +%FT%TZ > "$RESULT_DIR/stages/02_receipts_bound"

CURRENT_PHASE=model_load_lease
LEASE_CONTROL_FIFO="$RESULT_DIR/receipts/model-load-lease-control.fifo"
LEASE_EVENT_FIFO="$RESULT_DIR/receipts/model-load-lease-events.fifo"
mkfifo -m 600 "$LEASE_CONTROL_FIFO" "$LEASE_EVENT_FIFO"
timeout --signal=TERM --kill-after=60s 21600s \
  env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  "$PYTHON" -I -B -c \
  'import runpy,signal,sys; signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGIO}); script=sys.argv[1]; sys.path.insert(0,sys.argv[2]); sys.argv=[script,*sys.argv[3:]]; runpy.run_path(script,run_name="__main__")' \
  "$MANIFEST_BUILDER" "$CODE_DIR" \
  model-load-lease-keeper \
  --model-view "$MODEL_VIEW" \
  --model-weight-ledger "$MODEL_WEIGHT_LEDGER" \
  --expected-model-weight-ledger-raw-sha256 "$MODEL_WEIGHT_LEDGER_SHA256" \
  --expected-model-artifact-ledger-raw-sha256 "$MODEL_ARTIFACT_LEDGER_SHA256" \
  --model-view-manifest "$PRIVATE_MODEL_VIEW_MANIFEST" \
  --expected-model-view-manifest-raw-sha256 \
    "$PRIVATE_MODEL_VIEW_MANIFEST_SHA256" \
  --run-id "$RUN_ID" \
  --authority-output "$RESULT_DIR/receipts/model-load-authority.json" \
  --closure-output "$RESULT_DIR/receipts/model-load-closure.json" \
  < "$LEASE_CONTROL_FIFO" > "$LEASE_EVENT_FIFO" \
  2> "$RESULT_DIR/logs/model-load-lease-keeper.log" &
LEASE_KEEPER_PID=$!
exec 8> "$LEASE_CONTROL_FIFO"
exec 9< "$LEASE_EVENT_FIFO"
IFS= read -r MODEL_LOAD_READY <&9
if [[ ! "$MODEL_LOAD_READY" =~ ^READY[[:space:]]([0-9a-f]{64})$ ]]; then
  echo "model-load lease keeper READY schema drift" >&2
  exit 2
fi
MODEL_LOAD_AUTHORITY_SHA256="${BASH_REMATCH[1]}"
date -u +%FT%TZ > "$RESULT_DIR/stages/03_model_load_authority_ok"

CURRENT_PHASE=rank0_shard
set +e
CUDA_VISIBLE_DEVICES="$GPU_UUID" PYTHONPATH="$CODE_DIR" \
  timeout --signal=TERM --kill-after=60s 21600s \
  "$PYTHON" -B "$RUNNER" \
  --stage shard \
  --rank 0 \
  --run-id "$RUN_ID" \
  --artifact-root "$RESULT_DIR/raw" \
  --static-artifact "$RESULT_DIR/preregistration/static-artifact.json" \
  --expected-static-sha256 "$STATIC_ARTIFACT_SHA256" \
  --rr2-input-manifest "$RR2_INPUT_MANIFEST" \
  --expected-rr2-input-manifest-sha256 "$RR2_INPUT_MANIFEST_SHA256" \
  --pg19-data "$PG19_DATA" \
  --pg19-manifest "$PG19_MANIFEST" \
  --prior-capacity-manifest "$PRIOR_CAPACITY_MANIFEST" \
  --model-dir "$MODEL_VIEW" \
  --code-ledger "$RESULT_DIR/preregistration/code.sha256" \
  --model-artifact-ledger "$MODEL_ARTIFACT_LEDGER" \
  --model-weight-ledger "$MODEL_WEIGHT_LEDGER" \
  --protocol-manifest "$PROTOCOL_MANIFEST" \
  --run-id-receipt "$RESULT_DIR/receipts/run-id-receipt.json" \
  --expected-run-id-receipt-sha256 "$RUN_ID_RECEIPT_SHA256" \
  --gpu-assignment-receipt "$RESULT_DIR/receipts/gpu-assignment-receipt.json" \
  --expected-gpu-assignment-receipt-raw-sha256 \
    "$GPU_ASSIGNMENT_RECEIPT_SHA256" \
  --model-load-authority "$RESULT_DIR/receipts/model-load-authority.json" \
  --expected-model-load-authority-raw-sha256 \
    "$MODEL_LOAD_AUTHORITY_SHA256" \
  --private-model-view-manifest "$PRIVATE_MODEL_VIEW_MANIFEST" \
  --expected-private-model-view-manifest-raw-sha256 \
    "$PRIVATE_MODEL_VIEW_MANIFEST_SHA256" \
  --expected-gpu-uuid "$GPU_UUID" \
  --output "$RESULT_DIR/raw/shards/forkaudit-shard-0.json" \
  > "$RESULT_DIR/logs/shard-rank-0.log" 2>&1
SHARD_STATUS=$?
set -e

printf 'CLOSE %s\n' "$MODEL_LOAD_AUTHORITY_SHA256" >&8
IFS= read -r MODEL_LOAD_CLOSED <&9
exec 8>&-
exec 9<&-
wait "$LEASE_KEEPER_PID"
unlink "$LEASE_CONTROL_FIFO"
unlink "$LEASE_EVENT_FIFO"
if [[ ! "$MODEL_LOAD_CLOSED" =~ ^CLOSED[[:space:]]([0-9a-f]{64})$ ]]; then
  echo "model-load lease keeper CLOSED schema drift" >&2
  exit 2
fi
MODEL_LOAD_CLOSURE_SHA256="${BASH_REMATCH[1]}"
if [[ "$SHARD_STATUS" -ne 0 ]]; then
  echo "rank-0 shard recapture failed with status $SHARD_STATUS" >&2
  exit "$SHARD_STATUS"
fi

CURRENT_PHASE=terminal_verification
"$PYTHON" -I -B - "$RESULT_DIR/raw/shards/forkaudit-shard-0.json" \
  "$RUN_ID" "$STATIC_ARTIFACT_SHA256" "$GPU_UUID" <<'PY' \
  > "$RESULT_DIR/logs/rank0-terminal-audit.json"
import json,sys
value=json.load(open(sys.argv[1],encoding="utf-8"))
assert value["schema_version"] == "qcomem-forkaudit-review-shard-v1"
assert value["status"] == "completed_formal_gpu_shard"
assert value["artifact_mode"] == "formal_gpu"
assert value["rank"] == 0 and value["world_size"] == 8
assert value["run_id"] == sys.argv[2]
assert value["static_artifact_sha256"] == sys.argv[3]
assert value["hardware_audit"]["uuid"] == sys.argv[4]
print(json.dumps({
    "status":"independent_rank0_primary_receipt_recaptured",
    "run_id":value["run_id"],
    "rank":value["rank"],
},sort_keys=True))
PY

capture_identity "$RESULT_DIR/receipts/execution-identity-terminal.json"
"$PYTHON" -I -B "$IDENTITY_BUILDER" verify-stable \
  --before "$RESULT_DIR/preregistration/execution-identity-preregistration.json" \
  --after "$RESULT_DIR/receipts/execution-identity-terminal.json" \
  --output "$RESULT_DIR/receipts/execution-identity-verification.json"
(cd "$CODE_DIR" && sha256sum -c "$RESULT_DIR/preregistration/code.sha256") \
  > "$RESULT_DIR/logs/code-terminal.log"
if find "$CODE_DIR" -name '__pycache__' -o -name '*.pyc' | grep -q .; then
  echo "runtime-generated bytecode drifted into the source tree" >&2
  exit 2
fi

(
  cd "$RESULT_DIR"
  find preregistration raw receipts runtime-cache -type f -print0 | \
    LC_ALL=C sort -z | xargs -0 sha256sum
) > "$RESULT_DIR/primary-receipt-bundle.sha256"
(cd "$RESULT_DIR" && sha256sum -c primary-receipt-bundle.sha256) \
  > "$RESULT_DIR/logs/primary-receipt-bundle-integrity.log"

cat > "$RESULT_DIR/receipt-summary.tsv" <<EOF
field\tvalue
scope\taffected-path debug; independent rank-0 primary receipt recapture
formal_evidence\tfalse
preserved_w_run\t$OLD_RUN_DIR
run_id\t$RUN_ID
rank\t0
gpu_uuid\t$GPU_UUID
static_artifact_canonical_sha256\t$STATIC_ARTIFACT_SHA256
protocol_manifest_raw_sha256\t$PROTOCOL_MANIFEST_SHA256
code_ledger_raw_sha256\t$EXPECTED_CODE_LEDGER_RAW_SHA256
run_id_receipt_canonical_sha256\t$RUN_ID_RECEIPT_SHA256
gpu_assignment_receipt_raw_sha256\t$GPU_ASSIGNMENT_RECEIPT_SHA256
model_load_authority_raw_sha256\t$MODEL_LOAD_AUTHORITY_SHA256
model_load_closure_raw_sha256\t$MODEL_LOAD_CLOSURE_SHA256
rank0_shard_raw_sha256\t$(sha256sum "$RESULT_DIR/raw/shards/forkaudit-shard-0.json" | awk '{print $1}')
execution_identity_terminal_raw_sha256\t$(sha256sum "$RESULT_DIR/receipts/execution-identity-terminal.json" | awk '{print $1}')
execution_identity_verification_raw_sha256\t$(sha256sum "$RESULT_DIR/receipts/execution-identity-verification.json" | awk '{print $1}')
EOF
date -u +%FT%TZ > "$RESULT_DIR/stages/99_done"
trap - ERR INT TERM
