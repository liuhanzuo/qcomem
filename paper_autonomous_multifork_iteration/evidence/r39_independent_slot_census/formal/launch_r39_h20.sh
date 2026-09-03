#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

R39_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

: "${R39_RUN_DIR:?set a new absent output directory}"
: "${R39_GPU_INDEX:?set an integer GPU index in [0,7]}"
: "${R33_CODE_DIR:?}"
: "${R29_CODE_DIR:?}"
: "${IMPORTED_RR2_CODE_DIR:?}"
: "${IMPORTED_RR2_CODE_LEDGER_FILE:?}"
: "${ENV_DIR:?}"
: "${MODEL_DIR:?}"
: "${MODEL_WEIGHT_LEDGER_FILE:?}"
: "${MODEL_ARTIFACT_LEDGER_FILE:?}"
: "${PG19_DATA:?}"
: "${PG19_MANIFEST:?}"
: "${FROZEN_QUERY_BANKS:?}"

if [[ ! "$R39_GPU_INDEX" =~ ^[0-7]$ ]]; then
  echo "R39_GPU_INDEX must be an integer in [0,7]" >&2
  exit 2
fi

PYTHON="$ENV_DIR/bin/python"
PROTOCOL="$R39_ROOT/protocol.json"
GENERATOR="$R39_ROOT/scripts/generate_preexecution_census.py"
AUDITOR="$R39_ROOT/scripts/audit_independent_slot_census.py"
CONTROLS="$R39_ROOT/scripts/run_negative_controls.py"
AGGREGATOR="$R39_ROOT/scripts/aggregate_formal_run.py"
R33_LAUNCHER="$R33_CODE_DIR/r33_launch_h20_independent_capture_1gpu.sh"
R33_PREREG="$R33_CODE_DIR/preregistration.json"
R33_SOURCE_LEDGER="$R33_CODE_DIR/source-code.sha256"

test -x "$PYTHON"
test -f "$PROTOCOL"
test -f "$GENERATOR"
test -f "$AUDITOR"
test -f "$CONTROLS"
test -f "$AGGREGATOR"
test -f "$R33_LAUNCHER"
test -f "$R29_CODE_DIR/r29_run_independent_gdn_observer.py"
test ! -e "$R39_RUN_DIR"

mkdir -p "$R39_RUN_DIR/preexecution" "$R39_RUN_DIR/audit" \
  "$R39_RUN_DIR/logs" "$R39_RUN_DIR/receipts" "$R39_RUN_DIR/stages"

CURRENT_PHASE=r39_source_freeze
record_failure() {
  local status=$1
  trap - ERR INT TERM
  date -u +%FT%TZ > "$R39_RUN_DIR/stages/FAILED"
  printf '%s\n' "$CURRENT_PHASE" > "$R39_RUN_DIR/stages/FAILED_PHASE"
  exit "$status"
}
trap 'record_failure $?' ERR
trap 'record_failure 130' INT
trap 'record_failure 143' TERM
date -u +%FT%TZ > "$R39_RUN_DIR/stages/00_started"

(
  cd "$R39_ROOT"
  sha256sum \
    protocol.json \
    scripts/audit_independent_slot_census.py \
    scripts/run_negative_controls.py \
    scripts/generate_preexecution_census.py \
    scripts/aggregate_formal_run.py \
    formal/launch_r39_h20.sh \
    formal/launch_trial_1907358.sh \
    "$R33_CODE_DIR/r33_ipc_capture_protocol.py" \
    "$R33_CODE_DIR/r33_independent_capture_worker.py" \
    "$R33_CODE_DIR/r33_out_of_process_capture.py" \
    "$R33_CODE_DIR/r33_replay_independent_capture.py" \
    "$R33_CODE_DIR/r33_run_h20_independent_capture.py" \
    "$R33_CODE_DIR/r33_test_independent_capture.py" \
    "$R33_CODE_DIR/r33_launch_h20_independent_capture_1gpu.sh" \
    "$R33_CODE_DIR/source-code.sha256" \
    "$R33_CODE_DIR/preregistration.json" \
    "$R29_CODE_DIR/r29_independent_gdn_observer.py" \
    "$R29_CODE_DIR/r29_run_independent_gdn_observer.py"
) > "$R39_RUN_DIR/preexecution/r39-source-code.sha256"

CURRENT_PHASE=preexecution_census
CUDA_VISIBLE_DEVICES='' "$PYTHON" -B "$GENERATOR" \
  --protocol "$PROTOCOL" \
  --source-ledger "$R39_RUN_DIR/preexecution/r39-source-code.sha256" \
  --census-output "$R39_RUN_DIR/preexecution/expected-slot-census.json" \
  --receipt-output "$R39_RUN_DIR/preexecution/census-receipt.json" \
  > "$R39_RUN_DIR/logs/preexecution-census.log" 2>&1
EXPECTED_CENSUS_SHA256=$("$PYTHON" -I -B -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["census_semantic_sha256"])' \
  "$R39_RUN_DIR/preexecution/census-receipt.json")
test "${#EXPECTED_CENSUS_SHA256}" -eq 64
touch "$R39_RUN_DIR/stages/01_census_frozen_before_producer"

CURRENT_PHASE=fresh_h20_r33_capture
export R33_CODE_DIR R29_CODE_DIR IMPORTED_RR2_CODE_DIR
export IMPORTED_RR2_CODE_LEDGER_FILE ENV_DIR MODEL_DIR
export MODEL_WEIGHT_LEDGER_FILE MODEL_ARTIFACT_LEDGER_FILE
export PG19_DATA PG19_MANIFEST FROZEN_QUERY_BANKS
export SOURCE_LEDGER_FILE="$R33_SOURCE_LEDGER"
export PREREG_FILE="$R33_PREREG"
export RUN_DIR="$R39_RUN_DIR/r33-live"
export R33_GPU_INDEX="$R39_GPU_INDEX"
export EXPECTED_SOURCE_LEDGER_SHA256=08dee9b0e7f92edd65472618ebc3e9c2a38108296930255ffb62248bbe853319
export EXPECTED_PREREG_SHA256=67c001a04e61006967befe1ff1e26018b58bc02630b8e48dd40f16067a44ea65
export EXPECTED_IMPORTED_RR2_CODE_LEDGER_SHA256=837f7a488d75cbedbc01e35a236a97f00b85259746e6a368b7aeec873045e94a
export EXPECTED_MODEL_WEIGHT_LEDGER_SHA256=8314a82c9188b9b817193e039b0b0eb0636b328512f19b0c12455853b7e20014
export EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256=c0a23e9d3f9d220257af97b78fd97661f315f0c82a3a010b57a771e3eeefbbfb
export EXPECTED_PG19_SHA256=ef18893b6bfb0f4b8cb29eab85ccf2a0ef1fdb44606e9742a70405cca564e18c
export EXPECTED_PG19_MANIFEST_SHA256=5d789d67aa239f089e92de8a4267b86d2f1d2723d5f1370970883738f5f89a9c
export EXPECTED_PG19_WINDOWS_SHA256=39bc36bb2eb04d51122e66caaebfa72367c02b43b073f072a2da240ed068c166
export EXPECTED_FROZEN_QUERY_BANKS_SHA256=400921d147bc840e9802950dc542b002080f3f274661efbd5b4354ec364da7db
bash "$R33_LAUNCHER" > "$R39_RUN_DIR/logs/r33-launcher.log" 2>&1
touch "$R39_RUN_DIR/stages/02_fresh_h20_capture_and_r33_replay_passed"

RAW_INPUT="$R39_RUN_DIR/r33-live/raw/out-of-process-gdn-capture.json"
R33_REPLAY="$R39_RUN_DIR/r33-live/replay/out-of-process-gdn-replay.json"
RAW_SHA256=$(sha256sum "$RAW_INPUT" | awk '{print $1}')
printf '%s  %s\n' "$RAW_SHA256" "r33-live/raw/out-of-process-gdn-capture.json" \
  > "$R39_RUN_DIR/receipts/clean-raw-before-controls.sha256"

CURRENT_PHASE=independent_census_clean_audit
CUDA_VISIBLE_DEVICES='' "$PYTHON" -B "$AUDITOR" \
  --protocol "$PROTOCOL" \
  --input "$RAW_INPUT" \
  --expected-input-sha256 "$RAW_SHA256" \
  --preregistration "$R39_RUN_DIR/r33-live/preregistration/preregistration.json" \
  --expected-census-sha256 "$EXPECTED_CENSUS_SHA256" \
  --output "$R39_RUN_DIR/audit/clean-audit.json" \
  > "$R39_RUN_DIR/logs/clean-census-audit.log" 2>&1
touch "$R39_RUN_DIR/stages/03_live_capture_bound_to_preexecution_census"

CURRENT_PHASE=copy_only_negative_controls
CUDA_VISIBLE_DEVICES='' "$PYTHON" -B "$CONTROLS" \
  --protocol "$PROTOCOL" \
  --input "$RAW_INPUT" \
  --expected-input-sha256 "$RAW_SHA256" \
  --output "$R39_RUN_DIR/audit/negative-controls.json" \
  > "$R39_RUN_DIR/logs/negative-controls.log" 2>&1
test "$(sha256sum "$RAW_INPUT" | awk '{print $1}')" = "$RAW_SHA256"
printf '%s  %s\n' "$RAW_SHA256" "r33-live/raw/out-of-process-gdn-capture.json" \
  > "$R39_RUN_DIR/receipts/clean-raw-after-controls.sha256"
cmp "$R39_RUN_DIR/receipts/clean-raw-before-controls.sha256" \
  "$R39_RUN_DIR/receipts/clean-raw-after-controls.sha256"
touch "$R39_RUN_DIR/stages/04_copy_only_controls_failed_closed"

CURRENT_PHASE=formal_aggregation
CUDA_VISIBLE_DEVICES='' "$PYTHON" -B "$AGGREGATOR" \
  --protocol "$PROTOCOL" \
  --source-ledger "$R39_RUN_DIR/preexecution/r39-source-code.sha256" \
  --preexecution-census "$R39_RUN_DIR/preexecution/expected-slot-census.json" \
  --preexecution-receipt "$R39_RUN_DIR/preexecution/census-receipt.json" \
  --raw-input "$RAW_INPUT" \
  --r33-replay "$R33_REPLAY" \
  --clean-audit "$R39_RUN_DIR/audit/clean-audit.json" \
  --negative-controls "$R39_RUN_DIR/audit/negative-controls.json" \
  --output "$R39_RUN_DIR/audit/formal-aggregate.json" \
  > "$R39_RUN_DIR/logs/formal-aggregation.log" 2>&1
(
  cd "$R39_ROOT"
  sha256sum -c "$R39_RUN_DIR/preexecution/r39-source-code.sha256"
) > "$R39_RUN_DIR/logs/r39-terminal-source-integrity.log"
touch "$R39_RUN_DIR/stages/05_formal_aggregate_passed"

CURRENT_PHASE=terminal_receipt
(
  cd "$R39_RUN_DIR"
  find preexecution audit logs receipts stages r33-live -type f \
    ! -path '*/runtime-cache/*' \
    ! -name terminal-files.sha256 -print0 \
    | LC_ALL=C sort -z | xargs -0 sha256sum
) > "$R39_RUN_DIR/receipts/terminal-files.sha256"
date -u +%FT%TZ > "$R39_RUN_DIR/stages/06_complete"
