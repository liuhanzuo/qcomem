#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

DUAL_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

: "${DUAL_RUN_DIR:?set a new absent output directory}"
: "${DUAL_GPU_INDEX:?set an integer GPU index in [0,7]}"
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

if [[ ! "$DUAL_GPU_INDEX" =~ ^[0-7]$ ]]; then
  echo "DUAL_GPU_INDEX must be an integer in [0,7]" >&2
  exit 2
fi

PYTHON="$ENV_DIR/bin/python"
PREREG="$DUAL_ROOT/preregistration.json"
SLOT_PROTOCOL="$DUAL_ROOT/slot_protocol.json"
SOURCE_LEDGER="$DUAL_ROOT/source-code.sha256"
GENERATOR="$DUAL_ROOT/vendor/r39/generate_preexecution_census.py"
AUDITOR="$DUAL_ROOT/vendor/r39/audit_independent_slot_census.py"
VERIFIER="$DUAL_ROOT/scripts/verify_dual_producer_repeat.py"
TEST_DIR="$DUAL_ROOT/tests"
R33_LAUNCHER="$R33_CODE_DIR/r33_launch_h20_independent_capture_1gpu.sh"
R33_PREREG="$R33_CODE_DIR/preregistration.json"
R33_SOURCE_LEDGER="$R33_CODE_DIR/source-code.sha256"

EXPECTED_DUAL_PREREG_SHA256=fe3583907cd0cfadb4045509d3a103ab64052452e00184edcf72835742510b72

test -x "$PYTHON"
test -f "$PREREG"
test -f "$SLOT_PROTOCOL"
test -f "$SOURCE_LEDGER"
test -f "$GENERATOR"
test -f "$AUDITOR"
test -f "$VERIFIER"
test -f "$R33_LAUNCHER"
test -f "$R29_CODE_DIR/r29_run_independent_gdn_observer.py"
test ! -e "$DUAL_RUN_DIR"
test "$(sha256sum "$PREREG" | awk '{print $1}')" = "$EXPECTED_DUAL_PREREG_SHA256"

mkdir -p "$DUAL_RUN_DIR/preexecution" "$DUAL_RUN_DIR/audit" \
  "$DUAL_RUN_DIR/logs" "$DUAL_RUN_DIR/receipts" "$DUAL_RUN_DIR/stages"

CURRENT_PHASE=dual_source_preflight
record_failure() {
  local status=$1
  trap - ERR INT TERM
  date -u +%FT%TZ > "$DUAL_RUN_DIR/stages/FAILED"
  printf '%s\n' "$CURRENT_PHASE" > "$DUAL_RUN_DIR/stages/FAILED_PHASE"
  exit "$status"
}
trap 'record_failure $?' ERR
trap 'record_failure 130' INT
trap 'record_failure 143' TERM
date -u +%FT%TZ > "$DUAL_RUN_DIR/stages/00_started"

CURRENT_PHASE=source_integrity_and_cpu_tests
(
  cd "$DUAL_ROOT"
  sha256sum -c "$SOURCE_LEDGER"
) > "$DUAL_RUN_DIR/logs/dual-source-integrity.log"
CUDA_VISIBLE_DEVICES='' \
PYTHONPATH="$DUAL_ROOT/scripts:$DUAL_ROOT/vendor/r39" \
  "$PYTHON" -B -m unittest discover -v -s "$TEST_DIR" \
  > "$DUAL_RUN_DIR/logs/dual-unit-tests.log" 2>&1
cp -- "$PREREG" "$DUAL_RUN_DIR/preexecution/preregistration.json"
cp -- "$SOURCE_LEDGER" "$DUAL_RUN_DIR/preexecution/source-code.sha256"
cp -- "$SLOT_PROTOCOL" "$DUAL_RUN_DIR/preexecution/slot_protocol.json"
touch "$DUAL_RUN_DIR/stages/01_source_and_tests_passed"

CURRENT_PHASE=preexecution_census_before_both_producers
CUDA_VISIBLE_DEVICES='' \
PYTHONPATH="$DUAL_ROOT/vendor/r39" \
  "$PYTHON" -B "$GENERATOR" \
  --protocol "$SLOT_PROTOCOL" \
  --source-ledger "$SOURCE_LEDGER" \
  --census-output "$DUAL_RUN_DIR/preexecution/expected-slot-census.json" \
  --receipt-output "$DUAL_RUN_DIR/preexecution/census-receipt.json" \
  > "$DUAL_RUN_DIR/logs/preexecution-census.log" 2>&1
EXPECTED_CENSUS_SHA256=$("$PYTHON" -I -B -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["census_semantic_sha256"])' \
  "$DUAL_RUN_DIR/preexecution/census-receipt.json")
test "${#EXPECTED_CENSUS_SHA256}" -eq 64
touch "$DUAL_RUN_DIR/stages/02_census_frozen_before_both_producers"

export R33_CODE_DIR R29_CODE_DIR IMPORTED_RR2_CODE_DIR
export IMPORTED_RR2_CODE_LEDGER_FILE ENV_DIR MODEL_DIR
export MODEL_WEIGHT_LEDGER_FILE MODEL_ARTIFACT_LEDGER_FILE
export PG19_DATA PG19_MANIFEST FROZEN_QUERY_BANKS
export SOURCE_LEDGER_FILE="$R33_SOURCE_LEDGER"
export PREREG_FILE="$R33_PREREG"
export R33_GPU_INDEX="$DUAL_GPU_INDEX"
export EXPECTED_SOURCE_LEDGER_SHA256=08dee9b0e7f92edd65472618ebc3e9c2a38108296930255ffb62248bbe853319
export EXPECTED_PREREG_SHA256=67c001a04e61006967befe1ff1e26018b58bc02630b8e48dd40f16067a44ea65
export EXPECTED_IMPORTED_RR2_CODE_LEDGER_SHA256=837f7a488d75cbedbc01e35a236a97f00b85259746e6a368b7aeec873045e94a
export EXPECTED_MODEL_WEIGHT_LEDGER_SHA256=8314a82c9188b9b817193e039b0b0eb0636b328512f19b0c12455853b7e20014
export EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256=c0a23e9d3f9d220257af97b78fd97661f315f0c82a3a010b57a771e3eeefbbfb
export EXPECTED_PG19_SHA256=ef18893b6bfb0f4b8cb29eab85ccf2a0ef1fdb44606e9742a70405cca564e18c
export EXPECTED_PG19_MANIFEST_SHA256=5d789d67aa239f089e92de8a4267b86d2f1d2723d5f1370970883738f5f89a9c
export EXPECTED_PG19_WINDOWS_SHA256=39bc36bb2eb04d51122e66caaebfa72367c02b43b073f072a2da240ed068c166
export EXPECTED_FROZEN_QUERY_BANKS_SHA256=400921d147bc840e9802950dc542b002080f3f274661efbd5b4354ec364da7db

run_one_producer() {
  local label=$1
  local stage_marker=$2
  local producer_dir="$DUAL_RUN_DIR/producer-$label"
  local raw="$producer_dir/raw/out-of-process-gdn-capture.json"
  local audit="$DUAL_RUN_DIR/audit/producer-$label-census-audit.json"
  export RUN_DIR="$producer_dir"
  CURRENT_PHASE="producer_${label}_fresh_h20_capture"
  bash "$R33_LAUNCHER" \
    > "$DUAL_RUN_DIR/logs/producer-$label-r33-launcher.log" 2>&1
  local raw_sha
  raw_sha=$(sha256sum "$raw" | awk '{print $1}')
  printf '%s  %s\n' "$raw_sha" \
    "producer-$label/raw/out-of-process-gdn-capture.json" \
    > "$DUAL_RUN_DIR/receipts/producer-$label-raw.sha256"
  CURRENT_PHASE="producer_${label}_independent_census_audit"
  CUDA_VISIBLE_DEVICES='' \
  PYTHONPATH="$DUAL_ROOT/vendor/r39" \
    "$PYTHON" -B "$AUDITOR" \
    --protocol "$SLOT_PROTOCOL" \
    --input "$raw" \
    --expected-input-sha256 "$raw_sha" \
    --preregistration "$producer_dir/preregistration/preregistration.json" \
    --expected-census-sha256 "$EXPECTED_CENSUS_SHA256" \
    --output "$audit" \
    > "$DUAL_RUN_DIR/logs/producer-$label-census-audit.log" 2>&1
  test "$(sha256sum "$raw" | awk '{print $1}')" = "$raw_sha"
  touch "$DUAL_RUN_DIR/stages/$stage_marker"
}

run_one_producer a 03_producer_a_capture_replay_and_census_audit_passed
run_one_producer b 04_producer_b_capture_replay_and_census_audit_passed

cmp "$DUAL_RUN_DIR/producer-a/receipts/selected-gpu.csv" \
  "$DUAL_RUN_DIR/producer-b/receipts/selected-gpu.csv"

CURRENT_PHASE=dual_producer_exact_closure
PYTHONPATH="$DUAL_ROOT/scripts:$DUAL_ROOT/vendor/r39" \
  CUDA_VISIBLE_DEVICES='' "$PYTHON" -B "$VERIFIER" \
  --preregistration "$PREREG" \
  --slot-protocol "$SLOT_PROTOCOL" \
  --source-ledger "$SOURCE_LEDGER" \
  --census "$DUAL_RUN_DIR/preexecution/expected-slot-census.json" \
  --census-receipt "$DUAL_RUN_DIR/preexecution/census-receipt.json" \
  --producer-a "$DUAL_RUN_DIR/producer-a/raw/out-of-process-gdn-capture.json" \
  --producer-b "$DUAL_RUN_DIR/producer-b/raw/out-of-process-gdn-capture.json" \
  --producer-a-replay "$DUAL_RUN_DIR/producer-a/replay/out-of-process-gdn-replay.json" \
  --producer-b-replay "$DUAL_RUN_DIR/producer-b/replay/out-of-process-gdn-replay.json" \
  --producer-a-audit "$DUAL_RUN_DIR/audit/producer-a-census-audit.json" \
  --producer-b-audit "$DUAL_RUN_DIR/audit/producer-b-census-audit.json" \
  --output "$DUAL_RUN_DIR/audit/dual-producer-summary.json" \
  > "$DUAL_RUN_DIR/logs/dual-producer-verification.log" 2>&1
"$PYTHON" -I -B - "$DUAL_RUN_DIR/audit/dual-producer-summary.json" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert summary["passed"] is True
assert summary["captures_per_producer"] == 6
assert summary["matched_semantic_coordinates"] == 1080
assert summary["matched_content_digests"] == 1080
assert summary["matched_stable_descriptors"] == 1080
assert summary["matched_relation_labels"] == 96660
assert summary["numeric_tolerance"] == 0
assert summary["canonical_semantic_fallback"] is False
PY
touch "$DUAL_RUN_DIR/stages/05_dual_exact_closure_passed"

CURRENT_PHASE=terminal_source_integrity
(
  cd "$DUAL_ROOT"
  sha256sum -c "$SOURCE_LEDGER"
) > "$DUAL_RUN_DIR/logs/dual-terminal-source-integrity.log"
touch "$DUAL_RUN_DIR/stages/06_terminal_source_integrity_passed"

CURRENT_PHASE=terminal_ledger
date -u +%FT%TZ > "$DUAL_RUN_DIR/stages/07_scientific_execution_complete"
(
  cd "$DUAL_RUN_DIR"
  find preexecution audit logs receipts stages producer-a producer-b -type f \
    ! -path '*/runtime-cache/*' \
    ! -name terminal-files.sha256 -print0 \
    | LC_ALL=C sort -z | xargs -0 sha256sum
) > "$DUAL_RUN_DIR/receipts/terminal-files.sha256"
date -u +%FT%TZ > "$DUAL_RUN_DIR/stages/08_complete"
