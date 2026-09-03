#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

: "${R33_CODE_DIR:?}"
: "${R29_CODE_DIR:?}"
: "${IMPORTED_RR2_CODE_DIR:?}"
: "${IMPORTED_RR2_CODE_LEDGER_FILE:?}"
: "${SOURCE_LEDGER_FILE:?}"
: "${PREREG_FILE:?}"
: "${RUN_DIR:?}"
: "${ENV_DIR:?}"
: "${MODEL_DIR:?}"
: "${MODEL_WEIGHT_LEDGER_FILE:?}"
: "${MODEL_ARTIFACT_LEDGER_FILE:?}"
: "${PG19_DATA:?}"
: "${PG19_MANIFEST:?}"
: "${FROZEN_QUERY_BANKS:?}"
: "${EXPECTED_SOURCE_LEDGER_SHA256:?}"
: "${EXPECTED_PREREG_SHA256:?}"
: "${EXPECTED_IMPORTED_RR2_CODE_LEDGER_SHA256:?}"
: "${EXPECTED_MODEL_WEIGHT_LEDGER_SHA256:?}"
: "${EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256:?}"
: "${EXPECTED_PG19_SHA256:?}"
: "${EXPECTED_PG19_MANIFEST_SHA256:?}"
: "${EXPECTED_PG19_WINDOWS_SHA256:?}"
: "${EXPECTED_FROZEN_QUERY_BANKS_SHA256:?}"

R33_GPU_INDEX="${R33_GPU_INDEX:-7}"
if [[ ! "$R33_GPU_INDEX" =~ ^[0-7]$ ]]; then
  printf 'R33_GPU_INDEX must be an integer in [0,7], found %s\n' "$R33_GPU_INDEX" >&2
  exit 2
fi

PYTHON="$ENV_DIR/bin/python"
RUNNER="$R33_CODE_DIR/r33_run_h20_independent_capture.py"
REPLAY="$R33_CODE_DIR/r33_replay_independent_capture.py"
TEST="$R33_CODE_DIR/r33_test_independent_capture.py"
RESULT="$RUN_DIR/raw/out-of-process-gdn-capture.json"
REPLAY_RESULT="$RUN_DIR/replay/out-of-process-gdn-replay.json"
LOGS="$RUN_DIR/logs"
STAGES="$RUN_DIR/stages"
RECEIPTS="$RUN_DIR/receipts"

test -x "$PYTHON"
test -f "$RUNNER"
test -f "$REPLAY"
test -f "$TEST"
test -f "$R29_CODE_DIR/r29_run_independent_gdn_observer.py"
test -f "$SOURCE_LEDGER_FILE"
test -f "$PREREG_FILE"
test -f "$IMPORTED_RR2_CODE_LEDGER_FILE"
test ! -e "$RUN_DIR"

test "$(sha256sum "$SOURCE_LEDGER_FILE" | awk '{print $1}')" = "$EXPECTED_SOURCE_LEDGER_SHA256"
test "$(sha256sum "$PREREG_FILE" | awk '{print $1}')" = "$EXPECTED_PREREG_SHA256"
test "$(sha256sum "$IMPORTED_RR2_CODE_LEDGER_FILE" | awk '{print $1}')" = "$EXPECTED_IMPORTED_RR2_CODE_LEDGER_SHA256"
test "$(sha256sum "$MODEL_WEIGHT_LEDGER_FILE" | awk '{print $1}')" = "$EXPECTED_MODEL_WEIGHT_LEDGER_SHA256"
test "$(sha256sum "$MODEL_ARTIFACT_LEDGER_FILE" | awk '{print $1}')" = "$EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256"
test "$(sha256sum "$PG19_DATA" | awk '{print $1}')" = "$EXPECTED_PG19_SHA256"
test "$(sha256sum "$PG19_MANIFEST" | awk '{print $1}')" = "$EXPECTED_PG19_MANIFEST_SHA256"
test "$(sha256sum "$FROZEN_QUERY_BANKS" | awk '{print $1}')" = "$EXPECTED_FROZEN_QUERY_BANKS_SHA256"

mkdir -p "$RUN_DIR/preregistration" "$RUN_DIR/raw" "$RUN_DIR/replay" \
  "$LOGS" "$STAGES" "$RECEIPTS" "$RUN_DIR/runtime-cache/python" \
  "$RUN_DIR/runtime-cache/triton" "$RUN_DIR/runtime-cache/torchinductor" \
  "$RUN_DIR/runtime-cache/cuda"

CURRENT_PHASE=source_preflight
record_failure() {
  local status=$1
  trap - ERR INT TERM
  date -u +%FT%TZ > "$STAGES/FAILED"
  printf '%s\n' "$CURRENT_PHASE" > "$STAGES/FAILED_PHASE"
  exit "$status"
}
trap 'record_failure $?' ERR
trap 'record_failure 130' INT
trap 'record_failure 143' TERM
date -u +%FT%TZ > "$STAGES/00_started"

(
  cd "$R33_CODE_DIR"
  sha256sum -c "$SOURCE_LEDGER_FILE"
) > "$LOGS/r33-source-integrity.log"
(
  cd "$IMPORTED_RR2_CODE_DIR"
  sha256sum -c "$IMPORTED_RR2_CODE_LEDGER_FILE"
) > "$LOGS/imported-rr2-code-integrity.log"
cp -- "$PREREG_FILE" "$RUN_DIR/preregistration/preregistration.json"
cp -- "$SOURCE_LEDGER_FILE" "$RUN_DIR/preregistration/source-code.sha256"
touch "$STAGES/01_preregistered_before_candidate_outputs"

export PYTHONPATH="$R33_CODE_DIR:$R29_CODE_DIR:$IMPORTED_RR2_CODE_DIR"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="$RUN_DIR/runtime-cache/python"
export TRITON_CACHE_DIR="$RUN_DIR/runtime-cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$RUN_DIR/runtime-cache/torchinductor"
export CUDA_CACHE_PATH="$RUN_DIR/runtime-cache/cuda"

CURRENT_PHASE=unit_tests
CUDA_VISIBLE_DEVICES='' "$PYTHON" -B "$TEST" -v > "$LOGS/unit-tests.log" 2>&1
touch "$STAGES/02_unit_tests_passed"

CURRENT_PHASE=gpu_assignment
nvidia-smi --query-gpu=uuid,name,memory.total --format=csv,noheader,nounits \
  > "$RECEIPTS/gpu-inventory.csv"
mapfile -t GPU_UUIDS < <(
  nvidia-smi --query-gpu=uuid --format=csv,noheader,nounits \
    | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' | sed '/^$/d'
)
test "${#GPU_UUIDS[@]}" -eq 8
"$PYTHON" -I -B - "$RECEIPTS/gpu-inventory.csv" <<'PY'
import csv
import sys
from pathlib import Path

rows = list(csv.reader(Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()))
assert len(rows) == 8
for row in rows:
    assert len(row) == 3
    uuid, name, memory = (cell.strip() for cell in row)
    assert uuid.startswith("GPU-")
    assert name == "NVIDIA H20-3e"
    assert int(memory) > 0
PY
GPU_UUID="${GPU_UUIDS[$R33_GPU_INDEX]}"
printf 'gpu_index,gpu_uuid\n%s,%s\n' "$R33_GPU_INDEX" "$GPU_UUID" \
  > "$RECEIPTS/selected-gpu.csv"
touch "$STAGES/03_gpu_assignment_verified"

CURRENT_PHASE=scientific_execution
CUDA_VISIBLE_DEVICES="$GPU_UUID" timeout --signal=TERM --kill-after=60s 10800s \
  "$PYTHON" -B "$RUNNER" \
  --output "$RESULT" \
  --preregistration "$RUN_DIR/preregistration/preregistration.json" \
  --expected-preregistration-sha256 "$EXPECTED_PREREG_SHA256" \
  --source-ledger "$RUN_DIR/preregistration/source-code.sha256" \
  --expected-source-ledger-sha256 "$EXPECTED_SOURCE_LEDGER_SHA256" \
  --candidate-code-ledger "$IMPORTED_RR2_CODE_LEDGER_FILE" \
  --expected-candidate-code-ledger-sha256 "$EXPECTED_IMPORTED_RR2_CODE_LEDGER_SHA256" \
  --model-dir "$MODEL_DIR" \
  --model-weight-ledger "$MODEL_WEIGHT_LEDGER_FILE" \
  --model-artifact-ledger "$MODEL_ARTIFACT_LEDGER_FILE" \
  --expected-weight-ledger-sha256 "$EXPECTED_MODEL_WEIGHT_LEDGER_SHA256" \
  --expected-artifact-ledger-sha256 "$EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256" \
  --pg19-data "$PG19_DATA" \
  --pg19-manifest "$PG19_MANIFEST" \
  --expected-pg19-sha256 "$EXPECTED_PG19_SHA256" \
  --expected-pg19-manifest-sha256 "$EXPECTED_PG19_MANIFEST_SHA256" \
  --expected-windows-sha256 "$EXPECTED_PG19_WINDOWS_SHA256" \
  --frozen-query-banks "$FROZEN_QUERY_BANKS" \
  --expected-query-banks-sha256 "$EXPECTED_FROZEN_QUERY_BANKS_SHA256" \
  --expected-gpu-uuid "$GPU_UUID" \
  > "$LOGS/scientific-execution.log" 2>&1
touch "$STAGES/04_scientific_execution_complete"

CURRENT_PHASE=independent_cpu_replay
RESULT_SHA256="$(sha256sum "$RESULT" | awk '{print $1}')"
CUDA_VISIBLE_DEVICES='' "$PYTHON" -B "$REPLAY" \
  --input "$RESULT" \
  --expected-input-sha256 "$RESULT_SHA256" \
  --preregistration "$RUN_DIR/preregistration/preregistration.json" \
  --expected-preregistration-sha256 "$EXPECTED_PREREG_SHA256" \
  --output "$REPLAY_RESULT" \
  > "$LOGS/independent-replay.log" 2>&1
touch "$STAGES/05_independent_replay_passed"

CURRENT_PHASE=terminal_receipt
(
  cd "$RUN_DIR"
  find preregistration raw replay logs receipts stages -type f \
    ! -name terminal-files.sha256 -print0 \
    | LC_ALL=C sort -z | xargs -0 sha256sum
) > "$RECEIPTS/terminal-files.sha256"
date -u +%FT%TZ > "$STAGES/06_complete"
