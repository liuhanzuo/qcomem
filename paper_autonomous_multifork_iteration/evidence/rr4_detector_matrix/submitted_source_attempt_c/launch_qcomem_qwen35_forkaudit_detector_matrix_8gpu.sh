#!/usr/bin/env bash
set -euo pipefail

: "${CODE_DIR:?}"
: "${IMPORTED_RR2_CODE_DIR:?}"
: "${IMPORTED_RR2_CODE_LEDGER:?}"
: "${MODEL_DIR:?}"
: "${MODEL_WEIGHT_LEDGER_FILE:?}"
: "${MODEL_ARTIFACT_LEDGER_FILE:?}"
: "${PG19_DATA:?}"
: "${PG19_MANIFEST:?}"
: "${FROZEN_QUERY_BANKS:?}"
: "${ORIGINAL_RR2_RUN_ROOT:?}"
: "${RUN_DIR:?}"
: "${ENV_DIR:?}"
: "${EXPECTED_RUNNER_SHA256:?}"
: "${EXPECTED_BUILDER_SHA256:?}"
: "${EXPECTED_IMPORTED_RR2_CODE_LEDGER_SHA256:?}"
: "${EXPECTED_MODEL_WEIGHT_LEDGER_SHA256:?}"
: "${EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256:?}"
: "${EXPECTED_PG19_SHA256:?}"
: "${EXPECTED_PG19_MANIFEST_SHA256:?}"
: "${EXPECTED_PG19_WINDOWS_SHA256:?}"
: "${EXPECTED_FROZEN_QUERY_BANKS_SHA256:?}"

PYTHON="$ENV_DIR/bin/python"
RUNNER="$CODE_DIR/run_qcomem_qwen35_forkaudit_detector_matrix.py"
BUILDER="$CODE_DIR/build_qcomem_qwen35_forkaudit_detector_matrix.py"
ORIGINAL_RECEIPTS="$ORIGINAL_RR2_RUN_ROOT/receipts/detached-receipt-manifest.json"
PREREG="$RUN_DIR/preregistration/detector-matrix-plan.json"
RAW="$RUN_DIR/raw"
LOGS="$RUN_DIR/logs"
STAGES="$RUN_DIR/stages"

test -x "$PYTHON"
test -f "$RUNNER"
test -f "$BUILDER"
test -f "$IMPORTED_RR2_CODE_DIR/run_qcomem_qwen35_forkaudit_review_revision.py"
test -f "$ORIGINAL_RECEIPTS"
test ! -e "$RUN_DIR"
mkdir -p "$RUN_DIR/preregistration" "$RAW" "$LOGS" "$STAGES"
touch "$STAGES/00_start"

test "$(sha256sum "$RUNNER" | awk '{print $1}')" = "$EXPECTED_RUNNER_SHA256"
test "$(sha256sum "$BUILDER" | awk '{print $1}')" = "$EXPECTED_BUILDER_SHA256"
test "$(sha256sum "$IMPORTED_RR2_CODE_LEDGER" | awk '{print $1}')" = "$EXPECTED_IMPORTED_RR2_CODE_LEDGER_SHA256"
test "$(sha256sum "$MODEL_WEIGHT_LEDGER_FILE" | awk '{print $1}')" = "$EXPECTED_MODEL_WEIGHT_LEDGER_SHA256"
test "$(sha256sum "$MODEL_ARTIFACT_LEDGER_FILE" | awk '{print $1}')" = "$EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256"
test "$(sha256sum "$PG19_DATA" | awk '{print $1}')" = "$EXPECTED_PG19_SHA256"
test "$(sha256sum "$PG19_MANIFEST" | awk '{print $1}')" = "$EXPECTED_PG19_MANIFEST_SHA256"
test "$(sha256sum "$FROZEN_QUERY_BANKS" | awk '{print $1}')" = "$EXPECTED_FROZEN_QUERY_BANKS_SHA256"

export PYTHONPATH="$CODE_DIR:$IMPORTED_RR2_CODE_DIR"
export PYTHONDONTWRITEBYTECODE=1

"$PYTHON" -B "$BUILDER" \
  --stage preregister \
  --output "$PREREG" \
  --runner "$RUNNER" \
  --original-rr2-run-id 372384bd37cf7640ca210537a4360e1a \
  --original-receipt-manifest "$ORIGINAL_RECEIPTS" \
  --imported-rr2-code-ledger-sha256 "$EXPECTED_IMPORTED_RR2_CODE_LEDGER_SHA256"
PREREG_SHA=$(sha256sum "$PREREG" | awk '{print $1}')
touch "$STAGES/01_preregistered_before_outputs"

nvidia-smi --query-gpu=uuid,name,memory.total --format=csv,noheader,nounits > "$RUN_DIR/preregistration/gpu-inventory.csv"
mapfile -t GPU_UUIDS < <(nvidia-smi --query-gpu=uuid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')
test "${#GPU_UUIDS[@]}" -eq 8

pids=()
for rank in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES="${GPU_UUIDS[$rank]}" "$PYTHON" -B "$RUNNER" \
    --stage rank \
    --rank "$rank" \
    --output "$RAW/detector-matrix-rank-$rank.json" \
    --preregistration "$PREREG" \
    --expected-preregistration-sha256 "$PREREG_SHA" \
    --original-receipt-manifest "$ORIGINAL_RECEIPTS" \
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
    --expected-gpu-uuid "${GPU_UUIDS[$rank]}" \
    > "$LOGS/rank-$rank.log" 2>&1 &
  pids+=("$!")
done

failed=0
for index in $(seq 0 7); do
  if ! wait "${pids[$index]}"; then
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  touch "$STAGES/02_rank_failure"
  exit 2
fi
touch "$STAGES/02_all_ranks_complete"

"$PYTHON" -B "$BUILDER" \
  --stage aggregate \
  --output "$RUN_DIR/detector-matrix-summary.json" \
  --preregistration "$PREREG" \
  --expected-preregistration-sha256 "$PREREG_SHA" \
  --original-receipt-manifest "$ORIGINAL_RECEIPTS" \
  --original-rr2-root "$ORIGINAL_RR2_RUN_ROOT" \
  --rank-root "$RAW" \
  --expected-runner-sha256 "$EXPECTED_RUNNER_SHA256" \
  > "$LOGS/aggregate.log" 2>&1

(
  cd "$RUN_DIR"
  find preregistration raw logs -type f -print0 | sort -z | xargs -0 sha256sum
  sha256sum detector-matrix-summary.json
) > "$RUN_DIR/all-artifacts.sha256"
touch "$STAGES/03_aggregate_complete"

"$PYTHON" - "$RUN_DIR/detector-matrix-summary.json" <<'PY'
import json
import sys
from pathlib import Path

value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert value["schema_version"] == "forkaudit-detector-matrix-aggregate-v1"
assert value["summary"]["mutants"] == 9
assert value["summary"]["forkaudit_expected_gate_caught"] == 9
assert len(value["rows"]) == 9
print(json.dumps(value["summary"], sort_keys=True))
PY
touch "$STAGES/99_done"
