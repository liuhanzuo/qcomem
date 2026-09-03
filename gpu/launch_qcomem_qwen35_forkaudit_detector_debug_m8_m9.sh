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
: "${EXPECTED_LAUNCHER_SHA256:?}"
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
LAUNCHER="$CODE_DIR/launch_qcomem_qwen35_forkaudit_detector_debug_m8_m9.sh"
ORIGINAL_RECEIPTS="$ORIGINAL_RR2_RUN_ROOT/receipts/detached-receipt-manifest.json"
PREREG="$RUN_DIR/preregistration/debug-m8-m9-plan.json"
RAW="$RUN_DIR/raw"
LOGS="$RUN_DIR/logs"
STAGES="$RUN_DIR/stages"

test -x "$PYTHON"
test -f "$RUNNER"
test -f "$BUILDER"
test -f "$LAUNCHER"
test -f "$ORIGINAL_RECEIPTS"
test ! -e "$RUN_DIR"
mkdir -p "$RUN_DIR/preregistration" "$RAW" "$LOGS" "$STAGES"
touch "$STAGES/00_start_debug_only"

test "$(sha256sum "$RUNNER" | awk '{print $1}')" = "$EXPECTED_RUNNER_SHA256"
test "$(sha256sum "$BUILDER" | awk '{print $1}')" = "$EXPECTED_BUILDER_SHA256"
test "$(sha256sum "$LAUNCHER" | awk '{print $1}')" = "$EXPECTED_LAUNCHER_SHA256"
test "$(sha256sum "$IMPORTED_RR2_CODE_LEDGER" | awk '{print $1}')" = "$EXPECTED_IMPORTED_RR2_CODE_LEDGER_SHA256"
test "$(sha256sum "$MODEL_WEIGHT_LEDGER_FILE" | awk '{print $1}')" = "$EXPECTED_MODEL_WEIGHT_LEDGER_SHA256"
test "$(sha256sum "$MODEL_ARTIFACT_LEDGER_FILE" | awk '{print $1}')" = "$EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256"
test "$(sha256sum "$PG19_DATA" | awk '{print $1}')" = "$EXPECTED_PG19_SHA256"
test "$(sha256sum "$PG19_MANIFEST" | awk '{print $1}')" = "$EXPECTED_PG19_MANIFEST_SHA256"
test "$(sha256sum "$FROZEN_QUERY_BANKS" | awk '{print $1}')" = "$EXPECTED_FROZEN_QUERY_BANKS_SHA256"

export PYTHONPATH="$CODE_DIR:$IMPORTED_RR2_CODE_DIR"
export PYTHONDONTWRITEBYTECODE=1

"$PYTHON" -B "$BUILDER" \
  --stage preregister-debug \
  --output "$PREREG" \
  --runner "$RUNNER" \
  --original-rr2-run-id 372384bd37cf7640ca210537a4360e1a \
  --original-receipt-manifest "$ORIGINAL_RECEIPTS" \
  --imported-rr2-code-ledger-sha256 "$EXPECTED_IMPORTED_RR2_CODE_LEDGER_SHA256"
PREREG_SHA=$(sha256sum "$PREREG" | awk '{print $1}')
touch "$STAGES/01_debug_preregistered_before_outputs"

nvidia-smi --query-gpu=uuid,name,memory.total --format=csv,noheader,nounits > "$RUN_DIR/preregistration/gpu-inventory.csv"
mapfile -t GPU_UUIDS < <(nvidia-smi --query-gpu=uuid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')
test "${#GPU_UUIDS[@]}" -eq 8

pids=()
for spec in "M9:0:0" "M8:7:1"; do
  IFS=: read -r mutant rank gpu_index <<< "$spec"
  CUDA_VISIBLE_DEVICES="${GPU_UUIDS[$gpu_index]}" "$PYTHON" -B "$RUNNER" \
    --stage debug-path \
    --debug-mutant "$mutant" \
    --rank "$rank" \
    --output "$RAW/debug-$mutant-rank-$rank.json" \
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
    --expected-gpu-uuid "${GPU_UUIDS[$gpu_index]}" \
    > "$LOGS/debug-$mutant-rank-$rank.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  touch "$STAGES/02_debug_path_failure"
  exit 2
fi
touch "$STAGES/02_debug_paths_complete"

"$PYTHON" - "$RAW/debug-M8-rank-7.json" "$RAW/debug-M9-rank-0.json" <<'PY'
import json
import sys
from pathlib import Path

rows = [json.loads(Path(path).read_text(encoding="utf-8")) for path in sys.argv[1:]]
assert {row["debug_mutant"] for row in rows} == {"M8", "M9"}
assert all(row["schema_version"] == "forkaudit-detector-m8-m9-debug-v1" for row in rows)
assert all(row["debug_only"] is True for row in rows)
assert all(row["formal_evidence_eligible"] is False for row in rows)
assert all(row["all_debug_gates_passed"] is True for row in rows)
assert all(row["matched_clean"]["status"] == "completed" for row in rows)
assert all(len(row["matched_clean"]["logit_sidecars"]) == 1 for row in rows)
PY
touch "$STAGES/03_debug_validation_complete"

(
  cd "$RUN_DIR"
  find preregistration raw logs -type f -print0 | sort -z | xargs -0 sha256sum
) > "$RUN_DIR/all-debug-artifacts.sha256"
touch "$STAGES/99_debug_only_done"
