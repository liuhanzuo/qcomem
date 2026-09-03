#!/usr/bin/env bash
set -euo pipefail

: "${CODE_DIR:?}"
: "${IMPORTED_RR2_CODE_DIR:?}"
: "${IMPORTED_RR2_CODE_LEDGER_FILE:?}"
: "${CODE_LEDGER_FILE:?}"
: "${EXTERNAL_PIN_FILE:?}"
: "${QS_CONFIG_FILE:?}"
: "${SCOPE_SUPERSESSION_FILE:?}"
: "${MODEL_DIR:?}"
: "${MODEL_REVISION:?}"
: "${MODEL_WEIGHT_LEDGER_FILE:?}"
: "${MODEL_ARTIFACT_LEDGER_FILE:?}"
: "${PG19_DATA:?}"
: "${PG19_MANIFEST:?}"
: "${FROZEN_QUERY_BANKS:?}"
: "${ORIGINAL_RR2_RUN_ROOT:?}"
: "${ORIGINAL_RR2_RUN_ID:?}"
: "${RUN_DIR:?}"
: "${ENV_DIR:?}"
: "${EXPECTED_RUNNER_SHA256:?}"
: "${EXPECTED_BUILDER_SHA256:?}"
: "${EXPECTED_REPLAY_SHA256:?}"
: "${EXPECTED_TEST_SHA256:?}"
: "${EXPECTED_LAUNCHER_SHA256:?}"
: "${EXPECTED_GATE_POLICY_SHA256:?}"
: "${EXPECTED_SCOPE_SUPERSESSION_SHA256:?}"
: "${EXPECTED_CODE_LEDGER_SHA256:?}"
: "${EXPECTED_IMPORTED_RR2_CODE_LEDGER_SHA256:?}"
: "${EXPECTED_EXTERNAL_PIN_SHA256:?}"
: "${EXPECTED_MODEL_WEIGHT_LEDGER_SHA256:?}"
: "${EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256:?}"
: "${EXPECTED_PG19_SHA256:?}"
: "${EXPECTED_PG19_MANIFEST_SHA256:?}"
: "${EXPECTED_PG19_WINDOWS_SHA256:?}"
: "${EXPECTED_FROZEN_QUERY_BANKS_SHA256:?}"

PYTHON="$ENV_DIR/bin/python"
RUNNER="$CODE_DIR/run_qcomem_qwen35_forkaudit_detector_matrix_v2.py"
BUILDER="$CODE_DIR/build_qcomem_qwen35_forkaudit_detector_matrix_v2.py"
REPLAY="$CODE_DIR/replay_qcomem_qwen35_forkaudit_detector_matrix_v2.py"
TEST_FILE="$CODE_DIR/test_qcomem_qwen35_forkaudit_detector_matrix_v2.py"
LAUNCHER="$CODE_DIR/launch_qcomem_qwen35_forkaudit_detector_matrix_v2_8gpu.sh"
GATE_POLICY="$CODE_DIR/qcomem_forkaudit_selective_gate_policy.py"
ORIGINAL_RECEIPTS="$ORIGINAL_RR2_RUN_ROOT/receipts/detached-receipt-manifest.json"
PREREG="$RUN_DIR/preregistration/detector-matrix-v2-plan.json"
PIN_COPY="$RUN_DIR/preregistration/preexecution-external-pin-payload.json"
QS_CONFIG_COPY="$RUN_DIR/preregistration/qs-config.yaml"
SCOPE_COPY="$RUN_DIR/preregistration/scope-supersession.json"
RR2_CODE_LEDGER_COPY="$RUN_DIR/preregistration/imported-rr2-code.sha256"
RAW="$RUN_DIR/raw"
LOGS="$RUN_DIR/logs"
RECEIPTS="$RUN_DIR/receipts"
REPLAY_ROOT="$RUN_DIR/replay"
STAGES="$RUN_DIR/stages"

test -x "$PYTHON"
test -f "$RUNNER"
test -f "$BUILDER"
test -f "$REPLAY"
test -f "$TEST_FILE"
test -f "$LAUNCHER"
test -f "$GATE_POLICY"
test -f "$CODE_LEDGER_FILE"
test -f "$IMPORTED_RR2_CODE_LEDGER_FILE"
test -f "$EXTERNAL_PIN_FILE"
test -f "$QS_CONFIG_FILE"
test -f "$SCOPE_SUPERSESSION_FILE"
test -f "$IMPORTED_RR2_CODE_DIR/run_qcomem_qwen35_forkaudit_review_revision.py"
test -f "$ORIGINAL_RECEIPTS"
test ! -e "$RUN_DIR"

test "$(sha256sum "$RUNNER" | awk '{print $1}')" = "$EXPECTED_RUNNER_SHA256"
test "$(sha256sum "$BUILDER" | awk '{print $1}')" = "$EXPECTED_BUILDER_SHA256"
test "$(sha256sum "$REPLAY" | awk '{print $1}')" = "$EXPECTED_REPLAY_SHA256"
test "$(sha256sum "$TEST_FILE" | awk '{print $1}')" = "$EXPECTED_TEST_SHA256"
test "$(sha256sum "$LAUNCHER" | awk '{print $1}')" = "$EXPECTED_LAUNCHER_SHA256"
test "$(sha256sum "$GATE_POLICY" | awk '{print $1}')" = "$EXPECTED_GATE_POLICY_SHA256"
test "$(sha256sum "$SCOPE_SUPERSESSION_FILE" | awk '{print $1}')" = "$EXPECTED_SCOPE_SUPERSESSION_SHA256"
test "$(sha256sum "$CODE_LEDGER_FILE" | awk '{print $1}')" = "$EXPECTED_CODE_LEDGER_SHA256"
test "$(sha256sum "$IMPORTED_RR2_CODE_LEDGER_FILE" | awk '{print $1}')" = "$EXPECTED_IMPORTED_RR2_CODE_LEDGER_SHA256"
test "$(sha256sum "$EXTERNAL_PIN_FILE" | awk '{print $1}')" = "$EXPECTED_EXTERNAL_PIN_SHA256"
test "$(sha256sum "$MODEL_WEIGHT_LEDGER_FILE" | awk '{print $1}')" = "$EXPECTED_MODEL_WEIGHT_LEDGER_SHA256"
test "$(sha256sum "$MODEL_ARTIFACT_LEDGER_FILE" | awk '{print $1}')" = "$EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256"
test "$(sha256sum "$PG19_DATA" | awk '{print $1}')" = "$EXPECTED_PG19_SHA256"
test "$(sha256sum "$PG19_MANIFEST" | awk '{print $1}')" = "$EXPECTED_PG19_MANIFEST_SHA256"
test "$(sha256sum "$FROZEN_QUERY_BANKS" | awk '{print $1}')" = "$EXPECTED_FROZEN_QUERY_BANKS_SHA256"

mkdir -p "$RUN_DIR/preregistration" "$RAW/sidecars" "$LOGS" "$RECEIPTS" "$REPLAY_ROOT" "$STAGES"
(
  cd "$CODE_DIR"
  sha256sum -c "$CODE_LEDGER_FILE"
) > "$LOGS/r28-code-ledger-integrity.log"
(
  cd "$IMPORTED_RR2_CODE_DIR"
  sha256sum -c "$IMPORTED_RR2_CODE_LEDGER_FILE"
) > "$LOGS/imported-rr2-code-ledger-integrity.log"
cp -- "$EXTERNAL_PIN_FILE" "$PIN_COPY"
cp -- "$QS_CONFIG_FILE" "$QS_CONFIG_COPY"
cp -- "$SCOPE_SUPERSESSION_FILE" "$SCOPE_COPY"
cp -- "$IMPORTED_RR2_CODE_LEDGER_FILE" "$RR2_CODE_LEDGER_COPY"
test "$(sha256sum "$PIN_COPY" | awk '{print $1}')" = "$EXPECTED_EXTERNAL_PIN_SHA256"
test "$(sha256sum "$SCOPE_COPY" | awk '{print $1}')" = "$EXPECTED_SCOPE_SUPERSESSION_SHA256"
test "$(sha256sum "$RR2_CODE_LEDGER_COPY" | awk '{print $1}')" = "$EXPECTED_IMPORTED_RR2_CODE_LEDGER_SHA256"
touch "$STAGES/00_started"

export PYTHONPATH="$CODE_DIR:$IMPORTED_RR2_CODE_DIR"
export PYTHONDONTWRITEBYTECODE=1

"$PYTHON" -B "$BUILDER" \
  --stage preregister \
  --output "$PREREG" \
  --runner "$RUNNER" \
  --replay "$REPLAY" \
  --test-file "$TEST_FILE" \
  --launcher "$LAUNCHER" \
  --gate-policy "$GATE_POLICY" \
  --qs-config "$QS_CONFIG_COPY" \
  --scope-supersession "$SCOPE_COPY" \
  --original-receipt-manifest "$ORIGINAL_RECEIPTS" \
  --original-rr2-run-id "$ORIGINAL_RR2_RUN_ID" \
  --model-revision "$MODEL_REVISION" \
  --weight-ledger-sha256 "$EXPECTED_MODEL_WEIGHT_LEDGER_SHA256" \
  --artifact-ledger-sha256 "$EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256" \
  --pg19-sha256 "$EXPECTED_PG19_SHA256" \
  --pg19-manifest-sha256 "$EXPECTED_PG19_MANIFEST_SHA256" \
  --windows-sha256 "$EXPECTED_PG19_WINDOWS_SHA256" \
  --frozen-query-banks-sha256 "$EXPECTED_FROZEN_QUERY_BANKS_SHA256" \
  --code-ledger-sha256 "$EXPECTED_CODE_LEDGER_SHA256" \
  --imported-rr2-code-ledger-sha256 "$EXPECTED_IMPORTED_RR2_CODE_LEDGER_SHA256" \
  --external-pin-payload "$PIN_COPY" \
  --external-pin-payload-sha256 "$EXPECTED_EXTERNAL_PIN_SHA256"
PREREG_SHA="$(sha256sum "$PREREG" | awk '{print $1}')"
touch "$STAGES/01_preregistered_before_candidate_outputs"

(
  cd "$CODE_DIR"
  "$PYTHON" -B -m unittest -v test_qcomem_qwen35_forkaudit_detector_matrix_v2.py
) > "$LOGS/preflight-unit-tests.log" 2>&1
touch "$STAGES/01_preflight_unit_tests_passed"

nvidia-smi --query-gpu=uuid,name,memory.total --format=csv,noheader,nounits \
  > "$RECEIPTS/gpu-inventory.csv"
mapfile -t GPU_UUIDS < <(
  nvidia-smi --query-gpu=uuid --format=csv,noheader,nounits \
    | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' \
    | sed '/^$/d'
)
test "${#GPU_UUIDS[@]}" -eq 8
"$PYTHON" -B - "$RECEIPTS/gpu-inventory.csv" <<'PY'
import csv
import sys
from pathlib import Path

rows = list(csv.reader(Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()))
assert len(rows) == 8
uuids = []
for row in rows:
    assert len(row) == 3
    uuid, name, memory = (cell.strip() for cell in row)
    assert uuid.startswith("GPU-")
    assert name == "NVIDIA H20-3e"
    assert int(memory) > 0
    uuids.append(uuid)
assert len(set(uuids)) == 8
PY
touch "$STAGES/02_eight_distinct_h20s_verified"

pids=()
for rank in $(seq 0 7); do
  mkdir -p "$RAW/sidecars/rank-$rank"
  CUDA_VISIBLE_DEVICES="${GPU_UUIDS[$rank]}" "$PYTHON" -B "$RUNNER" \
    --stage rank \
    --rank "$rank" \
    --output "$RAW/detector-matrix-v2-rank-$rank.json" \
    --sidecar-root "$RAW/sidecars/rank-$rank" \
    --rank-root "$RAW" \
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
    --code-ledger "$CODE_LEDGER_FILE" \
    --expected-code-ledger-sha256 "$EXPECTED_CODE_LEDGER_SHA256" \
    --imported-rr2-code-ledger "$IMPORTED_RR2_CODE_LEDGER_FILE" \
    --expected-imported-rr2-code-ledger-sha256 "$EXPECTED_IMPORTED_RR2_CODE_LEDGER_SHA256" \
    --external-pin-payload "$PIN_COPY" \
    --expected-external-pin-payload-sha256 "$EXPECTED_EXTERNAL_PIN_SHA256" \
    --expected-gate-policy-sha256 "$EXPECTED_GATE_POLICY_SHA256" \
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
  touch "$STAGES/03_rank_operational_failure"
  exit 2
fi
touch "$STAGES/03_all_ranks_complete"

(
  cd "$RUN_DIR"
  find raw -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum \
    > receipts/raw-artifacts.sha256
  sha256sum -c receipts/raw-artifacts.sha256 \
    > logs/raw-artifact-integrity-before-aggregate.log
)

"$PYTHON" -B "$BUILDER" \
  --stage aggregate \
  --output "$RUN_DIR/detector-matrix-v2-summary.json" \
  --preregistration "$PREREG" \
  --expected-preregistration-sha256 "$PREREG_SHA" \
  --original-receipt-manifest "$ORIGINAL_RECEIPTS" \
  --original-rr2-root "$ORIGINAL_RR2_RUN_ROOT" \
  --rank-root "$RAW" \
  --expected-runner-sha256 "$EXPECTED_RUNNER_SHA256" \
  --runner "$RUNNER" \
  --replay "$REPLAY" \
  --test-file "$TEST_FILE" \
  --launcher "$LAUNCHER" \
  --gate-policy "$GATE_POLICY" \
  --qs-config "$QS_CONFIG_COPY" \
  --scope-supersession "$SCOPE_COPY" \
  --external-pin-payload "$PIN_COPY" \
  > "$LOGS/aggregate.log" 2>&1
touch "$STAGES/04_strict_aggregate_complete"

"$PYTHON" -B "$REPLAY" \
  --summary "$RUN_DIR/detector-matrix-v2-summary.json" \
  --output "$REPLAY_ROOT/detector-matrix-v2-summary.json" \
  --preregistration "$PREREG" \
  --expected-preregistration-sha256 "$PREREG_SHA" \
  --rank-root "$RAW" \
  --original-receipt-manifest "$ORIGINAL_RECEIPTS" \
  --original-rr2-root "$ORIGINAL_RR2_RUN_ROOT" \
  --expected-runner-sha256 "$EXPECTED_RUNNER_SHA256" \
  --runner "$RUNNER" \
  --test-file "$TEST_FILE" \
  --launcher "$LAUNCHER" \
  --gate-policy "$GATE_POLICY" \
  --qs-config "$QS_CONFIG_COPY" \
  --scope-supersession "$SCOPE_COPY" \
  --external-pin-payload "$PIN_COPY" \
  > "$LOGS/replay.log" 2>&1
cmp "$RUN_DIR/detector-matrix-v2-summary.json" \
  "$REPLAY_ROOT/detector-matrix-v2-summary.json"
touch "$STAGES/05_replay_byte_identical"

"$PYTHON" -B - "$RUN_DIR/detector-matrix-v2-summary.json" <<'PY'
import json
import sys
from pathlib import Path

value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert value["schema_version"] == "forkaudit-detector-matrix-summary-v2"
assert value["scientific_valid"] is True
assert value["scientific_outcome"] in ("positive", "negative", "mixed")
assert value["operational_invalid_count"] == 0
assert value["counts"]["ranks"] == 8
assert value["counts"]["cases"] == 18
assert value["counts"]["clean_cases"] == 9
assert value["counts"]["target_suppressed_mutant_cases"] == 9
assert value["counts"]["clean_fp32_sidecars"] >= 9
PY

(
  cd "$RUN_DIR"
  sha256sum -c receipts/raw-artifacts.sha256 \
    > logs/raw-artifact-integrity-terminal.log
  sha256sum \
    preregistration/detector-matrix-v2-plan.json \
    preregistration/preexecution-external-pin-payload.json \
    preregistration/qs-config.yaml \
    preregistration/scope-supersession.json \
    preregistration/imported-rr2-code.sha256 \
    receipts/gpu-inventory.csv \
    receipts/raw-artifacts.sha256 \
    detector-matrix-v2-summary.json \
    replay/detector-matrix-v2-summary.json \
    > receipts/terminal-products.sha256
  sha256sum -c receipts/terminal-products.sha256 \
    > logs/terminal-product-integrity.log
)
touch "$STAGES/COMPLETED_VALID_SCIENTIFIC_OUTCOME"
touch "$RUN_DIR/COMPLETED_VALID_SCIENTIFIC_OUTCOME"
