#!/usr/bin/env bash
set -euo pipefail

# Generic cross-execution launcher for the outcome-blind R29 suite.  The
# independent executor and aggregator are supplied later and are SHA-bound
# here before any candidate output can be created.

: "${CODE_DIR:?}"
: "${SUITE_FILE:?}"
: "${EXECUTION_INPUT_FILE:?}"
: "${EXECUTOR_FILE:?}"
: "${AGGREGATOR_FILE:?}"
: "${RUN_DIR:?}"
: "${ENV_DIR:?}"
: "${EXPECTED_SUITE_RAW_SHA256:?}"
: "${EXPECTED_SUITE_CANONICAL_SHA256:?}"
: "${EXPECTED_FAULT_MODULE_SHA256:?}"
: "${EXPECTED_TEST_SHA256:?}"
: "${EXPECTED_CROSS_TEST_SHA256:?}"
: "${EXPECTED_EXECUTION_INPUT_SHA256:?}"
: "${EXPECTED_EXECUTOR_SHA256:?}"
: "${EXPECTED_AGGREGATOR_SHA256:?}"
: "${EXPECTED_LAUNCHER_SHA256:?}"

PYTHON="$ENV_DIR/bin/python"
FAULT_MODULE="$CODE_DIR/r29_heldout_fault_suite.py"
TEST_FILE="$CODE_DIR/test_r29_heldout_fault_suite.py"
CROSS_TEST_FILE="$CODE_DIR/test_r29_heldout_fault_executor.py"
LAUNCHER="$CODE_DIR/r29_launch_heldout_faults.sh"
PREREG_DIR="$RUN_DIR/preregistration"
RAW_DIR="$RUN_DIR/raw"
LOG_DIR="$RUN_DIR/logs"
RECEIPT_DIR="$RUN_DIR/receipts"
STAGE_DIR="$RUN_DIR/stages"
SUITE_COPY="$PREREG_DIR/heldout-fault-suite.json"
INPUT_COPY="$PREREG_DIR/execution-input.json"

test -x "$PYTHON"
test -f "$SUITE_FILE"
test -f "$EXECUTION_INPUT_FILE"
test -f "$FAULT_MODULE"
test -f "$TEST_FILE"
test -f "$CROSS_TEST_FILE"
test -f "$EXECUTOR_FILE"
test -f "$AGGREGATOR_FILE"
test -f "$LAUNCHER"
test ! -e "$RUN_DIR"

test "$(sha256sum "$SUITE_FILE" | awk '{print $1}')" = "$EXPECTED_SUITE_RAW_SHA256"
test "$(sha256sum "$FAULT_MODULE" | awk '{print $1}')" = "$EXPECTED_FAULT_MODULE_SHA256"
test "$(sha256sum "$TEST_FILE" | awk '{print $1}')" = "$EXPECTED_TEST_SHA256"
test "$(sha256sum "$CROSS_TEST_FILE" | awk '{print $1}')" = "$EXPECTED_CROSS_TEST_SHA256"
test "$(sha256sum "$EXECUTION_INPUT_FILE" | awk '{print $1}')" = "$EXPECTED_EXECUTION_INPUT_SHA256"
test "$(sha256sum "$EXECUTOR_FILE" | awk '{print $1}')" = "$EXPECTED_EXECUTOR_SHA256"
test "$(sha256sum "$AGGREGATOR_FILE" | awk '{print $1}')" = "$EXPECTED_AGGREGATOR_SHA256"
test "$(sha256sum "$LAUNCHER" | awk '{print $1}')" = "$EXPECTED_LAUNCHER_SHA256"

mkdir -p "$PREREG_DIR" "$RAW_DIR/sidecars" "$LOG_DIR" "$RECEIPT_DIR" "$STAGE_DIR"
cp -- "$SUITE_FILE" "$SUITE_COPY"
cp -- "$EXECUTION_INPUT_FILE" "$INPUT_COPY"
test "$(sha256sum "$SUITE_COPY" | awk '{print $1}')" = "$EXPECTED_SUITE_RAW_SHA256"
test "$(sha256sum "$INPUT_COPY" | awk '{print $1}')" = "$EXPECTED_EXECUTION_INPUT_SHA256"

export PYTHONPATH="$CODE_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1

"$PYTHON" -B "$FAULT_MODULE" --verify-frozen "$SUITE_COPY" \
  > "$LOG_DIR/frozen-suite-validation.json"
"$PYTHON" -B - "$LOG_DIR/frozen-suite-validation.json" \
  "$EXPECTED_SUITE_CANONICAL_SHA256" <<'PY'
import json
import sys
from pathlib import Path

receipt = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert receipt["validated"] is True
assert receipt["contains_expected_detector_mapping"] is False
assert receipt["suite_sha256"] == sys.argv[2]
assert receipt["fault_ids"] == ["H01", "H02", "H03"]
PY
touch "$STAGE_DIR/00_fault_suite_bound_before_outputs"

"$PYTHON" -B - "$ENV_DIR" > "$RECEIPT_DIR/python-environment-identity-v2.json" <<'PY'
import json
import os
import sys
from pathlib import Path

env_dir = Path(os.path.abspath(sys.argv[1]))
invoked = Path(os.path.abspath(sys.executable))
prefix = Path(os.path.abspath(sys.prefix))
base_prefix = Path(os.path.abspath(sys.base_prefix))
expected = env_dir / "bin" / "python"
assert prefix == env_dir
assert invoked == expected
assert invoked.resolve(strict=True) == expected.resolve(strict=True)
print(json.dumps({
    "schema_version": "forkaudit-r29-python-environment-identity-v2",
    "frozen_env_dir": str(env_dir),
    "sys_prefix": str(prefix),
    "sys_base_prefix": str(base_prefix),
    "sys_executable": str(invoked),
    "frozen_env_python": str(expected),
    "sys_executable_realpath": str(invoked.resolve(strict=True)),
    "frozen_env_python_realpath": str(expected.resolve(strict=True)),
    "sys_prefix_exact": True,
    "lexical_invocation_exact": True,
    "resolved_interpreter_target_exact": True,
    "resolved_interpreter_required_below_env": False,
}, sort_keys=True))
PY
touch "$STAGE_DIR/00b_python_environment_identity_v2_verified"

(
  cd "$CODE_DIR"
  "$PYTHON" -B -m unittest -v test_r29_heldout_fault_suite.py
) > "$LOG_DIR/fault-suite-unit-tests.log" 2>&1
touch "$STAGE_DIR/01_fault_suite_unit_tests_passed"

(
  cd "$CODE_DIR"
  "$PYTHON" -B -m unittest -v test_r29_heldout_fault_executor.py
) > "$LOG_DIR/cross-executor-unit-tests.log" 2>&1
touch "$STAGE_DIR/01b_cross_executor_unit_tests_passed"

nvidia-smi --query-gpu=uuid,name,memory.total --format=csv,noheader,nounits \
  > "$RECEIPT_DIR/gpu-inventory.csv"
mapfile -t GPU_UUIDS < <(
  nvidia-smi --query-gpu=uuid --format=csv,noheader,nounits \
    | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' \
    | sed '/^$/d'
)
test "${#GPU_UUIDS[@]}" -ge 3
"$PYTHON" -B - "$RECEIPT_DIR/gpu-inventory.csv" <<'PY'
import csv
import sys
from pathlib import Path

rows = list(csv.reader(Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()))
assert len(rows) >= 3
for uuid, name, memory in ([cell.strip() for cell in row] for row in rows[:3]):
    assert uuid.startswith("GPU-")
    assert name == "NVIDIA H20-3e"
    assert int(memory) > 0
assert len({row[0].strip() for row in rows[:3]}) == 3
PY
touch "$STAGE_DIR/02_three_distinct_h20s_verified"

FAULT_IDS=(H01 H02 H03)
pids=()
for rank in 0 1 2; do
  fault_id="${FAULT_IDS[$rank]}"
  mkdir -p "$RAW_DIR/sidecars/rank-$rank"
  CUDA_VISIBLE_DEVICES="${GPU_UUIDS[$rank]}" "$PYTHON" -B "$EXECUTOR_FILE" \
    --suite "$SUITE_COPY" \
    --expected-suite-raw-sha256 "$EXPECTED_SUITE_RAW_SHA256" \
    --expected-suite-canonical-sha256 "$EXPECTED_SUITE_CANONICAL_SHA256" \
    --execution-input "$INPUT_COPY" \
    --expected-execution-input-sha256 "$EXPECTED_EXECUTION_INPUT_SHA256" \
    --fault-id "$fault_id" \
    --rank "$rank" \
    --expected-gpu-uuid "${GPU_UUIDS[$rank]}" \
    --output "$RAW_DIR/heldout-fault-rank-$rank.json" \
    --sidecar-root "$RAW_DIR/sidecars/rank-$rank" \
    > "$LOG_DIR/rank-$rank.log" 2>&1 &
  pids+=("$!")
done

failed=0
for index in 0 1 2; do
  if ! wait "${pids[$index]}"; then
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  touch "$STAGE_DIR/03_rank_operational_failure"
  exit 2
fi
touch "$STAGE_DIR/03_all_three_ranks_complete"

(
  cd "$RUN_DIR"
  find raw -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum \
    > receipts/raw-artifacts.sha256
  sha256sum -c receipts/raw-artifacts.sha256 \
    > logs/raw-artifact-integrity-before-aggregate.log
)

"$PYTHON" -B "$AGGREGATOR_FILE" \
  --suite "$SUITE_COPY" \
  --expected-suite-raw-sha256 "$EXPECTED_SUITE_RAW_SHA256" \
  --expected-suite-canonical-sha256 "$EXPECTED_SUITE_CANONICAL_SHA256" \
  --execution-input "$INPUT_COPY" \
  --expected-execution-input-sha256 "$EXPECTED_EXECUTION_INPUT_SHA256" \
  --rank-root "$RAW_DIR" \
  --output "$RUN_DIR/heldout-fault-summary.json" \
  > "$LOG_DIR/aggregate.log" 2>&1
touch "$STAGE_DIR/04_strict_aggregate_complete"

"$PYTHON" -B - "$RUN_DIR/heldout-fault-summary.json" <<'PY'
import json
import sys
from pathlib import Path

value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert value["schema_version"] == "forkaudit-r29-heldout-fault-summary-v1"
assert value["scientific_valid"] is True
assert value["operational_invalid_count"] == 0
assert value["fault_ids"] == ["H01", "H02", "H03"]
assert value["detection_rate_reported"] is False
assert value["naturally_occurring_claimed"] is False
PY

(
  cd "$RUN_DIR"
  sha256sum -c receipts/raw-artifacts.sha256 \
    > logs/raw-artifact-integrity-terminal.log
  sha256sum \
    preregistration/heldout-fault-suite.json \
    preregistration/execution-input.json \
    receipts/gpu-inventory.csv \
    receipts/raw-artifacts.sha256 \
    heldout-fault-summary.json \
    > receipts/terminal-products.sha256
  sha256sum -c receipts/terminal-products.sha256 \
    > logs/terminal-product-integrity.log
)
touch "$STAGE_DIR/COMPLETED_VALID_SCIENTIFIC_OUTCOME"
touch "$RUN_DIR/COMPLETED_VALID_SCIENTIFIC_OUTCOME"
