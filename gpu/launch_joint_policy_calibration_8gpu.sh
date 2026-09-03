#!/usr/bin/env bash
set -euo pipefail

CODE_DIR=${CODE_DIR:?set CODE_DIR}
MODEL_DIR=${MODEL_DIR:?set MODEL_DIR}
DATA_FILE=${DATA_FILE:?set DATA_FILE}
DATA_MANIFEST=${DATA_MANIFEST:?set DATA_MANIFEST}
EXPECTED_DATA_SHA256=${EXPECTED_DATA_SHA256:?set EXPECTED_DATA_SHA256}
EXPECTED_MANIFEST_SHA256=${EXPECTED_MANIFEST_SHA256:?set EXPECTED_MANIFEST_SHA256}
EXPECTED_WINDOWS_SHA256=${EXPECTED_WINDOWS_SHA256:?set EXPECTED_WINDOWS_SHA256}
RUN_DIR=${RUN_DIR:?set RUN_DIR}
ENV_DIR=${ENV_DIR:?set ENV_DIR}
CALIBRATION_BOOKS=${CALIBRATION_BOOKS:-32}
DOCUMENT_TOKENS=${DOCUMENT_TOKENS:-1024}
QUERY_TOKENS=${QUERY_TOKENS:-128}
QUERY_POSITIONS=${QUERY_POSITIONS:-8}
WINDOW_STRIDE=${WINDOW_STRIDE:-512}
CANDIDATE_WINDOWS_PER_BOOK=${CANDIDATE_WINDOWS_PER_BOOK:-4}
CANDIDATES_PER_BUDGET=${CANDIDATES_PER_BUDGET:-6}
SEED=${SEED:-20260812}

case "$(printf '%s' "$DATA_FILE $DATA_MANIFEST" | tr '[:upper:]_' '[:lower:]-')" in
  *longbench*|*qasper*|*2wikimqa*)
    echo "joint policy selection accepts only PG-19 train paths" >&2
    exit 1
    ;;
esac
if [ "$EXPECTED_DATA_SHA256" = "1553739496b3c209988de56a4ccf574d407379b6b7937ddfafacbe66925069fe" ] || \
   [ "$EXPECTED_DATA_SHA256" = "fe046477bf5b39629e9f66fd4def7a55c2d5d1f073c8bb601ee3833f08eaaa5f" ]; then
  echo "refusing known LongBench validation/test digest" >&2
  exit 1
fi
if [ -e "$RUN_DIR" ] && [ -n "$(find "$RUN_DIR" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]; then
  echo "run directory already contains artifacts; refusing duplicate task: $RUN_DIR" >&2
  exit 1
fi

mkdir -p "$RUN_DIR/logs" "$RUN_DIR/stages"
on_failure() {
  status=$?
  if [ "$status" -ne 0 ]; then
    date -u +%FT%TZ > "$RUN_DIR/stages/FAILED"
  fi
  exit "$status"
}
trap on_failure EXIT
date -u +%FT%TZ > "$RUN_DIR/stages/00_start"

GPU_COUNT=$(nvidia-smi -L | wc -l | tr -d ' ')
if [ "$GPU_COUNT" -ne 8 ]; then
  echo "expected exactly 8 GPUs, found $GPU_COUNT" >&2
  exit 1
fi
nvidia-smi \
  --query-gpu=index,uuid,name,memory.total,memory.used,driver_version,pstate,power.limit \
  --format=csv > "$RUN_DIR/gpus-before.csv"

FILES=(
  qcomem_joint_policy.py
  run_joint_policy_profile.py
  aggregate_joint_policy_candidates.py
  run_joint_policy_eval.py
  aggregate_joint_policy_eval.py
  qcomem_torch.py
  run_downstream.py
  test_joint_policy_calibration.py
)
for file in "${FILES[@]}"; do
  test -s "$CODE_DIR/$file"
done
(
  cd "$CODE_DIR"
  sha256sum "${FILES[@]}" > "$RUN_DIR/code.sha256"
)

"$ENV_DIR/bin/python" -m py_compile \
  "$CODE_DIR/qcomem_joint_policy.py" \
  "$CODE_DIR/run_joint_policy_profile.py" \
  "$CODE_DIR/aggregate_joint_policy_candidates.py" \
  "$CODE_DIR/run_joint_policy_eval.py" \
  "$CODE_DIR/aggregate_joint_policy_eval.py"
PYTHONPATH="$CODE_DIR" "$ENV_DIR/bin/python" -m unittest discover \
  -s "$CODE_DIR" -p 'test_joint_policy_calibration.py' -v \
  > "$RUN_DIR/joint-policy-tests.log" 2>&1

PYTHONPATH="$CODE_DIR" \
DATA_FILE="$DATA_FILE" DATA_MANIFEST="$DATA_MANIFEST" \
EXPECTED_DATA_SHA256="$EXPECTED_DATA_SHA256" \
EXPECTED_MANIFEST_SHA256="$EXPECTED_MANIFEST_SHA256" \
CALIBRATION_BOOKS="$CALIBRATION_BOOKS" \
"$ENV_DIR/bin/python" - <<'PY' > "$RUN_DIR/data-audit.json"
import json
import os
from pathlib import Path

from qcomem_joint_policy import audit_pg19_train_calibration

_, audit = audit_pg19_train_calibration(
    Path(os.environ["DATA_FILE"]),
    Path(os.environ["DATA_MANIFEST"]),
    expected_data_sha256=os.environ["EXPECTED_DATA_SHA256"],
    expected_manifest_sha256=os.environ["EXPECTED_MANIFEST_SHA256"],
    minimum_books=int(os.environ["CALIBRATION_BOOKS"]),
)
if audit["longbench_labels_used"] or audit["formal_validation_source_6_35_used"]:
    raise SystemExit("LongBench labels leaked into policy selection")
if audit["frozen_test_v2_source_68_99_used"]:
    raise SystemExit("frozen test-v2 leaked into policy selection")
print(json.dumps(audit, indent=2))
PY
date -u +%FT%TZ > "$RUN_DIR/stages/01_preflight_ok"

COMMON_ARGS=(
  --model "$MODEL_DIR"
  --data "$DATA_FILE"
  --manifest "$DATA_MANIFEST"
  --expected-data-sha256 "$EXPECTED_DATA_SHA256"
  --expected-manifest-sha256 "$EXPECTED_MANIFEST_SHA256"
  --expected-windows-sha256 "$EXPECTED_WINDOWS_SHA256"
  --run-dir "$RUN_DIR"
  --world-size 8
  --depth 7
  --calibration-books "$CALIBRATION_BOOKS"
  --document-tokens "$DOCUMENT_TOKENS"
  --query-tokens "$QUERY_TOKENS"
  --query-positions "$QUERY_POSITIONS"
  --window-stride "$WINDOW_STRIDE"
  --candidate-windows-per-book "$CANDIDATE_WINDOWS_PER_BOOK"
  --seed "$SEED"
)

PIDS=()
for RANK in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$RANK "$ENV_DIR/bin/python" \
    "$CODE_DIR/run_joint_policy_profile.py" \
    "${COMMON_ARGS[@]}" --rank "$RANK" \
    > "$RUN_DIR/logs/profile-${RANK}.log" 2>&1 &
  PIDS+=("$!")
  sleep 5
done
FAILED=0
for INDEX in 0 1 2 3 4 5 6 7; do
  if ! wait "${PIDS[$INDEX]}"; then
    echo "component profile rank $INDEX failed" >&2
    FAILED=1
  fi
done
if [ "$FAILED" -ne 0 ]; then
  exit 1
fi
date -u +%FT%TZ > "$RUN_DIR/stages/02_component_profiles_done"

PYTHONPATH="$CODE_DIR" "$ENV_DIR/bin/python" \
  "$CODE_DIR/aggregate_joint_policy_candidates.py" "$RUN_DIR" \
  --expected-shards 8 --candidates-per-budget "$CANDIDATES_PER_BUDGET" \
  > "$RUN_DIR/candidate-generation.log" 2>&1
test -s "$RUN_DIR/joint-policy-candidates.json"
date -u +%FT%TZ > "$RUN_DIR/stages/03_candidates_frozen"

EVAL_ARGS=("${COMMON_ARGS[@]}" --candidate-file "$RUN_DIR/joint-policy-candidates.json")
PIDS=()
for RANK in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$RANK "$ENV_DIR/bin/python" \
    "$CODE_DIR/run_joint_policy_eval.py" \
    "${EVAL_ARGS[@]}" --rank "$RANK" \
    > "$RUN_DIR/logs/joint-eval-${RANK}.log" 2>&1 &
  PIDS+=("$!")
  sleep 5
done
FAILED=0
for INDEX in 0 1 2 3 4 5 6 7; do
  if ! wait "${PIDS[$INDEX]}"; then
    echo "joint evaluation rank $INDEX failed" >&2
    FAILED=1
  fi
done
if [ "$FAILED" -ne 0 ]; then
  exit 1
fi
date -u +%FT%TZ > "$RUN_DIR/stages/04_joint_evaluation_done"

PYTHONPATH="$CODE_DIR" "$ENV_DIR/bin/python" \
  "$CODE_DIR/aggregate_joint_policy_eval.py" "$RUN_DIR" \
  --expected-shards 8 --bootstrap-seed "$SEED" \
  > "$RUN_DIR/joint-selection.log" 2>&1
test -s "$RUN_DIR/joint_policy.json"
nvidia-smi \
  --query-gpu=index,uuid,name,memory.used,pstate,power.draw,temperature.gpu \
  --format=csv > "$RUN_DIR/gpus-after.csv"
date -u +%FT%TZ > "$RUN_DIR/stages/99_done"
trap - EXIT
echo "Expanded PG-19 joint policy calibration complete: $RUN_DIR"
