#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 7 ]]; then
  echo "usage: $0 PACKAGE_ROOT OUTPUT_ROOT ENV_DIR RR2_CODE_DIR EXECUTION_INPUT TRIAL_ID EXPECTED_PACKAGE_SHA256" >&2
  exit 64
fi

PACKAGE_ROOT="$1"
OUTPUT_ROOT="$2"
ENV_DIR="$3"
RR2_CODE_DIR="$4"
EXECUTION_INPUT="$5"
TRIAL_ID="$6"
EXPECTED_PACKAGE_SHA256="$7"

EXECUTOR="$PACKAGE_ROOT/paper_autonomous_multifork_iteration/evidence/r39_blind_faults/executor"
FREEZE="$PACKAGE_ROOT/paper_autonomous_multifork_iteration/evidence/r39_blind_faults/designer_freeze"
PROTOCOL="$FREEZE/PROTOCOL.md"
PLAN="$FREEZE/plan.json"
SOURCE_MANIFEST="$EXECUTOR/source-code.sha256"
PYTHON="$ENV_DIR/bin/python"
LANE="$EXECUTOR/r39_lane.py"
REPLAY="$EXECUTOR/r39_replay.py"
RUN_FAULT="$EXECUTOR/r39_run_fault.py"
RR2_LEDGER="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["code"]["imported_rr2_code_ledger_path"])' "$EXECUTION_INPUT")"

[[ ! -e "$OUTPUT_ROOT" ]] || { echo "refusing existing output root: $OUTPUT_ROOT" >&2; exit 65; }
for item in "$PYTHON" "$PROTOCOL" "$PLAN" "$SOURCE_MANIFEST" "$EXECUTION_INPUT" "$RR2_LEDGER"; do
  [[ -e "$item" ]] || { echo "required path absent: $item" >&2; exit 66; }
done

mkdir -p "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/preflight" "$OUTPUT_ROOT/receipts" "$OUTPUT_ROOT/stages"
touch "$OUTPUT_ROOT/stages/00_started"
printf '%s\n' "$EXPECTED_PACKAGE_SHA256" > "$OUTPUT_ROOT/receipts/expected-package-sha256.txt"

sha256sum "$PROTOCOL" "$PLAN" "$EXECUTION_INPUT" > "$OUTPUT_ROOT/receipts/frozen-inputs.sha256"
(
  cd "$PACKAGE_ROOT"
  sha256sum -c "$SOURCE_MANIFEST"
) > "$OUTPUT_ROOT/logs/source-manifest-check.log" 2>&1
(
  cd "$RR2_CODE_DIR"
  sha256sum -c "$RR2_LEDGER"
) > "$OUTPUT_ROOT/logs/imported-rr2-code-check.log" 2>&1
touch "$OUTPUT_ROOT/stages/01_source_and_freeze_verified"

mapfile -t GPU_UUIDS < <(nvidia-smi --query-gpu=uuid --format=csv,noheader | tr -d ' ')
[[ "${#GPU_UUIDS[@]}" -eq 8 ]] || { echo "R39 blind faults require exactly eight visible physical GPUs" >&2; exit 67; }
printf 'physical_gpu_index,gpu_uuid\n' > "$OUTPUT_ROOT/receipts/gpu-assignment.csv"
for index in 0 1 2 3 4 5 6 7; do
  printf '%s,%s\n' "$index" "${GPU_UUIDS[$index]}" >> "$OUTPUT_ROOT/receipts/gpu-assignment.csv"
done
touch "$OUTPUT_ROOT/stages/02_gpu_assignment_frozen"

run_one_fault() {
  local gpu_index="$1"
  local fault_id="$2"
  local uuid="${GPU_UUIDS[$gpu_index]}"
  local preflight_root="$OUTPUT_ROOT/preflight/$fault_id"
  local feasibility="$preflight_root/feasibility.json"
  local fault_root="$OUTPUT_ROOT/$fault_id"
  local triton_cache="$OUTPUT_ROOT/triton-cache-gpu-$gpu_index"
  mkdir -p "$preflight_root" "$triton_cache"
  if ! CUDA_VISIBLE_DEVICES="$uuid" \
    TOKENIZERS_PARALLELISM=false \
    TRITON_CACHE_DIR="$triton_cache" \
    PYTHONPATH="$EXECUTOR:$RR2_CODE_DIR" \
    "$PYTHON" "$EXECUTOR/r39_preflight.py" \
      --fault-id "$fault_id" \
      --gpu-index "$gpu_index" \
      --expected-gpu-uuid "$uuid" \
      --protocol "$PROTOCOL" \
      --plan "$PLAN" \
      --execution-input "$EXECUTION_INPUT" \
      --source-root "$PACKAGE_ROOT" \
      --source-manifest "$SOURCE_MANIFEST" \
      --triton-cache-root "$triton_cache" \
      --output "$feasibility" \
      > "$OUTPUT_ROOT/logs/$fault_id-preflight.log" 2>&1; then
    mkdir -p "$fault_root"
    if [[ -f "$preflight_root/preflight-operational-invalid.json" ]]; then
      cp "$preflight_root/preflight-operational-invalid.json" "$fault_root/operational-invalid.json"
    else
      printf '{"schema_version":"forkaudit-r39-launch-operational-invalid-v1","run_id":"R39-BLIND-FAULTS-20260826A","fault_id":"%s","status":"operational_invalid","stage":"preflight","negative_outcome_preserved":true}\n' "$fault_id" > "$fault_root/operational-invalid.json"
    fi
    return 1
  fi
  CUDA_VISIBLE_DEVICES="$uuid" \
    TOKENIZERS_PARALLELISM=false \
    TRITON_CACHE_DIR="$triton_cache" \
    PYTHONPATH="$EXECUTOR:$RR2_CODE_DIR" \
    "$PYTHON" "$RUN_FAULT" \
      --fault-id "$fault_id" \
      --gpu-index "$gpu_index" \
      --expected-gpu-uuid "$uuid" \
      --trial-id "$TRIAL_ID" \
      --run-dir "$fault_root" \
      --protocol "$PROTOCOL" \
      --plan "$PLAN" \
      --execution-input "$EXECUTION_INPUT" \
      --feasibility "$feasibility" \
      --source-root "$PACKAGE_ROOT" \
      --source-manifest "$SOURCE_MANIFEST" \
      --lane-source "$LANE" \
      --replay-source "$REPLAY" \
      > "$OUTPUT_ROOT/logs/$fault_id-driver.log" 2>&1
}

worker() {
  local gpu_index="$1"
  shift
  local status=0
  for fault_id in "$@"; do
    if ! run_one_fault "$gpu_index" "$fault_id"; then
      status=1
    fi
  done
  return "$status"
}

PIDS=()
worker 0 R39-BF01 R39-BF09 > "$OUTPUT_ROOT/logs/gpu-0-worker.log" 2>&1 & PIDS+=("$!")
worker 1 R39-BF02 R39-BF10 > "$OUTPUT_ROOT/logs/gpu-1-worker.log" 2>&1 & PIDS+=("$!")
worker 2 R39-BF03 R39-BF11 > "$OUTPUT_ROOT/logs/gpu-2-worker.log" 2>&1 & PIDS+=("$!")
worker 3 R39-BF04 > "$OUTPUT_ROOT/logs/gpu-3-worker.log" 2>&1 & PIDS+=("$!")
worker 4 R39-BF05 > "$OUTPUT_ROOT/logs/gpu-4-worker.log" 2>&1 & PIDS+=("$!")
worker 5 R39-BF06 > "$OUTPUT_ROOT/logs/gpu-5-worker.log" 2>&1 & PIDS+=("$!")
worker 6 R39-BF07 > "$OUTPUT_ROOT/logs/gpu-6-worker.log" 2>&1 & PIDS+=("$!")
worker 7 R39-BF08 > "$OUTPUT_ROOT/logs/gpu-7-worker.log" 2>&1 & PIDS+=("$!")

worker_status=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    worker_status=1
  fi
done
touch "$OUTPUT_ROOT/stages/03_all_individual_outcomes_terminal"

PYTHONPATH="$EXECUTOR" "$PYTHON" "$EXECUTOR/r39_aggregate.py" \
  --run-root "$OUTPUT_ROOT" \
  --protocol "$PROTOCOL" \
  --plan "$PLAN" \
  --output "$OUTPUT_ROOT/summary.json" \
  > "$OUTPUT_ROOT/logs/aggregate.log" 2>&1
touch "$OUTPUT_ROOT/stages/04_aggregate_complete"
touch "$OUTPUT_ROOT/stages/05_complete"

find "$OUTPUT_ROOT" -type f ! -name terminal-files.sha256 -print0 \
  | sort -z | xargs -0 sha256sum > "$OUTPUT_ROOT/terminal-files.sha256"
sha256sum -c "$OUTPUT_ROOT/terminal-files.sha256"

# The aggregate is still emitted when one or more rows are operational-invalid;
# return nonzero so automation cannot mistake that campaign for all-valid.
exit "$worker_status"
