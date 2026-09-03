#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 8 ]]; then
  echo "usage: $0 PACKAGE_ROOT OUTPUT_ROOT PROTOCOL EXECUTION_INPUT FAULTS_JSON ENV_DIR RR2_CODE_DIR TRIAL_ID" >&2
  exit 64
fi

PACKAGE_ROOT="$1"
OUTPUT_ROOT="$2"
PROTOCOL="$3"
EXECUTION_INPUT="$4"
FAULTS_JSON="$5"
ENV_DIR="$6"
RR2_CODE_DIR="$7"
TRIAL_ID="$8"

[[ ! -e "$OUTPUT_ROOT" ]] || { echo "refusing existing output root: $OUTPUT_ROOT" >&2; exit 65; }
mkdir -p "$OUTPUT_ROOT/logs"

SCRIPTS="$PACKAGE_ROOT/paper_autonomous_multifork_iteration/scripts"
GPU_CODE="$PACKAGE_ROOT/gpu"
PYTHON="$ENV_DIR/bin/python"
REPLAY="$SCRIPTS/r33_fault_replay.py"
PROTOCOL_SHA="$($PYTHON -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$PROTOCOL")"

mapfile -t GPU_UUIDS < <(nvidia-smi --query-gpu=uuid --format=csv,noheader | tr -d ' ')
[[ "${#GPU_UUIDS[@]}" -ge 5 ]] || { echo "R33 requires five visible physical GPUs" >&2; exit 66; }

FAULT_IDS=(
  HF01_DELAYED_TAIL_DETACH
  HF02_INACTIVE_DOCUMENT_LANE_SCRIBBLE
  HF03_DUPLICATE_COMMITTED_DISPATCH
  HF04_EFFECTIVE_SCALE_DRIFT
  HF05_STALE_GDN_BINDING_TOKEN_AFTER_REBIND
)

PIDS=()
for rank in 0 1 2 3 4; do
  rank_run="$OUTPUT_ROOT/rank-run-$rank"
  log="$OUTPUT_ROOT/logs/rank-$rank.log"
  CUDA_VISIBLE_DEVICES="${GPU_UUIDS[$rank]}" \
  PYTHONPATH="$SCRIPTS:$GPU_CODE:$RR2_CODE_DIR" \
    "$PYTHON" "$SCRIPTS/r33_execute_fresh_faults.py" \
      --rank "$rank" \
      --fault-id "${FAULT_IDS[$rank]}" \
      --run-dir "$rank_run" \
      --protocol "$PROTOCOL" \
      --expected-protocol-sha256 "$PROTOCOL_SHA" \
      --faults-json "$FAULTS_JSON" \
      --execution-input "$EXECUTION_INPUT" \
      --expected-gpu-uuid "${GPU_UUIDS[$rank]}" \
      --replay-source "$REPLAY" \
      --trial-id "$TRIAL_ID" \
      >"$log" 2>&1 &
  PIDS+=("$!")
done

status=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
[[ "$status" -eq 0 ]] || { echo "one or more R33 ranks failed; aggregate not emitted" >&2; exit 67; }

PYTHONPATH="$SCRIPTS" "$PYTHON" "$SCRIPTS/r33_aggregate_fresh_faults.py" \
  --protocol "$PROTOCOL" \
  --expected-protocol-sha256 "$PROTOCOL_SHA" \
  --rank-run-root "$OUTPUT_ROOT" \
  --output "$OUTPUT_ROOT/summary.json" \
  >"$OUTPUT_ROOT/logs/aggregate.log" 2>&1

find "$OUTPUT_ROOT" -type f ! -name terminal-files.sha256 -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  >"$OUTPUT_ROOT/terminal-files.sha256"
sha256sum -c "$OUTPUT_ROOT/terminal-files.sha256"
