#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 2 || "$#" -gt 3 ]]; then
  echo "usage: $0 ARCHIVE EXPECTED_ARCHIVE_SHA256 [TRIAL_ID]" >&2
  exit 64
fi

ARCHIVE="$1"
EXPECTED_ARCHIVE_SHA256="$2"
TRIAL_ID="${3:-1907355}"
SHARED=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo
STAGE="$SHARED/indep-bench/qcomem_r39_blind_faults_20260826a"
OUTPUT="$SHARED/indep-bench_assets/runs/qcomem/r39-blind-faults-20260826a"
ENV_DIR="$SHARED/indep-bench_assets/envs/vllm-cu129-v1"
RR2_CODE_DIR="$SHARED/indep-bench/qcomem_gpu_forkaudit_rr2_formal_20260818w/gpu"

[[ -f "$ARCHIVE" ]] || { echo "archive absent: $ARCHIVE" >&2; exit 65; }
OBSERVED_SHA256="$(sha256sum "$ARCHIVE" | awk '{print $1}')"
[[ "$OBSERVED_SHA256" == "$EXPECTED_ARCHIVE_SHA256" ]] || {
  echo "archive SHA mismatch: $OBSERVED_SHA256" >&2
  exit 66
}
[[ ! -e "$STAGE" ]] || { echo "refusing existing stage: $STAGE" >&2; exit 67; }
[[ ! -e "$OUTPUT" ]] || { echo "refusing existing output: $OUTPUT" >&2; exit 68; }

mkdir -p "$STAGE"
tar -xzf "$ARCHIVE" -C "$STAGE" --no-same-owner
EXECUTION_INPUT="$STAGE/paper_autonomous_multifork_iteration/evidence/r29_heldout_faults/cross_execution/execution-input-v3.json"

bash "$STAGE/paper_autonomous_multifork_iteration/evidence/r39_blind_faults/executor/r39_launch_8gpu.sh" \
  "$STAGE" \
  "$OUTPUT" \
  "$ENV_DIR" \
  "$RR2_CODE_DIR" \
  "$EXECUTION_INPUT" \
  "$TRIAL_ID" \
  "$EXPECTED_ARCHIVE_SHA256"
