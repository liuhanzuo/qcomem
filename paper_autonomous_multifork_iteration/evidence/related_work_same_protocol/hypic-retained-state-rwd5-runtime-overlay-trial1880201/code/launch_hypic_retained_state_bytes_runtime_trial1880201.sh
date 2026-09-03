#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

RUNTIME_ROOT=/tmp/rwd5-hypic-store-runtime-trial1880201
RUNTIME_MANIFEST=${RUNTIME_ROOT}/RUNTIME-SHA256SUMS
EXPECTED_RUNTIME_MANIFEST_SHA256=${EXPECTED_RUNTIME_MANIFEST_SHA256:?supply the locally reported runtime-overlay manifest SHA256}
RUNTIME_LAUNCHER=${RUNTIME_ROOT}/code/launch_hypic_retained_state_bytes_8gpu.sh
MODEL_ROOT=/tmp/Qwen3.5-35B-A3B-hypic-model-view
ASSET_OBSERVATION=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/runs/qcomem/hypic-retained-state-rwd5-preflight-trial1880201/model-asset-observation.json
RUN_DIR_RUNTIME=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/runs/qcomem/hypic-retained-state-rwd5-direct-trial1880201-20260822
INSTRUMENTED_RUNTIME=/tmp/HYPIC-98147c0-rwd5-store-trial1880201

die() { printf '%s\n' "ERROR: $*" >&2; exit 1; }

[[ -f "$RUNTIME_MANIFEST" && -f "$RUNTIME_LAUNCHER" ]] || die "runtime overlay is incomplete"
[[ "$(/usr/bin/sha256sum "$RUNTIME_MANIFEST" | /usr/bin/awk '{print $1}')" == "$EXPECTED_RUNTIME_MANIFEST_SHA256" ]] \
  || die "runtime overlay manifest identity drift"
(cd "$RUNTIME_ROOT" && /usr/bin/sha256sum -c "$RUNTIME_MANIFEST") \
  || die "runtime overlay member drift"

require_pid1() {
  local key=$1 expected=$2 count value
  count=$(/usr/bin/tr '\0' '\n' </proc/1/environ | /usr/bin/awk -F= -v key="$key" '$1==key {n++} END {print n+0}')
  value=$(/usr/bin/tr '\0' '\n' </proc/1/environ | /usr/bin/awk -F= -v key="$key" '$1==key {print substr($0,length(key)+2)}')
  [[ "$count" -eq 1 && "$value" == "$expected" ]] || die "PID-1 identity mismatch: $key"
}
require_pid1 QS_JOB_ID 247699
require_pid1 QS_TRIAL_ID 1880201
require_pid1 QCOMEM_DEBUG_SCOPE ROUND27_HYPIC_STORE_FORMAL_W

[[ -f "$ASSET_OBSERVATION" ]] || die "read-only inventory/prep did not publish the exact asset observation"
[[ ! -e "$RUN_DIR_RUNTIME" ]] || die "fresh runtime RUN_DIR already exists"
[[ ! -e "$INSTRUMENTED_RUNTIME" ]] || die "fresh runtime instrumented repository already exists"

cd /
exec /usr/bin/env -i \
  PATH=/root/miniconda3/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/mpi/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/ucx/bin:/opt/amazon/efa/bin \
  HOME=/root USER=root LOGNAME=root SHELL=/bin/bash LANG=C.UTF-8 LC_ALL=C.UTF-8 TZ=UTC \
  LD_LIBRARY_PATH=/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
  LIBRARY_PATH=/usr/local/cuda/lib64/stubs CUDA_HOME=/usr/local/cuda \
  PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 RWD5_RUNTIME_OVERLAY_EXEC=1 \
  PYTHON_BIN=/tmp/round25-hypic-env/venv/bin/python \
  OFFICIAL_REPO=/tmp/HYPIC-98147c0 \
  FREEZE_ROOT="$RUNTIME_ROOT" CODE_DIR="$RUNTIME_ROOT/code" \
  FREEZE_MANIFEST="$RUNTIME_MANIFEST" EXPECTED_FREEZE_MANIFEST_SHA256="$EXPECTED_RUNTIME_MANIFEST_SHA256" \
  LIVE_DEBUG_ROOT="$RUNTIME_ROOT/live-debug-j-trial-1879097" \
  ALLOCATOR_DEBUG_ROOT="$RUNTIME_ROOT/live-allocator-debug-d-trial-1879456" \
  ALLOCATOR_DEBUG_PROVENANCE="$RUNTIME_ROOT/allocator-debug-d-provenance.json" \
  ALLOCATOR_DEBUG_LAUNCH_PLAN="$RUNTIME_ROOT/allocator-debug-d-launch-plan.json" \
  ALLOCATOR_DEBUG_FREEZE_MANIFEST="$RUNTIME_ROOT/allocator-debug-d-freeze-SHA256SUMS" \
  MODEL_DIR="$MODEL_ROOT" \
  MODEL_WEIGHT_LEDGER="$MODEL_ROOT/model-weights.sha256" \
  MODEL_ARTIFACT_LEDGER="$MODEL_ROOT/model-artifacts.sha256" \
  VALIDATION_DATA=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/data/qcomem-longbench-validation/longbench_validation.jsonl \
  RUN_DIR="$RUN_DIR_RUNTIME" INSTRUMENTED_REPO="$INSTRUMENTED_RUNTIME" \
  INHERITED_TEST="$RUNTIME_ROOT/code/test_run_hypic_same_protocol.py" \
  INHERITED_LAUNCHER="$RUNTIME_ROOT/code/launch_hypic_same_protocol_8gpu.sh" \
  SAFE_WRAPPER="$RUNTIME_ROOT/code/launch_hypic_retained_state_bytes_runtime_trial1880201.sh" \
  SAFE_CWD_GUARD="$RUNTIME_ROOT/code/rwd5_safe_cwd_guard.py" \
  MODEL_ASSET_SNAPSHOT="$RUNTIME_ROOT/code/rwd5_model_asset_snapshot.py" \
  ASSET_OBSERVATION_PATH="$ASSET_OBSERVATION" \
  INVALID_T_RECEIPT="$RUNTIME_ROOT/invalid-formal-t-job247574-trial1879456.json" \
  PLATFORM_AUTHORITY_RECEIPT="$RUNTIME_ROOT/platform-execution-authority-runtime-trial1880201.json" \
  RETIRED_W_TRIAL_RECEIPT="$RUNTIME_ROOT/external-manual-stop-trial1879843.json" \
  RWD5_PID1_ENV_PATH=/proc/1/environ RWD5_PLATFORM_JOB_ID=247699 RWD5_PLATFORM_TRIAL_ID=1880201 \
  /bin/bash --noprofile --norc "$RUNTIME_LAUNCHER"
