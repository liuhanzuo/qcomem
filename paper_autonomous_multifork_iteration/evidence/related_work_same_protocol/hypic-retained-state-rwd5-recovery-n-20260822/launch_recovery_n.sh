#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

PATH=/root/miniconda3/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/mpi/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/ucx/bin:/opt/amazon/efa/bin
export PATH
unset BASH_ENV ENV CDPATH PYTHONPATH PYTHONHOME CODE_DIR FREEZE_MANIFEST LIVE_DEBUG_ROOT \
  PYTHON_BIN OFFICIAL_REPO MODEL_DIR MODEL_WEIGHT_LEDGER MODEL_ARTIFACT_LEDGER \
  VALIDATION_DATA RUN_DIR INSTRUMENTED_REPO GIT_CONFIG_GLOBAL GIT_CONFIG_SYSTEM || true

K_ROOT=/tmp/rwd5-hypic-store-freeze-k
N_ROOT=/tmp/rwd5-hypic-store-recovery-n
MODEL_ROOT=/tmp/Qwen3.5-35B-A3B-hypic-model-view
K_MANIFEST=${K_ROOT}/SHA256SUMS
N_MANIFEST=${N_ROOT}/SHA256SUMS
EXPECTED_K_MANIFEST_SHA256=c7f0fcc0b44d6292af52f2d31a0770e7a74982c20eadc8317740106034dc3a7b
EXPECTED_N_MANIFEST_SHA256=${EXPECTED_RECOVERY_MANIFEST_SHA256:?supply externally audited recovery-N SHA256SUMS SHA256}
RUN_DIR_N=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/runs/qcomem/hypic-retained-state-rwd5-20260822n
INSTRUMENTED_REPO_N=/tmp/HYPIC-98147c0-rwd5-store-n
RECOVERY_RECEIPT_DIR=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/runs/qcomem/hypic-retained-state-rwd5-recovery-n-20260822
REPAIR_RECEIPT=${N_ROOT}/asset-mode-repair-receipt.json
K_LAUNCHER=${K_ROOT}/code/launch_hypic_retained_state_bytes_8gpu.sh

die() { printf '%s\n' "ERROR: $*" >&2; exit 1; }

for path in "$K_MANIFEST" "$N_MANIFEST" "$REPAIR_RECEIPT" "$K_LAUNCHER" "$MODEL_ROOT"; do
  [[ -e "$path" ]] || die "missing frozen path: $path"
done
[[ "$(/usr/bin/sha256sum "$K_MANIFEST" | /usr/bin/awk '{print $1}')" == "$EXPECTED_K_MANIFEST_SHA256" ]] || die "K manifest identity drift"
[[ "$(/usr/bin/sha256sum "$N_MANIFEST" | /usr/bin/awk '{print $1}')" == "$EXPECTED_N_MANIFEST_SHA256" ]] || die "N manifest identity drift"
(cd "$K_ROOT" && /usr/bin/sha256sum -c "$K_MANIFEST") || die "K frozen files drift"
(cd "$N_ROOT" && /usr/bin/sha256sum -c "$N_MANIFEST") || die "N frozen files drift"

check_asset() {
  local name=$1 expected_sha=$2 expected_size=$3 expected_inode=$4
  local path=${MODEL_ROOT}/${name}
  [[ -f "$path" && ! -L "$path" ]] || die "asset is not a regular non-symlink: $name"
  [[ "$(/usr/bin/sha256sum "$path" | /usr/bin/awk '{print $1}')" == "$expected_sha" ]] || die "asset SHA drift: $name"
  [[ "$(/usr/bin/stat -c '%a|%u|%g|%s|%i|%d|%Y|%Z' "$path")" == "444|0|0|${expected_size}|${expected_inode}|2097177|1787372672|1787376685" ]] \
    || die "asset post-repair stat drift: $name"
}

check_asset model-artifacts.sha256 \
  d78424684a27718b44616c80c788e7f8b726feed674fe015901b87de1993f7dd 778 58755972
check_asset preprocessor_config.json \
  27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516 390 58755952
check_asset video_preprocessor_config.json \
  7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13 385 58755969

mapfile -t writable_top < <(/usr/bin/find "$MODEL_ROOT" -maxdepth 1 -type f -perm /222 -printf '%f\n' | /usr/bin/sort)
[[ ${#writable_top[@]} -eq 0 ]] || die "writable top-level model assets remain: ${writable_top[*]}"
[[ ! -e "$RUN_DIR_N" ]] || die "fresh N RUN_DIR already exists"
[[ ! -e "$INSTRUMENTED_REPO_N" ]] || die "fresh N instrumented repo already exists"
[[ ! -e "$RECOVERY_RECEIPT_DIR" ]] || die "recovery receipt directory already exists"

/usr/bin/mkdir -p "$RECOVERY_RECEIPT_DIR"
/usr/bin/cp "$REPAIR_RECEIPT" "$RECOVERY_RECEIPT_DIR/asset-mode-repair-receipt.json"
/usr/bin/sha256sum "$K_MANIFEST" "$N_MANIFEST" "$RECOVERY_RECEIPT_DIR/asset-mode-repair-receipt.json" \
  > "$RECOVERY_RECEIPT_DIR/frozen-authority.sha256"
printf '%s\n' "preflight_passed_before_k_launcher" > "$RECOVERY_RECEIPT_DIR/STATUS"

# PINNED_CWD_BEGIN -- parsed and exercised by test_recovery_n_cwd_env.py.
for shadow in /test_hypic_retained_state_receipt.py /test_hypic_retained_state_receipt /sglang; do
  [[ ! -e "$shadow" ]] || die "unsafe fixed-cwd import shadow exists: $shadow"
done
cd /
[[ "$PWD" == "/" ]] || die "failed to enter fixed safe cwd"
/usr/bin/env -i PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 \
  /tmp/round25-hypic-env/venv/bin/python -c \
  'import os,sys; assert os.getcwd() == "/"; assert "" not in sys.path' \
  || die "frozen Python does not enforce safe sys.path from fixed cwd"
# PINNED_CWD_END

# PINNED_EXEC_ENV_BEGIN -- parsed by test_recovery_n_cwd_env.py.
exec /usr/bin/env -i \
  PATH=/root/miniconda3/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/mpi/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/ucx/bin:/opt/amazon/efa/bin \
  HOME=/root \
  USER=root \
  LOGNAME=root \
  SHELL=/bin/bash \
  LANG=C.UTF-8 \
  LC_ALL=C.UTF-8 \
  TZ=UTC \
  LD_LIBRARY_PATH=/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
  LIBRARY_PATH=/usr/local/cuda/lib64/stubs \
  CUDA_HOME=/usr/local/cuda \
  PYTHONNOUSERSITE=1 \
  PYTHONSAFEPATH=1 \
  PYTHON_BIN=/tmp/round25-hypic-env/venv/bin/python \
  OFFICIAL_REPO=/tmp/HYPIC-98147c0 \
  FREEZE_ROOT=/tmp/rwd5-hypic-store-freeze-k \
  CODE_DIR=/tmp/rwd5-hypic-store-freeze-k/code \
  FREEZE_MANIFEST=/tmp/rwd5-hypic-store-freeze-k/SHA256SUMS \
  EXPECTED_FREEZE_MANIFEST_SHA256=c7f0fcc0b44d6292af52f2d31a0770e7a74982c20eadc8317740106034dc3a7b \
  LIVE_DEBUG_ROOT=/tmp/rwd5-hypic-store-freeze-k/live-debug-j-trial-1879097 \
  MODEL_DIR=/tmp/Qwen3.5-35B-A3B-hypic-model-view \
  MODEL_WEIGHT_LEDGER=/tmp/Qwen3.5-35B-A3B-hypic-model-view/model-weights.sha256 \
  MODEL_ARTIFACT_LEDGER=/tmp/Qwen3.5-35B-A3B-hypic-model-view/model-artifacts.sha256 \
  VALIDATION_DATA=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/data/qcomem-longbench-validation/longbench_validation.jsonl \
  RUN_DIR=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/runs/qcomem/hypic-retained-state-rwd5-20260822n \
  INSTRUMENTED_REPO=/tmp/HYPIC-98147c0-rwd5-store-n \
  /bin/bash --noprofile --norc /tmp/rwd5-hypic-store-freeze-k/code/launch_hypic_retained_state_bytes_8gpu.sh
# PINNED_EXEC_ENV_END
