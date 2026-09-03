#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

PATH=/root/miniconda3/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/mpi/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/ucx/bin:/opt/amazon/efa/bin
export PATH
unset BASH_ENV ENV CDPATH PYTHONPATH PYTHONHOME CODE_DIR FREEZE_MANIFEST LIVE_DEBUG_ROOT \
  ALLOCATOR_DEBUG_ROOT ALLOCATOR_DEBUG_PROVENANCE ALLOCATOR_DEBUG_LAUNCH_PLAN \
  ALLOCATOR_DEBUG_FREEZE_MANIFEST \
  PYTHON_BIN OFFICIAL_REPO MODEL_DIR MODEL_WEIGHT_LEDGER MODEL_ARTIFACT_LEDGER \
  VALIDATION_DATA RUN_DIR INSTRUMENTED_REPO GIT_CONFIG_GLOBAL GIT_CONFIG_SYSTEM || true

Q_ROOT=/tmp/rwd5-hypic-store-freeze-q
R_ROOT=/tmp/rwd5-hypic-store-recovery-r
MODEL_ROOT=/tmp/Qwen3.5-35B-A3B-hypic-model-view
Q_MANIFEST=${Q_ROOT}/SHA256SUMS
R_MANIFEST=${R_ROOT}/SHA256SUMS
EXPECTED_Q_MANIFEST_SHA256=b56f35527ca56c27a3dc951752d95cc2335ed1489e7e655b721eea3710ca8274
EXPECTED_R_MANIFEST_SHA256=${EXPECTED_RECOVERY_MANIFEST_SHA256:?supply externally audited recovery-R SHA256SUMS SHA256}
Q_STOP=${Q_ROOT}/STOP
EXPECTED_Q_STOP_SHA256=21ffb897f17df1eb45c6ccb513678cdc51c2325e875cc5fb7b32ac58dae0d8df
RUN_DIR_R=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/runs/qcomem/hypic-retained-state-rwd5-20260822r
INSTRUMENTED_REPO_R=/tmp/HYPIC-98147c0-rwd5-store-r
RECOVERY_RECEIPT_DIR=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/runs/qcomem/hypic-retained-state-rwd5-recovery-r-20260822
REPAIR_RECEIPT=${R_ROOT}/asset-mode-repair-receipt.json
Q_LAUNCHER=${Q_ROOT}/code/launch_hypic_retained_state_bytes_8gpu.sh

die() { printf '%s\n' "ERROR: $*" >&2; exit 1; }

for path in "$Q_MANIFEST" "$Q_STOP" "$R_MANIFEST" "$REPAIR_RECEIPT" "$Q_LAUNCHER" "$MODEL_ROOT"; do
  [[ -e "$path" ]] || die "missing frozen path: $path"
done
[[ "$(/usr/bin/sha256sum "$Q_MANIFEST" | /usr/bin/awk '{print $1}')" == "$EXPECTED_Q_MANIFEST_SHA256" ]] || die "Q manifest identity drift"
[[ "$(/usr/bin/sha256sum "$Q_STOP" | /usr/bin/awk '{print $1}')" == "$EXPECTED_Q_STOP_SHA256" ]] || die "Q STOP identity drift"
[[ "$(/usr/bin/sha256sum "$R_MANIFEST" | /usr/bin/awk '{print $1}')" == "$EXPECTED_R_MANIFEST_SHA256" ]] || die "R manifest identity drift"
(cd "$Q_ROOT" && /usr/bin/sha256sum -c "$Q_MANIFEST") || die "Q frozen files drift"
(cd "$R_ROOT" && /usr/bin/sha256sum -c "$R_MANIFEST") || die "R frozen files drift"

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
[[ ! -e "$RUN_DIR_R" ]] || die "fresh R RUN_DIR already exists"
[[ ! -e "$INSTRUMENTED_REPO_R" ]] || die "fresh R instrumented repo already exists"
[[ ! -e "$RECOVERY_RECEIPT_DIR" ]] || die "recovery receipt directory already exists"

# PINNED_CWD_BEGIN -- parsed and exercised by test_recovery_r_publication.py.
/usr/bin/env -i PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 \
  /tmp/round25-hypic-env/venv/bin/python "${R_ROOT}/safe_cwd_guard.py" /
cd /
[[ "$PWD" == "/" ]] || die "failed to enter fixed safe cwd"

/usr/bin/env -i PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 \
  /tmp/round25-hypic-env/venv/bin/python -c \
  'import os,sys; assert os.getcwd() == "/"; assert "" not in sys.path; assert "/" not in sys.path' \
  || die "frozen Python does not enforce safe sys.path from fixed cwd"

/usr/bin/env -i PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 \
  PYTHONPATH=/tmp/rwd5-hypic-store-freeze-q/code \
  /tmp/round25-hypic-env/venv/bin/python -c \
  'import importlib.util,pathlib; s=importlib.util.find_spec("test_hypic_retained_state_receipt"); assert s and s.origin; assert pathlib.Path(s.origin).resolve(strict=True) == pathlib.Path("/tmp/rwd5-hypic-store-freeze-q/code/test_hypic_retained_state_receipt.py").resolve(strict=True)' \
  || die "focused unittest import authority is not exact Q code"

/usr/bin/env -i PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 \
  PYTHONPATH=/tmp/HYPIC-98147c0/python:/tmp/rwd5-hypic-store-freeze-q/code \
  /tmp/round25-hypic-env/venv/bin/python -c \
  'import importlib.util,pathlib; s=importlib.util.find_spec("sglang"); assert s and s.origin and s.submodule_search_locations; root=pathlib.Path("/tmp/HYPIC-98147c0/python/sglang").resolve(strict=True); assert pathlib.Path(s.origin).resolve(strict=True).is_relative_to(root); assert {pathlib.Path(p).resolve(strict=True) for p in s.submodule_search_locations} == {root}' \
  || die "SGLang import authority is not exact official repository"
# PINNED_CWD_END

# AUTHORITY_PUBLICATION_BEGIN -- must remain after every probe and before exec.
/usr/bin/mkdir -p "$RECOVERY_RECEIPT_DIR"
/usr/bin/cp "$REPAIR_RECEIPT" "$RECOVERY_RECEIPT_DIR/asset-mode-repair-receipt.json"
/usr/bin/sha256sum "$Q_MANIFEST" "$Q_STOP" "$R_MANIFEST" "$RECOVERY_RECEIPT_DIR/asset-mode-repair-receipt.json" \
  > "$RECOVERY_RECEIPT_DIR/frozen-authority.sha256"
printf '%s\n' "all_recovery_preflight_passed_before_q_exec" > "$RECOVERY_RECEIPT_DIR/STATUS"
# AUTHORITY_PUBLICATION_END

# PINNED_EXEC_ENV_BEGIN -- parsed by test_recovery_r_publication.py.
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
  FREEZE_ROOT=/tmp/rwd5-hypic-store-freeze-q \
  CODE_DIR=/tmp/rwd5-hypic-store-freeze-q/code \
  FREEZE_MANIFEST=/tmp/rwd5-hypic-store-freeze-q/SHA256SUMS \
  EXPECTED_FREEZE_MANIFEST_SHA256=b56f35527ca56c27a3dc951752d95cc2335ed1489e7e655b721eea3710ca8274 \
  LIVE_DEBUG_ROOT=/tmp/rwd5-hypic-store-freeze-q/live-debug-j-trial-1879097 \
  ALLOCATOR_DEBUG_ROOT=/tmp/rwd5-hypic-store-freeze-q/live-allocator-debug-d-trial-1879456 \
  ALLOCATOR_DEBUG_PROVENANCE=/tmp/rwd5-hypic-store-freeze-q/allocator-debug-d-provenance.json \
  ALLOCATOR_DEBUG_LAUNCH_PLAN=/tmp/rwd5-hypic-store-freeze-q/allocator-debug-d-launch-plan.json \
  ALLOCATOR_DEBUG_FREEZE_MANIFEST=/tmp/rwd5-hypic-store-freeze-q/allocator-debug-d-freeze-SHA256SUMS \
  MODEL_DIR=/tmp/Qwen3.5-35B-A3B-hypic-model-view \
  MODEL_WEIGHT_LEDGER=/tmp/Qwen3.5-35B-A3B-hypic-model-view/model-weights.sha256 \
  MODEL_ARTIFACT_LEDGER=/tmp/Qwen3.5-35B-A3B-hypic-model-view/model-artifacts.sha256 \
  VALIDATION_DATA=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/data/qcomem-longbench-validation/longbench_validation.jsonl \
  RUN_DIR=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/runs/qcomem/hypic-retained-state-rwd5-20260822r \
  INSTRUMENTED_REPO=/tmp/HYPIC-98147c0-rwd5-store-r \
  /bin/bash --noprofile --norc /tmp/rwd5-hypic-store-freeze-q/code/launch_hypic_retained_state_bytes_8gpu.sh
# PINNED_EXEC_ENV_END
