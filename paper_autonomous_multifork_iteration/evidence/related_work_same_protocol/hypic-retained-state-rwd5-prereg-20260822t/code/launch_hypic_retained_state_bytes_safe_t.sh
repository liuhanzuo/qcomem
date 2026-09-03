#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

PATH=/root/miniconda3/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/mpi/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/ucx/bin:/opt/amazon/efa/bin
export PATH
unset BASH_ENV ENV CDPATH PYTHONPATH PYTHONHOME CODE_DIR FREEZE_MANIFEST LIVE_DEBUG_ROOT \
  ALLOCATOR_DEBUG_ROOT ALLOCATOR_DEBUG_PROVENANCE ALLOCATOR_DEBUG_LAUNCH_PLAN \
  ALLOCATOR_DEBUG_FREEZE_MANIFEST PYTHON_BIN OFFICIAL_REPO MODEL_DIR \
  MODEL_WEIGHT_LEDGER MODEL_ARTIFACT_LEDGER VALIDATION_DATA RUN_DIR \
  INSTRUMENTED_REPO INHERITED_TEST INHERITED_LAUNCHER SAFE_WRAPPER SAFE_CWD_GUARD \
  GIT_CONFIG_GLOBAL GIT_CONFIG_SYSTEM || true

T_ROOT=/tmp/rwd5-hypic-store-freeze-t
T_MANIFEST=${T_ROOT}/SHA256SUMS
EXPECTED_T_MANIFEST_SHA256=${EXPECTED_T_MANIFEST_SHA256:?supply externally audited T SHA256SUMS SHA256}
T_STOP=${T_ROOT}/STOP
EXPECTED_T_STOP_SHA256=${EXPECTED_T_STOP_SHA256:?supply externally audited T STOP SHA256}
MODEL_ROOT=/tmp/Qwen3.5-35B-A3B-hypic-model-view
RUN_DIR_T=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/runs/qcomem/hypic-retained-state-rwd5-20260822t
INSTRUMENTED_REPO_T=/tmp/HYPIC-98147c0-rwd5-store-t
T_LAUNCHER=${T_ROOT}/code/launch_hypic_retained_state_bytes_8gpu.sh
T_WRAPPER=${T_ROOT}/code/launch_hypic_retained_state_bytes_safe_t.sh
T_GUARD=${T_ROOT}/code/rwd5_safe_cwd_guard.py

die() { printf '%s\n' "ERROR: $*" >&2; exit 1; }

for path in "$T_MANIFEST" "$T_STOP" "$T_LAUNCHER" "$T_WRAPPER" "$T_GUARD" "$MODEL_ROOT"; do
  [[ -e "$path" ]] || die "missing frozen path: $path"
done
[[ "$(/usr/bin/sha256sum "$T_MANIFEST" | /usr/bin/awk '{print $1}')" == "$EXPECTED_T_MANIFEST_SHA256" ]] || die "T manifest identity drift"
[[ "$(/usr/bin/sha256sum "$T_STOP" | /usr/bin/awk '{print $1}')" == "$EXPECTED_T_STOP_SHA256" ]] || die "T STOP identity drift"
(cd "$T_ROOT" && /usr/bin/sha256sum -c "$T_MANIFEST") || die "T frozen files drift"
[[ ! -e "$RUN_DIR_T" ]] || die "fresh T RUN_DIR already exists"
[[ ! -e "$INSTRUMENTED_REPO_T" ]] || die "fresh T instrumented repo already exists"

check_asset() {
  local name=$1 expected_sha=$2 expected_size=$3 expected_inode=$4
  local path=${MODEL_ROOT}/${name}
  [[ -f "$path" && ! -L "$path" ]] || die "asset is not a regular non-symlink: $name"
  [[ "$(/usr/bin/sha256sum "$path" | /usr/bin/awk '{print $1}')" == "$expected_sha" ]] || die "asset SHA drift: $name"
  [[ "$(/usr/bin/stat -c '%a|%u|%g|%s|%i|%d|%Y|%Z' "$path")" == "444|0|0|${expected_size}|${expected_inode}|2097177|1787372672|1787376685" ]] \
    || die "asset post-repair stat drift: $name"
}

check_asset model-artifacts.sha256 d78424684a27718b44616c80c788e7f8b726feed674fe015901b87de1993f7dd 778 58755972
check_asset preprocessor_config.json 27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516 390 58755952
check_asset video_preprocessor_config.json 7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13 385 58755969
mapfile -t writable_top < <(/usr/bin/find "$MODEL_ROOT" -maxdepth 1 -type f -perm /222 -printf '%f\n' | /usr/bin/sort)
[[ ${#writable_top[@]} -eq 0 ]] || die "writable top-level model assets remain: ${writable_top[*]}"

/usr/bin/env -i PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 \
  /tmp/round25-hypic-env/venv/bin/python "$T_GUARD" /
cd /
[[ "$PWD" == "/" ]] || die "failed to enter fixed safe cwd"

/usr/bin/env -i PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 \
  /tmp/round25-hypic-env/venv/bin/python -c \
  'import os,sys; assert os.getcwd() == "/"; assert "" not in sys.path; assert "/" not in sys.path' \
  || die "frozen Python does not enforce safe sys.path from fixed cwd"

/usr/bin/env -i PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONPATH=/tmp/rwd5-hypic-store-freeze-t/code \
  /tmp/round25-hypic-env/venv/bin/python -c \
  'import importlib.util,pathlib; root=pathlib.Path("/tmp/rwd5-hypic-store-freeze-t/code").resolve(strict=True); expected={"test_hypic_retained_state_receipt":root/"test_hypic_retained_state_receipt.py","test_run_hypic_same_protocol":root/"test_run_hypic_same_protocol.py"}; [(lambda s,p: (_ for _ in ()).throw(AssertionError()) if not (s and s.origin and pathlib.Path(s.origin).resolve(strict=True)==p.resolve(strict=True)) else None)(importlib.util.find_spec(n),p) for n,p in expected.items()]' \
  || die "frozen unittest import authority is not exact T code"

/usr/bin/env -i PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 \
  PYTHONPATH=/tmp/HYPIC-98147c0/python:/tmp/rwd5-hypic-store-freeze-t/code \
  /tmp/round25-hypic-env/venv/bin/python -c \
  'import importlib.util,pathlib; s=importlib.util.find_spec("sglang"); assert s and s.origin and s.submodule_search_locations; root=pathlib.Path("/tmp/HYPIC-98147c0/python/sglang").resolve(strict=True); assert pathlib.Path(s.origin).resolve(strict=True).is_relative_to(root); assert {pathlib.Path(p).resolve(strict=True) for p in s.submodule_search_locations} == {root}' \
  || die "SGLang import authority is not exact official repository"

# PINNED_EXEC_ENV_BEGIN -- parsed by the frozen adversarial tests.
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
  RWD5_SAFE_WRAPPER_EXEC=1 \
  PYTHON_BIN=/tmp/round25-hypic-env/venv/bin/python \
  OFFICIAL_REPO=/tmp/HYPIC-98147c0 \
  FREEZE_ROOT=/tmp/rwd5-hypic-store-freeze-t \
  CODE_DIR=/tmp/rwd5-hypic-store-freeze-t/code \
  FREEZE_MANIFEST=/tmp/rwd5-hypic-store-freeze-t/SHA256SUMS \
  EXPECTED_FREEZE_MANIFEST_SHA256="$EXPECTED_T_MANIFEST_SHA256" \
  LIVE_DEBUG_ROOT=/tmp/rwd5-hypic-store-freeze-t/live-debug-j-trial-1879097 \
  ALLOCATOR_DEBUG_ROOT=/tmp/rwd5-hypic-store-freeze-t/live-allocator-debug-d-trial-1879456 \
  ALLOCATOR_DEBUG_PROVENANCE=/tmp/rwd5-hypic-store-freeze-t/allocator-debug-d-provenance.json \
  ALLOCATOR_DEBUG_LAUNCH_PLAN=/tmp/rwd5-hypic-store-freeze-t/allocator-debug-d-launch-plan.json \
  ALLOCATOR_DEBUG_FREEZE_MANIFEST=/tmp/rwd5-hypic-store-freeze-t/allocator-debug-d-freeze-SHA256SUMS \
  MODEL_DIR=/tmp/Qwen3.5-35B-A3B-hypic-model-view \
  MODEL_WEIGHT_LEDGER=/tmp/Qwen3.5-35B-A3B-hypic-model-view/model-weights.sha256 \
  MODEL_ARTIFACT_LEDGER=/tmp/Qwen3.5-35B-A3B-hypic-model-view/model-artifacts.sha256 \
  VALIDATION_DATA=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/data/qcomem-longbench-validation/longbench_validation.jsonl \
  RUN_DIR=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/runs/qcomem/hypic-retained-state-rwd5-20260822t \
  INSTRUMENTED_REPO=/tmp/HYPIC-98147c0-rwd5-store-t \
  INHERITED_TEST=/tmp/rwd5-hypic-store-freeze-t/code/test_run_hypic_same_protocol.py \
  INHERITED_LAUNCHER=/tmp/rwd5-hypic-store-freeze-t/code/launch_hypic_same_protocol_8gpu.sh \
  SAFE_WRAPPER=/tmp/rwd5-hypic-store-freeze-t/code/launch_hypic_retained_state_bytes_safe_t.sh \
  SAFE_CWD_GUARD=/tmp/rwd5-hypic-store-freeze-t/code/rwd5_safe_cwd_guard.py \
  /bin/bash --noprofile --norc /tmp/rwd5-hypic-store-freeze-t/code/launch_hypic_retained_state_bytes_8gpu.sh
# PINNED_EXEC_ENV_END
