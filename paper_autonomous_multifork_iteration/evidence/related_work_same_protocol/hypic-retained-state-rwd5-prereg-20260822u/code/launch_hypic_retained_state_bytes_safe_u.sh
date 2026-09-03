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
  MODEL_ASSET_SNAPSHOT ASSET_OBSERVATION_PATH \
  INVALID_T_RECEIPT \
  GIT_CONFIG_GLOBAL GIT_CONFIG_SYSTEM || true

U_ROOT=/tmp/rwd5-hypic-store-freeze-u
U_MANIFEST=${U_ROOT}/SHA256SUMS
EXPECTED_U_MANIFEST_SHA256=${EXPECTED_U_MANIFEST_SHA256:?supply externally audited U SHA256SUMS SHA256}
U_STOP=${U_ROOT}/STOP
EXPECTED_U_STOP_SHA256=${EXPECTED_U_STOP_SHA256:?supply externally audited U STOP SHA256}
MODEL_ROOT=/tmp/Qwen3.5-35B-A3B-hypic-model-view
RUN_DIR_U=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/runs/qcomem/hypic-retained-state-rwd5-20260822u
INSTRUMENTED_REPO_U=/tmp/HYPIC-98147c0-rwd5-store-u
U_LAUNCHER=${U_ROOT}/code/launch_hypic_retained_state_bytes_8gpu.sh
U_WRAPPER=${U_ROOT}/code/launch_hypic_retained_state_bytes_safe_u.sh
U_GUARD=${U_ROOT}/code/rwd5_safe_cwd_guard.py
U_ASSET_SNAPSHOT=${U_ROOT}/code/rwd5_model_asset_snapshot.py
U_PREFLIGHT_DIR=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/runs/qcomem/hypic-retained-state-rwd5-preflight-u-20260822
U_ASSET_OBSERVATION=${U_PREFLIGHT_DIR}/model-asset-observation.json

die() { printf '%s\n' "ERROR: $*" >&2; exit 1; }

for path in "$U_MANIFEST" "$U_STOP" "$U_LAUNCHER" "$U_WRAPPER" "$U_GUARD" "$U_ASSET_SNAPSHOT" "$MODEL_ROOT"; do
  [[ -e "$path" ]] || die "missing frozen path: $path"
done
[[ "$(/usr/bin/sha256sum "$U_MANIFEST" | /usr/bin/awk '{print $1}')" == "$EXPECTED_U_MANIFEST_SHA256" ]] || die "U manifest identity drift"
[[ "$(/usr/bin/sha256sum "$U_STOP" | /usr/bin/awk '{print $1}')" == "$EXPECTED_U_STOP_SHA256" ]] || die "U STOP identity drift"
(cd "$U_ROOT" && /usr/bin/sha256sum -c "$U_MANIFEST") || die "U frozen files drift"
[[ ! -e "$RUN_DIR_U" ]] || die "fresh U RUN_DIR already exists"
[[ ! -e "$INSTRUMENTED_REPO_U" ]] || die "fresh U instrumented repo already exists"
[[ ! -e "$U_PREFLIGHT_DIR" ]] || die "fresh U preflight observation path already exists"

check_asset_semantic() {
  local name=$1 expected_sha=$2 expected_size=$3
  local path=${MODEL_ROOT}/${name}
  [[ -f "$path" && ! -L "$path" ]] || die "asset is not a regular non-symlink: $name"
  [[ "$(/usr/bin/sha256sum "$path" | /usr/bin/awk '{print $1}')" == "$expected_sha" ]] || die "asset SHA drift: $name"
  [[ "$(/usr/bin/stat -c '%a|%u|%g|%s' "$path")" == "444|0|0|${expected_size}" ]] \
    || die "asset stable semantic identity drift: $name"
}

check_asset_semantic model-artifacts.sha256 d78424684a27718b44616c80c788e7f8b726feed674fe015901b87de1993f7dd 778
check_asset_semantic preprocessor_config.json 27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516 390
check_asset_semantic video_preprocessor_config.json 7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13 385
[[ "$(/usr/bin/sha256sum "$MODEL_ROOT/model-weights.sha256" | /usr/bin/awk '{print $1}')" == "8314a82c9188b9b817193e039b0b0eb0636b328512f19b0c12455853b7e20014" ]] || die "model weight ledger SHA drift"
[[ "$(/usr/bin/sha256sum "$MODEL_ROOT/model-artifacts.sha256" | /usr/bin/awk '{print $1}')" == "d78424684a27718b44616c80c788e7f8b726feed674fe015901b87de1993f7dd" ]] || die "model artifact ledger SHA drift"
(cd "$MODEL_ROOT" && /usr/bin/sha256sum -c model-weights.sha256) || die "one or more of 14 model weight entries drifted"
(cd "$MODEL_ROOT" && /usr/bin/sha256sum -c model-artifacts.sha256) || die "one or more of 9 model artifact entries drifted"
mapfile -t writable_top < <(/usr/bin/find "$MODEL_ROOT" -maxdepth 1 -type f -perm /222 -printf '%f\n' | /usr/bin/sort)
[[ ${#writable_top[@]} -eq 0 ]] || die "writable top-level model assets remain: ${writable_top[*]}"

# The first O_NOFOLLOW/open/fstat/hash snapshot occurs only after every stable
# semantic check, every 14+9 ledger entry, and the top-level writable-file gate.
ASSET_PRE_SNAPSHOT=$(/usr/bin/env -i PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 \
  /tmp/round25-hypic-env/venv/bin/python "$U_ASSET_SNAPSHOT" --model-root "$MODEL_ROOT") \
  || die "pre-authority asset snapshot failed"

/usr/bin/env -i PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 \
  /tmp/round25-hypic-env/venv/bin/python "$U_GUARD" /
cd /
[[ "$PWD" == "/" ]] || die "failed to enter fixed safe cwd"

/usr/bin/env -i PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 \
  /tmp/round25-hypic-env/venv/bin/python -c \
  'import os,sys; assert os.getcwd() == "/"; assert "" not in sys.path; assert "/" not in sys.path' \
  || die "frozen Python does not enforce safe sys.path from fixed cwd"

/usr/bin/env -i PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONPATH=/tmp/rwd5-hypic-store-freeze-u/code \
  /tmp/round25-hypic-env/venv/bin/python -c \
  'import importlib.util,pathlib; root=pathlib.Path("/tmp/rwd5-hypic-store-freeze-u/code").resolve(strict=True); expected={"test_hypic_retained_state_receipt":root/"test_hypic_retained_state_receipt.py","test_run_hypic_same_protocol":root/"test_run_hypic_same_protocol.py"}; [(lambda s,p: (_ for _ in ()).throw(AssertionError()) if not (s and s.origin and pathlib.Path(s.origin).resolve(strict=True)==p.resolve(strict=True)) else None)(importlib.util.find_spec(n),p) for n,p in expected.items()]' \
  || die "frozen unittest import authority is not exact U code"

/usr/bin/env -i PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 \
  PYTHONPATH=/tmp/HYPIC-98147c0/python:/tmp/rwd5-hypic-store-freeze-u/code \
  /tmp/round25-hypic-env/venv/bin/python -c \
  'import importlib.util,pathlib; s=importlib.util.find_spec("sglang"); assert s and s.origin and s.submodule_search_locations; root=pathlib.Path("/tmp/HYPIC-98147c0/python/sglang").resolve(strict=True); assert pathlib.Path(s.origin).resolve(strict=True).is_relative_to(root); assert {pathlib.Path(p).resolve(strict=True) for p in s.submodule_search_locations} == {root}' \
  || die "SGLang import authority is not exact official repository"

# This is the last authority read before publication and exec.  Physical
# inode/device/time fields are not compared with an old node, but every field
# must be identical across these two same-preflight snapshots.
ASSET_POST_SNAPSHOT=$(/usr/bin/env -i PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 \
  /tmp/round25-hypic-env/venv/bin/python "$U_ASSET_SNAPSHOT" --model-root "$MODEL_ROOT") \
  || die "post-authority asset snapshot failed"
[[ "$ASSET_PRE_SNAPSHOT" == "$ASSET_POST_SNAPSHOT" ]] || die "same-preflight model asset identity drift"

# No receipt directory, copied authority, pass marker, or U run path exists
# before the exact post-snapshot equality above.
/usr/bin/mkdir "$U_PREFLIGHT_DIR" || die "cannot create fresh U preflight observation directory"
/usr/bin/printf '%s\n' "$ASSET_POST_SNAPSHOT" > "$U_ASSET_OBSERVATION"
[[ -s "$U_ASSET_OBSERVATION" ]] || die "U asset observation publication failed"

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
  FREEZE_ROOT=/tmp/rwd5-hypic-store-freeze-u \
  CODE_DIR=/tmp/rwd5-hypic-store-freeze-u/code \
  FREEZE_MANIFEST=/tmp/rwd5-hypic-store-freeze-u/SHA256SUMS \
  EXPECTED_FREEZE_MANIFEST_SHA256="$EXPECTED_U_MANIFEST_SHA256" \
  LIVE_DEBUG_ROOT=/tmp/rwd5-hypic-store-freeze-u/live-debug-j-trial-1879097 \
  ALLOCATOR_DEBUG_ROOT=/tmp/rwd5-hypic-store-freeze-u/live-allocator-debug-d-trial-1879456 \
  ALLOCATOR_DEBUG_PROVENANCE=/tmp/rwd5-hypic-store-freeze-u/allocator-debug-d-provenance.json \
  ALLOCATOR_DEBUG_LAUNCH_PLAN=/tmp/rwd5-hypic-store-freeze-u/allocator-debug-d-launch-plan.json \
  ALLOCATOR_DEBUG_FREEZE_MANIFEST=/tmp/rwd5-hypic-store-freeze-u/allocator-debug-d-freeze-SHA256SUMS \
  MODEL_DIR=/tmp/Qwen3.5-35B-A3B-hypic-model-view \
  MODEL_WEIGHT_LEDGER=/tmp/Qwen3.5-35B-A3B-hypic-model-view/model-weights.sha256 \
  MODEL_ARTIFACT_LEDGER=/tmp/Qwen3.5-35B-A3B-hypic-model-view/model-artifacts.sha256 \
  VALIDATION_DATA=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/data/qcomem-longbench-validation/longbench_validation.jsonl \
  RUN_DIR=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/runs/qcomem/hypic-retained-state-rwd5-20260822u \
  INSTRUMENTED_REPO=/tmp/HYPIC-98147c0-rwd5-store-u \
  INHERITED_TEST=/tmp/rwd5-hypic-store-freeze-u/code/test_run_hypic_same_protocol.py \
  INHERITED_LAUNCHER=/tmp/rwd5-hypic-store-freeze-u/code/launch_hypic_same_protocol_8gpu.sh \
  SAFE_WRAPPER=/tmp/rwd5-hypic-store-freeze-u/code/launch_hypic_retained_state_bytes_safe_u.sh \
  SAFE_CWD_GUARD=/tmp/rwd5-hypic-store-freeze-u/code/rwd5_safe_cwd_guard.py \
  MODEL_ASSET_SNAPSHOT=/tmp/rwd5-hypic-store-freeze-u/code/rwd5_model_asset_snapshot.py \
  ASSET_OBSERVATION_PATH=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/runs/qcomem/hypic-retained-state-rwd5-preflight-u-20260822/model-asset-observation.json \
  INVALID_T_RECEIPT=/tmp/rwd5-hypic-store-freeze-u/invalid-formal-t-job247574-trial1879456.json \
  /bin/bash --noprofile --norc /tmp/rwd5-hypic-store-freeze-u/code/launch_hypic_retained_state_bytes_8gpu.sh
# PINNED_EXEC_ENV_END
