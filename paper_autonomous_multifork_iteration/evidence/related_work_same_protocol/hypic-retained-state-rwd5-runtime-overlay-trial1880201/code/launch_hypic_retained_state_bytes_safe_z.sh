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
  INVALID_T_RECEIPT PLATFORM_AUTHORITY_RECEIPT RWD5_PID1_ENV_PATH \
  RETIRED_W_TRIAL_RECEIPT \
  RWD5_PLATFORM_JOB_ID RWD5_PLATFORM_TRIAL_ID \
  GIT_CONFIG_GLOBAL GIT_CONFIG_SYSTEM || true

Z_ROOT=/tmp/rwd5-hypic-store-freeze-z
Z_MANIFEST=${Z_ROOT}/SHA256SUMS
EXPECTED_Z_MANIFEST_SHA256=${EXPECTED_Z_MANIFEST_SHA256:?supply externally audited Z SHA256SUMS SHA256}
Z_STOP=${Z_ROOT}/STOP
EXPECTED_Z_STOP_SHA256=${EXPECTED_Z_STOP_SHA256:?supply externally audited Z STOP SHA256}
MODEL_ROOT=/tmp/Qwen3.5-35B-A3B-hypic-model-view
RUN_DIR_Z=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/runs/qcomem/hypic-retained-state-rwd5-20260822z
INSTRUMENTED_REPO_Z=/tmp/HYPIC-98147c0-rwd5-store-z
Z_LAUNCHER=${Z_ROOT}/code/launch_hypic_retained_state_bytes_8gpu.sh
Z_WRAPPER=${Z_ROOT}/code/launch_hypic_retained_state_bytes_safe_z.sh
Z_GUARD=${Z_ROOT}/code/rwd5_safe_cwd_guard.py
Z_ASSET_SNAPSHOT=${Z_ROOT}/code/rwd5_model_asset_snapshot.py
Z_PREFLIGHT_DIR=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/runs/qcomem/hypic-retained-state-rwd5-preflight-z-20260822
Z_ASSET_OBSERVATION=${Z_PREFLIGHT_DIR}/model-asset-observation.json
Z_PLATFORM_AUTHORITY=${Z_ROOT}/platform-execution-authority-z.json
Z_PID1_ENV_PATH=/proc/1/environ
Z_EXPECTED_MANIFEST_MEMBERS=73

die() { printf '%s\n' "ERROR: $*" >&2; exit 1; }

z_open_trusted_authority() {
  local sentinel manifest_capture stop_capture actual
  exec 10< "$Z_MANIFEST"
  exec 11< "$Z_STOP"
  sentinel='__RWD5_Z_EXACT_EOF_SENTINEL__'
  manifest_capture=$({ /usr/bin/cat /proc/self/fd/10; /usr/bin/printf '%s' "$sentinel"; })
  stop_capture=$({ /usr/bin/cat /proc/self/fd/11; /usr/bin/printf '%s' "$sentinel"; })
  [[ "$manifest_capture" == *"$sentinel" && "$stop_capture" == *"$sentinel" ]] \
    || die "Z authority capture failed"
  Z_TRUSTED_MANIFEST_BYTES=${manifest_capture%"$sentinel"}
  Z_TRUSTED_STOP_BYTES=${stop_capture%"$sentinel"}
  exec 10<&-
  exec 11<&-
  actual=$(/usr/bin/printf '%s' "$Z_TRUSTED_MANIFEST_BYTES" | /usr/bin/sha256sum | /usr/bin/awk '{print $1}')
  [[ "$actual" == "$EXPECTED_Z_MANIFEST_SHA256" ]] \
    || die "Z manifest identity drift"
  actual=$(/usr/bin/printf '%s' "$Z_TRUSTED_STOP_BYTES" | /usr/bin/sha256sum | /usr/bin/awk '{print $1}')
  [[ "$actual" == "$EXPECTED_Z_STOP_SHA256" ]] \
    || die "Z STOP identity drift"
  [[ "$Z_TRUSTED_MANIFEST_BYTES" == *$'\n' ]] || die "Z manifest lacks canonical terminal newline"
  /usr/bin/printf '%s' "$Z_TRUSTED_MANIFEST_BYTES" | /usr/bin/awk -v expected="$Z_EXPECTED_MANIFEST_MEMBERS" '
    BEGIN { n=0 }
    {
      if (NF != 2 || length($1) != 64 || $1 !~ /^[0-9a-f]+$/ || $2 !~ /^\.\//) exit 71
      path=substr($2,3)
      if (path == "" || path ~ /^\// || path ~ /\/\//) exit 72
      count=split(path, component, "/")
      for (i=1; i<=count; i++) if (component[i] == "" || component[i] == "." || component[i] == "..") exit 73
      if (seen[path]++) exit 74
      n++
    }
    END { if (n != expected) exit 75 }
  ' || die "Z manifest canonical member set is not exact"
  (cd "$Z_ROOT" && /usr/bin/printf '%s' "$Z_TRUSTED_MANIFEST_BYTES" | /usr/bin/sha256sum -c -) \
    || die "Z frozen files drift"
}

z_close_trusted_authority() {
  unset Z_TRUSTED_MANIFEST_BYTES Z_TRUSTED_STOP_BYTES
}

z_manifest_member_sha() {
  local relative=$1 row count
  row=$(/usr/bin/printf '%s' "$Z_TRUSTED_MANIFEST_BYTES" | /usr/bin/awk -v target="./${relative}" 'NF==2 && $2==target {print $1}')
  count=$(/usr/bin/printf '%s' "$Z_TRUSTED_MANIFEST_BYTES" | /usr/bin/awk -v target="./${relative}" 'NF==2 && $2==target {n++} END {print n+0}')
  [[ "$count" -eq 1 && "$row" =~ ^[0-9a-f]{64}$ ]] || die "Z manifest member is not exact: $relative"
  /usr/bin/printf '%s\n' "$row"
}

z_verify_open_fd() {
  local fd_path=$1 relative=$2 expected actual
  expected=$(z_manifest_member_sha "$relative")
  actual=$(/usr/bin/sha256sum "$fd_path" | /usr/bin/awk '{print $1}')
  [[ "$actual" == "$expected" ]] || die "opened frozen member drift: $relative"
}

z_require_pid1_env() {
  local key=$1 expected=$2 count value
  count=$(/usr/bin/tr '\0' '\n' < "$Z_PID1_ENV_PATH" | /usr/bin/awk -F= -v key="$key" '$1==key {n++} END {print n+0}')
  value=$(/usr/bin/tr '\0' '\n' < "$Z_PID1_ENV_PATH" | /usr/bin/awk -F= -v key="$key" '$1==key {print substr($0, length(key)+2)}')
  [[ "$count" -eq 1 && "$value" == "$expected" ]] || die "platform-owned pid1 environment mismatch: $key"
}

for path in "$Z_MANIFEST" "$Z_STOP" "$Z_LAUNCHER" "$Z_WRAPPER" "$Z_GUARD" "$Z_ASSET_SNAPSHOT" "$Z_PLATFORM_AUTHORITY" "$MODEL_ROOT"; do
  [[ -e "$path" ]] || die "missing frozen path: $path"
done
[[ -r "$Z_PID1_ENV_PATH" ]] || die "platform-owned pid1 environment is unreadable"
z_require_pid1_env QS_JOB_ID 247699
z_require_pid1_env QS_TRIAL_ID 1880085
z_require_pid1_env QCOMEM_DEBUG_SCOPE ROUND27_HYPIC_STORE_FORMAL_W
[[ ! -e "$RUN_DIR_Z" ]] || die "fresh Z RUN_DIR already exists"
[[ ! -e "$INSTRUMENTED_REPO_Z" ]] || die "fresh Z instrumented repo already exists"
[[ ! -e "$Z_PREFLIGHT_DIR" ]] || die "fresh Z preflight observation path already exists"

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
z_open_trusted_authority
exec 7< "$Z_ASSET_SNAPSHOT"
z_verify_open_fd /proc/self/fd/7 code/rwd5_model_asset_snapshot.py
z_close_trusted_authority
ASSET_PRE_SNAPSHOT=$(/usr/bin/env -i PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 \
  /tmp/round25-hypic-env/venv/bin/python /proc/self/fd/7 --model-root "$MODEL_ROOT") \
  || die "pre-authority asset snapshot failed"
exec 7<&-

/usr/bin/env -i PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 \
  /tmp/round25-hypic-env/venv/bin/python "$Z_GUARD" /
cd /
[[ "$PWD" == "/" ]] || die "failed to enter fixed safe cwd"

/usr/bin/env -i PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 \
  /tmp/round25-hypic-env/venv/bin/python -c \
  'import os,sys; assert os.getcwd() == "/"; assert "" not in sys.path; assert "/" not in sys.path' \
  || die "frozen Python does not enforce safe sys.path from fixed cwd"

/usr/bin/env -i PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONPATH=/tmp/rwd5-hypic-store-freeze-z/code \
  /tmp/round25-hypic-env/venv/bin/python -c \
  'import importlib.util,pathlib; root=pathlib.Path("/tmp/rwd5-hypic-store-freeze-z/code").resolve(strict=True); expected={"test_hypic_retained_state_receipt":root/"test_hypic_retained_state_receipt.py","test_run_hypic_same_protocol":root/"test_run_hypic_same_protocol.py"}; [(lambda s,p: (_ for _ in ()).throw(AssertionError()) if not (s and s.origin and pathlib.Path(s.origin).resolve(strict=True)==p.resolve(strict=True)) else None)(importlib.util.find_spec(n),p) for n,p in expected.items()]' \
  || die "frozen unittest import authority is not exact Z code"

/usr/bin/env -i PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 \
  PYTHONPATH=/tmp/HYPIC-98147c0/python:/tmp/rwd5-hypic-store-freeze-z/code \
  /tmp/round25-hypic-env/venv/bin/python -c \
  'import importlib.util,pathlib; s=importlib.util.find_spec("sglang"); assert s and s.origin and s.submodule_search_locations; root=pathlib.Path("/tmp/HYPIC-98147c0/python/sglang").resolve(strict=True); assert pathlib.Path(s.origin).resolve(strict=True).is_relative_to(root); assert {pathlib.Path(p).resolve(strict=True) for p in s.submodule_search_locations} == {root}' \
  || die "SGLang import authority is not exact official repository"

# Rebind the externally trusted manifest, STOP, and every frozen row after all
# imports.  Then open the exact manifest-bound helper by FD so replacing its
# pathname cannot change the program that produces POST.
z_open_trusted_authority
exec 8< "$Z_ASSET_SNAPSHOT"
z_verify_open_fd /proc/self/fd/8 code/rwd5_model_asset_snapshot.py
z_close_trusted_authority
ASSET_POST_SNAPSHOT=$(/usr/bin/env -i PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 \
  /tmp/round25-hypic-env/venv/bin/python /proc/self/fd/8 --model-root "$MODEL_ROOT") \
  || die "post-authority asset snapshot failed"
exec 8<&-
[[ "$ASSET_PRE_SNAPSHOT" == "$ASSET_POST_SNAPSHOT" ]] || die "same-preflight model asset identity drift"

# Trusted system sha256sum, not a bundle helper, rechecks the external
# manifest/STOP roots and all frozen authority bytes after POST.  FD 9 binds
# the exact internal launcher inode through exec, closing path replacement.
z_open_trusted_authority
exec 9< "$Z_LAUNCHER"
z_verify_open_fd /proc/self/fd/9 code/launch_hypic_retained_state_bytes_8gpu.sh
z_close_trusted_authority

# No receipt directory, copied authority, pass marker, or Z run path exists
# before all checks above.
/usr/bin/mkdir "$Z_PREFLIGHT_DIR" || die "cannot create fresh Z preflight observation directory"
/usr/bin/printf '%s\n' "$ASSET_POST_SNAPSHOT" > "$Z_ASSET_OBSERVATION"
[[ -s "$Z_ASSET_OBSERVATION" ]] || die "Z asset observation publication failed"

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
  FREEZE_ROOT=/tmp/rwd5-hypic-store-freeze-z \
  CODE_DIR=/tmp/rwd5-hypic-store-freeze-z/code \
  FREEZE_MANIFEST=/tmp/rwd5-hypic-store-freeze-z/SHA256SUMS \
  EXPECTED_FREEZE_MANIFEST_SHA256="$EXPECTED_Z_MANIFEST_SHA256" \
  LIVE_DEBUG_ROOT=/tmp/rwd5-hypic-store-freeze-z/live-debug-j-trial-1879097 \
  ALLOCATOR_DEBUG_ROOT=/tmp/rwd5-hypic-store-freeze-z/live-allocator-debug-d-trial-1879456 \
  ALLOCATOR_DEBUG_PROVENANCE=/tmp/rwd5-hypic-store-freeze-z/allocator-debug-d-provenance.json \
  ALLOCATOR_DEBUG_LAUNCH_PLAN=/tmp/rwd5-hypic-store-freeze-z/allocator-debug-d-launch-plan.json \
  ALLOCATOR_DEBUG_FREEZE_MANIFEST=/tmp/rwd5-hypic-store-freeze-z/allocator-debug-d-freeze-SHA256SUMS \
  MODEL_DIR=/tmp/Qwen3.5-35B-A3B-hypic-model-view \
  MODEL_WEIGHT_LEDGER=/tmp/Qwen3.5-35B-A3B-hypic-model-view/model-weights.sha256 \
  MODEL_ARTIFACT_LEDGER=/tmp/Qwen3.5-35B-A3B-hypic-model-view/model-artifacts.sha256 \
  VALIDATION_DATA=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/data/qcomem-longbench-validation/longbench_validation.jsonl \
  RUN_DIR=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/runs/qcomem/hypic-retained-state-rwd5-20260822z \
  INSTRUMENTED_REPO=/tmp/HYPIC-98147c0-rwd5-store-z \
  INHERITED_TEST=/tmp/rwd5-hypic-store-freeze-z/code/test_run_hypic_same_protocol.py \
  INHERITED_LAUNCHER=/tmp/rwd5-hypic-store-freeze-z/code/launch_hypic_same_protocol_8gpu.sh \
  SAFE_WRAPPER=/tmp/rwd5-hypic-store-freeze-z/code/launch_hypic_retained_state_bytes_safe_z.sh \
  SAFE_CWD_GUARD=/tmp/rwd5-hypic-store-freeze-z/code/rwd5_safe_cwd_guard.py \
  MODEL_ASSET_SNAPSHOT=/tmp/rwd5-hypic-store-freeze-z/code/rwd5_model_asset_snapshot.py \
  ASSET_OBSERVATION_PATH=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/runs/qcomem/hypic-retained-state-rwd5-preflight-z-20260822/model-asset-observation.json \
  INVALID_T_RECEIPT=/tmp/rwd5-hypic-store-freeze-z/invalid-formal-t-job247574-trial1879456.json \
  PLATFORM_AUTHORITY_RECEIPT=/tmp/rwd5-hypic-store-freeze-z/platform-execution-authority-z.json \
  RETIRED_W_TRIAL_RECEIPT=/tmp/rwd5-hypic-store-freeze-z/external-manual-stop-trial1879843.json \
  RWD5_PID1_ENV_PATH=/proc/1/environ \
  RWD5_PLATFORM_JOB_ID=247699 \
  RWD5_PLATFORM_TRIAL_ID=1880085 \
  /bin/bash --noprofile --norc /proc/self/fd/9
# PINNED_EXEC_ENV_END
