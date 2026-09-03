#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

PACKAGE_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
PACKAGE_MANIFEST=${PACKAGE_ROOT}/PACKAGE-SHA256SUMS
EXPECTED_PACKAGE_MANIFEST_SHA256=${EXPECTED_BOOTSTRAP_PACKAGE_SHA256:?missing EXPECTED_BOOTSTRAP_PACKAGE_SHA256}
JOB_ID=${QS_JOB_ID:?missing QS_JOB_ID}
TRIAL_ID=${QS_TRIAL_ID:?missing QS_TRIAL_ID}
SCOPE=${QCOMEM_DEBUG_SCOPE:?missing QCOMEM_DEBUG_SCOPE}
[[ "$JOB_ID" =~ ^[0-9]+$ && "$TRIAL_ID" =~ ^[0-9]+$ ]] || { echo "ERROR: nonnumeric QS identity" >&2; exit 1; }
[[ "$SCOPE" == ROUND27_HYPIC_STORE_FORMAL_W ]] || { echo "ERROR: scope drift" >&2; exit 1; }
[[ $(id -u) -eq 0 && $(id -g) -eq 0 ]] || { echo "ERROR: bootstrap must run as root" >&2; exit 1; }

ASSET_BASE=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets
TRANSPORT=${ASSET_BASE}/transport/hypic-retained-state-rwd5-trial1880346
TRANSPORT_MANIFEST=${TRANSPORT}/TRANSPORT-SHA256SUMS
EXPECTED_TRANSPORT_MANIFEST_SHA256=7b8a8585684487ac6d44d900653607d90c6289982a9793b2923fc95ae6ff8233
EXPECTED_RUNTIME_MANIFEST_SHA256=7722e39036b5a71a03848f42bd16db793b78d3db92ae0a13db8665ac81d71d92
RUNTIME_ROOT=/tmp/rwd5-hypic-store-runtime-${TRIAL_ID}
OFFICIAL_REPO=/tmp/HYPIC-98147c0
INSTALL_SOURCE=/tmp/HYPIC-98147c0-install-source-${TRIAL_ID}
VENV_ROOT=/tmp/round25-hypic-env
WHEEL_ROOT=/tmp/rwd5-cu129-wheels-${TRIAL_ID}
MODEL_SOURCE=${ASSET_BASE}/models/Qwen3.5-35B-A3B-59d61f3
MODEL_VIEW=/tmp/Qwen3.5-35B-A3B-hypic-model-view
DATA=${ASSET_BASE}/data/qcomem-longbench-validation/longbench_validation.jsonl
BOOT_ROOT=${ASSET_BASE}/runs/qcomem/hypic-rwd5-autobootstrap-job${JOB_ID}-trial${TRIAL_ID}
FORMAL_RUN=${BOOT_ROOT}/formal-run
PLATFORM_RECEIPT=${BOOT_ROOT}/dynamic-platform-command-authority.json
ASSET_OBSERVATION=${BOOT_ROOT}/model-asset-observation.json
EXPECTED_COMMIT=98147c01909004e66d98bcb18b886927d41b0ee5
EXPECTED_DATA_SHA=1553739496b3c209988de56a4ccf574d407379b6b7937ddfafacbe66925069fe
EXPECTED_WEIGHT_LEDGER_SHA=8314a82c9188b9b817193e039b0b0eb0636b328512f19b0c12455853b7e20014
EXPECTED_ARTIFACT_LEDGER_SHA=d78424684a27718b44616c80c788e7f8b726feed674fe015901b87de1993f7dd

die() { printf '%s\n' "ERROR: $*" >&2; exit 1; }
sha() { /usr/bin/sha256sum "$1" | /usr/bin/awk '{print $1}'; }

[[ ! -e "$BOOT_ROOT" ]] || die "bootstrap output already exists: $BOOT_ROOT"
mkdir -p "$BOOT_ROOT"
exec > >(tee -a "$BOOT_ROOT/bootstrap.log") 2>&1
printf '%s\n' "$(date -u +%FT%TZ) bootstrap start job=$JOB_ID trial=$TRIAL_ID pid=$$"

BOOT_CHILD=''
on_exit() {
  local rc=$?
  trap - EXIT INT TERM
  if [[ -n "$BOOT_CHILD" ]] && kill -0 "$BOOT_CHILD" 2>/dev/null; then
    kill -TERM "$BOOT_CHILD" 2>/dev/null || true
    for _ in $(seq 1 40); do kill -0 "$BOOT_CHILD" 2>/dev/null || break; sleep 0.25; done
    kill -KILL "$BOOT_CHILD" 2>/dev/null || true
    wait "$BOOT_CHILD" 2>/dev/null || true
  fi
  if [[ $rc -eq 0 && -f "$FORMAL_RUN/COMPLETED" && ! -e "$FORMAL_RUN/FAILED" ]]; then
    printf '%s\n' "$(date -u +%FT%TZ)" > "$BOOT_ROOT/BOOTSTRAP_COMPLETED"
  else
    printf '{"exit_code":%d,"timestamp_utc":"%s"}\n' "$rc" "$(date -u +%FT%TZ)" > "$BOOT_ROOT/FAILED"
  fi
  exit "$rc"
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

[[ -f "$PACKAGE_MANIFEST" ]] || die "package manifest missing"
[[ "$(sha "$PACKAGE_MANIFEST")" == "$EXPECTED_PACKAGE_MANIFEST_SHA256" ]] || die "package manifest identity drift"
(cd "$PACKAGE_ROOT" && /usr/bin/sha256sum -c PACKAGE-SHA256SUMS) || die "package member drift"
[[ -f "$TRANSPORT_MANIFEST" ]] || die "transport manifest missing"
[[ "$(sha "$TRANSPORT_MANIFEST")" == "$EXPECTED_TRANSPORT_MANIFEST_SHA256" ]] || die "transport manifest identity drift"
(cd "$TRANSPORT" && /usr/bin/sha256sum -c TRANSPORT-SHA256SUMS) || die "transport member drift"

mapfile -t GPU_ROWS < <(nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used --format=csv,noheader,nounits)
[[ ${#GPU_ROWS[@]} -eq 8 ]] || die "expected 8 GPUs"
printf '%s\n' "${GPU_ROWS[@]}" | awk -F, '
  {gsub(/^ +| +$/, "", $1); gsub(/^ +| +$/, "", $2); gsub(/^ +| +$/, "", $3); gsub(/^ +| +$/, "", $4); gsub(/^ +| +$/, "", $5)}
  $1 != NR-1 || $2 !~ /^GPU-/ || $3 != "NVIDIA H20-3e" || $4 != 143771 || $5 != 0 {exit 1}
' || die "8xH20 idle inventory drift"
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sed '/^$/d')" ]] || die "GPU compute process active before prep"

for fresh in "$RUNTIME_ROOT" "$OFFICIAL_REPO" "$INSTALL_SOURCE" "$VENV_ROOT" "$WHEEL_ROOT" "$MODEL_VIEW" "$FORMAL_RUN" "/tmp/HYPIC-98147c0-rwd5-store-${TRIAL_ID}"; do
  [[ ! -e "$fresh" ]] || die "fresh path already exists: $fresh"
done

cp -a "$PACKAGE_ROOT/runtime-overlay" "$RUNTIME_ROOT"
[[ "$(sha "$RUNTIME_ROOT/RUNTIME-SHA256SUMS")" == "$EXPECTED_RUNTIME_MANIFEST_SHA256" ]] || die "runtime manifest drift after stage"
(cd "$RUNTIME_ROOT" && /usr/bin/sha256sum -c RUNTIME-SHA256SUMS) || die "runtime overlay drift after stage"

[[ "$(sha "$TRANSPORT/HYPIC-98147c0.tgz")" == 6f28b699c91259629897d5810cc992b809f4f85f47bd4e9cc7c0c5033422e4cd ]] || die "official transport SHA"
OFFICIAL_TMP=/tmp/HYPIC-98147c0-extract-${TRIAL_ID}
[[ ! -e "$OFFICIAL_TMP" ]] || die "official extract path exists"
mkdir -p "$OFFICIAL_TMP"
tar --warning=no-unknown-keyword -xzf "$TRANSPORT/HYPIC-98147c0.tgz" -C "$OFFICIAL_TMP"
mapfile -t OFFICIAL_TOP < <(find "$OFFICIAL_TMP" -mindepth 1 -maxdepth 1 -print)
[[ ${#OFFICIAL_TOP[@]} -eq 1 && -d "${OFFICIAL_TOP[0]}/.git" ]] || die "official tar topology"
mv "${OFFICIAL_TOP[0]}" "$OFFICIAL_REPO"
rmdir "$OFFICIAL_TMP"
[[ "$(git -C "$OFFICIAL_REPO" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || die "official commit drift"
[[ -z "$(git -C "$OFFICIAL_REPO" status --porcelain --untracked-files=all)" ]] || die "official worktree dirty"

[[ "$(sha "$DATA")" == "$EXPECTED_DATA_SHA" ]] || die "validation data SHA"
[[ -d "$MODEL_SOURCE" && ! -L "$MODEL_SOURCE" ]] || die "model source unavailable"
[[ "$(sha "$MODEL_SOURCE/model-weights.sha256")" == "$EXPECTED_WEIGHT_LEDGER_SHA" ]] || die "model weight ledger SHA"
[[ $(wc -l < "$MODEL_SOURCE/model-weights.sha256") -eq 14 ]] || die "model weight ledger count"
(cd "$MODEL_SOURCE" && /usr/bin/sha256sum -c model-weights.sha256) || die "model weight payload drift"
mkdir "$MODEL_VIEW"
while IFS= read -r -d '' item; do
  name=${item##*/}
  case "$name" in
    model-artifacts.sha256|model-weights.sha256|preprocessor_config.json|video_preprocessor_config.json) continue ;;
  esac
  ln -s "$item" "$MODEL_VIEW/$name"
done < <(find "$MODEL_SOURCE" -mindepth 1 -maxdepth 1 -print0)
cp "$MODEL_SOURCE/model-weights.sha256" "$MODEL_VIEW/model-weights.sha256"
cp "$MODEL_SOURCE/model-artifacts.sha256" "$MODEL_VIEW/model-artifacts.sha256"
cp "$TRANSPORT/preprocessor_config.json" "$MODEL_VIEW/preprocessor_config.json"
cp "$TRANSPORT/video_preprocessor_config.json" "$MODEL_VIEW/video_preprocessor_config.json"
if [[ "$(sha "$MODEL_VIEW/model-artifacts.sha256")" != "$EXPECTED_ARTIFACT_LEDGER_SHA" ]]; then
  printf '%s  %s\n' 27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516 preprocessor_config.json >> "$MODEL_VIEW/model-artifacts.sha256"
  printf '%s  %s\n' 7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13 video_preprocessor_config.json >> "$MODEL_VIEW/model-artifacts.sha256"
fi
chmod 0444 "$MODEL_VIEW/model-weights.sha256" "$MODEL_VIEW/model-artifacts.sha256" "$MODEL_VIEW/preprocessor_config.json" "$MODEL_VIEW/video_preprocessor_config.json"
chown 0:0 "$MODEL_VIEW/model-weights.sha256" "$MODEL_VIEW/model-artifacts.sha256" "$MODEL_VIEW/preprocessor_config.json" "$MODEL_VIEW/video_preprocessor_config.json"
[[ "$(sha "$MODEL_VIEW/model-artifacts.sha256")" == "$EXPECTED_ARTIFACT_LEDGER_SHA" ]] || die "final model artifact ledger SHA"
[[ $(wc -l < "$MODEL_VIEW/model-artifacts.sha256") -eq 9 ]] || die "model artifact ledger count"
(cd "$MODEL_VIEW" && /usr/bin/sha256sum -c model-weights.sha256 && /usr/bin/sha256sum -c model-artifacts.sha256) || die "model-view payload drift"
[[ -z "$(find "$MODEL_VIEW" -maxdepth 1 -type f -perm /222 -print -quit)" ]] || die "writable top-level model-view authority"

[[ "$(sha "$TRANSPORT/rwd5-cu129-wheels.tgz")" == 1e74541a6e71610d598d90cbb7bd5df33cf9c5148f14621ba34dbe1f9c840bee ]] || die "wheel bundle SHA"
[[ "$(sha "$TRANSPORT/sglang_kernel-0.4.4+cu129-cp310-abi3-manylinux2014_x86_64.whl")" == 4ea2f7176965c3d1a5697254befe45c2ac995793577108129710d0f66ca85b33 ]] || die "full kernel wheel SHA"
mkdir "$WHEEL_ROOT"
tar --warning=no-unknown-keyword -xzf "$TRANSPORT/rwd5-cu129-wheels.tgz" -C "$WHEEL_ROOT"
mapfile -t DEEP_WHEELS < <(find "$WHEEL_ROOT" -type f -name 'sgl_deep_gemm-0.1.3+cu129-*.whl')
[[ ${#DEEP_WHEELS[@]} -eq 1 ]] || die "deep-gemm wheel topology"

python3 -m venv "$VENV_ROOT/venv"
PY=$VENV_ROOT/venv/bin/python
PIP=$VENV_ROOT/venv/bin/pip
"$PIP" --disable-pip-version-check install --upgrade pip setuptools wheel
"$PIP" --disable-pip-version-check install --index-url https://download.pytorch.org/whl/cu129 'torch==2.11.0'
"$PIP" --disable-pip-version-check install --force-reinstall --no-deps \
  "$TRANSPORT/sglang_kernel-0.4.4+cu129-cp310-abi3-manylinux2014_x86_64.whl" "${DEEP_WHEELS[0]}"
cp -a "$OFFICIAL_REPO" "$INSTALL_SOURCE"
"$PIP" --disable-pip-version-check install "$INSTALL_SOURCE/python" \
  'transformers==5.8.1' 'flashinfer-python==0.5.3' 'flashinfer-cubin==0.5.3'
"$PIP" --disable-pip-version-check install --force-reinstall --no-deps \
  "$TRANSPORT/sglang_kernel-0.4.4+cu129-cp310-abi3-manylinux2014_x86_64.whl" "${DEEP_WHEELS[0]}"
"$PIP" check
"$PY" - <<'PY'
import importlib.metadata as m
import importlib.util
import torch
expected={"torch":"2.11.0","transformers":"5.8.1","sglang":"0.5.14","sglang-kernel":"0.4.4","sgl-deep-gemm":"0.1.3","flashinfer-python":"0.5.3","flashinfer-cubin":"0.5.3"}
for name, base in expected.items():
    dist=m.distribution(name)
    assert dist.version.split("+")[0] == base, (name, dist.version)
    records=[p for p in (dist.files or []) if str(p).endswith(".dist-info/RECORD")]
    assert len(records)==1 and dist.locate_file(records[0]).is_file(), name
assert torch.__version__ == "2.11.0+cu129", torch.__version__
assert torch.version.cuda == "12.9", torch.version.cuda
assert torch.cuda.device_count() == 8
assert importlib.util.find_spec("flashinfer") is not None
import sgl_kernel
PY
[[ "$(git -C "$OFFICIAL_REPO" rev-parse HEAD)" == "$EXPECTED_COMMIT" && -z "$(git -C "$OFFICIAL_REPO" status --porcelain --untracked-files=all)" ]] || die "official worktree changed during env prep"

"$PY" "$RUNTIME_ROOT/code/rwd5_model_asset_snapshot.py" --model-root "$MODEL_VIEW" > "$ASSET_OBSERVATION.tmp"
mv "$ASSET_OBSERVATION.tmp" "$ASSET_OBSERVATION"

AUTH_PID=$$
AUTH_ENV=/proc/${AUTH_PID}/environ
[[ -r "$AUTH_ENV" && -r "/proc/${AUTH_PID}/cmdline" ]] || die "bootstrap process authority unavailable"
AUTH_PID="$AUTH_PID" AUTH_ENV="$AUTH_ENV" JOB_ID="$JOB_ID" TRIAL_ID="$TRIAL_ID" SCOPE="$SCOPE" PLATFORM_RECEIPT="$PLATFORM_RECEIPT" "$PY" - <<'PY'
import hashlib,json,os,subprocess
from pathlib import Path
pid=int(os.environ["AUTH_PID"]); env_path=Path(os.environ["AUTH_ENV"])
raw=env_path.read_bytes(); relevant={"QS_JOB_ID":[],"QS_TRIAL_ID":[],"QCOMEM_DEBUG_SCOPE":[]}
for item in raw.split(b"\0"):
    if not item: continue
    k,v=item.split(b"=",1); key=k.decode()
    if key in relevant: relevant[key].append(v.decode())
expected={"QS_JOB_ID":os.environ["JOB_ID"],"QS_TRIAL_ID":os.environ["TRIAL_ID"],"QCOMEM_DEBUG_SCOPE":os.environ["SCOPE"]}
assert all(relevant[k]==[v] for k,v in expected.items()), (relevant,expected)
cmd=[x.decode() for x in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0") if x]
rows=[]
for line in subprocess.check_output(["nvidia-smi","--query-gpu=uuid,name,memory.total","--format=csv,noheader,nounits"],text=True).splitlines():
    uuid,name,memory=[x.strip() for x in line.split(",",2)]; rows.append((uuid,name,int(memory)))
assert len(rows)==8 and len({x[0] for x in rows})==8 and all(x[1:]==("NVIDIA H20-3e",143771) for x in rows)
receipt={"schema":"hypic-rwd5-dynamic-platform-command-authority-v1","platform_job_id":int(expected["QS_JOB_ID"]),"platform_trial_id":int(expected["QS_TRIAL_ID"]),"scope":expected["QCOMEM_DEBUG_SCOPE"],"status_at_bootstrap":"Running","platform_command_pid":pid,"platform_command":cmd,"platform_command_environ_sha256":hashlib.sha256(raw).hexdigest(),"runtime_platform_command_environ_required":expected,"gpu_count":8,"gpu_name":"NVIDIA H20-3e","gpu_memory_mib":143771,"gpu_uuids":[x[0] for x in rows],"paper_evidence":False,"formal_cells_before_bootstrap":0}
Path(os.environ["PLATFORM_RECEIPT"]+".tmp").write_bytes((json.dumps(receipt,sort_keys=True,separators=(",",":"))+"\n").encode())
os.replace(os.environ["PLATFORM_RECEIPT"]+".tmp",os.environ["PLATFORM_RECEIPT"])
PY

EXPECTED_RUNTIME_MANIFEST_SHA256="$EXPECTED_RUNTIME_MANIFEST_SHA256" \
RWD5_ASSET_OBSERVATION="$ASSET_OBSERVATION" RWD5_FORMAL_RUN_DIR="$FORMAL_RUN" \
RWD5_DYNAMIC_PLATFORM_RECEIPT="$PLATFORM_RECEIPT" RWD5_PLATFORM_COMMAND_PID="$AUTH_PID" \
RWD5_PLATFORM_COMMAND_ENV_PATH="$AUTH_ENV" \
/bin/bash --noprofile --norc "$RUNTIME_ROOT/code/launch_hypic_retained_state_bytes_runtime_dynamic.sh" &
BOOT_CHILD=$!
wait "$BOOT_CHILD"
BOOT_CHILD=''
[[ -f "$FORMAL_RUN/COMPLETED" && ! -e "$FORMAL_RUN/FAILED" ]] || die "formal terminal closure failed"
