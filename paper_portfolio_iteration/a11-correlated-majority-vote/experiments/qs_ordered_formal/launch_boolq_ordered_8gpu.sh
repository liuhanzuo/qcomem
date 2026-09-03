#!/usr/bin/env bash
set -euo pipefail

# Frozen A11 formal launcher.  It performs no exploratory/smoke experiment.
# Any failure before the runner's COMPLETED marker is infrastructure/preflight,
# CAL-screen rejection, or integrity failure rather than TEST evidence.

CODE_DIR=${CODE_DIR:?set frozen CODE_DIR}
RUN_DIR=${RUN_DIR:?set fresh formal RUN_DIR}
DATASET=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench/a11_boolq_ordered_formal_20260822a_actual/data/boolq_35b264d_all_12697.jsonl
MODEL_DIR=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench/a11_boolq_ordered_formal_20260822a_actual/model/Qwen2.5-7B-Instruct-a09a354
MODEL_REVISION=a09a35458c702b33eeacc393d103063234e8bc28
SERVED_MODEL=a11-qwen25-7b
PROTOCOL="$CODE_DIR/protocol.json"
RUNNER="$CODE_DIR/run_boolq_ordered.py"

EXPECTED_PROTOCOL_SHA256=1a6593a97b67509d0afc954bef653924f7f9cdc055592d0ba75b08987b71cb31
EXPECTED_RUNNER_SHA256=f36e9ee066f7c37d455490b74f78056cb987ed382f5d8e7e35c0ecbb00f75a29
EXPECTED_DATASET_SHA256=13c2f4143ae320a0191c6de5be919248a20c15515f58c6deb7d3732068f2d31a
EXPECTED_DATASET_BYTES=8835655
EXPECTED_IMAGE=artifactory.devops.xiaohongshu.com/media/verlai/verl:vllm017.latest

preflight_fail() {
  echo "failure_class=infrastructure_preflight message=$*" >&2
  exit 2
}
preflight_error_trap() {
  local status=$?
  trap - ERR
  echo "failure_class=infrastructure_preflight launcher_status=$status" >&2
  exit "$status"
}
trap preflight_error_trap ERR

[[ ${QS_IMAGE:-} == "$EXPECTED_IMAGE" ]] || preflight_fail "QS_IMAGE mismatch: ${QS_IMAGE:-unset}"
[[ ! -e "$RUN_DIR" ]] || preflight_fail "formal RUN_DIR already exists: $RUN_DIR"
[[ -f "$PROTOCOL" && -f "$RUNNER" && -f "$DATASET" ]] || preflight_fail "missing frozen input"
[[ $(sha256sum "$PROTOCOL" | awk '{print $1}') == "$EXPECTED_PROTOCOL_SHA256" ]] || preflight_fail "protocol SHA mismatch"
[[ $(sha256sum "$RUNNER" | awk '{print $1}') == "$EXPECTED_RUNNER_SHA256" ]] || preflight_fail "runner SHA mismatch"
[[ $(sha256sum "$DATASET" | awk '{print $1}') == "$EXPECTED_DATASET_SHA256" ]] || preflight_fail "dataset SHA mismatch"
[[ $(stat -c '%s' "$DATASET") == "$EXPECTED_DATASET_BYTES" ]] || preflight_fail "dataset byte count mismatch"

mapfile -t GPU_UUIDS < <(nvidia-smi --query-gpu=uuid --format=csv,noheader | sed 's/[[:space:]]//g')
[[ ${#GPU_UUIDS[@]} -eq 8 ]] || preflight_fail "formal run requires exactly 8 visible GPUs"

MODEL_DIR="$MODEL_DIR" python3 - <<'PY'
import hashlib
import os
from pathlib import Path

root = Path(os.environ["MODEL_DIR"])
expected = {
    ".gitattributes": "11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361",
    "LICENSE": "832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e",
    "README.md": "f366f33bbf6bcadbb7d87f0a21a7b65584a56b8d58b0743c77c88bee625b93a6",
    "config.json": "7463bb0ea78315365e6c6b74de4e73bbcc8359dfb0c5a737584e077d42c0b03c",
    "generation_config.json": "3a8f9087e486054c8a4a08dae2e5a3ba62e23da212b5b8c08bc42cb983c3459f",
    "merges.txt": "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
    "model-00001-of-00004.safetensors": "a1333e6293854747c481288ea83b348226af178dd565c49b6f9495ba1966aba7",
    "model-00002-of-00004.safetensors": "f5d25a2772cb825164a2a2c0fb6d51a87e282abf21e4dd75bc5cfb3cd0ea6185",
    "model-00003-of-00004.safetensors": "8efdec4c1bc12317ae1a38dc42b595ce777738a64deea3fcb8a0a91381bcdfd5",
    "model-00004-of-00004.safetensors": "1a72d403cdf0c1ec3cb7f289f17b394a01e64394c2e9b3c0f94dbce3faf879bd",
    "model.safetensors.index.json": "624bf7c47cd12468fdc16e38a47cf4f19e0415b859a223ba3c027eed2f0e1028",
    "tokenizer.json": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
    "tokenizer_config.json": "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
}
expected_ledger_sha = "3ee6c9510b7e50bfcd46d6df33cafa3e2019f13a6a09bf1d2f9e80cdfe1164e8"
if not root.is_dir() or root.is_symlink():
    raise SystemExit("model snapshot root is not a regular directory")
actual = {}
for entry in root.iterdir():
    if entry.name == ".cache":
        if not entry.is_dir() or entry.is_symlink():
            raise SystemExit("model .cache entry is not a regular directory")
        continue
    if entry.is_symlink() or not entry.is_file():
        raise SystemExit("model snapshot contains non-regular entry: " + entry.name)
    actual[entry.name] = entry
if set(actual) != set(expected):
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    raise SystemExit(f"model snapshot file-set mismatch; missing={missing}; extra={extra}")
computed = {}
for name in sorted(actual):
    digest = hashlib.sha256()
    with actual[name].open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    computed[name] = digest.hexdigest()
    if computed[name] != expected[name]:
        raise SystemExit("model snapshot SHA mismatch: " + name)
ledger = "".join(f"{computed[name]}  {name}\n" for name in sorted(computed)).encode("utf-8")
if hashlib.sha256(ledger).hexdigest() != expected_ledger_sha:
    raise SystemExit("model snapshot ledger SHA mismatch")
print("model snapshot verified", expected_ledger_sha)
PY

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python3 - <<'PY'
import importlib.metadata
required = {"vllm": "0.17", "transformers": None, "torch": None, "huggingface-hub": None}
for package, prefix in required.items():
    version = importlib.metadata.version(package)
    print(package, version)
    if prefix is not None and not version.startswith(prefix):
        raise SystemExit(f"{package} version {version} does not start with frozen prefix {prefix}")
PY

HELP_TEXT=$(vllm serve --help=all 2>&1)
grep -q -- '--no-enable-prefix-caching' <<<"$HELP_TEXT" || preflight_fail "vLLM cannot explicitly disable prefix caching"

LAUNCH_TMP=$(mktemp -d /tmp/a11-boolq-launch.XXXXXX)
SERVER_LOG_DIR="$LAUNCH_TMP/server_logs"
mkdir -p "$SERVER_LOG_DIR"
pids=()
cleanup_servers() {
  local pid
  for pid in "${pids[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  for pid in "${pids[@]:-}"; do
    wait "$pid" 2>/dev/null || true
  done
}
trap cleanup_servers EXIT INT TERM

BASE_PORT=28100
for rank in $(seq 0 7); do
  port=$((BASE_PORT + rank))
  CUDA_VISIBLE_DEVICES=${GPU_UUIDS[$rank]} \
  TOKENIZERS_PARALLELISM=false \
  vllm serve "$MODEL_DIR" \
    --served-model-name "$SERVED_MODEL" \
    --host 127.0.0.1 \
    --port "$port" \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 2048 \
    --max-num-seqs 16 \
    --no-enable-prefix-caching \
    --generation-config vllm \
    --seed 20260822 \
    >"$SERVER_LOG_DIR/rank-$rank.stdout.log" \
    2>"$SERVER_LOG_DIR/rank-$rank.stderr.log" &
  pids+=("$!")
done

for rank in $(seq 0 7); do
  port=$((BASE_PORT + rank))
  healthy=0
  for _ in $(seq 1 180); do
    if curl --fail --silent "http://127.0.0.1:$port/health" >/dev/null; then
      healthy=1
      break
    fi
    sleep 2
  done
  [[ $healthy -eq 1 ]] || preflight_fail "vLLM rank $rank failed health preflight"
done

SERVER_ARGS=()
for rank in $(seq 0 7); do
  SERVER_ARGS+=(--server "http://127.0.0.1:$((BASE_PORT + rank))")
done

runner_status=0
QS_IMAGE="$EXPECTED_IMAGE" python3 -B "$RUNNER" \
  --protocol "$PROTOCOL" \
  --expected-protocol-sha256 "$EXPECTED_PROTOCOL_SHA256" \
  --dataset "$DATASET" \
  --model "$MODEL_DIR" \
  --served-model "$SERVED_MODEL" \
  "${SERVER_ARGS[@]}" \
  --run-dir "$RUN_DIR" || runner_status=$?

cleanup_servers
trap - EXIT INT TERM

if [[ -d "$RUN_DIR" ]]; then
  mkdir -p "$RUN_DIR/server_logs"
  cp "$SERVER_LOG_DIR"/*.log "$RUN_DIR/server_logs/"
  RUNNER_STATUS="$runner_status" RUN_DIR="$RUN_DIR" EXPECTED_IMAGE="$EXPECTED_IMAGE" python3 - <<'PY'
import hashlib, json, os
from pathlib import Path

root = Path(os.environ["RUN_DIR"])
logs = {}
for path in sorted((root / "server_logs").glob("*.log")):
    logs[path.name] = {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
payload = {
    "schema": "a11-boolq-ordered-launcher-receipt-v1",
    "runner_exit_status": int(os.environ["RUNNER_STATUS"]),
    "image": os.environ["EXPECTED_IMAGE"],
    "server_logs": logs,
}
target = root / "launcher_receipt.json"
with target.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
    handle.write("\n")
PY
fi

exit "$runner_status"
