#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

PATH=/root/miniconda3/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/mpi/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/ucx/bin:/opt/amazon/efa/bin
export PATH
unset BASH_ENV ENV CDPATH PYTHONPATH PYTHONHOME CODE_DIR FREEZE_ROOT FREEZE_MANIFEST \
  PYTHON_BIN OFFICIAL_REPO MODEL_DIR VALIDATION_DATA OUTPUT_ROOT \
  INSTRUMENTED_REPO GPU_INDEX PORT || true

PYTHON_BIN=/tmp/round25-hypic-env/venv/bin/python
OFFICIAL_REPO=/tmp/HYPIC-98147c0
K_ROOT=/tmp/rwd5-hypic-store-freeze-k
FREEZE_ROOT=/tmp/rwd5-hypic-mamba-allocator-debug-freeze-d
FREEZE_MANIFEST=${FREEZE_ROOT}/SHA256SUMS
EXPECTED_FREEZE_MANIFEST_SHA256=${EXPECTED_FREEZE_MANIFEST_SHA256:?external audited manifest SHA required}
CODE_DIR=${FREEZE_ROOT}/code
INSTRUMENTED_REPO=/tmp/HYPIC-98147c0-rwd5-mamba-allocator-debug-d
MODEL_DIR=/tmp/Qwen3.5-35B-A3B-hypic-model-view
VALIDATION_DATA=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/data/qcomem-longbench-validation/longbench_validation.jsonl
OUTPUT_ROOT=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/runs/qcomem/hypic-mamba-allocator-debug-20260822d
GPU_INDEX=0
PORT=33600
EXPECTED_COMMIT=98147c01909004e66d98bcb18b886927d41b0ee5
EXPECTED_K_MANIFEST_SHA256=c7f0fcc0b44d6292af52f2d31a0770e7a74982c20eadc8317740106034dc3a7b
EXPECTED_MODEL_WEIGHT_LEDGER_SHA256=8314a82c9188b9b817193e039b0b0eb0636b328512f19b0c12455853b7e20014
EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256=d78424684a27718b44616c80c788e7f8b726feed674fe015901b87de1993f7dd
SERVER_NAME=qwen35-hypic
MODE=transition_rope_recompute
RUNNER=${CODE_DIR}/run_hypic_mamba_allocator_debug.py
PATCH=${CODE_DIR}/hypic_retained_state_instrumentation.patch
RECEIPT_MODULE=${CODE_DIR}/hypic_retained_state_receipt.py
SAFE_CWD_GUARD=${CODE_DIR}/safe_cwd_guard.py
SERVER_PID=""

die() { printf '%s\n' "ERROR: $*" >&2; exit 1; }

cleanup_server() {
  local alive=0 attempt old_pid
  [[ "${SERVER_PID:-}" =~ ^[0-9]+$ ]] || return 0
  old_pid=$SERVER_PID
  kill -TERM -- "-${old_pid}" 2>/dev/null || kill -TERM "${old_pid}" 2>/dev/null || true
  for attempt in $(seq 1 40); do
    alive=0
    kill -0 -- "-${old_pid}" 2>/dev/null && alive=1
    kill -0 "${old_pid}" 2>/dev/null && alive=1
    [[ "$alive" -eq 0 ]] && break
    sleep 0.25
  done
  alive=0
  kill -0 -- "-${old_pid}" 2>/dev/null && alive=1
  kill -0 "${old_pid}" 2>/dev/null && alive=1
  if [[ "$alive" -ne 0 ]]; then
    kill -KILL -- "-${old_pid}" 2>/dev/null || kill -KILL "${old_pid}" 2>/dev/null || true
  fi
  wait "${old_pid}" 2>/dev/null || true
  for attempt in $(seq 1 40); do
    alive=0
    kill -0 -- "-${old_pid}" 2>/dev/null && alive=1
    kill -0 "${old_pid}" 2>/dev/null && alive=1
    [[ "$alive" -eq 0 ]] && break
    sleep 0.25
  done
  if kill -0 -- "-${old_pid}" 2>/dev/null || kill -0 "${old_pid}" 2>/dev/null; then
    printf '%s\n' "server PID/PGID survived cleanup: ${old_pid}" >&2
    return 1
  fi
  SERVER_PID=""
}

on_exit() {
  local status=$?
  trap - EXIT INT TERM
  local cleanup_status=0
  cleanup_server || cleanup_status=$?
  [[ "$cleanup_status" -eq 0 ]] || status=$cleanup_status
  if [[ -d "${OUTPUT_ROOT:-}" && "$status" -ne 0 ]]; then
    rm -f "${OUTPUT_ROOT}/COMPLETED_DEBUG_ONLY"
    printf '%s\n' "$status" > "${OUTPUT_ROOT}/FAILED_DEBUG_ONLY"
  fi
  exit "$status"
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

for path in "$PYTHON_BIN" "$OFFICIAL_REPO/.git" "$K_ROOT/SHA256SUMS" \
  "$FREEZE_MANIFEST" "$RUNNER" "$PATCH" "$RECEIPT_MODULE" "$SAFE_CWD_GUARD" \
  "$CODE_DIR/run_hypic_retained_state_bytes.py" "$CODE_DIR/run_hypic_same_protocol.py" \
  "$MODEL_DIR/config.json" "$MODEL_DIR/model-weights.sha256" \
  "$MODEL_DIR/model-artifacts.sha256" "$VALIDATION_DATA"; do
  [[ -e "$path" ]] || die "missing path: $path"
done
[[ ! -e "$OUTPUT_ROOT" ]] || die "debug output already exists: $OUTPUT_ROOT"
[[ ! -e "$INSTRUMENTED_REPO" ]] || die "debug instrumented repo already exists: $INSTRUMENTED_REPO"
[[ "$(sha256sum "$FREEZE_MANIFEST" | awk '{print $1}')" == "$EXPECTED_FREEZE_MANIFEST_SHA256" ]] || die "external debug manifest identity drift"
(cd "$FREEZE_ROOT" && sha256sum -c "$FREEZE_MANIFEST") || die "external debug freeze drift"
[[ "$(sha256sum "$K_ROOT/SHA256SUMS" | awk '{print $1}')" == "$EXPECTED_K_MANIFEST_SHA256" ]] || die "K manifest identity drift"
(cd "$K_ROOT" && sha256sum -c "$K_ROOT/SHA256SUMS") || die "K freeze drift"
[[ "$(sha256sum "$MODEL_DIR/model-weights.sha256" | awk '{print $1}')" == "$EXPECTED_MODEL_WEIGHT_LEDGER_SHA256" ]] || die "model weight ledger drift"
[[ "$(sha256sum "$MODEL_DIR/model-artifacts.sha256" | awk '{print $1}')" == "$EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256" ]] || die "model artifact ledger drift"
[[ "$(wc -l < "$MODEL_DIR/model-weights.sha256" | tr -d ' ')" == "14" ]] || die "model weight ledger entry count drift"
[[ "$(wc -l < "$MODEL_DIR/model-artifacts.sha256" | tr -d ' ')" == "9" ]] || die "model artifact ledger entry count drift"
(cd "$MODEL_DIR" && sha256sum -c "$MODEL_DIR/model-weights.sha256") || die "model weight payload drift"
(cd "$MODEL_DIR" && sha256sum -c "$MODEL_DIR/model-artifacts.sha256") || die "model artifact payload drift"
[[ "$(git -C "$OFFICIAL_REPO" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || die "official commit drift"
[[ -z "$(git -C "$OFFICIAL_REPO" status --porcelain --untracked-files=all)" ]] || die "official repo dirty"
git -C "$OFFICIAL_REPO" apply --check "$PATCH"
/usr/bin/env -i PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 "$PYTHON_BIN" "$SAFE_CWD_GUARD" /
cd /
[[ "$PWD" == "/" ]] || die "failed to enter safe fixed cwd"

mkdir -p "$OUTPUT_ROOT"/{commands,debug-receipts,formal-receipts-disabled,logs,run-summaries,targets}
git clone --quiet --no-hardlinks "$OFFICIAL_REPO" "$INSTRUMENTED_REPO"
git -C "$INSTRUMENTED_REPO" checkout --quiet --detach "$EXPECTED_COMMIT"
git -C "$INSTRUMENTED_REPO" apply "$PATCH"
cp "$RECEIPT_MODULE" "$INSTRUMENTED_REPO/python/sglang/srt/retained_state_receipt.py"
mapfile -t overlay_status < <(git -C "$INSTRUMENTED_REPO" status --porcelain --untracked-files=all)
expected_status=(
  " M python/sglang/srt/managers/scheduler.py"
  " M python/sglang/srt/mem_cache/common.py"
  "?? python/sglang/srt/retained_state_receipt.py"
)
mapfile -t overlay_status_sorted < <(printf '%s\n' "${overlay_status[@]}" | sort)
mapfile -t expected_status_sorted < <(printf '%s\n' "${expected_status[@]}" | sort)
[[ "${overlay_status_sorted[*]}" == "${expected_status_sorted[*]}" ]] || die "instrumentation-only overlay status drift"
git -C "$INSTRUMENTED_REPO" diff --binary > "$OUTPUT_ROOT/instrumentation-overlay.diff"

GPU_UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$GPU_INDEX" | tr -d '[:space:]')
[[ "$GPU_UUID" == GPU-* ]] || die "missing GPU UUID"
nvidia-smi --query-gpu=index,name,uuid,memory.used,utilization.gpu --format=csv,noheader > "$OUTPUT_ROOT/nvidia-smi-before.txt"
TARGET="$OUTPUT_ROOT/targets/${MODE}-rank-0.json"
DEBUG_RECEIPT="$OUTPUT_ROOT/debug-receipts/${MODE}-rank-0.json"
VALIDATION_RECEIPT="$OUTPUT_ROOT/debug-receipts/${MODE}-validation.json"
MODE_ARGS=(--page-size 1 --chunked-prefill-size -1 --mamba-radix-cache-strategy no_buffer --pic-enable --pic-mode transition_rope_recompute --pic-separator-str '<<PIC_SEP>>')

printf '%q ' env CUDA_VISIBLE_DEVICES="$GPU_UUID" PIC_SEAM_SINK=8 \
  SGLANG_MAMBA_CONV_DTYPE=bfloat16 SGLANG_MAMBA_SSM_DTYPE=float32 \
  FORKAUDIT_RWD5_TARGET_PATH="$TARGET" \
  FORKAUDIT_RWD5_RECEIPT_DIR="$OUTPUT_ROOT/formal-receipts-disabled" \
  FORKAUDIT_RWD5_MAMBA_ALLOCATOR_DEBUG_PATH="$DEBUG_RECEIPT" \
  FORKAUDIT_RWD5_MODE="$MODE" FORKAUDIT_RWD5_RANK=0 \
  PYTHONPATH="$INSTRUMENTED_REPO/python:$CODE_DIR" "$PYTHON_BIN" -m sglang.launch_server \
  --model-path "$MODEL_DIR" --served-model-name "$SERVER_NAME" --host 127.0.0.1 --port "$PORT" \
  --tp-size 1 --dtype bfloat16 --context-length 8192 --max-running-requests 1 --max-total-tokens 8192 \
  --mem-fraction-static 0.80 --random-seed 20260821 --sampling-backend pytorch \
  --disable-cuda-graph --disable-piecewise-cuda-graph --disable-overlap-schedule --enable-cache-report "${MODE_ARGS[@]}" \
  > "$OUTPUT_ROOT/commands/server.txt"
printf '\n' >> "$OUTPUT_ROOT/commands/server.txt"

setsid /usr/bin/env -i PATH="$PATH" HOME=/root USER=root LOGNAME=root SHELL=/bin/bash \
  LANG=C.UTF-8 LC_ALL=C.UTF-8 TZ=UTC \
  LD_LIBRARY_PATH=/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
  LIBRARY_PATH=/usr/local/cuda/lib64/stubs CUDA_HOME=/usr/local/cuda \
  PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 CUDA_VISIBLE_DEVICES="$GPU_UUID" PIC_SEAM_SINK=8 \
  SGLANG_MAMBA_CONV_DTYPE=bfloat16 SGLANG_MAMBA_SSM_DTYPE=float32 \
  FORKAUDIT_RWD5_TARGET_PATH="$TARGET" \
  FORKAUDIT_RWD5_RECEIPT_DIR="$OUTPUT_ROOT/formal-receipts-disabled" \
  FORKAUDIT_RWD5_MAMBA_ALLOCATOR_DEBUG_PATH="$DEBUG_RECEIPT" \
  FORKAUDIT_RWD5_MODE="$MODE" FORKAUDIT_RWD5_RANK=0 \
  SGLANG_NUMA_BIND_V2=0 SGLANG_IS_FLASHINFER_AVAILABLE=0 PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="$INSTRUMENTED_REPO/python:$CODE_DIR" \
  "$PYTHON_BIN" -m sglang.launch_server --model-path "$MODEL_DIR" --served-model-name "$SERVER_NAME" \
  --host 127.0.0.1 --port "$PORT" --tp-size 1 --dtype bfloat16 --context-length 8192 \
  --max-running-requests 1 --max-total-tokens 8192 --mem-fraction-static 0.80 \
  --random-seed 20260821 --sampling-backend pytorch --disable-cuda-graph \
  --disable-piecewise-cuda-graph --disable-overlap-schedule --enable-cache-report "${MODE_ARGS[@]}" \
  > "$OUTPUT_ROOT/logs/server.log" 2>&1 &
SERVER_PID=$!
ready=0
for _ in $(seq 1 240); do
  kill -0 "$SERVER_PID" 2>/dev/null || die "debug server exited during model_info wait"
  if curl -fsS --max-time 3 "http://127.0.0.1:${PORT}/model_info" >/dev/null 2>&1; then ready=1; break; fi
  sleep 5
done
[[ "$ready" -eq 1 ]] || die "debug model_info readiness timeout"
curl -fsS --retry 100 --retry-all-errors --retry-delay 1 --max-time 3 \
  "http://127.0.0.1:${PORT}/server_info" > "$OUTPUT_ROOT/debug-receipts/server-info.json"

/usr/bin/env -i PATH="$PATH" HOME=/root PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 \
  PYTHONPATH="$CODE_DIR" "$PYTHON_BIN" "$RUNNER" --stage run --mode "$MODE" --rank 0 \
  --model "$MODEL_DIR" --data "$VALIDATION_DATA" --base-url "http://127.0.0.1:${PORT}" \
  --served-model-name "$SERVER_NAME" --target-file "$TARGET" \
  --allocator-debug-receipt "$DEBUG_RECEIPT" --output "$OUTPUT_ROOT/run-summaries/${MODE}-rank-0.json"
WORKLOAD_ID=$(/usr/bin/env -i PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 "$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["workload_id"])' "$OUTPUT_ROOT/run-summaries/${MODE}-rank-0.json")
[[ "$WORKLOAD_ID" == "qasper-6" ]] || die "debug workload literal drift"
TARGET_SHA=$(sha256sum "$TARGET" | awk '{print $1}')
/usr/bin/env -i PATH="$PATH" HOME=/root PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 \
  PYTHONPATH="$CODE_DIR" "$PYTHON_BIN" "$RUNNER" --stage validate \
  --allocator-debug-receipt "$DEBUG_RECEIPT" --target-file "$TARGET" \
  --target-sha256 "$TARGET_SHA" --output "$VALIDATION_RECEIPT"

[[ -s "$DEBUG_RECEIPT" && -s "$VALIDATION_RECEIPT" ]] || die "allocator debug receipts missing"
[[ -z "$(find "$OUTPUT_ROOT/formal-receipts-disabled" -mindepth 1 -print -quit)" ]] || die "formal receipt member emitted in debug mode"
/usr/bin/env -i PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 "$PYTHON_BIN" - \
  "$OUTPUT_ROOT/run-summaries/${MODE}-rank-0.json" "$DEBUG_RECEIPT" \
  "$VALIDATION_RECEIPT" "$TARGET" "$EXPECTED_COMMIT" <<'PY'
import hashlib, json, pathlib, sys
run_path, raw_path, validation_path, target_path = map(pathlib.Path, sys.argv[1:5])
commit = sys.argv[5]
run = json.loads(run_path.read_text())
raw = json.loads(raw_path.read_text())
validation = json.loads(validation_path.read_text())
target = json.loads(target_path.read_text())
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
assert set(run) == {
    "schema", "status", "official_commit", "mode", "rank", "workload_id",
    "target_sha256", "allocator_debug_receipt_sha256", "validation",
    "warm_prime", "warmup", "prime", "flush_response",
    "formal_receipts_emitted", "paper_evidence",
}
assert run["schema"] == "hypic-rwd5-mamba-allocator-debug-run-v1"
assert run["status"] == "completed_debug_only_not_formal_evidence"
assert run["official_commit"] == commit and run["mode"] == "transition_rope_recompute"
assert run["rank"] == 0 and run["workload_id"] == "qasper-6"
assert run["formal_receipts_emitted"] == 0 and run["paper_evidence"] is False
assert target["snapshot_id"] == "allocator-debug-transition_rope_recompute-rank-0"
assert target["workload_id"] == "qasper-6" and target["mode"] == run["mode"] and target["rank"] == 0
assert run["target_sha256"] == sha(target_path)
assert run["allocator_debug_receipt_sha256"] == sha(raw_path)
assert validation["allocator_debug_receipt_sha256"] == sha(raw_path)
assert validation["schema"] == "hypic-rwd5-mamba-allocator-debug-validation-v1"
assert validation["status"] == "passed_exact_duplicate_representation_capture"
assert validation["official_commit"] == commit and validation["paper_evidence"] is False
assert validation["workload_id"] == "qasper-6"
assert validation["duplicate_excess_count"] > 0 and validation["duplicates"]
replayed = dict(validation)
replayed.pop("allocator_debug_receipt_sha256")
assert run["validation"] == replayed
terminal = {
    "schema": "hypic-rwd5-mamba-allocator-debug-terminal-v1",
    "status": "passed_exact_debug_only_terminal_binding",
    "paper_evidence": False,
    "mode": run["mode"], "rank": run["rank"], "workload_id": run["workload_id"],
    "run_summary_sha256": sha(run_path), "raw_receipt_sha256": sha(raw_path),
    "validation_receipt_sha256": sha(validation_path), "target_sha256": sha(target_path),
}
(validation_path.parent / "terminal-binding.json").write_text(
    json.dumps(terminal, sort_keys=True, separators=(",", ":")) + "\n"
)
PY

cleanup_server || die "debug server cleanup failed"
nvidia-smi --query-gpu=index,name,uuid,memory.used,utilization.gpu --format=csv,noheader > "$OUTPUT_ROOT/nvidia-smi-after.txt"
[[ "$(awk -F, 'NR==1 {gsub(/ MiB/,"",$4); gsub(/ /,"",$4); print $4}' "$OUTPUT_ROOT/nvidia-smi-after.txt")" == "0" ]] || die "GPU not released after debug"
find "$OUTPUT_ROOT" -type f ! -name all-debug-artifacts.sha256 ! -name COMPLETED_DEBUG_ONLY -print0 \
  | sort -z | xargs -0 sha256sum > "$OUTPUT_ROOT/all-debug-artifacts.sha256"
sha256sum -c "$OUTPUT_ROOT/all-debug-artifacts.sha256" >/dev/null
touch "$OUTPUT_ROOT/COMPLETED_DEBUG_ONLY"
rm -f "$OUTPUT_ROOT/FAILED_DEBUG_ONLY"
trap - EXIT ERR INT TERM
