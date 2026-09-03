#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

PYTHON_BIN=${PYTHON_BIN:-/tmp/round25-hypic-env/venv/bin/python}
OFFICIAL_REPO=${OFFICIAL_REPO:-/tmp/HYPIC-98147c0}
INSTRUMENTED_REPO=${INSTRUMENTED_REPO:-/tmp/HYPIC-98147c0-rwd5-dtype-debug-j}
CODE_DIR=${CODE_DIR:?copy the reviewed J debug bundle and supply CODE_DIR}
MODEL_DIR=${MODEL_DIR:-/tmp/Qwen3.5-35B-A3B-hypic-model-view}
VALIDATION_DATA=${VALIDATION_DATA:-/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/data/qcomem-longbench-validation/longbench_validation.jsonl}
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/runs/qcomem/hypic-component-dtype-debug-trial1879097-20260822j}
GPU_INDEX=${GPU_INDEX:-0}
PORT=${PORT:-33500}
EXPECTED_COMMIT=98147c01909004e66d98bcb18b886927d41b0ee5
CLIENT=${CODE_DIR}/run_hypic_retained_state_bytes.py
PATCH=${CODE_DIR}/hypic_retained_state_instrumentation.patch
RECEIPT_MODULE=${CODE_DIR}/hypic_retained_state_receipt.py
SERVER_NAME=qwen35-hypic
SERVER_PID=""

die() { printf '%s\n' "ERROR: $*" >&2; exit 1; }

cleanup_server() {
  local alive=0 attempt
  [[ "${SERVER_PID:-}" =~ ^[0-9]+$ ]] || return 0
  kill -TERM -- "-${SERVER_PID}" 2>/dev/null || kill -TERM "${SERVER_PID}" 2>/dev/null || true
  for attempt in $(seq 1 40); do
    alive=0
    kill -0 -- "-${SERVER_PID}" 2>/dev/null && alive=1
    kill -0 "${SERVER_PID}" 2>/dev/null && alive=1
    [[ "$alive" -eq 0 ]] && break
    sleep 0.25
  done
  alive=0
  kill -0 -- "-${SERVER_PID}" 2>/dev/null && alive=1
  kill -0 "${SERVER_PID}" 2>/dev/null && alive=1
  if [[ "$alive" -ne 0 ]]; then
    kill -KILL -- "-${SERVER_PID}" 2>/dev/null || kill -KILL "${SERVER_PID}" 2>/dev/null || true
  fi
  wait "${SERVER_PID}" 2>/dev/null || true
  SERVER_PID=""
}

on_exit() {
  local status=$?
  trap - EXIT ERR INT TERM
  if [[ -d "${OUTPUT_ROOT:-}" && "$status" -ne 0 ]]; then
    rm -f "${OUTPUT_ROOT}/COMPLETED_DEBUG_ONLY"
    printf '%s\n' "$status" > "${OUTPUT_ROOT}/FAILED_DEBUG_ONLY"
  fi
  cleanup_server
  exit "$status"
}
trap on_exit EXIT ERR INT TERM

for path in "$PYTHON_BIN" "$OFFICIAL_REPO/.git" "$CLIENT" "$PATCH" "$RECEIPT_MODULE" "$MODEL_DIR/config.json" "$VALIDATION_DATA"; do
  [[ -e "$path" ]] || die "missing path: $path"
done
[[ ! -e "$OUTPUT_ROOT" ]] || die "debug output already exists: $OUTPUT_ROOT"
[[ ! -e "$INSTRUMENTED_REPO" ]] || die "debug instrumented repo already exists: $INSTRUMENTED_REPO"
[[ "$(git -C "$OFFICIAL_REPO" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || die "official commit drift"
[[ -z "$(git -C "$OFFICIAL_REPO" status --porcelain --untracked-files=all)" ]] || die "official repo dirty"
git -C "$OFFICIAL_REPO" apply --check "$PATCH"

mkdir -p "$OUTPUT_ROOT"/{commands,debug-receipts,formal-receipts-disabled,logs,run-summaries,targets}
git clone --quiet --no-hardlinks "$OFFICIAL_REPO" "$INSTRUMENTED_REPO"
git -C "$INSTRUMENTED_REPO" checkout --quiet --detach "$EXPECTED_COMMIT"
git -C "$INSTRUMENTED_REPO" apply "$PATCH"
cp "$RECEIPT_MODULE" "$INSTRUMENTED_REPO/python/sglang/srt/retained_state_receipt.py"
GPU_UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$GPU_INDEX" | tr -d '[:space:]')
[[ "$GPU_UUID" == GPU-* ]] || die "missing GPU UUID"
nvidia-smi --query-gpu=index,name,uuid,memory.used,utilization.gpu --format=csv,noheader > "$OUTPUT_ROOT/nvidia-smi-before.txt"

run_debug_mode() {
  local mode
  mode=$1
  local target="$OUTPUT_ROOT/targets/${mode}-rank-0.json"
  local debug_receipt="$OUTPUT_ROOT/debug-receipts/${mode}-rank-0.json"
  local validation_receipt="$OUTPUT_ROOT/debug-receipts/${mode}-validation.json"
  local mode_args=() ready=0
  case "$mode" in
    prefix_cache) mode_args=(--mamba-radix-cache-strategy extra_buffer) ;;
    transition_rope_recompute)
      mode_args=(--page-size 1 --chunked-prefill-size -1 --mamba-radix-cache-strategy no_buffer --pic-enable --pic-mode transition_rope_recompute --pic-separator-str '<<PIC_SEP>>') ;;
    *) die "unapproved debug mode: $mode" ;;
  esac
  if [[ "${RWD5_DTYPE_DEBUG_DECLARATION_SMOKE_ONLY:-0}" == 1 ]]; then
    printf '%s\n' "$mode|$target|$debug_receipt|$validation_receipt"
    return 0
  fi
  printf '%q ' env CUDA_VISIBLE_DEVICES="$GPU_UUID" PIC_SEAM_SINK=8 \
    SGLANG_MAMBA_CONV_DTYPE=bfloat16 SGLANG_MAMBA_SSM_DTYPE=float32 \
    FORKAUDIT_RWD5_TARGET_PATH="$target" \
    FORKAUDIT_RWD5_RECEIPT_DIR="$OUTPUT_ROOT/formal-receipts-disabled" \
    FORKAUDIT_RWD5_DTYPE_DEBUG_PATH="$debug_receipt" \
    FORKAUDIT_RWD5_MODE="$mode" FORKAUDIT_RWD5_RANK=0 \
    PYTHONPATH="$INSTRUMENTED_REPO/python:$CODE_DIR" "$PYTHON_BIN" -m sglang.launch_server \
    --model-path "$MODEL_DIR" --served-model-name "$SERVER_NAME" --host 127.0.0.1 --port "$PORT" \
    --tp-size 1 --dtype bfloat16 --context-length 8192 --max-running-requests 1 --max-total-tokens 8192 \
    --mem-fraction-static 0.80 --random-seed 20260821 --sampling-backend pytorch \
    --disable-cuda-graph --disable-piecewise-cuda-graph --disable-overlap-schedule --enable-cache-report "${mode_args[@]}" \
    > "$OUTPUT_ROOT/commands/${mode}.txt"
  printf '\n' >> "$OUTPUT_ROOT/commands/${mode}.txt"
  setsid env CUDA_VISIBLE_DEVICES="$GPU_UUID" PIC_SEAM_SINK=8 \
    SGLANG_MAMBA_CONV_DTYPE=bfloat16 SGLANG_MAMBA_SSM_DTYPE=float32 \
    FORKAUDIT_RWD5_TARGET_PATH="$target" \
    FORKAUDIT_RWD5_RECEIPT_DIR="$OUTPUT_ROOT/formal-receipts-disabled" \
    FORKAUDIT_RWD5_DTYPE_DEBUG_PATH="$debug_receipt" \
    FORKAUDIT_RWD5_MODE="$mode" FORKAUDIT_RWD5_RANK=0 \
    SGLANG_NUMA_BIND_V2=0 SGLANG_IS_FLASHINFER_AVAILABLE=0 PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="$INSTRUMENTED_REPO/python:$CODE_DIR" \
    "$PYTHON_BIN" -m sglang.launch_server --model-path "$MODEL_DIR" --served-model-name "$SERVER_NAME" \
    --host 127.0.0.1 --port "$PORT" --tp-size 1 --dtype bfloat16 --context-length 8192 \
    --max-running-requests 1 --max-total-tokens 8192 --mem-fraction-static 0.80 \
    --random-seed 20260821 --sampling-backend pytorch --disable-cuda-graph \
    --disable-piecewise-cuda-graph --disable-overlap-schedule --enable-cache-report "${mode_args[@]}" \
    > "$OUTPUT_ROOT/logs/${mode}.log" 2>&1 &
  SERVER_PID=$!
  for _ in $(seq 1 240); do
    kill -0 "$SERVER_PID" 2>/dev/null || die "$mode server exited during model_info wait"
    if curl -fsS --max-time 3 "http://127.0.0.1:${PORT}/model_info" >/dev/null 2>&1; then ready=1; break; fi
    sleep 5
  done
  [[ "$ready" -eq 1 ]] || die "$mode model_info readiness timeout"
  curl -fsS --retry 100 --retry-all-errors --retry-delay 1 --max-time 3 \
    "http://127.0.0.1:${PORT}/server_info" > "$OUTPUT_ROOT/debug-receipts/${mode}-server-info.json"
  PYTHONPATH="$CODE_DIR" "$PYTHON_BIN" "$CLIENT" --stage dtype_debug \
    --mode "$mode" --rank 0 --model "$MODEL_DIR" --data "$VALIDATION_DATA" \
    --base-url "http://127.0.0.1:${PORT}" --served-model-name "$SERVER_NAME" \
    --target-file "$target" --dtype-debug-receipt "$debug_receipt" \
    --output "$OUTPUT_ROOT/run-summaries/${mode}-rank-0.json"
  PYTHONPATH="$CODE_DIR" "$PYTHON_BIN" "$CLIENT" --stage dtype_debug_validate \
    --mode "$mode" --rank 0 --model "$MODEL_DIR" \
    --base-url "http://127.0.0.1:${PORT}" \
    --dtype-debug-receipt "$debug_receipt" --output "$validation_receipt"
  [[ -s "$debug_receipt" && -s "$OUTPUT_ROOT/run-summaries/${mode}-rank-0.json" \
      && -s "$validation_receipt" ]] || die "$mode debug validation receipts missing"
  [[ -z "$(find "$OUTPUT_ROOT/formal-receipts-disabled" -type f -print -quit)" ]] || die "formal receipt emitted in debug mode"
  cleanup_server
}

run_debug_mode prefix_cache
run_debug_mode transition_rope_recompute
for mode in prefix_cache transition_rope_recompute; do
  "$PYTHON_BIN" - "$OUTPUT_ROOT/debug-receipts/${mode}-validation.json" "$mode" "$EXPECTED_COMMIT" <<'PY'
import json
import sys

path, mode, commit = sys.argv[1:]
row = json.load(open(path))
assert row["schema"] == "hypic-rwd5-component-dtype-debug-validation-v1"
assert row["status"] == "passed_exact_live_component_contract"
assert row["official_commit"] == commit and row["mode"] == mode
assert row["paper_evidence"] is False
assert len(row["debug_receipt_sha256"]) == 64
PY
done
[[ -z "$(find "$OUTPUT_ROOT/formal-receipts-disabled" -type f -print -quit)" ]] || die "formal receipt emitted in debug run"
nvidia-smi --query-gpu=index,name,uuid,memory.used,utilization.gpu --format=csv,noheader > "$OUTPUT_ROOT/nvidia-smi-after.txt"
find "$OUTPUT_ROOT" -type f ! -name all-debug-artifacts.sha256 ! -name COMPLETED_DEBUG_ONLY -print0 \
  | sort -z | xargs -0 sha256sum > "$OUTPUT_ROOT/all-debug-artifacts.sha256"
sha256sum -c "$OUTPUT_ROOT/all-debug-artifacts.sha256" >/dev/null
touch "$OUTPUT_ROOT/COMPLETED_DEBUG_ONLY"
rm -f "$OUTPUT_ROOT/FAILED_DEBUG_ONLY"
trap - EXIT ERR INT TERM
