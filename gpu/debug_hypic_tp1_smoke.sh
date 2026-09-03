#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-transition_rope_recompute}
PORT=${2:-33201}
PYTHON_BIN=/tmp/round25-hypic-env/venv/bin/python
HYPIC_REPO=/tmp/HYPIC-98147c0
CODE_DIR=/tmp/round25-hypic-formal-code
MODEL_DIR=/tmp/Qwen3.5-35B-A3B-hypic-model-view
GPU_UUID=GPU-73650b19-ac2c-d385-e0c9-6f93c6f2bb57
DEBUG_DIR=/tmp/round25-related-direct-compare-${MODE}-${PORT}
PID_FILE="$DEBUG_DIR/server.pid"

case "$MODE" in
  full_recompute)
    MODE_ARGS=(--disable-radix-cache --mamba-radix-cache-strategy no_buffer)
    ;;
  prefix_cache)
    MODE_ARGS=(--mamba-radix-cache-strategy extra_buffer)
    ;;
  transition_rope_recompute)
    MODE_ARGS=(
      --page-size 1 --chunked-prefill-size -1
      --mamba-radix-cache-strategy no_buffer
      --pic-enable --pic-mode transition_rope_recompute
      --pic-separator-str '<<PIC_SEP>>'
    )
    ;;
  *) echo "unknown mode: $MODE" >&2; exit 2 ;;
esac

[[ ! -e "$DEBUG_DIR" ]] || { echo "debug directory already exists" >&2; exit 1; }
mkdir -p "$DEBUG_DIR/cache"
setsid bash -c \
  'pid_file=$1; shift; printf "%s\n" "$$" > "$pid_file"; exec "$@"' \
  hypic-server "$PID_FILE" env \
  CUDA_VISIBLE_DEVICES="$GPU_UUID" PIC_SEAM_SINK=8 SGLANG_NUMA_BIND_V2=0 \
  SGLANG_IS_FLASHINFER_AVAILABLE=0 \
  PYTHONPATH="$HYPIC_REPO/python:$CODE_DIR" PYTHONUNBUFFERED=1 \
  PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX="$DEBUG_DIR/cache/pycache" \
  TORCHINDUCTOR_CACHE_DIR="$DEBUG_DIR/cache/torchinductor" \
  TRITON_CACHE_DIR="$DEBUG_DIR/cache/triton" \
  XDG_CACHE_HOME="$DEBUG_DIR/cache/xdg" HF_HOME="$DEBUG_DIR/cache/huggingface" \
  "$PYTHON_BIN" -m sglang.launch_server \
  --model-path "$MODEL_DIR" --served-model-name qwen35-hypic \
  --host 127.0.0.1 --port "$PORT" --tp-size 1 --dtype bfloat16 \
  --context-length 8192 --max-running-requests 1 --max-total-tokens 8192 \
  --mem-fraction-static 0.80 --random-seed 20260821 \
  --sampling-backend pytorch \
  --disable-cuda-graph --disable-piecewise-cuda-graph --disable-overlap-schedule \
  --enable-cache-report \
  "${MODE_ARGS[@]}" \
  > "$DEBUG_DIR/server.log" 2>&1 &

for _ in $(seq 1 100); do
  [[ -s "$PID_FILE" ]] && break
  sleep 0.05
done
[[ -s "$PID_FILE" ]]
pid=$(cat "$PID_FILE")
[[ "$pid" =~ ^[0-9]+$ ]]
kill -0 "$pid"
printf 'SERVER_PID=%s\n' "$pid"
