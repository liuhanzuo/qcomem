#!/usr/bin/env bash
set -Eeuo pipefail

# Debug-only reproduction for the three cache-on clients that exposed SGLang
# 0.5.17 streaming edge cases in formal attempt B. This script is not a
# scientific result and deliberately does not rerun the five unaffected ranks
# or the cache-off phase.

ENV_DIR=/tmp/qcomem_sglang_v0517_cu129_env
CODE_DIR=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench/qcomem_sglang_related_debug_20260821c
MODEL_DIR=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/models/Qwen3.5-35B-A3B-59d61f3
DATA=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/data/qcomem-longbench-validation/longbench_validation.jsonl
PROCESSOR_VIEW=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/runs/qcomem/related-sglang-radix-node-20260821b/processor-view
OUT=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/debug_sessions/sglang-radix-cache-on-3gpu-20260821c
DATA_SHA=1553739496b3c209988de56a4ccf574d407379b6b7937ddfafacbe66925069fe
RANKS=(0 2 3)
PIDS=()

cleanup() {
  local pid
  for pid in "${PIDS[@]:-}"; do kill -TERM "$pid" 2>/dev/null || true; done
  for pid in "${PIDS[@]:-}"; do wait "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT INT TERM

[[ ! -e "$OUT" ]] || { echo "debug output already exists: $OUT" >&2; exit 2; }
mkdir -p "$OUT"/{raw,logs}
mapfile -t GPU_UUIDS < <(nvidia-smi --query-gpu=uuid --format=csv,noheader | sed 's/[[:space:]]//g')
[[ ${#GPU_UUIDS[@]} -eq 8 ]]

for slot in "${!RANKS[@]}"; do
  rank=${RANKS[$slot]}
  port=$((18600 + slot))
  setsid env CUDA_VISIBLE_DEVICES="${GPU_UUIDS[$rank]}" PYTHONUNBUFFERED=1 \
    "$ENV_DIR/bin/python" -m sglang.launch_server \
    --model-path "$MODEL_DIR" --served-model-name qwen35-related-baseline \
    --tokenizer-path "$PROCESSOR_VIEW" --host 127.0.0.1 --port "$port" \
    --tp-size 1 --dtype bfloat16 --context-length 8192 \
    --max-running-requests 1 --max-total-tokens 8192 \
    --chunked-prefill-size 8192 --mem-fraction-static 0.82 \
    --random-seed $((20260820 + rank)) --disable-cuda-graph \
    --mamba-radix-cache-strategy extra_buffer --page-size 64 \
    --enable-cache-report --enable-metrics \
    > "$OUT/logs/server-rank-${rank}.log" 2>&1 &
  PIDS[$slot]=$!
done

for slot in "${!RANKS[@]}"; do
  port=$((18600 + slot))
  pid=${PIDS[$slot]}
  for _ in $(seq 1 240); do
    kill -0 "$pid" 2>/dev/null || { echo "server exited: $pid" >&2; exit 3; }
    curl -fsS --max-time 3 "http://127.0.0.1:${port}/model_info" >/dev/null 2>&1 && break
    sleep 5
  done
  curl -fsS --max-time 3 "http://127.0.0.1:${port}/model_info" >/dev/null
done

CLIENTS=()
for slot in "${!RANKS[@]}"; do
  rank=${RANKS[$slot]}
  port=$((18600 + slot))
  "$ENV_DIR/bin/python" "$CODE_DIR/run_related_work_serving_baseline.py" \
    --stage client --system sglang-0.5.17-qwen35-radix-extra-buffer \
    --phase cache_on --rank "$rank" --world-size 8 \
    --model "$MODEL_DIR" --data "$DATA" --expected-data-sha256 "$DATA_SHA" \
    --base-url "http://127.0.0.1:${port}" \
    --served-model-name qwen35-related-baseline --max-new-tokens 32 \
    --output "$OUT/raw/cache_on-rank-${rank}.json" \
    > "$OUT/logs/client-rank-${rank}.log" 2>&1 &
  CLIENTS[$slot]=$!
done
for slot in "${!RANKS[@]}"; do wait "${CLIENTS[$slot]}"; done

sha256sum "$OUT"/raw/*.json > "$OUT/raw.sha256"
date -u +%FT%TZ > "$OUT/DEBUG_COMPLETE"
