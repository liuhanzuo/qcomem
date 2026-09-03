#!/usr/bin/env bash
set -Eeuo pipefail
export LC_ALL=C

CODE_DIR=${CODE_DIR:?set CODE_DIR}
MODEL_DIR=${MODEL_DIR:?set MODEL_DIR}
MODEL_ARTIFACT_LEDGER_FILE=${MODEL_ARTIFACT_LEDGER_FILE:?set MODEL_ARTIFACT_LEDGER_FILE}
MODEL_WEIGHT_LEDGER_FILE=${MODEL_WEIGHT_LEDGER_FILE:?set MODEL_WEIGHT_LEDGER_FILE}
VALIDATION_DATA=${VALIDATION_DATA:?set VALIDATION_DATA}
SGLANG_ENV=${SGLANG_ENV:?set SGLANG_ENV}
ENV_FREEZE_FILE=${ENV_FREEZE_FILE:?set ENV_FREEZE_FILE}
PROCESSOR_CONFIG_FILE=${PROCESSOR_CONFIG_FILE:?set PROCESSOR_CONFIG_FILE}
RUN_DIR=${RUN_DIR:?set RUN_DIR}
CODE_LEDGER_FILE=${CODE_LEDGER_FILE:?set CODE_LEDGER_FILE}
PROTOCOL_FILE=${PROTOCOL_FILE:?set PROTOCOL_FILE}
EXPECTED_CODE_LEDGER_SHA256=${EXPECTED_CODE_LEDGER_SHA256:?set EXPECTED_CODE_LEDGER_SHA256}
EXPECTED_PROTOCOL_SHA256=${EXPECTED_PROTOCOL_SHA256:?set EXPECTED_PROTOCOL_SHA256}
EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256=${EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256:?set EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256}
EXPECTED_MODEL_WEIGHT_LEDGER_SHA256=${EXPECTED_MODEL_WEIGHT_LEDGER_SHA256:?set EXPECTED_MODEL_WEIGHT_LEDGER_SHA256}
EXPECTED_VALIDATION_SHA256=${EXPECTED_VALIDATION_SHA256:?set EXPECTED_VALIDATION_SHA256}
EXPECTED_ENV_FREEZE_SHA256=${EXPECTED_ENV_FREEZE_SHA256:?set EXPECTED_ENV_FREEZE_SHA256}
EXPECTED_PROCESSOR_CONFIG_SHA256=${EXPECTED_PROCESSOR_CONFIG_SHA256:?set EXPECTED_PROCESSOR_CONFIG_SHA256}

SERVER_NAME=qwen35-related-baseline
SYSTEM_NAME=sglang-0.5.17-qwen35-radix-extra-buffer
WORLD_SIZE=8
BASE_PORT=18500
SERVER_PIDS=()
PROCESSOR_VIEW=

fail() {
  local message=$1
  printf '%s\n' "$message" >&2
  mkdir -p "$RUN_DIR/stages" 2>/dev/null || true
  date -u +%FT%TZ > "$RUN_DIR/stages/FAILED" 2>/dev/null || true
  exit 1
}

cleanup_servers() {
  local pid
  for pid in "${SERVER_PIDS[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    fi
  done
  for pid in "${SERVER_PIDS[@]:-}"; do
    wait "$pid" 2>/dev/null || true
  done
  SERVER_PIDS=()
}

on_exit() {
  local rc=$?
  local log_file
  trap - EXIT INT TERM
  if (( rc != 0 )); then
    printf 'launcher_exit_code=%d\n' "$rc" >&2
    if [[ -d "${RUN_DIR:-}/logs" ]]; then
      while IFS= read -r -d '' log_file; do
        printf '\n===== %s =====\n' "$log_file" >&2
        tail -n 200 "$log_file" >&2 || true
      done < <(find "$RUN_DIR/logs" -maxdepth 1 -type f -print0 | sort -z)
    fi
    if [[ -d "${RUN_DIR:-}/server-logs" ]]; then
      while IFS= read -r -d '' log_file; do
        printf '\n===== %s =====\n' "$log_file" >&2
        tail -n 120 "$log_file" >&2 || true
      done < <(find "$RUN_DIR/server-logs" -maxdepth 1 -type f -print0 | sort -z)
    fi
  fi
  cleanup_servers
  exit "$rc"
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

for digest in "$EXPECTED_CODE_LEDGER_SHA256" "$EXPECTED_PROTOCOL_SHA256" \
  "$EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256" "$EXPECTED_MODEL_WEIGHT_LEDGER_SHA256" \
  "$EXPECTED_VALIDATION_SHA256" "$EXPECTED_ENV_FREEZE_SHA256"; do
  [[ "$digest" =~ ^[0-9a-f]{64}$ ]] || fail "invalid frozen SHA-256"
done
[[ "$EXPECTED_PROCESSOR_CONFIG_SHA256" =~ ^[0-9a-f]{64}$ ]] \
  || fail "invalid processor-config SHA-256"
[[ "$EXPECTED_VALIDATION_SHA256" == 1553739496b3c209988de56a4ccf574d407379b6b7937ddfafacbe66925069fe ]] \
  || fail "validation contract drift"
[[ ! -e "$RUN_DIR" ]] || fail "RUN_DIR must be absent"
mkdir -p "$RUN_DIR"/{raw,logs,stages,server-logs}
date -u +%FT%TZ > "$RUN_DIR/stages/00_start"

test -x "$SGLANG_ENV/bin/python" || fail "frozen SGLang Python missing"
for input_path in "$MODEL_DIR" "$MODEL_ARTIFACT_LEDGER_FILE" \
  "$MODEL_WEIGHT_LEDGER_FILE" "$VALIDATION_DATA" "$CODE_LEDGER_FILE" "$PROTOCOL_FILE" \
  "$ENV_FREEZE_FILE"; do
  test -e "$input_path" || fail "frozen input missing: $input_path"
done
test -f "$PROCESSOR_CONFIG_FILE" || fail "frozen processor config missing"

[[ "$(sha256sum "$CODE_LEDGER_FILE" | awk '{print $1}')" == "$EXPECTED_CODE_LEDGER_SHA256" ]] \
  || fail "code ledger raw SHA drift"
[[ "$(sha256sum "$PROTOCOL_FILE" | awk '{print $1}')" == "$EXPECTED_PROTOCOL_SHA256" ]] \
  || fail "protocol raw SHA drift"
[[ "$(sha256sum "$MODEL_ARTIFACT_LEDGER_FILE" | awk '{print $1}')" == "$EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256" ]] \
  || fail "model artifact ledger raw SHA drift"
[[ "$(sha256sum "$MODEL_WEIGHT_LEDGER_FILE" | awk '{print $1}')" == "$EXPECTED_MODEL_WEIGHT_LEDGER_SHA256" ]] \
  || fail "model weight ledger raw SHA drift"
[[ "$(sha256sum "$VALIDATION_DATA" | awk '{print $1}')" == "$EXPECTED_VALIDATION_SHA256" ]] \
  || fail "validation raw SHA drift"
[[ "$(sha256sum "$ENV_FREEZE_FILE" | awk '{print $1}')" == "$EXPECTED_ENV_FREEZE_SHA256" ]] \
  || fail "environment freeze raw SHA drift"
[[ "$(sha256sum "$PROCESSOR_CONFIG_FILE" | awk '{print $1}')" == "$EXPECTED_PROCESSOR_CONFIG_SHA256" ]] \
  || fail "processor-config raw SHA drift"

cp "$CODE_LEDGER_FILE" "$RUN_DIR/code.sha256"
cp "$PROTOCOL_FILE" "$RUN_DIR/protocol.json"
cp "$MODEL_ARTIFACT_LEDGER_FILE" "$RUN_DIR/model-artifacts.sha256"
cp "$MODEL_WEIGHT_LEDGER_FILE" "$RUN_DIR/model-weights.sha256"
cp "$ENV_FREEZE_FILE" "$RUN_DIR/environment-freeze.txt"
cp "$PROCESSOR_CONFIG_FILE" "$RUN_DIR/qwen35-sglang-preprocessor-config.json"
(cd / && sha256sum -c "$RUN_DIR/code.sha256") > "$RUN_DIR/logs/code-check.log"

"$SGLANG_ENV/bin/python" - <<'PY' > "$RUN_DIR/logs/cuda-stack-check.log"
import importlib.metadata
import json
import torch
import torchaudio
import torchvision

observed = {
    "sglang": importlib.metadata.version("sglang"),
    "sglang_kernel": importlib.metadata.version("sglang-kernel"),
    "sgl_deep_gemm": importlib.metadata.version("sgl-deep-gemm"),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "torchaudio": torchaudio.__version__,
    "torchvision": torchvision.__version__,
}
expected = {
    "sglang": "0.5.17",
    "sglang_kernel": "0.4.5+cu129",
    "sgl_deep_gemm": "0.1.5.post1+cu129",
    "torch": "2.11.0+cu129",
    "torch_cuda": "12.9",
    "torchaudio": "2.11.0+cu129",
    "torchvision": "0.26.0+cu129",
}
print(json.dumps(observed, sort_keys=True, separators=(",", ":")))
if observed != expected:
    raise SystemExit(f"CUDA stack drift: expected={expected!r}, observed={observed!r}")
PY
set +e
"$SGLANG_ENV/bin/python" -m pip check > "$RUN_DIR/logs/pip-check.log" 2>&1
printf '%s\n' "$?" > "$RUN_DIR/logs/pip-check.exit"
set -e
"$SGLANG_ENV/bin/python" -m pip freeze --all | sort > "$RUN_DIR/logs/environment-freeze.live.txt"
cmp "$ENV_FREEZE_FILE" "$RUN_DIR/logs/environment-freeze.live.txt" \
  || fail "live Python environment differs from frozen package set"

PYTHONPATH="$CODE_DIR" "$SGLANG_ENV/bin/python" -m py_compile \
  "$CODE_DIR/run_related_work_serving_baseline.py" \
  "$CODE_DIR/test_run_related_work_serving_baseline.py"
PYTHONPATH="$CODE_DIR" "$SGLANG_ENV/bin/python" -m unittest -q \
  test_run_related_work_serving_baseline \
  > "$RUN_DIR/logs/focused-tests.log" 2>&1

"$SGLANG_ENV/bin/python" -m sglang.launch_server --help \
  > "$RUN_DIR/logs/sglang-serve-help.txt" 2>&1
for flag in --disable-radix-cache --mamba-radix-cache-strategy --page-size \
  --enable-cache-report --enable-metrics --disable-cuda-graph; do
  grep -q -- "$flag" "$RUN_DIR/logs/sglang-serve-help.txt" \
    || fail "pinned SGLang lacks required flag: $flag"
done

(cd "$MODEL_DIR" && sha256sum -c "$RUN_DIR/model-artifacts.sha256") \
  > "$RUN_DIR/logs/model-artifact-preflight.log"
(cd "$MODEL_DIR" && sha256sum -c "$RUN_DIR/model-weights.sha256") \
  > "$RUN_DIR/logs/model-weight-preflight.log"

# The frozen weight view intentionally contains only model/tokenizer assets.
# SGLang's official Qwen3.5 multimodal wrapper also asks AutoProcessor for the
# upstream 390-byte image-processor descriptor even for text-only requests.
# Build a run-private small-file view; never mutate the frozen model directory.
PROCESSOR_VIEW="$RUN_DIR/processor-view"
mkdir -p "$PROCESSOR_VIEW"
for name in config.json generation_config.json tokenizer_config.json \
  vocab.json merges.txt chat_template.jinja; do
  test -f "$MODEL_DIR/$name" || fail "processor-view input missing: $name"
  cp "$MODEL_DIR/$name" "$PROCESSOR_VIEW/$name"
done
cp "$PROCESSOR_CONFIG_FILE" "$PROCESSOR_VIEW/preprocessor_config.json"
find "$PROCESSOR_VIEW" -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum \
  > "$RUN_DIR/processor-view.sha256"

"$SGLANG_ENV/bin/python" - <<'PY' > "$RUN_DIR/environment.json"
import json, platform
import sglang, torch, transformers
print(json.dumps({
  "python": platform.python_version(),
  "sglang": sglang.__version__,
  "torch": torch.__version__,
  "transformers": transformers.__version__,
}, sort_keys=True, separators=(",", ":")))
PY

mapfile -t GPU_UUIDS < <(nvidia-smi --query-gpu=uuid --format=csv,noheader | sed 's/[[:space:]]//g')
mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader | sed 's/[[:space:]]*$//')
[[ ${#GPU_UUIDS[@]} -eq 8 ]] || fail "formal node must expose exactly eight GPUs"
[[ $(printf '%s\n' "${GPU_UUIDS[@]}" | sort -u | wc -l) -eq 8 ]] || fail "GPU UUIDs not unique"
for gpu_name in "${GPU_NAMES[@]}"; do
  [[ "$gpu_name" == "NVIDIA H20-3e" ]] || fail "formal GPU name drift: $gpu_name"
done
"$SGLANG_ENV/bin/python" - "${GPU_UUIDS[@]}" <<'PY' > "$RUN_DIR/gpu-assignment.json"
import json, sys
print(json.dumps({"schema":"related-baseline-gpu-assignment-v1",
                  "uuids":sys.argv[1:]}, sort_keys=True, separators=(",", ":")))
PY
date -u +%FT%TZ > "$RUN_DIR/stages/10_preflight_complete"

wait_ready() {
  local rank=$1
  local port=$((BASE_PORT + rank))
  local pid=${SERVER_PIDS[$rank]}
  local attempt
  for attempt in $(seq 1 240); do
    if ! kill -0 "$pid" 2>/dev/null; then
      fail "server rank $rank exited during startup"
    fi
    if curl -fsS --max-time 3 "http://127.0.0.1:${port}/model_info" >/dev/null 2>&1; then
      return 0
    fi
    sleep 5
  done
  fail "server rank $rank readiness timeout"
}

run_phase() {
  local phase=$1
  local cache_args=()
  local rank port
  SERVER_PIDS=()
  if [[ "$phase" == cache_off ]]; then
    cache_args=(--disable-radix-cache)
  fi

  for rank in $(seq 0 7); do
    port=$((BASE_PORT + rank))
    setsid env CUDA_VISIBLE_DEVICES="${GPU_UUIDS[$rank]}" PYTHONUNBUFFERED=1 \
      "$SGLANG_ENV/bin/python" -m sglang.launch_server \
      --model-path "$MODEL_DIR" --served-model-name "$SERVER_NAME" \
      --tokenizer-path "$PROCESSOR_VIEW" \
      --host 127.0.0.1 --port "$port" --tp-size 1 --dtype bfloat16 \
      --context-length 8192 --max-running-requests 1 --max-total-tokens 8192 \
      --chunked-prefill-size 8192 --mem-fraction-static 0.82 \
      --random-seed $((20260820 + rank)) --disable-cuda-graph \
      --mamba-radix-cache-strategy extra_buffer --page-size 64 \
      --enable-cache-report --enable-metrics "${cache_args[@]}" \
      > "$RUN_DIR/server-logs/${phase}-rank-${rank}.log" 2>&1 &
    SERVER_PIDS[$rank]=$!
  done
  for rank in $(seq 0 7); do wait_ready "$rank"; done

  local client_pids=()
  for rank in $(seq 0 7); do
    port=$((BASE_PORT + rank))
    PYTHONPATH="$CODE_DIR" "$SGLANG_ENV/bin/python" \
      "$CODE_DIR/run_related_work_serving_baseline.py" \
      --stage client --system "$SYSTEM_NAME" \
      --phase "$phase" --rank "$rank" --world-size 8 \
      --model "$MODEL_DIR" --data "$VALIDATION_DATA" \
      --expected-data-sha256 "$EXPECTED_VALIDATION_SHA256" \
      --base-url "http://127.0.0.1:${port}" \
      --served-model-name "$SERVER_NAME" --max-new-tokens 32 \
      --output "$RUN_DIR/raw/${phase}-rank-${rank}.json" \
      > "$RUN_DIR/logs/client-${phase}-rank-${rank}.log" 2>&1 &
    client_pids[$rank]=$!
  done
  for rank in $(seq 0 7); do
    wait "${client_pids[$rank]}" || fail "client $phase rank $rank failed"
  done
  cleanup_servers
  date -u +%FT%TZ > "$RUN_DIR/stages/20_${phase}_complete"
}

run_phase cache_off
run_phase cache_on

PYTHONPATH="$CODE_DIR" "$SGLANG_ENV/bin/python" \
  "$CODE_DIR/run_related_work_serving_baseline.py" \
  --stage aggregate --system "$SYSTEM_NAME" --input-dir "$RUN_DIR/raw" \
  --output "$RUN_DIR/summary.json" \
  > "$RUN_DIR/logs/aggregate.log" 2>&1
date -u +%FT%TZ > "$RUN_DIR/stages/30_aggregate_complete"

[[ "$(sha256sum "$VALIDATION_DATA" | awk '{print $1}')" == "$EXPECTED_VALIDATION_SHA256" ]] \
  || fail "terminal validation drift"
[[ "$(sha256sum "$MODEL_WEIGHT_LEDGER_FILE" | awk '{print $1}')" == "$EXPECTED_MODEL_WEIGHT_LEDGER_SHA256" ]] \
  || fail "terminal model-ledger drift"
[[ "$(sha256sum "$MODEL_ARTIFACT_LEDGER_FILE" | awk '{print $1}')" == "$EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256" ]] \
  || fail "terminal model-artifact-ledger drift"
(cd / && sha256sum -c "$RUN_DIR/code.sha256") >> "$RUN_DIR/logs/code-check.log"
find "$RUN_DIR" -type f ! -name 'artifact-ledger.sha256' ! -path '*/stages/COMPLETE' \
  -print0 | sort -z | xargs -0 sha256sum > "$RUN_DIR/artifact-ledger.sha256"
sha256sum -c "$RUN_DIR/artifact-ledger.sha256" > "$RUN_DIR/logs/artifact-check.log"
date -u +%FT%TZ > "$RUN_DIR/stages/COMPLETE"
trap - EXIT INT TERM
