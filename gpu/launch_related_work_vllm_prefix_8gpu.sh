#!/usr/bin/env bash
set -Eeuo pipefail
export LC_ALL=C

CODE_DIR=${CODE_DIR:?set CODE_DIR}
MODEL_DIR=${MODEL_DIR:?set MODEL_DIR}
MODEL_ARTIFACT_LEDGER_FILE=${MODEL_ARTIFACT_LEDGER_FILE:?set MODEL_ARTIFACT_LEDGER_FILE}
MODEL_WEIGHT_LEDGER_FILE=${MODEL_WEIGHT_LEDGER_FILE:?set MODEL_WEIGHT_LEDGER_FILE}
VALIDATION_DATA=${VALIDATION_DATA:?set VALIDATION_DATA}
ENV_DIR=${ENV_DIR:?set ENV_DIR}
RUN_DIR=${RUN_DIR:?set RUN_DIR}
CODE_LEDGER_FILE=${CODE_LEDGER_FILE:?set CODE_LEDGER_FILE}
PROTOCOL_FILE=${PROTOCOL_FILE:?set PROTOCOL_FILE}
EXPECTED_CODE_LEDGER_SHA256=${EXPECTED_CODE_LEDGER_SHA256:?set EXPECTED_CODE_LEDGER_SHA256}
EXPECTED_PROTOCOL_SHA256=${EXPECTED_PROTOCOL_SHA256:?set EXPECTED_PROTOCOL_SHA256}
EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256=${EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256:?set EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256}
EXPECTED_MODEL_WEIGHT_LEDGER_SHA256=${EXPECTED_MODEL_WEIGHT_LEDGER_SHA256:?set EXPECTED_MODEL_WEIGHT_LEDGER_SHA256}
EXPECTED_VALIDATION_SHA256=${EXPECTED_VALIDATION_SHA256:?set EXPECTED_VALIDATION_SHA256}

SERVER_NAME=qwen35-related-baseline
SYSTEM_NAME=vllm-0.26-qwen35-prefix-align
WORLD_SIZE=8
BASE_PORT=18400
SERVER_PIDS=()

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
        tail -n 160 "$log_file" >&2 || true
      done < <(find "$RUN_DIR/logs" -maxdepth 1 -type f -print0 | sort -z)
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
  "$EXPECTED_VALIDATION_SHA256"; do
  [[ "$digest" =~ ^[0-9a-f]{64}$ ]] || fail "invalid frozen SHA-256"
done
[[ "$EXPECTED_VALIDATION_SHA256" == 1553739496b3c209988de56a4ccf574d407379b6b7937ddfafacbe66925069fe ]] \
  || fail "validation contract drift"
[[ ! -e "$RUN_DIR" ]] || fail "RUN_DIR must be absent"
mkdir -p "$RUN_DIR"/{raw,logs,stages,server-logs}
date -u +%FT%TZ > "$RUN_DIR/stages/00_start"

test -x "$ENV_DIR/bin/python" || fail "formal Python missing"
test -x "$ENV_DIR/bin/vllm" || fail "vLLM executable missing"
for path in "$MODEL_DIR" "$MODEL_ARTIFACT_LEDGER_FILE" "$MODEL_WEIGHT_LEDGER_FILE" "$VALIDATION_DATA" \
  "$CODE_LEDGER_FILE" "$PROTOCOL_FILE"; do
  test -e "$path" || fail "frozen input missing: $path"
done

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

cp "$CODE_LEDGER_FILE" "$RUN_DIR/code.sha256"
cp "$PROTOCOL_FILE" "$RUN_DIR/protocol.json"
cp "$MODEL_ARTIFACT_LEDGER_FILE" "$RUN_DIR/model-artifacts.sha256"
cp "$MODEL_WEIGHT_LEDGER_FILE" "$RUN_DIR/model-weights.sha256"
(cd / && sha256sum -c "$RUN_DIR/code.sha256") > "$RUN_DIR/logs/code-check.log"
PYTHONPATH="$CODE_DIR" "$ENV_DIR/bin/python" -m py_compile \
  "$CODE_DIR/run_related_work_serving_baseline.py" \
  "$CODE_DIR/test_run_related_work_serving_baseline.py"
PYTHONPATH="$CODE_DIR" "$ENV_DIR/bin/python" -m unittest -q \
  test_run_related_work_serving_baseline \
  > "$RUN_DIR/logs/focused-tests.log" 2>&1

"$ENV_DIR/bin/vllm" serve --help=all > "$RUN_DIR/logs/vllm-serve-help.txt"
for flag in --enable-prefix-caching --no-enable-prefix-caching --mamba-cache-mode \
  --enable-chunked-prefill --block-size --language-model-only; do
  grep -q -- "$flag" "$RUN_DIR/logs/vllm-serve-help.txt" \
    || fail "pinned vLLM lacks required flag: $flag"
done

(cd "$MODEL_DIR" && sha256sum -c "$RUN_DIR/model-artifacts.sha256") \
  > "$RUN_DIR/logs/model-artifact-preflight.log"
(cd "$MODEL_DIR" && sha256sum -c "$RUN_DIR/model-weights.sha256") \
  > "$RUN_DIR/logs/model-weight-preflight.log"

"$ENV_DIR/bin/python" - <<'PY' > "$RUN_DIR/environment.json"
import json, platform
import torch, transformers, vllm
print(json.dumps({
  "python": platform.python_version(),
  "torch": torch.__version__,
  "transformers": transformers.__version__,
  "vllm": vllm.__version__,
}, sort_keys=True, separators=(",", ":")))
PY

mapfile -t GPU_UUIDS < <(nvidia-smi --query-gpu=uuid --format=csv,noheader | sed 's/[[:space:]]//g')
mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader | sed 's/[[:space:]]*$//')
[[ ${#GPU_UUIDS[@]} -eq 8 ]] || fail "formal node must expose exactly eight GPUs"
[[ $(printf '%s\n' "${GPU_UUIDS[@]}" | sort -u | wc -l) -eq 8 ]] || fail "GPU UUIDs not unique"
for name in "${GPU_NAMES[@]}"; do
  [[ "$name" == "NVIDIA H20-3e" ]] || fail "formal GPU name drift: $name"
done
"$ENV_DIR/bin/python" - "${GPU_UUIDS[@]}" <<'PY' > "$RUN_DIR/gpu-assignment.json"
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
  for attempt in $(seq 1 180); do
    if ! kill -0 "$pid" 2>/dev/null; then
      fail "server rank $rank exited during startup"
    fi
    if curl -fsS --max-time 3 "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
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
  if [[ "$phase" == cache_on ]]; then
    cache_args=(--enable-prefix-caching --mamba-cache-mode align)
  else
    cache_args=(--no-enable-prefix-caching)
  fi

  for rank in $(seq 0 7); do
    port=$((BASE_PORT + rank))
    setsid env CUDA_VISIBLE_DEVICES="${GPU_UUIDS[$rank]}" \
      PYTHONUNBUFFERED=1 "$ENV_DIR/bin/vllm" serve "$MODEL_DIR" \
      --served-model-name "$SERVER_NAME" \
      --host 127.0.0.1 --port "$port" \
      --tensor-parallel-size 1 --dtype bfloat16 --language-model-only \
      --max-model-len 8192 --max-num-seqs 1 --max-num-batched-tokens 8192 \
      --gpu-memory-utilization 0.85 --enforce-eager \
      --enable-chunked-prefill "${cache_args[@]}" \
      --generation-config vllm --seed $((20260820 + rank)) \
      > "$RUN_DIR/server-logs/${phase}-rank-${rank}.log" 2>&1 &
    SERVER_PIDS[$rank]=$!
  done
  for rank in $(seq 0 7); do wait_ready "$rank"; done

  local client_pids=()
  for rank in $(seq 0 7); do
    port=$((BASE_PORT + rank))
    PYTHONPATH="$CODE_DIR" "$ENV_DIR/bin/python" \
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

PYTHONPATH="$CODE_DIR" "$ENV_DIR/bin/python" \
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
