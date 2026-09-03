#!/usr/bin/env bash
set -Eeuo pipefail
export LC_ALL=C
export PYTHONDONTWRITEBYTECODE=1

CODE_DIR=${CODE_DIR:?set CODE_DIR}
MODEL_DIR=${MODEL_DIR:?set MODEL_DIR}
VALIDATION_DATA=${VALIDATION_DATA:?set VALIDATION_DATA}
ENV_DIR=${ENV_DIR:?set ENV_DIR}
RUN_DIR=${RUN_DIR:?set RUN_DIR}
CODE_LEDGER_FILE=${CODE_LEDGER_FILE:?set CODE_LEDGER_FILE}
MODEL_ARTIFACT_LEDGER_FILE=${MODEL_ARTIFACT_LEDGER_FILE:?set MODEL_ARTIFACT_LEDGER_FILE}
MODEL_WEIGHT_LEDGER_FILE=${MODEL_WEIGHT_LEDGER_FILE:?set MODEL_WEIGHT_LEDGER_FILE}
VLLM_SUMMARY=${VLLM_SUMMARY:?set VLLM_SUMMARY}
SGLANG_SUMMARY=${SGLANG_SUMMARY:?set SGLANG_SUMMARY}
EXPECTED_CODE_LEDGER_SHA256=${EXPECTED_CODE_LEDGER_SHA256:?set EXPECTED_CODE_LEDGER_SHA256}
EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256=${EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256:?set EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256}
EXPECTED_MODEL_WEIGHT_LEDGER_SHA256=${EXPECTED_MODEL_WEIGHT_LEDGER_SHA256:?set EXPECTED_MODEL_WEIGHT_LEDGER_SHA256}
EXPECTED_VALIDATION_SHA256=${EXPECTED_VALIDATION_SHA256:?set EXPECTED_VALIDATION_SHA256}
EXPECTED_VLLM_SUMMARY_SHA256=${EXPECTED_VLLM_SUMMARY_SHA256:?set EXPECTED_VLLM_SUMMARY_SHA256}
EXPECTED_SGLANG_SUMMARY_SHA256=${EXPECTED_SGLANG_SUMMARY_SHA256:?set EXPECTED_SGLANG_SUMMARY_SHA256}

WORLD_SIZE=8
BASE_PORT=18600
SERVER_PIDS=()
CONFIGS=(
  qcomem-d7-r8-a8-l8
  full-prefix-q16
  qcomem-d7-r4-a4-l8
  qcomem-d7-r16-a16-l16
  qcomem-d7-mixed
  qcomem-d7-r4-a4-l4
)

fail() {
  printf '%s\n' "$1" >&2
  mkdir -p "$RUN_DIR/stages" 2>/dev/null || true
  date -u +%FT%TZ > "$RUN_DIR/stages/FAILED" 2>/dev/null || true
  exit 1
}

cleanup() {
  local pid
  for pid in "${SERVER_PIDS[@]:-}"; do
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  done
  for pid in "${SERVER_PIDS[@]:-}"; do
    wait "$pid" 2>/dev/null || true
  done
}

on_exit() {
  local rc=$?
  trap - EXIT INT TERM
  cleanup
  if (( rc != 0 )); then
    printf 'launcher_exit_code=%d\n' "$rc" >&2
    find "$RUN_DIR/logs" "$RUN_DIR/server-logs" -maxdepth 1 -type f -print0 2>/dev/null \
      | sort -z | while IFS= read -r -d '' path; do
          printf '\n===== %s =====\n' "$path" >&2
          tail -n 120 "$path" >&2 || true
        done
  fi
  exit "$rc"
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

for digest in "$EXPECTED_CODE_LEDGER_SHA256" \
  "$EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256" "$EXPECTED_MODEL_WEIGHT_LEDGER_SHA256" \
  "$EXPECTED_VALIDATION_SHA256" "$EXPECTED_VLLM_SUMMARY_SHA256" \
  "$EXPECTED_SGLANG_SUMMARY_SHA256"; do
  [[ "$digest" =~ ^[0-9a-f]{64}$ ]] || fail "invalid frozen SHA-256"
done
[[ "$EXPECTED_VALIDATION_SHA256" == "1553739496b3c209988de56a4ccf574d407379b6b7937ddfafacbe66925069fe" ]] \
  || fail "validation SHA contract drift"
[[ ! -e "$RUN_DIR" ]] || fail "RUN_DIR must be absent"
mkdir -p "$RUN_DIR"/{raw,logs,server-logs,stages}
date -u +%FT%TZ > "$RUN_DIR/stages/00_start"

for path in "$CODE_DIR" "$MODEL_DIR" "$VALIDATION_DATA" "$ENV_DIR/bin/python" \
  "$CODE_LEDGER_FILE" "$MODEL_ARTIFACT_LEDGER_FILE" "$MODEL_WEIGHT_LEDGER_FILE" \
  "$VLLM_SUMMARY" "$SGLANG_SUMMARY"; do
  [[ -e "$path" ]] || fail "missing frozen input: $path"
done
[[ "$(sha256sum "$CODE_LEDGER_FILE" | awk '{print $1}')" == "$EXPECTED_CODE_LEDGER_SHA256" ]] \
  || fail "code ledger raw SHA drift"
[[ "$(sha256sum "$MODEL_ARTIFACT_LEDGER_FILE" | awk '{print $1}')" == "$EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256" ]] \
  || fail "model artifact ledger raw SHA drift"
[[ "$(sha256sum "$MODEL_WEIGHT_LEDGER_FILE" | awk '{print $1}')" == "$EXPECTED_MODEL_WEIGHT_LEDGER_SHA256" ]] \
  || fail "model weight ledger raw SHA drift"
[[ "$(sha256sum "$VALIDATION_DATA" | awk '{print $1}')" == "$EXPECTED_VALIDATION_SHA256" ]] \
  || fail "validation data SHA drift"
[[ "$(sha256sum "$VLLM_SUMMARY" | awk '{print $1}')" == "$EXPECTED_VLLM_SUMMARY_SHA256" ]] \
  || fail "vLLM imported summary SHA drift"
[[ "$(sha256sum "$SGLANG_SUMMARY" | awk '{print $1}')" == "$EXPECTED_SGLANG_SUMMARY_SHA256" ]] \
  || fail "SGLang imported summary SHA drift"
cp "$CODE_LEDGER_FILE" "$RUN_DIR/code.sha256"
cp "$MODEL_ARTIFACT_LEDGER_FILE" "$RUN_DIR/model-artifacts.sha256"
cp "$MODEL_WEIGHT_LEDGER_FILE" "$RUN_DIR/model-weights.sha256"
(cd / && sha256sum -c "$RUN_DIR/code.sha256") > "$RUN_DIR/logs/code-check.log"
PYTHONPATH="$CODE_DIR" "$ENV_DIR/bin/python" -B -m py_compile \
  "$CODE_DIR/run_qcomem_streaming_related_compare.py" \
  "$CODE_DIR/test_run_qcomem_streaming_related_compare.py"
PYTHONPATH="$CODE_DIR" "$ENV_DIR/bin/python" -B -m unittest -q \
  test_run_qcomem_streaming_related_compare \
  > "$RUN_DIR/logs/focused-tests.log" 2>&1

(cd "$MODEL_DIR" && sha256sum -c "$RUN_DIR/model-artifacts.sha256") \
  > "$RUN_DIR/logs/model-artifact-preflight.log"
(cd "$MODEL_DIR" && sha256sum -c "$RUN_DIR/model-weights.sha256") \
  > "$RUN_DIR/logs/model-weight-preflight.log"

mapfile -t GPU_UUIDS < <(nvidia-smi --query-gpu=uuid --format=csv,noheader | sed 's/[[:space:]]//g')
mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader | sed 's/[[:space:]]*$//')
[[ ${#GPU_UUIDS[@]} -eq 8 ]] || fail "formal node must expose eight GPUs"
[[ $(printf '%s\n' "${GPU_UUIDS[@]}" | sort -u | wc -l) -eq 8 ]] || fail "GPU UUIDs not unique"
for name in "${GPU_NAMES[@]}"; do
  [[ "$name" == "NVIDIA H20-3e" ]] || fail "GPU name drift: $name"
done
"$ENV_DIR/bin/python" -B - "${GPU_UUIDS[@]}" <<'PY' > "$RUN_DIR/gpu-assignment.json"
import json, sys
print(json.dumps({"schema":"comem-related-direct-gpu-assignment-v1","uuids":sys.argv[1:]},sort_keys=True,separators=(",",":")))
PY
date -u +%FT%TZ > "$RUN_DIR/stages/10_preflight_complete"

wait_ready() {
  local rank=$1 port=$((BASE_PORT + rank)) pid=${SERVER_PIDS[$rank]}
  local attempt
  for attempt in $(seq 1 240); do
    kill -0 "$pid" 2>/dev/null || fail "server rank $rank exited during startup"
    if curl -fsS --max-time 3 "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 5
  done
  fail "server rank $rank readiness timeout"
}

for rank in $(seq 0 7); do
  port=$((BASE_PORT + rank))
  setsid env CUDA_VISIBLE_DEVICES="${GPU_UUIDS[$rank]}" PYTHONPATH="$CODE_DIR" \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    "$ENV_DIR/bin/python" -B "$CODE_DIR/run_qcomem_streaming_related_compare.py" \
      --stage server --model "$MODEL_DIR" --data "$VALIDATION_DATA" \
      --rank "$rank" --world-size "$WORLD_SIZE" --port "$port" \
      > "$RUN_DIR/server-logs/rank-${rank}.log" 2>&1 &
  SERVER_PIDS[$rank]=$!
done
for rank in $(seq 0 7); do wait_ready "$rank"; done
date -u +%FT%TZ > "$RUN_DIR/stages/20_servers_ready"

run_cell() {
  local config=$1 phase=$2 rank port
  local pids=()
  for rank in $(seq 0 7); do
    port=$((BASE_PORT + rank))
    PYTHONPATH="$CODE_DIR" "$ENV_DIR/bin/python" -B \
      "$CODE_DIR/run_qcomem_streaming_related_compare.py" \
      --stage client --model "$MODEL_DIR" --data "$VALIDATION_DATA" \
      --rank "$rank" --world-size "$WORLD_SIZE" --config "$config" --phase "$phase" \
      --base-url "http://127.0.0.1:${port}" --max-new-tokens 32 \
      --output "$RUN_DIR/raw/${config}-${phase}-rank-${rank}.json" \
      > "$RUN_DIR/logs/${config}-${phase}-rank-${rank}.log" 2>&1 &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
}

for config in "${CONFIGS[@]}"; do
  run_cell "$config" cache_off
  run_cell "$config" cache_on
done
[[ $(find "$RUN_DIR/raw" -maxdepth 1 -name '*.json' | wc -l) -eq 96 ]] \
  || fail "raw cell count drift"
date -u +%FT%TZ > "$RUN_DIR/stages/30_raw_complete"

PYTHONPATH="$CODE_DIR" "$ENV_DIR/bin/python" -B \
  "$CODE_DIR/run_qcomem_streaming_related_compare.py" \
  --stage aggregate --input-dir "$RUN_DIR/raw" \
  --vllm-summary "$VLLM_SUMMARY" --sglang-summary "$SGLANG_SUMMARY" \
  --output "$RUN_DIR/summary.json"
"$ENV_DIR/bin/python" -B - "$RUN_DIR/summary.json" <<'PY'
import json, sys
value=json.load(open(sys.argv[1]))
if not value.get("scientific_run_valid") or not value.get("hypothesis_passed"):
    raise SystemExit("direct-comparison aggregate did not pass")
PY
date -u +%FT%TZ > "$RUN_DIR/stages/40_aggregate_complete"

cleanup
SERVER_PIDS=()
(cd "$MODEL_DIR" && sha256sum -c "$RUN_DIR/model-artifacts.sha256") \
  > "$RUN_DIR/logs/model-artifact-terminal.log"
(cd "$MODEL_DIR" && sha256sum -c "$RUN_DIR/model-weights.sha256") \
  > "$RUN_DIR/logs/model-weight-terminal.log"
find "$RUN_DIR" -type f ! -name 'all-artifacts.sha256' -print0 | sort -z \
  | xargs -0 sha256sum > "$RUN_DIR/all-artifacts.sha256"
(cd / && sha256sum -c "$RUN_DIR/all-artifacts.sha256") \
  > "$RUN_DIR/logs/artifact-ledger-check.log"
date -u +%FT%TZ > "$RUN_DIR/stages/COMPLETED"
trap - EXIT INT TERM
exit 0
