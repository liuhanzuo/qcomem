#!/usr/bin/env bash
set -euo pipefail
umask 077

die() {
  printf '%s\n' "ERROR: $*" >&2
  if declare -F cleanup_servers >/dev/null 2>&1; then
    cleanup_servers
  fi
  if [[ -n "${RUN_DIR:-}" && -d "${RUN_DIR:-}" ]]; then
    printf '%s\n' 1 > "$RUN_DIR/FAILED"
  fi
  exit 1
}

PYTHON_BIN=${PYTHON_BIN:-/tmp/round25-hypic-env/venv/bin/python}
HYPIC_REPO=${HYPIC_REPO:-/tmp/HYPIC-98147c0}
CODE_DIR=${CODE_DIR:-/tmp/round25-hypic-formal-code}
MODEL_DIR=${MODEL_DIR:-/tmp/Qwen3.5-35B-A3B-hypic-model-view}
MODEL_WEIGHT_LEDGER=${MODEL_WEIGHT_LEDGER:-${MODEL_DIR}/model-weights.sha256}
MODEL_ARTIFACT_LEDGER=${MODEL_ARTIFACT_LEDGER:-${MODEL_DIR}/model-artifacts.sha256}
VALIDATION_DATA=${VALIDATION_DATA:-/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/data/qcomem-longbench-validation/longbench_validation.jsonl}
RUN_DIR=${RUN_DIR:-/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/runs/qcomem/hypic-same-protocol-20260821a}
BASE_PORT=${BASE_PORT:-33100}
SERVER_NAME=qwen35-hypic

CLIENT=${CODE_DIR}/run_hypic_same_protocol.py
HELPER=${CODE_DIR}/run_related_work_serving_baseline.py
TEST=${CODE_DIR}/test_run_hypic_same_protocol.py
STATIC_BUILDER=${CODE_DIR}/build_hypic_formal_static.py
LAUNCHER=${CODE_DIR}/launch_hypic_same_protocol_8gpu.sh
EXPECTED_COMMIT=98147c01909004e66d98bcb18b886927d41b0ee5
EXPECTED_DATA_SHA=1553739496b3c209988de56a4ccf574d407379b6b7937ddfafacbe66925069fe
EXPECTED_MODEL_LEDGER_SHA=8314a82c9188b9b817193e039b0b0eb0636b328512f19b0c12455853b7e20014
EXPECTED_MODEL_ARTIFACT_LEDGER_SHA=d78424684a27718b44616c80c788e7f8b726feed674fe015901b87de1993f7dd

for path in "$PYTHON_BIN" "$CLIENT" "$HELPER" "$TEST" "$STATIC_BUILDER" "$LAUNCHER" "$VALIDATION_DATA" "$MODEL_WEIGHT_LEDGER" "$MODEL_ARTIFACT_LEDGER"; do
  [[ -f "$path" ]] || die "missing file: $path"
done
[[ -d "$HYPIC_REPO/.git" && -d "$MODEL_DIR" ]] || die "missing repo/model directory"
[[ ! -e "$RUN_DIR" ]] || die "RUN_DIR already exists: $RUN_DIR"
[[ "$(git -C "$HYPIC_REPO" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || die "HYPIC commit drift"
[[ -z "$(git -C "$HYPIC_REPO" status --porcelain --untracked-files=all)" ]] || die "HYPIC source dirty or untracked"
[[ "$(sha256sum "$VALIDATION_DATA" | awk '{print $1}')" == "$EXPECTED_DATA_SHA" ]] || die "data SHA drift"
[[ "$(sha256sum "$MODEL_WEIGHT_LEDGER" | awk '{print $1}')" == "$EXPECTED_MODEL_LEDGER_SHA" ]] || die "model ledger SHA drift"
[[ "$(sha256sum "$MODEL_ARTIFACT_LEDGER" | awk '{print $1}')" == "$EXPECTED_MODEL_ARTIFACT_LEDGER_SHA" ]] || die "model artifact ledger SHA drift"

mapfile -t GPU_UUIDS < <(nvidia-smi --query-gpu=uuid --format=csv,noheader)
[[ ${#GPU_UUIDS[@]} -eq 8 ]] || die "expected eight GPUs"
[[ $(printf '%s\n' "${GPU_UUIDS[@]}" | sort -u | wc -l) -eq 8 ]] || die "duplicate GPU UUID"
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sed '/^$/d')" ]] || die "GPU compute process already active"

mkdir -p "$RUN_DIR"/{raw,receipts,server-logs,logs,commands,stages,caches}
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="$RUN_DIR/caches/pycache"
export TORCHINDUCTOR_CACHE_DIR="$RUN_DIR/caches/torchinductor"
export TRITON_CACHE_DIR="$RUN_DIR/caches/triton"
export XDG_CACHE_HOME="$RUN_DIR/caches/xdg"
export HF_HOME="$RUN_DIR/caches/huggingface"
export PIC_SEAM_SINK=8
unset SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK || true
date -u +%FT%TZ > "$RUN_DIR/stages/00_started"

"$PYTHON_BIN" -m py_compile "$CLIENT" "$HELPER" "$TEST" "$STATIC_BUILDER"
PYTHONPATH="$CODE_DIR" "$PYTHON_BIN" -m unittest -q test_run_hypic_same_protocol \
  > "$RUN_DIR/logs/focused-tests.log" 2>&1
date -u +%FT%TZ > "$RUN_DIR/stages/01_focused_tests_passed"

CUDA_VISIBLE_DEVICES="${GPU_UUIDS[0]}" PYTHON="$PYTHON_BIN" "$PYTHON_BIN" "$STATIC_BUILDER" \
  --stage build --repo "$HYPIC_REPO" --model "$MODEL_DIR" \
  --model-weight-ledger "$MODEL_WEIGHT_LEDGER" --model-artifact-ledger "$MODEL_ARTIFACT_LEDGER" \
  --data "$VALIDATION_DATA" \
  --client "$CLIENT" --helper "$HELPER" --test "$TEST" --launcher "$LAUNCHER" \
  --static-builder "$STATIC_BUILDER" --output-dir "$RUN_DIR/static"
date -u +%FT%TZ > "$RUN_DIR/stages/02_preregistered_before_outputs"

PYTHONPATH="$CODE_DIR" "$PYTHON_BIN" "$CLIENT" \
  --stage token_gate --model "$MODEL_DIR" --data "$VALIDATION_DATA" \
  --output "$RUN_DIR/static/token-identity.json"
date -u +%FT%TZ > "$RUN_DIR/stages/03_token_identity_passed"

declare -a SERVER_PIDS=()

cleanup_servers() {
  local pid attempt
  for pid in "${SERVER_PIDS[@]:-}"; do
    [[ -n "$pid" ]] || continue
    if kill -0 "$pid" 2>/dev/null; then
      kill -- "-$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
    fi
  done
  for pid in "${SERVER_PIDS[@]:-}"; do
    [[ -n "$pid" ]] || continue
    for attempt in $(seq 1 30); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "$pid" 2>/dev/null; then
      kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
    fi
    wait "$pid" 2>/dev/null || true
  done
  SERVER_PIDS=()
}

failure_trap() {
  local status=$?
  cleanup_servers
  rm -f "$RUN_DIR/COMPLETED" "$RUN_DIR/stages/99_done"
  printf '%s\n' "$status" > "$RUN_DIR/FAILED"
  exit "$status"
}
trap failure_trap ERR INT TERM

wait_ready() {
  local rank=$1
  local port=$((BASE_PORT + rank))
  local pid=${SERVER_PIDS[$rank]}
  local attempt
  for attempt in $(seq 1 240); do
    kill -0 "$pid" 2>/dev/null || die "server rank $rank exited during startup"
    if curl -fsS --max-time 3 "http://127.0.0.1:${port}/model_info" >/dev/null 2>&1; then
      return 0
    fi
    sleep 5
  done
  die "server rank $rank readiness timeout"
}

run_mode() {
  local mode=$1
  local mode_args=()
  local rank port command_file
  case "$mode" in
    full_recompute)
      mode_args=(--disable-radix-cache --mamba-radix-cache-strategy no_buffer)
      ;;
    prefix_cache)
      mode_args=(--mamba-radix-cache-strategy extra_buffer)
      ;;
    transition_rope_recompute)
      mode_args=(
        --page-size 1 --chunked-prefill-size -1
        --mamba-radix-cache-strategy no_buffer
        --pic-enable --pic-mode transition_rope_recompute
        --pic-separator-str '<<PIC_SEP>>'
      )
      ;;
    *) die "unknown mode: $mode" ;;
  esac
  SERVER_PIDS=()
  for rank in $(seq 0 7); do
    port=$((BASE_PORT + rank))
    command_file="$RUN_DIR/commands/${mode}-rank-${rank}.txt"
    mkdir -p "$RUN_DIR/caches/rank-${rank}"/{pycache,torchinductor,triton,xdg,huggingface}
    printf '%q ' \
      env CUDA_VISIBLE_DEVICES="${GPU_UUIDS[$rank]}" PIC_SEAM_SINK=8 \
      SGLANG_NUMA_BIND_V2=0 SGLANG_IS_FLASHINFER_AVAILABLE=0 \
      PYTHONPATH="$HYPIC_REPO/python:$CODE_DIR" PYTHONUNBUFFERED=1 \
      PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX="$RUN_DIR/caches/rank-${rank}/pycache" \
      TORCHINDUCTOR_CACHE_DIR="$RUN_DIR/caches/rank-${rank}/torchinductor" \
      TRITON_CACHE_DIR="$RUN_DIR/caches/rank-${rank}/triton" \
      XDG_CACHE_HOME="$RUN_DIR/caches/rank-${rank}/xdg" \
      HF_HOME="$RUN_DIR/caches/rank-${rank}/huggingface" \
      "$PYTHON_BIN" -m sglang.launch_server \
      --model-path "$MODEL_DIR" --served-model-name "$SERVER_NAME" \
      --host 127.0.0.1 --port "$port" --tp-size 1 --dtype bfloat16 \
      --context-length 8192 --max-running-requests 1 --max-total-tokens 8192 \
      --mem-fraction-static 0.80 --random-seed $((20260821 + rank)) \
      --sampling-backend pytorch \
      --disable-cuda-graph --disable-piecewise-cuda-graph --disable-overlap-schedule \
      --enable-cache-report "${mode_args[@]}" > "$command_file"
    printf '\n' >> "$command_file"
    pid_file="$RUN_DIR/server-logs/${mode}-rank-${rank}.pid"
    rm -f "$pid_file"
    setsid bash -c \
      'pid_file=$1; shift; printf "%s\n" "$$" > "$pid_file"; exec "$@"' \
      hypic-server "$pid_file" env \
      CUDA_VISIBLE_DEVICES="${GPU_UUIDS[$rank]}" PIC_SEAM_SINK=8 \
      SGLANG_NUMA_BIND_V2=0 SGLANG_IS_FLASHINFER_AVAILABLE=0 \
      PYTHONPATH="$HYPIC_REPO/python:$CODE_DIR" PYTHONUNBUFFERED=1 \
      PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX="$RUN_DIR/caches/rank-${rank}/pycache" \
      TORCHINDUCTOR_CACHE_DIR="$RUN_DIR/caches/rank-${rank}/torchinductor" \
      TRITON_CACHE_DIR="$RUN_DIR/caches/rank-${rank}/triton" \
      XDG_CACHE_HOME="$RUN_DIR/caches/rank-${rank}/xdg" \
      HF_HOME="$RUN_DIR/caches/rank-${rank}/huggingface" \
      "$PYTHON_BIN" -m sglang.launch_server \
      --model-path "$MODEL_DIR" --served-model-name "$SERVER_NAME" \
      --host 127.0.0.1 --port "$port" --tp-size 1 --dtype bfloat16 \
      --context-length 8192 --max-running-requests 1 --max-total-tokens 8192 \
      --mem-fraction-static 0.80 --random-seed $((20260821 + rank)) \
      --sampling-backend pytorch \
      --disable-cuda-graph --disable-piecewise-cuda-graph --disable-overlap-schedule \
      --enable-cache-report "${mode_args[@]}" \
      > "$RUN_DIR/server-logs/${mode}-rank-${rank}.log" 2>&1 &
    for _ in $(seq 1 100); do
      [[ -s "$pid_file" ]] && break
      sleep 0.05
    done
    [[ -s "$pid_file" ]] || die "server rank $rank did not publish its PID"
    SERVER_PIDS[$rank]=$(cat "$pid_file")
    [[ "${SERVER_PIDS[$rank]}" =~ ^[0-9]+$ ]] || die "invalid server PID for rank $rank"
    kill -0 "${SERVER_PIDS[$rank]}" 2>/dev/null || die "server rank $rank exited before readiness"
  done
  for rank in $(seq 0 7); do wait_ready "$rank"; done

  for rank in $(seq 0 7); do
    port=$((BASE_PORT + rank))
    CUDA_VISIBLE_DEVICES="${GPU_UUIDS[$rank]}" PIC_SEAM_SINK=8 \
      SGLANG_NUMA_BIND_V2=0 SGLANG_IS_FLASHINFER_AVAILABLE=0 \
      PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$HYPIC_REPO/python:$CODE_DIR" \
      "$PYTHON_BIN" "$CLIENT" --stage server_receipt \
      --mode "$mode" --rank "$rank" --expected-tp-size 1 \
      --model "$MODEL_DIR" --data "$VALIDATION_DATA" \
      --base-url "http://127.0.0.1:${port}" --hypic-repo "$HYPIC_REPO" \
      --source-ledger "$RUN_DIR/static/source-ledger.json" \
      --environment-ledger "$RUN_DIR/static/environment-ledger.json" \
      --preregistration "$RUN_DIR/static/preregistration.json" \
      --launch-command-file "$RUN_DIR/commands/${mode}-rank-${rank}.txt" \
      --model-weight-ledger "$MODEL_WEIGHT_LEDGER" \
      --model-artifact-ledger "$MODEL_ARTIFACT_LEDGER" \
      --server-pid "${SERVER_PIDS[$rank]}" \
      --expected-gpu-uuid "${GPU_UUIDS[$rank]}" \
      --output "$RUN_DIR/receipts/${mode}-rank-${rank}.json"
  done

  local -a client_pids=()
  for rank in $(seq 0 7); do
    port=$((BASE_PORT + rank))
    PYTHONPATH="$CODE_DIR" "$PYTHON_BIN" "$CLIENT" --stage client \
      --mode "$mode" --rank "$rank" --world-size 8 --expected-tp-size 1 \
      --model "$MODEL_DIR" --data "$VALIDATION_DATA" \
      --base-url "http://127.0.0.1:${port}" --served-model-name "$SERVER_NAME" \
      --max-new-tokens 32 --server-receipt "$RUN_DIR/receipts/${mode}-rank-${rank}.json" \
      --output "$RUN_DIR/raw/${mode}-rank-${rank}.json" \
      > "$RUN_DIR/logs/client-${mode}-rank-${rank}.log" 2>&1 &
    client_pids[$rank]=$!
  done
  for rank in $(seq 0 7); do
    wait "${client_pids[$rank]}" || die "client $mode rank $rank failed"
  done
  cleanup_servers
  date -u +%FT%TZ > "$RUN_DIR/stages/20_${mode}_complete"
}

run_mode full_recompute
run_mode prefix_cache
run_mode transition_rope_recompute

PYTHONPATH="$CODE_DIR" "$PYTHON_BIN" "$CLIENT" --stage aggregate \
  --input-dir "$RUN_DIR/raw" --output "$RUN_DIR/summary.json"
date -u +%FT%TZ > "$RUN_DIR/stages/30_aggregate_complete"

CUDA_VISIBLE_DEVICES="${GPU_UUIDS[0]}" PYTHON="$PYTHON_BIN" "$PYTHON_BIN" "$STATIC_BUILDER" \
  --stage verify --repo "$HYPIC_REPO" --model "$MODEL_DIR" \
  --model-weight-ledger "$MODEL_WEIGHT_LEDGER" --model-artifact-ledger "$MODEL_ARTIFACT_LEDGER" \
  --verify-model-bytes --data "$VALIDATION_DATA" \
  --client "$CLIENT" --helper "$HELPER" --test "$TEST" --launcher "$LAUNCHER" \
  --static-builder "$STATIC_BUILDER" --output-dir "$RUN_DIR/static" \
  --validation-output "$RUN_DIR/terminal-static-verification.json"

find "$RUN_DIR" -type f ! -name all-artifacts.sha256 ! -name COMPLETED ! -path '*/stages/99_done' -print0 \
  | sort -z | xargs -0 sha256sum > "$RUN_DIR/all-artifacts.sha256"
sha256sum -c "$RUN_DIR/all-artifacts.sha256" >/dev/null
date -u +%FT%TZ > "$RUN_DIR/stages/99_done.tmp"
mv "$RUN_DIR/stages/99_done.tmp" "$RUN_DIR/stages/99_done"
touch "$RUN_DIR/COMPLETED.tmp"
mv "$RUN_DIR/COMPLETED.tmp" "$RUN_DIR/COMPLETED"
trap - ERR INT TERM
