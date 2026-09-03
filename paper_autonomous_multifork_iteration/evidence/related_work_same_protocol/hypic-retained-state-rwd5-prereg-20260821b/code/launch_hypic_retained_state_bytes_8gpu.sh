#!/usr/bin/env bash
set -euo pipefail
umask 077

die() {
  printf '%s\n' "ERROR: $*" >&2
  if declare -F cleanup_servers >/dev/null 2>&1; then cleanup_servers; fi
  if [[ -n "${RUN_DIR:-}" && -d "${RUN_DIR:-}" ]]; then printf '%s\n' 1 > "$RUN_DIR/FAILED"; fi
  exit 1
}

PYTHON_BIN=${PYTHON_BIN:-/tmp/round25-hypic-env/venv/bin/python}
OFFICIAL_REPO=${OFFICIAL_REPO:-/tmp/HYPIC-98147c0}
INSTRUMENTED_REPO=${INSTRUMENTED_REPO:-/tmp/HYPIC-98147c0-rwd5-store}
FREEZE_ROOT=${FREEZE_ROOT:-/tmp/rwd5-hypic-store-freeze}
CODE_DIR=${CODE_DIR:-${FREEZE_ROOT}/code}
FREEZE_MANIFEST=${FREEZE_MANIFEST:-${FREEZE_ROOT}/SHA256SUMS}
EXPECTED_FREEZE_MANIFEST_SHA256=${EXPECTED_FREEZE_MANIFEST_SHA256:?supply externally audited frozen manifest SHA256}
MODEL_DIR=${MODEL_DIR:-/tmp/Qwen3.5-35B-A3B-hypic-model-view}
MODEL_WEIGHT_LEDGER=${MODEL_WEIGHT_LEDGER:-${MODEL_DIR}/model-weights.sha256}
MODEL_ARTIFACT_LEDGER=${MODEL_ARTIFACT_LEDGER:-${MODEL_DIR}/model-artifacts.sha256}
VALIDATION_DATA=${VALIDATION_DATA:-/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/data/qcomem-longbench-validation/longbench_validation.jsonl}
RUN_DIR=${RUN_DIR:-/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/runs/qcomem/hypic-retained-state-rwd5-20260821b}
BASE_PORT=${BASE_PORT:-33400}
SERVER_NAME=qwen35-hypic

CLIENT=${CODE_DIR}/run_hypic_retained_state_bytes.py
FORMAL_HELPER=${CODE_DIR}/run_hypic_same_protocol.py
FORMAL_STATIC_HELPER=${CODE_DIR}/build_hypic_formal_static.py
SERVING_HELPER=${CODE_DIR}/run_related_work_serving_baseline.py
RECEIPT_MODULE=${CODE_DIR}/hypic_retained_state_receipt.py
PATCH=${CODE_DIR}/hypic_retained_state_instrumentation.patch
RECEIPT_TEST=${CODE_DIR}/test_hypic_retained_state_receipt.py
REPLAY=${CODE_DIR}/replay_hypic_retained_state_bytes.py
STATIC_BUILDER=${CODE_DIR}/build_hypic_retained_state_static.py
LAUNCHER=${CODE_DIR}/launch_hypic_retained_state_bytes_8gpu.sh
EXPECTED_COMMIT=98147c01909004e66d98bcb18b886927d41b0ee5
EXPECTED_DATA_SHA=1553739496b3c209988de56a4ccf574d407379b6b7937ddfafacbe66925069fe
EXPECTED_MODEL_LEDGER_SHA=8314a82c9188b9b817193e039b0b0eb0636b328512f19b0c12455853b7e20014
EXPECTED_MODEL_ARTIFACT_LEDGER_SHA=d78424684a27718b44616c80c788e7f8b726feed674fe015901b87de1993f7dd

for path in "$PYTHON_BIN" "$CLIENT" "$FORMAL_HELPER" "$FORMAL_STATIC_HELPER" "$SERVING_HELPER" "$RECEIPT_MODULE" "$PATCH" "$RECEIPT_TEST" "$REPLAY" "$STATIC_BUILDER" "$LAUNCHER" "$VALIDATION_DATA" "$MODEL_WEIGHT_LEDGER" "$MODEL_ARTIFACT_LEDGER"; do
  [[ -f "$path" ]] || die "missing file: $path"
done
[[ -f "$FREEZE_MANIFEST" ]] || die "missing external frozen manifest"
[[ "$(sha256sum "$FREEZE_MANIFEST" | awk '{print $1}')" == "$EXPECTED_FREEZE_MANIFEST_SHA256" ]] || die "external frozen manifest SHA drift"
(cd "$FREEZE_ROOT" && sha256sum -c "$FREEZE_MANIFEST") || die "external frozen files drift"
[[ -d "$OFFICIAL_REPO/.git" && -d "$MODEL_DIR" ]] || die "missing repo/model"
[[ ! -e "$RUN_DIR" ]] || die "RUN_DIR already exists: $RUN_DIR"
[[ ! -e "$INSTRUMENTED_REPO" ]] || die "instrumented repo path already exists: $INSTRUMENTED_REPO"
[[ "$(git -C "$OFFICIAL_REPO" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || die "HYPIC commit drift"
[[ -z "$(git -C "$OFFICIAL_REPO" status --porcelain --untracked-files=all)" ]] || die "official HYPIC worktree dirty"
git -C "$OFFICIAL_REPO" apply --check "$PATCH" || die "instrumentation patch does not apply"
[[ "$(sha256sum "$VALIDATION_DATA" | awk '{print $1}')" == "$EXPECTED_DATA_SHA" ]] || die "data SHA drift"
[[ "$(sha256sum "$MODEL_WEIGHT_LEDGER" | awk '{print $1}')" == "$EXPECTED_MODEL_LEDGER_SHA" ]] || die "model ledger SHA drift"
[[ "$(sha256sum "$MODEL_ARTIFACT_LEDGER" | awk '{print $1}')" == "$EXPECTED_MODEL_ARTIFACT_LEDGER_SHA" ]] || die "model artifact ledger SHA drift"

mapfile -t GPU_UUIDS < <(nvidia-smi --query-gpu=uuid --format=csv,noheader)
[[ ${#GPU_UUIDS[@]} -eq 8 ]] || die "expected eight GPUs"
[[ $(printf '%s\n' "${GPU_UUIDS[@]}" | sort -u | wc -l) -eq 8 ]] || die "duplicate GPU UUID"
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sed '/^$/d')" ]] || die "GPU compute process already active"

cp -a "$OFFICIAL_REPO" "$INSTRUMENTED_REPO"
git -C "$INSTRUMENTED_REPO" apply "$PATCH"
cp "$RECEIPT_MODULE" "$INSTRUMENTED_REPO/python/sglang/srt/retained_state_receipt.py"
git -C "$INSTRUMENTED_REPO" apply --reverse --check "$PATCH" || die "instrumentation reverse check"
[[ "$(sha256sum "$INSTRUMENTED_REPO/python/sglang/srt/retained_state_receipt.py" | awk '{print $1}')" == "$(sha256sum "$RECEIPT_MODULE" | awk '{print $1}')" ]] || die "installed receipt module drift"

mkdir -p "$RUN_DIR"/{raw,targets,store-receipts,server-receipts,scheduler-workers,server-logs,logs,commands,stages,caches}
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="$RUN_DIR/caches/pycache"
export TORCHINDUCTOR_CACHE_DIR="$RUN_DIR/caches/torchinductor"
export TRITON_CACHE_DIR="$RUN_DIR/caches/triton"
export XDG_CACHE_HOME="$RUN_DIR/caches/xdg"
export HF_HOME="$RUN_DIR/caches/huggingface"
export PIC_SEAM_SINK=8
unset SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK || true
date -u +%FT%TZ > "$RUN_DIR/stages/00_started"

"$PYTHON_BIN" -m py_compile "$CLIENT" "$FORMAL_HELPER" "$SERVING_HELPER" "$RECEIPT_MODULE" "$RECEIPT_TEST" "$REPLAY" "$STATIC_BUILDER"
PYTHONPATH="$CODE_DIR" "$PYTHON_BIN" -m unittest -v test_hypic_retained_state_receipt \
  > "$RUN_DIR/logs/focused-tests.log" 2>&1
date -u +%FT%TZ > "$RUN_DIR/stages/01_focused_tests_passed"

COMMON_STATIC_ARGS=(
  --official-repo "$OFFICIAL_REPO" --instrumented-repo "$INSTRUMENTED_REPO"
  --model "$MODEL_DIR" --model-weight-ledger "$MODEL_WEIGHT_LEDGER"
  --model-artifact-ledger "$MODEL_ARTIFACT_LEDGER" --data "$VALIDATION_DATA"
  --client "$CLIENT" --formal-helper "$FORMAL_HELPER" --serving-helper "$SERVING_HELPER"
  --formal-static-helper "$FORMAL_STATIC_HELPER"
  --receipt-module "$RECEIPT_MODULE" --patch "$PATCH" --receipt-test "$RECEIPT_TEST"
  --replay "$REPLAY" --launcher "$LAUNCHER" --static-builder "$STATIC_BUILDER"
  --freeze-manifest "$FREEZE_MANIFEST"
  --expected-freeze-manifest-sha256 "$EXPECTED_FREEZE_MANIFEST_SHA256"
)
CUDA_VISIBLE_DEVICES="${GPU_UUIDS[0]}" PYTHON="$PYTHON_BIN" PYTHONPATH="$CODE_DIR" \
  "$PYTHON_BIN" "$STATIC_BUILDER" --stage build "${COMMON_STATIC_ARGS[@]}" \
  --output-dir "$RUN_DIR/static"
date -u +%FT%TZ > "$RUN_DIR/stages/02_preregistered_before_outputs"

declare -a SERVER_PIDS=()
cleanup_servers() {
  local pid attempt
  for pid in "${SERVER_PIDS[@]:-}"; do
    [[ -n "$pid" ]] || continue
    if kill -0 "$pid" 2>/dev/null; then kill -- "-$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true; fi
  done
  for pid in "${SERVER_PIDS[@]:-}"; do
    [[ -n "$pid" ]] || continue
    for attempt in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    if kill -0 "$pid" 2>/dev/null; then kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true; fi
    wait "$pid" 2>/dev/null || true
  done
  SERVER_PIDS=()
}
trap 'status=$?; cleanup_servers; rm -f "$RUN_DIR/COMPLETED"; printf "%s\n" "$status" > "$RUN_DIR/FAILED"; exit "$status"' ERR INT TERM

wait_ready() {
  local rank=$1 port=$((BASE_PORT + rank)) pid=${SERVER_PIDS[$rank]}
  for _ in $(seq 1 240); do
    kill -0 "$pid" 2>/dev/null || die "server rank $rank exited"
    if curl -fsS --max-time 3 "http://127.0.0.1:${port}/model_info" >/dev/null 2>&1; then return 0; fi
    sleep 5
  done
  die "server rank $rank readiness timeout"
}

run_mode() {
  local mode=$1 rank port pid_file target_file receipt_dir server_receipt worker_receipt
  local mode_args=()
  case "$mode" in
    prefix_cache) mode_args=(--mamba-radix-cache-strategy extra_buffer) ;;
    transition_rope_recompute)
      mode_args=(--page-size 1 --chunked-prefill-size -1 --mamba-radix-cache-strategy no_buffer --pic-enable --pic-mode transition_rope_recompute --pic-separator-str '<<PIC_SEP>>') ;;
    *) die "unapproved mode: $mode" ;;
  esac
  SERVER_PIDS=()
  for rank in $(seq 0 7); do
    port=$((BASE_PORT + rank))
    target_file="$RUN_DIR/targets/${mode}-rank-${rank}.json"
    receipt_dir="$RUN_DIR/store-receipts"
    server_receipt="$RUN_DIR/server-receipts/${mode}-rank-${rank}.json"
    worker_receipt="$RUN_DIR/scheduler-workers/${mode}-rank-${rank}.json"
    mkdir -p "$RUN_DIR/caches/${mode}-rank-${rank}"/{pycache,torchinductor,triton,xdg,huggingface}
    printf '%q ' env CUDA_VISIBLE_DEVICES="${GPU_UUIDS[$rank]}" PIC_SEAM_SINK=8 \
      FORKAUDIT_RWD5_TARGET_PATH="$target_file" FORKAUDIT_RWD5_RECEIPT_DIR="$receipt_dir" \
      FORKAUDIT_RWD5_WORKER_RECEIPT_PATH="$worker_receipt" FORKAUDIT_RWD5_SERVER_RECEIPT_PATH="$server_receipt" \
      FORKAUDIT_RWD5_PREREGISTRATION_PATH="$RUN_DIR/static/preregistration.json" \
      FORKAUDIT_RWD5_FREEZE_MANIFEST_PATH="$FREEZE_MANIFEST" FORKAUDIT_RWD5_FREEZE_MANIFEST_SHA256="$EXPECTED_FREEZE_MANIFEST_SHA256" \
      FORKAUDIT_RWD5_MODE="$mode" FORKAUDIT_RWD5_RANK="$rank" \
      PYTHONPATH="$INSTRUMENTED_REPO/python:$CODE_DIR" "$PYTHON_BIN" -m sglang.launch_server \
      --model-path "$MODEL_DIR" --served-model-name "$SERVER_NAME" --host 127.0.0.1 --port "$port" \
      --tp-size 1 --dtype bfloat16 --context-length 8192 --max-running-requests 1 --max-total-tokens 8192 \
      --mem-fraction-static 0.80 --random-seed $((20260821 + rank)) --sampling-backend pytorch \
      --disable-cuda-graph --disable-piecewise-cuda-graph --disable-overlap-schedule --enable-cache-report "${mode_args[@]}" \
      > "$RUN_DIR/commands/${mode}-rank-${rank}.txt"
    printf '\n' >> "$RUN_DIR/commands/${mode}-rank-${rank}.txt"
    pid_file="$RUN_DIR/server-logs/${mode}-rank-${rank}.pid"
    setsid bash -c 'pid_file=$1; shift; printf "%s\n" "$$" > "$pid_file"; export FORKAUDIT_RWD5_FRONTEND_PID=$$; exec "$@"' \
      rwd5-server "$pid_file" env CUDA_VISIBLE_DEVICES="${GPU_UUIDS[$rank]}" PIC_SEAM_SINK=8 \
      FORKAUDIT_RWD5_TARGET_PATH="$target_file" FORKAUDIT_RWD5_RECEIPT_DIR="$receipt_dir" \
      FORKAUDIT_RWD5_WORKER_RECEIPT_PATH="$worker_receipt" FORKAUDIT_RWD5_SERVER_RECEIPT_PATH="$server_receipt" \
      FORKAUDIT_RWD5_PREREGISTRATION_PATH="$RUN_DIR/static/preregistration.json" \
      FORKAUDIT_RWD5_FREEZE_MANIFEST_PATH="$FREEZE_MANIFEST" FORKAUDIT_RWD5_FREEZE_MANIFEST_SHA256="$EXPECTED_FREEZE_MANIFEST_SHA256" \
      FORKAUDIT_RWD5_MODE="$mode" FORKAUDIT_RWD5_RANK="$rank" \
      SGLANG_NUMA_BIND_V2=0 SGLANG_IS_FLASHINFER_AVAILABLE=0 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
      PYTHONPATH="$INSTRUMENTED_REPO/python:$CODE_DIR" \
      PYTHONPYCACHEPREFIX="$RUN_DIR/caches/${mode}-rank-${rank}/pycache" \
      TORCHINDUCTOR_CACHE_DIR="$RUN_DIR/caches/${mode}-rank-${rank}/torchinductor" \
      TRITON_CACHE_DIR="$RUN_DIR/caches/${mode}-rank-${rank}/triton" \
      XDG_CACHE_HOME="$RUN_DIR/caches/${mode}-rank-${rank}/xdg" HF_HOME="$RUN_DIR/caches/${mode}-rank-${rank}/huggingface" \
      "$PYTHON_BIN" -m sglang.launch_server --model-path "$MODEL_DIR" --served-model-name "$SERVER_NAME" \
      --host 127.0.0.1 --port "$port" --tp-size 1 --dtype bfloat16 --context-length 8192 \
      --max-running-requests 1 --max-total-tokens 8192 --mem-fraction-static 0.80 \
      --random-seed $((20260821 + rank)) --sampling-backend pytorch --disable-cuda-graph \
      --disable-piecewise-cuda-graph --disable-overlap-schedule --enable-cache-report "${mode_args[@]}" \
      > "$RUN_DIR/server-logs/${mode}-rank-${rank}.log" 2>&1 &
    for _ in $(seq 1 100); do [[ -s "$pid_file" ]] && break; sleep 0.05; done
    [[ -s "$pid_file" ]] || die "missing server PID"
    SERVER_PIDS[$rank]=$(cat "$pid_file")
  done
  for rank in $(seq 0 7); do wait_ready "$rank"; done

  for rank in $(seq 0 7); do
    port=$((BASE_PORT + rank))
    CUDA_VISIBLE_DEVICES="${GPU_UUIDS[$rank]}" PYTHONPATH="$CODE_DIR" "$PYTHON_BIN" "$CLIENT" \
      --stage server_receipt --mode "$mode" --rank "$rank" --model "$MODEL_DIR" --data "$VALIDATION_DATA" \
      --base-url "http://127.0.0.1:${port}" --official-repo "$OFFICIAL_REPO" --instrumented-repo "$INSTRUMENTED_REPO" \
      --patch "$PATCH" --receipt-module "$RECEIPT_MODULE" --code-dir "$CODE_DIR" --server-pid "${SERVER_PIDS[$rank]}" \
      --expected-gpu-uuid "${GPU_UUIDS[$rank]}" --model-weight-ledger "$MODEL_WEIGHT_LEDGER" \
      --model-artifact-ledger "$MODEL_ARTIFACT_LEDGER" --preregistration "$RUN_DIR/static/preregistration.json" \
      --worker-receipt "$RUN_DIR/scheduler-workers/${mode}-rank-${rank}.json" \
      --freeze-manifest "$FREEZE_MANIFEST" --expected-freeze-manifest-sha256 "$EXPECTED_FREEZE_MANIFEST_SHA256" \
      --output "$RUN_DIR/server-receipts/${mode}-rank-${rank}.json"
  done

  local -a client_pids=()
  for rank in $(seq 0 7); do
    port=$((BASE_PORT + rank))
    PYTHONPATH="$CODE_DIR" "$PYTHON_BIN" "$CLIENT" --stage client --mode "$mode" --rank "$rank" --world-size 8 \
      --model "$MODEL_DIR" --data "$VALIDATION_DATA" --base-url "http://127.0.0.1:${port}" \
      --served-model-name "$SERVER_NAME" --server-receipt "$RUN_DIR/server-receipts/${mode}-rank-${rank}.json" \
      --target-file "$RUN_DIR/targets/${mode}-rank-${rank}.json" \
      --store-receipt "$RUN_DIR/store-receipts/${mode}-rank-${rank}.json" \
      --terminal-receipt "$RUN_DIR/store-receipts/${mode}-rank-${rank}.terminal.json" \
      --preregistration "$RUN_DIR/static/preregistration.json" --freeze-manifest "$FREEZE_MANIFEST" \
      --expected-freeze-manifest-sha256 "$EXPECTED_FREEZE_MANIFEST_SHA256" \
      --output "$RUN_DIR/raw/${mode}-rank-${rank}.json" \
      > "$RUN_DIR/logs/client-${mode}-rank-${rank}.log" 2>&1 &
    client_pids[$rank]=$!
  done
  for rank in $(seq 0 7); do wait "${client_pids[$rank]}" || die "client $mode rank $rank failed"; done
  cleanup_servers
  date -u +%FT%TZ > "$RUN_DIR/stages/20_${mode}_complete"
}

# Affected-only: these are the only two GPU arms in this launcher.
run_mode prefix_cache
run_mode transition_rope_recompute

PYTHONPATH="$CODE_DIR" "$PYTHON_BIN" "$REPLAY" --root "$RUN_DIR" --output "$RUN_DIR/blind-replay.json" \
  --freeze-manifest "$FREEZE_MANIFEST" --expected-freeze-manifest-sha256 "$EXPECTED_FREEZE_MANIFEST_SHA256"
date -u +%FT%TZ > "$RUN_DIR/stages/30_blind_replay_complete"
CUDA_VISIBLE_DEVICES="${GPU_UUIDS[0]}" PYTHON="$PYTHON_BIN" PYTHONPATH="$CODE_DIR" \
  "$PYTHON_BIN" "$STATIC_BUILDER" --stage verify "${COMMON_STATIC_ARGS[@]}" --output-dir "$RUN_DIR/static" \
  --validation-output "$RUN_DIR/terminal-static-verification.json" --verify-model-bytes

find "$RUN_DIR" -type f ! -name all-artifacts.sha256 ! -name COMPLETED ! -path '*/stages/99_done' -print0 \
  | sort -z | xargs -0 sha256sum > "$RUN_DIR/all-artifacts.sha256"
sha256sum -c "$RUN_DIR/all-artifacts.sha256" >/dev/null
date -u +%FT%TZ > "$RUN_DIR/stages/99_done"
touch "$RUN_DIR/COMPLETED"
trap - ERR INT TERM
