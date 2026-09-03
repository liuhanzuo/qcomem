#!/usr/bin/env bash
set -euo pipefail

CODE_DIR=${CODE_DIR:?set CODE_DIR}
MODEL_DIR=${MODEL_DIR:?set MODEL_DIR}
DATA_FILE=${DATA_FILE:?set DATA_FILE}
RUN_DIR=${RUN_DIR:?set RUN_DIR}
ENV_DIR=${ENV_DIR:?set ENV_DIR}

EXPECTED_DATA_SHA256=${EXPECTED_DATA_SHA256:?set EXPECTED_DATA_SHA256}
EXPECTED_SOURCE_REVISION=${EXPECTED_SOURCE_REVISION:?set EXPECTED_SOURCE_REVISION}
EXPECTED_SOURCE_INDICES=${EXPECTED_SOURCE_INDICES:-6,7,8,9}
EXPECTED_WORKLOADS=${EXPECTED_WORKLOADS:-8}
SOURCE_INDEX_START=${SOURCE_INDEX_START:-6}
SOURCE_INDEX_END=${SOURCE_INDEX_END:-9}
LIMIT_PER_DATASET=${LIMIT_PER_DATASET:-4}
MAX_INPUT_TOKENS=${MAX_INPUT_TOKENS:-4096}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-8}
GATE_DOCUMENT_TOKENS=${GATE_DOCUMENT_TOKENS:-256}
GATE_QUERY_TOKENS=${GATE_QUERY_TOKENS:-32}
PAGE_SIZE=${PAGE_SIZE:-128}
APPEND_PAGE_SIZE=${APPEND_PAGE_SIZE:-16}
GROUP_SIZE=${GROUP_SIZE:-64}
BENCHMARK_BITS=${BENCHMARK_BITS:-16,8,4}
QUERIES_PER_DOCUMENT=${QUERIES_PER_DOCUMENT:-2}
SAFETY_HEADROOM_GIB=${SAFETY_HEADROOM_GIB:-4}
WARMUP_COUNT=${WARMUP_COUNT:-1}
GATE_ONLY=${GATE_ONLY:-0}

case "$DATA_FILE" in
  *test-v2*|*test_v2*)
    echo "test-v2 paths are forbidden" >&2
    exit 2
    ;;
esac
if [[ "$SOURCE_INDEX_START" -ne 6 || "$SOURCE_INDEX_END" -ne 9 ]]; then
  echo "paged real short freezes source indices 6-9" >&2
  exit 2
fi
if [[ "$EXPECTED_WORKLOADS" -ne 8 ]]; then
  echo "paged real short freezes eight validation workloads" >&2
  exit 2
fi
if [[ "$GATE_ONLY" != 0 && "$GATE_ONLY" != 1 ]]; then
  echo "GATE_ONLY must be 0 or 1" >&2
  exit 2
fi
if [[ -e "$RUN_DIR" ]] && [[ -n "$(find "$RUN_DIR" -mindepth 1 -print -quit)" ]]; then
  echo "RUN_DIR must be absent or empty: $RUN_DIR" >&2
  exit 2
fi

ACTUAL_DATA_SHA256=$(sha256sum "$DATA_FILE" | awk '{print $1}')
if [[ "$ACTUAL_DATA_SHA256" != "$EXPECTED_DATA_SHA256" ]]; then
  echo "validation data SHA256 mismatch" >&2
  exit 2
fi
GPU_COUNT=$(nvidia-smi -L | wc -l | tr -d ' ')
if [[ "$GPU_COUNT" -ne 8 ]]; then
  echo "expected exactly 8 GPUs, found $GPU_COUNT" >&2
  exit 2
fi

mkdir -p "$RUN_DIR/logs" "$RUN_DIR/stages"
trap 'date -u +%FT%TZ > "$RUN_DIR/stages/FAILED"' ERR
date -u +%FT%TZ > "$RUN_DIR/stages/00_start"
nvidia-smi \
  --query-gpu=index,uuid,name,driver_version,memory.total,pstate,power.limit \
  --format=csv > "$RUN_DIR/gpus-before.csv"
sha256sum "$DATA_FILE" > "$RUN_DIR/data.sha256"
LC_ALL=C find "$CODE_DIR" -maxdepth 1 -type f -name '*.py' -print0 \
  | LC_ALL=C sort -z \
  | xargs -0 sha256sum > "$RUN_DIR/code.sha256"
sha256sum "$CODE_DIR/launch_qcomem_qwen35_paged_real_8gpu.sh" \
  >> "$RUN_DIR/code.sha256"
sha256sum "$RUN_DIR/code.sha256" > "$RUN_DIR/code-manifest.sha256"

"$ENV_DIR/bin/python" -m py_compile \
  "$CODE_DIR/qcomem_paged_attention.py" \
  "$CODE_DIR/qcomem_qwen35_paged_integration.py" \
  "$CODE_DIR/qcomem_qwen35_native_cache.py" \
  "$CODE_DIR/qcomem_qwen35_gdn_functional.py" \
  "$CODE_DIR/qcomem_qwen35_functional_stack.py" \
  "$CODE_DIR/diagnose_qcomem_qwen35_paged_dtype.py" \
  "$CODE_DIR/run_qcomem_qwen35_paged_real.py" \
  "$CODE_DIR/aggregate_qcomem_qwen35_paged_real.py"
PYTHONPATH="$CODE_DIR" "$ENV_DIR/bin/python" -m unittest -v \
  test_qcomem_paged_attention \
  test_qcomem_qwen35_paged_integration \
  test_qcomem_qwen35_native_cache \
  test_qcomem_qwen35_gdn_functional \
  test_qcomem_qwen35_functional_stack \
  test_diagnose_qcomem_qwen35_paged_dtype \
  test_run_qcomem_qwen35_paged_real \
  test_aggregate_qcomem_qwen35_paged_real \
  > "$RUN_DIR/logs/preflight-tests.log" 2>&1
date -u +%FT%TZ > "$RUN_DIR/stages/01_preflight_ok"

IFS=',' read -r -a SOURCE_INDICES <<< "$EXPECTED_SOURCE_INDICES"
COMMON_ARGS=(
  --model "$MODEL_DIR"
  --data "$DATA_FILE"
  --world-size 8
  --expected-data-sha256 "$EXPECTED_DATA_SHA256"
  --expected-source-revision "$EXPECTED_SOURCE_REVISION"
  --expected-source-indices "${SOURCE_INDICES[@]}"
  --expected-workloads "$EXPECTED_WORKLOADS"
  --source-index-start "$SOURCE_INDEX_START"
  --source-index-end "$SOURCE_INDEX_END"
  --limit-per-dataset "$LIMIT_PER_DATASET"
  --max-input-tokens "$MAX_INPUT_TOKENS"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --gate-document-tokens "$GATE_DOCUMENT_TOKENS"
  --gate-query-tokens "$GATE_QUERY_TOKENS"
  --page-size "$PAGE_SIZE"
  --append-page-size "$APPEND_PAGE_SIZE"
  --group-size "$GROUP_SIZE"
  --queries-per-document "$QUERIES_PER_DOCUMENT"
  --safety-headroom-gib "$SAFETY_HEADROOM_GIB"
  --warmup-count "$WARMUP_COUNT"
)
IFS=',' read -r -a BIT_LIST <<< "$BENCHMARK_BITS"
COMMON_ARGS+=(--benchmark-bits "${BIT_LIST[@]}")

# Rank-0-only arithmetic diagnosis.  GPUs 1-7 have no process during this
# stage; the shell itself is the fail-closed coordinator.  A timeout, missing
# artifact, incomplete ten-layer intercept, or original-threshold failure
# ends the job before any all-rank capability or benchmark phase starts.
DIAGNOSTIC_DIR="$RUN_DIR/rank0-dtype-diagnostic"
mkdir -p "$DIAGNOSTIC_DIR"
timeout 900 env CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$CODE_DIR" \
  "$ENV_DIR/bin/python" "$CODE_DIR/diagnose_qcomem_qwen35_paged_dtype.py" \
  --model "$MODEL_DIR" \
  --data "$DATA_FILE" \
  --output "$DIAGNOSTIC_DIR/paged-dtype-diagnostic.json" \
  --expected-data-sha256 "$EXPECTED_DATA_SHA256" \
  --expected-source-revision "$EXPECTED_SOURCE_REVISION" \
  --expected-source-indices "${SOURCE_INDICES[@]}" \
  --expected-workloads "$EXPECTED_WORKLOADS" \
  --source-index-start "$SOURCE_INDEX_START" \
  --source-index-end "$SOURCE_INDEX_END" \
  --limit-per-dataset "$LIMIT_PER_DATASET" \
  --max-input-tokens "$MAX_INPUT_TOKENS" \
  --document-tokens "$GATE_DOCUMENT_TOKENS" \
  --query-tokens "$GATE_QUERY_TOKENS" \
  --page-size "$PAGE_SIZE" \
  --append-page-size "$APPEND_PAGE_SIZE" \
  --rtol 0.02 \
  --atol 0.05 \
  > "$RUN_DIR/logs/rank0-dtype-diagnostic.log" 2>&1
PYTHONPATH="$CODE_DIR" "$ENV_DIR/bin/python" - \
  "$DIAGNOSTIC_DIR/paged-dtype-diagnostic.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
value = json.loads(path.read_text())
if value.get("status") != "diagnostic_complete":
    raise SystemExit("rank0 dtype diagnostic did not complete")
if value.get("formal_benchmark_authorized") is not False:
    raise SystemExit("diagnostic artifact improperly self-authorized benchmark")
indices = value.get("full_attention_layer_indices")
rows = value.get("per_layer_attention_output")
if not isinstance(indices, list) or len(indices) != 10 or len(set(indices)) != 10:
    raise SystemExit("rank0 diagnostic did not derive ten unique full-attention layers")
if not isinstance(rows, list) or [row.get("layer") for row in rows] != indices:
    raise SystemExit("rank0 per-layer diagnostic coverage is incomplete or reordered")
for label in ("legacy_intercept", "two_pass_intercept"):
    intercept = value.get(label)
    if not isinstance(intercept, dict) or intercept.get("verified") is not True:
        raise SystemExit(f"{label} is not verified")
    if intercept.get("total_calls") != 10 or intercept.get("dense_fallback_calls") != 0:
        raise SystemExit(f"{label} did not intercept exactly ten layers")
final = value.get("final_logits", {})
fixed = final.get("two_pass_bf16_vs_eager", {})
if fixed.get("rtol") != 0.02 or fixed.get("atol") != 0.05:
    raise SystemExit("rank0 fixed diagnostic changed the original thresholds")
if fixed.get("close") is not True or final.get("two_pass_token_exact") is not True:
    raise SystemExit("rank0 fixed final-logit/token gate failed")
if value.get("fixed_intercept_complete") is not True:
    raise SystemExit("rank0 fixed intercept summary failed")
if value.get("fixed_gate_passed") is not True or value.get("next_stage_authorized") is not True:
    raise SystemExit("rank0 diagnostic did not authorize the next stage")
PY
date -u +%FT%TZ > "$RUN_DIR/stages/01a_rank0_dtype_diagnostic_ok"

run_ranks() {
  local stage_name=$1
  local stage_run_dir=$2
  shift 2
  local extra_args=("$@")
  local pids=()
  local failed=0
  mkdir -p "$stage_run_dir"
  for rank in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$rank "$ENV_DIR/bin/python" \
      "$CODE_DIR/run_qcomem_qwen35_paged_real.py" \
      "${COMMON_ARGS[@]}" --run-dir "$stage_run_dir" --rank "$rank" \
      "${extra_args[@]}" \
      > "$RUN_DIR/logs/${stage_name}-rank-${rank}.log" 2>&1 &
    pids+=("$!")
    sleep 5
  done
  for index in 0 1 2 3 4 5 6 7; do
    if ! wait "${pids[$index]}"; then
      echo "$stage_name rank $index failed; see logs/${stage_name}-rank-${index}.log" >&2
      failed=1
    fi
  done
  if [[ "$failed" -ne 0 ]]; then
    return 1
  fi
}

GATE_RUN_DIR="$RUN_DIR/capability-gate"
run_ranks capability-gate "$GATE_RUN_DIR" --gate-only
PYTHONPATH="$CODE_DIR" "$ENV_DIR/bin/python" - "$GATE_RUN_DIR" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
paths = sorted(run_dir.glob("paged-real-shard-*.json"))
if len(paths) != 8:
    raise SystemExit(f"expected 8 gate shards, found {len(paths)}")
for path in paths:
    value = json.loads(path.read_text())
    if value.get("status") != "exactness_gate_passed":
        raise SystemExit(f"gate shard status failed: {path}")
    if value.get("gate", {}).get("passed") is not True:
        raise SystemExit(f"top-level gate failed: {path}")
    if value.get("rows") != []:
        raise SystemExit(f"gate-only shard unexpectedly contains benchmark rows: {path}")
PY
date -u +%FT%TZ > "$RUN_DIR/stages/02_all_rank_capability_gate_ok"

if [[ "$GATE_ONLY" == 1 ]]; then
  date -u +%FT%TZ > "$RUN_DIR/stages/99_done"
  echo "Real Qwen3.5 paged capability gate passed: $RUN_DIR"
  exit 0
fi

# Benchmark is a separate all-rank phase. No rank can enter it until every
# exactness/capability shard above has passed. Each benchmark process repeats
# its local gate after reloading the model, so the authorization is not merely
# inherited from a previous process.
run_ranks benchmark "$RUN_DIR"
date -u +%FT%TZ > "$RUN_DIR/stages/03_benchmark_shards_done"

PYTHONPATH="$CODE_DIR" "$ENV_DIR/bin/python" \
  "$CODE_DIR/aggregate_qcomem_qwen35_paged_real.py" "$RUN_DIR" \
  --expected-shards 8 \
  --expected-data-sha256 "$EXPECTED_DATA_SHA256" \
  --expected-source-revision "$EXPECTED_SOURCE_REVISION" \
  --expected-source-indices "${SOURCE_INDICES[@]}" \
  --expected-workloads "$EXPECTED_WORKLOADS" \
  --expected-max-new-tokens "$MAX_NEW_TOKENS" \
  > "$RUN_DIR/logs/aggregate.log" 2>&1
test -s "$RUN_DIR/paged-real-summary.json"
nvidia-smi \
  --query-gpu=index,uuid,name,memory.used,pstate,power.draw,temperature.gpu \
  --format=csv > "$RUN_DIR/gpus-after.csv"
date -u +%FT%TZ > "$RUN_DIR/stages/99_done"
echo "Real Qwen3.5 paged reference benchmark complete: $RUN_DIR"
