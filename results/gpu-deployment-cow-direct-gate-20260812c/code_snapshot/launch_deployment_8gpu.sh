#!/usr/bin/env bash
set -euo pipefail

CODE_DIR=${CODE_DIR:?set CODE_DIR}
MODEL_DIR=${MODEL_DIR:?set MODEL_DIR}
RUN_DIR=${RUN_DIR:?set RUN_DIR}
ENV_DIR=${ENV_DIR:?set ENV_DIR}
WORKLOAD=${WORKLOAD:-longbench}
DATA_FILE=${DATA_FILE:-}
LIMIT_PER_DATASET=${LIMIT_PER_DATASET:-4}
SOURCE_INDEX_START=${SOURCE_INDEX_START:-6}
SOURCE_INDEX_END=${SOURCE_INDEX_END:-35}
MAX_INPUT_TOKENS=${MAX_INPUT_TOKENS:-4096}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-16}
GATE_DOCUMENT_TOKENS=${GATE_DOCUMENT_TOKENS:-256}
GATE_QUERY_TOKENS=${GATE_QUERY_TOKENS:-64}
GATE_NEW_TOKENS=${GATE_NEW_TOKENS:-4}
WARMUPS=${WARMUPS:-1}
REPEATS=${REPEATS:-3}
SEED=${SEED:-20260812}
MIXED_POLICY_FILE=${MIXED_POLICY_FILE:-}
MIXED_POLICY_NAME=${MIXED_POLICY_NAME:-same_memory_as_frozen}
FORK_STRATEGY=${FORK_STRATEGY:-deep-clone}
GATE_ONLY=${GATE_ONLY:-0}
CONFIGS=${CONFIGS:-}

if [ "$WORKLOAD" = "longbench" ] && [ -z "$DATA_FILE" ]; then
  echo "DATA_FILE is required for WORKLOAD=longbench" >&2
  exit 1
fi
if [ "$WORKLOAD" != "longbench" ] && [ "$WORKLOAD" != "synthetic" ]; then
  echo "WORKLOAD must be longbench or synthetic" >&2
  exit 1
fi

mkdir -p "$RUN_DIR/logs" "$RUN_DIR/stages"
date -u +%FT%TZ > "$RUN_DIR/stages/00_start"
GPU_COUNT=$(nvidia-smi -L | wc -l | tr -d ' ')
if [ "$GPU_COUNT" -ne 8 ]; then
  echo "expected 8 GPUs, found $GPU_COUNT" >&2
  exit 1
fi
nvidia-smi \
  --query-gpu=index,uuid,name,driver_version,memory.total,pstate,power.limit \
  --format=csv > "$RUN_DIR/gpus-before.csv"
"$ENV_DIR/bin/python" -m py_compile \
  "$CODE_DIR/qcomem_torch.py" \
  "$CODE_DIR/qcomem_paged.py" \
  "$CODE_DIR/qcomem_deployment.py" \
  "$CODE_DIR/run_deployment_bench.py" \
  "$CODE_DIR/aggregate_deployment.py"
"$ENV_DIR/bin/python" -m unittest discover \
  -s "$CODE_DIR" -p 'test_qcomem_deployment.py' -v \
  > "$RUN_DIR/deployment-tests.log" 2>&1
"$ENV_DIR/bin/python" -m unittest discover \
  -s "$CODE_DIR" -p 'test_qcomem_paged.py' -v \
  >> "$RUN_DIR/deployment-tests.log" 2>&1
PYTHONPATH="$CODE_DIR${PYTHONPATH:+:$PYTHONPATH}" \
  "$ENV_DIR/bin/python" -m unittest \
  test_qcomem_torch.QuantizationTest.test_q16_packed_cache_forks_own_independent_mutable_storage \
  -v >> "$RUN_DIR/deployment-tests.log" 2>&1
date -u +%FT%TZ > "$RUN_DIR/stages/01_preflight_ok"

COMMON_ARGS=(
  --model "$MODEL_DIR"
  --run-dir "$RUN_DIR"
  --world-size 8
  --workload "$WORKLOAD"
  --limit-per-dataset "$LIMIT_PER_DATASET"
  --source-index-start "$SOURCE_INDEX_START"
  --source-index-end "$SOURCE_INDEX_END"
  --exclude-source-indices 4 5
  --max-input-tokens "$MAX_INPUT_TOKENS"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --gate-document-tokens "$GATE_DOCUMENT_TOKENS"
  --gate-query-tokens "$GATE_QUERY_TOKENS"
  --gate-new-tokens "$GATE_NEW_TOKENS"
  --warmups "$WARMUPS"
  --repeats "$REPEATS"
  --seed "$SEED"
  --fork-strategy "$FORK_STRATEGY"
)
if [ "$WORKLOAD" = "longbench" ]; then
  COMMON_ARGS+=(--data "$DATA_FILE")
else
  COMMON_ARGS+=(--context-lengths 4096 8192 16384 32768 --synthetic-repetitions 2)
fi
if [ -n "$MIXED_POLICY_FILE" ]; then
  COMMON_ARGS+=(
    --mixed-policy-file "$MIXED_POLICY_FILE"
    --mixed-policy-name "$MIXED_POLICY_NAME"
  )
fi
if [ "$GATE_ONLY" = "1" ]; then
  COMMON_ARGS+=(--gate-only)
fi
if [ -n "$CONFIGS" ]; then
  IFS=',' read -r -a CONFIG_LIST <<< "$CONFIGS"
  if [ "${#CONFIG_LIST[@]}" -eq 0 ]; then
    echo "CONFIGS must be a non-empty comma-separated list" >&2
    exit 1
  fi
  COMMON_ARGS+=(--configs "${CONFIG_LIST[@]}")
fi

PIDS=()
for RANK in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$RANK "$ENV_DIR/bin/python" \
    "$CODE_DIR/run_deployment_bench.py" \
    "${COMMON_ARGS[@]}" --rank "$RANK" \
    > "$RUN_DIR/logs/rank-${RANK}.log" 2>&1 &
  PIDS+=("$!")
  sleep 5
done

FAILED=0
for INDEX in 0 1 2 3 4 5 6 7; do
  if ! wait "${PIDS[$INDEX]}"; then
    echo "rank $INDEX failed; see logs/rank-${INDEX}.log" >&2
    FAILED=1
  fi
done
if [ "$FAILED" -ne 0 ]; then
  date -u +%FT%TZ > "$RUN_DIR/stages/FAILED"
  exit 1
fi

date -u +%FT%TZ > "$RUN_DIR/stages/02_shards_done"
if [ "$GATE_ONLY" = "1" ]; then
  "$ENV_DIR/bin/python" - "$RUN_DIR" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
shards = sorted(run_dir.glob("deployment-shard-*.json"))
if len(shards) != 8:
    raise SystemExit(f"expected 8 exactness shards, found {len(shards)}")
failed = [
    str(path)
    for path in shards
    if json.loads(path.read_text()).get("status") != "exactness_gate_passed"
]
if failed:
    raise SystemExit(f"exactness shards did not pass: {failed}")
for path in shards:
    gate = json.loads(path.read_text()).get("exactness_gate", {})
    if not gate.get("passed"):
        raise SystemExit(f"top-level exactness gate failed: {path}")
    if gate.get("fork_strategy") != "paged-cow-staging":
        continue
    direct = gate.get("cow_vs_deep_clone_q16")
    comparison = direct.get("comparison", {}) if isinstance(direct, dict) else {}
    immutable = (
        direct.get("cow_immutable_audit", {}) if isinstance(direct, dict) else {}
    )
    direct_ok = bool(
        isinstance(direct, dict)
        and direct.get("passed")
        and direct.get("same_persistent_source")
        and direct.get("cow_was_exercised")
        and direct.get("strategy_effective") == "paged-cow-staging"
        and direct.get("source_after_eager", {}).get("verified")
        and direct.get("source_after_cow", {}).get("verified")
        and immutable.get("verified")
        and comparison.get("passed")
        and comparison.get("token_sequence_exact")
        and comparison.get("logits_bitwise_exact")
    )
    if not direct_ok:
        raise SystemExit(f"same-source direct COW gate failed or missing: {path}")
PY
  date -u +%FT%TZ > "$RUN_DIR/stages/02_exactness_ok"
  nvidia-smi \
    --query-gpu=index,uuid,name,memory.used,pstate,power.draw,temperature.gpu \
    --format=csv > "$RUN_DIR/gpus-after.csv"
  date -u +%FT%TZ > "$RUN_DIR/stages/99_done"
  echo "Q-CoMem deployment exactness smoke complete: $RUN_DIR"
  exit 0
fi
"$ENV_DIR/bin/python" "$CODE_DIR/aggregate_deployment.py" \
  "$RUN_DIR" --expected-shards 8 \
  > "$RUN_DIR/aggregate.log" 2>&1
nvidia-smi \
  --query-gpu=index,uuid,name,memory.used,pstate,power.draw,temperature.gpu \
  --format=csv > "$RUN_DIR/gpus-after.csv"
date -u +%FT%TZ > "$RUN_DIR/stages/99_done"
echo "Q-CoMem deployment benchmark complete: $RUN_DIR"
