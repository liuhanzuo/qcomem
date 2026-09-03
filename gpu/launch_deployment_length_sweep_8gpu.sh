#!/usr/bin/env bash
# A4/A5: honest dense baseline + generation-length sweep + quantized exact caches.
#
# Modelled on launch_deployment_8gpu.sh and takes the same CODE_DIR / MODEL_DIR /
# DATA_FILE / RUN_DIR / ENV_DIR environment variables in the same way.  It runs
# run_deployment_length_sweep.py on eight GPUs, one rank per GPU, then
# re-validates every shard with the torch-free aggregator.
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
# A4 requirement 3: the same items at each of these generation lengths
MAX_NEW_TOKENS_SWEEP=${MAX_NEW_TOKENS_SWEEP:-8,128,512}
# dense-recompute recomputes the whole sequence per token; capping it keeps the
# sweep finite while leaving the published arm in the comparison at n=8
CONFIG_LENGTH_LIMITS=${CONFIG_LENGTH_LIMITS:-dense-recompute=8}
EOS_POLICY=${EOS_POLICY:-ignore}
GENERATION_LIMIT_POLICY=${GENERATION_LIMIT_POLICY:-fixed}
GATE_DOCUMENT_TOKENS=${GATE_DOCUMENT_TOKENS:-256}
GATE_QUERY_TOKENS=${GATE_QUERY_TOKENS:-64}
GATE_NEW_TOKENS=${GATE_NEW_TOKENS:-4}
WARMUPS=${WARMUPS:-1}
REPEATS=${REPEATS:-3}
SEED=${SEED:-20260902}
GROUP_SIZE=${GROUP_SIZE:-64}
MIXED_POLICY_FILE=${MIXED_POLICY_FILE:-}
MIXED_POLICY_NAME=${MIXED_POLICY_NAME:-same_memory_as_frozen}
FORK_STRATEGY=${FORK_STRATEGY:-deep-clone}
GATE_ONLY=${GATE_ONLY:-0}
SKIP_PUBLISHED_EXACTNESS_GATE=${SKIP_PUBLISHED_EXACTNESS_GATE:-0}
DROP_MEMORY_SAMPLES=${DROP_MEMORY_SAMPLES:-0}
DECODE_SAMPLE_STRIDE=${DECODE_SAMPLE_STRIDE:-0}
SAFETY_HEADROOM_GIB=${SAFETY_HEADROOM_GIB:-4.0}
CONFIGS=${CONFIGS:-dense-recompute,dense-prefill-once,full-prefix-q16,full-prefix-q8,full-prefix-q4,full-prefix-frozen-static,qcomem-d7-frozen-static}
EXPECTED_DATA_SHA256=${EXPECTED_DATA_SHA256:-}
EXPECTED_SOURCE_REVISION=${EXPECTED_SOURCE_REVISION:-}
EXPECTED_SOURCE_INDICES=${EXPECTED_SOURCE_INDICES:-}
EXPECTED_WORKLOADS=${EXPECTED_WORKLOADS:-}
PROTOCOL_LABEL=${PROTOCOL_LABEL:-a4-a5-length-sweep}
GPUS=${GPUS:-8}
RANK_START_DELAY=${RANK_START_DELAY:-5}

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
if [ "$GPU_COUNT" -ne "$GPUS" ]; then
  echo "expected $GPUS GPUs, found $GPU_COUNT" >&2
  exit 1
fi
nvidia-smi \
  --query-gpu=index,uuid,name,driver_version,memory.total,pstate,power.limit \
  --format=csv > "$RUN_DIR/gpus-before.csv"

# ---- preflight: every file this run touches must compile, and the byte
# ---- accounting must pass its torch-free unit tests before a GPU is claimed
"$ENV_DIR/bin/python" -m py_compile \
  "$CODE_DIR/qcomem_torch.py" \
  "$CODE_DIR/qcomem_paged.py" \
  "$CODE_DIR/qcomem_deployment.py" \
  "$CODE_DIR/qcomem_eq3_accounting.py" \
  "$CODE_DIR/qcomem_deployment_arms.py" \
  "$CODE_DIR/run_deployment_bench.py" \
  "$CODE_DIR/run_deployment_length_sweep.py" \
  "$CODE_DIR/aggregate_deployment_length_sweep.py"
PYTHONPATH="$CODE_DIR${PYTHONPATH:+:$PYTHONPATH}" \
  "$ENV_DIR/bin/python" -m unittest discover \
  -s "$CODE_DIR" -p 'test_qcomem_eq3_accounting.py' -v \
  > "$RUN_DIR/length-sweep-tests.log" 2>&1
PYTHONPATH="$CODE_DIR${PYTHONPATH:+:$PYTHONPATH}" \
  "$ENV_DIR/bin/python" -m unittest discover \
  -s "$CODE_DIR" -p 'test_aggregate_deployment_length_sweep.py' -v \
  >> "$RUN_DIR/length-sweep-tests.log" 2>&1
PYTHONPATH="$CODE_DIR${PYTHONPATH:+:$PYTHONPATH}" \
  "$ENV_DIR/bin/python" -m unittest discover \
  -s "$CODE_DIR" -p 'test_qcomem_deployment_arms.py' -v \
  >> "$RUN_DIR/length-sweep-tests.log" 2>&1
PYTHONPATH="$CODE_DIR${PYTHONPATH:+:$PYTHONPATH}" \
  "$ENV_DIR/bin/python" -m unittest discover \
  -s "$CODE_DIR" -p 'test_qcomem_deployment.py' -v \
  >> "$RUN_DIR/length-sweep-tests.log" 2>&1
date -u +%FT%TZ > "$RUN_DIR/stages/01_preflight_ok"

IFS=',' read -r -a SWEEP_LIST <<< "$MAX_NEW_TOKENS_SWEEP"
IFS=',' read -r -a CONFIG_LIST <<< "$CONFIGS"
if [ "${#CONFIG_LIST[@]}" -eq 0 ]; then
  echo "CONFIGS must be a non-empty comma-separated list" >&2
  exit 1
fi

COMMON_ARGS=(
  --model "$MODEL_DIR"
  --run-dir "$RUN_DIR"
  --world-size "$GPUS"
  --workload "$WORKLOAD"
  --limit-per-dataset "$LIMIT_PER_DATASET"
  --source-index-start "$SOURCE_INDEX_START"
  --source-index-end "$SOURCE_INDEX_END"
  --exclude-source-indices 4 5
  --max-input-tokens "$MAX_INPUT_TOKENS"
  --max-new-tokens-sweep "${SWEEP_LIST[@]}"
  --configs "${CONFIG_LIST[@]}"
  --eos-policy "$EOS_POLICY"
  --generation-limit-policy "$GENERATION_LIMIT_POLICY"
  --group-size "$GROUP_SIZE"
  --gate-document-tokens "$GATE_DOCUMENT_TOKENS"
  --gate-query-tokens "$GATE_QUERY_TOKENS"
  --gate-new-tokens "$GATE_NEW_TOKENS"
  --warmups "$WARMUPS"
  --repeats "$REPEATS"
  --seed "$SEED"
  --fork-strategy "$FORK_STRATEGY"
  --protocol-label "$PROTOCOL_LABEL"
  --decode-sample-stride "$DECODE_SAMPLE_STRIDE"
  --safety-headroom-gib "$SAFETY_HEADROOM_GIB"
)
if [ -n "$CONFIG_LENGTH_LIMITS" ]; then
  IFS=',' read -r -a LIMIT_LIST <<< "$CONFIG_LENGTH_LIMITS"
  COMMON_ARGS+=(--config-length-limit "${LIMIT_LIST[@]}")
else
  COMMON_ARGS+=(--config-length-limit)
fi
if [ "$WORKLOAD" = "longbench" ]; then
  COMMON_ARGS+=(--data "$DATA_FILE")
else
  COMMON_ARGS+=(--context-lengths 4096 8192 16384 32768 --synthetic-repetitions 2)
fi
if [ -n "$EXPECTED_DATA_SHA256" ]; then
  COMMON_ARGS+=(--expected-data-sha256 "$EXPECTED_DATA_SHA256")
fi
if [ -n "$EXPECTED_SOURCE_REVISION" ]; then
  COMMON_ARGS+=(--expected-source-revision "$EXPECTED_SOURCE_REVISION")
fi
if [ -n "$EXPECTED_SOURCE_INDICES" ]; then
  IFS=',' read -r -a EXPECTED_SOURCE_INDEX_LIST <<< "$EXPECTED_SOURCE_INDICES"
  COMMON_ARGS+=(--expected-source-indices "${EXPECTED_SOURCE_INDEX_LIST[@]}")
fi
if [ -n "$EXPECTED_WORKLOADS" ]; then
  COMMON_ARGS+=(--expected-workloads "$EXPECTED_WORKLOADS")
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
if [ "$SKIP_PUBLISHED_EXACTNESS_GATE" = "1" ]; then
  COMMON_ARGS+=(--skip-published-exactness-gate)
fi
if [ "$DROP_MEMORY_SAMPLES" = "1" ]; then
  COMMON_ARGS+=(--drop-memory-samples)
fi

PIDS=()
RANKS=()
LAST_RANK=$((GPUS - 1))
for RANK in $(seq 0 "$LAST_RANK"); do
  CUDA_VISIBLE_DEVICES=$RANK PYTHONPATH="$CODE_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    "$ENV_DIR/bin/python" \
    "$CODE_DIR/run_deployment_length_sweep.py" \
    "${COMMON_ARGS[@]}" --rank "$RANK" \
    > "$RUN_DIR/logs/rank-${RANK}.log" 2>&1 &
  PIDS+=("$!")
  RANKS+=("$RANK")
  sleep "$RANK_START_DELAY"
done

FAILED=0
for INDEX in "${!PIDS[@]}"; do
  if ! wait "${PIDS[$INDEX]}"; then
    echo "rank ${RANKS[$INDEX]} failed; see logs/rank-${RANKS[$INDEX]}.log" >&2
    FAILED=1
  fi
done
if [ "$FAILED" -ne 0 ]; then
  date -u +%FT%TZ > "$RUN_DIR/stages/FAILED"
  exit 1
fi

date -u +%FT%TZ > "$RUN_DIR/stages/02_shards_done"
if [ "$GATE_ONLY" = "1" ]; then
  PYTHONPATH="$CODE_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    "$ENV_DIR/bin/python" - "$RUN_DIR" "$GPUS" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
expected = int(sys.argv[2])
shards = sorted(run_dir.glob("length-sweep-shard-*.json"))
if len(shards) != expected:
    raise SystemExit(f"expected {expected} gate shards, found {len(shards)}")
for path in shards:
    payload = json.loads(path.read_text())
    if payload.get("status") != "gate_passed":
        raise SystemExit(f"gate shard did not pass: {path}")
    for name, gate in (payload.get("gates") or {}).items():
        if not gate.get("passed"):
            raise SystemExit(f"gate {name} failed in {path}")
print("A4/A5 gates passed on every rank")
PY
  date -u +%FT%TZ > "$RUN_DIR/stages/02_gates_ok"
  nvidia-smi \
    --query-gpu=index,uuid,name,memory.used,pstate,power.draw,temperature.gpu \
    --format=csv > "$RUN_DIR/gpus-after.csv"
  date -u +%FT%TZ > "$RUN_DIR/stages/99_done"
  echo "A4/A5 gate-only run complete: $RUN_DIR"
  exit 0
fi

PYTHONPATH="$CODE_DIR${PYTHONPATH:+:$PYTHONPATH}" \
  "$ENV_DIR/bin/python" "$CODE_DIR/aggregate_deployment_length_sweep.py" \
  "$RUN_DIR" --expected-shards "$GPUS" \
  > "$RUN_DIR/aggregate.log" 2>&1
nvidia-smi \
  --query-gpu=index,uuid,name,memory.used,pstate,power.draw,temperature.gpu \
  --format=csv > "$RUN_DIR/gpus-after.csv"
date -u +%FT%TZ > "$RUN_DIR/stages/99_done"
echo "A4/A5 length-sweep benchmark complete: $RUN_DIR"
