#!/usr/bin/env bash
# C1: one quantized depth-split entry shared across N>1 concurrent requests,
# with the ForkAudit contract instantiated on that path.
#
# Modelled on launch_deployment_length_sweep_8gpu.sh and takes the same
# CODE_DIR / MODEL_DIR / DATA_FILE / RUN_DIR / ENV_DIR environment variables in
# the same way.  It runs run_shared_packed_multifork.py on eight GPUs, one rank
# per GPU, then re-validates every shard with the torch-free aggregator.
#
# GATE_ONLY=1 runs the per-rank preflight gate and stops.  Run it first: the
# gate builds a 256-token entry, forks it twice, and fails in minutes if the
# shared mode falls back, if sharing is vacuous, if the N>1 output is not
# token-identical to the published N=1 private path, or if any contract target
# is uncovered or failing.
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
CONFIG=${CONFIG:-qcomem-d7-frozen-static}
# at least one fanout must exceed 1; N=1 is the reference the shared arm is
# compared against and is also needed for cross-N prefix consistency
FANOUTS=${FANOUTS:-1,2,4}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-32}
# borrowed-prefix keeps the document borrowed for the whole request;
# materialized-tail is the conservative fallback if the borrowed-prefix layer
# fails its preflight on this Transformers build
TAIL_POLICY=${TAIL_POLICY:-borrowed-prefix}
REBIND_POLICY=${REBIND_POLICY:-transition}
QUERY_SOURCE=${QUERY_SOURCE:-cross-item}
EOS_POLICY=${EOS_POLICY:-ignore}
GROUP_SIZE=${GROUP_SIZE:-64}
WARMUPS=${WARMUPS:-1}
REPEATS=${REPEATS:-1}
SEED=${SEED:-20260903}
GATE_DOCUMENT_TOKENS=${GATE_DOCUMENT_TOKENS:-256}
GATE_QUERY_TOKENS=${GATE_QUERY_TOKENS:-32}
GATE_NEW_TOKENS=${GATE_NEW_TOKENS:-4}
GATE_FANOUT=${GATE_FANOUT:-2}
GATE_ONLY=${GATE_ONLY:-0}
DROP_RECEIPT_DETAILS=${DROP_RECEIPT_DETAILS:-0}
STRICT_ACCOUNTING=${STRICT_ACCOUNTING:-1}
MIXED_POLICY_FILE=${MIXED_POLICY_FILE:-}
MIXED_POLICY_NAME=${MIXED_POLICY_NAME:-same_memory_as_frozen}
EXPECTED_DATA_SHA256=${EXPECTED_DATA_SHA256:-}
EXPECTED_SOURCE_REVISION=${EXPECTED_SOURCE_REVISION:-}
EXPECTED_SOURCE_INDICES=${EXPECTED_SOURCE_INDICES:-}
EXPECTED_WORKLOADS=${EXPECTED_WORKLOADS:-}
PROTOCOL_LABEL=${PROTOCOL_LABEL:-c1-shared-packed-multifork}
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

# ---- preflight: every file this run touches must compile, and the torch-free
# ---- bookkeeping must pass its unit tests before a GPU is claimed
"$ENV_DIR/bin/python" -m py_compile \
  "$CODE_DIR/qcomem_torch.py" \
  "$CODE_DIR/qcomem_paged.py" \
  "$CODE_DIR/qcomem_deployment.py" \
  "$CODE_DIR/qcomem_multifork_accounting.py" \
  "$CODE_DIR/qcomem_shared_packed_fork.py" \
  "$CODE_DIR/qcomem_shared_packed_forkaudit.py" \
  "$CODE_DIR/run_shared_packed_multifork.py" \
  "$CODE_DIR/aggregate_shared_packed_multifork.py"
PYTHONPATH="$CODE_DIR${PYTHONPATH:+:$PYTHONPATH}" \
  "$ENV_DIR/bin/python" -m unittest discover \
  -s "$CODE_DIR" -p 'test_qcomem_multifork_accounting.py' -v \
  > "$RUN_DIR/multifork-tests.log" 2>&1
PYTHONPATH="$CODE_DIR${PYTHONPATH:+:$PYTHONPATH}" \
  "$ENV_DIR/bin/python" -m unittest discover \
  -s "$CODE_DIR" -p 'test_aggregate_shared_packed_multifork.py' -v \
  >> "$RUN_DIR/multifork-tests.log" 2>&1
PYTHONPATH="$CODE_DIR${PYTHONPATH:+:$PYTHONPATH}" \
  "$ENV_DIR/bin/python" -m unittest discover \
  -s "$CODE_DIR" -p 'test_qcomem_shared_packed_fork.py' -v \
  >> "$RUN_DIR/multifork-tests.log" 2>&1
# the published paths this experiment must not have changed
PYTHONPATH="$CODE_DIR${PYTHONPATH:+:$PYTHONPATH}" \
  "$ENV_DIR/bin/python" -m unittest discover \
  -s "$CODE_DIR" -p 'test_qcomem_deployment.py' -v \
  >> "$RUN_DIR/multifork-tests.log" 2>&1
PYTHONPATH="$CODE_DIR${PYTHONPATH:+:$PYTHONPATH}" \
  "$ENV_DIR/bin/python" -m unittest discover \
  -s "$CODE_DIR" -p 'test_qcomem_paged.py' -v \
  >> "$RUN_DIR/multifork-tests.log" 2>&1
date -u +%FT%TZ > "$RUN_DIR/stages/01_preflight_ok"

IFS=',' read -r -a FANOUT_LIST <<< "$FANOUTS"
if [ "${#FANOUT_LIST[@]}" -eq 0 ]; then
  echo "FANOUTS must be a non-empty comma-separated list" >&2
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
  --config "$CONFIG"
  --fanouts "${FANOUT_LIST[@]}"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --tail-policy "$TAIL_POLICY"
  --rebind-policy "$REBIND_POLICY"
  --query-source "$QUERY_SOURCE"
  --eos-policy "$EOS_POLICY"
  --group-size "$GROUP_SIZE"
  --warmups "$WARMUPS"
  --repeats "$REPEATS"
  --seed "$SEED"
  --gate-document-tokens "$GATE_DOCUMENT_TOKENS"
  --gate-query-tokens "$GATE_QUERY_TOKENS"
  --gate-new-tokens "$GATE_NEW_TOKENS"
  --gate-fanout "$GATE_FANOUT"
  --protocol-label "$PROTOCOL_LABEL"
)
if [ "$WORKLOAD" = "longbench" ]; then
  COMMON_ARGS+=(--data "$DATA_FILE")
else
  COMMON_ARGS+=(--context-lengths 4096 8192 --synthetic-repetitions 2)
fi
if [ -n "$EXPECTED_DATA_SHA256" ]; then
  COMMON_ARGS+=(--expected-data-sha256 "$EXPECTED_DATA_SHA256")
fi
if [ -n "$EXPECTED_SOURCE_REVISION" ]; then
  COMMON_ARGS+=(--expected-source-revision "$EXPECTED_SOURCE_REVISION")
fi
if [ -n "$EXPECTED_SOURCE_INDICES" ]; then
  IFS=',' read -r -a EXPECTED_INDEX_LIST <<< "$EXPECTED_SOURCE_INDICES"
  COMMON_ARGS+=(--expected-source-indices "${EXPECTED_INDEX_LIST[@]}")
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
if [ "$DROP_RECEIPT_DETAILS" = "1" ]; then
  COMMON_ARGS+=(--drop-receipt-details)
fi
if [ "$STRICT_ACCOUNTING" != "1" ]; then
  COMMON_ARGS+=(--no-strict-accounting)
fi

PIDS=()
RANKS=()
LAST_RANK=$((GPUS - 1))
for RANK in $(seq 0 "$LAST_RANK"); do
  CUDA_VISIBLE_DEVICES=$RANK PYTHONPATH="$CODE_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    "$ENV_DIR/bin/python" \
    "$CODE_DIR/run_shared_packed_multifork.py" \
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
shards = sorted(run_dir.glob("multifork-shard-*.json"))
if len(shards) != expected:
    raise SystemExit(f"expected {expected} gate shards, found {len(shards)}")
for path in shards:
    payload = json.loads(path.read_text())
    if payload.get("status") != "gate_passed":
        raise SystemExit(f"gate shard did not pass: {path}")
    for name, gate in (payload.get("gates") or {}).items():
        if not gate.get("passed"):
            raise SystemExit(f"gate {name} failed in {path}")
        summary = gate.get("contract_summary") or {}
        print(
            json.dumps(
                {
                    "shard": path.name,
                    "gate": name,
                    "tail_policy": gate.get("tail_policy"),
                    "sharing_window": gate.get("sharing_window"),
                    "status_vector": summary.get("status_vector"),
                    "coverage_vector": summary.get("coverage_vector"),
                }
            )
        )
print("C1 shared-packed multifork gates passed on every rank")
PY
  date -u +%FT%TZ > "$RUN_DIR/stages/02_gates_ok"
  nvidia-smi \
    --query-gpu=index,uuid,name,memory.used,pstate,power.draw,temperature.gpu \
    --format=csv > "$RUN_DIR/gpus-after.csv"
  date -u +%FT%TZ > "$RUN_DIR/stages/99_done"
  echo "C1 gate-only run complete: $RUN_DIR"
  exit 0
fi

PYTHONPATH="$CODE_DIR${PYTHONPATH:+:$PYTHONPATH}" \
  "$ENV_DIR/bin/python" "$CODE_DIR/aggregate_shared_packed_multifork.py" \
  "$RUN_DIR" --expected-shards "$GPUS" --require-complete-record \
  > "$RUN_DIR/aggregate.log" 2>&1
nvidia-smi \
  --query-gpu=index,uuid,name,memory.used,pstate,power.draw,temperature.gpu \
  --format=csv > "$RUN_DIR/gpus-after.csv"
date -u +%FT%TZ > "$RUN_DIR/stages/99_done"
echo "C1 shared-packed multifork run complete: $RUN_DIR"
