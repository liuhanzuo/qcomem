#!/usr/bin/env bash
set -euo pipefail

CODE_DIR=${CODE_DIR:?set CODE_DIR}
MODEL_DIR=${MODEL_DIR:?set MODEL_DIR}
DATA_FILE=${DATA_FILE:?set DATA_FILE}
EXPECTED_DATA_SHA256=${EXPECTED_DATA_SHA256:?set EXPECTED_DATA_SHA256}
RUN_DIR=${RUN_DIR:?set RUN_DIR}
ENV_DIR=${ENV_DIR:?set ENV_DIR}
CONFIG_FILE=${CONFIG_FILE:?set CONFIG_FILE}
SEMANTIC_SAMPLES=${SEMANTIC_SAMPLES:-8}

mkdir -p "$RUN_DIR/logs" "$RUN_DIR/stages"
trap 'date -u +%FT%TZ > "$RUN_DIR/stages/FAILED"' ERR
date -u +%FT%TZ > "$RUN_DIR/stages/00_start"

GPU_COUNT=$(nvidia-smi -L | wc -l | tr -d ' ')
if [[ "$GPU_COUNT" -ne 8 ]]; then
  echo "detached capability requires exactly 8 GPUs, found $GPU_COUNT" >&2
  exit 1
fi
if [[ "$SEMANTIC_SAMPLES" -ne 8 ]]; then
  echo "detached capability semantic sample count is frozen to 8" >&2
  exit 2
fi
nvidia-smi \
  --query-gpu=index,uuid,name,memory.total,driver_version,pstate,power.limit \
  --format=csv > "$RUN_DIR/gpus-before.csv"

ACTUAL_DATA_SHA256=$(sha256sum "$DATA_FILE" | awk '{print $1}')
if [[ "$ACTUAL_DATA_SHA256" != "$EXPECTED_DATA_SHA256" ]]; then
  echo "PG-19 capability data SHA256 mismatch" >&2
  exit 2
fi
sha256sum "$DATA_FILE" > "$RUN_DIR/training-data.sha256"
sha256sum \
  "$CODE_DIR/qcomem_torch.py" \
  "$CODE_DIR/qcomem_lora.py" \
  "$CODE_DIR/train_qcomem_lora.py" \
  "$CODE_DIR/audit_detached_lora_checkpoint.py" \
  "$CODE_DIR/run_detached_lora_semantic_gate.py" \
  "$CODE_DIR/aggregate_detached_lora_semantic_gate.py" \
  "$CODE_DIR/launch_detached_lora_capability_8gpu.sh" \
  "$CONFIG_FILE" > "$RUN_DIR/code.sha256"

"$ENV_DIR/bin/python" -m py_compile \
  "$CODE_DIR/qcomem_torch.py" \
  "$CODE_DIR/qcomem_lora.py" \
  "$CODE_DIR/train_qcomem_lora.py" \
  "$CODE_DIR/audit_detached_lora_checkpoint.py" \
  "$CODE_DIR/run_detached_lora_semantic_gate.py" \
  "$CODE_DIR/aggregate_detached_lora_semantic_gate.py"
PYTHONPATH="$CODE_DIR" "$ENV_DIR/bin/python" -m unittest discover \
  -s "$CODE_DIR" -p 'test_qcomem_lora.py' -v \
  > "$RUN_DIR/logs/preflight-tests.log" 2>&1
PYTHONPATH="$CODE_DIR" "$ENV_DIR/bin/python" -m unittest discover \
  -s "$CODE_DIR" -p 'test_detached_lora_capability.py' -v \
  >> "$RUN_DIR/logs/preflight-tests.log" 2>&1
"$ENV_DIR/bin/python" - "$CONFIG_FILE" <<'PY' \
  > "$RUN_DIR/logs/protocol-preflight.log"
import json
import sys

config = json.load(open(sys.argv[1]))
assert config["mode"] == "quant"
assert config["student_suffix_execution"] == "detached-document-cache"
assert config["steps"] == 1
assert config["context_tokens"] == 512
assert config["query_tokens"] == 128
assert config["residual_bits"] == 4
assert config["attention_bits"] == 4
assert config["linear_bits"] == 8
assert config["cache_layer_bits"] == [8, 8, 8, 4, 8, 8, 8]
print({"status": "passed", "claim": "query-continuation-only capability"})
PY
date -u +%FT%TZ > "$RUN_DIR/stages/01_preflight_ok"

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export TOKENIZERS_PARALLELISM=false
"$ENV_DIR/bin/torchrun" \
  --standalone \
  --nproc-per-node=8 \
  "$CODE_DIR/train_qcomem_lora.py" \
  --config "$CONFIG_FILE" \
  --model "$MODEL_DIR" \
  --data "$DATA_FILE" \
  --output-dir "$RUN_DIR" \
  > "$RUN_DIR/logs/train.log" 2>&1

CHECKPOINT="$RUN_DIR/checkpoint-000001.pt"
test -s "$RUN_DIR/metadata.json"
test -s "$CHECKPOINT"
CHECKPOINT_SHA256=$(sha256sum "$CHECKPOINT" | awk '{print $1}')
sha256sum "$CHECKPOINT" > "$RUN_DIR/checkpoint-000001.sha256"
PYTHONPATH="$CODE_DIR" "$ENV_DIR/bin/python" \
  "$CODE_DIR/audit_detached_lora_checkpoint.py" \
  --checkpoint "$CHECKPOINT" \
  --output "$RUN_DIR/train-capability-gate.json" \
  --expected-sha256 "$CHECKPOINT_SHA256" \
  --expected-world-size 8 \
  --expected-modules 36 \
  --expected-query-positions 128 \
  > "$RUN_DIR/logs/train-capability-gate.log" 2>&1
date -u +%FT%TZ > "$RUN_DIR/stages/02_train_capability_ok"

PIDS=()
for RANK in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$RANK "$ENV_DIR/bin/python" \
    "$CODE_DIR/run_detached_lora_semantic_gate.py" \
    --model "$MODEL_DIR" \
    --data "$DATA_FILE" \
    --checkpoint "$CHECKPOINT" \
    --output "$RUN_DIR/detached-semantic-shard-${RANK}.json" \
    --depth 7 \
    --residual-bits 4 --attention-bits 4 --linear-bits 8 \
    --cache-layer-bits 8,8,8,4,8,8,8 \
    --context-tokens 512 --query-tokens 128 \
    --samples "$SEMANTIC_SAMPLES" \
    --projection-block-size 16 \
    --min-top1-match 1.0 --max-mean-kl 0.000001 --max-logit-error 0.0 \
    --rank "$RANK" --world-size 8 \
    > "$RUN_DIR/logs/detached-semantic-rank-${RANK}.log" 2>&1 &
  PIDS+=("$!")
  sleep 5
done

FAILED=0
for INDEX in 0 1 2 3 4 5 6 7; do
  if ! wait "${PIDS[$INDEX]}"; then
    echo "detached semantic rank $INDEX failed" >&2
    FAILED=1
  fi
done
if [[ "$FAILED" -ne 0 ]]; then
  exit 1
fi
PYTHONPATH="$CODE_DIR" "$ENV_DIR/bin/python" \
  "$CODE_DIR/aggregate_detached_lora_semantic_gate.py" \
  "$RUN_DIR" \
  --expected-world-size 8 \
  --expected-data-sha256 "$EXPECTED_DATA_SHA256" \
  --expected-checkpoint-sha256 "$CHECKPOINT_SHA256" \
  --expected-query-positions 128 \
  > "$RUN_DIR/logs/detached-semantic-gate.log" 2>&1
test -s "$RUN_DIR/detached-semantic-gate.json"
date -u +%FT%TZ > "$RUN_DIR/stages/03_semantic_ok"

nvidia-smi \
  --query-gpu=index,uuid,name,memory.used,pstate,power.draw,temperature.gpu \
  --format=csv > "$RUN_DIR/gpus-after.csv"
date -u +%FT%TZ > "$RUN_DIR/stages/99_done"
echo "Detached-document-cache LoRA capability passed: $RUN_DIR"
