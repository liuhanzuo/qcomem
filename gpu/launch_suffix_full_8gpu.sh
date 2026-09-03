#!/usr/bin/env bash
set -euo pipefail

CODE_DIR=${CODE_DIR:?set CODE_DIR}
MODEL_DIR=${MODEL_DIR:?set MODEL_DIR}
DATA_FILE=${DATA_FILE:?set DATA_FILE}
RUN_DIR=${RUN_DIR:?set RUN_DIR}
ENV_DIR=${ENV_DIR:?set ENV_DIR}
CONFIG_FILE=${CONFIG_FILE:?set CONFIG_FILE}
EXPECTED_DATA_SHA256=${EXPECTED_DATA_SHA256:?set EXPECTED_DATA_SHA256}

mkdir -p "$RUN_DIR/logs" "$RUN_DIR/stages"
date -u +%FT%TZ > "$RUN_DIR/stages/00_start"
GPU_COUNT=$(nvidia-smi -L | wc -l | tr -d ' ')
if [[ "$GPU_COUNT" -ne 8 ]]; then
  echo "suffix_full_distillation requires exactly 8 GPUs, found $GPU_COUNT" >&2
  exit 1
fi
nvidia-smi --query-gpu=index,name,memory.total --format=csv > "$RUN_DIR/gpus.csv"
sha256sum \
  "$CODE_DIR/qcomem_torch.py" \
  "$CODE_DIR/qcomem_lora.py" \
  "$CODE_DIR/qcomem_suffix_full.py" \
  "$CODE_DIR/train_qcomem_suffix_full.py" \
  "$CODE_DIR/launch_suffix_full_8gpu.sh" \
  "$CONFIG_FILE" > "$RUN_DIR/code.sha256"
ACTUAL_DATA_SHA256=$(sha256sum "$DATA_FILE" | awk '{print $1}')
if [[ "$ACTUAL_DATA_SHA256" != "$EXPECTED_DATA_SHA256" ]]; then
  echo "training data SHA256 mismatch" >&2
  exit 1
fi
printf '%s  %s\n' "$ACTUAL_DATA_SHA256" "$DATA_FILE" > "$RUN_DIR/training-data.sha256"

"$ENV_DIR/bin/python" -m py_compile \
  "$CODE_DIR/qcomem_torch.py" \
  "$CODE_DIR/qcomem_lora.py" \
  "$CODE_DIR/qcomem_suffix_full.py" \
  "$CODE_DIR/train_qcomem_suffix_full.py"
PYTHONPATH="$CODE_DIR" "$ENV_DIR/bin/python" -m unittest \
  test_qcomem_lora test_qcomem_suffix_full -v \
  > "$RUN_DIR/logs/preflight-tests.log" 2>&1
"$ENV_DIR/bin/python" -c 'from torch.distributed.fsdp import FullyShardedDataParallel, ShardingStrategy; from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper, CheckpointImpl; assert ShardingStrategy.FULL_SHARD is not None; assert CheckpointImpl.NO_REENTRANT is not None' \
  > "$RUN_DIR/logs/fsdp-import-preflight.log" 2>&1
date -u +%FT%TZ > "$RUN_DIR/stages/01_preflight_ok"

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
"$ENV_DIR/bin/torchrun" \
  --standalone \
  --nproc-per-node=8 \
  "$CODE_DIR/train_qcomem_suffix_full.py" \
  --config "$CONFIG_FILE" \
  --model "$MODEL_DIR" \
  --data "$DATA_FILE" \
  --output-dir "$RUN_DIR" \
  > "$RUN_DIR/logs/train.log" 2>&1

test -s "$RUN_DIR/metadata.json"
test -s "$RUN_DIR/metrics.jsonl"
date -u +%FT%TZ > "$RUN_DIR/stages/99_done"
echo "Q-CoMem suffix_full_distillation FSDP smoke complete: $RUN_DIR"
