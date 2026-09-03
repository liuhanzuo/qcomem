#!/usr/bin/env bash
set -euo pipefail

CODE_DIR=${CODE_DIR:?set CODE_DIR}
MODEL_DIR=${MODEL_DIR:?set MODEL_DIR}
DATA_FILE=${DATA_FILE:?set DATA_FILE}
RUN_DIR=${RUN_DIR:?set RUN_DIR}
ENV_DIR=${ENV_DIR:?set ENV_DIR}
CONFIG_FILE=${CONFIG_FILE:?set CONFIG_FILE}
RESUME_FILE=${RESUME_FILE:-}
INIT_ADAPTER_FILE=${INIT_ADAPTER_FILE:-}
EXPECTED_DATA_SHA256=${EXPECTED_DATA_SHA256:-}

mkdir -p "$RUN_DIR/logs" "$RUN_DIR/stages"
date -u +%FT%TZ > "$RUN_DIR/stages/00_start"
GPU_COUNT=$(nvidia-smi -L | wc -l | tr -d ' ')
if [[ "$GPU_COUNT" -ne 8 ]]; then
  echo "expected 8 GPUs, found $GPU_COUNT" >&2
  exit 1
fi
nvidia-smi --query-gpu=index,name,memory.total --format=csv > "$RUN_DIR/gpus.csv"
sha256sum \
  "$CODE_DIR/qcomem_torch.py" \
  "$CODE_DIR/qcomem_lora.py" \
  "$CODE_DIR/train_qcomem_lora.py" \
  "$CODE_DIR/launch_lora_8gpu.sh" \
  "$CONFIG_FILE" > "$RUN_DIR/code.sha256"
sha256sum "$DATA_FILE" > "$RUN_DIR/training-data.sha256"
if [[ -n "$EXPECTED_DATA_SHA256" ]]; then
  ACTUAL_DATA_SHA256=$(sha256sum "$DATA_FILE" | awk '{print $1}')
  if [[ "$ACTUAL_DATA_SHA256" != "$EXPECTED_DATA_SHA256" ]]; then
    echo "training data SHA256 mismatch" >&2
    exit 1
  fi
fi
"$ENV_DIR/bin/python" -m py_compile \
  "$CODE_DIR/qcomem_torch.py" \
  "$CODE_DIR/qcomem_lora.py" \
  "$CODE_DIR/train_qcomem_lora.py"
PYTHONPATH="$CODE_DIR" "$ENV_DIR/bin/python" -m unittest discover \
  -s "$CODE_DIR" -p 'test_qcomem_lora.py' -v \
  > "$RUN_DIR/logs/preflight-tests.log" 2>&1
date -u +%FT%TZ > "$RUN_DIR/stages/01_preflight_ok"

RESUME_ARGS=()
if [[ -n "$RESUME_FILE" && -n "$INIT_ADAPTER_FILE" ]]; then
  echo "RESUME_FILE and INIT_ADAPTER_FILE are mutually exclusive" >&2
  exit 1
elif [[ -n "$RESUME_FILE" ]]; then
  RESUME_ARGS=(--resume "$RESUME_FILE")
elif [[ -n "$INIT_ADAPTER_FILE" ]]; then
  RESUME_ARGS=(--init-adapter "$INIT_ADAPTER_FILE")
fi

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
  "${RESUME_ARGS[@]}" \
  > "$RUN_DIR/logs/train.log" 2>&1

test -s "$RUN_DIR/metadata.json"
test -s "$RUN_DIR/latest"
date -u +%FT%TZ > "$RUN_DIR/stages/99_done"
echo "Q-CoMem suffix LoRA smoke complete: $RUN_DIR"
