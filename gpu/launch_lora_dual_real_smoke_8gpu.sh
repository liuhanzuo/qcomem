#!/usr/bin/env bash
set -euo pipefail

CODE_DIR=${CODE_DIR:?set CODE_DIR}
MODEL_DIR=${MODEL_DIR:?set MODEL_DIR}
DATA_FILE=${DATA_FILE:?set DATA_FILE}
RUN_DIR=${RUN_DIR:?set RUN_DIR}
ENV_DIR=${ENV_DIR:?set ENV_DIR}

INTERFACE_CONFIG=${INTERFACE_CONFIG:-$CODE_DIR/configs/lora_interface_real_smoke_1.json}
QUANT_CONFIG=${QUANT_CONFIG:-$CODE_DIR/configs/lora_quant_real_smoke_1.json}

CONFIG_FILE="$INTERFACE_CONFIG" RUN_DIR="$RUN_DIR/interface" \
  CODE_DIR="$CODE_DIR" MODEL_DIR="$MODEL_DIR" DATA_FILE="$DATA_FILE" \
  ENV_DIR="$ENV_DIR" bash "$CODE_DIR/launch_lora_8gpu.sh"

CONFIG_FILE="$QUANT_CONFIG" RUN_DIR="$RUN_DIR/quant" \
  CODE_DIR="$CODE_DIR" MODEL_DIR="$MODEL_DIR" DATA_FILE="$DATA_FILE" \
  ENV_DIR="$ENV_DIR" bash "$CODE_DIR/launch_lora_8gpu.sh"

date -u +%FT%TZ > "$RUN_DIR/99_dual_smoke_done"
