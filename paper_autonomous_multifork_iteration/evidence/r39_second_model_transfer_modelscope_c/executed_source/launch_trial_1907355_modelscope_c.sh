#!/usr/bin/env bash
set -euo pipefail

# Exact wrapper for the already allocated Trial 1907355.  It neither creates,
# stops, nor evicts a QS resource.  The source/static digests below are filled
# only after the reviewed package is frozen; this bootstrap wrapper itself is
# intentionally excluded from the source manifest to avoid a hash cycle.

R39_STAGE=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench/qcomem_r39_second_model_transfer_20260826c
ASSET_ROOT=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets
ENV_DIR="$ASSET_ROOT/envs/vllm-cu129-v1"
MODEL_ROOT="$ASSET_ROOT/models/Qwen3.5-0.8B-hf-2fc06364715b967f1860aea9cf38778875588b17-modelscope-4d58a7b524cd33ed843d5125be8cd8f0a452d9bf-20260826c"
RUN_ROOT="$ASSET_ROOT/runs/qcomem/r39-second-model-transfer-20260826c"

EXPECTED_SOURCE_SHA256=cc1c0b47642e88c126e817f4c3b1f438ab27b4ecaa88667667e9711b754ceba8
EXPECTED_STATIC_SHA256=3934058526c73b28efcad2dd7e6f5aeb994070b4152eedc92bc8140f321b3fec

REPO_ROOT="$R39_STAGE" \
ENV_DIR="$ENV_DIR" \
MODEL_ROOT="$MODEL_ROOT" \
RUN_ROOT="$RUN_ROOT" \
EXPECTED_SOURCE_SHA256="$EXPECTED_SOURCE_SHA256" \
EXPECTED_STATIC_SHA256="$EXPECTED_STATIC_SHA256" \
bash "$R39_STAGE/paper_autonomous_multifork_iteration/evidence/r39_second_model_transfer_modelscope_c/executed_source/launch_r39_second_model_transfer_8gpu.sh"
