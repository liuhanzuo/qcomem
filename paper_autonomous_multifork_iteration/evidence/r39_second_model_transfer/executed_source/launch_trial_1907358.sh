#!/usr/bin/env bash
set -euo pipefail

# Exact wrapper for the already allocated Trial 1907358.  It neither creates,
# stops, nor evicts a QS resource.  The source/static digests below are filled
# only after the reviewed package is frozen; this bootstrap wrapper itself is
# intentionally excluded from the source manifest to avoid a hash cycle.

R39_STAGE=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench/qcomem_r39_second_model_transfer_20260826a
ASSET_ROOT=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets
ENV_DIR="$ASSET_ROOT/envs/vllm-cu129-v1"
MODEL_ROOT="$ASSET_ROOT/models/Qwen3.5-0.8B-2fc06364715b967f1860aea9cf38778875588b17"
RUN_ROOT="$ASSET_ROOT/runs/qcomem/r39-second-model-transfer-20260826a"

EXPECTED_SOURCE_SHA256=98ce42e7dd1ab70c35edab177488b94292864979831199d596e392b1d559a7d9
EXPECTED_STATIC_SHA256=6eb585539aa4311583d75b291b1d8fc5fd90f4e371fd7dee3dcae3ec16638d49

REPO_ROOT="$R39_STAGE" \
ENV_DIR="$ENV_DIR" \
MODEL_ROOT="$MODEL_ROOT" \
RUN_ROOT="$RUN_ROOT" \
EXPECTED_SOURCE_SHA256="$EXPECTED_SOURCE_SHA256" \
EXPECTED_STATIC_SHA256="$EXPECTED_STATIC_SHA256" \
bash "$R39_STAGE/paper_autonomous_multifork_iteration/evidence/r39_second_model_transfer/executed_source/launch_r39_second_model_transfer_8gpu.sh"
