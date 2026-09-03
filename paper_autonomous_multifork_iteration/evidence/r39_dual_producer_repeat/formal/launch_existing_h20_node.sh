#!/usr/bin/env bash
set -euo pipefail

# Single-command wrapper for an already-created eight-H20 node.  It allocates,
# stops, and deletes no QS resource.  The two producer runs are serial and use
# only the one selected GPU.

DUAL_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SHARED=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo

export R33_CODE_DIR="${R33_CODE_DIR:-$DUAL_ROOT/vendor/r33}"
export R29_CODE_DIR="${R29_CODE_DIR:-$DUAL_ROOT/vendor/r29}"
export IMPORTED_RR2_CODE_DIR="${IMPORTED_RR2_CODE_DIR:-$SHARED/indep-bench/qcomem_gpu_forkaudit_rr2_formal_20260818w/gpu}"
export IMPORTED_RR2_CODE_LEDGER_FILE="${IMPORTED_RR2_CODE_LEDGER_FILE:-$SHARED/indep-bench/qcomem_gpu_forkaudit_rr2_formal_20260818w-inputs/code.sha256}"
export ENV_DIR="${ENV_DIR:-$SHARED/indep-bench_assets/envs/vllm-cu129-v1}"
export MODEL_DIR="${MODEL_DIR:-$SHARED/indep-bench_assets/models/Qwen3.5-35B-A3B-59d61f3}"
export MODEL_WEIGHT_LEDGER_FILE="${MODEL_WEIGHT_LEDGER_FILE:-$SHARED/indep-bench/qcomem_gpu_forkaudit_rr2_formal_20260818w-inputs/model-weights.canonical.sha256}"
export MODEL_ARTIFACT_LEDGER_FILE="${MODEL_ARTIFACT_LEDGER_FILE:-$SHARED/indep-bench/qcomem_gpu_forkaudit_rr2_formal_20260818w-inputs/model-artifacts.formal.sha256}"
export PG19_DATA="${PG19_DATA:-$SHARED/indep-bench_assets/data/pg19/qcomem_pg19_train_smoke64.jsonl}"
export PG19_MANIFEST="${PG19_MANIFEST:-$SHARED/indep-bench_assets/data/pg19/qcomem_pg19_train_smoke64.manifest.json}"
export FROZEN_QUERY_BANKS="${FROZEN_QUERY_BANKS:-$SHARED/indep-bench/qcomem_gpu_forkaudit_rr2_formal_20260818w-inputs/rr2-frozen-query-banks.json}"
export DUAL_GPU_INDEX="${DUAL_GPU_INDEX:-7}"
export DUAL_RUN_DIR="${DUAL_RUN_DIR:-$SHARED/indep-bench_assets/runs/qcomem/r39-dual-producer-repeat-20260826a}"

exec bash "$DUAL_ROOT/formal/launch_dual_producer_h20.sh"
