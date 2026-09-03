#!/usr/bin/env bash
set -euo pipefail

# Formal one-H20 capture on the already allocated QS node.  This
# script neither creates, stops, nor deletes any QS resource.  It runs the
# pre-existing R29 one-GPU audited adapter under the Round-39 hook wrapper.

R39_STAGE=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench/qcomem_r39_compiled_dispatch_20260826g
R29_STAGE=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench/qcomem_r29_live_overhead_20260825b
UPSTREAM_CODE_DIR=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench/qcomem_gpu_forkaudit_lifecycle_transfer_20260819c/gpu
ASSET_ROOT=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets
RUN_ROOT="$ASSET_ROOT/runs/qcomem/r39-compiled-dispatch-20260826g"
ARGS_JSON="$ASSET_ROOT/runs/qcomem/r39-compiled-dispatch-20260826g-args.json"

PYTHON="$ASSET_ROOT/envs/vllm-cu129-v1/bin/python"
RUNTIME_ROOT="$ASSET_ROOT/envs/vllm-cu129-v1/lib/python3.11/site-packages"
MODEL_DIR="$ASSET_ROOT/models/Qwen3.5-35B-A3B-59d61f3"
MODEL_ARTIFACT_LEDGER="$MODEL_DIR/model-artifacts.sha256"
MODEL_WEIGHT_LEDGER="$MODEL_DIR/model-weights.sha256"
PG19_DATA="$ASSET_ROOT/data/pg19/qcomem_pg19_train_smoke64.jsonl"
PG19_MANIFEST="$ASSET_ROOT/data/pg19/qcomem_pg19_train_smoke64.manifest.json"
DESIGN="$R29_STAGE/paper_autonomous_multifork_iteration/evidence/r29_live_overhead/preregistration.json"

[[ ! -e "$RUN_ROOT" ]] || { echo "RUN_ROOT already exists: $RUN_ROOT" >&2; exit 2; }
[[ ! -e "$ARGS_JSON" ]] || { echo "argument file already exists: $ARGS_JSON" >&2; exit 2; }
for item in "$PYTHON" "$RUNTIME_ROOT" "$MODEL_DIR" "$MODEL_ARTIFACT_LEDGER" "$MODEL_WEIGHT_LEDGER" "$PG19_DATA" "$PG19_MANIFEST" "$DESIGN" "$UPSTREAM_CODE_DIR/code.sha256"; do
  [[ -e "$item" ]] || { echo "required path absent: $item" >&2; exit 2; }
done
(cd "$UPSTREAM_CODE_DIR" && sha256sum -c code.sha256)

"$PYTHON" - "$ARGS_JSON" "$RUN_ROOT" "$DESIGN" "$MODEL_DIR" "$MODEL_ARTIFACT_LEDGER" "$MODEL_WEIGHT_LEDGER" "$PG19_DATA" "$PG19_MANIFEST" "$UPSTREAM_CODE_DIR/code.sha256" <<'PY'
import json
import sys

(output, run_root, design, model, artifact_ledger, weight_ledger, pg19_data,
 pg19_manifest, code_ledger) = sys.argv[1:]
args = [
    "--stage", "formal",
    "--design-preregistration", design,
    "--expected-design-sha256", "2114d2cd85bedc1eafa5d1398fd0afd0d57819c0360c3be3f9ec20f1b2878939",
    "--model", model,
    "--model-artifact-ledger", artifact_ledger,
    "--model-weight-ledger", weight_ledger,
    "--pg19-data", pg19_data,
    "--pg19-manifest", pg19_manifest,
    "--upstream-code-ledger", code_ledger,
    "--artifact-dir", run_root + "/raw/r29-audit",
    "--output", run_root + "/raw/r29-formal-result.json",
]
with open(output, "w", encoding="utf-8") as handle:
    json.dump(args, handle, separators=(",", ":"))
    handle.write("\n")
PY

CUDA_VISIBLE_DEVICES=0 \
TOKENIZERS_PARALLELISM=false \
PYTHON="$PYTHON" \
CODE_ROOT="$UPSTREAM_CODE_DIR" \
RUNTIME_ROOT="$RUNTIME_ROOT" \
ENTRYPOINT_ROOT="$R29_STAGE/gpu" \
ENTRYPOINT_RELATIVE=r29_live_overhead.py \
ENTRYPOINT_ARGS_JSON="$ARGS_JSON" \
OUTPUT_ROOT="$RUN_ROOT" \
R29_RESULT="$RUN_ROOT/raw/r29-formal-result.json" \
R29_SEMANTIC_SIDECAR="$RUN_ROOT/raw/r29-audit/semantic-logits.fp32.bin" \
DESIGN_PREREGISTRATION="$DESIGN" \
bash "$R39_STAGE/paper_autonomous_multifork_iteration/evidence/r39_compiled_dispatch/executed_source/r39_launch_h20.sh"
