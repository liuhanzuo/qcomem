#!/usr/bin/env bash
set -euo pipefail

# Frozen launcher for the fresh formal run.  The completed debug run is never
# copied into this output directory and is never eligible for aggregation.
A2_DIR=${A2_DIR:?set immutable A2_DIR}
UPSTREAM_CODE_DIR=${UPSTREAM_CODE_DIR:?set frozen upstream code directory}
MODEL_DIR=${MODEL_DIR:?set MODEL_DIR}
PG19_DATA=${PG19_DATA:?set PG19_DATA}
PG19_MANIFEST=${PG19_MANIFEST:?set PG19_MANIFEST}
ENV_DIR=${ENV_DIR:?set frozen ENV_DIR}
RUN_DIR=${RUN_DIR:?set a fresh formal RUN_DIR}

DESIGN="$A2_DIR/a2_scheduler_interleave_design_preregistration_b.json"
STATIC="$A2_DIR/static-input-preregistration-b.json"
RUNNER="$A2_DIR/run_qcomem_qwen35_forkaudit_scheduler_interleave.py"
HELPER="$A2_DIR/qcomem_forkaudit_scheduler_contract.py"
PYTHON="$ENV_DIR/bin/python"
MODEL_ARTIFACT_LEDGER="$MODEL_DIR/model-artifacts.sha256"
MODEL_WEIGHT_LEDGER="$MODEL_DIR/model-weights.sha256"
UPSTREAM_LEDGER="$UPSTREAM_CODE_DIR/code.sha256"

DESIGN_SHA=c7e80ff62d68a2d942888d3f0a1c1027d69180ae6aa1726bb49aeccc38019847
STATIC_SHA=2c7480e9301860fd24a87fa8aa05b25360456824181cc9b456b0ee0b855a85eb
RUNNER_SHA=8a53591e53d4b9ff1efafca60fd2c42f48986c9b9719d60a93dc5a49549f32f4
HELPER_SHA=e9eb78d7981bf2c6a56032774a3bb64904e6e9892dcc16d07e2fb6b42205617f
UPSTREAM_LEDGER_SHA=7620f05821fc5435a9aaa260ae82577988a5a20eff0f42901b66e6c6871fd2b9
MODEL_ARTIFACT_LEDGER_SHA=c0a23e9d3f9d220257af97b78fd97661f315f0c82a3a010b57a771e3eeefbbfb
MODEL_WEIGHT_LEDGER_SHA=8314a82c9188b9b817193e039b0b0eb0636b328512f19b0c12455853b7e20014
PG19_SHA=ef18893b6bfb0f4b8cb29eab85ccf2a0ef1fdb44606e9742a70405cca564e18c
PG19_MANIFEST_SHA=5d789d67aa239f089e92de8a4267b86d2f1d2723d5f1370970883738f5f89a9c

[[ ! -e "$RUN_DIR" ]] || { echo "formal RUN_DIR already exists: $RUN_DIR" >&2; exit 2; }
[[ $(sha256sum "$DESIGN" | awk '{print $1}') == "$DESIGN_SHA" ]]
[[ $(sha256sum "$STATIC" | awk '{print $1}') == "$STATIC_SHA" ]]
[[ $(sha256sum "$RUNNER" | awk '{print $1}') == "$RUNNER_SHA" ]]
[[ $(sha256sum "$HELPER" | awk '{print $1}') == "$HELPER_SHA" ]]
[[ $(sha256sum "$UPSTREAM_LEDGER" | awk '{print $1}') == "$UPSTREAM_LEDGER_SHA" ]]
[[ $(sha256sum "$MODEL_ARTIFACT_LEDGER" | awk '{print $1}') == "$MODEL_ARTIFACT_LEDGER_SHA" ]]
[[ $(sha256sum "$MODEL_WEIGHT_LEDGER" | awk '{print $1}') == "$MODEL_WEIGHT_LEDGER_SHA" ]]
[[ $(sha256sum "$PG19_DATA" | awk '{print $1}') == "$PG19_SHA" ]]
[[ $(sha256sum "$PG19_MANIFEST" | awk '{print $1}') == "$PG19_MANIFEST_SHA" ]]

mkdir -p "$RUN_DIR/logs" "$RUN_DIR/raw/shards" "$RUN_DIR/receipts" "$RUN_DIR/stages"
date -u +%FT%TZ > "$RUN_DIR/stages/00-started"
(cd "$UPSTREAM_CODE_DIR" && sha256sum -c code.sha256 > "$RUN_DIR/logs/upstream-code-ledger-check.log")
date -u +%FT%TZ > "$RUN_DIR/stages/01-preflight-passed"

mapfile -t GPU_UUIDS < <(nvidia-smi --query-gpu=uuid --format=csv,noheader | sed 's/[[:space:]]//g')
[[ ${#GPU_UUIDS[@]} -eq 8 ]] || { echo "formal run requires exactly eight visible GPUs" >&2; exit 2; }
printf '%s\n' "${GPU_UUIDS[@]}" > "$RUN_DIR/receipts/gpu-uuids.txt"

pids=()
for rank in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=${GPU_UUIDS[$rank]} \
  PYTHONPATH="$A2_DIR:$UPSTREAM_CODE_DIR" \
  "$PYTHON" -B "$RUNNER" \
    --stage formal-shard \
    --rank "$rank" \
    --design-preregistration "$DESIGN" \
    --expected-design-sha256 "$DESIGN_SHA" \
    --static-manifest "$STATIC" \
    --expected-static-sha256 "$STATIC_SHA" \
    --model "$MODEL_DIR" \
    --model-artifact-ledger "$MODEL_ARTIFACT_LEDGER" \
    --model-weight-ledger "$MODEL_WEIGHT_LEDGER" \
    --pg19-data "$PG19_DATA" \
    --pg19-manifest "$PG19_MANIFEST" \
    --upstream-code-ledger "$UPSTREAM_LEDGER" \
    --output "$RUN_DIR/raw/shards/scheduler-interleave-formal-shard-$rank.json" \
    > "$RUN_DIR/logs/rank-$rank.stdout.log" \
    2> "$RUN_DIR/logs/rank-$rank.stderr.log" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=$?
done
if [[ $status -ne 0 ]]; then
  date -u +%FT%TZ > "$RUN_DIR/stages/FAILED-shards"
  exit "$status"
fi
date -u +%FT%TZ > "$RUN_DIR/stages/02-shards-passed"

PYTHONPATH="$A2_DIR:$UPSTREAM_CODE_DIR" "$PYTHON" -B "$RUNNER" \
  --stage formal-aggregate \
  --design-preregistration "$DESIGN" \
  --expected-design-sha256 "$DESIGN_SHA" \
  --static-manifest "$STATIC" \
  --expected-static-sha256 "$STATIC_SHA" \
  --upstream-code-ledger "$UPSTREAM_LEDGER" \
  --shard-dir "$RUN_DIR/raw/shards" \
  --output "$RUN_DIR/scheduler-interleave-formal-summary.json" \
  > "$RUN_DIR/logs/aggregate.stdout.log" \
  2> "$RUN_DIR/logs/aggregate.stderr.log"

(cd "$RUN_DIR" && find raw -type f -print0 | sort -z | xargs -0 sha256sum > receipts/raw-artifacts.sha256)
date -u +%FT%TZ > "$RUN_DIR/stages/COMPLETED"
