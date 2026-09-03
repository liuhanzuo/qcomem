#!/usr/bin/env bash
set -euo pipefail

export LC_ALL=C
export PYTHONDONTWRITEBYTECODE=1

CODE_DIR=${CODE_DIR:?set immutable CODE_DIR}
MODEL_DIR=${MODEL_DIR:?set MODEL_DIR}
MODEL_ARTIFACT_LEDGER_FILE=${MODEL_ARTIFACT_LEDGER_FILE:?set MODEL_ARTIFACT_LEDGER_FILE}
MODEL_WEIGHT_LEDGER_FILE=${MODEL_WEIGHT_LEDGER_FILE:?set MODEL_WEIGHT_LEDGER_FILE}
PG19_DATA=${PG19_DATA:?set PG19_DATA}
PG19_MANIFEST=${PG19_MANIFEST:?set PG19_MANIFEST}
STATIC_MANIFEST=${STATIC_MANIFEST:?set preregistered STATIC_MANIFEST}
RUN_DIR=${RUN_DIR:?set a fresh RUN_DIR}
ENV_DIR=${ENV_DIR:?set frozen ENV_DIR}
EXPECTED_CODE_LEDGER_SHA256=${EXPECTED_CODE_LEDGER_SHA256:?set code ledger raw SHA}
EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256=${EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256:?set model ledger raw SHA}
EXPECTED_MODEL_WEIGHT_LEDGER_SHA256=${EXPECTED_MODEL_WEIGHT_LEDGER_SHA256:?set model weight ledger raw SHA}
EXPECTED_STATIC_MANIFEST_SHA256=${EXPECTED_STATIC_MANIFEST_SHA256:?set static manifest raw SHA}
EXPECTED_WINDOWS_SHA256=${EXPECTED_WINDOWS_SHA256:?set aligned input windows SHA}
EXPECTED_PG19_SHA256=${EXPECTED_PG19_SHA256:?set PG19 raw SHA}
EXPECTED_PG19_MANIFEST_SHA256=${EXPECTED_PG19_MANIFEST_SHA256:?set PG19 manifest raw SHA}

PYTHON="$ENV_DIR/bin/python"
RUNNER="$CODE_DIR/run_qcomem_qwen35_forkaudit_lifecycle_transfer.py"
CODE_LEDGER="$CODE_DIR/code.sha256"

if [[ -e "$RUN_DIR" ]]; then
  echo "RUN_DIR already exists: $RUN_DIR" >&2
  exit 2
fi
for value in \
  "$EXPECTED_CODE_LEDGER_SHA256" "$EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256" \
  "$EXPECTED_MODEL_WEIGHT_LEDGER_SHA256" \
  "$EXPECTED_STATIC_MANIFEST_SHA256" "$EXPECTED_WINDOWS_SHA256" \
  "$EXPECTED_PG19_SHA256" "$EXPECTED_PG19_MANIFEST_SHA256"; do
  [[ "$value" =~ ^[0-9a-f]{64}$ ]] || { echo "invalid frozen SHA: $value" >&2; exit 2; }
done
[[ $(sha256sum "$CODE_LEDGER" | awk '{print $1}') == "$EXPECTED_CODE_LEDGER_SHA256" ]]
[[ $(sha256sum "$MODEL_ARTIFACT_LEDGER_FILE" | awk '{print $1}') == "$EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256" ]]
[[ $(sha256sum "$MODEL_WEIGHT_LEDGER_FILE" | awk '{print $1}') == "$EXPECTED_MODEL_WEIGHT_LEDGER_SHA256" ]]
[[ $(sha256sum "$STATIC_MANIFEST" | awk '{print $1}') == "$EXPECTED_STATIC_MANIFEST_SHA256" ]]
[[ $(sha256sum "$PG19_DATA" | awk '{print $1}') == "$EXPECTED_PG19_SHA256" ]]
[[ $(sha256sum "$PG19_MANIFEST" | awk '{print $1}') == "$EXPECTED_PG19_MANIFEST_SHA256" ]]

mkdir -p "$RUN_DIR/logs" "$RUN_DIR/raw/shards" "$RUN_DIR/receipts" "$RUN_DIR/stages" "$RUN_DIR/pycache"
export PYTHONPYCACHEPREFIX="$RUN_DIR/pycache"
date -u +%FT%TZ > "$RUN_DIR/stages/00-started"

(
  cd "$CODE_DIR"
  sha256sum -c code.sha256 > "$RUN_DIR/logs/code-ledger-check.log"
)
# This is the one formal launch; the full model ledger check is part of it,
# not a separate GPU smoke or a post-result integrity assertion.
(
  cd "$MODEL_DIR"
  sha256sum -c "$MODEL_ARTIFACT_LEDGER_FILE" > "$RUN_DIR/logs/model-artifact-check.log"
  sha256sum -c "$MODEL_WEIGHT_LEDGER_FILE" > "$RUN_DIR/logs/model-weight-check.log"
)
date -u +%FT%TZ > "$RUN_DIR/stages/01-preflight-passed"

mapfile -t GPU_UUIDS < <(nvidia-smi --query-gpu=uuid --format=csv,noheader | sed 's/[[:space:]]//g')
[[ ${#GPU_UUIDS[@]} -eq 8 ]] || { echo "expected exactly eight GPUs" >&2; exit 2; }
printf '%s\n' "${GPU_UUIDS[@]}" > "$RUN_DIR/receipts/gpu-uuids.txt"

pids=()
for rank in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=${GPU_UUIDS[$rank]} "$PYTHON" -B "$RUNNER" \
    --stage shard \
    --rank "$rank" \
    --expected-gpu-uuid "${GPU_UUIDS[$rank]}" \
    --model "$MODEL_DIR" \
    --pg19-data "$PG19_DATA" \
    --pg19-manifest "$PG19_MANIFEST" \
    --expected-pg19-sha256 "$EXPECTED_PG19_SHA256" \
    --expected-pg19-manifest-sha256 "$EXPECTED_PG19_MANIFEST_SHA256" \
    --expected-windows-sha256 "$EXPECTED_WINDOWS_SHA256" \
    --static-manifest "$STATIC_MANIFEST" \
    --expected-static-manifest-sha256 "$EXPECTED_STATIC_MANIFEST_SHA256" \
    --expected-code-ledger-sha256 "$EXPECTED_CODE_LEDGER_SHA256" \
    --expected-model-weight-ledger-sha256 "$EXPECTED_MODEL_WEIGHT_LEDGER_SHA256" \
    --output "$RUN_DIR/raw/shards/forkaudit-lifecycle-shard-$rank.json" \
    > "$RUN_DIR/logs/rank-$rank.log" 2>&1 &
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

"$PYTHON" -B "$RUNNER" \
  --stage aggregate \
  --static-manifest "$STATIC_MANIFEST" \
  --expected-static-manifest-sha256 "$EXPECTED_STATIC_MANIFEST_SHA256" \
  --expected-model-weight-ledger-sha256 "$EXPECTED_MODEL_WEIGHT_LEDGER_SHA256" \
  --shard-dir "$RUN_DIR/raw/shards" \
  --output "$RUN_DIR/forkaudit-lifecycle-summary.json" \
  > "$RUN_DIR/logs/aggregate.log" 2>&1

(
  cd "$RUN_DIR"
  find raw -type f -print0 | sort -z | xargs -0 sha256sum > receipts/raw-artifacts.sha256
)
date -u +%FT%TZ > "$RUN_DIR/stages/03-aggregate-passed"
date -u +%FT%TZ > "$RUN_DIR/stages/COMPLETED"
