#!/usr/bin/env bash
set -euo pipefail

CODE_DIR=${CODE_DIR:?set CODE_DIR}
RR2_CODE_DIR=${RR2_CODE_DIR:?set RR2_CODE_DIR}
MODEL_DIR=${MODEL_DIR:?set MODEL_DIR}
MODEL_WEIGHT_LEDGER_FILE=${MODEL_WEIGHT_LEDGER_FILE:?set MODEL_WEIGHT_LEDGER_FILE}
PG19_DATA=${PG19_DATA:?set PG19_DATA}
PG19_MANIFEST=${PG19_MANIFEST:?set PG19_MANIFEST}
PREREGISTRATION=${PREREGISTRATION:?set PREREGISTRATION}
SOURCE_LEDGER_FILE=${SOURCE_LEDGER_FILE:?set SOURCE_LEDGER_FILE}
PREEXECUTION_PIN=${PREEXECUTION_PIN:?set PREEXECUTION_PIN}
RUN_DIR=${RUN_DIR:?set RUN_DIR}
ENV_DIR=${ENV_DIR:?set ENV_DIR}
EXPECTED_RUNNER_SHA256=${EXPECTED_RUNNER_SHA256:?set EXPECTED_RUNNER_SHA256}
EXPECTED_REFERENCE_SHA256=${EXPECTED_REFERENCE_SHA256:?set EXPECTED_REFERENCE_SHA256}
EXPECTED_PREREGISTRATION_SHA256=${EXPECTED_PREREGISTRATION_SHA256:?set EXPECTED_PREREGISTRATION_SHA256}
EXPECTED_SOURCE_LEDGER_SHA256=${EXPECTED_SOURCE_LEDGER_SHA256:?set EXPECTED_SOURCE_LEDGER_SHA256}
EXPECTED_PREEXECUTION_PIN_SHA256=${EXPECTED_PREEXECUTION_PIN_SHA256:?set EXPECTED_PREEXECUTION_PIN_SHA256}
EXPECTED_MODEL_WEIGHT_LEDGER_SHA256=${EXPECTED_MODEL_WEIGHT_LEDGER_SHA256:?set EXPECTED_MODEL_WEIGHT_LEDGER_SHA256}

PYTHON="$ENV_DIR/bin/python"
RUNNER="$CODE_DIR/run_qcomem_qwen35_gdn_transition_oracle_preregistered.py"
REFERENCE="$CODE_DIR/qcomem_gdn_transition_oracle_reference_preregistered.py"

if [[ -e "$RUN_DIR" ]]; then
  echo "RUN_DIR must be absent: $RUN_DIR" >&2
  exit 2
fi
mkdir -p "$RUN_DIR/stages"
date -u +%FT%TZ > "$RUN_DIR/stages/00-started"

verify_sha() {
  local path=$1 expected=$2 label=$3 actual
  actual=$(sha256sum "$path" | awk '{print $1}')
  if [[ "$actual" != "$expected" ]]; then
    echo "$label SHA mismatch: expected=$expected actual=$actual" >&2
    exit 2
  fi
}

verify_sha "$RUNNER" "$EXPECTED_RUNNER_SHA256" runner
verify_sha "$REFERENCE" "$EXPECTED_REFERENCE_SHA256" reference
verify_sha "$PREREGISTRATION" "$EXPECTED_PREREGISTRATION_SHA256" preregistration
verify_sha "$SOURCE_LEDGER_FILE" "$EXPECTED_SOURCE_LEDGER_SHA256" source-ledger
verify_sha "$PREEXECUTION_PIN" "$EXPECTED_PREEXECUTION_PIN_SHA256" preexecution-pin
verify_sha "$MODEL_WEIGHT_LEDGER_FILE" "$EXPECTED_MODEL_WEIGHT_LEDGER_SHA256" model-weight-ledger
(cd "$CODE_DIR" && sha256sum -c "$SOURCE_LEDGER_FILE")
if grep -Eq '^(import torch|from torch|import qcomem|from qcomem)' "$REFERENCE"; then
  echo "candidate-code-free reference import gate failed" >&2
  exit 2
fi
if [[ "${CUDA_VISIBLE_DEVICES:-}" != "0" ]]; then
  echo "formal oracle must expose only package-local GPU 0" >&2
  exit 2
fi
"$PYTHON" -I -c 'import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 1; assert "H20" in torch.cuda.get_device_properties(0).name'

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$CODE_DIR:$RR2_CODE_DIR"
date -u +%FT%TZ > "$RUN_DIR/stages/01-preflight-passed"

timeout --signal=TERM --kill-after=60s 1800s "$PYTHON" "$RUNNER" capture \
  --model-dir "$MODEL_DIR" \
  --model-weight-ledger "$MODEL_WEIGHT_LEDGER_FILE" \
  --expected-model-weight-ledger-sha256 "$EXPECTED_MODEL_WEIGHT_LEDGER_SHA256" \
  --pg19-data "$PG19_DATA" \
  --pg19-manifest "$PG19_MANIFEST" \
  --preregistration "$PREREGISTRATION" \
  --output-dir "$RUN_DIR/raw" \
  > "$RUN_DIR/capture.stdout.jsonl"
date -u +%FT%TZ > "$RUN_DIR/stages/02-capture-passed"

timeout --signal=TERM --kill-after=30s 600s "$PYTHON" "$RUNNER" aggregate \
  --capture-manifest "$RUN_DIR/raw/capture-manifest.json" \
  --preregistration "$PREREGISTRATION" \
  --reference-module "$REFERENCE" \
  --output "$RUN_DIR/oracle-result.json" \
  > "$RUN_DIR/aggregate.stdout.jsonl"
date -u +%FT%TZ > "$RUN_DIR/stages/03-independent-replay-passed"

find "$RUN_DIR/raw" -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum > "$RUN_DIR/raw-artifacts.sha256"
sha256sum "$RUN_DIR/oracle-result.json" "$RUN_DIR/raw-artifacts.sha256" > "$RUN_DIR/terminal-products.sha256"
(cd / && sha256sum -c "$RUN_DIR/raw-artifacts.sha256")
(cd / && sha256sum -c "$RUN_DIR/terminal-products.sha256")
date -u +%FT%TZ > "$RUN_DIR/stages/04-terminal-audit-passed"
date -u +%FT%TZ > "$RUN_DIR/COMPLETED"
