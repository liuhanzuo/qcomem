#!/usr/bin/env bash
set -euo pipefail

CODE_DIR=${CODE_DIR:?set CODE_DIR}
RR2_CODE_DIR=${RR2_CODE_DIR:?set RR2_CODE_DIR}
ENV_DIR=${ENV_DIR:?set ENV_DIR}
MODEL_DIR=${MODEL_DIR:?set MODEL_DIR}
MODEL_WEIGHT_LEDGER=${MODEL_WEIGHT_LEDGER:?set MODEL_WEIGHT_LEDGER}
PG19_DATA=${PG19_DATA:?set PG19_DATA}
PG19_MANIFEST=${PG19_MANIFEST:?set PG19_MANIFEST}
INPUT_MANIFEST=${INPUT_MANIFEST:?set INPUT_MANIFEST}
PREREGISTRATION=${PREREGISTRATION:?set PREREGISTRATION}
SOURCE_LEDGER=${SOURCE_LEDGER:?set SOURCE_LEDGER}
PREEXECUTION_PIN=${PREEXECUTION_PIN:?set PREEXECUTION_PIN}
RUN_DIR=${RUN_DIR:?set RUN_DIR}
EXPECTED_PRODUCER_SHA256=${EXPECTED_PRODUCER_SHA256:?set EXPECTED_PRODUCER_SHA256}
EXPECTED_REFERENCE_SHA256=${EXPECTED_REFERENCE_SHA256:?set EXPECTED_REFERENCE_SHA256}
EXPECTED_LAUNCHER_SHA256=${EXPECTED_LAUNCHER_SHA256:?set EXPECTED_LAUNCHER_SHA256}
EXPECTED_INPUT_MANIFEST_SHA256=${EXPECTED_INPUT_MANIFEST_SHA256:?set EXPECTED_INPUT_MANIFEST_SHA256}
EXPECTED_PREREGISTRATION_SHA256=${EXPECTED_PREREGISTRATION_SHA256:?set EXPECTED_PREREGISTRATION_SHA256}
EXPECTED_SOURCE_LEDGER_SHA256=${EXPECTED_SOURCE_LEDGER_SHA256:?set EXPECTED_SOURCE_LEDGER_SHA256}
EXPECTED_PREEXECUTION_PIN_SHA256=${EXPECTED_PREEXECUTION_PIN_SHA256:?set EXPECTED_PREEXECUTION_PIN_SHA256}

PYTHON="$ENV_DIR/bin/python"
PRODUCER="$CODE_DIR/r30_expanded_oracle_producer.py"
REFERENCE="$CODE_DIR/r30_expanded_oracle_reference.py"
LAUNCHER="$CODE_DIR/launch_r30_expanded_oracle.sh"

verify_sha() {
  local path=$1 expected=$2 label=$3 actual
  actual=$(sha256sum "$path" | awk '{print $1}')
  if [[ "$actual" != "$expected" ]]; then
    echo "$label SHA mismatch: expected=$expected actual=$actual" >&2
    exit 2
  fi
}

if [[ "${CUDA_VISIBLE_DEVICES:-}" != "3" ]]; then
  echo "candidate must expose physical GPU3 only" >&2
  exit 2
fi
if [[ -e "$RUN_DIR" ]]; then
  echo "RUN_DIR must be absent: $RUN_DIR" >&2
  exit 2
fi
verify_sha "$PRODUCER" "$EXPECTED_PRODUCER_SHA256" producer
verify_sha "$REFERENCE" "$EXPECTED_REFERENCE_SHA256" reference
verify_sha "$LAUNCHER" "$EXPECTED_LAUNCHER_SHA256" launcher
verify_sha "$INPUT_MANIFEST" "$EXPECTED_INPUT_MANIFEST_SHA256" input-manifest
verify_sha "$PREREGISTRATION" "$EXPECTED_PREREGISTRATION_SHA256" preregistration
verify_sha "$SOURCE_LEDGER" "$EXPECTED_SOURCE_LEDGER_SHA256" source-ledger
verify_sha "$PREEXECUTION_PIN" "$EXPECTED_PREEXECUTION_PIN_SHA256" preexecution-pin
(cd "$CODE_DIR" && sha256sum -c "$SOURCE_LEDGER")
if grep -Eq '^(import torch|from torch|import qcomem|from qcomem|import transformers|from transformers|import vllm|from vllm)' "$REFERENCE"; then
  echo "candidate-import-free reference import gate failed" >&2
  exit 2
fi
if nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -q '^GPU-c009c4fe-2866-3f91-0080-5d5d895efa4f'; then
  echo "physical GPU3 is not idle" >&2
  exit 2
fi
"$PYTHON" -c 'import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 1; assert "H20" in torch.cuda.get_device_properties(0).name'

mkdir -p "$RUN_DIR/stages"
date -u +%FT%TZ > "$RUN_DIR/stages/00-started"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$CODE_DIR:$RR2_CODE_DIR"
timeout --signal=TERM --kill-after=60s 1800s "$PYTHON" "$PRODUCER" capture \
  --model-dir "$MODEL_DIR" \
  --model-weight-ledger "$MODEL_WEIGHT_LEDGER" \
  --pg19-data "$PG19_DATA" \
  --pg19-manifest "$PG19_MANIFEST" \
  --input-manifest "$INPUT_MANIFEST" \
  --preregistration "$PREREGISTRATION" \
  --output-dir "$RUN_DIR/raw" \
  > "$RUN_DIR/capture.stdout.jsonl"
date -u +%FT%TZ > "$RUN_DIR/stages/01-capture-passed"

CUDA_VISIBLE_DEVICES='' timeout --signal=TERM --kill-after=30s 1800s "$PYTHON" -I "$REFERENCE" \
  --capture-manifest "$RUN_DIR/raw/capture-manifest.json" \
  --preregistration "$PREREGISTRATION" \
  --output "$RUN_DIR/oracle-result.json" \
  > "$RUN_DIR/reference.stdout.jsonl"
date -u +%FT%TZ > "$RUN_DIR/stages/02-candidate-import-free-numpy-replay-passed"

find "$RUN_DIR/raw" -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum > "$RUN_DIR/raw-artifacts.sha256"
sha256sum "$RUN_DIR/oracle-result.json" "$RUN_DIR/raw-artifacts.sha256" > "$RUN_DIR/terminal-products.sha256"
(cd / && sha256sum -c "$RUN_DIR/raw-artifacts.sha256")
(cd / && sha256sum -c "$RUN_DIR/terminal-products.sha256")
date -u +%FT%TZ > "$RUN_DIR/stages/03-terminal-audit-passed"
date -u +%FT%TZ > "$RUN_DIR/COMPLETED"
