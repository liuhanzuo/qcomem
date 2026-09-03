#!/usr/bin/env bash
set -euo pipefail

CODE_DIR=${CODE_DIR:?set CODE_DIR}
MODEL_DIR=${MODEL_DIR:?set MODEL_DIR}
PG19_DATA=${PG19_DATA:?set PG19_DATA}
PG19_MANIFEST=${PG19_MANIFEST:?set PG19_MANIFEST}
VALIDATION_DATA=${VALIDATION_DATA:?set VALIDATION_DATA}
RUN_DIR=${RUN_DIR:?set RUN_DIR}
ENV_DIR=${ENV_DIR:?set ENV_DIR}
MODEL_WEIGHT_LEDGER_FILE=${MODEL_WEIGHT_LEDGER_FILE:?set MODEL_WEIGHT_LEDGER_FILE}

EXPECTED_PG19_SHA256=${EXPECTED_PG19_SHA256:?set EXPECTED_PG19_SHA256}
EXPECTED_PG19_MANIFEST_SHA256=${EXPECTED_PG19_MANIFEST_SHA256:?set EXPECTED_PG19_MANIFEST_SHA256}
EXPECTED_PG19_WINDOWS_SHA256=${EXPECTED_PG19_WINDOWS_SHA256:?set EXPECTED_PG19_WINDOWS_SHA256}
EXPECTED_VALIDATION_SHA256=${EXPECTED_VALIDATION_SHA256:?set EXPECTED_VALIDATION_SHA256}
EXPECTED_SOURCE_REVISION=${EXPECTED_SOURCE_REVISION:?set EXPECTED_SOURCE_REVISION}
EXPECTED_MODEL_MANIFEST_SHA256=${EXPECTED_MODEL_MANIFEST_SHA256:?set EXPECTED_MODEL_MANIFEST_SHA256}
EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256=${EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256:?set EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256}
EXPECTED_MODEL_WEIGHT_LEDGER_SHA256=${EXPECTED_MODEL_WEIGHT_LEDGER_SHA256:?set EXPECTED_MODEL_WEIGHT_LEDGER_SHA256}
EXPECTED_CODE_LEDGER_SHA256=${EXPECTED_CODE_LEDGER_SHA256:?set EXPECTED_CODE_LEDGER_SHA256}
PAGE_SIZE=${PAGE_SIZE:-128}
MAX_INPUT_TOKENS=${MAX_INPUT_TOKENS:-4096}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-8}

for input in "$PG19_DATA" "$PG19_MANIFEST" "$VALIDATION_DATA"; do
  normalized=$(printf '%s' "$input" | tr '[:upper:]_' '[:lower:]-')
  case "$normalized" in
    *test-v2*) echo "test-v2 path is forbidden: $input" >&2; exit 2 ;;
  esac
done
if [[ "$PAGE_SIZE" -lt 16 || $((PAGE_SIZE % 16)) -ne 0 ]]; then
  echo "PAGE_SIZE must be a multiple of 16" >&2
  exit 2
fi
if [[ -e "$RUN_DIR" ]] && [[ -n "$(find "$RUN_DIR" -mindepth 1 -print -quit)" ]]; then
  echo "RUN_DIR must be absent or empty: $RUN_DIR" >&2
  exit 2
fi
if [[ "$(sha256sum "$PG19_DATA" | awk '{print $1}')" != "$EXPECTED_PG19_SHA256" ]]; then
  echo "PG19 SHA mismatch" >&2; exit 2
fi
if [[ "$(sha256sum "$PG19_MANIFEST" | awk '{print $1}')" != "$EXPECTED_PG19_MANIFEST_SHA256" ]]; then
  echo "PG19 manifest SHA mismatch" >&2; exit 2
fi
if [[ "$(sha256sum "$VALIDATION_DATA" | awk '{print $1}')" != "$EXPECTED_VALIDATION_SHA256" ]]; then
  echo "validation SHA mismatch" >&2; exit 2
fi
GPU_COUNT=$(nvidia-smi -L | wc -l | tr -d ' ')
if [[ "$GPU_COUNT" -ne 8 ]]; then
  echo "formal protocol requires exactly 8 visible GPUs, found $GPU_COUNT" >&2
  exit 2
fi

mkdir -p "$RUN_DIR/logs" "$RUN_DIR/stages" "$RUN_DIR/validation"
mkdir -p "$RUN_DIR/pg19-gate-shards"
trap 'date -u +%FT%TZ > "$RUN_DIR/stages/FAILED"' ERR
date -u +%FT%TZ > "$RUN_DIR/stages/00_start"

write_code_ledger() {
  local destination=$1
  while IFS= read -r file; do
    local digest name
    digest=$(sha256sum "$file" | awk '{print $1}')
    name=${file#"$CODE_DIR"/}
    printf '%s  %s\n' "$digest" "$name"
  done < <(
    LC_ALL=C find "$CODE_DIR" -maxdepth 1 -type f \
      \( -name '*.py' -o -name 'launch_qcomem_qwen35_vllm_paged_formal_8gpu.sh' \) \
      | LC_ALL=C sort
  ) > "$destination"
}

verify_code_ledger() {
  local candidate actual
  candidate=$(mktemp "$RUN_DIR/code-ledger-candidate.XXXXXX")
  write_code_ledger "$candidate"
  actual=$(sha256sum "$candidate" | awk '{print $1}')
  if [[ "$actual" != "$EXPECTED_CODE_LEDGER_SHA256" ]]; then
    echo "frozen code ledger SHA mismatch: $actual" >&2
    rm -f "$candidate"
    return 2
  fi
  if [[ -f "$RUN_DIR/code.sha256" ]] && ! cmp -s "$candidate" "$RUN_DIR/code.sha256"; then
    echo "code snapshot changed after run start" >&2
    rm -f "$candidate"
    return 2
  fi
  mv "$candidate" "$RUN_DIR/code.sha256"
  (cd "$CODE_DIR" && sha256sum -c "$RUN_DIR/code.sha256") \
    >> "$RUN_DIR/logs/code-integrity.log"
}

verify_code_ledger

MODEL_ARTIFACT_FILES=(
  "$MODEL_DIR/config.json"
  "$MODEL_DIR/model.safetensors.index.json"
  "$MODEL_DIR/tokenizer_config.json"
  "$MODEL_DIR/vocab.json"
  "$MODEL_DIR/merges.txt"
  "$MODEL_DIR/chat_template.jinja"
)
for artifact in "${MODEL_ARTIFACT_FILES[@]}" "$MODEL_WEIGHT_LEDGER_FILE"; do
  if [[ ! -f "$artifact" ]]; then
    echo "required frozen model artifact is missing: $artifact" >&2
    exit 2
  fi
done
if [[ "$(wc -l < "$MODEL_WEIGHT_LEDGER_FILE" | tr -d ' ')" -ne 14 ]]; then
  echo "model weight ledger must contain exactly 14 shards" >&2
  exit 2
fi
sha256sum "${MODEL_ARTIFACT_FILES[@]}" > "$RUN_DIR/model-artifacts.sha256"
cp "$MODEL_WEIGHT_LEDGER_FILE" "$RUN_DIR/model-weights.sha256"
if [[ "$(sha256sum "$RUN_DIR/model-artifacts.sha256" | awk '{print $1}')" \
  != "$EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256" ]]; then
  echo "model artifact ledger SHA mismatch" >&2
  exit 2
fi
if [[ "$(sha256sum "$RUN_DIR/model-weights.sha256" | awk '{print $1}')" \
  != "$EXPECTED_MODEL_WEIGHT_LEDGER_SHA256" ]]; then
  echo "model weight ledger SHA mismatch" >&2
  exit 2
fi
sha256sum -c "$RUN_DIR/model-artifacts.sha256" \
  > "$RUN_DIR/logs/model-artifact-integrity.log"
# Hash all 14 shards once before any model process starts. Subsequent ranks are
# bound to this immutable ledger instead of rereading roughly 70 GB eight times.
sha256sum -c "$RUN_DIR/model-weights.sha256" \
  > "$RUN_DIR/logs/model-weight-integrity.log"

PYTHONPATH="$CODE_DIR" "$ENV_DIR/bin/python" -m unittest -v \
  test_qcomem_vllm_paged_kernel \
  test_qcomem_qwen35_vllm_paged_integration \
  test_run_qcomem_qwen35_vllm_paged_formal \
  > "$RUN_DIR/logs/preflight-tests.log" 2>&1
date -u +%FT%TZ > "$RUN_DIR/stages/01_preflight_ok"

COMMON=(
  --model "$MODEL_DIR"
  --pg19-data "$PG19_DATA"
  --pg19-manifest "$PG19_MANIFEST"
  --validation-data "$VALIDATION_DATA"
  --expected-pg19-sha256 "$EXPECTED_PG19_SHA256"
  --expected-pg19-manifest-sha256 "$EXPECTED_PG19_MANIFEST_SHA256"
  --expected-pg19-windows-sha256 "$EXPECTED_PG19_WINDOWS_SHA256"
  --expected-validation-sha256 "$EXPECTED_VALIDATION_SHA256"
  --expected-source-revision "$EXPECTED_SOURCE_REVISION"
  --expected-model-manifest-sha256 "$EXPECTED_MODEL_MANIFEST_SHA256"
  --bits 16
  --page-size "$PAGE_SIZE"
  --source-index-start 6
  --source-index-end 9
  --limit-per-dataset 4
  --max-input-tokens "$MAX_INPUT_TOKENS"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --pg19-books 8
  --pg19-document-tokens 1024
  --pg19-query-tokens 32
  --pg19-window-stride 512
  --pg19-candidate-windows 4
  --pg19-seed 20260813
  --isolated-rtol 0.02
  --isolated-atol 0.05
  --semantic-mean-kl-threshold 0.001
)

PYTHONPATH="$CODE_DIR" "$ENV_DIR/bin/python" \
  "$CODE_DIR/run_qcomem_qwen35_vllm_paged_formal.py" \
  --stage static-dry-run --output "$RUN_DIR/static-dry-run.json" "${COMMON[@]}" \
  > "$RUN_DIR/logs/static-dry-run.log" 2>&1
date -u +%FT%TZ > "$RUN_DIR/stages/02_static_dry_run_ok"

verify_code_ledger
gate_pids=()
for rank in 0 1 2 3 4 5 6 7; do
  timeout 5400 env CUDA_VISIBLE_DEVICES=$rank PYTHONPATH="$CODE_DIR" \
    "$ENV_DIR/bin/python" "$CODE_DIR/run_qcomem_qwen35_vllm_paged_formal.py" \
    --stage pg19-gate --rank "$rank" --world-size 8 \
    --output "$RUN_DIR/pg19-gate-shards/pg19-gate-shard-$rank.json" \
    "${COMMON[@]}" > "$RUN_DIR/logs/pg19-gate-rank-$rank.log" 2>&1 &
  gate_pids+=("$!")
  sleep 2
done
gate_failed=0
for rank in 0 1 2 3 4 5 6 7; do
  if ! wait "${gate_pids[$rank]}"; then
    echo "PG19 gate rank $rank failed" >&2
    gate_failed=1
  fi
done
if [[ "$gate_failed" -ne 0 ]]; then exit 1; fi

verify_code_ledger
AUTHORIZATION="$RUN_DIR/pg19-kernel-authorization.json"
PYTHONPATH="$CODE_DIR" "$ENV_DIR/bin/python" - \
  "$RUN_DIR" "$AUTHORIZATION" "$EXPECTED_PG19_WINDOWS_SHA256" <<'PY'
import sys
from pathlib import Path
from run_downstream import atomic_json
from run_qcomem_qwen35_vllm_paged_formal import aggregate_pg19_gate_shards
run_dir, output, windows_sha = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
paths=sorted((run_dir/"pg19-gate-shards").glob("pg19-gate-shard-*.json"))
result=aggregate_pg19_gate_shards(
    paths,
    expected_windows_sha256=windows_sha,
    mean_kl_threshold=0.001,
)
atomic_json(output, result)
PY
AUTHORIZATION_SHA256=$(sha256sum "$AUTHORIZATION" | awk '{print $1}')
printf '%s  %s\n' "$AUTHORIZATION_SHA256" "$AUTHORIZATION" \
  > "$RUN_DIR/pg19-kernel-authorization.sha256"
date -u +%FT%TZ > "$RUN_DIR/stages/03_pg19_gate_authorized"

verify_code_ledger
pids=()
for rank in 0 1 2 3 4 5 6 7; do
  timeout 10800 env CUDA_VISIBLE_DEVICES=$rank PYTHONPATH="$CODE_DIR" \
    "$ENV_DIR/bin/python" "$CODE_DIR/run_qcomem_qwen35_vllm_paged_formal.py" \
    --stage validation --rank "$rank" --world-size 8 \
    --authorization "$AUTHORIZATION" \
    --expected-authorization-sha256 "$AUTHORIZATION_SHA256" \
    --output "$RUN_DIR/validation/vllm-paged-q16-shard-$rank.json" \
    "${COMMON[@]}" > "$RUN_DIR/logs/validation-rank-$rank.log" 2>&1 &
  pids+=("$!")
  sleep 5
done
failed=0
for rank in 0 1 2 3 4 5 6 7; do
  if ! wait "${pids[$rank]}"; then
    echo "validation rank $rank failed" >&2
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then exit 1; fi

verify_code_ledger
PYTHONPATH="$CODE_DIR" "$ENV_DIR/bin/python" - \
  "$RUN_DIR" "$AUTHORIZATION" "$AUTHORIZATION_SHA256" \
  "$EXPECTED_CODE_LEDGER_SHA256" "$EXPECTED_MODEL_MANIFEST_SHA256" \
  "$EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256" \
  "$EXPECTED_MODEL_WEIGHT_LEDGER_SHA256" "$EXPECTED_SOURCE_REVISION" \
  "$MAX_NEW_TOKENS" <<'PY'
import sys
from pathlib import Path
from run_downstream import atomic_json
from run_qcomem_qwen35_vllm_paged_formal import summarize_validation_shards
root=Path(sys.argv[1])
summary=summarize_validation_shards(
    root,
    authorization_path=Path(sys.argv[2]),
    expected_authorization_sha256=sys.argv[3],
    expected_code_ledger_sha256=sys.argv[4],
    expected_model_manifest_sha256=sys.argv[5],
    expected_model_artifact_ledger_sha256=sys.argv[6],
    expected_model_weight_ledger_sha256=sys.argv[7],
    expected_source_revision=sys.argv[8],
    expected_calls_per_layer=int(sys.argv[9]),
)
atomic_json(root/"summary.json", summary)
PY
date -u +%FT%TZ > "$RUN_DIR/stages/99_done"
echo "Qwen3.5 vLLM Q16 formal run complete: $RUN_DIR"
