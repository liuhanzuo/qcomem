#!/usr/bin/env bash
set -euo pipefail

required=(CODE_DIR MODEL_DIR PG19_DATA PG19_MANIFEST PRIOR_CAPACITY_MANIFEST RR2_INPUT_MANIFEST EXPECTED_RR2_INPUT_MANIFEST_SHA256 OUTPUT_ROOT ENV_DIR RUN_ID)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "missing required environment variable: $name" >&2
    exit 2
  fi
done

[[ "$RUN_ID" =~ ^[0-9a-f]{32}$ ]] || { echo "invalid RUN_ID" >&2; exit 2; }
[[ ! -e "$OUTPUT_ROOT" ]] || { echo "OUTPUT_ROOT already exists" >&2; exit 2; }
for path in "$CODE_DIR" "$MODEL_DIR" "$PG19_DATA" "$PG19_MANIFEST" "$PRIOR_CAPACITY_MANIFEST" "$RR2_INPUT_MANIFEST" "$ENV_DIR/bin/python"; do
  [[ -e "$path" ]] || { echo "missing prerequisite: $path" >&2; exit 2; }
done

export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$CODE_DIR"

exec "$ENV_DIR/bin/python" "$CODE_DIR/run_qcomem_qwen35_forkaudit_clean_rerun.py" \
  --code-dir "$CODE_DIR" \
  --model-dir "$MODEL_DIR" \
  --pg19-data "$PG19_DATA" \
  --pg19-manifest "$PG19_MANIFEST" \
  --prior-capacity-manifest "$PRIOR_CAPACITY_MANIFEST" \
  --rr2-input-manifest "$RR2_INPUT_MANIFEST" \
  --expected-rr2-input-manifest-sha256 "$EXPECTED_RR2_INPUT_MANIFEST_SHA256" \
  --run-id "$RUN_ID" \
  --output-root "$OUTPUT_ROOT"
