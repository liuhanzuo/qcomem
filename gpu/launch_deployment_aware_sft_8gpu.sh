#!/usr/bin/env bash
set -euo pipefail

CODE_DIR=${CODE_DIR:?set CODE_DIR}
MODEL_DIR=${MODEL_DIR:?set MODEL_DIR}
DATA_DIR=${DATA_DIR:?set DATA_DIR}
RUN_DIR=${RUN_DIR:?set RUN_DIR}
ENV_DIR=${ENV_DIR:?set ENV_DIR}
CONFIG_FILE=${CONFIG_FILE:?set CONFIG_FILE}
TRAIN_FILE=${TRAIN_FILE:-$DATA_DIR/deployment-aware-train-1024.jsonl}
HELDOUT_FILE=${HELDOUT_FILE:-$DATA_DIR/deployment-aware-heldout-64.jsonl}
DATA_MANIFEST_FILE=${DATA_MANIFEST_FILE:-$DATA_DIR/deployment-aware-manifest.json}
TRAIN_SHA256=${TRAIN_SHA256:?set TRAIN_SHA256}
HELDOUT_SHA256=${HELDOUT_SHA256:?set HELDOUT_SHA256}
DATA_MANIFEST_SHA256=${DATA_MANIFEST_SHA256:?set DATA_MANIFEST_SHA256}
MODEL_WEIGHT_LEDGER_FILE=${MODEL_WEIGHT_LEDGER_FILE:?set MODEL_WEIGHT_LEDGER_FILE}

for VALUE in "$TRAIN_SHA256" "$HELDOUT_SHA256" "$DATA_MANIFEST_SHA256"; do
  if [[ ! "$VALUE" =~ ^[0-9a-f]{64}$ ]]; then
    echo "data hashes must be lowercase SHA256 values" >&2
    exit 2
  fi
done
if [ -e "$RUN_DIR" ] && [ -n "$(find "$RUN_DIR" -mindepth 1 -print -quit 2>/dev/null)" ]; then
  echo "RUN_DIR must be absent or empty: $RUN_DIR" >&2
  exit 2
fi
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/stages"
date -u +%FT%TZ > "$RUN_DIR/stages/00_start"

CODE_FILES=(
  "$CODE_DIR/train_deployment_aware_sft.py"
  "$CODE_DIR/deployment_aware_sft.py"
  "$CODE_DIR/build_deployment_aware_sft.py"
  "$CODE_DIR/audit_deployment_aware_sft.py"
  "$CODE_DIR/test_deployment_aware_sft.py"
  "$CODE_DIR/deployment_aware_fsdp_preflight.py"
  "$CODE_DIR/launch_deployment_aware_sft_8gpu.sh"
  "$CONFIG_FILE"
  "$CODE_DIR/fp32_master.py"
  "$CODE_DIR/qcomem_torch.py"
  "$CODE_DIR/sft_dcp_checkpoint.py"
  "$CODE_DIR/supervised_sft.py"
  "$CODE_DIR/supervised_sft_longrun.py"
)
MODEL_ARTIFACT_FILES=(
  "$MODEL_DIR/config.json"
  "$MODEL_DIR/model.safetensors.index.json"
  "$MODEL_DIR/tokenizer_config.json"
  "$MODEL_DIR/vocab.json"
  "$MODEL_DIR/merges.txt"
  "$MODEL_DIR/chat_template.jinja"
)
MODEL_WEIGHT_FILES=()
for INDEX in $(seq -w 1 14); do
  MODEL_WEIGHT_FILES+=("$MODEL_DIR/model.safetensors-000${INDEX}-of-00014.safetensors")
done
for ARTIFACT in \
  "${CODE_FILES[@]}" \
  "${MODEL_ARTIFACT_FILES[@]}" \
  "${MODEL_WEIGHT_FILES[@]}" \
  "$MODEL_WEIGHT_LEDGER_FILE" \
  "$TRAIN_FILE" \
  "$HELDOUT_FILE" \
  "$DATA_MANIFEST_FILE"; do
  if [ ! -f "$ARTIFACT" ]; then
    echo "required artifact is missing: $ARTIFACT" >&2
    exit 2
  fi
done

sha256sum "${CODE_FILES[@]}" > "$RUN_DIR/code.sha256"
sha256sum "${MODEL_ARTIFACT_FILES[@]}" > "$RUN_DIR/model-artifacts.sha256"
cp "$MODEL_WEIGHT_LEDGER_FILE" "$RUN_DIR/model-weights.sha256"
CODE_LEDGER_SHA256=$(sha256sum "$RUN_DIR/code.sha256" | awk '{print $1}')
MODEL_ARTIFACT_LEDGER_SHA256=$(sha256sum "$RUN_DIR/model-artifacts.sha256" | awk '{print $1}')
MODEL_WEIGHT_LEDGER_SHA256=$(sha256sum "$RUN_DIR/model-weights.sha256" | awk '{print $1}')
sha256sum -c "$RUN_DIR/code.sha256" > "$RUN_DIR/logs/code-integrity.log"
sha256sum -c "$RUN_DIR/model-artifacts.sha256" > "$RUN_DIR/logs/model-artifact-integrity.log"
sha256sum -c "$RUN_DIR/model-weights.sha256" > "$RUN_DIR/logs/model-weight-integrity.log"
if [ "$(sha256sum "$TRAIN_FILE" | awk '{print $1}')" != "$TRAIN_SHA256" ]; then
  echo "train SHA256 mismatch" >&2
  exit 2
fi
if [ "$(sha256sum "$HELDOUT_FILE" | awk '{print $1}')" != "$HELDOUT_SHA256" ]; then
  echo "heldout SHA256 mismatch" >&2
  exit 2
fi
if [ "$(sha256sum "$DATA_MANIFEST_FILE" | awk '{print $1}')" != "$DATA_MANIFEST_SHA256" ]; then
  echo "data manifest SHA256 mismatch" >&2
  exit 2
fi

GPU_COUNT=$(nvidia-smi -L | wc -l | tr -d ' ')
if [ "$GPU_COUNT" -ne 8 ]; then
  echo "formal run requires exactly eight GPUs, found $GPU_COUNT" >&2
  exit 2
fi
nvidia-smi \
  --query-gpu=index,uuid,name,memory.total,driver_version,pstate,power.limit \
  --format=csv > "$RUN_DIR/gpus-before.csv"

"$ENV_DIR/bin/python" -m py_compile \
  "$CODE_DIR/train_deployment_aware_sft.py" \
  "$CODE_DIR/deployment_aware_sft.py" \
  "$CODE_DIR/build_deployment_aware_sft.py" \
  "$CODE_DIR/audit_deployment_aware_sft.py" \
  "$CODE_DIR/test_deployment_aware_sft.py" \
  "$CODE_DIR/deployment_aware_fsdp_preflight.py"
bash -n "$CODE_DIR/launch_deployment_aware_sft_8gpu.sh"
PYTHONPATH="$CODE_DIR" "$ENV_DIR/bin/python" -m unittest \
  test_deployment_aware_sft -v > "$RUN_DIR/logs/unit-tests.log" 2>&1
PYTHONPATH="$CODE_DIR" "$ENV_DIR/bin/python" \
  "$CODE_DIR/audit_deployment_aware_sft.py" \
  --manifest "$DATA_MANIFEST_FILE" \
  --train "$TRAIN_FILE" \
  --heldout "$HELDOUT_FILE" \
  --expected-manifest-sha256 "$DATA_MANIFEST_SHA256" \
  --expected-train-sha256 "$TRAIN_SHA256" \
  --expected-heldout-sha256 "$HELDOUT_SHA256" \
  > "$RUN_DIR/logs/data-audit.json"
PYTHONPATH="$CODE_DIR" "$ENV_DIR/bin/torchrun" \
  --standalone --nproc-per-node=2 \
  "$CODE_DIR/deployment_aware_fsdp_preflight.py" \
  --output-root "$RUN_DIR/fsdp-dcp-tiny-preflight" \
  > "$RUN_DIR/logs/fsdp-dcp-tiny-preflight.log" 2>&1
date -u +%FT%TZ > "$RUN_DIR/stages/01_static_and_tiny_preflight_ok"

sha256sum -c "$RUN_DIR/code.sha256" >> "$RUN_DIR/logs/code-integrity.log"
sha256sum -c "$RUN_DIR/model-artifacts.sha256" >> "$RUN_DIR/logs/model-artifact-integrity.log"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ENABLE_MONITORING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=14400
PYTHONPATH="$CODE_DIR" "$ENV_DIR/bin/torchrun" \
  --standalone --nproc-per-node=8 \
  "$CODE_DIR/train_deployment_aware_sft.py" \
  --config "$CONFIG_FILE" \
  --model "$MODEL_DIR" \
  --train-data "$TRAIN_FILE" \
  --heldout-data "$HELDOUT_FILE" \
  --data-manifest "$DATA_MANIFEST_FILE" \
  --expected-train-sha256 "$TRAIN_SHA256" \
  --expected-heldout-sha256 "$HELDOUT_SHA256" \
  --expected-data-manifest-sha256 "$DATA_MANIFEST_SHA256" \
  --code-ledger "$RUN_DIR/code.sha256" \
  --expected-code-ledger-sha256 "$CODE_LEDGER_SHA256" \
  --model-artifact-ledger "$RUN_DIR/model-artifacts.sha256" \
  --expected-model-artifact-ledger-sha256 "$MODEL_ARTIFACT_LEDGER_SHA256" \
  --model-weight-ledger "$RUN_DIR/model-weights.sha256" \
  --expected-model-weight-ledger-sha256 "$MODEL_WEIGHT_LEDGER_SHA256" \
  --output-dir "$RUN_DIR/artifacts" \
  > "$RUN_DIR/logs/train.log" 2>&1

test -s "$RUN_DIR/artifacts/metadata.json"
test -s "$RUN_DIR/artifacts/train-metrics.jsonl"
test -s "$RUN_DIR/artifacts/heldout-metrics.jsonl"
test -s "$RUN_DIR/artifacts/checkpoint-metrics.jsonl"
test -s "$RUN_DIR/artifacts/best-checkpoint.json"
"$ENV_DIR/bin/python" - "$RUN_DIR/artifacts" <<'PY'
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
metadata = json.loads((root / "metadata.json").read_text())
if metadata.get("format") != "qcomem_dense_long_instruction_preservation_full_sft_control_v1":
    raise SystemExit("wrong formal metadata format")
if metadata.get("last_step") != 128:
    raise SystemExit("formal training did not complete 128 steps")
if metadata.get("initialization") != "post_trained_not_base":
    raise SystemExit("model initialization contract failed")
algorithm = metadata.get("algorithm_path", {})
if (
    algorithm.get("use_cache") is not False
    or algorithm.get("qcomem_replay_or_quantization_in_training_forward") is not False
    or algorithm.get("cache_aware_training") is not False
):
    raise SystemExit("dense-control claim boundary failed")
if metadata.get("raw_longbench_validation_or_test_read") is not False:
    raise SystemExit("blind LongBench governance failed")
if metadata.get("step1_gate", {}).get("passed") is not True:
    raise SystemExit("4K step-1 memory/delta gate failed")
teacher = metadata.get("teacher_targets", {}).get("manifest", {})
if teacher.get("records") != 307 or teacher.get("total_shard_bytes", 0) <= 0:
    raise SystemExit("teacher target artifact contract failed")
if len(teacher.get("shards", [])) != 8:
    raise SystemExit("teacher target shard count failed")
rows = [json.loads(line) for line in (root / "train-metrics.jsonl").read_text().splitlines()]
if [row.get("step") for row in rows] != list(range(1, 129)):
    raise SystemExit("train metrics do not contain exactly steps 1..128")
if not all(math.isfinite(row["summary"]["overall"]["example_equal_mean_ce"]) for row in rows):
    raise SystemExit("train CE is non-finite")
heldout = [json.loads(line) for line in (root / "heldout-metrics.jsonl").read_text().splitlines()]
if [row.get("step") for row in heldout] != [0, 64, 128]:
    raise SystemExit("heldout phases drifted")
best = json.loads((root / "best-checkpoint.json").read_text())
if best.get("selected_step") not in {64, 128}:
    raise SystemExit("best checkpoint selection failed")
print({"status": "passed", "steps": 128, "best_step": best["selected_step"]})
PY
date -u +%FT%TZ > "$RUN_DIR/stages/02_complete"
nvidia-smi \
  --query-gpu=index,uuid,name,memory.total,memory.used,driver_version,pstate,power.draw,power.limit \
  --format=csv > "$RUN_DIR/gpus-after.csv"
