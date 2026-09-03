#!/usr/bin/env bash
set -euo pipefail

CODE_DIR=${CODE_DIR:?set CODE_DIR}
MODEL_DIR=${MODEL_DIR:?set MODEL_DIR}
DATA_FILE=${DATA_FILE:?set DATA_FILE}
MANIFEST_FILE=${MANIFEST_FILE:?set MANIFEST_FILE}
TRAIN_JSONL_SHA256=${TRAIN_JSONL_SHA256:?set TRAIN_JSONL_SHA256}
TRAIN_MANIFEST_SHA256=${TRAIN_MANIFEST_SHA256:?set TRAIN_MANIFEST_SHA256}
EXPECTED_CODE_LEDGER_SHA256=${EXPECTED_CODE_LEDGER_SHA256:?set EXPECTED_CODE_LEDGER_SHA256}
EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256=${EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256:?set EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256}
EXPECTED_RUNTIME_DEPENDENCY_LEDGER_SHA256=${EXPECTED_RUNTIME_DEPENDENCY_LEDGER_SHA256:?set EXPECTED_RUNTIME_DEPENDENCY_LEDGER_SHA256}
EXPECTED_MODEL_WEIGHT_LEDGER_SHA256=${EXPECTED_MODEL_WEIGHT_LEDGER_SHA256:?set EXPECTED_MODEL_WEIGHT_LEDGER_SHA256}
MODEL_WEIGHT_LEDGER_FILE=${MODEL_WEIGHT_LEDGER_FILE:?set MODEL_WEIGHT_LEDGER_FILE}
RUN_DIR=${RUN_DIR:?set RUN_DIR}
ENV_DIR=${ENV_DIR:?set ENV_DIR}
CONFIG_FILE=${CONFIG_FILE:?set CONFIG_FILE}

if [[ ! "$TRAIN_JSONL_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "TRAIN_JSONL_SHA256 must be one pinned lowercase SHA256" >&2
  exit 2
fi
if [[ ! "$TRAIN_MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "TRAIN_MANIFEST_SHA256 must be one pinned lowercase SHA256" >&2
  exit 2
fi
if [[ ! "$EXPECTED_CODE_LEDGER_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "EXPECTED_CODE_LEDGER_SHA256 must be one pinned lowercase SHA256" >&2
  exit 2
fi
if [[ ! "$EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256 must be one pinned lowercase SHA256" >&2
  exit 2
fi
if [[ ! "$EXPECTED_RUNTIME_DEPENDENCY_LEDGER_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "EXPECTED_RUNTIME_DEPENDENCY_LEDGER_SHA256 must be one pinned lowercase SHA256" >&2
  exit 2
fi
if [[ ! "$EXPECTED_MODEL_WEIGHT_LEDGER_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "EXPECTED_MODEL_WEIGHT_LEDGER_SHA256 must be one pinned lowercase SHA256" >&2
  exit 2
fi

if [ -e "$RUN_DIR" ] && [ -n "$(find "$RUN_DIR" -mindepth 1 -print -quit 2>/dev/null)" ]; then
  echo "RUN_DIR must be absent or empty: $RUN_DIR" >&2
  exit 2
fi
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/stages"
date -u +%FT%TZ > "$RUN_DIR/stages/00_start"

CODE_FILES=(
  "$CODE_DIR/supervised_sft.py"
  "$CODE_DIR/train_supervised_sft.py"
  "$CODE_DIR/preflight_supervised_sft.py"
  "$CODE_DIR/launch_supervised_sft_8gpu.sh"
  "$CONFIG_FILE"
  "$CODE_DIR/run_downstream.py"
)
RUNTIME_DEPENDENCY_FILES=(
  "$CODE_DIR/qcomem_torch.py"
  "$CODE_DIR/fp32_master.py"
  "$CODE_DIR/test_supervised_sft.py"
  "$CODE_DIR/test_fp32_master.py"
  "$CODE_DIR/fsdp_fp32_preflight.py"
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
  MODEL_WEIGHT_FILES+=(
    "$MODEL_DIR/model.safetensors-000${INDEX}-of-00014.safetensors"
  )
done
if [ ! -f "$MODEL_WEIGHT_LEDGER_FILE" ]; then
  echo "frozen model weight ledger is missing: $MODEL_WEIGHT_LEDGER_FILE" >&2
  exit 2
fi
for ARTIFACT in \
  "${CODE_FILES[@]}" \
  "${RUNTIME_DEPENDENCY_FILES[@]}" \
  "${MODEL_ARTIFACT_FILES[@]}" \
  "${MODEL_WEIGHT_FILES[@]}"; do
  if [ ! -f "$ARTIFACT" ]; then
    echo "required frozen code/model artifact is missing: $ARTIFACT" >&2
    exit 2
  fi
done
sha256sum "${CODE_FILES[@]}" > "$RUN_DIR/code.sha256"
sha256sum "${RUNTIME_DEPENDENCY_FILES[@]}" \
  > "$RUN_DIR/runtime-dependencies.sha256"
sha256sum "${MODEL_ARTIFACT_FILES[@]}" > "$RUN_DIR/model-artifacts.sha256"
cp "$MODEL_WEIGHT_LEDGER_FILE" "$RUN_DIR/model-weights.sha256"
ACTUAL_CODE_LEDGER_SHA256=$(sha256sum "$RUN_DIR/code.sha256" | awk '{print $1}')
ACTUAL_MODEL_ARTIFACT_LEDGER_SHA256=$(
  sha256sum "$RUN_DIR/model-artifacts.sha256" | awk '{print $1}'
)
ACTUAL_RUNTIME_DEPENDENCY_LEDGER_SHA256=$(
  sha256sum "$RUN_DIR/runtime-dependencies.sha256" | awk '{print $1}'
)
ACTUAL_MODEL_WEIGHT_LEDGER_SHA256=$(
  sha256sum "$RUN_DIR/model-weights.sha256" | awk '{print $1}'
)
if [ "$ACTUAL_CODE_LEDGER_SHA256" != "$EXPECTED_CODE_LEDGER_SHA256" ]; then
  echo "frozen code ledger SHA256 mismatch" >&2
  exit 2
fi
if [ "$ACTUAL_MODEL_ARTIFACT_LEDGER_SHA256" != "$EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256" ]; then
  echo "frozen model artifact ledger SHA256 mismatch" >&2
  exit 2
fi
if [ "$ACTUAL_RUNTIME_DEPENDENCY_LEDGER_SHA256" != "$EXPECTED_RUNTIME_DEPENDENCY_LEDGER_SHA256" ]; then
  echo "frozen runtime dependency ledger SHA256 mismatch" >&2
  exit 2
fi
if [ "$ACTUAL_MODEL_WEIGHT_LEDGER_SHA256" != "$EXPECTED_MODEL_WEIGHT_LEDGER_SHA256" ]; then
  echo "frozen model weight ledger SHA256 mismatch" >&2
  exit 2
fi
sha256sum -c "$RUN_DIR/code.sha256" > "$RUN_DIR/logs/code-integrity.log"
sha256sum -c "$RUN_DIR/runtime-dependencies.sha256" \
  > "$RUN_DIR/logs/runtime-dependency-integrity.log"
sha256sum -c "$RUN_DIR/model-artifacts.sha256" \
  > "$RUN_DIR/logs/model-artifact-integrity.log"
sha256sum -c "$RUN_DIR/model-weights.sha256" \
  > "$RUN_DIR/logs/model-weight-integrity.log"

GPU_COUNT=$(nvidia-smi -L | wc -l | tr -d ' ')
if [ "$GPU_COUNT" -ne 8 ]; then
  echo "dense_full_model_sft_smoke requires exactly 8 GPUs, found $GPU_COUNT" >&2
  exit 2
fi
NORMALIZED_DATA=$(printf '%s' "$DATA_FILE" | tr '[:upper:]_' '[:lower:]-')
case "$NORMALIZED_DATA" in
  *validation*|*test-v2*|*/test/*|*-test.*|*-dev.*|*/dev/*)
    echo "supervised SFT refuses validation/test data paths" >&2
    exit 2
    ;;
esac
ACTUAL_DATA_SHA256=$(sha256sum "$DATA_FILE" | awk '{print $1}')
ACTUAL_MANIFEST_SHA256=$(sha256sum "$MANIFEST_FILE" | awk '{print $1}')
if [ "$ACTUAL_DATA_SHA256" != "$TRAIN_JSONL_SHA256" ]; then
  echo "supervised train JSONL SHA256 mismatch" >&2
  exit 2
fi
if [ "$ACTUAL_MANIFEST_SHA256" != "$TRAIN_MANIFEST_SHA256" ]; then
  echo "supervised converter manifest SHA256 mismatch" >&2
  exit 2
fi
printf '%s  %s\n' "$ACTUAL_DATA_SHA256" "$DATA_FILE" \
  > "$RUN_DIR/training-data.sha256"
printf '%s  %s\n' "$ACTUAL_MANIFEST_SHA256" "$MANIFEST_FILE" \
  > "$RUN_DIR/training-manifest.sha256"
nvidia-smi \
  --query-gpu=index,uuid,name,memory.total,driver_version,pstate,power.limit \
  --format=csv > "$RUN_DIR/gpus-before.csv"

"$ENV_DIR/bin/python" -m py_compile \
  "$CODE_DIR/run_downstream.py" \
  "$CODE_DIR/supervised_sft.py" \
  "$CODE_DIR/fp32_master.py" \
  "$CODE_DIR/fsdp_fp32_preflight.py" \
  "$CODE_DIR/preflight_supervised_sft.py" \
  "$CODE_DIR/train_supervised_sft.py"
bash -n "$CODE_DIR/launch_supervised_sft_8gpu.sh"
PYTHONPATH="$CODE_DIR" "$ENV_DIR/bin/python" -m unittest \
  test_supervised_sft test_fp32_master -v \
  > "$RUN_DIR/logs/preflight-tests.log" 2>&1
"$ENV_DIR/bin/python" -c \
  'from torch.distributed.fsdp import FullyShardedDataParallel, ShardingStrategy; from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl, checkpoint_wrapper; assert ShardingStrategy.FULL_SHARD is not None; assert CheckpointImpl.NO_REENTRANT is not None; assert callable(checkpoint_wrapper)' \
  > "$RUN_DIR/logs/fsdp-import-preflight.log" 2>&1
PYTHONPATH="$CODE_DIR" "$ENV_DIR/bin/torchrun" \
  --standalone --nproc-per-node=2 \
  "$CODE_DIR/fsdp_fp32_preflight.py" \
  > "$RUN_DIR/logs/fsdp-fp32-preflight.log" 2>&1

PYTHONPATH="$CODE_DIR" "$ENV_DIR/bin/python" \
  "$CODE_DIR/preflight_supervised_sft.py" \
  --data "$DATA_FILE" \
  --manifest "$MANIFEST_FILE" \
  --config "$CONFIG_FILE" \
  --model "$MODEL_DIR" \
  --expected-data-sha256 "$TRAIN_JSONL_SHA256" \
  --expected-manifest-sha256 "$TRAIN_MANIFEST_SHA256" \
  --code-ledger "$RUN_DIR/code.sha256" \
  --expected-code-ledger-sha256 "$EXPECTED_CODE_LEDGER_SHA256" \
  --model-artifact-ledger "$RUN_DIR/model-artifacts.sha256" \
  --expected-model-artifact-ledger-sha256 \
    "$EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256" \
  > "$RUN_DIR/logs/protocol-preflight.log"
date -u +%FT%TZ > "$RUN_DIR/stages/01_preflight_ok"

# Recheck after imports/tests/preflight so no artifact can drift before torchrun.
sha256sum -c "$RUN_DIR/code.sha256" >> "$RUN_DIR/logs/code-integrity.log"
sha256sum -c "$RUN_DIR/runtime-dependencies.sha256" \
  >> "$RUN_DIR/logs/runtime-dependency-integrity.log"
sha256sum -c "$RUN_DIR/model-artifacts.sha256" \
  >> "$RUN_DIR/logs/model-artifact-integrity.log"
# Weight shards total roughly 72 GB. The one full verification pass above is
# intentionally not repeated; the trainer pins the ledger SHA and exact paths.

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
"$ENV_DIR/bin/torchrun" \
  --standalone \
  --nproc-per-node=8 \
  "$CODE_DIR/train_supervised_sft.py" \
  --config "$CONFIG_FILE" \
  --model "$MODEL_DIR" \
  --data "$DATA_FILE" \
  --manifest "$MANIFEST_FILE" \
  --expected-data-sha256 "$TRAIN_JSONL_SHA256" \
  --expected-manifest-sha256 "$TRAIN_MANIFEST_SHA256" \
  --code-ledger "$RUN_DIR/code.sha256" \
  --expected-code-ledger-sha256 "$EXPECTED_CODE_LEDGER_SHA256" \
  --model-artifact-ledger "$RUN_DIR/model-artifacts.sha256" \
  --expected-model-artifact-ledger-sha256 \
    "$EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256" \
  --runtime-dependency-ledger "$RUN_DIR/runtime-dependencies.sha256" \
  --expected-runtime-dependency-ledger-sha256 \
    "$EXPECTED_RUNTIME_DEPENDENCY_LEDGER_SHA256" \
  --model-weight-ledger "$RUN_DIR/model-weights.sha256" \
  --expected-model-weight-ledger-sha256 \
    "$EXPECTED_MODEL_WEIGHT_LEDGER_SHA256" \
  --output-dir "$RUN_DIR" \
  > "$RUN_DIR/logs/train.log" 2>&1

test -s "$RUN_DIR/metadata.json"
test -s "$RUN_DIR/metrics.jsonl"
"$ENV_DIR/bin/python" - "$RUN_DIR/metadata.json" \
  "$EXPECTED_CODE_LEDGER_SHA256" \
  "$EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256" \
  "$EXPECTED_RUNTIME_DEPENDENCY_LEDGER_SHA256" \
  "$EXPECTED_MODEL_WEIGHT_LEDGER_SHA256" <<'PY'
import json
import sys
from pathlib import Path

metadata = json.loads(Path(sys.argv[1]).read_text())
expected_code_sha256 = sys.argv[2]
expected_model_sha256 = sys.argv[3]
expected_runtime_sha256 = sys.argv[4]
expected_weight_sha256 = sys.argv[5]
if metadata.get("training_scope") != "dense_full_model_sft_smoke":
    raise SystemExit("wrong supervised training scope")
if metadata.get("last_step") != 1 or not metadata.get("smoke_only"):
    raise SystemExit("run escaped the frozen 1-step smoke boundary")
plan = metadata.get("parameter_plan", {})
if not plan.get("exact_parameter_count_gate_passed"):
    raise SystemExit("exact full-model parameter gate did not pass")
if plan.get("trainable_parameters") != 34_660_610_688:
    raise SystemExit("unexpected trainable parameter count")
distributed = metadata.get("distributed", {})
if distributed.get("kind") != "FSDP1_FULL_SHARD" or distributed.get("ddp_used"):
    raise SystemExit("only FSDP FULL_SHARD is an admissible result")
checkpoint = metadata.get("checkpoint", {})
if checkpoint.get("mode") != "metadata-only" or checkpoint.get(
    "model_or_optimizer_artifact_written"
):
    raise SystemExit("smoke must remain metadata-only")
suffix_gate = metadata.get("qcomem_suffix_supervised_sft_capability", {})
if suffix_gate.get("capability_gate_passed") or suffix_gate.get("implemented"):
    raise SystemExit("Q-CoMem suffix supervised SFT must remain fail-closed")
if suffix_gate.get("observed_blocker", {}).get("trial_id") != 1830867:
    raise SystemExit("missing cached-two-stage mutable-cache autograd failure evidence")
integrity = metadata.get("integrity", {})
code = integrity.get("code", {})
model_artifacts = integrity.get("model_artifacts", {})
runtime_dependencies = integrity.get("runtime_dependencies", {})
model_weights = integrity.get("model_weights", {})
if (
    code.get("ledger_sha256") != expected_code_sha256
    or not code.get("all_artifacts_exist_and_match")
):
    raise SystemExit("code.sha256 was not preserved in trainer metadata")
if (
    model_artifacts.get("ledger_sha256") != expected_model_sha256
    or not model_artifacts.get("all_artifacts_exist_and_match")
):
    raise SystemExit("model artifact SHA ledger was not preserved in trainer metadata")
if (
    runtime_dependencies.get("ledger_sha256") != expected_runtime_sha256
    or not runtime_dependencies.get("all_artifacts_exist_and_match")
):
    raise SystemExit("runtime dependency SHA ledger was not preserved in metadata")
if (
    model_weights.get("ledger_sha256") != expected_weight_sha256
    or not model_weights.get("all_artifacts_exist_and_match")
):
    raise SystemExit("model weight SHA ledger was not preserved in metadata")
delta = metadata.get("parameter_delta", {})
if delta.get("fp32_logical", {}).get("total", {}).get("nonzero_elements", 0) <= 0:
    raise SystemExit("FP32 parameter delta gate did not pass")
gradient = metadata.get("gradient_coverage", {}).get("total", {})
if gradient.get("missing_elements") or gradient.get("nonfinite_elements"):
    raise SystemExit("full-model gradient coverage gate did not pass")
optimizer_gates = metadata.get("optimizer_state_gate_by_rank", [])
if len(optimizer_gates) != 8 or not all(
    gate.get("all_steps_match")
    and gate.get("nonfinite_moment_elements") == 0
    and gate.get("moment_dtype_elements", {}).get("torch.float32")
        == 2 * gate.get("parameter_elements", -1)
    for gate in optimizer_gates
):
    raise SystemExit("FP32 AdamW state/step gate did not pass")
print({"status": "passed", "last_step": 1, "trainable_parameters": 34_660_610_688})
PY
nvidia-smi \
  --query-gpu=index,uuid,name,memory.used,pstate,power.draw,temperature.gpu \
  --format=csv > "$RUN_DIR/gpus-after.csv"
date -u +%FT%TZ > "$RUN_DIR/stages/99_done"
echo "dense_full_model_sft_smoke complete: $RUN_DIR"
