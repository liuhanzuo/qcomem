#!/usr/bin/env bash
set -euo pipefail

CODE_DIR=${CODE_DIR:?set CODE_DIR}
MODEL_DIR=${MODEL_DIR:?set MODEL_DIR}
TRAIN_DATA_FILE=${TRAIN_DATA_FILE:?set TRAIN_DATA_FILE}
HELDOUT_DATA_FILE=${HELDOUT_DATA_FILE:?set HELDOUT_DATA_FILE}
SPLIT_MANIFEST_FILE=${SPLIT_MANIFEST_FILE:?set SPLIT_MANIFEST_FILE}
PARENT_JSONL_FILE=${PARENT_JSONL_FILE:?set PARENT_JSONL_FILE}
ASSIGNMENT_LEDGER_FILE=${ASSIGNMENT_LEDGER_FILE:?set ASSIGNMENT_LEDGER_FILE}
TRAIN_SHA256=${TRAIN_SHA256:?set TRAIN_SHA256}
HELDOUT_SHA256=${HELDOUT_SHA256:?set HELDOUT_SHA256}
SPLIT_MANIFEST_SHA256=${SPLIT_MANIFEST_SHA256:?set SPLIT_MANIFEST_SHA256}
EXPECTED_CODE_LEDGER_SHA256=${EXPECTED_CODE_LEDGER_SHA256:?set EXPECTED_CODE_LEDGER_SHA256}
EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256=${EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256:?set EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256}
EXPECTED_MODEL_WEIGHT_LEDGER_SHA256=${EXPECTED_MODEL_WEIGHT_LEDGER_SHA256:?set EXPECTED_MODEL_WEIGHT_LEDGER_SHA256}
MODEL_WEIGHT_LEDGER_FILE=${MODEL_WEIGHT_LEDGER_FILE:?set MODEL_WEIGHT_LEDGER_FILE}
RUN_DIR=${RUN_DIR:?set RUN_DIR}
ENV_DIR=${ENV_DIR:?set ENV_DIR}
CONFIG_FILE=${CONFIG_FILE:?set CONFIG_FILE}

for VALUE in \
  "$TRAIN_SHA256" \
  "$HELDOUT_SHA256" \
  "$SPLIT_MANIFEST_SHA256" \
  "$EXPECTED_CODE_LEDGER_SHA256" \
  "$EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256" \
  "$EXPECTED_MODEL_WEIGHT_LEDGER_SHA256"; do
  if [[ ! "$VALUE" =~ ^[0-9a-f]{64}$ ]]; then
    echo "all expected hashes must be lowercase SHA256 values" >&2
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
  "$CODE_DIR/train_supervised_sft_longrun.py"
  "$CODE_DIR/supervised_sft_longrun.py"
  "$CODE_DIR/sft_dcp_checkpoint.py"
  "$CODE_DIR/sft_quality_validation.py"
  "$CODE_DIR/test_supervised_sft_longrun.py"
  "$CODE_DIR/test_sft_quality_validation.py"
  "$CODE_DIR/fsdp_dcp_longrun_preflight.py"
  "$CODE_DIR/launch_supervised_sft_longrun_8gpu.sh"
  "$CONFIG_FILE"
  "$CODE_DIR/supervised_sft.py"
  "$CODE_DIR/fp32_master.py"
  "$CODE_DIR/qcomem_torch.py"
  "$CODE_DIR/run_downstream.py"
  "$CODE_DIR/split_supervised_sft_scale.py"
  "$CODE_DIR/audit_supervised_sft_scale.py"
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
for ARTIFACT in \
  "${CODE_FILES[@]}" \
  "${MODEL_ARTIFACT_FILES[@]}" \
  "${MODEL_WEIGHT_FILES[@]}" \
  "$MODEL_WEIGHT_LEDGER_FILE" \
  "$TRAIN_DATA_FILE" \
  "$HELDOUT_DATA_FILE" \
  "$SPLIT_MANIFEST_FILE" \
  "$PARENT_JSONL_FILE" \
  "$ASSIGNMENT_LEDGER_FILE"; do
  if [ ! -f "$ARTIFACT" ]; then
    echo "required formal artifact is missing: $ARTIFACT" >&2
    exit 2
  fi
done

sha256sum "${CODE_FILES[@]}" > "$RUN_DIR/code.sha256"
sha256sum "${MODEL_ARTIFACT_FILES[@]}" > "$RUN_DIR/model-artifacts.sha256"
cp "$MODEL_WEIGHT_LEDGER_FILE" "$RUN_DIR/model-weights.sha256"
ACTUAL_CODE_LEDGER_SHA256=$(sha256sum "$RUN_DIR/code.sha256" | awk '{print $1}')
ACTUAL_MODEL_ARTIFACT_LEDGER_SHA256=$(
  sha256sum "$RUN_DIR/model-artifacts.sha256" | awk '{print $1}'
)
ACTUAL_MODEL_WEIGHT_LEDGER_SHA256=$(
  sha256sum "$RUN_DIR/model-weights.sha256" | awk '{print $1}'
)
if [ "$ACTUAL_CODE_LEDGER_SHA256" != "$EXPECTED_CODE_LEDGER_SHA256" ]; then
  echo "formal code ledger SHA256 mismatch" >&2
  exit 2
fi
if [ "$ACTUAL_MODEL_ARTIFACT_LEDGER_SHA256" != "$EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256" ]; then
  echo "formal model-artifact ledger SHA256 mismatch" >&2
  exit 2
fi
if [ "$ACTUAL_MODEL_WEIGHT_LEDGER_SHA256" != "$EXPECTED_MODEL_WEIGHT_LEDGER_SHA256" ]; then
  echo "formal model-weight ledger SHA256 mismatch" >&2
  exit 2
fi
sha256sum -c "$RUN_DIR/code.sha256" > "$RUN_DIR/logs/code-integrity.log"
sha256sum -c "$RUN_DIR/model-artifacts.sha256" \
  > "$RUN_DIR/logs/model-artifact-integrity.log"
# One full model-weight verification before torchrun. The trainer subsequently
# pins this verified ledger and exact paths without rereading all shards 8 times.
sha256sum -c "$RUN_DIR/model-weights.sha256" \
  > "$RUN_DIR/logs/model-weight-integrity.log"

if [ "$(sha256sum "$TRAIN_DATA_FILE" | awk '{print $1}')" != "$TRAIN_SHA256" ]; then
  echo "formal train JSONL SHA256 mismatch" >&2
  exit 2
fi
if [ "$(sha256sum "$HELDOUT_DATA_FILE" | awk '{print $1}')" != "$HELDOUT_SHA256" ]; then
  echo "formal CE-heldout JSONL SHA256 mismatch" >&2
  exit 2
fi
if [ "$(sha256sum "$SPLIT_MANIFEST_FILE" | awk '{print $1}')" != "$SPLIT_MANIFEST_SHA256" ]; then
  echo "formal split manifest SHA256 mismatch" >&2
  exit 2
fi
printf '%s  %s\n' "$TRAIN_SHA256" "$TRAIN_DATA_FILE" \
  > "$RUN_DIR/train-data.sha256"
printf '%s  %s\n' "$HELDOUT_SHA256" "$HELDOUT_DATA_FILE" \
  > "$RUN_DIR/heldout-data.sha256"
printf '%s  %s\n' "$SPLIT_MANIFEST_SHA256" "$SPLIT_MANIFEST_FILE" \
  > "$RUN_DIR/split-manifest.sha256"

GPU_COUNT=$(nvidia-smi -L | wc -l | tr -d ' ')
if [ "$GPU_COUNT" -ne 8 ]; then
  echo "formal dense SFT requires exactly 8 GPUs, found $GPU_COUNT" >&2
  exit 2
fi
nvidia-smi \
  --query-gpu=index,uuid,name,memory.total,driver_version,pstate,power.limit \
  --format=csv > "$RUN_DIR/gpus-before.csv"

# Compile only Python sources; the deliberately separate list above also binds
# shell/config files in the ledger.
"$ENV_DIR/bin/python" -m py_compile \
  "$CODE_DIR/train_supervised_sft_longrun.py" \
  "$CODE_DIR/supervised_sft_longrun.py" \
  "$CODE_DIR/sft_dcp_checkpoint.py" \
  "$CODE_DIR/sft_quality_validation.py" \
  "$CODE_DIR/fsdp_dcp_longrun_preflight.py" \
  "$CODE_DIR/supervised_sft.py" \
  "$CODE_DIR/fp32_master.py" \
  "$CODE_DIR/split_supervised_sft_scale.py" \
  "$CODE_DIR/audit_supervised_sft_scale.py"
bash -n "$CODE_DIR/launch_supervised_sft_longrun_8gpu.sh"
PYTHONPATH="$CODE_DIR" "$ENV_DIR/bin/python" -m unittest \
  test_supervised_sft_longrun test_sft_quality_validation -v \
  > "$RUN_DIR/logs/unit-tests.log" 2>&1
PYTHONPATH="$CODE_DIR" "$ENV_DIR/bin/python" \
  "$CODE_DIR/audit_supervised_sft_scale.py" \
  --manifest "$SPLIT_MANIFEST_FILE" \
  --expected-manifest-sha256 "$SPLIT_MANIFEST_SHA256" \
  --parent-jsonl "$PARENT_JSONL_FILE" \
  --train-jsonl "$TRAIN_DATA_FILE" \
  --heldout-ce-jsonl "$HELDOUT_DATA_FILE" \
  --assignment-ledger "$ASSIGNMENT_LEDGER_FILE" \
  > "$RUN_DIR/logs/split-audit.json"
PYTHONPATH="$CODE_DIR" "$ENV_DIR/bin/torchrun" \
  --standalone --nproc-per-node=2 \
  "$CODE_DIR/fsdp_dcp_longrun_preflight.py" \
  --output-root "$RUN_DIR/dcp-tiny-preflight" \
  > "$RUN_DIR/logs/dcp-tiny-preflight.log" 2>&1
date -u +%FT%TZ > "$RUN_DIR/stages/01_preflight_ok"

# Recheck small code/data artifacts immediately before the expensive job. Model
# weights were already read once in full and remain bound by the immutable ledger.
sha256sum -c "$RUN_DIR/code.sha256" >> "$RUN_DIR/logs/code-integrity.log"
sha256sum -c "$RUN_DIR/model-artifacts.sha256" \
  >> "$RUN_DIR/logs/model-artifact-integrity.log"
sha256sum -c "$RUN_DIR/train-data.sha256" > /dev/null
sha256sum -c "$RUN_DIR/heldout-data.sha256" > /dev/null
sha256sum -c "$RUN_DIR/split-manifest.sha256" > /dev/null

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ENABLE_MONITORING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200
PYTHONPATH="$CODE_DIR" "$ENV_DIR/bin/torchrun" \
  --standalone --nproc-per-node=8 \
  "$CODE_DIR/train_supervised_sft_longrun.py" \
  --config "$CONFIG_FILE" \
  --model "$MODEL_DIR" \
  --train-data "$TRAIN_DATA_FILE" \
  --heldout-data "$HELDOUT_DATA_FILE" \
  --split-manifest "$SPLIT_MANIFEST_FILE" \
  --expected-train-sha256 "$TRAIN_SHA256" \
  --expected-heldout-sha256 "$HELDOUT_SHA256" \
  --expected-split-manifest-sha256 "$SPLIT_MANIFEST_SHA256" \
  --code-ledger "$RUN_DIR/code.sha256" \
  --expected-code-ledger-sha256 "$EXPECTED_CODE_LEDGER_SHA256" \
  --model-artifact-ledger "$RUN_DIR/model-artifacts.sha256" \
  --expected-model-artifact-ledger-sha256 \
    "$EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256" \
  --model-weight-ledger "$RUN_DIR/model-weights.sha256" \
  --expected-model-weight-ledger-sha256 "$EXPECTED_MODEL_WEIGHT_LEDGER_SHA256" \
  --output-dir "$RUN_DIR/artifacts" \
  > "$RUN_DIR/logs/train.log" 2>&1

test -s "$RUN_DIR/artifacts/metadata.json"
test -s "$RUN_DIR/artifacts/train-metrics.jsonl"
test -s "$RUN_DIR/artifacts/heldout-metrics.jsonl"
test -s "$RUN_DIR/artifacts/checkpoint-metrics.jsonl"
test -s "$RUN_DIR/artifacts/best-checkpoint.json"
"$ENV_DIR/bin/python" - "$RUN_DIR/artifacts" <<'PY'
import hashlib
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
metadata = json.loads((root / "metadata.json").read_text())
if metadata.get("format") != "qcomem_dense_full_sft_formal_v1":
    raise SystemExit("wrong formal metadata format")
if metadata.get("last_step") != 384:
    raise SystemExit("formal trainer did not finish 384 steps")
if metadata.get("raw_test_v2_read") is not False:
    raise SystemExit("formal metadata lost the unread test-v2 gate")
if metadata.get("cumulative_dataset_counts") != {
    "2wikimqa": 1536,
    "qasper": 1536,
}:
    raise SystemExit("formal run did not consume exactly three balanced epochs")
gate = metadata.get("fp32_gradient_optimizer_delta_gates", {}).get("1", {})
if gate.get("parameter_delta", {}).get("fp32_logical", {}).get(
    "total", {}
).get("nonzero_elements", 0) <= 0:
    raise SystemExit("initial-to-step-1 FP32 parameter delta gate did not pass")
checkpoint = metadata.get("checkpoint", {})
if checkpoint.get("observed_completed_steps") != [128, 256, 384]:
    raise SystemExit("the three model-only checkpoints did not all complete")
if checkpoint.get("model_write_observed") is not True:
    raise SystemExit("formal metadata does not attest observed checkpoint writes")
train_rows = [json.loads(line) for line in (root / "train-metrics.jsonl").read_text().splitlines()]
if [row.get("step") for row in train_rows] != list(range(1, 385)):
    raise SystemExit("formal train metrics are not exactly steps 1..384")
if not all(row.get("learning_rate_used", 0) > 0 for row in train_rows):
    raise SystemExit("one or more of the 384 optimizer updates used zero LR")
if train_rows[-1].get("next_learning_rate") != 0.0:
    raise SystemExit("cosine schedule did not reach zero after the final update")
heldout_rows = [json.loads(line) for line in (root / "heldout-metrics.jsonl").read_text().splitlines()]
if [row.get("step") for row in heldout_rows] != [0, 128, 256, 384]:
    raise SystemExit("heldout CE phases are not the four frozen boundaries")

checkpoint_records = metadata.get("checkpoint_records", {})
for step in (128, 256, 384):
    record = checkpoint_records.get(str(step), {})
    checkpoint_root = Path(record.get("checkpoint_path", ""))
    manifest_path = checkpoint_root / "checkpoint-manifest.json"
    success_path = checkpoint_root / "_SUCCESS"
    if not manifest_path.is_file() or not success_path.is_file():
        raise SystemExit(f"checkpoint {step} is not atomically complete")
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if manifest_sha != record.get("checkpoint_manifest_sha256"):
        raise SystemExit(f"checkpoint {step} manifest pointer mismatch")
    manifest = json.loads(manifest_path.read_text())
    success = json.loads(success_path.read_text())
    if (
        manifest.get("contract") != "eval_model_only_fp32"
        or manifest.get("step") != step
        or manifest.get("world_size") != 8
        or manifest.get("global_parameter_count") != 34660610688
        or manifest.get("persistent_parameter_dtype") != "torch.float32"
        or manifest.get("rank0_full_gather_used") is not False
        or manifest.get("state", {}).get("optimizer") is not False
        or manifest.get("actual_payload_bytes", 0) < manifest.get("logical_model_bytes", 1)
        or success.get("checkpoint_manifest_sha256") != manifest_sha
        or success.get("payload_directory_sha256")
            != manifest.get("payload_directory_sha256")
    ):
        raise SystemExit(f"checkpoint {step} manifest contract failed")

best = json.loads((root / "best-checkpoint.json").read_text())
candidate_ce = {
    row["step"]: row["summary"]["overall"]["token_weighted_ce"]
    for row in heldout_rows
    if row["step"] in {128, 256, 384}
}
if not all(math.isfinite(value) for value in candidate_ce.values()):
    raise SystemExit("checkpoint candidate CE is non-finite")
expected_best = min(candidate_ce, key=lambda step: (candidate_ce[step], step))
if best.get("selected_step") != expected_best:
    raise SystemExit("best checkpoint does not minimize heldout token-weighted CE")
print({"status": "passed", "steps": 384, "best_step": expected_best})
PY
nvidia-smi \
  --query-gpu=index,uuid,name,memory.used,pstate,power.draw,temperature.gpu \
  --format=csv > "$RUN_DIR/gpus-after.csv"
date -u +%FT%TZ > "$RUN_DIR/stages/99_done"
echo "dense_full_model_sft_formal_384 complete: $RUN_DIR"
