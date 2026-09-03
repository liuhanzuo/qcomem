#!/usr/bin/env bash
set -euo pipefail

# Formal-only launcher.  There is intentionally no smoke/fallback path.
export LC_ALL=C

CODE_DIR=${CODE_DIR:?set immutable gpu source directory}
MODEL_DIR=${MODEL_DIR:?set pinned Qwen3.5 model directory}
MODEL_ARTIFACT_LEDGER_FILE=${MODEL_ARTIFACT_LEDGER_FILE:?set artifact SHA-256 ledger}
MODEL_WEIGHT_LEDGER_FILE=${MODEL_WEIGHT_LEDGER_FILE:?set fourteen-shard weight ledger}
PG19_DATA=${PG19_DATA:?set exact train64 JSONL}
PG19_MANIFEST=${PG19_MANIFEST:?set exact train64 source manifest}
SOURCE_MANIFEST=${SOURCE_MANIFEST:?set frozen transfer source manifest}
STATIC_MANIFEST=${STATIC_MANIFEST:?set frozen transfer preregistration}
RUN_DIR=${RUN_DIR:?set a new output directory}
ENV_DIR=${ENV_DIR:?set frozen Python environment}
EXPECTED_SOURCE_MANIFEST_SHA256=${EXPECTED_SOURCE_MANIFEST_SHA256:?set frozen source manifest raw SHA}
EXPECTED_STATIC_MANIFEST_SHA256=${EXPECTED_STATIC_MANIFEST_SHA256:?set frozen static manifest raw SHA}
EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256=${EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256:?set artifact ledger raw SHA}
EXPECTED_MODEL_WEIGHT_LEDGER_SHA256=${EXPECTED_MODEL_WEIGHT_LEDGER_SHA256:?set weight ledger raw SHA}

PYTHON="$ENV_DIR/bin/python"
RUNNER="$CODE_DIR/run_qcomem_transformers_forkaudit_transfer.py"
BUILDER="$CODE_DIR/build_qcomem_transformers_forkaudit_transfer_prereg.py"

if [[ -e "$RUN_DIR" ]]; then
  echo "RUN_DIR must not exist before the formal launch" >&2
  exit 2
fi
for digest in \
  "$EXPECTED_SOURCE_MANIFEST_SHA256" "$EXPECTED_STATIC_MANIFEST_SHA256" \
  "$EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256" "$EXPECTED_MODEL_WEIGHT_LEDGER_SHA256"; do
  [[ "$digest" =~ ^[0-9a-f]{64}$ ]] || { echo "invalid frozen SHA-256" >&2; exit 2; }
done

mkdir -p "$RUN_DIR"/{logs,stages,receipts,raw/shards,raw/logits,pycache}
cp "$SOURCE_MANIFEST" "$RUN_DIR/receipts/frozen-source-manifest.json"
cp "$STATIC_MANIFEST" "$RUN_DIR/receipts/frozen-static-manifest.json"
export PYTHONPYCACHEPREFIX="$RUN_DIR/pycache"
export PYTHONPATH="$CODE_DIR"
PIDS=()
cleanup_failure() {
  status=$?
  trap - ERR INT TERM
  for pid in "${PIDS[@]:-}"; do kill -TERM "$pid" 2>/dev/null || true; done
  for pid in "${PIDS[@]:-}"; do wait "$pid" 2>/dev/null || true; done
  date -u +%FT%TZ > "$RUN_DIR/stages/FAILED"
  exit "$status"
}
trap cleanup_failure ERR INT TERM
date -u +%FT%TZ > "$RUN_DIR/stages/00_start"

# Source authenticity and input reconstruction both occur before any model output.
"$PYTHON" -B "$BUILDER" --stage verify-source \
  --source-root "$CODE_DIR" --source-manifest "$SOURCE_MANIFEST" \
  --expected-source-manifest-sha256 "$EXPECTED_SOURCE_MANIFEST_SHA256" \
  --output "$RUN_DIR/receipts/source-verification.json"

# Exactly one pre-output full read of all 14 weight shards.
"$PYTHON" -B "$BUILDER" --stage model-authority \
  --model "$MODEL_DIR" --model-artifact-ledger "$MODEL_ARTIFACT_LEDGER_FILE" \
  --model-weight-ledger "$MODEL_WEIGHT_LEDGER_FILE" \
  --expected-model-artifact-ledger-sha256 "$EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256" \
  --expected-model-weight-ledger-sha256 "$EXPECTED_MODEL_WEIGHT_LEDGER_SHA256" \
  --output "$RUN_DIR/receipts/model-authority-pre.json"
MODEL_AUTHORITY_SHA256=$(sha256sum "$RUN_DIR/receipts/model-authority-pre.json" | awk '{print $1}')

"$PYTHON" -B "$BUILDER" --stage verify-static \
  --source-root "$CODE_DIR" --source-manifest "$SOURCE_MANIFEST" \
  --expected-source-manifest-sha256 "$EXPECTED_SOURCE_MANIFEST_SHA256" \
  --pg19-data "$PG19_DATA" --pg19-manifest "$PG19_MANIFEST" \
  --model "$MODEL_DIR" --model-artifact-ledger "$MODEL_ARTIFACT_LEDGER_FILE" \
  --model-weight-ledger "$MODEL_WEIGHT_LEDGER_FILE" \
  --expected-model-artifact-ledger-sha256 "$EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256" \
  --expected-model-weight-ledger-sha256 "$EXPECTED_MODEL_WEIGHT_LEDGER_SHA256" \
  --static-manifest "$STATIC_MANIFEST" \
  --expected-static-manifest-sha256 "$EXPECTED_STATIC_MANIFEST_SHA256" \
  --output "$RUN_DIR/receipts/static-bytewise-rebuild.json"

"$PYTHON" -B "$BUILDER" --stage gpu-assignment \
  --output "$RUN_DIR/receipts/gpu-assignment.json"
GPU_ASSIGNMENT_SHA256=$(sha256sum "$RUN_DIR/receipts/gpu-assignment.json" | awk '{print $1}')
mapfile -t GPU_UUIDS < <("$PYTHON" -B - "$RUN_DIR/receipts/gpu-assignment.json" <<'PY'
import json, sys
for row in json.load(open(sys.argv[1], encoding="utf-8"))["rows"]:
    print(row["uuid"])
PY
)
[[ ${#GPU_UUIDS[@]} -eq 8 ]] || { echo "GPU assignment does not contain eight ranks" >&2; exit 2; }
date -u +%FT%TZ > "$RUN_DIR/stages/10_preflight_complete"

for rank in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=${GPU_UUIDS[$rank]} "$PYTHON" -B "$RUNNER" --stage shard \
    --model "$MODEL_DIR" --model-artifact-ledger "$MODEL_ARTIFACT_LEDGER_FILE" \
    --model-weight-ledger "$MODEL_WEIGHT_LEDGER_FILE" \
    --model-authority "$RUN_DIR/receipts/model-authority-pre.json" \
    --expected-model-authority-sha256 "$MODEL_AUTHORITY_SHA256" \
    --gpu-assignment "$RUN_DIR/receipts/gpu-assignment.json" \
    --expected-gpu-assignment-sha256 "$GPU_ASSIGNMENT_SHA256" \
    --static-manifest "$STATIC_MANIFEST" --source-manifest "$SOURCE_MANIFEST" \
    --source-root "$CODE_DIR" \
    --expected-static-manifest-sha256 "$EXPECTED_STATIC_MANIFEST_SHA256" \
    --expected-source-manifest-sha256 "$EXPECTED_SOURCE_MANIFEST_SHA256" \
    --expected-model-artifact-ledger-sha256 "$EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256" \
    --expected-model-weight-ledger-sha256 "$EXPECTED_MODEL_WEIGHT_LEDGER_SHA256" \
    --rank "$rank" --expected-gpu-uuid "${GPU_UUIDS[$rank]}" \
    --logit-sidecar "$RUN_DIR/raw/logits/forkaudit-transformers-transfer-logits-rank-$rank.bin" \
    --output "$RUN_DIR/raw/shards/forkaudit-transformers-transfer-shard-$rank.json" \
    >"$RUN_DIR/logs/rank-$rank.log" 2>&1 &
  PIDS+=("$!")
done
for pid in "${PIDS[@]}"; do wait "$pid"; done
PIDS=()
[[ $(find "$RUN_DIR/raw/shards" -type f -name '*.json' | wc -l) -eq 8 ]] || exit 2
[[ $(find "$RUN_DIR/raw/logits" -type f -name '*.bin' | wc -l) -eq 8 ]] || exit 2
date -u +%FT%TZ > "$RUN_DIR/stages/20_shards_complete"

# Exactly one terminal full reread; byte equality closes model identity over outputs.
"$PYTHON" -B "$BUILDER" --stage model-authority \
  --model "$MODEL_DIR" --model-artifact-ledger "$MODEL_ARTIFACT_LEDGER_FILE" \
  --model-weight-ledger "$MODEL_WEIGHT_LEDGER_FILE" \
  --expected-model-artifact-ledger-sha256 "$EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256" \
  --expected-model-weight-ledger-sha256 "$EXPECTED_MODEL_WEIGHT_LEDGER_SHA256" \
  --output "$RUN_DIR/receipts/model-authority-terminal.json"
cmp "$RUN_DIR/receipts/model-authority-pre.json" "$RUN_DIR/receipts/model-authority-terminal.json"

"$PYTHON" -B "$RUNNER" --stage aggregate \
  --model "$MODEL_DIR" --model-artifact-ledger "$MODEL_ARTIFACT_LEDGER_FILE" \
  --model-weight-ledger "$MODEL_WEIGHT_LEDGER_FILE" \
  --model-authority "$RUN_DIR/receipts/model-authority-pre.json" \
  --model-closure "$RUN_DIR/receipts/model-authority-terminal.json" \
  --expected-model-authority-sha256 "$MODEL_AUTHORITY_SHA256" \
  --gpu-assignment "$RUN_DIR/receipts/gpu-assignment.json" \
  --expected-gpu-assignment-sha256 "$GPU_ASSIGNMENT_SHA256" \
  --static-manifest "$STATIC_MANIFEST" --source-manifest "$SOURCE_MANIFEST" \
  --source-root "$CODE_DIR" --shard-dir "$RUN_DIR/raw/shards" \
  --sidecar-dir "$RUN_DIR/raw/logits" \
  --expected-static-manifest-sha256 "$EXPECTED_STATIC_MANIFEST_SHA256" \
  --expected-source-manifest-sha256 "$EXPECTED_SOURCE_MANIFEST_SHA256" \
  --expected-model-artifact-ledger-sha256 "$EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256" \
  --expected-model-weight-ledger-sha256 "$EXPECTED_MODEL_WEIGHT_LEDGER_SHA256" \
  --output "$RUN_DIR/forkaudit-transformers-transfer-aggregate.json" \
  >"$RUN_DIR/logs/aggregate.log" 2>&1

"$PYTHON" -B - "$RUN_DIR" <<'PY'
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1])
paths = sorted(
    [root / "forkaudit-transformers-transfer-aggregate.json"]
    + list((root / "raw" / "shards").glob("*.json"))
    + list((root / "raw" / "logits").glob("*.bin"))
    + list((root / "receipts").glob("*.json")),
    key=lambda path: path.relative_to(root).as_posix(),
)
rows=[]
for path in paths:
    raw=path.read_bytes()
    rows.append({"path":path.relative_to(root).as_posix(),"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()})
out={"schema_version":"forkaudit-transformers-transfer-artifact-ledger-v1","rows":rows}
payload=json.dumps(out,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()+b"\n"
(root/"artifact-ledger.json").write_bytes(payload)
PY
date -u +%FT%TZ > "$RUN_DIR/stages/COMPLETE"
trap - ERR INT TERM
