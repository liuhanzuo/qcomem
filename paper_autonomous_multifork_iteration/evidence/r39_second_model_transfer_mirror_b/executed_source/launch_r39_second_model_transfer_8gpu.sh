#!/usr/bin/env bash
set -euo pipefail

# Formal-only eight-rank launcher.  No smoke, fallback, rank downshift, or
# selective retry exists.  This script does not call QS or manage resources.

export LC_ALL=C
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1

REPO_ROOT=${REPO_ROOT:?set immutable staged repository root}
PACKAGE_ROOT="$REPO_ROOT/paper_autonomous_multifork_iteration/evidence/r39_second_model_transfer_mirror_b"
ENV_DIR=${ENV_DIR:?set frozen Transformers environment}
MODEL_ROOT=${MODEL_ROOT:?set dedicated revision-named immutable model directory}
RUN_ROOT=${RUN_ROOT:?set a fresh R39 output directory}
EXPECTED_SOURCE_SHA256=${EXPECTED_SOURCE_SHA256:?set frozen source-manifest SHA-256}
EXPECTED_STATIC_SHA256=${EXPECTED_STATIC_SHA256:?set frozen static-preregistration SHA-256}

PYTHON="$ENV_DIR/bin/python"
SOURCE_MANIFEST="$PACKAGE_ROOT/preregistration/source-manifest.json"
STATIC_MANIFEST="$PACKAGE_ROOT/preregistration/static-preregistration.json"
BUILDER="$PACKAGE_ROOT/executed_source/build_r39_preregistration.py"
MODEL_PREPARER="$PACKAGE_ROOT/executed_source/prepare_r39_model_snapshot.py"
RUNNER="$PACKAGE_ROOT/executed_source/run_r39_second_model_transfer.py"
REPLAY="$PACKAGE_ROOT/executed_source/replay_r39_second_model_transfer.py"

[[ ! -e "$RUN_ROOT" ]] || { echo "RUN_ROOT already exists: $RUN_ROOT" >&2; exit 2; }
[[ "$EXPECTED_SOURCE_SHA256" =~ ^[0-9a-f]{64}$ ]] || { echo "bad source SHA" >&2; exit 2; }
[[ "$EXPECTED_STATIC_SHA256" =~ ^[0-9a-f]{64}$ ]] || { echo "bad static SHA" >&2; exit 2; }
for item in "$PYTHON" "$SOURCE_MANIFEST" "$STATIC_MANIFEST" "$BUILDER" "$MODEL_PREPARER" "$RUNNER" "$REPLAY"; do
  [[ -e "$item" ]] || { echo "required path absent: $item" >&2; exit 2; }
done

mkdir -p "$RUN_ROOT"/{logs,raw/shards,raw/logits,receipts,stages,pycache}
cp "$SOURCE_MANIFEST" "$RUN_ROOT/receipts/frozen-source-manifest.json"
cp "$STATIC_MANIFEST" "$RUN_ROOT/receipts/frozen-static-preregistration.json"
export PYTHONPYCACHEPREFIX="$RUN_ROOT/pycache"
export PYTHONPATH="$PACKAGE_ROOT/executed_source"

PIDS=()
fail_closed() {
  status=$?
  trap - ERR INT TERM
  for pid in "${PIDS[@]:-}"; do kill -TERM "$pid" 2>/dev/null || true; done
  for pid in "${PIDS[@]:-}"; do wait "$pid" 2>/dev/null || true; done
  date -u +%FT%TZ > "$RUN_ROOT/stages/FAILED"
  exit "$status"
}
trap fail_closed ERR INT TERM
date -u +%FT%TZ > "$RUN_ROOT/stages/00_started"

# Freeze verification and optional public snapshot acquisition happen before
# any rank starts or model output exists.
"$PYTHON" -B "$BUILDER" verify-source \
  --repo-root "$REPO_ROOT" \
  --manifest "$SOURCE_MANIFEST" \
  --expected-sha256 "$EXPECTED_SOURCE_SHA256" \
  --output "$RUN_ROOT/receipts/source-verification-pre.json"
"$PYTHON" -B "$BUILDER" verify-static \
  --repo-root "$REPO_ROOT" \
  --static "$STATIC_MANIFEST" \
  --expected-sha256 "$EXPECTED_STATIC_SHA256" \
  --output "$RUN_ROOT/receipts/static-verification-pre.json"
"$PYTHON" -B "$MODEL_PREPARER" prepare \
  --model-root "$MODEL_ROOT" \
  --output "$RUN_ROOT/receipts/model-prepare.json"
"$PYTHON" -B "$MODEL_PREPARER" authority \
  --model-root "$MODEL_ROOT" \
  --output "$RUN_ROOT/receipts/model-authority-pre.json"
MODEL_AUTHORITY_SHA256=$(sha256sum "$RUN_ROOT/receipts/model-authority-pre.json" | awk '{print $1}')

"$PYTHON" -B "$BUILDER" gpu-assignment \
  --output "$RUN_ROOT/receipts/gpu-assignment.json"
GPU_ASSIGNMENT_SHA256=$(sha256sum "$RUN_ROOT/receipts/gpu-assignment.json" | awk '{print $1}')
mapfile -t GPU_UUIDS < <("$PYTHON" -B - "$RUN_ROOT/receipts/gpu-assignment.json" <<'PY'
import json
import sys
for row in json.load(open(sys.argv[1], encoding="utf-8"))["rows"]:
    print(row["uuid"])
PY
)
[[ ${#GPU_UUIDS[@]} -eq 8 ]] || { echo "formal cell requires eight H20 UUIDs" >&2; exit 2; }
date -u +%FT%TZ > "$RUN_ROOT/stages/10_preflight_complete"

for rank in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=${GPU_UUIDS[$rank]} \
  "$PYTHON" -B "$RUNNER" \
    --model-root "$MODEL_ROOT" \
    --static "$STATIC_MANIFEST" \
    --source-manifest "$SOURCE_MANIFEST" \
    --model-authority "$RUN_ROOT/receipts/model-authority-pre.json" \
    --gpu-assignment "$RUN_ROOT/receipts/gpu-assignment.json" \
    --expected-static-sha256 "$EXPECTED_STATIC_SHA256" \
    --expected-source-sha256 "$EXPECTED_SOURCE_SHA256" \
    --expected-model-authority-sha256 "$MODEL_AUTHORITY_SHA256" \
    --expected-gpu-assignment-sha256 "$GPU_ASSIGNMENT_SHA256" \
    --rank "$rank" \
    --expected-gpu-uuid "${GPU_UUIDS[$rank]}" \
    --sidecar "$RUN_ROOT/raw/logits/r39-second-model-logits-$rank.bin" \
    --output "$RUN_ROOT/raw/shards/r39-second-model-shard-$rank.json" \
    >"$RUN_ROOT/logs/rank-$rank.log" 2>&1 &
  PIDS+=("$!")
done
for pid in "${PIDS[@]}"; do wait "$pid"; done
PIDS=()
[[ $(find "$RUN_ROOT/raw/shards" -type f -name '*.json' | wc -l) -eq 8 ]] || exit 2
[[ $(find "$RUN_ROOT/raw/logits" -type f -name '*.bin' | wc -l) -eq 8 ]] || exit 2
date -u +%FT%TZ > "$RUN_ROOT/stages/20_raw_complete"

# Close every mutable authority over the scientific outputs.
"$PYTHON" -B "$MODEL_PREPARER" authority \
  --model-root "$MODEL_ROOT" \
  --output "$RUN_ROOT/receipts/model-authority-terminal.json"
cmp "$RUN_ROOT/receipts/model-authority-pre.json" "$RUN_ROOT/receipts/model-authority-terminal.json"
"$PYTHON" -B "$BUILDER" verify-source \
  --repo-root "$REPO_ROOT" \
  --manifest "$SOURCE_MANIFEST" \
  --expected-sha256 "$EXPECTED_SOURCE_SHA256" \
  --output "$RUN_ROOT/receipts/source-verification-terminal.json"
cmp "$RUN_ROOT/receipts/source-verification-pre.json" "$RUN_ROOT/receipts/source-verification-terminal.json"
"$PYTHON" -B "$BUILDER" verify-static \
  --repo-root "$REPO_ROOT" \
  --static "$STATIC_MANIFEST" \
  --expected-sha256 "$EXPECTED_STATIC_SHA256" \
  --output "$RUN_ROOT/receipts/static-verification-terminal.json"
cmp "$RUN_ROOT/receipts/static-verification-pre.json" "$RUN_ROOT/receipts/static-verification-terminal.json"

"$PYTHON" -B "$REPLAY" \
  --package-root "$PACKAGE_ROOT" \
  --run-root "$RUN_ROOT" \
  --output "$RUN_ROOT/r39-second-model-transfer-aggregate.json" \
  >"$RUN_ROOT/logs/aggregate.log" 2>&1
"$PYTHON" -B "$REPLAY" \
  --package-root "$PACKAGE_ROOT" \
  --run-root "$RUN_ROOT" \
  --verify-existing "$RUN_ROOT/r39-second-model-transfer-aggregate.json" \
  >"$RUN_ROOT/receipts/detached-replay.json"
date -u +%FT%TZ > "$RUN_ROOT/stages/30_complete"

"$PYTHON" -B - "$RUN_ROOT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
paths = sorted(
    [path for path in root.rglob("*") if path.is_file() and path.name not in {"artifact-ledger.json", "TERMINAL.sha256", "COMPLETE"}],
    key=lambda path: path.relative_to(root).as_posix(),
)
rows = []
for path in paths:
    raw = path.read_bytes()
    rows.append({
        "path": path.relative_to(root).as_posix(),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    })
payload = json.dumps(
    {"schema_version": "r39-second-model-transfer-artifact-ledger-v1", "rows": rows},
    sort_keys=True,
    separators=(",", ":"),
).encode() + b"\n"
(root / "artifact-ledger.json").write_bytes(payload)
PY

(
  cd "$RUN_ROOT"
  find . -type f ! -name TERMINAL.sha256 ! -name COMPLETE -print0 \
    | sort -z \
    | xargs -0 sha256sum > TERMINAL.sha256
)
sha256sum "$RUN_ROOT/TERMINAL.sha256" | awk '{print $1}' > "$RUN_ROOT/COMPLETE"
trap - ERR INT TERM
