#!/usr/bin/env bash
set -euo pipefail

# Formal eight-rank launcher. It never invokes QS or manages resources. The
# independent official reference finishes before candidate code starts.

export LC_ALL=C
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# This must precede every Transformers import. It disables Hub-loaded wrapper
# kernels as well as avoiding any unregistered network/code-loading route.
export USE_HUB_KERNELS=NO

REPO_ROOT=${REPO_ROOT:?set immutable staged repository root}
PACKAGE_ROOT="$REPO_ROOT/paper_autonomous_multifork_iteration/evidence/r39_falcon_h1_transfer"
ENV_DIR=${ENV_DIR:?set frozen vllm-cu129-v1 Transformers environment}
MODEL_ROOT=${MODEL_ROOT:?set fresh dedicated Falcon model directory}
RUN_ROOT=${RUN_ROOT:?set a fresh R39 Falcon output directory}
EXPECTED_SOURCE_SHA256=${EXPECTED_SOURCE_SHA256:?set source-manifest SHA-256}
EXPECTED_STATIC_SHA256=${EXPECTED_STATIC_SHA256:?set static-preregistration SHA-256}
EXPECTED_FREEZE_SHA256=${EXPECTED_FREEZE_SHA256:?set freeze SHA-256}

PYTHON="$ENV_DIR/bin/python"
SOURCE_MANIFEST="$PACKAGE_ROOT/preregistration/source-manifest.json"
STATIC_MANIFEST="$PACKAGE_ROOT/preregistration/static-preregistration.json"
FREEZE_MANIFEST="$PACKAGE_ROOT/preregistration/freeze.json"
FROZEN_TREE="$PACKAGE_ROOT/preregistration/modelscope-tree.json"
BUILDER="$PACKAGE_ROOT/executed_source/build_r39_falcon_preregistration.py"
MODEL_PREPARER="$PACKAGE_ROOT/executed_source/prepare_r39_falcon_snapshot.py"
REFERENCE="$PACKAGE_ROOT/executed_source/run_r39_falcon_reference.py"
CANDIDATE="$PACKAGE_ROOT/executed_source/run_r39_falcon_candidate.py"
REPLAY="$PACKAGE_ROOT/executed_source/replay_r39_falcon_transfer.py"
TESTS="$PACKAGE_ROOT/executed_source/test_r39_falcon_transfer.py"

[[ ! -e "$RUN_ROOT" ]] || { echo "RUN_ROOT already exists: $RUN_ROOT" >&2; exit 2; }
[[ ! -e "$MODEL_ROOT" ]] || { echo "MODEL_ROOT must be fresh and absent: $MODEL_ROOT" >&2; exit 2; }
for digest in "$EXPECTED_SOURCE_SHA256" "$EXPECTED_STATIC_SHA256" "$EXPECTED_FREEZE_SHA256"; do
  [[ "$digest" =~ ^[0-9a-f]{64}$ ]] || { echo "bad frozen SHA-256" >&2; exit 2; }
done
for item in "$PYTHON" "$SOURCE_MANIFEST" "$STATIC_MANIFEST" "$FREEZE_MANIFEST" "$FROZEN_TREE" "$BUILDER" "$MODEL_PREPARER" "$REFERENCE" "$CANDIDATE" "$REPLAY" "$TESTS"; do
  [[ -e "$item" ]] || { echo "required path absent: $item" >&2; exit 2; }
done

mkdir -p "$RUN_ROOT"/{logs,raw/reference,raw/candidate,raw/logits,receipts,stages,pycache}
cp "$SOURCE_MANIFEST" "$RUN_ROOT/receipts/frozen-source-manifest.json"
cp "$STATIC_MANIFEST" "$RUN_ROOT/receipts/frozen-static-preregistration.json"
cp "$FREEZE_MANIFEST" "$RUN_ROOT/receipts/frozen-freeze.json"
export PYTHONPYCACHEPREFIX="$RUN_ROOT/pycache"
export PYTHONPATH="$PACKAGE_ROOT/executed_source"

PIDS=()
fail_closed() {
  status=$?
  [[ $status -ne 0 ]] || status=1
  trap - ERR INT TERM
  for pid in "${PIDS[@]:-}"; do kill -TERM "$pid" 2>/dev/null || true; done
  for pid in "${PIDS[@]:-}"; do wait "$pid" 2>/dev/null || true; done
  date -u +%FT%TZ > "$RUN_ROOT/stages/FAILED"
  exit "$status"
}
trap fail_closed ERR INT TERM
date -u +%FT%TZ > "$RUN_ROOT/stages/00_started"

# Bootstrap with stdlib-only isolated Python before executing any package
# source. The externally supplied manifest hashes bind this verifier's inputs.
"$PYTHON" -I -B - \
  "$REPO_ROOT" "$SOURCE_MANIFEST" "$STATIC_MANIFEST" "$FREEZE_MANIFEST" \
  "$EXPECTED_SOURCE_SHA256" "$EXPECTED_STATIC_SHA256" "$EXPECTED_FREEZE_SHA256" \
  >"$RUN_ROOT/receipts/bootstrap-source-verification.json" <<'PY'
import hashlib
import json
import stat
import sys
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


repo, source_path, static_path, freeze_path = map(Path, sys.argv[1:5])
expected_source, expected_static, expected_freeze = sys.argv[5:8]
assert sys.flags.isolated
assert sha256_file(source_path) == expected_source
assert sha256_file(static_path) == expected_static
assert sha256_file(freeze_path) == expected_freeze
manifest = json.loads(source_path.read_text(encoding="utf-8"))
assert manifest["file_count"] == len(manifest["files"]) and manifest["files"]
for row in manifest["files"]:
    relative = Path(row["path"])
    assert not relative.is_absolute() and ".." not in relative.parts
    path = repo / relative
    status = path.stat()
    assert stat.S_ISREG(status.st_mode) and not path.is_symlink()
    assert status.st_size == row["bytes"] and sha256_file(path) == row["sha256"]
freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
assert freeze["source_manifest_sha256"] == expected_source
assert freeze["static_manifest_sha256"] == expected_static
receipt = {
    "schema_version": "r39-falcon-h1-bootstrap-source-verification-v1",
    "source_manifest_sha256": expected_source,
    "static_manifest_sha256": expected_static,
    "freeze_sha256": expected_freeze,
    "verified_file_count": len(manifest["files"]),
    "verified": True,
}
print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
PY

"$PYTHON" -B "$TESTS" >"$RUN_ROOT/logs/package-tests.log" 2>&1
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
"$PYTHON" -B "$BUILDER" verify-freeze \
  --package-root "$PACKAGE_ROOT" \
  --freeze "$FREEZE_MANIFEST" \
  --expected-sha256 "$EXPECTED_FREEZE_SHA256" \
  --output "$RUN_ROOT/receipts/freeze-verification-pre.json"
"$PYTHON" -B "$MODEL_PREPARER" prepare \
  --model-root "$MODEL_ROOT" \
  --frozen-tree "$FROZEN_TREE" \
  --output "$RUN_ROOT/receipts/model-prepare.json"
"$PYTHON" -B "$MODEL_PREPARER" authority \
  --model-root "$MODEL_ROOT" \
  --frozen-tree "$FROZEN_TREE" \
  --output "$RUN_ROOT/receipts/model-authority-pre.json"
MODEL_AUTHORITY_SHA256=$(sha256sum "$RUN_ROOT/receipts/model-authority-pre.json" | awk '{print $1}')

"$PYTHON" -B "$BUILDER" gpu-assignment --output "$RUN_ROOT/receipts/gpu-assignment.json"
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

# Independent official reference: no candidate module is on its import path.
for rank in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=${GPU_UUIDS[$rank]} \
  PYTHONPATH= \
  "$PYTHON" -I -B "$REFERENCE" \
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
    --sidecar "$RUN_ROOT/raw/logits/r39-falcon-reference-logits-$rank.bin" \
    --output "$RUN_ROOT/raw/reference/r39-falcon-reference-$rank.json" \
    >"$RUN_ROOT/logs/reference-rank-$rank.log" 2>&1 &
  PIDS+=("$!")
done
for pid in "${PIDS[@]}"; do wait "$pid"; done
PIDS=()
[[ $(find "$RUN_ROOT/raw/reference" -type f -name '*.json' | wc -l) -eq 8 ]] || exit 2
[[ $(find "$RUN_ROOT/raw/logits" -type f -name 'r39-falcon-reference-*.bin' | wc -l) -eq 8 ]] || exit 2
date -u +%FT%TZ > "$RUN_ROOT/stages/20_reference_complete"

# Candidate starts only after all independent reference processes exit.
for rank in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=${GPU_UUIDS[$rank]} \
  "$PYTHON" -B "$CANDIDATE" \
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
    --sidecar "$RUN_ROOT/raw/logits/r39-falcon-candidate-logits-$rank.bin" \
    --output "$RUN_ROOT/raw/candidate/r39-falcon-candidate-$rank.json" \
    >"$RUN_ROOT/logs/candidate-rank-$rank.log" 2>&1 &
  PIDS+=("$!")
done
for pid in "${PIDS[@]}"; do wait "$pid"; done
PIDS=()
[[ $(find "$RUN_ROOT/raw/candidate" -type f -name '*.json' | wc -l) -eq 8 ]] || exit 2
[[ $(find "$RUN_ROOT/raw/logits" -type f -name 'r39-falcon-candidate-*.bin' | wc -l) -eq 8 ]] || exit 2
date -u +%FT%TZ > "$RUN_ROOT/stages/30_candidate_complete"

"$PYTHON" -B "$MODEL_PREPARER" authority \
  --model-root "$MODEL_ROOT" \
  --frozen-tree "$FROZEN_TREE" \
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
"$PYTHON" -B "$BUILDER" verify-freeze \
  --package-root "$PACKAGE_ROOT" \
  --freeze "$FREEZE_MANIFEST" \
  --expected-sha256 "$EXPECTED_FREEZE_SHA256" \
  --output "$RUN_ROOT/receipts/freeze-verification-terminal.json"
cmp "$RUN_ROOT/receipts/freeze-verification-pre.json" "$RUN_ROOT/receipts/freeze-verification-terminal.json"

"$PYTHON" -B "$REPLAY" \
  --package-root "$PACKAGE_ROOT" \
  --run-root "$RUN_ROOT" \
  --output "$RUN_ROOT/r39-falcon-h1-transfer-aggregate.json" \
  >"$RUN_ROOT/logs/aggregate.log" 2>&1
"$PYTHON" -B "$REPLAY" \
  --package-root "$PACKAGE_ROOT" \
  --run-root "$RUN_ROOT" \
  --verify-existing "$RUN_ROOT/r39-falcon-h1-transfer-aggregate.json" \
  >"$RUN_ROOT/receipts/detached-replay.json"
date -u +%FT%TZ > "$RUN_ROOT/stages/40_replay_complete"

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
    rows.append({"path": path.relative_to(root).as_posix(), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
payload = json.dumps(
    {"schema_version": "r39-falcon-h1-transfer-artifact-ledger-v1", "rows": rows},
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
