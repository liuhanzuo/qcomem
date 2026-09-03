#!/usr/bin/env bash
set -euo pipefail

# Ledger ordering is protocol state, not a host preference.
export LC_ALL=C

# This is a source-controlled second gate.  It must be changed only in the
# same reviewed release that changes runner.GPU_LOOP_IMPLEMENTED to true.
FORMAL_PIPELINE_RELEASED=true
FORMAL_MODEL_ID=Qwen/Qwen3.5-35B-A3B
FORMAL_MODEL_REVISION=59d61f3ce65a6d9863b86d2e96597125219dc754

CODE_DIR=${CODE_DIR:?set CODE_DIR to the immutable release snapshot}
MODEL_DIR=${MODEL_DIR:?set MODEL_DIR}
MODEL_ARTIFACT_LEDGER_FILE=${MODEL_ARTIFACT_LEDGER_FILE:?set a path-independent model artifact ledger}
MODEL_WEIGHT_LEDGER_FILE=${MODEL_WEIGHT_LEDGER_FILE:?set a path-independent 14-shard weight ledger}
PG19_DATA=${PG19_DATA:?set PG19_DATA}
PG19_MANIFEST=${PG19_MANIFEST:?set PG19_MANIFEST}
PG19_INPUT_MANIFEST=${PG19_INPUT_MANIFEST:?set the frozen RR2 main input manifest}
PRIOR_CAPACITY_MANIFEST=${PRIOR_CAPACITY_MANIFEST:?set the exact prior capacity protocol manifest}
FROZEN_QUERY_BANKS_INPUT=${FROZEN_QUERY_BANKS_INPUT:?set the RR2 query-bank sidecar}
PROTOCOL_SOURCE_MANIFEST=${PROTOCOL_SOURCE_MANIFEST:?set PROTOCOL_SOURCE_MANIFEST}
ORACLE_SELECTION_INPUT=${ORACLE_SELECTION_INPUT:?set the pre-output oracle selection plan}
PRIOR_FP32_CONTEXT_MANIFEST=${PRIOR_FP32_CONTEXT_MANIFEST:?set the frozen prior FP32 context manifest}
REVIEW_RESPONSE_PLAN=${REVIEW_RESPONSE_PLAN:?set the reviewer-response experiment plan}
RUN_DIR=${RUN_DIR:?set a new RUN_DIR outside CODE_DIR}
ENV_DIR=${ENV_DIR:?set the frozen Python environment}
MODEL_ID=${MODEL_ID:?set MODEL_ID}
MODEL_REVISION=${MODEL_REVISION:?set the full 40-character model commit}

EXPECTED_CODE_LEDGER_SHA256=${EXPECTED_CODE_LEDGER_SHA256:?set EXPECTED_CODE_LEDGER_SHA256}
EXPECTED_MODEL_MANIFEST_SHA256=${EXPECTED_MODEL_MANIFEST_SHA256:?set EXPECTED_MODEL_MANIFEST_SHA256}
EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256=${EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256:?set EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256}
EXPECTED_MODEL_WEIGHT_LEDGER_SHA256=${EXPECTED_MODEL_WEIGHT_LEDGER_SHA256:?set EXPECTED_MODEL_WEIGHT_LEDGER_SHA256}
EXPECTED_PG19_SHA256=${EXPECTED_PG19_SHA256:?set EXPECTED_PG19_SHA256}
EXPECTED_PG19_MANIFEST_SHA256=${EXPECTED_PG19_MANIFEST_SHA256:?set EXPECTED_PG19_MANIFEST_SHA256}
EXPECTED_PG19_WINDOWS_SHA256=${EXPECTED_PG19_WINDOWS_SHA256:?set the RR2 algorithm windows SHA}
EXPECTED_PG19_INPUT_MANIFEST_SHA256=${EXPECTED_PG19_INPUT_MANIFEST_SHA256:?set the RR2 main raw-byte SHA}
EXPECTED_PRIOR_CAPACITY_MANIFEST_SHA256=${EXPECTED_PRIOR_CAPACITY_MANIFEST_SHA256:?set the prior capacity raw-byte SHA}
EXPECTED_FROZEN_QUERY_BANKS_INPUT_SHA256=${EXPECTED_FROZEN_QUERY_BANKS_INPUT_SHA256:?set query-bank sidecar raw-byte SHA}
EXPECTED_ORACLE_SELECTION_INPUT_SHA256=${EXPECTED_ORACLE_SELECTION_INPUT_SHA256:?set oracle sidecar raw-byte SHA}
EXPECTED_PRIOR_FP32_CONTEXT_MANIFEST_SHA256=${EXPECTED_PRIOR_FP32_CONTEXT_MANIFEST_SHA256:?set prior FP32 context raw-byte SHA}
EXPECTED_REVIEW_RESPONSE_PLAN_SHA256=${EXPECTED_REVIEW_RESPONSE_PLAN_SHA256:?set review-response plan raw-byte SHA}
EXPECTED_PROTOCOL_SOURCE_MANIFEST_SHA256=${EXPECTED_PROTOCOL_SOURCE_MANIFEST_SHA256:?set EXPECTED_PROTOCOL_SOURCE_MANIFEST_SHA256}
EXPECTED_RELEASE_MANIFEST_SHA256=${EXPECTED_RELEASE_MANIFEST_SHA256:?set EXPECTED_RELEASE_MANIFEST_SHA256}
EXPECTED_ORACLE_SELECTION_PLAN_SHA256=${EXPECTED_ORACLE_SELECTION_PLAN_SHA256:?set EXPECTED_ORACLE_SELECTION_PLAN_SHA256}
EXPECTED_FROZEN_QUERY_BANKS_SHA256=${EXPECTED_FROZEN_QUERY_BANKS_SHA256:?set EXPECTED_FROZEN_QUERY_BANKS_SHA256}
EXPECTED_RUNNER_SHA256=${EXPECTED_RUNNER_SHA256:?set the independently reviewed final runner SHA}

RUNNER="$CODE_DIR/run_qcomem_qwen35_forkaudit_review_revision.py"
MANIFEST_BUILDER="$CODE_DIR/build_qcomem_qwen35_forkaudit_review_manifest.py"
RR2_INPUT_BUILDER="$CODE_DIR/build_qcomem_forkaudit_rr2_input_manifest.py"
LAUNCHER="$CODE_DIR/launch_qcomem_qwen35_forkaudit_review_revision_8gpu.sh"
PROTOCOL_DOC="$CODE_DIR/FORKAUDIT_REVIEW_REVISION_PROTOCOL_ZH.md"
PYTHON="$ENV_DIR/bin/python"
PRIVATE_MODEL_VIEW="$RUN_DIR/model-view"

EXPECTED_MUTANT_ASSIGNMENT="0:M1,M9;1:M2;2:M3;3:M4;4:M5;5:M6;6:M7;7:M8"

if [[ -e "$RUN_DIR" ]]; then
  echo "RUN_DIR must be absent before release: $RUN_DIR" >&2
  exit 2
fi
if [[ "$MODEL_ID" != "$FORMAL_MODEL_ID" ]]; then
  echo "formal model ID drift" >&2
  exit 2
fi
if [[ "$MODEL_REVISION" != "$FORMAL_MODEL_REVISION" || ! "$MODEL_REVISION" =~ ^[0-9a-f]{40}$ ]]; then
  echo "MODEL_REVISION must equal the frozen full 40-character commit" >&2
  exit 2
fi

for VALUE in \
  "$EXPECTED_CODE_LEDGER_SHA256" "$EXPECTED_MODEL_MANIFEST_SHA256" \
  "$EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256" \
  "$EXPECTED_MODEL_WEIGHT_LEDGER_SHA256" "$EXPECTED_PG19_SHA256" \
  "$EXPECTED_PG19_MANIFEST_SHA256" "$EXPECTED_PG19_WINDOWS_SHA256" \
  "$EXPECTED_PG19_INPUT_MANIFEST_SHA256" \
  "$EXPECTED_PRIOR_CAPACITY_MANIFEST_SHA256" \
  "$EXPECTED_FROZEN_QUERY_BANKS_INPUT_SHA256" \
  "$EXPECTED_ORACLE_SELECTION_INPUT_SHA256" \
  "$EXPECTED_PRIOR_FP32_CONTEXT_MANIFEST_SHA256" \
  "$EXPECTED_REVIEW_RESPONSE_PLAN_SHA256" \
  "$EXPECTED_PROTOCOL_SOURCE_MANIFEST_SHA256" \
  "$EXPECTED_RELEASE_MANIFEST_SHA256" \
  "$EXPECTED_ORACLE_SELECTION_PLAN_SHA256" \
  "$EXPECTED_FROZEN_QUERY_BANKS_SHA256" "$EXPECTED_RUNNER_SHA256"; do
  if [[ ! "$VALUE" =~ ^[0-9a-f]{64}$ ]]; then
    echo "every frozen digest must be one lowercase SHA-256" >&2
    exit 2
  fi
done
if [[ "$EXPECTED_PG19_WINDOWS_SHA256" != "39bc36bb2eb04d51122e66caaebfa72367c02b43b073f072a2da240ed068c166" ]]; then
  echo "RR2 algorithm windows SHA drift" >&2
  exit 2
fi
if [[ "$EXPECTED_PRIOR_CAPACITY_MANIFEST_SHA256" != "975bc6a12f43447024b889889d4156ca71c2f89b68de6157ac609b4a9687e9c0" ]]; then
  echo "prior capacity manifest SHA drift" >&2
  exit 2
fi
if [[ "$EXPECTED_PRIOR_FP32_CONTEXT_MANIFEST_SHA256" != "fa64f663bb74a190a0a5c0898fda2a55528171c77a91af2b1321c24a5f310a1d" ]]; then
  echo "prior FP32 context manifest SHA drift" >&2
  exit 2
fi
if [[ "$EXPECTED_REVIEW_RESPONSE_PLAN_SHA256" != "e2be05e198d6c86276f229d4e862c3579c65e311d07c219e9e9c792a390cbfbb" ]]; then
  echo "review-response experiment plan SHA drift" >&2
  exit 2
fi

case "${PG19_DATA}:${PG19_MANIFEST}:${PG19_INPUT_MANIFEST}:${FROZEN_QUERY_BANKS_INPUT}:${ORACLE_SELECTION_INPUT}" in
  *[Ll][Oo][Nn][Gg][Bb][Ee][Nn][Cc][Hh]*|*[Tt][Ee][Ss][Tt]-[Vv]2*|*[Tt][Ee][Ss][Tt]_[Vv]2*)
    echo "ForkAudit accepts PG19 train-only inputs; LongBench/test-v2 is forbidden" >&2
    exit 2
    ;;
esac

CODE_REAL=$(realpath "$CODE_DIR")
RUN_REAL=$(realpath -m "$RUN_DIR")
case "$RUN_REAL/" in
  "$CODE_REAL/"*)
    echo "RUN_DIR and PYTHONPYCACHEPREFIX must be outside the immutable code snapshot" >&2
    exit 2
    ;;
esac

mkdir -p \
  "$RUN_DIR/logs" "$RUN_DIR/stages" "$RUN_DIR/preregistration" \
  "$RUN_DIR/raw/shards" "$RUN_DIR/receipts" "$RUN_DIR/pycache"
export PYTHONPYCACHEPREFIX="$RUN_DIR/pycache"

CURRENT_PHASE=preflight
PIDS=()
LEASE_KEEPER_PID=""
LEASE_CONTROL_FIFO=""
LEASE_EVENT_FIFO=""
terminate_children() {
  local pid
  for pid in "${PIDS[@]:-}"; do
    kill -TERM "$pid" 2>/dev/null || true
  done
  for pid in "${PIDS[@]:-}"; do
    wait "$pid" 2>/dev/null || true
  done
  PIDS=()
  if [[ -n "$LEASE_KEEPER_PID" ]]; then
    kill -TERM "$LEASE_KEEPER_PID" 2>/dev/null || true
    wait "$LEASE_KEEPER_PID" 2>/dev/null || true
    LEASE_KEEPER_PID=""
  fi
  if [[ -n "$LEASE_CONTROL_FIFO" && -p "$LEASE_CONTROL_FIFO" ]]; then
    rm -f "$LEASE_CONTROL_FIFO"
  fi
  if [[ -n "$LEASE_EVENT_FIFO" && -p "$LEASE_EVENT_FIFO" ]]; then
    rm -f "$LEASE_EVENT_FIFO"
  fi
}
record_failure() {
  local status=$1
  trap - ERR INT TERM
  terminate_children
  date -u +%FT%TZ > "$RUN_DIR/stages/FAILED"
  printf '%s\n' "$CURRENT_PHASE" > "$RUN_DIR/stages/FAILED_PHASE"
  date -u +%FT%TZ > "$RUN_DIR/stages/FAILED_${CURRENT_PHASE}"
  exit "$status"
}
on_error() {
  local status=$?
  record_failure "$status"
}
trap on_error ERR
trap 'record_failure 130' INT
trap 'record_failure 143' TERM
date -u +%FT%TZ > "$RUN_DIR/stages/00_start"

fail_stage() {
  echo "$1" >&2
  return 2
}
audit_code_snapshot() {
  local code_root=$1 list_output=$2 ledger_output=$3
  "$PYTHON" -I - "$code_root" "$list_output" "$ledger_output" <<'PY_CODE_SNAPSHOT_AUDIT'
import hashlib
import os
import stat
import sys
from pathlib import Path

root_arg = Path(sys.argv[1])
list_output = Path(sys.argv[2])
ledger_output = Path(sys.argv[3])
write_mask = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH


def reject(message):
    raise SystemExit(f"ForkAudit code snapshot rejected: {message}")


try:
    root_lstat = root_arg.lstat()
except OSError as exc:
    reject(f"CODE_DIR cannot be inspected: {exc}")
if stat.S_ISLNK(root_lstat.st_mode):
    reject("CODE_DIR itself is a symbolic link")
if not stat.S_ISDIR(root_lstat.st_mode):
    reject("CODE_DIR is not a directory")
if root_lstat.st_mode & write_mask:
    reject("CODE_DIR root is writable")
root = root_arg.resolve(strict=True)

regular_files = []


def walk(directory, relative_parent):
    try:
        with os.scandir(directory) as iterator:
            entries = sorted(iterator, key=lambda entry: os.fsencode(entry.name))
    except OSError as exc:
        reject(f"cannot scan {relative_parent.as_posix() or '.'}: {exc}")
    for entry in entries:
        relative = relative_parent / entry.name
        relative_text = relative.as_posix()
        if any(character in relative_text for character in ("\\", "\n", "\r")):
            reject(f"unsupported code path spelling: {relative_text!r}")
        try:
            metadata = entry.stat(follow_symlinks=False)
        except OSError as exc:
            reject(f"cannot stat {relative_text}: {exc}")
        mode = metadata.st_mode
        if stat.S_ISLNK(mode):
            reject(f"symbolic link present: {relative_text}")
        if mode & write_mask:
            reject(f"writable code entry present: {relative_text}")
        if "__pycache__" in relative.parts:
            reject(f"Python bytecode cache present: {relative_text}")
        if stat.S_ISDIR(mode):
            walk(Path(entry.path), relative)
        elif stat.S_ISREG(mode):
            if relative.suffix in {".pyc", ".pyo"}:
                reject(f"Python bytecode file present: {relative_text}")
            regular_files.append((relative_text, Path(entry.path), metadata))
        else:
            reject(f"non-regular code entry present: {relative_text}")


walk(root, Path())
# Every regular file in the curated release snapshot is part of the closure.
# This includes native Python extensions and non-Python import/config inputs;
# extension loaders must never be able to select an unreceipted same-stem .so.
selected = list(regular_files)
selected.sort(key=lambda row: os.fsencode(row[0]))
required = {
    "run_qcomem_qwen35_forkaudit_review_revision.py",
    "build_qcomem_qwen35_forkaudit_review_manifest.py",
    "build_qcomem_forkaudit_rr2_input_manifest.py",
    "launch_qcomem_qwen35_forkaudit_review_revision_8gpu.sh",
    "FORKAUDIT_REVIEW_REVISION_PROTOCOL_ZH.md",
}
selected_names = {row[0] for row in selected}
missing = sorted(required - selected_names)
if missing:
    reject(f"required release files missing: {missing}")
if len(selected) < 15:
    reject("ForkAudit recursive code snapshot is unexpectedly incomplete")

ledger_lines = []
list_bytes = bytearray()
for relative_text, path, scanned in selected:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            opened = os.fstat(handle.fileno())
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_dev != scanned.st_dev
                or opened.st_ino != scanned.st_ino
                or opened.st_mode & write_mask
            ):
                reject(f"code entry changed during audit: {relative_text}")
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
            closed = os.fstat(handle.fileno())
    except OSError as exc:
        reject(f"cannot hash {relative_text}: {exc}")
    if (
        closed.st_size != opened.st_size
        or closed.st_mtime_ns != opened.st_mtime_ns
        or closed.st_mode != opened.st_mode
    ):
        reject(f"code entry changed while hashing: {relative_text}")
    ledger_lines.append(f"{digest.hexdigest()}  ./{relative_text}\n")
    list_bytes.extend(f"./{relative_text}".encode("utf-8"))
    list_bytes.append(0)


def atomic_write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


atomic_write(list_output, bytes(list_bytes))
atomic_write(ledger_output, "".join(ledger_lines).encode("utf-8"))
PY_CODE_SNAPSHOT_AUDIT
}
verify_sha() {
  local path=$1 expected=$2 label=$3 actual
  actual=$(sha256sum "$path" | awk '{print $1}')
  if [[ "$actual" != "$expected" ]]; then
    echo "$label SHA-256 mismatch: expected=$expected actual=$actual" >&2
    return 2
  fi
}
json_digest() {
  timeout --signal=TERM --kill-after=15s 120s \
    env PYTHONPATH="$CODE_DIR" "$PYTHON" "$MANIFEST_BUILDER" \
    digest-json --input "$1"
}

for INPUT in \
  "$RUNNER" "$MANIFEST_BUILDER" "$RR2_INPUT_BUILDER" "$LAUNCHER" "$PROTOCOL_DOC" \
  "$MODEL_ARTIFACT_LEDGER_FILE" "$MODEL_WEIGHT_LEDGER_FILE" \
  "$PG19_DATA" "$PG19_MANIFEST" "$PG19_INPUT_MANIFEST" \
  "$PRIOR_CAPACITY_MANIFEST" "$FROZEN_QUERY_BANKS_INPUT" \
  "$PROTOCOL_SOURCE_MANIFEST" "$ORACLE_SELECTION_INPUT" \
  "$PRIOR_FP32_CONTEXT_MANIFEST" "$REVIEW_RESPONSE_PLAN"; do
  test -s "$INPUT"
done
test -x "$PYTHON"
# The isolated audit recursively rejects symlinks, bytecode, writable
# files/directories, and special files.  Its selected closure is every regular
# file, with relative C-byte-sorted names so relocation cannot change it.
audit_code_snapshot \
  "$CODE_DIR" \
  "$RUN_DIR/preregistration/code-files.nul" \
  "$RUN_DIR/preregistration/code.sha256"
CODE_FILES=()
while IFS= read -r -d '' CODE_FILE; do
  case "$CODE_FILE" in
    *.py) CODE_FILES+=("$CODE_FILE") ;;
  esac
done < "$RUN_DIR/preregistration/code-files.nul"
if ((${#CODE_FILES[@]} < 12)); then
  fail_stage "recursive code snapshot contains too few Python sources"
fi
verify_sha "$RUN_DIR/preregistration/code.sha256" \
  "$EXPECTED_CODE_LEDGER_SHA256" code-ledger
verify_sha "$MODEL_ARTIFACT_LEDGER_FILE" \
  "$EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256" model-artifact-ledger
verify_sha "$MODEL_WEIGHT_LEDGER_FILE" \
  "$EXPECTED_MODEL_WEIGHT_LEDGER_SHA256" model-weight-ledger
verify_sha "$RUNNER" "$EXPECTED_RUNNER_SHA256" independently-reviewed-runner
verify_sha "$PG19_DATA" "$EXPECTED_PG19_SHA256" pg19-train
verify_sha "$PG19_MANIFEST" "$EXPECTED_PG19_MANIFEST_SHA256" pg19-manifest
verify_sha "$PG19_INPUT_MANIFEST" \
  "$EXPECTED_PG19_INPUT_MANIFEST_SHA256" rr2-pg19-input-manifest
verify_sha "$PRIOR_CAPACITY_MANIFEST" \
  "$EXPECTED_PRIOR_CAPACITY_MANIFEST_SHA256" prior-capacity-manifest
verify_sha "$FROZEN_QUERY_BANKS_INPUT" \
  "$EXPECTED_FROZEN_QUERY_BANKS_INPUT_SHA256" rr2-query-bank-sidecar
verify_sha "$ORACLE_SELECTION_INPUT" \
  "$EXPECTED_ORACLE_SELECTION_INPUT_SHA256" rr2-oracle-sidecar
verify_sha "$PRIOR_FP32_CONTEXT_MANIFEST" \
  "$EXPECTED_PRIOR_FP32_CONTEXT_MANIFEST_SHA256" prior-fp32-context
verify_sha "$REVIEW_RESPONSE_PLAN" \
  "$EXPECTED_REVIEW_RESPONSE_PLAN_SHA256" review-response-plan
verify_sha "$PROTOCOL_SOURCE_MANIFEST" \
  "$EXPECTED_PROTOCOL_SOURCE_MANIFEST_SHA256" protocol-source-manifest

timeout --signal=TERM --kill-after=30s 300s bash -c \
  'cd "$1" && sha256sum -c "$2"' _ "$CODE_DIR" \
  "$RUN_DIR/preregistration/code.sha256" \
  > "$RUN_DIR/logs/code-integrity.log"
timeout --signal=TERM --kill-after=30s 900s bash -c \
  'cd "$1" && sha256sum -c "$2"' _ "$MODEL_DIR" \
  "$(realpath "$MODEL_ARTIFACT_LEDGER_FILE")" \
  > "$RUN_DIR/logs/model-artifact-integrity.log"
timeout --signal=TERM --kill-after=60s 7200s bash -c \
  'cd "$1" && sha256sum -c "$2"' _ "$MODEL_DIR" \
  "$(realpath "$MODEL_WEIGHT_LEDGER_FILE")" \
  > "$RUN_DIR/logs/model-weight-integrity.log"

bash -n "$LAUNCHER"
timeout --signal=TERM --kill-after=30s 300s \
  "$PYTHON" -m py_compile "${CODE_FILES[@]/#/$CODE_DIR/}"

CURRENT_PHASE=focused_tests
timeout --signal=TERM --kill-after=60s 1800s \
  env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  PYTHONPATH="$CODE_DIR" "$PYTHON" -m unittest -v \
  test_qcomem_forkaudit_storage_witness \
  test_qcomem_forkaudit_oracle \
  test_qcomem_forkaudit_mutants \
  test_qcomem_vllm_paged_multifork_resident \
  test_qcomem_qwen35_vllm_paged_integration \
  test_build_qcomem_forkaudit_rr2_input_manifest \
  test_build_qcomem_forkaudit_fp32_calibration_manifest \
  test_run_qcomem_qwen35_forkaudit_review_revision \
  test_launch_qcomem_qwen35_forkaudit_review_revision \
  > "$RUN_DIR/logs/focused-tests.log" 2>&1
if grep -Eq 'skipped=|\.\.\. skipped|SKIP' "$RUN_DIR/logs/focused-tests.log"; then
  fail_stage "focused test suite contained a skip; real Transformers coverage is mandatory"
fi
grep -Eq \
  '^test_real_tf514_qwen_call_consumes_and_advances_position_ids .* \.\.\. ok$' \
  "$RUN_DIR/logs/focused-tests.log"
grep -Eq '^OK$' "$RUN_DIR/logs/focused-tests.log"
date -u +%FT%TZ > "$RUN_DIR/stages/01_focused_tests_ok"

CURRENT_PHASE=preregistration
# First clean source replay: reconstruct main/banks/oracle directly from the
# exact PG19 bytes, exact prior manifest, and local-only frozen tokenizer.
timeout --signal=TERM --kill-after=30s 1800s \
  env PYTHONPATH="$CODE_DIR" "$PYTHON" "$RR2_INPUT_BUILDER" \
  --pg19-data "$PG19_DATA" \
  --pg19-manifest "$PG19_MANIFEST" \
  --prior-capacity-manifest "$PRIOR_CAPACITY_MANIFEST" \
  --model-dir "$MODEL_DIR" \
  --model-id "$MODEL_ID" \
  --model-revision "$MODEL_REVISION" \
  --output "$RUN_DIR/preregistration/source-rebuilt-rr2-input-manifest.json" \
  --frozen-query-banks-output \
    "$RUN_DIR/preregistration/source-rebuilt-frozen-query-banks.json" \
  --oracle-selection-output \
    "$RUN_DIR/preregistration/source-rebuilt-oracle-selection-plan.json" \
  > "$RUN_DIR/logs/rr2-source-rebuild.json"
if ! cmp -s "$RUN_DIR/preregistration/source-rebuilt-rr2-input-manifest.json" \
  "$PG19_INPUT_MANIFEST"; then
  fail_stage "RR2 main manifest does not byte-replay from PG19 and tokenizer"
fi
if ! cmp -s "$RUN_DIR/preregistration/source-rebuilt-frozen-query-banks.json" \
  "$FROZEN_QUERY_BANKS_INPUT"; then
  fail_stage "RR2 query banks do not byte-replay from PG19 and tokenizer"
fi
if ! cmp -s "$RUN_DIR/preregistration/source-rebuilt-oracle-selection-plan.json" \
  "$ORACLE_SELECTION_INPUT"; then
  fail_stage "RR2 oracle plan does not byte-replay from PG19 and tokenizer"
fi

# Second independent source replay happens inside the release builder before
# it emits the final static inputs and frozen receipts.
timeout --signal=TERM --kill-after=30s 600s \
  env PYTHONPATH="$CODE_DIR" "$PYTHON" "$MANIFEST_BUILDER" preregister \
  --output "$RUN_DIR/preregistration/release-manifest.json" \
  --frozen-identity-output "$RUN_DIR/preregistration/frozen-identity.json" \
  --frozen-query-banks-output "$RUN_DIR/preregistration/frozen-query-banks.json" \
  --oracle-selection-output "$RUN_DIR/preregistration/oracle-selection-plan.json" \
  --model-id "$MODEL_ID" --model-revision "$MODEL_REVISION" \
  --model-dir "$MODEL_DIR" \
  --code-ledger "$RUN_DIR/preregistration/code.sha256" \
  --model-artifact-ledger "$MODEL_ARTIFACT_LEDGER_FILE" \
  --model-weight-ledger "$MODEL_WEIGHT_LEDGER_FILE" \
  --pg19-data "$PG19_DATA" --pg19-manifest "$PG19_MANIFEST" \
  --pg19-input-manifest "$PG19_INPUT_MANIFEST" \
  --expected-pg19-input-manifest-sha256 \
    "$EXPECTED_PG19_INPUT_MANIFEST_SHA256" \
  --prior-capacity-manifest "$PRIOR_CAPACITY_MANIFEST" \
  --frozen-query-banks-input "$FROZEN_QUERY_BANKS_INPUT" \
  --expected-frozen-query-banks-input-sha256 \
    "$EXPECTED_FROZEN_QUERY_BANKS_INPUT_SHA256" \
  --protocol-source-manifest "$PROTOCOL_SOURCE_MANIFEST" \
  --oracle-selection-input "$ORACLE_SELECTION_INPUT" \
  --expected-oracle-selection-input-sha256 \
    "$EXPECTED_ORACLE_SELECTION_INPUT_SHA256" \
  --prior-fp32-context-manifest "$PRIOR_FP32_CONTEXT_MANIFEST" \
  --expected-prior-fp32-context-manifest-sha256 \
    "$EXPECTED_PRIOR_FP32_CONTEXT_MANIFEST_SHA256" \
  --review-response-plan "$REVIEW_RESPONSE_PLAN" \
  --expected-review-response-plan-sha256 \
    "$EXPECTED_REVIEW_RESPONSE_PLAN_SHA256" \
  > "$RUN_DIR/logs/manifest-build.json"

RELEASE_MANIFEST_SHA256=$(json_digest "$RUN_DIR/preregistration/release-manifest.json")
ORACLE_SELECTION_PLAN_SHA256=$(json_digest "$RUN_DIR/preregistration/oracle-selection-plan.json")
FROZEN_QUERY_BANKS_SHA256=$(json_digest "$RUN_DIR/preregistration/frozen-query-banks.json")
if [[ "$RELEASE_MANIFEST_SHA256" != "$EXPECTED_RELEASE_MANIFEST_SHA256" ]]; then
  fail_stage "release manifest differs from the externally pinned preregistration"
fi
if [[ "$ORACLE_SELECTION_PLAN_SHA256" != "$EXPECTED_ORACLE_SELECTION_PLAN_SHA256" ]]; then
  fail_stage "oracle sample selection was not pinned before candidate outputs"
fi
if [[ "$FROZEN_QUERY_BANKS_SHA256" != "$EXPECTED_FROZEN_QUERY_BANKS_SHA256" ]]; then
  fail_stage "query banks differ from the externally pinned preregistration"
fi
if ! cmp -s "$RUN_DIR/preregistration/frozen-query-banks.json" \
  "$FROZEN_QUERY_BANKS_INPUT"; then
  fail_stage "generated query-bank sidecar differs bytewise from the frozen input"
fi
if ! cmp -s "$RUN_DIR/preregistration/oracle-selection-plan.json" \
  "$ORACLE_SELECTION_INPUT"; then
  fail_stage "generated oracle sidecar differs bytewise from the frozen input"
fi

timeout --signal=TERM --kill-after=30s 300s \
  env PYTHONPATH="$CODE_DIR" "$PYTHON" "$RUNNER" \
  --stage static \
  --frozen-identity "$RUN_DIR/preregistration/frozen-identity.json" \
  --oracle-selection-plan "$RUN_DIR/preregistration/oracle-selection-plan.json" \
  --frozen-query-banks "$RUN_DIR/preregistration/frozen-query-banks.json" \
  --rr2-input-manifest "$PG19_INPUT_MANIFEST" \
  --expected-rr2-input-manifest-sha256 \
    "$EXPECTED_PG19_INPUT_MANIFEST_SHA256" \
  --prior-fp32-context-manifest "$PRIOR_FP32_CONTEXT_MANIFEST" \
  --expected-prior-fp32-context-manifest-sha256 \
    "$EXPECTED_PRIOR_FP32_CONTEXT_MANIFEST_SHA256" \
  --review-experiment-plan "$REVIEW_RESPONSE_PLAN" \
  --expected-review-experiment-plan-sha256 \
    "$EXPECTED_REVIEW_RESPONSE_PLAN_SHA256" \
  --output "$RUN_DIR/preregistration/static-artifact.json" \
  > "$RUN_DIR/logs/static.log" 2>&1
STATIC_ARTIFACT_SHA256=$(json_digest "$RUN_DIR/preregistration/static-artifact.json")

timeout --signal=TERM --kill-after=15s 120s \
  env PYTHONPATH="$CODE_DIR" "$PYTHON" - \
  "$RUN_DIR/preregistration/release-manifest.json" \
  "$RUN_DIR/preregistration/static-artifact.json" \
  "$EXPECTED_MUTANT_ASSIGNMENT" \
  "$EXPECTED_MODEL_MANIFEST_SHA256" \
  "$EXPECTED_PG19_INPUT_MANIFEST_SHA256" \
  "$EXPECTED_PRIOR_FP32_CONTEXT_MANIFEST_SHA256" \
  "$EXPECTED_REVIEW_RESPONSE_PLAN_SHA256" <<'PY' \
  > "$RUN_DIR/logs/static-final-audit.json"
import json,sys
release=json.load(open(sys.argv[1],encoding="utf-8"))
static=json.load(open(sys.argv[2],encoding="utf-8"))
assignment=";".join(
    f'{row["rank"]}:{",".join(row["mutant_ids"])}'
    for row in release["rank_assignments"]
)
assert assignment==sys.argv[3]
assert release["frozen_identity"]["model_manifest_sha256"]==sys.argv[4]
assert release["frozen_identity"]["pg19_input_manifest_sha256"]==sys.argv[5]
assert release["frozen_identity"]["prior_fp32_context_manifest_sha256"]==sys.argv[6]
assert release["frozen_identity"]["review_response_plan_sha256"]==sys.argv[7]
assert len(release["frozen_identity"]["model_revision"])==40
assert release["data_policy"]=={
    "dataset":"PG19","split":"train","distinct_books":8,
    "longbench_consumed":False,"validation_consumed":False,
    "test_v2_consumed":False,
}
assert release["mutant_case_isolation"]["fresh_document_cache_per_case"] is True
assert release["mutant_case_isolation"]["fresh_request_cache_per_case"] is True
assert release["raw_artifact_integrity"]["detached_external_sha256_receipts_required"] is True
assert release["measurement_cell_isolation"]["cell_ids_must_differ"] is True
assert static["formal_ready"] is False and static["passed"] is True
assert static["input_provenance"]["mode"]=="formal_preoutput_inputs"
assert static["input_provenance"]["rr2_input_manifest_raw_sha256"]==sys.argv[5]
assert static["input_provenance"]["prior_fp32_context_manifest_raw_sha256"]==sys.argv[6]
assert static["input_provenance"]["review_response_plan_raw_sha256"]==sys.argv[7]
assert static["protocol_config"]["world_size"]==8
assert static["protocol_config"]["mutant_cache_rebuilt_per_case"] is True
assert static["protocol_config"]["ownership_witness_cell_separate_from_memory_cell"] is True
print(json.dumps({"status":"static_release_gates_passed","formal_ready":False},sort_keys=True))
PY
date -u +%FT%TZ > "$RUN_DIR/stages/02_static_preregistration_ok"

CURRENT_PHASE=run_identity
# Generate exactly one run identity after static preregistration and before any
# candidate output.  The nonce prevents hard-coding/reuse; the derivation and
# receipt bind the 128-bit ID to this static artifact and protocol receipt.
RUN_ID=$(
  timeout --signal=TERM --kill-after=15s 120s \
    env PYTHONPATH="$CODE_DIR" "$PYTHON" "$MANIFEST_BUILDER" \
    run-id-receipt \
    --static-artifact-sha256 "$STATIC_ARTIFACT_SHA256" \
    --protocol-manifest-sha256 "$EXPECTED_PROTOCOL_SOURCE_MANIFEST_SHA256" \
    --output "$RUN_DIR/receipts/run-id-receipt.json"
)
if [[ ! "$RUN_ID" =~ ^[0-9a-f]{32}$ ]]; then
  fail_stage "run ID is not one non-empty lowercase 128-bit identifier"
fi
RUN_ID_RECEIPT_SHA256=$(json_digest "$RUN_DIR/receipts/run-id-receipt.json")
printf '%s\n' "$RUN_ID_RECEIPT_SHA256" \
  > "$RUN_DIR/receipts/run-id-receipt.canonical-json.sha256"
date -u +%FT%TZ > "$RUN_DIR/stages/02_run_identity_bound"

CURRENT_PHASE=producer_release_gate
PRODUCER_STATE=$(
  timeout --signal=TERM --kill-after=15s 120s \
    env PYTHONPATH="$CODE_DIR" "$PYTHON" - <<'PY'
import run_qcomem_qwen35_forkaudit_review_revision as runner
print("true" if runner.GPU_LOOP_IMPLEMENTED else "false")
PY
)
if [[ "$PRODUCER_STATE" != true || "$FORMAL_PIPELINE_RELEASED" != true ]]; then
  printf '%s\n' \
    "formal GPU release gates are not jointly enabled" \
    > "$RUN_DIR/stages/BLOCKED_GPU_PRODUCER_NOT_IMPLEMENTED"
  printf '%s\n' "$STATIC_ARTIFACT_SHA256" \
    > "$RUN_DIR/receipts/static-artifact.canonical-json.sha256"
  # Deliberate fail-closed terminal state: no GPU query, shard, aggregate, or
  # scheduler create command is reached.  99_done must not exist.
  trap - ERR INT TERM
  exit 3
fi

# Unreachable in this release.  This section is staged for the joint reviewed
# change that enables both gates; it never provisions resources and can only
# use eight GPUs already allocated to this process.
CURRENT_PHASE=private_model_view
timeout --signal=TERM --kill-after=60s 21600s \
  env PYTHONPATH="$CODE_DIR" "$PYTHON" "$MANIFEST_BUILDER" \
  materialize-private-model-view \
  --source-model-dir "$MODEL_DIR" \
  --private-model-view "$PRIVATE_MODEL_VIEW" \
  --model-artifact-ledger "$MODEL_ARTIFACT_LEDGER_FILE" \
  --expected-model-artifact-ledger-raw-sha256 \
    "$EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256" \
  --model-weight-ledger "$MODEL_WEIGHT_LEDGER_FILE" \
  --expected-model-weight-ledger-raw-sha256 \
    "$EXPECTED_MODEL_WEIGHT_LEDGER_SHA256" \
  --model-id "$MODEL_ID" \
  --model-revision "$MODEL_REVISION" \
  --manifest-output "$RUN_DIR/preregistration/private-model-view-manifest.json" \
  > "$RUN_DIR/logs/private-model-view-materialization.json"
PRIVATE_MODEL_VIEW_MANIFEST_RAW_SHA256=$(
  sha256sum "$RUN_DIR/preregistration/private-model-view-manifest.json" | \
    awk '{print $1}'
)
if [[ ! "$PRIVATE_MODEL_VIEW_MANIFEST_RAW_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  fail_stage "private model-view manifest raw SHA is invalid"
fi
LEASE_CONTROL_FIFO="$RUN_DIR/receipts/model-load-lease-control.fifo"
LEASE_EVENT_FIFO="$RUN_DIR/receipts/model-load-lease-events.fifo"
mkfifo -m 600 "$LEASE_CONTROL_FIFO" "$LEASE_EVENT_FIFO"
timeout --signal=TERM --kill-after=60s 21600s \
  env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  "$PYTHON" -I -c \
  'import runpy,signal,sys; signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGIO}); script=sys.argv[1]; sys.path.insert(0,sys.argv[2]); sys.argv=[script,*sys.argv[3:]]; runpy.run_path(script,run_name="__main__")' \
  "$MANIFEST_BUILDER" "$CODE_DIR" \
  model-load-lease-keeper \
  --model-view "$PRIVATE_MODEL_VIEW" \
  --model-weight-ledger "$MODEL_WEIGHT_LEDGER_FILE" \
  --expected-model-weight-ledger-raw-sha256 \
    "$EXPECTED_MODEL_WEIGHT_LEDGER_SHA256" \
  --expected-model-artifact-ledger-raw-sha256 \
    "$EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256" \
  --model-view-manifest \
    "$RUN_DIR/preregistration/private-model-view-manifest.json" \
  --expected-model-view-manifest-raw-sha256 \
    "$PRIVATE_MODEL_VIEW_MANIFEST_RAW_SHA256" \
  --run-id "$RUN_ID" \
  --authority-output "$RUN_DIR/receipts/model-load-authority.json" \
  --closure-output "$RUN_DIR/receipts/model-load-closure.json" \
  < "$LEASE_CONTROL_FIFO" > "$LEASE_EVENT_FIFO" \
  2> "$RUN_DIR/logs/model-load-lease-keeper.log" &
LEASE_KEEPER_PID=$!
exec 8> "$LEASE_CONTROL_FIFO"
exec 9< "$LEASE_EVENT_FIFO"
if ! IFS= read -r MODEL_LOAD_READY <&9; then
  fail_stage "model-load lease keeper did not emit READY"
fi
if [[ ! "$MODEL_LOAD_READY" =~ ^READY[[:space:]]([0-9a-f]{64})$ ]]; then
  fail_stage "model-load lease keeper READY schema drift"
fi
MODEL_LOAD_AUTHORITY_RAW_SHA256="${BASH_REMATCH[1]}"
verify_sha "$RUN_DIR/receipts/model-load-authority.json" \
  "$MODEL_LOAD_AUTHORITY_RAW_SHA256" model-load-authority-pre-output
date -u +%FT%TZ > "$RUN_DIR/stages/03_private_model_view_ok"

CURRENT_PHASE=formal_gpu_preflight
GPU_COUNT=$(timeout --signal=TERM --kill-after=15s 120s nvidia-smi -L | wc -l | tr -d ' ')
if [[ "$GPU_COUNT" -ne 8 ]]; then
  fail_stage "formal ForkAudit requires exactly eight already-allocated GPUs"
fi
nvidia-smi \
  --query-gpu=index,uuid,name,memory.total,driver_version,pstate,power.limit \
  --format=csv > "$RUN_DIR/gpus-before.csv"
nvidia-smi \
  --query-gpu=index,uuid,name,memory.total,compute_cap \
  --format=csv,noheader,nounits \
  > "$RUN_DIR/preregistration/gpu-assignment-inventory.csv"
timeout --signal=TERM --kill-after=15s 120s \
  env PYTHONPATH="$CODE_DIR" "$PYTHON" "$MANIFEST_BUILDER" \
  gpu-assignment-receipt \
  --inventory "$RUN_DIR/preregistration/gpu-assignment-inventory.csv" \
  --output "$RUN_DIR/receipts/gpu-assignment-receipt.json" \
  > "$RUN_DIR/logs/gpu-assignment-receipt-build.json"
GPU_ASSIGNMENT_RECEIPT_RAW_SHA256=$(
  sha256sum "$RUN_DIR/receipts/gpu-assignment-receipt.json" | awk '{print $1}'
)
if [[ ! "$GPU_ASSIGNMENT_RECEIPT_RAW_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  fail_stage "GPU assignment receipt raw SHA is invalid"
fi
printf '%s\n' "$GPU_ASSIGNMENT_RECEIPT_RAW_SHA256" \
  > "$RUN_DIR/receipts/gpu-assignment-receipt.raw.sha256"
timeout --signal=TERM --kill-after=15s 120s \
  "$PYTHON" -I - \
  "$RUN_DIR/receipts/gpu-assignment-receipt.json" \
  "$RUN_DIR/receipts/gpu-assignment-rank-map.tsv" <<'PY'
import json,sys
from pathlib import Path
value=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
rows=value["rows"]
assert len(rows)==8 and [row["rank"] for row in rows]==list(range(8))
Path(sys.argv[2]).write_text(
    "".join(f'{row["rank"]}\t{row["visible_index"]}\t{row["uuid"]}\n' for row in rows),
    encoding="utf-8",
)
PY
GPU_VISIBLE_INDICES=()
GPU_UUIDS=()
while IFS=$'\t' read -r GPU_RANK GPU_VISIBLE_INDEX GPU_UUID; do
  if [[ "$GPU_RANK" -ne "${#GPU_VISIBLE_INDICES[@]}" ]]; then
    fail_stage "GPU assignment rank order drift"
  fi
  GPU_VISIBLE_INDICES+=("$GPU_VISIBLE_INDEX")
  GPU_UUIDS+=("$GPU_UUID")
done < "$RUN_DIR/receipts/gpu-assignment-rank-map.tsv"
if [[ "${#GPU_UUIDS[@]}" -ne 8 ]]; then
  fail_stage "formal GPU assignment differs from eight"
fi
date -u +%FT%TZ > "$RUN_DIR/stages/03_formal_gpu_preflight_ok"

CURRENT_PHASE=eight_rank_shards
PIDS=()
for RANK in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES="${GPU_UUIDS[$RANK]}" \
  PYTHONPATH="$CODE_DIR" \
    timeout --signal=TERM --kill-after=60s 21600s \
    "$PYTHON" "$RUNNER" \
    --stage shard \
    --rank "$RANK" \
    --run-id "$RUN_ID" \
    --artifact-root "$RUN_DIR/raw" \
    --static-artifact "$RUN_DIR/preregistration/static-artifact.json" \
    --expected-static-sha256 "$STATIC_ARTIFACT_SHA256" \
    --rr2-input-manifest "$PG19_INPUT_MANIFEST" \
    --expected-rr2-input-manifest-sha256 \
      "$EXPECTED_PG19_INPUT_MANIFEST_SHA256" \
    --pg19-data "$PG19_DATA" \
    --pg19-manifest "$PG19_MANIFEST" \
    --prior-capacity-manifest "$PRIOR_CAPACITY_MANIFEST" \
    --model-dir "$PRIVATE_MODEL_VIEW" \
    --code-ledger "$RUN_DIR/preregistration/code.sha256" \
    --model-artifact-ledger "$MODEL_ARTIFACT_LEDGER_FILE" \
    --model-weight-ledger "$MODEL_WEIGHT_LEDGER_FILE" \
    --protocol-manifest "$PROTOCOL_SOURCE_MANIFEST" \
    --run-id-receipt "$RUN_DIR/receipts/run-id-receipt.json" \
    --expected-run-id-receipt-sha256 "$RUN_ID_RECEIPT_SHA256" \
    --gpu-assignment-receipt \
      "$RUN_DIR/receipts/gpu-assignment-receipt.json" \
    --expected-gpu-assignment-receipt-raw-sha256 \
      "$GPU_ASSIGNMENT_RECEIPT_RAW_SHA256" \
    --model-load-authority "$RUN_DIR/receipts/model-load-authority.json" \
    --expected-model-load-authority-raw-sha256 \
      "$MODEL_LOAD_AUTHORITY_RAW_SHA256" \
    --private-model-view-manifest \
      "$RUN_DIR/preregistration/private-model-view-manifest.json" \
    --expected-private-model-view-manifest-raw-sha256 \
      "$PRIVATE_MODEL_VIEW_MANIFEST_RAW_SHA256" \
    --expected-gpu-uuid "${GPU_UUIDS[$RANK]}" \
    --output "$RUN_DIR/raw/shards/forkaudit-shard-$RANK.json" \
    > "$RUN_DIR/logs/shard-rank-$RANK.log" 2>&1 &
  PIDS+=("$!")
done
for INDEX in 0 1 2 3 4 5 6 7; do
  if ! wait "${PIDS[$INDEX]}"; then
    echo "ForkAudit rank $INDEX failed" >&2
    terminate_children
    fail_stage "eight-rank shard phase failed"
  fi
done
PIDS=()
printf 'CLOSE %s\n' "$MODEL_LOAD_AUTHORITY_RAW_SHA256" >&8
if ! IFS= read -r MODEL_LOAD_CLOSED <&9; then
  fail_stage "model-load lease keeper did not emit CLOSED"
fi
exec 8>&-
exec 9<&-
if ! wait "$LEASE_KEEPER_PID"; then
  fail_stage "model-load lease keeper failed closure"
fi
LEASE_KEEPER_PID=""
rm -f "$LEASE_CONTROL_FIFO" "$LEASE_EVENT_FIFO"
LEASE_CONTROL_FIFO=""
LEASE_EVENT_FIFO=""
if [[ ! "$MODEL_LOAD_CLOSED" =~ ^CLOSED[[:space:]]([0-9a-f]{64})$ ]]; then
  fail_stage "model-load lease keeper CLOSED schema drift"
fi
MODEL_LOAD_CLOSURE_RAW_SHA256="${BASH_REMATCH[1]}"
verify_sha "$RUN_DIR/receipts/model-load-closure.json" \
  "$MODEL_LOAD_CLOSURE_RAW_SHA256" model-load-closure-pre-aggregate
SHARD_COUNT=$(find "$RUN_DIR/raw/shards" -maxdepth 1 -type f \
  -name 'forkaudit-shard-*.json' | wc -l | tr -d ' ')
if [[ "$SHARD_COUNT" -ne 8 ]]; then
  fail_stage "raw shard cardinality differs from eight"
fi
date -u +%FT%TZ > "$RUN_DIR/stages/04_eight_rank_shards_ok"

CURRENT_PHASE=detached_raw_receipts
(
  cd "$RUN_DIR"
  find raw -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum
) > "$RUN_DIR/receipts/all-raw-artifacts.sha256"
timeout --signal=TERM --kill-after=30s 1800s bash -c \
  'cd "$1" && sha256sum -c receipts/all-raw-artifacts.sha256' _ "$RUN_DIR" \
  > "$RUN_DIR/logs/raw-artifact-integrity.log"
timeout --signal=TERM --kill-after=30s 300s \
  env PYTHONPATH="$CODE_DIR" "$PYTHON" "$MANIFEST_BUILDER" receipts \
  --artifact-root "$RUN_DIR/raw" \
  --static-artifact "$RUN_DIR/preregistration/static-artifact.json" \
  --run-id-receipt "$RUN_DIR/receipts/run-id-receipt.json" \
  --expected-run-id-receipt-sha256 "$RUN_ID_RECEIPT_SHA256" \
  --run-id "$RUN_ID" \
  --protocol-manifest-sha256 "$EXPECTED_PROTOCOL_SOURCE_MANIFEST_SHA256" \
  --output "$RUN_DIR/receipts/detached-receipt-manifest.json" \
  > "$RUN_DIR/logs/receipt-build.json"
RECEIPT_MANIFEST_SHA256=$(json_digest "$RUN_DIR/receipts/detached-receipt-manifest.json")
printf '%s\n' "$RECEIPT_MANIFEST_SHA256" \
  > "$RUN_DIR/receipts/detached-receipt-manifest.canonical-json.sha256"
date -u +%FT%TZ > "$RUN_DIR/stages/05_detached_raw_receipts_ok"

CURRENT_PHASE=blind_aggregate
timeout --signal=TERM --kill-after=60s 1800s \
  env PYTHONPATH="$CODE_DIR" "$PYTHON" "$RUNNER" \
  --stage aggregate \
  --run-id "$RUN_ID" \
  --output "$RUN_DIR/forkaudit-summary.json" \
  --artifact-root "$RUN_DIR/raw" \
  --static-artifact "$RUN_DIR/preregistration/static-artifact.json" \
  --expected-static-sha256 "$STATIC_ARTIFACT_SHA256" \
  --receipt-manifest "$RUN_DIR/receipts/detached-receipt-manifest.json" \
  --expected-receipt-manifest-sha256 "$RECEIPT_MANIFEST_SHA256" \
  --run-id-receipt "$RUN_DIR/receipts/run-id-receipt.json" \
  --expected-run-id-receipt-sha256 "$RUN_ID_RECEIPT_SHA256" \
  --gpu-assignment-receipt "$RUN_DIR/receipts/gpu-assignment-receipt.json" \
  --expected-gpu-assignment-receipt-raw-sha256 \
    "$GPU_ASSIGNMENT_RECEIPT_RAW_SHA256" \
  --model-load-authority "$RUN_DIR/receipts/model-load-authority.json" \
  --expected-model-load-authority-raw-sha256 \
    "$MODEL_LOAD_AUTHORITY_RAW_SHA256" \
  --model-load-closure "$RUN_DIR/receipts/model-load-closure.json" \
  --expected-model-load-closure-raw-sha256 \
    "$MODEL_LOAD_CLOSURE_RAW_SHA256" \
  --private-model-view-manifest \
    "$RUN_DIR/preregistration/private-model-view-manifest.json" \
  --expected-private-model-view-manifest-raw-sha256 \
    "$PRIVATE_MODEL_VIEW_MANIFEST_RAW_SHA256" \
  --model-artifact-ledger "$MODEL_ARTIFACT_LEDGER_FILE" \
  --model-weight-ledger "$MODEL_WEIGHT_LEDGER_FILE" \
  > "$RUN_DIR/logs/aggregate.log" 2>&1
timeout --signal=TERM --kill-after=15s 120s "$PYTHON" - \
  "$RUN_DIR/forkaudit-summary.json" "$RUN_ID" <<'PY' \
  > "$RUN_DIR/logs/aggregate-final-audit.json"
import json,sys
value=json.load(open(sys.argv[1],encoding="utf-8"))
assert value["run_id"]==sys.argv[2]
assert value["formal_ready"] is True and value["passed"] is True
assert value["rank_count"]==8
assert value["factorial_four_cell_exact"] is True
assert value["oracle_all_ranks_passed"] is True
assert value["mutant_campaign"]["passed"] is True
assert value["mutant_campaign"]["escaped_mutant_ids"]==[]
assert value["mutant_campaign"]["wrong_gate_mutant_ids"]==[]
assert value["mutant_campaign"]["unexpected_crash_mutant_ids"]==[]
print(json.dumps({"status":"formal_aggregate_passed"},sort_keys=True))
PY
date -u +%FT%TZ > "$RUN_DIR/stages/06_blind_aggregate_ok"

CURRENT_PHASE=terminal_integrity
CURRENT_RUN_ID_RECEIPT_SHA256=$(json_digest "$RUN_DIR/receipts/run-id-receipt.json")
if [[ "$CURRENT_RUN_ID_RECEIPT_SHA256" != "$RUN_ID_RECEIPT_SHA256" ]]; then
  fail_stage "run-ID binding receipt changed after candidate outputs"
fi
verify_sha "$RUN_DIR/receipts/gpu-assignment-receipt.json" \
  "$GPU_ASSIGNMENT_RECEIPT_RAW_SHA256" gpu-assignment-receipt-terminal
nvidia-smi \
  --query-gpu=index,uuid,name,memory.total,compute_cap \
  --format=csv,noheader,nounits \
  > "$RUN_DIR/receipts/gpu-assignment-inventory-terminal.csv"
timeout --signal=TERM --kill-after=15s 120s \
  env PYTHONPATH="$CODE_DIR" "$PYTHON" "$MANIFEST_BUILDER" \
  gpu-assignment-receipt \
  --inventory "$RUN_DIR/receipts/gpu-assignment-inventory-terminal.csv" \
  --output "$RUN_DIR/receipts/gpu-assignment-receipt-terminal.json" \
  > "$RUN_DIR/logs/gpu-assignment-terminal-build.json"
verify_sha "$RUN_DIR/receipts/gpu-assignment-receipt-terminal.json" \
  "$GPU_ASSIGNMENT_RECEIPT_RAW_SHA256" gpu-assignment-rebuilt-terminal
if ! cmp -s "$RUN_DIR/receipts/gpu-assignment-receipt.json" \
  "$RUN_DIR/receipts/gpu-assignment-receipt-terminal.json"; then
  fail_stage "GPU assignment receipt changed after candidate outputs"
fi
verify_sha "$RUN_DIR/receipts/model-load-authority.json" \
  "$MODEL_LOAD_AUTHORITY_RAW_SHA256" model-load-authority-terminal
verify_sha "$RUN_DIR/receipts/model-load-closure.json" \
  "$MODEL_LOAD_CLOSURE_RAW_SHA256" model-load-closure-terminal
verify_sha "$RUN_DIR/preregistration/private-model-view-manifest.json" \
  "$PRIVATE_MODEL_VIEW_MANIFEST_RAW_SHA256" private-model-view-manifest-terminal
timeout --signal=TERM --kill-after=15s 120s "$PYTHON" - \
  "$RUN_DIR/receipts/run-id-receipt.json" "$RUN_ID" \
  "$STATIC_ARTIFACT_SHA256" "$EXPECTED_PROTOCOL_SOURCE_MANIFEST_SHA256" <<'PY' \
  > "$RUN_DIR/logs/run-id-terminal-audit.json"
import hashlib,json,sys
value=json.load(open(sys.argv[1],encoding="utf-8"))
assert value["run_id"]==sys.argv[2]
assert value["run_id_bits"]==128
assert value["static_artifact_sha256"]==sys.argv[3]
assert value["protocol_manifest_sha256"]==sys.argv[4]
domain=bytes.fromhex(value["domain_hex"])
nonce=bytes.fromhex(value["nonce_hex"])
replayed=hashlib.sha256(
    domain+bytes.fromhex(sys.argv[3])+bytes.fromhex(sys.argv[4])+nonce
).hexdigest()[:32]
assert replayed==sys.argv[2]
print(json.dumps({"status":"run_id_receipt_replayed","run_id":replayed},sort_keys=True))
PY
CURRENT_RECEIPT_SHA256=$(json_digest "$RUN_DIR/receipts/detached-receipt-manifest.json")
if [[ "$CURRENT_RECEIPT_SHA256" != "$RECEIPT_MANIFEST_SHA256" ]]; then
  fail_stage "detached receipt manifest changed after aggregation"
fi
audit_code_snapshot \
  "$CODE_DIR" \
  "$RUN_DIR/receipts/code-files-terminal.nul" \
  "$RUN_DIR/receipts/code-terminal.sha256"
# Re-anchor both the retained preregistration ledger and the freshly rebuilt
# terminal ledger to the immutable external receipt.  Comparing the two files
# alone is insufficient: a coordinated post-preflight rewrite could otherwise
# replace both with a self-consistent ledger for modified code.
verify_sha "$RUN_DIR/preregistration/code.sha256" \
  "$EXPECTED_CODE_LEDGER_SHA256" code-ledger-preregistration-terminal
verify_sha "$RUN_DIR/receipts/code-terminal.sha256" \
  "$EXPECTED_CODE_LEDGER_SHA256" code-ledger-rebuilt-terminal
if ! cmp -s \
  "$RUN_DIR/preregistration/code-files.nul" \
  "$RUN_DIR/receipts/code-files-terminal.nul"; then
  fail_stage "recursive code closure changed after preflight"
fi
if ! cmp -s \
  "$RUN_DIR/preregistration/code.sha256" \
  "$RUN_DIR/receipts/code-terminal.sha256"; then
  fail_stage "recursive code ledger changed after preflight"
fi
timeout --signal=TERM --kill-after=30s 300s bash -c \
  'cd "$1" && sha256sum -c "$2"' _ "$CODE_DIR" \
  "$RUN_DIR/preregistration/code.sha256" \
  > "$RUN_DIR/logs/code-terminal-integrity.log"
timeout --signal=TERM --kill-after=30s 900s bash -c \
  'cd "$1" && sha256sum -c "$2"' _ "$MODEL_DIR" \
  "$(realpath "$MODEL_ARTIFACT_LEDGER_FILE")" \
  > "$RUN_DIR/logs/model-artifact-terminal-integrity.log"
timeout --signal=TERM --kill-after=60s 7200s bash -c \
  'cd "$1" && sha256sum -c "$2"' _ "$MODEL_DIR" \
  "$(realpath "$MODEL_WEIGHT_LEDGER_FILE")" \
  > "$RUN_DIR/logs/model-weight-terminal-integrity.log"
verify_sha "$PG19_DATA" "$EXPECTED_PG19_SHA256" pg19-train-terminal
verify_sha "$PG19_MANIFEST" "$EXPECTED_PG19_MANIFEST_SHA256" pg19-manifest-terminal
verify_sha "$RUNNER" "$EXPECTED_RUNNER_SHA256" reviewed-runner-terminal
verify_sha "$PG19_INPUT_MANIFEST" \
  "$EXPECTED_PG19_INPUT_MANIFEST_SHA256" rr2-pg19-input-terminal
verify_sha "$PRIOR_CAPACITY_MANIFEST" \
  "$EXPECTED_PRIOR_CAPACITY_MANIFEST_SHA256" prior-capacity-terminal
verify_sha "$FROZEN_QUERY_BANKS_INPUT" \
  "$EXPECTED_FROZEN_QUERY_BANKS_INPUT_SHA256" rr2-query-banks-input-terminal
verify_sha "$ORACLE_SELECTION_INPUT" \
  "$EXPECTED_ORACLE_SELECTION_INPUT_SHA256" rr2-oracle-input-terminal
verify_sha "$PRIOR_FP32_CONTEXT_MANIFEST" \
  "$EXPECTED_PRIOR_FP32_CONTEXT_MANIFEST_SHA256" prior-fp32-context-terminal
verify_sha "$REVIEW_RESPONSE_PLAN" \
  "$EXPECTED_REVIEW_RESPONSE_PLAN_SHA256" review-response-plan-terminal
verify_sha "$PROTOCOL_SOURCE_MANIFEST" \
  "$EXPECTED_PROTOCOL_SOURCE_MANIFEST_SHA256" protocol-source-terminal
timeout --signal=TERM --kill-after=30s 1800s bash -c \
  'cd "$1" && sha256sum -c receipts/all-raw-artifacts.sha256' _ "$RUN_DIR" \
  > "$RUN_DIR/logs/raw-artifact-terminal-integrity.log"
(
  cd "$RUN_DIR"
  find preregistration raw receipts -type f -print0 | LC_ALL=C sort -z | \
    xargs -0 sha256sum
  sha256sum forkaudit-summary.json
) > "$RUN_DIR/scientific-artifacts.sha256"
timeout --signal=TERM --kill-after=30s 1800s bash -c \
  'cd "$1" && sha256sum -c scientific-artifacts.sha256' _ "$RUN_DIR" \
  > "$RUN_DIR/logs/scientific-artifact-integrity.log"
date -u +%FT%TZ > "$RUN_DIR/stages/99_done"
trap - ERR INT TERM
