#!/usr/bin/env bash
set -euo pipefail

R29_PACKAGE_DIR=${R29_PACKAGE_DIR:?set immutable R29_PACKAGE_DIR}
UPSTREAM_CODE_DIR=${UPSTREAM_CODE_DIR:?set frozen UPSTREAM_CODE_DIR}
MODEL_DIR=${MODEL_DIR:?set MODEL_DIR}
MODEL_ARTIFACT_LEDGER=${MODEL_ARTIFACT_LEDGER:?set exact model artifact ledger}
MODEL_WEIGHT_LEDGER=${MODEL_WEIGHT_LEDGER:?set exact model weight ledger}
PG19_DATA=${PG19_DATA:?set PG19_DATA}
PG19_MANIFEST=${PG19_MANIFEST:?set PG19_MANIFEST}
ENV_DIR=${ENV_DIR:?set frozen ENV_DIR}
RUN_DIR=${RUN_DIR:?set a fresh experiment-private RUN_DIR}
R29_GPU_UUID=${R29_GPU_UUID:?set one exclusively assigned H20 UUID}

DESIGN="$R29_PACKAGE_DIR/paper_autonomous_multifork_iteration/evidence/r29_true_concurrency/design_preregistration.json"
RUNNER="$R29_PACKAGE_DIR/gpu/r29_true_concurrent_lifecycle.py"
REPLAY="$R29_PACKAGE_DIR/gpu/r29_replay_true_concurrent_lifecycle.py"
TEST="$R29_PACKAGE_DIR/gpu/test_r29_true_concurrent_lifecycle.py"
PYTHON="$ENV_DIR/bin/python"
UPSTREAM_LEDGER="$UPSTREAM_CODE_DIR/code.sha256"

DESIGN_SHA=5c9fc301ec63e2702d097b9d9be9c68758164c653c6c7b53fedad290428a9a96
RUNNER_SHA=401a19314ea3efd24731ed4f2fea9515e961f4532359af566fafe38148c98302
REPLAY_SHA=0ff6b384a5e1be772777578265882b8b50ac5d759a80d42092771483e4eb7042
TEST_SHA=39a83031d88f12fa3f06391c19fb24bd8b21a131ff40f6365376f286f5ddcbc4
UPSTREAM_LEDGER_SHA=7620f05821fc5435a9aaa260ae82577988a5a20eff0f42901b66e6c6871fd2b9
MODEL_ARTIFACT_LEDGER_SHA=c0a23e9d3f9d220257af97b78fd97661f315f0c82a3a010b57a771e3eeefbbfb
MODEL_WEIGHT_LEDGER_SHA=8314a82c9188b9b817193e039b0b0eb0636b328512f19b0c12455853b7e20014
PG19_SHA=ef18893b6bfb0f4b8cb29eab85ccf2a0ef1fdb44606e9742a70405cca564e18c
PG19_MANIFEST_SHA=5d789d67aa239f089e92de8a4267b86d2f1d2723d5f1370970883738f5f89a9c

[[ ! -e "$RUN_DIR" ]] || { echo "RUN_DIR already exists: $RUN_DIR" >&2; exit 2; }
[[ $(sha256sum "$DESIGN" | awk '{print $1}') == "$DESIGN_SHA" ]]
[[ $(sha256sum "$RUNNER" | awk '{print $1}') == "$RUNNER_SHA" ]]
[[ $(sha256sum "$REPLAY" | awk '{print $1}') == "$REPLAY_SHA" ]]
[[ $(sha256sum "$TEST" | awk '{print $1}') == "$TEST_SHA" ]]
[[ $(sha256sum "$UPSTREAM_LEDGER" | awk '{print $1}') == "$UPSTREAM_LEDGER_SHA" ]]
[[ $(sha256sum "$MODEL_ARTIFACT_LEDGER" | awk '{print $1}') == "$MODEL_ARTIFACT_LEDGER_SHA" ]]
[[ $(sha256sum "$MODEL_WEIGHT_LEDGER" | awk '{print $1}') == "$MODEL_WEIGHT_LEDGER_SHA" ]]
[[ $(sha256sum "$PG19_DATA" | awk '{print $1}') == "$PG19_SHA" ]]
[[ $(sha256sum "$PG19_MANIFEST" | awk '{print $1}') == "$PG19_MANIFEST_SHA" ]]
[[ $(nvidia-smi --query-gpu=uuid --format=csv,noheader | sed 's/[[:space:]]//g' | grep -Fxc "$R29_GPU_UUID") -eq 1 ]]

mkdir -p "$RUN_DIR/logs" "$RUN_DIR/raw/sidecars" "$RUN_DIR/replay" "$RUN_DIR/receipts" "$RUN_DIR/stages" "$RUN_DIR/caches/triton" "$RUN_DIR/caches/torchinductor"
date -u +%FT%TZ > "$RUN_DIR/stages/00-started"

(cd "$UPSTREAM_CODE_DIR" && sha256sum -c code.sha256 > "$RUN_DIR/logs/upstream-code-ledger-check.log")
PYTHONPATH="$R29_PACKAGE_DIR/gpu:$UPSTREAM_CODE_DIR" "$PYTHON" -B -m py_compile "$RUNNER" "$REPLAY" "$TEST"
(cd "$R29_PACKAGE_DIR" && PYTHONPATH="$R29_PACKAGE_DIR/gpu:$UPSTREAM_CODE_DIR" "$PYTHON" -B -m unittest -v gpu/test_r29_true_concurrent_lifecycle.py > "$RUN_DIR/logs/focused-tests.log" 2>&1)
PYTHONPATH="$R29_PACKAGE_DIR/gpu:$UPSTREAM_CODE_DIR" "$PYTHON" -B "$RUNNER" \
  --stage mock \
  --design-preregistration "$DESIGN" \
  --expected-design-sha256 "$DESIGN_SHA" \
  --output "$RUN_DIR/receipts/node-mock.json" \
  > "$RUN_DIR/logs/node-mock.stdout.log" \
  2> "$RUN_DIR/logs/node-mock.stderr.log"
date -u +%FT%TZ > "$RUN_DIR/stages/01-preflight-passed"

nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv,noheader > "$RUN_DIR/receipts/node-gpu-inventory.csv"
printf '%s\n' "$R29_GPU_UUID" > "$RUN_DIR/receipts/assigned-gpu-uuid.txt"

CUDA_VISIBLE_DEVICES="$R29_GPU_UUID" \
TOKENIZERS_PARALLELISM=false \
TRITON_CACHE_DIR="$RUN_DIR/caches/triton" \
TORCHINDUCTOR_CACHE_DIR="$RUN_DIR/caches/torchinductor" \
PYTHONPATH="$R29_PACKAGE_DIR/gpu:$UPSTREAM_CODE_DIR" \
"$PYTHON" -B "$RUNNER" \
  --stage formal \
  --design-preregistration "$DESIGN" \
  --expected-design-sha256 "$DESIGN_SHA" \
  --model "$MODEL_DIR" \
  --model-artifact-ledger "$MODEL_ARTIFACT_LEDGER" \
  --model-weight-ledger "$MODEL_WEIGHT_LEDGER" \
  --pg19-data "$PG19_DATA" \
  --pg19-manifest "$PG19_MANIFEST" \
  --upstream-code-ledger "$UPSTREAM_LEDGER" \
  --artifact-dir "$RUN_DIR/raw/sidecars" \
  --output "$RUN_DIR/raw/formal-result.json" \
  > "$RUN_DIR/logs/formal.stdout.log" \
  2> "$RUN_DIR/logs/formal.stderr.log"
date -u +%FT%TZ > "$RUN_DIR/stages/02-formal-complete"

FORMAL_RESULT_SHA=$(sha256sum "$RUN_DIR/raw/formal-result.json" | awk '{print $1}')
PYTHONPATH="$R29_PACKAGE_DIR/gpu:$UPSTREAM_CODE_DIR" "$PYTHON" -B "$REPLAY" \
  --design-preregistration "$DESIGN" \
  --expected-design-sha256 "$DESIGN_SHA" \
  --formal-result "$RUN_DIR/raw/formal-result.json" \
  --expected-formal-result-sha256 "$FORMAL_RESULT_SHA" \
  --artifact-dir "$RUN_DIR/raw/sidecars" \
  --output "$RUN_DIR/replay/independent-replay.json" \
  > "$RUN_DIR/logs/replay.stdout.log" \
  2> "$RUN_DIR/logs/replay.stderr.log"
date -u +%FT%TZ > "$RUN_DIR/stages/03-independent-replay-complete"

(cd "$RUN_DIR" && find raw replay -type f -print0 | sort -z | xargs -0 sha256sum > receipts/raw-and-replay.sha256)
date -u +%FT%TZ > "$RUN_DIR/stages/COMPLETED"
