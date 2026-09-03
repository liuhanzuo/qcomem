#!/usr/bin/env bash
set -euo pipefail

R29_PACKAGE_DIR=${R29_PACKAGE_DIR:?set immutable R29_PACKAGE_DIR}
UPSTREAM_CODE_DIR=${UPSTREAM_CODE_DIR:?set frozen UPSTREAM_CODE_DIR for focused tests}
ENV_DIR=${ENV_DIR:?set frozen ENV_DIR}
RUN_DIR=${RUN_DIR:?set the completed formal-b RUN_DIR}
PRE_REPLAY_V2_LEDGER=${PRE_REPLAY_V2_LEDGER:?set the frozen 13-entry raw/audit pre-replay-v2 ledger}

DESIGN="$R29_PACKAGE_DIR/paper_autonomous_multifork_iteration/evidence/r29_live_overhead/preregistration.json"
RUNNER="$R29_PACKAGE_DIR/gpu/r29_live_overhead.py"
REPLAY="$R29_PACKAGE_DIR/gpu/r29_replay_live_overhead.py"
TEST="$R29_PACKAGE_DIR/gpu/test_r29_live_overhead.py"
PYTHON="$ENV_DIR/bin/python"
UPSTREAM_LEDGER="$UPSTREAM_CODE_DIR/code.sha256"
FORMAL_RESULT="$RUN_DIR/raw/formal-result.json"
ARTIFACT_DIR="$RUN_DIR/raw/audit"
REPLAY_OUTPUT="$RUN_DIR/replay/independent-replay-v2.json"

DESIGN_SHA=2114d2cd85bedc1eafa5d1398fd0afd0d57819c0360c3be3f9ec20f1b2878939
RUNNER_SHA=d4d6e04ba07e90438472b72412a6cb302c431e0626f495cae8ce18b2344c825b
REPLAY_SHA=53e70a1f1af5989c5e3a2a18d7097b37fcc83fa9821c18a54d05953328d7c54d
TEST_SHA=b39a203793f686a0f18c7a1c2944f9e0e8dae4ce292f06dc7f5e302d1d8c38e6
UPSTREAM_LEDGER_SHA=7620f05821fc5435a9aaa260ae82577988a5a20eff0f42901b66e6c6871fd2b9
FORMAL_RESULT_SHA=3ccf86e2233b560f003d965fdae05a8e3b0773e15976a05c8d70af881338bc22
FORMAL_STDOUT_SHA=717c93f37b5e2e2b7313b694606ee629bb87fb7330258d74d13546d88ba6e76a
FORMAL_STDERR_SHA=78b2e4662b8fbf3729516e7ad2fe00d1c0de64d18d0455abc8a9714589081a8c
OLD_REPLAY_STDOUT_SHA=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
OLD_REPLAY_STDERR_SHA=eee313d85c24b5f23f21e633a1973890d388e04f77cc8fd3b0de0b38d427ce97
PRE_REPLAY_V2_LEDGER_SHA=995a7cdfb0f502e4a0b4d603e59482f479d400c46e090ac7ff6908ca4d4b14fe

[[ -d "$RUN_DIR" ]]
[[ -f "$RUN_DIR/stages/00-started" ]]
[[ -f "$RUN_DIR/stages/01-preflight-passed" ]]
[[ -f "$RUN_DIR/stages/02-formal-complete" ]]
[[ ! -e "$RUN_DIR/stages/03-independent-replay-complete" ]]
[[ ! -e "$RUN_DIR/stages/03-independent-replay-v2-complete" ]]
[[ ! -e "$RUN_DIR/stages/COMPLETED" ]]
[[ ! -e "$RUN_DIR/stages/COMPLETED-v2" ]]
[[ ! -e "$REPLAY_OUTPUT" ]]
[[ ! -e "$RUN_DIR/logs/replay-v2.stdout.log" ]]
[[ ! -e "$RUN_DIR/logs/replay-v2.stderr.log" ]]
[[ ! -e "$RUN_DIR/logs/focused-tests-replay-v2.log" ]]
[[ ! -e "$RUN_DIR/receipts/raw-and-replay-v2.sha256" ]]

[[ $(sha256sum "$DESIGN" | awk '{print $1}') == "$DESIGN_SHA" ]]
[[ $(sha256sum "$RUNNER" | awk '{print $1}') == "$RUNNER_SHA" ]]
[[ $(sha256sum "$REPLAY" | awk '{print $1}') == "$REPLAY_SHA" ]]
[[ $(sha256sum "$TEST" | awk '{print $1}') == "$TEST_SHA" ]]
[[ $(sha256sum "$UPSTREAM_LEDGER" | awk '{print $1}') == "$UPSTREAM_LEDGER_SHA" ]]
[[ $(sha256sum "$FORMAL_RESULT" | awk '{print $1}') == "$FORMAL_RESULT_SHA" ]]
[[ $(sha256sum "$RUN_DIR/logs/formal.stdout.log" | awk '{print $1}') == "$FORMAL_STDOUT_SHA" ]]
[[ $(sha256sum "$RUN_DIR/logs/formal.stderr.log" | awk '{print $1}') == "$FORMAL_STDERR_SHA" ]]
[[ $(sha256sum "$RUN_DIR/logs/replay.stdout.log" | awk '{print $1}') == "$OLD_REPLAY_STDOUT_SHA" ]]
[[ $(sha256sum "$RUN_DIR/logs/replay.stderr.log" | awk '{print $1}') == "$OLD_REPLAY_STDERR_SHA" ]]
[[ $(sha256sum "$PRE_REPLAY_V2_LEDGER" | awk '{print $1}') == "$PRE_REPLAY_V2_LEDGER_SHA" ]]

(cd "$UPSTREAM_CODE_DIR" && sha256sum -c code.sha256 > /dev/null)
(cd "$RUN_DIR" && sha256sum -c "$PRE_REPLAY_V2_LEDGER" > /dev/null)

CUDA_VISIBLE_DEVICES="" \
PYTHONPATH="$R29_PACKAGE_DIR/gpu:$UPSTREAM_CODE_DIR" \
"$PYTHON" -B -m py_compile "$REPLAY" "$TEST"

(cd "$R29_PACKAGE_DIR" && \
  CUDA_VISIBLE_DEVICES="" \
  PYTHONPATH="$R29_PACKAGE_DIR/gpu:$UPSTREAM_CODE_DIR" \
  "$PYTHON" -B -m unittest -v gpu/test_r29_live_overhead.py \
  > "$RUN_DIR/logs/focused-tests-replay-v2.log" 2>&1)

CUDA_VISIBLE_DEVICES="" \
TOKENIZERS_PARALLELISM=false \
PYTHONPATH="$R29_PACKAGE_DIR/gpu" \
"$PYTHON" -B "$REPLAY" \
  --design-preregistration "$DESIGN" \
  --expected-design-sha256 "$DESIGN_SHA" \
  --formal-result "$FORMAL_RESULT" \
  --expected-formal-result-sha256 "$FORMAL_RESULT_SHA" \
  --artifact-dir "$ARTIFACT_DIR" \
  --output "$REPLAY_OUTPUT" \
  > "$RUN_DIR/logs/replay-v2.stdout.log" \
  2> "$RUN_DIR/logs/replay-v2.stderr.log"

date -u +%FT%TZ > "$RUN_DIR/stages/03-independent-replay-v2-complete"
(cd "$RUN_DIR" && \
  find raw replay -type f -print0 | sort -z | xargs -0 sha256sum \
  > receipts/raw-and-replay-v2.sha256)
date -u +%FT%TZ > "$RUN_DIR/stages/COMPLETED-v2"
