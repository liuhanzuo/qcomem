#!/usr/bin/env bash
set -euo pipefail

CODE_DIR=${CODE_DIR:?set CODE_DIR}
DATA_FILE=${DATA_FILE:?set DATA_FILE}
RUN_DIR=${RUN_DIR:?set RUN_DIR}

FROZEN_DATA_SHA256=1553739496b3c209988de56a4ccf574d407379b6b7937ddfafacbe66925069fe
FROZEN_SOURCE_REVISION=5e628be450b7e67fb7ae6e201bd6d8f7056f7672
FROZEN_CONFIGS=full-prefix-q16,qcomem-d7-r16-a16-l16,qcomem-d7-frozen-static

case "$DATA_FILE" in
  *test-v2*|*test_v2*)
    echo "test-v2 paths are forbidden for the COW short protocol" >&2
    exit 1
    ;;
esac
ACTUAL_DATA_SHA256=$(sha256sum "$DATA_FILE" | awk '{print $1}')
if [ "$ACTUAL_DATA_SHA256" != "$FROZEN_DATA_SHA256" ]; then
  echo "validation data SHA256 mismatch" >&2
  exit 1
fi
if [ -e "$RUN_DIR" ] && [ -n "$(find "$RUN_DIR" -mindepth 1 -print -quit)" ]; then
  echo "RUN_DIR must be absent or empty: $RUN_DIR" >&2
  exit 1
fi

export WORKLOAD=longbench
export LIMIT_PER_DATASET=4
export SOURCE_INDEX_START=6
export SOURCE_INDEX_END=9
export MAX_INPUT_TOKENS=4096
export MAX_NEW_TOKENS=8
export GATE_DOCUMENT_TOKENS=256
export GATE_QUERY_TOKENS=32
export GATE_NEW_TOKENS=4
export WARMUPS=1
export REPEATS=1
export FORK_STRATEGY=paged-cow-staging
export GATE_ONLY=0
export CONFIGS=$FROZEN_CONFIGS
export EXPECTED_DATA_SHA256=$FROZEN_DATA_SHA256
export EXPECTED_SOURCE_REVISION=$FROZEN_SOURCE_REVISION
export EXPECTED_SOURCE_INDICES=6,7,8,9
export EXPECTED_WORKLOADS=8
export PROTOCOL_LABEL=qcomem-cow-4k-short-incremental-three-way-v1
export REQUIRE_COMPLETE_MEASUREMENTS=1
export REQUIRE_NO_TEST_V2=1
unset MIXED_POLICY_FILE

exec bash "$CODE_DIR/launch_deployment_8gpu.sh"
