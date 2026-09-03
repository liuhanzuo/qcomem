#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUN_ROOT=${1:?usage: run_replay.sh /path/to/completed/run}
PYTHON=${PYTHON:-python3}

"$PYTHON" -B "$PACKAGE_ROOT/executed_source/replay_r39_second_model_transfer.py" \
  --package-root "$PACKAGE_ROOT" \
  --run-root "$RUN_ROOT" \
  --verify-existing "$RUN_ROOT/r39-second-model-transfer-aggregate.json"

