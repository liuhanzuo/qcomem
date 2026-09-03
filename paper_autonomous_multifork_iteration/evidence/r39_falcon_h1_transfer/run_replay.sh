#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUN_ROOT=${1:?usage: run_replay.sh /path/to/completed/r39-falcon-run}
PYTHON=${PYTHON:-python3}

[[ -f "$RUN_ROOT/COMPLETE" ]] || { echo "completed run marker absent" >&2; exit 2; }

PYTHONPATH= "$PYTHON" -B \
  "$PACKAGE_ROOT/executed_source/replay_r39_falcon_transfer.py" \
  --package-root "$PACKAGE_ROOT" \
  --run-root "$RUN_ROOT" \
  --verify-existing "$RUN_ROOT/r39-falcon-h1-transfer-aggregate.json"
