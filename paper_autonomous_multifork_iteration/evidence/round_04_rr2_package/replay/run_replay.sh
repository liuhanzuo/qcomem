#!/bin/sh
set -eu
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PACKAGE_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
export PYTHONDONTWRITEBYTECODE=1
TMP=$(mktemp -d "${TMPDIR:-/tmp}/rr2-reviewer-replay.XXXXXX")
trap 'rm -rf "$TMP"' EXIT HUP INT TERM
python3 "$SCRIPT_DIR/verify_package_manifest.py" --package-root "$PACKAGE_ROOT"
python3 "$SCRIPT_DIR/replay_rr2.py" --package-root "$PACKAGE_ROOT" --output "$TMP/derived"
(cd "$SCRIPT_DIR" && python3 test_storage_witness.py)
