#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export PYTHONDONTWRITEBYTECODE=1
TMPDIR_REPLAY=$(mktemp -d)
trap 'rm -rf "$TMPDIR_REPLAY"' EXIT HUP INT TERM
python3 "$ROOT/replay.py" --package-root "$ROOT" --output "$TMPDIR_REPLAY/replayed-result.json"
