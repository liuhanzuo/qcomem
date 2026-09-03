#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
TMP_REPORT=$(mktemp "${TMPDIR:-/tmp}/qcomem60-replay.XXXXXX.json")
trap 'rm -f "$TMP_REPORT"' EXIT

python3 "$ROOT_DIR/replay/verify.py" --output "$TMP_REPORT"
cmp "$TMP_REPORT" "$ROOT_DIR/validation_report.json"
printf 'PASS: manifest, 66 remote-mirror files, 48 shards, 360 F1 rows, 24 bootstrap intervals, Store accounting, and archived aggregate verified.\n'
