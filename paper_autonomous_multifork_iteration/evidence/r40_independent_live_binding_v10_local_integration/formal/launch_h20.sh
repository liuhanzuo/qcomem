#!/usr/bin/env bash
set -euo pipefail
: "${R40_H20_EXECUTION_AUTHORIZED:?authorization required}"
[[ "$R40_H20_EXECUTION_AUTHORIZED" == yes ]] || { echo "authorization must equal yes" >&2; exit 2; }
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
EVIDENCE=$(cd -- "$HERE/.." && pwd)
MARKER=${R40_ONE_SHOT_MARKER:-${TMPDIR:-/tmp}/r40-v10-formal-launch-used}
[[ ! -e "$MARKER" ]] || { echo "one-shot marker exists" >&2; exit 2; }
(cd "$HERE" && sha256sum -c source-code.sha256)
TMP=$(mktemp "${TMPDIR:-/tmp}/r40-v10.XXXXXX"); trap 'rm -f "$TMP"' EXIT; rm -f "$TMP"
python3 "$HERE/scripts/build_formal_launcher.py" --v6 "$EVIDENCE/r39_primary_compiled_dispatch_v6/executed_source/r39_primary_formal_h20.sh" --output "$TMP"
bash -n "$TMP"
touch "$MARKER"
exec bash "$TMP"
