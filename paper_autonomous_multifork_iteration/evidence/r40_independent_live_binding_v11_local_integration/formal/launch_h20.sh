#!/usr/bin/env bash
set -euo pipefail
FAILURE_LEDGER=${R40_FAILURE_LEDGER:-${TMPDIR:-/tmp}/r40-v11-formal-failure.json}
failure_ledger(){ code=$?; if [[ $code -ne 0 && ! -e "$FAILURE_LEDGER" ]]; then python3 - "$FAILURE_LEDGER" "$code" <<'PY'
import json,sys
with open(sys.argv[1],"x") as stream:json.dump({"schema_version":"forkaudit-r40-v11-launch-failure-v1","exit_code":int(sys.argv[2]),"science_accepted":False},stream,sort_keys=True,separators=(",",":"))
PY
fi; }
trap failure_ledger EXIT
: "${R40_H20_EXECUTION_AUTHORIZED:?authorization required}"
[[ "$R40_H20_EXECUTION_AUTHORIZED" == yes ]] || { echo "authorization must equal yes" >&2; exit 2; }
: "${R40_V11_FRESH_AUDIT_APPROVED:?fresh independent audit approval required}"
[[ "$R40_V11_FRESH_AUDIT_APPROVED" == yes ]] || { echo "fresh audit must approve v11 before formal execution" >&2; exit 2; }
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
EVIDENCE=$(cd -- "$HERE/.." && pwd)
MARKER=${R40_ONE_SHOT_MARKER:-${TMPDIR:-/tmp}/r40-v11-formal-launch-used}
[[ ! -e "$MARKER" ]] || { echo "one-shot marker exists" >&2; exit 2; }
(cd "$HERE" && sha256sum -c source-code.sha256)
TMP=$(mktemp "${TMPDIR:-/tmp}/r40-v11.XXXXXX"); trap 'rm -f "$TMP"' EXIT; rm -f "$TMP"
python3 "$HERE/scripts/build_formal_launcher.py" --v6 "$EVIDENCE/r39_primary_compiled_dispatch_v6/executed_source/r39_primary_formal_h20.sh" --output "$TMP"
bash -n "$TMP"
touch "$MARKER"
bash "$TMP"
trap - EXIT
