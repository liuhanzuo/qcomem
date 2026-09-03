#!/usr/bin/env bash
set -euo pipefail
FAILURE_LEDGER=${R40_FAILURE_LEDGER:-${TMPDIR:-/tmp}/r40-v13-formal-failure.json}
TMP_DIR=""
on_exit(){ code=$?; if [[ -n "$TMP_DIR" && -d "$TMP_DIR" ]]; then rm -rf -- "$TMP_DIR"; fi; if [[ $code -ne 0 && ! -e "$FAILURE_LEDGER" ]]; then python3 - "$FAILURE_LEDGER" "$code" <<'PY'
import json,sys
with open(sys.argv[1],"x") as stream:json.dump({"schema_version":"forkaudit-r40-v13-launch-failure-v1","exit_code":int(sys.argv[2]),"science_accepted":False},stream,sort_keys=True,separators=(",",":"))
PY
fi; exit "$code"; }
trap on_exit EXIT
if [[ "${R40_LAUNCHER_HANDLER_SELFTEST:-}" == success ]]; then TMP_DIR=$(mktemp -d "${TMPDIR:-/tmp}/r40-v13-handler-success.XXXXXX"); exit 0; fi
if [[ "${R40_LAUNCHER_HANDLER_SELFTEST:-}" == failure ]]; then TMP_DIR=$(mktemp -d "${TMPDIR:-/tmp}/r40-v13-handler-failure.XXXXXX"); exit 7; fi
[[ -n "${R40_H20_EXECUTION_AUTHORIZED:-}" && "$R40_H20_EXECUTION_AUTHORIZED" == yes ]] || { echo "authorization must be explicitly nonempty and equal yes" >&2; exit 2; }
[[ -n "${R40_V13_FRESH_AUDIT_APPROVED:-}" && "$R40_V13_FRESH_AUDIT_APPROVED" == yes ]] || { echo "fresh audit must be explicitly nonempty and equal yes for v13" >&2; exit 2; }
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
EVIDENCE=$(cd -- "$HERE/.." && pwd)
MARKER=${R40_ONE_SHOT_MARKER:-${TMPDIR:-/tmp}/r40-v13-formal-launch-used}
[[ ! -e "$MARKER" ]] || { echo "one-shot marker exists" >&2; exit 2; }
(cd "$HERE" && sha256sum -c source-code.sha256)
TMP_DIR=$(mktemp -d "${TMPDIR:-/tmp}/r40-v13.XXXXXX"); TMP="$TMP_DIR/launcher.sh"
python3 "$HERE/scripts/build_formal_launcher.py" --v6 "$EVIDENCE/r39_primary_compiled_dispatch_v6/executed_source/r39_primary_formal_h20.sh" --output "$TMP"
bash -n "$TMP"
touch "$MARKER"
bash "$TMP"
