#!/usr/bin/env bash
set -euo pipefail
FAILURE_LEDGER=${R40_FAILURE_LEDGER:-${TMPDIR:-/tmp}/r40-v14-formal-failure.json}
TMP_DIR=""
on_exit(){ code=$?; if [[ -n "$TMP_DIR" && -d "$TMP_DIR" ]]; then rm -rf -- "$TMP_DIR"; fi; if [[ $code -ne 0 && ! -e "$FAILURE_LEDGER" ]]; then python3 - "$FAILURE_LEDGER" "$code" <<'PY'
import json,sys
with open(sys.argv[1],"x") as stream:json.dump({"schema_version":"forkaudit-r40-v14-launch-failure-v1","exit_code":int(sys.argv[2]),"science_accepted":False},stream,sort_keys=True,separators=(",",":"))
PY
fi; exit "$code"; }
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
[[ -n "${R40_H20_EXECUTION_AUTHORIZED:-}" && "$R40_H20_EXECUTION_AUTHORIZED" == yes ]] || { echo "authorization must be explicitly nonempty and equal yes" >&2; exit 2; }
[[ -n "${R40_V14_FRESH_AUDIT_APPROVED:-}" && "$R40_V14_FRESH_AUDIT_APPROVED" == yes ]] || { echo "fresh audit must be explicitly nonempty and equal yes for v14" >&2; exit 2; }
if [[ "${R40_LAUNCHER_HANDLER_SELFTEST:-}" == success ]]; then TMP_DIR=$(mktemp -d "${TMPDIR:-/tmp}/r40-v14-handler-success.XXXXXX"); exit 0; fi
if [[ "${R40_LAUNCHER_HANDLER_SELFTEST:-}" == failure ]]; then TMP_DIR=$(mktemp -d "${TMPDIR:-/tmp}/r40-v14-handler-failure.XXXXXX"); exit 7; fi
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
EVIDENCE=$(cd -- "$HERE/.." && pwd)
if [[ "${R40_LAUNCHER_ATOMIC_GATE_SELFTEST:-}" != yes ]]; then
  [[ "${R40_V14_APPROVED_SOURCE_LEDGER_SHA256:-}" =~ ^[0-9a-f]{64}$ ]] || { echo "external approved source-ledger SHA missing/invalid" >&2; exit 2; }
  [[ "${R40_V14_APPROVED_ARCHIVE_SHA256:-}" =~ ^[0-9a-f]{64}$ ]] || { echo "external approved archive SHA missing/invalid" >&2; exit 2; }
  [[ "$(sha256sum "$HERE/source-code.sha256" | awk '{print $1}')" == "$R40_V14_APPROVED_SOURCE_LEDGER_SHA256" ]] || { echo "operator-approved source ledger mismatch" >&2; exit 2; }
  TMP_DIR=$(mktemp -d "${TMPDIR:-/tmp}/r40-v14-approval.XXXXXX")
  python3 "$HERE/scripts/build_deterministic_archive.py" --root "$HERE" --output "$TMP_DIR/candidate.tar.gz"
  [[ "$(sha256sum "$TMP_DIR/candidate.tar.gz" | awk '{print $1}')" == "$R40_V14_APPROVED_ARCHIVE_SHA256" ]] || { echo "operator-approved archive mismatch" >&2; exit 2; }
fi
MARKER=${R40_ONE_SHOT_MARKER:-${TMPDIR:-/tmp}/r40-v14-formal-launch-used}
mkdir -- "$MARKER" 2>/dev/null || { echo "atomic one-shot ownership acquisition failed" >&2; exit 2; }
if [[ "${R40_LAUNCHER_ATOMIC_GATE_SELFTEST:-}" == yes ]]; then sleep 0.2; [[ -n "${R40_SELFTEST_RESULT_ACTION:-}" ]]; (set -o noclobber; : > "$R40_SELFTEST_RESULT_ACTION"); exit 0; fi
(cd "$HERE" && sha256sum -c source-code.sha256)
if [[ -z "$TMP_DIR" ]]; then TMP_DIR=$(mktemp -d "${TMPDIR:-/tmp}/r40-v14.XXXXXX"); fi; TMP="$TMP_DIR/launcher.sh"
python3 "$HERE/scripts/build_formal_launcher.py" --v6 "$EVIDENCE/r39_primary_compiled_dispatch_v6/executed_source/r39_primary_formal_h20.sh" --output "$TMP"
bash -n "$TMP"
bash "$TMP"
