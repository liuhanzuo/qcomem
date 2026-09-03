#!/usr/bin/env bash
set -euo pipefail

FAILURE_LEDGER=${R40_FAILURE_LEDGER:-${TMPDIR:-/tmp}/r40-v20-formal-failure.json}
TMP_DIR=""
on_exit(){ code=$?; if [[ -n "$TMP_DIR" && -d "$TMP_DIR" ]]; then rm -rf -- "$TMP_DIR"; fi; if [[ $code -ne 0 && ! -e "$FAILURE_LEDGER" ]]; then python3 - "$FAILURE_LEDGER" "$code" <<'PY'
import json,sys
with open(sys.argv[1],"x") as stream:json.dump({"schema_version":"forkaudit-r40-v20-launch-failure-v1","status":"HOLD_PENDING_FRESH_AUDIT_AND_H20","exit_code":int(sys.argv[2]),"science_accepted":False},stream,sort_keys=True,separators=(",",":"))
PY
fi; exit "$code"; }
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

[[ -n "${R40_H20_EXECUTION_AUTHORIZED:-}" && "$R40_H20_EXECUTION_AUTHORIZED" == yes ]] || { echo "authorization must be explicitly nonempty and equal yes" >&2; exit 2; }
[[ -n "${R40_V20_FRESH_AUDIT_APPROVED:-}" && "$R40_V20_FRESH_AUDIT_APPROVED" == yes ]] || { echo "fresh audit must be explicitly nonempty and equal yes for v20" >&2; exit 2; }
if [[ "${R40_LAUNCHER_HANDLER_SELFTEST:-}" == success ]]; then TMP_DIR=$(mktemp -d "${TMPDIR:-/tmp}/r40-v20-handler-success.XXXXXX"); exit 0; fi
if [[ "${R40_LAUNCHER_HANDLER_SELFTEST:-}" == failure ]]; then TMP_DIR=$(mktemp -d "${TMPDIR:-/tmp}/r40-v20-handler-failure.XXXXXX"); exit 7; fi

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
EVIDENCE=$(cd -- "$HERE/.." && pwd)
STAGE_ROOT=$(cd -- "$HERE/../../.." && pwd)
EXPECTED_STAGE_ROOT=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench/qcomem_r40_v20_clean_20260828a

if [[ "${R40_LAUNCHER_ATOMIC_GATE_SELFTEST:-}" != yes ]]; then
  [[ "$STAGE_ROOT" == "$EXPECTED_STAGE_ROOT" ]] || { echo "v20 exact fixed stage root mismatch" >&2; exit 2; }
  [[ "${R40_V20_APPROVED_SOURCE_LEDGER_SHA256:-}" =~ ^[0-9a-f]{64}$ ]] || { echo "external approved v20 source-ledger SHA missing/invalid" >&2; exit 2; }
  [[ "${R40_V20_APPROVED_ARCHIVE_SHA256:-}" =~ ^[0-9a-f]{64}$ ]] || { echo "external approved v20 overlay archive SHA missing/invalid" >&2; exit 2; }
  [[ "${R40_V20_APPROVED_V6_ARCHIVE_SHA256:-}" == 306daba7b79b045a306f0b22d6434143dd568cf1f3b6af7114ad1a4ebe1d6f82 ]] || { echo "operator-approved canonical v6 archive SHA mismatch" >&2; exit 2; }
  [[ -n "${R40_V20_CANONICAL_V6_ARCHIVE:-}" ]] || { echo "canonical v6 archive path missing" >&2; exit 2; }
  [[ -n "${R40_V20_OVERLAY_ARCHIVE:-}" ]] || { echo "v20 overlay archive path missing" >&2; exit 2; }
  [[ "$(sha256sum "$HERE/source-code.sha256" | awk '{print $1}')" == "$R40_V20_APPROVED_SOURCE_LEDGER_SHA256" ]] || { echo "operator-approved v20 source ledger mismatch" >&2; exit 2; }
  TMP_DIR=$(mktemp -d "${TMPDIR:-/tmp}/r40-v20-approval.XXXXXX")
  python3 -B "$HERE/scripts/build_deterministic_archive.py" --root "$HERE" --output-root "$TMP_DIR" --output "$TMP_DIR/candidate.tar.gz"
  [[ "$(sha256sum "$TMP_DIR/candidate.tar.gz" | awk '{print $1}')" == "$R40_V20_APPROVED_ARCHIVE_SHA256" ]] || { echo "operator-approved v20 overlay archive mismatch" >&2; exit 2; }
  python3 -B "$HERE/scripts/stage_v6_clean.py" verify \
    --stage-root "$STAGE_ROOT" \
    --v6-archive "$R40_V20_CANONICAL_V6_ARCHIVE" \
    --overlay-archive "$R40_V20_OVERLAY_ARCHIVE" \
    --clean-ledger "$HERE/v6-clean-members.json" \
    --exclusion-ledger "$HERE/v6-appledouble-exclusions.json" \
    --expected-v6-sha256 "$R40_V20_APPROVED_V6_ARCHIVE_SHA256" \
    --expected-overlay-sha256 "$R40_V20_APPROVED_ARCHIVE_SHA256"
  if find "$STAGE_ROOT" -name '._*' -print -quit | grep -q .; then echo "AppleDouble path remains before result action" >&2; exit 2; fi
fi

MARKER=${R40_ONE_SHOT_MARKER:-${TMPDIR:-/tmp}/r40-v20-formal-launch-used}
mkdir -- "$MARKER" 2>/dev/null || { echo "atomic one-shot ownership acquisition failed" >&2; exit 2; }
if [[ "${R40_LAUNCHER_ATOMIC_GATE_SELFTEST:-}" == yes ]]; then
  if [[ "${R40_SELFTEST_SIGNAL_HOLD:-}" == yes ]]; then while :; do :; done; fi
  [[ -n "${R40_SELFTEST_RESULT_ACTION:-}" ]]
  (set -o noclobber; : > "$R40_SELFTEST_RESULT_ACTION")
  exit 0
fi

(cd "$HERE" && sha256sum -c source-code.sha256)
if [[ -z "$TMP_DIR" ]]; then TMP_DIR=$(mktemp -d "${TMPDIR:-/tmp}/r40-v20.XXXXXX"); fi
TMP="$TMP_DIR/launcher.sh"
python3 -B "$HERE/scripts/build_formal_launcher.py" --v6 "$EVIDENCE/r39_primary_compiled_dispatch_v6/executed_source/r39_primary_formal_h20.sh" --output "$TMP"
bash -n "$TMP"
bash "$TMP"
