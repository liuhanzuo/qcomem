#!/usr/bin/env bash
set -euo pipefail

# Deliberately inert without explicit future authorization. This script does
# not allocate resources; it only wraps an already-provisioned 8-rank shell.
: "${R40_H20_EXECUTION_AUTHORIZED:?set only after explicit authorization}"
[[ "$R40_H20_EXECUTION_AUTHORIZED" == "yes" ]] || { echo "authorization must equal yes" >&2; exit 2; }

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
R40_V2_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
EVIDENCE_ROOT=$(cd -- "$R40_V2_ROOT/.." && pwd)
V6_LAUNCHER="$EVIDENCE_ROOT/r39_primary_compiled_dispatch_v6/executed_source/r39_primary_formal_h20.sh"

(cd "$R40_V2_ROOT" && shasum -a 256 -c source-code.sha256)
export R40_V2_EXPECTED_PREREGISTRATION_SHA256="a1e7bb45817096980595c0afdebd839ccc1902a80cc4182ef68d1b80c2ea684e"
COMBINED=$(mktemp "${TMPDIR:-/tmp}/r40-combined-launcher.XXXXXX")
trap 'rm -f "$COMBINED"' EXIT
rm -f "$COMBINED"
python3 "$R40_V2_ROOT/scripts/build_combined_launcher.py" \
  --v6-launcher "$V6_LAUNCHER" --output "$COMBINED"
bash -n "$COMBINED"

# The inherited v6 launcher performs all environment, model, PG19, rank-count,
# immutable-runner, result-root and terminal-ledger gates.
exec bash "$COMBINED"
