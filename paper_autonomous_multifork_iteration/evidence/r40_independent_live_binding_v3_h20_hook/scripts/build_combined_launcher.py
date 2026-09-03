from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


EXPECTED_V6_LAUNCHER_SHA256 = "299907b4f95e7f5d8873ef5d810698640cc525ec9d2af647d325465b150e69ee"
OLD_RUN_ID = "qcomem_r39_primary_compiled_dispatch_20260827f"
OLD_RESULT_ID = "r39-primary-compiled-dispatch-20260827f"
NEW_RUN_ID = "qcomem_r40_h20_preserialization_20260827b"
NEW_RESULT_ID = "r40-h20-preserialization-hook-20260827b"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def transform(source: str) -> str:
    if source.count(OLD_RUN_ID) != 1 or source.count(OLD_RESULT_ID) != 1:
        raise RuntimeError("v6 run/result literal count drift")
    result = source.replace(OLD_RUN_ID, NEW_RUN_ID).replace(OLD_RESULT_ID, NEW_RESULT_ID)
    anchor = 'R39_SOURCE="$R39_PRIMARY_ROOT/executed_source"\n'
    insertion = (
        anchor
        + 'R40_V3_ROOT="$EVIDENCE_ROOT/r40_independent_live_binding_v3_h20_hook"\n'
        + 'R40_V3_SOURCE="$R40_V3_ROOT/executed_source"\n'
        + 'R40_V3_CAPTURE_ROOT="$RESULT_ROOT/independent-live-binding"\n'
        + 'R40_V3_FORMAL="$RESULT_ROOT/live-binding-formal"\n'
    )
    if result.count(anchor) != 1:
        raise RuntimeError("v6 source anchor drift")
    result = result.replace(anchor, insertion)
    wrapper = 'export R39_PRIMARY_RANK_WRAPPER="$R39_SOURCE/r39_primary_rank_entrypoint.py"\n'
    replacement = (
        'export R39_PRIMARY_RANK_WRAPPER="$R40_V3_SOURCE/r40_combined_rank_entrypoint.py"\n'
        + 'export R40_V3_V6_ENTRYPOINT="$R39_SOURCE/r39_primary_rank_entrypoint.py"\n'
        + 'export R40_V3_PREREGISTRATION="$R40_V3_ROOT/preregistration.json"\n'
        + 'export R40_V3_EXPECTED_PREREGISTRATION_SHA256="${R40_V3_EXPECTED_PREREGISTRATION_SHA256:?missing preregistration hash}"\n'
        + 'export R40_V3_CAPTURE_ROOT="$R40_V3_CAPTURE_ROOT"\n'
        + 'export R40_V3_SOURCE_ROOT="$R40_V3_SOURCE"\n'
    )
    if result.count(wrapper) != 1:
        raise RuntimeError("v6 wrapper export anchor drift")
    result = result.replace(wrapper, replacement)
    terminal_anchor = '(\n  cd "$RESULT_ROOT"\n  find preflight primary compiled-dispatch-capture formal-binding -type f \\\n'
    finalize = (
        'mkdir -p "$R40_V3_FORMAL"\n'
        + 'PYTHONPATH="$R40_V3_SOURCE" "$REAL_PYTHON" -B "$R40_V3_SOURCE/r40_h20_finalize.py" \\\n'
        + '  --capture-root "$R40_V3_CAPTURE_ROOT" \\\n'
        + '  --preregistration "$R40_V3_PREREGISTRATION" \\\n'
        + '  --expected-preregistration-sha256 "$R40_V3_EXPECTED_PREREGISTRATION_SHA256" \\\n'
        + '  --expected-ranks 8 \\\n'
        + '  --output "$R40_V3_FORMAL/aggregate.json"\n\n'
        + '(\n  cd "$RESULT_ROOT"\n  find preflight primary compiled-dispatch-capture formal-binding independent-live-binding live-binding-formal -type f \\\n'
    )
    if result.count(terminal_anchor) != 1:
        raise RuntimeError("v6 terminal-ledger anchor drift")
    result = result.replace(terminal_anchor, finalize)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v6-launcher", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sha256_file(args.v6_launcher) != EXPECTED_V6_LAUNCHER_SHA256:
        raise RuntimeError("v6 formal wrapper hash drift")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    value = transform(args.v6_launcher.read_text(encoding="utf-8"))
    args.output.write_text(value, encoding="utf-8")
    print(hashlib.sha256(value.encode()).hexdigest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
