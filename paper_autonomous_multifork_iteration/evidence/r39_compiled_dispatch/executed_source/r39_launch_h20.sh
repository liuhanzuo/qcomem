#!/usr/bin/env bash
set -euo pipefail

# This launcher is deliberately a wrapper around an already frozen, reviewed
# model entry point.  It creates no QS resources.  The hook wrapper is always
# the Python process that executes the entry point, so a caller cannot
# accidentally omit the interception step.

R39_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:?set PYTHON to the frozen H20 environment interpreter}
CODE_ROOT=${CODE_ROOT:?set CODE_ROOT to the immutable audited adapter snapshot}
RUNTIME_ROOT=${RUNTIME_ROOT:?set RUNTIME_ROOT to the frozen Python site-packages root}
ENTRYPOINT_ROOT=${ENTRYPOINT_ROOT:?set ENTRYPOINT_ROOT to the immutable reviewed entrypoint package}
ENTRYPOINT_RELATIVE=${ENTRYPOINT_RELATIVE:?set ENTRYPOINT_RELATIVE inside ENTRYPOINT_ROOT}
ENTRYPOINT_ARGS_JSON=${ENTRYPOINT_ARGS_JSON:?set ENTRYPOINT_ARGS_JSON to a JSON list of reviewed entrypoint arguments}
OUTPUT_ROOT=${OUTPUT_ROOT:?set OUTPUT_ROOT to a new empty Round-39 output directory}
R29_RESULT=${R29_RESULT:?set R29_RESULT to the result path written by the reviewed entrypoint}
R29_SEMANTIC_SIDECAR=${R29_SEMANTIC_SIDECAR:?set R29_SEMANTIC_SIDECAR to the sidecar path written by the reviewed entrypoint}
DESIGN_PREREGISTRATION=${DESIGN_PREREGISTRATION:?set DESIGN_PREREGISTRATION to the frozen R29 design}

if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "OUTPUT_ROOT must be absent" >&2
  exit 2
fi
mkdir -p "$OUTPUT_ROOT"
CODE_ROOT=$(realpath "$CODE_ROOT")
RUNTIME_ROOT=$(realpath "$RUNTIME_ROOT")
ENTRYPOINT_ROOT=$(realpath "$ENTRYPOINT_ROOT")
DESIGN_PREREGISTRATION=$(realpath "$DESIGN_PREREGISTRATION")
ENTRYPOINT="$ENTRYPOINT_ROOT/$ENTRYPOINT_RELATIVE"
case "$(realpath "$ENTRYPOINT")" in
  "$ENTRYPOINT_ROOT"/*) ;;
  *) echo "ENTRYPOINT_RELATIVE escapes ENTRYPOINT_ROOT" >&2; exit 2 ;;
esac
if [[ ! -f "$ENTRYPOINT" ]]; then
  echo "entry point is not a regular file" >&2
  exit 2
fi
TRITON_CACHE_ROOT="$OUTPUT_ROOT/runtime-cache/triton"
RECEIPT="$OUTPUT_ROOT/raw/compiled-dispatch-receipt.json"
mkdir -p "$TRITON_CACHE_ROOT" "$OUTPUT_ROOT/raw"
cp "$ENTRYPOINT_ARGS_JSON" "$OUTPUT_ROOT/entrypoint-args.json"
(cd "$R39_ROOT" && sha256sum -c source-code.sha256) \
  > "$OUTPUT_ROOT/r39-source-code-check.log"
cp "$R39_ROOT/source-code.sha256" "$OUTPUT_ROOT/r39-source-code.sha256"

# The cache directory is run-local and held fixed, so replay binds the exact
# selected cubin/PTX bundle.  Do not use the historical shared Triton cache.
export TRITON_CACHE_DIR="$TRITON_CACHE_ROOT"
run_isolated_script() {
  local script=$1
  shift
  "$PYTHON" -I -B -c \
    'import runpy,sys; source=sys.argv.pop(1); script=sys.argv.pop(1); sys.path.insert(0,source); sys.argv[0]=script; runpy.run_path(script,run_name="__main__")' \
    "$R39_ROOT/executed_source" "$script" "$@"
}

run_isolated_script "$R39_ROOT/executed_source/r39_hooked_entrypoint.py" \
  --code-root "$CODE_ROOT" \
  --runtime-root "$RUNTIME_ROOT" \
  --entrypoint-root "$ENTRYPOINT_ROOT" \
  --triton-cache-root "$TRITON_CACHE_ROOT" \
  --receipt "$RECEIPT" \
  --entrypoint "$ENTRYPOINT" \
  --entrypoint-args-json "$OUTPUT_ROOT/entrypoint-args.json"

# Snapshot only the source files named by the emitted receipt. Detached replay
# and controls below use these run-local copies, not mutable live installations.
run_isolated_script "$R39_ROOT/executed_source/r39_compiled_dispatch_receipts.py" snapshot-sources \
  --receipt "$RECEIPT" \
  --code-root "$CODE_ROOT" \
  --runtime-root "$RUNTIME_ROOT" \
  --target "$OUTPUT_ROOT/source-snapshot" \
  --output "$OUTPUT_ROOT/source-snapshot-manifest.json"
SNAPSHOT_CODE_ROOT="$OUTPUT_ROOT/source-snapshot/code"
SNAPSHOT_RUNTIME_ROOT="$OUTPUT_ROOT/source-snapshot/runtime"

run_isolated_script "$R39_ROOT/executed_source/r39_compiled_dispatch_receipts.py" replay \
  --receipt "$RECEIPT" \
  --triton-cache-root "$TRITON_CACHE_ROOT" \
  --code-root "$SNAPSHOT_CODE_ROOT" \
  --runtime-root "$SNAPSHOT_RUNTIME_ROOT" \
  --output "$OUTPUT_ROOT/replay.json"

[[ -f "$R29_RESULT" ]] || { echo "R29 formal result was not written" >&2; exit 2; }
[[ -f "$R29_SEMANTIC_SIDECAR" ]] || { echo "R29 semantic sidecar was not written" >&2; exit 2; }

run_isolated_script "$R39_ROOT/executed_source/r39_compiled_dispatch_receipts.py" bound-negative-controls \
  --receipt "$RECEIPT" \
  --triton-cache-root "$TRITON_CACHE_ROOT" \
  --code-root "$SNAPSHOT_CODE_ROOT" \
  --runtime-root "$SNAPSHOT_RUNTIME_ROOT" \
  --output "$OUTPUT_ROOT/negative-controls.json"

run_isolated_script "$R39_ROOT/executed_source/r39_verify_formal_binding.py" \
  --r29-result "$R29_RESULT" \
  --semantic-sidecar "$R29_SEMANTIC_SIDECAR" \
  --design-preregistration "$DESIGN_PREREGISTRATION" \
  --receipt "$RECEIPT" \
  --replay "$OUTPUT_ROOT/replay.json" \
  --negative-controls "$OUTPUT_ROOT/negative-controls.json" \
  --triton-cache-root "$TRITON_CACHE_ROOT" \
  --code-root "$SNAPSHOT_CODE_ROOT" \
  --runtime-root "$SNAPSHOT_RUNTIME_ROOT" \
  --output "$OUTPUT_ROOT/formal-aggregate.json"

(
  cd "$OUTPUT_ROOT"
  {
    find raw runtime-cache source-snapshot -type f -print0
    printf '%s\0' entrypoint-args.json replay.json negative-controls.json \
      formal-aggregate.json source-snapshot-manifest.json \
      r39-source-code-check.log r39-source-code.sha256
  } | sort -z | xargs -0 sha256sum > terminal-files.sha256
)
touch "$OUTPUT_ROOT/COMPLETE"
