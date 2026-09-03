from __future__ import annotations

import argparse
import hashlib
import os
import stat
from pathlib import Path


EXPECTED = "299907b4f95e7f5d8873ef5d810698640cc525ec9d2af647d325465b150e69ee"


def read_regular_snapshot(path: Path) -> bytes:
    """Read one stable regular-file snapshot without following a final symlink."""
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("O_NOFOLLOW is required for the v6 launcher snapshot")
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(os.fspath(path), flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise RuntimeError("v6 launcher must be a singly linked regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            snapshot = stream.read()
        after = os.fstat(descriptor)
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_nlink,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if before_identity != after_identity or len(snapshot) != before.st_size:
            raise RuntimeError("v6 launcher changed during stable descriptor snapshot")
        return snapshot
    finally:
        os.close(descriptor)


def transform(s: str) -> str:
    s = s.replace(
        "qcomem_r39_primary_compiled_dispatch_20260827f",
        "qcomem_r40_v19_clean_20260828a",
    ).replace(
        "r39-primary-compiled-dispatch-20260827f",
        "r40-v19-clean-20260828a",
    )
    anchor = 'R39_SOURCE="$R39_PRIMARY_ROOT/executed_source"\n'
    insert = anchor + (
        'R40_ROOT="$EVIDENCE_ROOT/r40_independent_live_binding_v19_terminal_closure_fix"\n'
        'R40_CAPTURE_ROOT="$RESULT_ROOT/r40-clean-binding"\n'
        'R40_FORMAL_ROOT="$RESULT_ROOT/r40-formal"\n'
    )
    if s.count(anchor) != 1:
        raise RuntimeError("root anchor")
    s = s.replace(anchor, insert)
    wrapper = 'export R39_PRIMARY_RANK_WRAPPER="$R39_SOURCE/r39_primary_rank_entrypoint.py"\n'
    replacement = 'R40_SOURCE_LEDGER_SHA256=$(sha256sum "$R40_ROOT/source-code.sha256"|awk \'{print $1}\')\n(cd "$R40_ROOT" && sha256sum -c source-code.sha256)\nchmod -R a-w "$R40_ROOT"\nexport R39_PRIMARY_RANK_WRAPPER="$R40_ROOT/executed_source/r40_rank_entrypoint.py"\nexport R40_ROOT R40_CAPTURE_ROOT R40_SOURCE_LEDGER_SHA256\nexport R40_PREREG_SHA256=$(sha256sum "$R40_ROOT/preregistration.json"|awk \'{print $1}\')\nexport R40_BASE_ENTRYPOINT="$R39_SOURCE/r39_primary_rank_entrypoint.py"\nexport R40_BINDINGS_JSON=$("$REAL_PYTHON" - "$PRIMARY_CODE/run_qcomem_qwen35_forkaudit_review_revision.py" "$PRIMARY_CODE/launch_qcomem_qwen35_forkaudit_review_revision_8gpu.sh" "$PRIMARY_INPUTS/model-artifacts.formal.sha256" "$PRIMARY_INPUTS/model-weights.canonical.sha256" "$PG19_DATA" "$PG19_MANIFEST" "$PRIMARY_PROTOCOL_MANIFEST" "$R39_SOURCE/r39_primary_rank_entrypoint.py" "$R40_ROOT/source-code.sha256" <<\'PY\'\nimport hashlib,json,sys\nnames=["immutable_runner_sha256","immutable_launcher_sha256","model_artifact_ledger_sha256","model_weight_ledger_sha256","pg19_data_sha256","pg19_manifest_sha256","protocol_manifest_sha256","r39_v6_entrypoint_sha256","r40_source_ledger_sha256"]\nprint(json.dumps(dict(zip(names,[hashlib.sha256(open(p,"rb").read()).hexdigest() for p in sys.argv[1:]])),sort_keys=True,separators=(",",":")))\nPY\n)\n'
    if s.count(wrapper) != 1:
        raise RuntimeError("wrapper anchor")
    s = s.replace(wrapper, replacement)
    science = 'CODE_DIR="$PRIMARY_CODE" \\\n'
    smoke = 'mkdir -- "$R40_FORMAL_ROOT"\n"$REAL_PYTHON" -B "$R40_ROOT/executed_source/r40_formal_preflight.py" --root "$R40_ROOT" --repo-root "$STAGE_ROOT"\nPYTHONPATH="$R40_ROOT/executed_source:$PRIMARY_CODE" "$REAL_PYTHON" -B "$R40_ROOT/executed_source/r40_cuda_smoke.py" --root "$RESULT_ROOT" --runner "$PRIMARY_CODE/run_qcomem_qwen35_forkaudit_review_revision.py" --output "$R40_FORMAL_ROOT/cuda-smoke.json"\n[[ "$("$REAL_PYTHON" -c \'import json,sys; print(str(json.load(open(sys.argv[1]))["passed"]).lower())\' "$R40_FORMAL_ROOT/cuda-smoke.json")" == true ]]\n' + science
    if s.count(science) != 1:
        raise RuntimeError("science anchor")
    s = s.replace(science, smoke)
    terminal = '(\n  cd "$RESULT_ROOT"\n  find preflight primary compiled-dispatch-capture formal-binding -type f \\\n'
    final = '"$REAL_PYTHON" -B "$R40_ROOT/executed_source/r40_finalize.py" --terminal-root "$RESULT_ROOT" --capture-root "$R40_CAPTURE_ROOT" --preregistration "$R40_ROOT/preregistration.json" --expected-prereg-sha256 "$R40_PREREG_SHA256" --output "$R40_FORMAL_ROOT/aggregate.json"\n[[ "$(sha256sum "$R40_ROOT/source-code.sha256"|awk \'{print $1}\')" == "$R40_SOURCE_LEDGER_SHA256" ]]\n(cd "$R40_ROOT" && sha256sum -c source-code.sha256)\n(\n  cd "$RESULT_ROOT"\n  find preflight primary compiled-dispatch-capture formal-binding r40-clean-binding r40-formal -type f \\\n'
    if s.count(terminal) != 1:
        raise RuntimeError("terminal anchor")
    s = s.replace(terminal, final)
    old = '''(
  cd "$RESULT_ROOT"
  find preflight primary compiled-dispatch-capture formal-binding r40-clean-binding r40-formal -type f \\
    ! -path '*/terminal-files.sha256' ! -name COMPLETE -print0 | \\
    sort -z | xargs -0 sha256sum > terminal-files.sha256
)
touch "$RESULT_ROOT/COMPLETE"
'''
    new = '''R40_EXPECTED_PATH_ARGS=()
while IFS= read -r R40_EXPECTED_PATH; do
  [[ -n "$R40_EXPECTED_PATH" ]] || { echo "empty formal expected path" >&2; exit 2; }
  R40_EXPECTED_PATH_ARGS+=(--expected-existing-path "$R40_EXPECTED_PATH")
done < <("$REAL_PYTHON" -B "$R40_ROOT/executed_source/r40_tree_closure.py" expected-paths \\
  --root "$RESULT_ROOT")
[[ "${#R40_EXPECTED_PATH_ARGS[@]}" -gt 0 ]] || { echo "formal exact expected path whitelist absent" >&2; exit 2; }
"$REAL_PYTHON" -B "$R40_ROOT/executed_source/r40_tree_closure.py" prepare \\
  --root "$RESULT_ROOT" \\
  --output "$R40_FORMAL_ROOT/terminal-closure.json" \\
  --terminal-tree-output "$R40_FORMAL_ROOT/terminal-tree.json" \\
  --complete-output "$RESULT_ROOT/COMPLETE" \\
  --source-ledger-sha256 "$R40_SOURCE_LEDGER_SHA256" \\
  --profile r40-v16-formal \\
  "${R40_EXPECTED_PATH_ARGS[@]}"
"$REAL_PYTHON" -B "$R40_ROOT/executed_source/r40_tree_closure.py" complete \\
  --root "$RESULT_ROOT" --output "$RESULT_ROOT/COMPLETE"
"$REAL_PYTHON" -B "$R40_ROOT/executed_source/r40_tree_closure.py" close \\
  --root "$RESULT_ROOT" \\
  --output "$R40_FORMAL_ROOT/terminal-tree.json" \\
  --expectation "$R40_FORMAL_ROOT/terminal-closure.json"
'''
    if s.count(old) != 1:
        raise RuntimeError("terminal lexical closure anchor")
    return s.replace(old, new)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v6", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    snapshot = read_regular_snapshot(args.v6)
    if hashlib.sha256(snapshot).hexdigest() != EXPECTED:
        raise RuntimeError("v6 drift")
    source = snapshot.decode("utf-8", errors="strict")
    value = transform(source)
    with args.output.open("x", encoding="utf-8", errors="strict") as stream:
        stream.write(value)
    print(hashlib.sha256(value.encode("utf-8")).hexdigest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
