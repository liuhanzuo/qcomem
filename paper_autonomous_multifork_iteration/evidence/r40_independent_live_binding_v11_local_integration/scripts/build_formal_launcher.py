from __future__ import annotations
import argparse,hashlib
from pathlib import Path
EXPECTED="299907b4f95e7f5d8873ef5d810698640cc525ec9d2af647d325465b150e69ee"
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def transform(s):
    s=s.replace("qcomem_r39_primary_compiled_dispatch_20260827f","qcomem_r40_v11_clean_20260827a").replace("r39-primary-compiled-dispatch-20260827f","r40-v11-clean-20260827a")
    anchor='R39_SOURCE="$R39_PRIMARY_ROOT/executed_source"\n';insert=anchor+'R40_ROOT="$EVIDENCE_ROOT/r40_independent_live_binding_v11_local_integration"\nR40_CAPTURE_ROOT="$RESULT_ROOT/r40-clean-binding"\nR40_FORMAL_ROOT="$RESULT_ROOT/r40-formal"\n'
    if s.count(anchor)!=1: raise RuntimeError("root anchor")
    s=s.replace(anchor,insert)
    wrapper='export R39_PRIMARY_RANK_WRAPPER="$R39_SOURCE/r39_primary_rank_entrypoint.py"\n';replacement='R40_SOURCE_LEDGER_SHA256=$(sha256sum "$R40_ROOT/source-code.sha256"|awk \'{print $1}\')\n(cd "$R40_ROOT" && sha256sum -c source-code.sha256)\nchmod -R a-w "$R40_ROOT"\nexport R39_PRIMARY_RANK_WRAPPER="$R40_ROOT/executed_source/r40_rank_entrypoint.py"\nexport R40_ROOT R40_CAPTURE_ROOT R40_SOURCE_LEDGER_SHA256\nexport R40_PREREG_SHA256=$(sha256sum "$R40_ROOT/preregistration.json"|awk \'{print $1}\')\nexport R40_BASE_ENTRYPOINT="$R39_SOURCE/r39_primary_rank_entrypoint.py"\nexport R40_BINDINGS_JSON=$("$REAL_PYTHON" - "$PRIMARY_CODE/run_qcomem_qwen35_forkaudit_review_revision.py" "$PRIMARY_CODE/launch_qcomem_qwen35_forkaudit_review_revision_8gpu.sh" "$PRIMARY_INPUTS/model-artifacts.formal.sha256" "$PRIMARY_INPUTS/model-weights.canonical.sha256" "$PG19_DATA" "$PG19_MANIFEST" "$PRIMARY_PROTOCOL_MANIFEST" "$R39_SOURCE/r39_primary_rank_entrypoint.py" "$R40_ROOT/source-code.sha256" <<\'PY\'\nimport hashlib,json,sys\nnames=["immutable_runner_sha256","immutable_launcher_sha256","model_artifact_ledger_sha256","model_weight_ledger_sha256","pg19_data_sha256","pg19_manifest_sha256","protocol_manifest_sha256","r39_v6_entrypoint_sha256","r40_source_ledger_sha256"]\nprint(json.dumps(dict(zip(names,[hashlib.sha256(open(p,"rb").read()).hexdigest() for p in sys.argv[1:]])),sort_keys=True,separators=(",",":")))\nPY\n)\n'
    if s.count(wrapper)!=1: raise RuntimeError("wrapper anchor")
    s=s.replace(wrapper,replacement)
    science='CODE_DIR="$PRIMARY_CODE" \\\n';smoke='mkdir -p "$R40_FORMAL_ROOT"\nPYTHONPATH="$R40_ROOT/executed_source:$PRIMARY_CODE" "$REAL_PYTHON" -B "$R40_ROOT/executed_source/r40_cuda_smoke.py" --runner "$PRIMARY_CODE/run_qcomem_qwen35_forkaudit_review_revision.py" --output "$R40_FORMAL_ROOT/cuda-smoke.json"\n[[ "$("$REAL_PYTHON" -c \'import json,sys; print(str(json.load(open(sys.argv[1]))["passed"]).lower())\' "$R40_FORMAL_ROOT/cuda-smoke.json")" == true ]]\n'+science
    if s.count(science)!=1: raise RuntimeError("science anchor")
    s=s.replace(science,smoke)
    terminal='(\n  cd "$RESULT_ROOT"\n  find preflight primary compiled-dispatch-capture formal-binding -type f \\\n';final='"$REAL_PYTHON" -B "$R40_ROOT/executed_source/r40_finalize.py" --capture-root "$R40_CAPTURE_ROOT" --preregistration "$R40_ROOT/preregistration.json" --expected-prereg-sha256 "$R40_PREREG_SHA256" --output "$R40_FORMAL_ROOT/aggregate.json"\n[[ "$(sha256sum "$R40_ROOT/source-code.sha256"|awk \'{print $1}\')" == "$R40_SOURCE_LEDGER_SHA256" ]]\n(cd "$R40_ROOT" && sha256sum -c source-code.sha256)\nprintf \'%s\\n\' "{\\"terminal_source_rehash\\":true,\\"read_only_staging\\":true,\\"source_ledger_sha256\\":\\"$R40_SOURCE_LEDGER_SHA256\\"}" > "$R40_FORMAL_ROOT/terminal-closure.json"\n(\n  cd "$RESULT_ROOT"\n  find preflight primary compiled-dispatch-capture formal-binding r40-clean-binding r40-formal -type f \\\n'
    if s.count(terminal)!=1: raise RuntimeError("terminal anchor")
    return s.replace(terminal,final)
def main():
    p=argparse.ArgumentParser();p.add_argument("--v6",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    if sha(a.v6)!=EXPECTED:raise RuntimeError("v6 drift")
    if a.output.exists():raise FileExistsError("overwrite")
    value=transform(a.v6.read_text());a.output.write_text(value);print(hashlib.sha256(value.encode()).hexdigest())
if __name__=="__main__":main()
