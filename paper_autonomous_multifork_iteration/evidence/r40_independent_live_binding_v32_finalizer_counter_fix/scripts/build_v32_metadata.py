from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent / "r40_independent_live_binding_v31_borrowed_compact_descriptor_fix"
RELATIVES = (
    "preregistration.json",
    "absorbed-lineage.json",
    "executed_source/r40_compact_rebind_fix.py",
    "executed_source/r40_no_bytecode_python",
    "executed_source/r40_cuda_smoke.py",
    "executed_source/r40_finalize.py",
    "executed_source/r40_formal_preflight.py",
    "executed_source/r40_passive_clone_lineage.py",
    "executed_source/r40_rank_entrypoint.py",
    "executed_source/r40_real_binding.py",
    "executed_source/r40_real_binding_hook.py",
    "executed_source/r40_tree_closure.py",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def classification(relative: str, identical: bool) -> str:
    if identical:
        return "byte-identical"
    if relative == "executed_source/r40_finalize.py":
        return "v32-terminal-finalizer-counter-contract-fix"
    if relative in {"preregistration.json", "executed_source/r40_formal_preflight.py"}:
        return "v32-counter-contract-metadata-or-regression"
    return "v32-identity-only"


rows = []
for relative in RELATIVES:
    old, new = BASE / relative, ROOT / relative
    identical = old.read_bytes() == new.read_bytes()
    rows.append(
        {
            "path": relative,
            "classification": classification(relative, identical),
            "byte_identical": identical,
            "v31_sha256": sha(old),
            "v31_size": old.stat().st_size,
            "v32_sha256": sha(new),
            "v32_size": new.stat().st_size,
        }
    )

old_pre = json.loads((BASE / "preregistration.json").read_text())
new_pre = json.loads((ROOT / "preregistration.json").read_text())
for value in (old_pre, new_pre):
    value.pop("schema_version")
    value.pop("experiment_id")
    value.pop("claim_boundary")
new_pre["acceptance"].pop("rank_artifact_primary_memory_calls_observed")
if old_pre != new_pre:
    raise RuntimeError("V31-V32 scientific configuration drift")

manifest = "".join(f"{sha(ROOT / relative)}  {relative}\n" for relative in RELATIVES)
(ROOT / "v32-current-payload.sha256").write_text(manifest, encoding="ascii")

from build_formal_launcher import read_regular_snapshot, transform


v6 = ROOT.parent / "r39_primary_compiled_dispatch_v6/executed_source/r39_primary_formal_h20.sh"
launcher = hashlib.sha256(transform(read_regular_snapshot(v6).decode()).encode()).hexdigest()
historical = json.loads((BASE / "v30-v31-controlled-diff.json").read_text())
if (ROOT / "v31-current-payload.sha256").read_bytes() != (BASE / "v31-current-payload.sha256").read_bytes():
    raise RuntimeError("copied V31 payload manifest drift")

value = {
    "schema_version": "forkaudit-r40-v31-v32-controlled-diff-v1",
    "status": "HOLD_PENDING_FRESH_AUDIT_AND_H20",
    "base_package": BASE.name,
    "current_package": ROOT.name,
    "base_payload_manifest_sha256": sha(ROOT / "v31-current-payload.sha256"),
    "current_payload_manifest_sha256": sha(ROOT / "v32-current-payload.sha256"),
    "base_payload_file_count": len(rows),
    "current_payload_file_count": len(rows),
    "byte_identical_file_count": sum(row["byte_identical"] for row in rows),
    "controlled_change_file_count": sum(not row["byte_identical"] for row in rows),
    "scientific_configuration_unchanged_after_identity_and_counter_metadata_normalization": True,
    "sole_non_identity_runtime_change": "terminal-finalizer-rank-artifact-primary_memory_calls_observed-strict-integer-7-to-8",
    "immutable_external_runner_sha256": "9da619fc037e2c670b146d778fd9f4d5344212b7e525f3d3f26a077f79d67775",
    "immutable_external_builder_sha256": "546efd59e2833034bc2e24d4cc0e6077f5a408275e359af43cd96f7f71cad16e",
    "v31_generated_launcher_sha256": historical["v31_generated_launcher_sha256"],
    "v32_generated_launcher_sha256": launcher,
    "rows": rows,
    "payload_sha256": None,
}
value["payload_sha256"] = hashlib.sha256(
    json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
(ROOT / "v31-v32-controlled-diff.json").write_text(json.dumps(value, indent=2) + "\n")
print(launcher)
