from __future__ import annotations

import ast
import hashlib
import json
import re
import tempfile
from pathlib import Path

from build_combined_launcher import EXPECTED_V6_LAUNCHER_SHA256, transform


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT.parent
V6 = EVIDENCE / "r39_primary_compiled_dispatch_v6"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    prereg = json.loads((ROOT / "preregistration.json").read_text())
    acceptance = json.loads((ROOT / "acceptance.json").read_text())
    launcher = V6 / "executed_source/r39_primary_formal_h20.sh"
    checks = {
        "v6_launcher_hash_pinned": sha(launcher) == EXPECTED_V6_LAUNCHER_SHA256,
        "v6_entrypoint_hash_pinned": sha(V6 / "executed_source/r39_primary_rank_entrypoint.py")
        == prereg["combination"]["v6_rank_entrypoint_sha256"],
        "formal_status_not_run": prereg["formal_gpu_execution"] == "not-run"
        and acceptance["formal_gpu_execution"] == "not-run"
        and acceptance["formal_evidence_eligible"] is False,
        "four_frozen_faults": len(prereg["faults"]) == 4,
        "six_frozen_coordinates": len(prereg["selected_coordinates"]) == 6,
        "wire_forbids_manifest_and_slot_id": {"manifest", "slot_manifest", "slot_id"}
        <= set(prereg["registration_wire"]["forbidden_fields"]),
    }
    for path in sorted((ROOT / "executed_source").glob("*.py")):
        ast.parse(path.read_text(), filename=str(path))
    for path in sorted((ROOT / "scripts").glob("*.py")):
        ast.parse(path.read_text(), filename=str(path))
    generated = transform(launcher.read_text())
    checks.update(
        {
            "combined_launcher_syntax_shape": generated.count("r40_combined_rank_entrypoint.py") == 1,
            "same_v6_scientific_launcher_body": "PG19_DATA" in generated and "MODEL_DIR" in generated,
            "v2_finalize_in_terminal_flow": "r40_h20_finalize.py" in generated
            and "independent-live-binding live-binding-formal" in generated,
            "no_qs_commands": not any(
                re.search(r"(^|\s)qs\s", path.read_text(), re.MULTILINE)
                for folder in (ROOT / "executed_source", ROOT / "formal", ROOT / "scripts")
                for path in folder.glob("*") if path.is_file()
            ),
        }
    )
    failed = sorted(key for key, value in checks.items() if not value)
    result = {
        "schema_version": "forkaudit-r40-h20-local-static-audit-v1",
        "checks": checks,
        "passed": not failed,
        "failed_checks": failed,
        "generated_launcher_sha256": hashlib.sha256(generated.encode()).hexdigest(),
        "formal_gpu_execution": "not-run",
    }
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
