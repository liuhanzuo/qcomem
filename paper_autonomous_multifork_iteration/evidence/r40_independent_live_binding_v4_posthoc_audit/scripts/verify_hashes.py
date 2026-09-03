from __future__ import annotations

"""Read-only terminal verification for the frozen posthoc audit package."""

import hashlib
import json
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[1]
V4_ROOT = ROOT.parent / "r40_independent_live_binding_v4_real_binder"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_ledger(path: Path, root: Path, *, exact_count: int, allowed_root: Path) -> List[str]:
    checked: List[str] = []
    for line_number, line in enumerate(path.read_text(encoding="ascii").splitlines(), 1):
        if not line.strip():
            continue
        expected, relative = line.split(None, 1)
        relative = relative.strip()
        target = (root / relative).resolve()
        try:
            target.relative_to(allowed_root.resolve())
        except ValueError as exc:
            raise RuntimeError(f"ledger path escapes allowed root at {path.name}:{line_number}") from exc
        observed = sha256(target)
        if observed != expected:
            raise RuntimeError(
                f"hash mismatch for {relative}: expected {expected}, observed {observed}"
            )
        checked.append(relative)
    if len(checked) != exact_count or len(set(checked)) != exact_count:
        raise RuntimeError(
            f"{path.name} expected {exact_count} unique rows, observed {len(checked)}"
        )
    return checked


def main() -> int:
    v4_files = verify_ledger(
        ROOT / "V4_SOURCE_FILES.sha256",
        ROOT,
        exact_count=10,
        allowed_root=V4_ROOT,
    )
    receipt = json.loads((ROOT / "TERMINAL_RECEIPT.json").read_text(encoding="utf-8"))
    package_files = verify_ledger(
        ROOT / "SHA256SUMS",
        ROOT,
        exact_count=int(receipt["package_payload_file_count"]),
        allowed_root=ROOT,
    )
    expected = {
        "sha256sums_sha256": sha256(ROOT / "SHA256SUMS"),
        "v4_external_binding_sha256": sha256(ROOT / "V4_SOURCE_FILES.sha256"),
        "v4_source_ledger_sha256": sha256(V4_ROOT / "source-code.sha256"),
        "probe_stdout_sha256": sha256(ROOT / "raw" / "probe.stdout.txt"),
        "probe_json_sha256": sha256(ROOT / "raw" / "probe.results.json"),
        "test_stdout_sha256": sha256(ROOT / "raw" / "tests.stdout.txt"),
    }
    for key, observed in expected.items():
        if receipt.get(key) != observed:
            raise RuntimeError(
                f"terminal receipt mismatch for {key}: expected {receipt.get(key)}, observed {observed}"
            )
    hard = {
        "audit_kind": "posthoc-read-only-audit",
        "v4_formal_status": "HOLD",
        "formal_gpu_execution": "not-run",
        "qs_invocation": "not-run",
        "positive_paper_evidence": False,
        "all_counterexamples_accepted": True,
        "exact_file_inventory": True,
    }
    for key, value in hard.items():
        if receipt.get(key) != value:
            raise RuntimeError(f"terminal receipt hard predicate failed: {key}")
    expected_inventory = {
        (ROOT / relative).resolve() for relative in package_files
    } | {
        (ROOT / "SHA256SUMS").resolve(),
        (ROOT / "TERMINAL_RECEIPT.json").resolve(),
    }
    observed_inventory = {
        path.resolve() for path in ROOT.rglob("*") if path.is_file()
    }
    if observed_inventory != expected_inventory:
        missing = sorted(str(path.relative_to(ROOT)) for path in expected_inventory - observed_inventory)
        extra = sorted(str(path.relative_to(ROOT)) for path in observed_inventory - expected_inventory)
        raise RuntimeError(f"terminal inventory drift: missing={missing}, extra={extra}")
    result: Dict[str, object] = {
        "schema_version": "forkaudit-r40-v4-posthoc-terminal-verify-v1",
        "passed": True,
        "v4_files_verified": len(v4_files),
        "package_payload_files_verified": len(package_files),
        **hard,
    }
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
