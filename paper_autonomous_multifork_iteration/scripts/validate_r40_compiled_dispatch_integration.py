#!/usr/bin/env python3
"""Read-only integrity check for the bounded R40 dispatch result.

This checker validates only artifacts available in the local repository.  It
does not re-execute the H20 run, duplicate the remote model view, or establish
the identity/independence of the reported post-run auditor.
"""

from __future__ import annotations

import hashlib
import json
import re
import tarfile
from pathlib import Path


PAPER = Path(__file__).resolve().parents[1]
SOURCE = PAPER / "evidence/r40_primary_compiled_dispatch_v11"
MIRROR = PAPER / "evidence/r40_compiled_v11_postrun_audit_mirror"

EXPECTED = {
    SOURCE / "packages/r40-primary-compiled-dispatch-v11-20260827k.tar.gz":
        "0013e1e458711263342b37c1a274b6a36d227a602a885201f12892a8968b3641",
    SOURCE / "source-code.sha256":
        "958e795ef473d87cd9addfc2924cb20c50df2434c0a66069d08ed2ad0d4c08a3",
    MIRROR / "r40-compiled-v11-postrun-minimal-mirror.tar.gz":
        "c9ef02c21ce782bef65dde1ad76fd18e8fda233e7d97d6fd20ea22428c99929d",
    MIRROR / "formal-binding/formal-aggregate.json":
        "04b5ae63dc2f2dbe7c116a7136c2cdda2d9cab2e433b72b31d57cd28125c7a1f",
    MIRROR / "primary/forkaudit-summary.json":
        "5221d9ae0eb12092e311929fed6269122c290baddc97b6014a69f0266e634353",
    MIRROR / "preflight/runtime-preflight.json":
        "e4467acfbf440fff5b9a4c4ca99b46a282edea79f31d9cd44ef5d90036991651",
    MIRROR / "terminal-files.sha256":
        "909d47d38ba3e37f196ceca340b4a0d2e40bbe6b8c494f63bc78286ca217fa5d",
    MIRROR / "formal-binding/terminal-files.sha256":
        "b01d76704b4155d826ebc21fdce8abe85a9ed8ce9aac64c9308feecf49b4e525",
    MIRROR / "primary/receipts/all-raw-artifacts.sha256":
        "cc8a39aedd87ee196dd6424db5403c3b5ac7cc2b86c68b089dfa730989b780de",
    MIRROR / "primary/scientific-artifacts.sha256":
        "50097b75ea925cc4ef7b6393113e10bcdf78d1508573dac218652d0270cc4758",
}

MIRROR_MEMBERS = {
    "COMPLETE",
    "formal-binding/COMPLETE",
    "formal-binding/formal-aggregate.json",
    "formal-binding/terminal-files.sha256",
    "preflight/runtime-preflight.json",
    "primary/forkaudit-summary.json",
    "primary/logs/aggregate-final-audit.json",
    "primary/receipts/all-raw-artifacts.sha256",
    "primary/scientific-artifacts.sha256",
    "terminal-files.sha256",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ledger_rows(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        require(match is not None, f"malformed ledger row {path}:{number}")
        rows.append((match.group(1), match.group(2)))
    require(rows and len({name for _, name in rows}) == len(rows),
            f"empty or duplicate ledger paths: {path}")
    return rows


def verify_source_ledger() -> int:
    rows = ledger_rows(SOURCE / "source-code.sha256")
    for expected, raw_name in rows:
        relative = Path(raw_name.removeprefix("./"))
        require(not relative.is_absolute() and ".." not in relative.parts,
                f"unsafe source-ledger path: {raw_name}")
        path = SOURCE / relative
        require(path.exists(), f"source-ledger entry missing: {relative}")
        require(sha256_file(path) == expected,
                f"source-ledger hash drift: {relative}")
    return len(rows)


def verify_minimal_mirror_archive() -> None:
    archive = MIRROR / "r40-compiled-v11-postrun-minimal-mirror.tar.gz"
    with tarfile.open(archive, "r:gz") as handle:
        members = {item.name for item in handle.getmembers() if item.isfile()}
        require(members == MIRROR_MEMBERS,
                f"minimal-mirror member drift: {sorted(members ^ MIRROR_MEMBERS)}")
        for name in sorted(members):
            extracted = handle.extractfile(name)
            require(extracted is not None, f"cannot read archive member: {name}")
            require(extracted.read() == (MIRROR / name).read_bytes(),
                    f"archive member differs from mirror: {name}")


def main() -> None:
    for path, expected in EXPECTED.items():
        require(path.is_file(), f"missing required artifact: {path}")
        require(sha256_file(path) == expected,
                f"artifact hash drift: {path.relative_to(PAPER)}")

    source_ledger_entries = verify_source_ledger()
    require(source_ledger_entries == 46,
            f"source-ledger count drift: {source_ledger_entries}")
    verify_minimal_mirror_archive()

    formal = json.loads((MIRROR / "formal-binding/formal-aggregate.json").read_text())
    require(formal.get("status") == "pass", "formal aggregate is not pass")
    require(formal.get("formal_evidence_eligible") is True,
            "formal aggregate is not evidence eligible")
    require(formal.get("target_5_status_at_declared_scope") == "pass",
            "target 5 does not pass at declared scope")
    totals = formal.get("totals", {})
    expected_totals = {
        "attention_call_count": 209_920,
        "gdn_call_count": 635_520,
        "gdn_document_prefill_call_count": 5_760,
        "gdn_request_call_count": 629_760,
        "primary_configuration_count": 96,
        "primary_execution_cell_count": 192,
        "rank_count": 8,
    }
    for key, value in expected_totals.items():
        require(totals.get(key) == value, f"formal total drift: {key}")
    require(len(totals.get("distinct_compiled_artifact_ids", [])) == 8,
            "compiled artifact-ID count drift")
    require(len(totals.get("distinct_selected_compile_configurations", [])) == 1,
            "selected compile-configuration count drift")

    rank_rows = formal.get("rank_replays", [])
    require(len(rank_rows) == 8, "rank-replay count drift")
    require({row.get("rank") for row in rank_rows} == set(range(8)),
            "rank identities are incomplete")
    require(all(row.get("replay", {}).get("replay_verdict") == "pass"
                for row in rank_rows), "a rank replay did not pass")
    require(all(row.get("negative_controls", {}).get("all_rejected") is True
                for row in rank_rows), "a rank negative-control suite did not close")
    require(all(len(row.get("negative_controls", {}).get("negative_controls", {})) == 28
                for row in rank_rows), "negative-control count drift")

    root_hashes = {digest for digest, _ in ledger_rows(MIRROR / "terminal-files.sha256")}
    formal_hashes = {
        digest for digest, _ in ledger_rows(MIRROR / "formal-binding/terminal-files.sha256")
    }
    raw_hashes = {
        digest for digest, _ in ledger_rows(
            MIRROR / "primary/receipts/all-raw-artifacts.sha256"
        )
    }
    receipt_hashes = {row.get("receipt_sha256") for row in rank_rows}
    shard_hashes = {row.get("primary_shard_sha256") for row in rank_rows}
    require(receipt_hashes <= root_hashes and receipt_hashes <= formal_hashes,
            "rank receipts are not closed by both terminal ledgers")
    require(shard_hashes <= raw_hashes,
            "primary shards are not closed by the raw ledger")

    primary = json.loads((MIRROR / "primary/forkaudit-summary.json").read_text())
    require(primary.get("passed") is True, "primary aggregate did not pass")
    require(primary.get("scientific_outcome") == "valid_positive",
            "primary aggregate is not a valid positive")
    require(primary.get("scientific_run_valid") is True,
            "primary scientific run is invalid")
    require(primary.get("negative_reasons") == [], "primary aggregate has negative reasons")

    old_acceptance = json.loads((SOURCE / "acceptance.json").read_text())
    require(old_acceptance.get("package_status") == "HOLD",
            "immutable pre-run acceptance history drifted")

    report = {
        "status": "pass_for_local_mirror_integration",
        "source_ledger_entries_verified": source_ledger_entries,
        "minimal_mirror_members_verified": len(MIRROR_MEMBERS),
        "rank_replays_passed": len(rank_rows),
        "negative_controls_rejected": 8 * 28,
        "target_5_status_at_declared_scope": "pass",
        "full_remote_raw_audit_reexecuted_locally": False,
        "standalone_postrun_auditor_identity_receipt_present": False,
        "device_side_completion_attested": False,
        "compiled_gdn_attested": False,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
