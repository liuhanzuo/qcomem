#!/usr/bin/env python3
"""Build and transactionally publish the deterministic v9 method freeze."""

from __future__ import annotations

import json
import os
from pathlib import Path

from v9_guard import (
    FREEZE_TIMESTAMP,
    TRUST_ROOT_DOCUMENT_SHA256,
    TRUST_ROOT_ID,
    _read_regular,
    canonical_bytes,
    canonical_tar,
    deterministic_gzip,
    digest_bytes,
    snapshot_commitments,
)
from v9_runtime import ProtectedParent


ARCHIVE_NAME = "r40-method-v9-freeze-20260828a.tar.gz"
TEST_COUNT = 11
SOURCE_FILES = tuple(
    sorted(
        (
            "DESIGN.md",
            "COUNTEREXAMPLES.md",
            "OPERATOR_TRUST_ROOT.json",
            "README.md",
            "authorized_launcher_v9.py",
            "designer_snapshot/ATTESTATION_SCHEMA.json",
            "designer_snapshot/PUBLIC_CONTRACT.md",
            "designer_snapshot/README.md",
            "freeze_v9.py",
            "fixtures_v9.py",
            "operator-binding.template.json",
            "static_audit_v9.py",
            "test_v9.py",
            "torch_probe_v9.py",
            "v9_guard.py",
            "v9_runtime.py",
        )
    )
)


def build_outputs(source_root: os.PathLike[str] | str) -> tuple[dict[str, bytes], dict]:
    root = Path(source_root)
    source_entries = []
    rows = []
    for relative in SOURCE_FILES:
        raw, item_stat = _read_regular(root / relative)
        source_entries.append((relative, raw))
        rows.append({"path": relative, "sha256": digest_bytes(raw), "size": item_stat.st_size})
    ledger = canonical_bytes(
        {
            "files": rows,
            "freeze_timestamp": FREEZE_TIMESTAMP,
            "schema_version": "forkaudit-v9-source-ledger-v1",
        }
    )
    tar_raw = canonical_tar(source_entries + [("source-ledger.json", ledger)])
    archive = deterministic_gzip(tar_raw)
    snapshot_sha, snapshot_inventory_sha, _ = snapshot_commitments(root / "designer_snapshot")
    method = {
        "archive_members": len(source_entries) + 1,
        "archive_sha256": digest_bytes(archive),
        "designer_snapshot_inventory_sha256": snapshot_inventory_sha,
        "designer_snapshot_released": False,
        "designer_snapshot_sha256": snapshot_sha,
        "fault_set": None,
        "formal_config": None,
        "freeze_timestamp": FREEZE_TIMESTAMP,
        "gpu_execution": False,
        "operator_binding": None,
        "operator_trust_root_id": TRUST_ROOT_ID,
        "operator_trust_root_sha256": TRUST_ROOT_DOCUMENT_SHA256,
        "schema_version": "forkaudit-v9-method-freeze-v1",
        "source_ledger_sha256": digest_bytes(ledger),
        "status": "HOLD_PENDING_FRESH_AUDIT_AND_EXTERNAL_OPERATOR_BINDING",
        "tests": TEST_COUNT,
    }
    outputs = {
        "source-ledger.json": ledger,
        ARCHIVE_NAME: archive,
        "METHOD_FROZEN.json": canonical_bytes(method),
    }
    return outputs, method


def freeze_package(
    source_root: os.PathLike[str] | str, target_root: os.PathLike[str] | str
) -> dict:
    outputs, method = build_outputs(source_root)
    with ProtectedParent(target_root) as parent:
        parent.publish_many(outputs, mode=0o444)
    return method


def main() -> int:
    root = Path(__file__).resolve().parent
    method = freeze_package(root, root)
    print(json.dumps(method, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
