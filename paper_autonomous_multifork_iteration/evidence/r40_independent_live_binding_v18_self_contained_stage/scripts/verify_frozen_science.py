from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import stat
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
STATUS = "HOLD_PENDING_FRESH_AUDIT_AND_H20"
PACKAGE_NAME = "r40_independent_live_binding_v18_self_contained_stage"
MANIFEST_NAME = "v16-scientific-payload.sha256"
EQUIVALENCE_NAME = "v16-v18-scientific-equivalence.json"
EXPECTED_MANIFEST_SHA256 = "ae5b7e404bb8e3fd004e69df716c4a80c29f7928230210dd9798356bb0efa59a"
EXPECTED_EQUIVALENCE_SHA256 = "6c8775053b745dac835026b4435f1ac3edcce6e65cfc628bd63276f6c0ff12d1"
EXPECTED_EQUIVALENCE_PAYLOAD_SHA256 = "6cb91c0d305987191f1969b0338102be497fdb51bfcd870605dc7bd8967be81a"
EXPECTED_V6_LAUNCHER_SHA256 = "299907b4f95e7f5d8873ef5d810698640cc525ec9d2af647d325465b150e69ee"
EXPECTED_V16_GENERATED_LAUNCHER_SHA256 = "dba428edb5030d930b8892747fdbf5f8ae79f0fc07b25605458bdff5c673c0b0"
EXPECTED_V18_GENERATED_LAUNCHER_SHA256 = "8d5ba77f9b61b760346334b4bca041e1ac0176719c5b8bd2e616a29b24226636"
EXPECTED_RUNNER_SHA256 = "9da619fc037e2c670b146d778fd9f4d5344212b7e525f3d3f26a077f79d67775"
EXPECTED_BUILDER_SHA256 = "546efd59e2833034bc2e24d4cc0e6077f5a408275e359af43cd96f7f71cad16e"
SCIENCE_RELATIVES = (
    "preregistration.json",
    "absorbed-lineage.json",
    "executed_source/r40_cuda_smoke.py",
    "executed_source/r40_finalize.py",
    "executed_source/r40_formal_preflight.py",
    "executed_source/r40_passive_clone_lineage.py",
    "executed_source/r40_rank_entrypoint.py",
    "executed_source/r40_real_binding.py",
    "executed_source/r40_real_binding_hook.py",
    "executed_source/r40_tree_closure.py",
)
EQUIVALENCE_FIELDS = {
    "all_scientific_payload_files_byte_identical",
    "generated_launcher_diff_is_only_stage_result_package_identifiers",
    "immutable_external_builder_sha256",
    "immutable_external_runner_sha256",
    "payload_sha256",
    "rows",
    "schema_version",
    "scientific_payload_file_count",
    "scientific_payload_manifest_sha256",
    "sibling_source_required",
    "status",
    "v16_generated_launcher_sha256",
    "v16_package",
    "v18_generated_launcher_sha256",
    "v18_package",
    "verification_mode",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stable_regular_bytes(path: Path, *, label: str) -> bytes:
    require(hasattr(os, "O_NOFOLLOW"), f"O_NOFOLLOW unavailable for {label}")
    descriptor = os.open(os.fspath(path), os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0))
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1, f"{label} must be a singly linked regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            data = stream.read()
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
        require(before_identity == after_identity and len(data) == before.st_size, f"{label} changed during exact read")
        return data
    finally:
        os.close(descriptor)


def reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_exact_json(data: bytes, *, label: str) -> dict[str, Any]:
    value = json.loads(data.decode("utf-8", errors="strict"), object_pairs_hook=reject_duplicate_pairs)
    require(type(value) is dict, f"{label} must be an object")
    return value


def verify_payload_seal(value: dict[str, Any]) -> None:
    supplied = value.get("payload_sha256")
    require(supplied == EXPECTED_EQUIVALENCE_PAYLOAD_SHA256, "scientific equivalence payload seal external pin drift")
    candidate = dict(value)
    candidate["payload_sha256"] = None
    expected = sha256(json.dumps(candidate, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    require(supplied == expected, "scientific equivalence payload seal mismatch")


def parse_manifest(data: bytes) -> list[tuple[str, str]]:
    require(sha256(data) == EXPECTED_MANIFEST_SHA256, "scientific manifest external pin drift")
    text = data.decode("ascii", errors="strict")
    require(text.endswith("\n") and "\r" not in text, "scientific manifest canonical line encoding drift")
    lines = text.splitlines()
    require(len(lines) == len(SCIENCE_RELATIVES), "scientific manifest file count drift")
    rows: list[tuple[str, str]] = []
    for index, line in enumerate(lines):
        pieces = line.split("  ", 1)
        require(len(pieces) == 2, "scientific manifest row format drift")
        digest, relative = pieces
        require(len(digest) == 64 and digest == digest.lower() and all(c in "0123456789abcdef" for c in digest), "scientific manifest digest format drift")
        require(relative == SCIENCE_RELATIVES[index], "scientific manifest exact path/order drift")
        rows.append((relative, digest))
    return rows


def load_launcher_builder(path: Path):
    specification = importlib.util.spec_from_file_location("r40_v18_launcher_builder", path)
    require(specification is not None and specification.loader is not None, "launcher builder import unavailable")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def verify_scientific_payload(*, root: Path = ROOT, repo_root: Path | None = None) -> dict[str, Any]:
    root = root.resolve(strict=True)
    require(root.name == PACKAGE_NAME, "v18 package root identity drift")
    repository = (repo_root if repo_root is not None else root.parents[2]).resolve(strict=True)
    manifest_bytes = stable_regular_bytes(root / MANIFEST_NAME, label="scientific manifest")
    manifest_rows = parse_manifest(manifest_bytes)
    equivalence_bytes = stable_regular_bytes(root / EQUIVALENCE_NAME, label="scientific equivalence")
    require(sha256(equivalence_bytes) == EXPECTED_EQUIVALENCE_SHA256, "scientific equivalence external file pin drift")
    equivalence = load_exact_json(equivalence_bytes, label="scientific equivalence")
    require(set(equivalence) == EQUIVALENCE_FIELDS, "scientific equivalence exact schema drift")
    verify_payload_seal(equivalence)
    require(equivalence["schema_version"] == "forkaudit-r40-v18-v16-scientific-equivalence-v1", "scientific equivalence schema identity drift")
    require(equivalence["status"] == STATUS, "scientific equivalence status drift")
    require(equivalence["v16_package"] == "r40_independent_live_binding_v16_local_integration", "frozen v16 package identity drift")
    require(equivalence["v18_package"] == PACKAGE_NAME, "v18 package identity drift")
    require(equivalence["verification_mode"] == "self-contained-frozen-pins-no-sibling", "scientific verification mode drift")
    require(equivalence["sibling_source_required"] is False, "scientific verification unexpectedly requires sibling source")
    require(equivalence["scientific_payload_manifest_sha256"] == EXPECTED_MANIFEST_SHA256, "equivalence manifest pin drift")
    require(equivalence["scientific_payload_file_count"] == len(SCIENCE_RELATIVES), "equivalence file count drift")
    require(equivalence["all_scientific_payload_files_byte_identical"] is True, "frozen scientific identity assertion drift")
    require(equivalence["immutable_external_runner_sha256"] == EXPECTED_RUNNER_SHA256, "immutable runner pin drift")
    require(equivalence["immutable_external_builder_sha256"] == EXPECTED_BUILDER_SHA256, "immutable builder pin drift")
    rows = equivalence["rows"]
    require(type(rows) is list and len(rows) == len(SCIENCE_RELATIVES), "scientific equivalence rows drift")
    for index, ((relative, manifest_digest), row) in enumerate(zip(manifest_rows, rows)):
        require(type(row) is dict and set(row) == {"path", "sha256", "size", "byte_identical"}, "scientific equivalence row schema drift")
        require(row["path"] == SCIENCE_RELATIVES[index] == relative, "scientific equivalence row path/order drift")
        require(row["sha256"] == manifest_digest and row["byte_identical"] is True, "scientific equivalence row identity drift")
        current = stable_regular_bytes(root / relative, label=f"scientific payload {relative}")
        require(sha256(current) == manifest_digest and len(current) == row["size"], f"current scientific payload hash/size drift: {relative}")
    current_python_paths = tuple(
        f"executed_source/{path.name}" for path in sorted((root / "executed_source").glob("*.py"))
    )
    require(current_python_paths == SCIENCE_RELATIVES[2:], "scientific executed-source exact file set drift")
    v6_launcher = repository / "paper_autonomous_multifork_iteration/evidence/r39_primary_compiled_dispatch_v6/executed_source/r39_primary_formal_h20.sh"
    builder = load_launcher_builder(root / "scripts/build_formal_launcher.py")
    v6_snapshot = builder.read_regular_snapshot(v6_launcher)
    require(sha256(v6_snapshot) == EXPECTED_V6_LAUNCHER_SHA256, "v6 launcher external pin drift")
    generated = builder.transform(v6_snapshot.decode("utf-8", errors="strict"))
    generated_sha256 = sha256(generated.encode("utf-8"))
    require(generated_sha256 == EXPECTED_V18_GENERATED_LAUNCHER_SHA256 == equivalence["v18_generated_launcher_sha256"], "v18 generated launcher pin drift")
    normalized = (
        generated.replace("qcomem_r40_v18_clean_20260828a", "qcomem_r40_v16_clean_20260827a")
        .replace("r40-v18-clean-20260828a", "r40-v16-clean-20260827a")
        .replace(PACKAGE_NAME, "r40_independent_live_binding_v16_local_integration")
    )
    normalized_sha256 = sha256(normalized.encode("utf-8"))
    require(normalized_sha256 == EXPECTED_V16_GENERATED_LAUNCHER_SHA256 == equivalence["v16_generated_launcher_sha256"], "normalized v16 launcher external pin drift")
    require(equivalence["generated_launcher_diff_is_only_stage_result_package_identifiers"] is True, "generated launcher equivalence assertion drift")
    return {
        "schema_version": "forkaudit-r40-v18-self-contained-science-verification-v1",
        "status": STATUS,
        "scientific_payload_file_count": len(SCIENCE_RELATIVES),
        "scientific_payload_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "scientific_equivalence_file_sha256": EXPECTED_EQUIVALENCE_SHA256,
        "scientific_equivalence_payload_sha256": EXPECTED_EQUIVALENCE_PAYLOAD_SHA256,
        "v18_generated_launcher_sha256": generated_sha256,
        "normalized_v16_generated_launcher_sha256": normalized_sha256,
        "sibling_source_accessed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--repo-root", type=Path)
    args = parser.parse_args()
    result = verify_scientific_payload(root=args.root, repo_root=args.repo_root)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
