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
PACKAGE_NAME = "r40_independent_live_binding_v19_terminal_closure_fix"
V18_PACKAGE_NAME = "r40_independent_live_binding_v18_self_contained_stage"
MANIFEST_NAME = "v19-current-payload.sha256"
CONTROLLED_DIFF_NAME = "v18-v19-controlled-diff.json"
EXPECTED_MANIFEST_SHA256 = "6007133c45dc24f51f1482598e17c739eb8eae7f4579bed6eb101d3a249896ab"
EXPECTED_CONTROLLED_DIFF_SHA256 = "fc81a6a0b6cbac46708ef4f9680db9829ba7c2e22f437d9e8f8629be81c76f9c"
EXPECTED_CONTROLLED_DIFF_PAYLOAD_SHA256 = "73fae1086b003a4233546b6d6c179bf0cac4e44445efb790e9aa54801d44b6f6"
EXPECTED_V6_LAUNCHER_SHA256 = "299907b4f95e7f5d8873ef5d810698640cc525ec9d2af647d325465b150e69ee"
EXPECTED_V18_GENERATED_LAUNCHER_SHA256 = "8d5ba77f9b61b760346334b4bca041e1ac0176719c5b8bd2e616a29b24226636"
EXPECTED_V19_GENERATED_LAUNCHER_SHA256 = "ef1f68028fbec4180925c60701ed6d850975c0eeeff541964cab60fafa2e20ed"
EXPECTED_RUNNER_SHA256 = "9da619fc037e2c670b146d778fd9f4d5344212b7e525f3d3f26a077f79d67775"
EXPECTED_BUILDER_SHA256 = "546efd59e2833034bc2e24d4cc0e6077f5a408275e359af43cd96f7f71cad16e"
EXPECTED_V18_TREE_CLOSURE_SHA256 = "c4b18fd6636e9bf70c055cbd2103ae0cc7a3c04d3989d2bb9b9919495962daf4"
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
CONTROLLED_DIFF_FIELDS = {
    "byte_identical_file_count",
    "controlled_change_file_count",
    "controlled_change_scope",
    "current_payload_file_count",
    "current_payload_manifest_sha256",
    "generated_launcher_diff_is_only_stage_result_package_identifiers",
    "immutable_external_builder_sha256",
    "immutable_external_runner_sha256",
    "payload_sha256",
    "rows",
    "schema_version",
    "sibling_source_required",
    "status",
    "v18_generated_launcher_sha256",
    "v18_package",
    "v19_generated_launcher_sha256",
    "v19_package",
    "verification_mode",
}
ROW_FIELDS = {
    "byte_identical",
    "classification",
    "path",
    "v18_sha256",
    "v18_size",
    "v19_sha256",
    "v19_size",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def stable_regular_bytes(path: Path, *, label: str) -> bytes:
    require(hasattr(os, "O_NOFOLLOW"), f"O_NOFOLLOW unavailable for {label}")
    descriptor = os.open(
        os.fspath(path),
        os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        before = os.fstat(descriptor)
        require(
            stat.S_ISREG(before.st_mode) and before.st_nlink == 1,
            f"{label} must be a singly linked regular file",
        )
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
        require(
            before_identity == after_identity and len(data) == before.st_size,
            f"{label} changed during exact read",
        )
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
    value = json.loads(
        data.decode("utf-8", errors="strict"),
        object_pairs_hook=reject_duplicate_pairs,
    )
    require(type(value) is dict, f"{label} must be an object")
    return value


def canonical_root(path: Path, *, label: str) -> Path:
    lexical = Path(os.path.abspath(os.fspath(path)))
    metadata = os.lstat(lexical)
    require(
        stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode),
        f"{label} must be an exact directory",
    )
    require(lexical.resolve(strict=True) == lexical, f"{label} canonical path drift")
    return lexical


def parse_manifest(data: bytes) -> list[tuple[str, str]]:
    require(sha256(data) == EXPECTED_MANIFEST_SHA256, "current payload manifest external pin drift")
    text = data.decode("ascii", errors="strict")
    require(text.endswith("\n") and "\r" not in text, "current payload manifest encoding drift")
    lines = text.splitlines()
    require(len(lines) == len(SCIENCE_RELATIVES), "current payload manifest file count drift")
    rows: list[tuple[str, str]] = []
    for index, line in enumerate(lines):
        pieces = line.split("  ", 1)
        require(len(pieces) == 2, "current payload manifest row format drift")
        digest, relative = pieces
        require(is_sha256(digest), "current payload manifest digest format drift")
        require(relative == SCIENCE_RELATIVES[index], "current payload manifest path/order drift")
        rows.append((relative, digest))
    return rows


def load_launcher_builder(path: Path):
    specification = importlib.util.spec_from_file_location("r40_v19_launcher_builder", path)
    require(
        specification is not None and specification.loader is not None,
        "launcher builder import unavailable",
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def verify_controlled_diff(
    *, root: Path, manifest_rows: list[tuple[str, str]]
) -> dict[str, Any]:
    data = stable_regular_bytes(root / CONTROLLED_DIFF_NAME, label="v18-v19 controlled diff")
    require(sha256(data) == EXPECTED_CONTROLLED_DIFF_SHA256, "controlled diff external file pin drift")
    value = load_exact_json(data, label="v18-v19 controlled diff")
    require(set(value) == CONTROLLED_DIFF_FIELDS, "controlled diff exact schema drift")
    supplied = value["payload_sha256"]
    require(
        supplied == EXPECTED_CONTROLLED_DIFF_PAYLOAD_SHA256,
        "controlled diff payload seal external pin drift",
    )
    candidate = dict(value)
    candidate["payload_sha256"] = None
    expected_seal = sha256(
        json.dumps(candidate, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    require(supplied == expected_seal, "controlled diff payload seal mismatch")
    require(
        value["schema_version"] == "forkaudit-r40-v18-v19-controlled-diff-v1"
        and value["status"] == STATUS,
        "controlled diff identity drift",
    )
    require(
        value["v18_package"] == V18_PACKAGE_NAME
        and value["v19_package"] == PACKAGE_NAME,
        "controlled diff package identity drift",
    )
    require(
        value["verification_mode"] == "self-contained-v18-external-pins-plus-current-v19"
        and value["sibling_source_required"] is False,
        "controlled diff verification mode drift",
    )
    require(
        value["current_payload_manifest_sha256"] == EXPECTED_MANIFEST_SHA256
        and value["current_payload_file_count"] == len(SCIENCE_RELATIVES)
        and value["byte_identical_file_count"] == 9
        and value["controlled_change_file_count"] == 1,
        "controlled diff counts/manifest drift",
    )
    require(
        value["immutable_external_runner_sha256"] == EXPECTED_RUNNER_SHA256
        and value["immutable_external_builder_sha256"] == EXPECTED_BUILDER_SHA256,
        "controlled diff external producer pins drift",
    )
    rows = value["rows"]
    require(type(rows) is list and len(rows) == len(SCIENCE_RELATIVES), "controlled diff rows drift")
    changed: list[str] = []
    for index, ((relative, manifest_digest), row) in enumerate(zip(manifest_rows, rows)):
        require(type(row) is dict and set(row) == ROW_FIELDS, "controlled diff row exact schema drift")
        require(row["path"] == SCIENCE_RELATIVES[index] == relative, "controlled diff row path/order drift")
        require(
            row["v19_sha256"] == manifest_digest
            and is_sha256(row["v18_sha256"])
            and type(row["v18_size"]) is int
            and type(row["v19_size"]) is int,
            f"controlled diff row digest/size drift: {relative}",
        )
        current = stable_regular_bytes(root / relative, label=f"current payload {relative}")
        require(
            sha256(current) == row["v19_sha256"] and len(current) == row["v19_size"],
            f"current payload hash/size drift: {relative}",
        )
        if row["byte_identical"] is True:
            require(
                row["classification"] == "byte-identical"
                and row["v18_sha256"] == row["v19_sha256"]
                and row["v18_size"] == row["v19_size"],
                f"controlled diff identical row drift: {relative}",
            )
        else:
            changed.append(relative)
            require(
                row["classification"] == "terminal-governance-invocation-closure-repair",
                "controlled diff change classification drift",
            )
    require(changed == ["executed_source/r40_tree_closure.py"], "controlled diff change set drift")
    require(
        rows[-1]["v18_sha256"] == EXPECTED_V18_TREE_CLOSURE_SHA256,
        "controlled diff v18 tree-closure pin drift",
    )
    return value


def verify_scientific_payload(*, root: Path = ROOT, repo_root: Path | None = None) -> dict[str, Any]:
    root = canonical_root(root, label="v19 package root")
    require(root.name == PACKAGE_NAME, "v19 package root identity drift")
    repository = canonical_root(
        repo_root if repo_root is not None else root.parents[2],
        label="repository root",
    )
    manifest_rows = parse_manifest(
        stable_regular_bytes(root / MANIFEST_NAME, label="current payload manifest")
    )
    controlled = verify_controlled_diff(root=root, manifest_rows=manifest_rows)
    current_python_paths = tuple(
        f"executed_source/{path.name}"
        for path in sorted((root / "executed_source").glob("*.py"))
    )
    require(current_python_paths == SCIENCE_RELATIVES[2:], "executed-source exact file set drift")

    runner = repository / "paper_autonomous_multifork_iteration/evidence/round_04_rr2_package/executed_source/gpu/run_qcomem_qwen35_forkaudit_review_revision.py"
    production_builder = repository / "paper_autonomous_multifork_iteration/evidence/round_04_rr2_package/executed_source/gpu/qcomem_vllm_paged_multifork_resident.py"
    require(sha256(stable_regular_bytes(runner, label="immutable runner")) == EXPECTED_RUNNER_SHA256, "immutable runner drift")
    require(sha256(stable_regular_bytes(production_builder, label="immutable builder")) == EXPECTED_BUILDER_SHA256, "immutable builder drift")

    v6_launcher = repository / "paper_autonomous_multifork_iteration/evidence/r39_primary_compiled_dispatch_v6/executed_source/r39_primary_formal_h20.sh"
    builder = load_launcher_builder(root / "scripts/build_formal_launcher.py")
    v6_snapshot = builder.read_regular_snapshot(v6_launcher)
    require(sha256(v6_snapshot) == EXPECTED_V6_LAUNCHER_SHA256, "v6 launcher external pin drift")
    generated = builder.transform(v6_snapshot.decode("utf-8", errors="strict"))
    generated_sha256 = sha256(generated.encode("utf-8"))
    require(
        generated_sha256
        == EXPECTED_V19_GENERATED_LAUNCHER_SHA256
        == controlled["v19_generated_launcher_sha256"],
        "v19 generated launcher pin drift",
    )
    normalized = (
        generated.replace("qcomem_r40_v19_clean_20260828a", "qcomem_r40_v18_clean_20260828a")
        .replace("r40-v19-clean-20260828a", "r40-v18-clean-20260828a")
        .replace(PACKAGE_NAME, V18_PACKAGE_NAME)
    )
    require(
        sha256(normalized.encode("utf-8"))
        == EXPECTED_V18_GENERATED_LAUNCHER_SHA256
        == controlled["v18_generated_launcher_sha256"],
        "normalized v18 launcher pin drift",
    )
    require(
        controlled["generated_launcher_diff_is_only_stage_result_package_identifiers"] is True,
        "generated launcher controlled-diff assertion drift",
    )
    return {
        "schema_version": "forkaudit-r40-v19-controlled-payload-verification-v1",
        "status": STATUS,
        "current_payload_file_count": len(SCIENCE_RELATIVES),
        "current_payload_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "controlled_diff_file_sha256": EXPECTED_CONTROLLED_DIFF_SHA256,
        "controlled_diff_payload_sha256": EXPECTED_CONTROLLED_DIFF_PAYLOAD_SHA256,
        "byte_identical_file_count": 9,
        "controlled_change_file_count": 1,
        "controlled_change_path": "executed_source/r40_tree_closure.py",
        "v19_generated_launcher_sha256": generated_sha256,
        "normalized_v18_generated_launcher_sha256": EXPECTED_V18_GENERATED_LAUNCHER_SHA256,
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
