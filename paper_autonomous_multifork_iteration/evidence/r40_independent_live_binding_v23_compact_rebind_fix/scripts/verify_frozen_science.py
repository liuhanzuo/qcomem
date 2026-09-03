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
PACKAGE_NAME = "r40_independent_live_binding_v23_compact_rebind_fix"
MANIFEST_NAME = "v23-current-payload.sha256"
DIFF_NAME = "v22-v23-controlled-diff.json"
EXPECTED_MANIFEST_SHA256 = "768fbd11fe00b6f650b326f46697ea79491e0d71d17d7478375d746cc55e419d"
EXPECTED_DIFF_SHA256 = "44d575b5dd5259a607cd90dd00e350d718a0febb370e8f07de0ed17034976405"
EXPECTED_DIFF_PAYLOAD_SHA256 = "fd7066dcc1a6650d33f68390714f0ea3c43c3a399b573db75dc6eab459d9d7c5"
EXPECTED_V6_LAUNCHER_SHA256 = "299907b4f95e7f5d8873ef5d810698640cc525ec9d2af647d325465b150e69ee"
EXPECTED_V22_GENERATED_LAUNCHER_SHA256 = "3dbeb7a5e4fbbcaf3075c60e371915acdd4a1b0d723c75b749528eda5fe4ca13"
EXPECTED_V23_GENERATED_LAUNCHER_SHA256 = "800a9147c76291fec754b27c9a04165d7cb061a1437d1bd9f706dda1d2ec75f0"
EXPECTED_RUNNER_SHA256 = "9da619fc037e2c670b146d778fd9f4d5344212b7e525f3d3f26a077f79d67775"
EXPECTED_BUILDER_SHA256 = "546efd59e2833034bc2e24d4cc0e6077f5a408275e359af43cd96f7f71cad16e"
EXPECTED_VERIFIER_SHA256 = "f5ec8b7dbfd5efd3a71af58d39e7f9bd9879dc8d4225453b48fc8eab41bc265c"
SCIENCE_RELATIVES = (
    "preregistration.json",
    "absorbed-lineage.json",
    "executed_source/r40_compact_rebind_fix.py",
    "executed_source/r40_cuda_smoke.py",
    "executed_source/r40_finalize.py",
    "executed_source/r40_formal_preflight.py",
    "executed_source/r40_passive_clone_lineage.py",
    "executed_source/r40_rank_entrypoint.py",
    "executed_source/r40_real_binding.py",
    "executed_source/r40_real_binding_hook.py",
    "executed_source/r40_tree_closure.py",
)
ROW_FIELDS = {
    "path", "classification", "byte_identical", "v22_sha256", "v22_size",
    "v23_sha256", "v23_size",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def is_sha256(value: object) -> bool:
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def stable_regular_bytes(path: Path, *, label: str) -> bytes:
    require(hasattr(os, "O_NOFOLLOW"), f"O_NOFOLLOW unavailable for {label}")
    descriptor = os.open(os.fspath(path), os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0))
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1, f"{label} must be singly linked regular")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            data = stream.read()
        after = os.fstat(descriptor)
        before_id = (before.st_dev, before.st_ino, before.st_mode, before.st_nlink, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        after_id = (after.st_dev, after.st_ino, after.st_mode, after.st_nlink, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
        require(before_id == after_id and len(data) == before.st_size, f"{label} changed during exact read")
        return data
    finally:
        os.close(descriptor)


def canonical_root(path: Path, *, label: str) -> Path:
    lexical = Path(os.path.abspath(os.fspath(path)))
    metadata = os.lstat(lexical)
    require(stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode), f"{label} must be exact directory")
    require(lexical.resolve(strict=True) == lexical, f"{label} canonical path drift")
    return lexical


def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(data: bytes, *, label: str) -> dict[str, Any]:
    value = json.loads(data.decode("utf-8", errors="strict"), object_pairs_hook=reject_duplicates)
    require(type(value) is dict, f"{label} must be object")
    return value


def parse_manifest(data: bytes) -> list[tuple[str, str]]:
    require(sha256(data) == EXPECTED_MANIFEST_SHA256, "v23 payload manifest external pin drift")
    text = data.decode("ascii", errors="strict")
    require(text.endswith("\n") and "\r" not in text, "v23 payload manifest encoding drift")
    lines = text.splitlines()
    require(len(lines) == len(SCIENCE_RELATIVES), "v23 payload manifest file count drift")
    rows = []
    for index, line in enumerate(lines):
        pieces = line.split("  ", 1)
        require(len(pieces) == 2 and is_sha256(pieces[0]), "v23 payload manifest row drift")
        require(pieces[1] == SCIENCE_RELATIVES[index], "v23 payload manifest path/order drift")
        rows.append((pieces[1], pieces[0]))
    return rows


def load_launcher_builder(path: Path):
    spec = importlib.util.spec_from_file_location("r40_v23_launcher_builder", path)
    require(spec is not None and spec.loader is not None, "launcher builder import unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_scientific_payload(*, root: Path = ROOT, repo_root: Path | None = None) -> dict[str, Any]:
    root = canonical_root(root, label="v23 package root")
    require(root.name == PACKAGE_NAME, "v23 package root identity drift")
    repository = canonical_root(repo_root if repo_root is not None else root.parents[2], label="repository root")
    manifest_rows = parse_manifest(stable_regular_bytes(root / MANIFEST_NAME, label="v23 payload manifest"))

    diff_data = stable_regular_bytes(root / DIFF_NAME, label="v22-v23 controlled diff")
    require(sha256(diff_data) == EXPECTED_DIFF_SHA256, "v22-v23 controlled diff external pin drift")
    diff = load_json(diff_data, label="v22-v23 controlled diff")
    supplied = diff.get("payload_sha256")
    candidate = dict(diff)
    candidate["payload_sha256"] = None
    seal = sha256(json.dumps(candidate, sort_keys=True, separators=(",", ":")).encode())
    require(supplied == seal == EXPECTED_DIFF_PAYLOAD_SHA256, "v22-v23 controlled diff seal drift")
    require(diff.get("schema_version") == "forkaudit-r40-v22-v23-controlled-diff-v1" and diff.get("status") == STATUS, "v22-v23 diff identity drift")
    require(diff.get("current_payload_manifest_sha256") == EXPECTED_MANIFEST_SHA256, "v22-v23 diff manifest pin drift")
    require((diff.get("current_payload_file_count"), diff.get("byte_identical_file_count"), diff.get("controlled_change_file_count")) == (11, 8, 3), "v22-v23 diff counts drift")
    require(diff.get("verifier_byte_identical_to_v22") is True, "verifier change forbidden")
    rows = diff.get("rows")
    require(type(rows) is list and len(rows) == len(SCIENCE_RELATIVES), "v22-v23 diff rows drift")
    changed = []
    for (relative, digest), row in zip(manifest_rows, rows):
        require(type(row) is dict and set(row) == ROW_FIELDS and row["path"] == relative, f"v22-v23 row schema/path drift: {relative}")
        current = stable_regular_bytes(root / relative, label=f"v23 payload {relative}")
        require(sha256(current) == digest == row["v23_sha256"] and len(current) == row["v23_size"], f"v23 payload hash/size drift: {relative}")
        if row["byte_identical"] is True:
            require(row["classification"] == "byte-identical" and row["v22_sha256"] == row["v23_sha256"] and row["v22_size"] == row["v23_size"], f"false identical row: {relative}")
        else:
            changed.append(relative)
    require(changed == ["preregistration.json", "executed_source/r40_compact_rebind_fix.py", "executed_source/r40_rank_entrypoint.py"], "v22-v23 controlled change set drift")
    current_python = tuple(f"executed_source/{p.name}" for p in sorted((root / "executed_source").glob("*.py")))
    require(current_python == SCIENCE_RELATIVES[2:], "executed-source exact file set drift")
    require(sha256(stable_regular_bytes(root / "executed_source/r40_real_binding.py", label="unchanged verifier")) == EXPECTED_VERIFIER_SHA256, "v22 verifier byte identity drift")

    gpu = repository / "paper_autonomous_multifork_iteration/evidence/round_04_rr2_package/executed_source/gpu"
    runner = gpu / "run_qcomem_qwen35_forkaudit_review_revision.py"
    builder_source = gpu / "qcomem_vllm_paged_multifork_resident.py"
    require(sha256(stable_regular_bytes(runner, label="immutable runner")) == EXPECTED_RUNNER_SHA256, "immutable runner drift")
    require(sha256(stable_regular_bytes(builder_source, label="immutable builder")) == EXPECTED_BUILDER_SHA256, "immutable builder drift")
    require(diff["immutable_external_runner_sha256"] == EXPECTED_RUNNER_SHA256 and diff["immutable_external_builder_sha256"] == EXPECTED_BUILDER_SHA256, "diff producer pins drift")

    v6 = repository / "paper_autonomous_multifork_iteration/evidence/r39_primary_compiled_dispatch_v6/executed_source/r39_primary_formal_h20.sh"
    launcher_builder = load_launcher_builder(root / "scripts/build_formal_launcher.py")
    snapshot = launcher_builder.read_regular_snapshot(v6)
    require(sha256(snapshot) == EXPECTED_V6_LAUNCHER_SHA256, "v6 launcher drift")
    generated = launcher_builder.transform(snapshot.decode("utf-8", errors="strict"))
    generated_sha = sha256(generated.encode())
    require(generated_sha == EXPECTED_V23_GENERATED_LAUNCHER_SHA256 == diff["v23_generated_launcher_sha256"], "v23 generated launcher pin drift")
    normalized = generated.replace("qcomem_r40_v23_compact_rebind_fix_20260829a", "qcomem_r40_v22_descriptor_diag_20260829a").replace("r40-v23-compact-rebind-fix-20260829a", "r40-v22-descriptor-diag-20260829a").replace(PACKAGE_NAME, "r40_independent_live_binding_v22_descriptor_diagnostic")
    require(sha256(normalized.encode()) == EXPECTED_V22_GENERATED_LAUNCHER_SHA256 == diff["v22_generated_launcher_sha256"], "normalized v22 launcher pin drift")
    require(diff["generated_launcher_diff_is_only_stage_result_package_identifiers"] is True, "launcher diff scope drift")
    return {
        "schema_version": "forkaudit-r40-v23-compact-rebind-payload-verification-v1",
        "status": STATUS,
        "science_accepted": False,
        "current_payload_file_count": 11,
        "current_payload_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "controlled_diff_file_sha256": EXPECTED_DIFF_SHA256,
        "controlled_diff_payload_sha256": EXPECTED_DIFF_PAYLOAD_SHA256,
        "byte_identical_file_count": 8,
        "controlled_change_file_count": 3,
        "verifier_sha256": EXPECTED_VERIFIER_SHA256,
        "v23_generated_launcher_sha256": generated_sha,
        "sibling_source_accessed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--repo-root", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify_scientific_payload(root=args.root, repo_root=args.repo_root), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
