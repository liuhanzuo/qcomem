from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V16_ROOT = ROOT.parent / "r40_independent_live_binding_v16_local_integration"
STATUS = "HOLD_PENDING_FRESH_AUDIT_AND_H20"
RUNNER_SHA256 = "9da619fc037e2c670b146d778fd9f4d5344212b7e525f3d3f26a077f79d67775"
BUILDER_SHA256 = "546efd59e2833034bc2e24d4cc0e6077f5a408275e359af43cd96f7f71cad16e"
MANIFEST = ROOT / "v16-scientific-payload.sha256"
EQUIVALENCE = ROOT / "v16-v17-scientific-equivalence.json"


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def seal(value: dict[str, object]) -> dict[str, object]:
    result = dict(value)
    result["payload_sha256"] = None
    result["payload_sha256"] = sha256(
        json.dumps(result, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    return result


def load_launcher_builder(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError("launcher builder import specification unavailable")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def main() -> int:
    if not V16_ROOT.is_dir():
        raise RuntimeError("canonical v16 source root unavailable")
    relatives = ["preregistration.json", "absorbed-lineage.json"]
    relatives.extend(
        str(path.relative_to(ROOT))
        for path in sorted((ROOT / "executed_source").glob("*.py"))
    )
    rows: list[dict[str, object]] = []
    manifest_lines: list[str] = []
    for relative in relatives:
        current = (ROOT / relative).read_bytes()
        prior = (V16_ROOT / relative).read_bytes()
        if current != prior:
            raise RuntimeError(f"v16/v17 scientific payload byte drift: {relative}")
        digest = sha256(current)
        manifest_lines.append(f"{digest}  {relative}")
        rows.append({"path": relative, "sha256": digest, "size": len(current), "byte_identical": True})
    manifest_bytes = ("\n".join(manifest_lines) + "\n").encode("ascii")
    with MANIFEST.open("xb") as stream:
        stream.write(manifest_bytes)
    v6_launcher = (ROOT.parent / "r39_primary_compiled_dispatch_v6/executed_source/r39_primary_formal_h20.sh").read_text()
    v16_builder = load_launcher_builder("r40_v16_launcher_builder", V16_ROOT / "scripts/build_formal_launcher.py")
    v17_builder = load_launcher_builder("r40_v17_launcher_builder", ROOT / "scripts/build_formal_launcher.py")
    v16_launcher = v16_builder.transform(v6_launcher)
    v17_launcher = v17_builder.transform(v6_launcher)
    normalized_v17 = (
        v17_launcher.replace("qcomem_r40_v17_clean_20260827b", "qcomem_r40_v16_clean_20260827a")
        .replace("r40-v17-clean-20260827b", "r40-v16-clean-20260827a")
        .replace("r40_independent_live_binding_v17_linux_stage", "r40_independent_live_binding_v16_local_integration")
    )
    if normalized_v17 != v16_launcher:
        raise RuntimeError("generated v17 launcher differs from v16 beyond exact stage/result/package identifiers")
    equivalence = seal(
        {
            "schema_version": "forkaudit-r40-v17-v16-scientific-equivalence-v1",
            "status": STATUS,
            "v16_package": V16_ROOT.name,
            "v17_package": ROOT.name,
            "scientific_payload_manifest_sha256": sha256(manifest_bytes),
            "scientific_payload_file_count": len(rows),
            "all_scientific_payload_files_byte_identical": True,
            "immutable_external_runner_sha256": RUNNER_SHA256,
            "immutable_external_builder_sha256": BUILDER_SHA256,
            "v16_generated_launcher_sha256": sha256(v16_launcher.encode("utf-8")),
            "v17_generated_launcher_sha256": sha256(v17_launcher.encode("utf-8")),
            "generated_launcher_diff_is_only_stage_result_package_identifiers": True,
            "rows": rows,
            "payload_sha256": None,
        }
    )
    with EQUIVALENCE.open("xb") as stream:
        stream.write((json.dumps(equivalence, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"))
    print(sha256(manifest_bytes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
