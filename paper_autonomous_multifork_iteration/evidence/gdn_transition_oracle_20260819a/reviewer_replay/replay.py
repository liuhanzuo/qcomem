#!/usr/bin/env python3
"""Manifest-first replay for the anonymous GDN transition-oracle package."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any


EXCLUDED = {"MANIFEST.json", "MANIFEST.sha256"}
PORTABILITY_ABS = 1e-6


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = [row["relative_path"] for row in manifest["files"]]
    actual = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.relative_to(root).as_posix() not in EXCLUDED
    )
    if expected != sorted(expected) or len(expected) != len(set(expected)):
        raise ValueError("manifest paths are not unique and sorted")
    if actual != expected:
        raise ValueError("reviewer package file-set drift")
    total = 0
    for row in manifest["files"]:
        path = root / row["relative_path"]
        if path.stat().st_size != row["bytes"] or sha256_file(path) != row["sha256"]:
            raise ValueError(f"reviewer package file drift: {row['relative_path']}")
        total += row["bytes"]
    if total != manifest["total_bytes"] or len(expected) != manifest["file_count"]:
        raise ValueError("reviewer package manifest summary drift")
    sidecar = (root / "MANIFEST.sha256").read_text(encoding="ascii").strip()
    if sidecar != f"{sha256_file(manifest_path)}  MANIFEST.json":
        raise ValueError("reviewer package manifest sidecar drift")
    return manifest


def load_reference(root: Path):
    path = root / "reference" / "qcomem_gdn_transition_oracle_reference.py"
    spec = importlib.util.spec_from_file_location("anonymous_gdn_reference", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load reference source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compare_scalar(actual: Any, expected: Any, path: str) -> None:
    if isinstance(expected, bool) or isinstance(expected, str) or expected is None:
        if actual != expected:
            raise ValueError(f"decision drift at {path}: {actual!r} != {expected!r}")
    elif isinstance(expected, (int, float)):
        if not isinstance(actual, (int, float)) or not math.isfinite(float(actual)):
            raise ValueError(f"non-finite or non-numeric replay value at {path}")
        if abs(float(actual) - float(expected)) > PORTABILITY_ABS:
            raise ValueError(f"numeric replay drift at {path}: {actual!r} != {expected!r}")
    else:
        raise TypeError(f"unexpected scalar type at {path}")


def compare_tree(actual: Any, expected: Any, path: str = "result") -> None:
    if isinstance(expected, dict):
        for key, value in expected.items():
            if key in {"aggregator_runner_raw_sha256", "reference_source_raw_sha256", "capture_manifest_raw_sha256"}:
                continue
            if key not in actual:
                raise ValueError(f"missing replay field {path}.{key}")
            compare_tree(actual[key], value, f"{path}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"list-shape drift at {path}")
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            compare_tree(actual_item, expected_item, f"{path}[{index}]")
    else:
        compare_scalar(actual, expected, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.package_root.resolve()
    manifest = verify_manifest(root)
    reference = load_reference(root)
    replayed = reference.evaluate_capture(
        root / "raw" / "capture-manifest.json",
        root / "preregistration.json",
        args.output,
    )
    expected = json.loads(
        (root / "expected" / "original-oracle-result.json").read_text(encoding="utf-8")
    )
    compare_tree(replayed, expected)
    recorded_output_max = max(row["output_metrics"]["relative_l2"] for row in expected["rows"])
    replayed_output_max = max(row["output_metrics"]["relative_l2"] for row in replayed["rows"])
    recorded_state_max = max(row["state_metrics"]["relative_l2"] for row in expected["rows"])
    replayed_state_max = max(row["state_metrics"]["relative_l2"] for row in replayed["rows"])
    print(
        json.dumps(
            {
                "all_clean_rows_pass": replayed["all_clean_rows_pass"],
                "all_seeded_wrong_transitions_rejected": replayed[
                    "all_seeded_wrong_transitions_rejected"
                ],
                "manifest_sha256": sha256_file(root / "MANIFEST.json"),
                "package_files": manifest["file_count"],
                "recorded_output_max_relative_l2": recorded_output_max,
                "recorded_state_max_relative_l2": recorded_state_max,
                "replayed_output_max_relative_l2": replayed_output_max,
                "replayed_state_max_relative_l2": replayed_state_max,
                "scalar_portability_abs": PORTABILITY_ABS,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
