#!/usr/bin/env python3
"""Manifest-first replay for the anonymous fresh-preregistered GDN derivative."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from typing import Any


class ReplayError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReplayError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_inputs(root: Path, manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    manifest_path = root / "INPUTS_MANIFEST.json"
    package_root = root.parent
    manifest = load_json(manifest_path) if manifest is None else manifest
    require(
        manifest.get("schema_version") == "forkaudit-fresh-gdn-reviewer-inputs-v1",
        "input manifest schema drift",
    )
    rows = manifest.get("files")
    require(isinstance(rows, list) and len(rows) == 46, "input manifest cardinality drift")
    seen: set[str] = set()
    total_bytes = 0
    for row in rows:
        require(set(row) == {"relative_path", "sha256", "bytes"}, "input row schema drift")
        relative = Path(row["relative_path"])
        require(not relative.is_absolute() and ".." not in relative.parts, "unsafe input path")
        relative_text = relative.as_posix()
        require(relative_text not in seen, "duplicate input path")
        seen.add(relative_text)
        path = package_root / relative
        require(path.is_file(), f"missing input: {relative_text}")
        require(path.stat().st_size == row["bytes"], f"input byte drift: {relative_text}")
        require(sha256_file(path) == row["sha256"], f"input SHA drift: {relative_text}")
        total_bytes += row["bytes"]
    required = {
        "reviewer_replay/reference.py",
        "reviewer_replay/replay.py",
        "reviewer_replay/preregistration.reviewer.json",
        "reviewer_replay/capture-manifest.reviewer.json",
        "artifacts/oracle-result.json",
        "reviewer_replay/EXECUTION_TO_REVIEWER_BINDING.json",
    }
    require(required <= seen, "required replay authority missing")
    require(manifest.get("file_count") == len(rows), "declared input count drift")
    require(manifest.get("total_bytes") == total_bytes, "declared input bytes drift")
    return {"file_count": len(rows), "total_bytes": total_bytes, "all_sha256_verified": True}


def load_reference(path: Path):
    spec = importlib.util.spec_from_file_location("fresh_gdn_reviewer_reference", path)
    require(spec is not None and spec.loader is not None, "cannot load reference module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_execution_binding(root: Path) -> dict[str, Any]:
    binding = load_json(root / "EXECUTION_TO_REVIEWER_BINDING.json")
    require(
        binding.get("schema_version") == "forkaudit-execution-to-reviewer-derivative-v1",
        "execution/derivative binding schema drift",
    )
    prereg = root / "preregistration.reviewer.json"
    capture = root / "capture-manifest.reviewer.json"
    require(
        binding["reviewer_derivative"]["preregistration_sha256"] == sha256_file(prereg),
        "reviewer preregistration binding drift",
    )
    require(
        binding["reviewer_derivative"]["capture_manifest_sha256"] == sha256_file(capture),
        "reviewer capture binding drift",
    )
    capture_json = load_json(capture)
    require(
        capture_json["preregistration_raw_sha256"] == sha256_file(prereg),
        "reviewer capture/preregistration binding drift",
    )
    require(
        binding["execution_authority"]["preregistration_sha256"]
        == "3d37356eb58f1a3f8f4eb0890d1631ff520b4f01fd5ff485fa19740ff2fcc0fe",
        "original execution preregistration authority drift",
    )
    return binding


def replay(root: Path, output: Path) -> dict[str, Any]:
    inputs = verify_inputs(root)
    binding = validate_execution_binding(root)
    reference = load_reference(root / "reference.py")
    result = reference.evaluate_capture(
        root / "capture-manifest.reviewer.json",
        root / "preregistration.reviewer.json",
        output,
    )
    require(result.get("candidate_code_imported") is False, "reference imported candidate code")
    require(result.get("all_clean_rows_pass") is True, "a clean GDN row failed")
    require(
        result.get("all_seeded_wrong_transitions_rejected") is True,
        "a seeded wrong transition escaped",
    )
    require(len(result.get("rows", [])) == 4 and len(result.get("faults", [])) == 4, "row/fault coverage drift")
    producer = load_json(root.parent / "artifacts/oracle-result.json")
    require(
        producer["all_clean_rows_pass"] == result["all_clean_rows_pass"]
        and producer["all_seeded_wrong_transitions_rejected"]
        == result["all_seeded_wrong_transitions_rejected"]
        and producer["claim_boundary"] == result["claim_boundary"],
        "producer/reviewer decision projection drift",
    )
    metric_atol = {
        "relative_l2": 5e-7,
        "normalized_max_abs": 5e-7,
        "max_abs": 5e-6,
        "coordinate_max_abs": 1e-6,
    }
    max_delta = {key: 0.0 for key in metric_atol}
    for producer_row, replay_row in zip(producer["rows"], result["rows"]):
        require(
            (producer_row["row_id"], producer_row["layer_index"], producer_row["clean_pass"])
            == (replay_row["row_id"], replay_row["layer_index"], replay_row["clean_pass"]),
            "clean row identity/decision drift",
        )
        for family in ("output_metrics", "state_metrics"):
            require(
                producer_row[family]["finite"] == replay_row[family]["finite"],
                f"{family} finite decision drift",
            )
            for metric in ("relative_l2", "normalized_max_abs", "max_abs"):
                delta = abs(producer_row[family][metric] - replay_row[family][metric])
                max_delta[metric] = max(max_delta[metric], delta)
                require(delta <= metric_atol[metric], f"portable {family}/{metric} drift")
        for coordinate in ("output_coordinate_max_abs", "state_coordinate_max_abs"):
            delta = abs(producer_row[coordinate] - replay_row[coordinate])
            max_delta["coordinate_max_abs"] = max(max_delta["coordinate_max_abs"], delta)
            require(delta <= metric_atol["coordinate_max_abs"], "portable coordinate metric drift")
    for producer_fault, replay_fault in zip(producer["faults"], result["faults"]):
        require(
            (producer_fault["row_id"], producer_fault["fault"], producer_fault["rejected"])
            == (replay_fault["row_id"], replay_fault["fault"], replay_fault["rejected"]),
            "fault identity/decision drift",
        )
        for family in ("output_metrics", "state_metrics"):
            require(
                producer_fault[family]["finite"] == replay_fault[family]["finite"],
                f"fault {family} finite decision drift",
            )
            for metric in ("relative_l2", "normalized_max_abs", "max_abs"):
                delta = abs(producer_fault[family][metric] - replay_fault[family][metric])
                max_delta[metric] = max(max_delta[metric], delta)
                require(delta <= metric_atol[metric], f"portable fault {family}/{metric} drift")
    summary = {
        "passed": True,
        "input_files": inputs["file_count"],
        "input_bytes": inputs["total_bytes"],
        "clean_rows_passed": 4,
        "seeded_wrong_transitions_rejected": 4,
        "producer_scientific_decisions_exact": True,
        "portable_metric_atol": metric_atol,
        "portable_metric_max_absolute_delta": max_delta,
        "execution_preregistration_sha256": binding["execution_authority"][
            "preregistration_sha256"
        ],
        "reviewer_preregistration_sha256": binding["reviewer_derivative"][
            "preregistration_sha256"
        ],
    }
    print(json.dumps(summary, sort_keys=True))
    return summary


def self_test(root: Path) -> None:
    manifest = load_json(root / "INPUTS_MANIFEST.json")
    mutant = copy.deepcopy(manifest)
    mutant["files"][0]["sha256"] = "0" * 64
    try:
        verify_inputs(root, mutant)
    except ReplayError:
        pass
    else:
        raise AssertionError("manifest SHA tamper was accepted")

    reference = load_reference(root / "reference.py")
    with tempfile.TemporaryDirectory(prefix="fresh-gdn-binding-test-") as temp:
        temp_root = Path(temp)
        prereg = root / "preregistration.reviewer.json"
        capture = root / "capture-manifest.reviewer.json"
        tampered_prereg = temp_root / "preregistration.json"
        tampered_capture = temp_root / "capture.json"
        tampered_prereg.write_bytes(prereg.read_bytes() + b"\n")
        tampered_capture.write_bytes(capture.read_bytes())
        try:
            reference.evaluate_capture(
                tampered_capture, tampered_prereg, temp_root / "result.json"
            )
        except ValueError as error:
            require("binding drift" in str(error), "wrong preregistration tamper failure")
        else:
            raise AssertionError("capture/preregistration tamper was accepted")
    print("PASS fresh-GDN manifest and preregistration-binding adversarial tests")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    replay(root, args.output.resolve())
    if args.self_test:
        self_test(root)


if __name__ == "__main__":
    main()
