#!/usr/bin/env python3
"""Raw-shard replay for the anonymous same-model Transformers transfer run."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence


class ReplayError(RuntimeError):
    pass


PORTABLE_NUMERIC_ATOL = {"max_abs": 5e-5, "relative_l2": 5e-5}
PORTABLE_NUMERIC_MAX_DELTA = {"max_abs": 0.0, "relative_l2": 0.0}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReplayError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_inputs(root: Path, manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    manifest = load_json(root / "INPUTS_MANIFEST.json") if manifest is None else manifest
    require(
        manifest.get("schema_version") == "forkaudit-transformers-reviewer-inputs-v1",
        "A4 input manifest schema drift",
    )
    rows = manifest.get("files")
    require(isinstance(rows, list) and len(rows) == 31, "A4 input cardinality drift")
    seen: set[str] = set()
    total_bytes = 0
    for row in rows:
        require(set(row) == {"relative_path", "sha256", "bytes"}, "A4 input row schema drift")
        relative = Path(row["relative_path"])
        require(not relative.is_absolute() and ".." not in relative.parts, "unsafe A4 input path")
        text = relative.as_posix()
        require(text not in seen, "duplicate A4 input path")
        seen.add(text)
        path = root / relative
        require(path.is_file(), f"missing A4 input: {text}")
        require(path.stat().st_size == row["bytes"], f"A4 byte drift: {text}")
        require(sha256_file(path) == row["sha256"], f"A4 SHA drift: {text}")
        total_bytes += row["bytes"]
    require(manifest.get("file_count") == len(rows), "A4 declared file count drift")
    require(manifest.get("total_bytes") == total_bytes, "A4 declared byte count drift")
    return {"file_count": len(rows), "total_bytes": total_bytes, "all_sha256_verified": True}


def load_helper(root: Path):
    source = root / "executed_source"
    helper_path = source / "qcomem_transformers_forkaudit_transfer.py"
    spec = importlib.util.spec_from_file_location("a4_transformers_helper", helper_path)
    require(spec is not None and spec.loader is not None, "cannot load A4 helper")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    install_portable_numeric_replay(module)
    return module


def install_portable_numeric_replay(helper: Any) -> None:
    """Keep the executed replay logic but bound cross-PyTorch FP32 reductions."""

    def portable_numeric_oracle(
        value: Any,
        label: str,
        *,
        threshold: float,
        candidate: Sequence[Any],
        reference: Sequence[Any],
    ) -> bool:
        require(
            isinstance(value, Mapping) and set(value) == {"predicate_id", "passed", "rows"},
            f"{label} schema drift",
        )
        require(
            value["predicate_id"] == "INDEPENDENT_DENSE_SEMANTIC_ORACLE",
            f"{label} predicate drift",
        )
        rows = value["rows"]
        require(
            isinstance(rows, list) and len(rows) == len(candidate) == len(reference) and rows,
            f"{label} cardinality drift",
        )
        replayed = []
        for index, (row, actual, expected) in enumerate(zip(rows, candidate, reference)):
            require(
                isinstance(row, Mapping)
                and set(row)
                == {
                    "step",
                    "finite",
                    "top1_equal",
                    "max_abs",
                    "relative_l2",
                    "relative_l2_threshold",
                    "passed",
                },
                f"{label}[{index}] schema drift",
            )
            require(type(row["step"]) is int and row["step"] == index, f"{label} step drift")
            require(actual.shape == expected.shape, f"{label} sidecar shape mismatch")
            delta = actual - expected
            maximum = float(delta.abs().max().item())
            reference_norm = float(helper.torch.linalg.vector_norm(expected).item())
            relative = float(helper.torch.linalg.vector_norm(delta).item()) / max(
                reference_norm, 1e-12
            )
            for metric, recomputed in (("max_abs", maximum), ("relative_l2", relative)):
                recorded = float(row[metric])
                difference = abs(recorded - recomputed)
                PORTABLE_NUMERIC_MAX_DELTA[metric] = max(
                    PORTABLE_NUMERIC_MAX_DELTA[metric], difference
                )
                require(
                    difference <= PORTABLE_NUMERIC_ATOL[metric],
                    f"{label} portable {metric} drift: "
                    f"recorded={recorded} recomputed={recomputed} delta={difference}",
                )
            require(float(row["relative_l2_threshold"]) == threshold, f"{label} threshold drift")
            finite = math.isfinite(maximum) and math.isfinite(relative)
            top1 = int(actual.argmax().item()) == int(expected.argmax().item())
            passed = top1 and relative <= threshold
            require(type(row["finite"]) is bool and row["finite"] == finite, f"{label} finite drift")
            require(
                type(row["top1_equal"]) is bool and row["top1_equal"] == top1,
                f"{label} top1 drift",
            )
            require(type(row["passed"]) is bool and row["passed"] == passed, f"{label} pass drift")
            require(
                abs(float(row["relative_l2"]) - threshold)
                > PORTABLE_NUMERIC_ATOL["relative_l2"],
                f"{label} decision is within portable ambiguity band",
            )
            replayed.append(passed)
        overall = all(replayed)
        require(type(value["passed"]) is bool and value["passed"] == overall, f"{label} overall drift")
        return overall

    helper._replay_numeric_oracle = portable_numeric_oracle


def replay_from_paths(root: Path, shard_paths: list[Path]) -> dict[str, Any]:
    helper = load_helper(root)
    source_manifest_path = root / "results/receipts/frozen-source-manifest.json"
    static_manifest_path = root / "results/receipts/frozen-static-manifest.json"
    gpu_assignment_path = root / "results/receipts/gpu-assignment.json"
    source_manifest = load_json(source_manifest_path)
    static_manifest = load_json(static_manifest_path)
    gpu_assignment = load_json(gpu_assignment_path)
    helper.validate_source_manifest(root / "executed_source", source_manifest)
    pre_authority = root / "results/receipts/model-authority-pre.json"
    terminal_authority = root / "results/receipts/model-authority-terminal.json"
    require(pre_authority.read_bytes() == terminal_authority.read_bytes(), "A4 model authority closure drift")
    producer = load_json(root / "results/forkaudit-transformers-transfer-aggregate.json")
    reconstructed = helper.aggregate_shards(
        shard_paths,
        static_manifest=static_manifest,
        sidecar_dir=root / "results/raw/logits",
        static_manifest_raw_sha256=sha256_file(static_manifest_path),
        source_manifest_raw_sha256=sha256_file(source_manifest_path),
        model_authority_raw_sha256=sha256_file(pre_authority),
        gpu_assignment=gpu_assignment,
        gpu_assignment_raw_sha256=sha256_file(gpu_assignment_path),
    )
    require(
        canonical_bytes(reconstructed) == canonical_bytes(producer),
        "A4 raw-shard replay differs from producer aggregate",
    )
    return reconstructed


def replay(root: Path) -> dict[str, Any]:
    inputs = verify_inputs(root)
    shard_paths = sorted((root / "results/raw/shards").glob("*.json"))
    require(len(shard_paths) == 8, "A4 shard coverage drift")
    result = replay_from_paths(root, shard_paths)
    require(result["scientific_run_valid"] is True, "A4 scientific run invalid")
    require(result["passed"] is False, "A4 negative unexpectedly became positive")
    require(
        result["scientific_outcome"] == "valid_negative_transformers_runtime_transfer",
        "A4 scientific outcome drift",
    )
    require(
        result["fault_outcome_counts"]
        == {
            "detected_expected_predicate": 24,
            "detected_wrong_predicate": 0,
            "clean_false_positive": 16,
            "escaped": 0,
        },
        "A4 fault count drift",
    )
    summary = {
        "passed": True,
        "input_files": inputs["file_count"],
        "input_bytes": inputs["total_bytes"],
        "raw_shards_replayed": 8,
        "fp32_sidecars_replayed": 8,
        "producer_aggregate_exact": True,
        "portable_numeric_atol": PORTABLE_NUMERIC_ATOL,
        "portable_numeric_max_absolute_delta": PORTABLE_NUMERIC_MAX_DELTA,
        "scientific_outcome": result["scientific_outcome"],
        "clean_all_applicable_predicates_passed": result[
            "clean_all_applicable_predicates_passed"
        ],
        "fault_outcome_counts": result["fault_outcome_counts"],
    }
    print(json.dumps(summary, sort_keys=True))
    return summary


def self_test(root: Path) -> None:
    manifest = load_json(root / "INPUTS_MANIFEST.json")
    mutant_manifest = copy.deepcopy(manifest)
    mutant_manifest["files"][0]["bytes"] += 1
    try:
        verify_inputs(root, mutant_manifest)
    except ReplayError:
        pass
    else:
        raise AssertionError("A4 manifest byte tamper was accepted")

    shard_paths = sorted((root / "results/raw/shards").glob("*.json"))
    with tempfile.TemporaryDirectory(prefix="a4-shard-tamper-") as temp:
        mutant = load_json(shard_paths[0])
        mutant["scientific_run_valid"] = False
        mutant_path = Path(temp) / shard_paths[0].name
        mutant_path.write_bytes(canonical_bytes(mutant) + b"\n")
        paths = [mutant_path, *shard_paths[1:]]
        try:
            replay_from_paths(root, paths)
        except Exception as error:
            require(
                "shard invalid" in str(error),
                f"unexpected A4 shard-tamper failure: {error}",
            )
        else:
            raise AssertionError("A4 shard validity tamper was accepted")
    print("PASS A4 manifest and raw-shard adversarial tests")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    replay(root)
    if args.self_test:
        self_test(root)


if __name__ == "__main__":
    main()
