from __future__ import annotations

"""CPU-only preregistration and aggregation for the RR4 detector matrix."""

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence


PREREG_SCHEMA = "forkaudit-detector-matrix-preregistration-v1"
RANK_SCHEMA = "forkaudit-detector-matrix-rank-v1"
AGGREGATE_SCHEMA = "forkaudit-detector-matrix-aggregate-v1"
MUTANT_IDS = tuple(f"M{index}" for index in range(1, 10))
ASSIGNMENT = {
    0: ("M1", "M9"),
    1: ("M2",),
    2: ("M3",),
    3: ("M4",),
    4: ("M5",),
    5: ("M6",),
    6: ("M7",),
    7: ("M8",),
}
EXPECTED_GATES = {
    "M1": "KV_RESERVATION_DISJOINT",
    "M2": "KV_SEQUENCE_ID",
    "M3": "KV_TAIL_COW",
    "M4": "gdn_completed_vs_base_disjoint",
    "M5": "gdn_completed_vs_peers_disjoint",
    "M6": "POSITION_CANONICAL_VALUES",
    "M7": "MASK_CONTRACT",
    "M8": "KERNEL_CALLABLE_ID",
    "M9": "KV_PAGED_VIEW",
}
TARGET_REQUESTS = {
    "M1": 1,
    "M2": 0,
    "M3": 0,
    "M4": 0,
    "M5": 1,
    "M6": 0,
    "M7": 0,
    "M8": 0,
    "M9": 0,
}
MEASURED_STEPS = {
    "M1": "one request-1 full-model step after reservation mutation",
    "M2": "one request-0 full-model step after ledger sequence mutation",
    "M3": "request-0, request-1, then request-0 continuation",
    "M4": "request-0 prefix then one request-0 continuation after alias",
    "M5": "request-0/request-1 prefixes then one request-1 continuation after alias",
    "M6": "one request-0 full-model step with first-layer post-RoPE position +1",
    "M7": "one request-0 full-model step with first-layer materialized all-true mask",
    "M8": "one request-0 full-model step after callable swap",
    "M9": "one request-0 full-model step after first-layer dense-key substitution",
}


class BuildError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BuildError(message)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_bytes(canonical_bytes(value) + b"\n")
    pending.replace(path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def check_file(path: Path, expected: str, label: str) -> bytes:
    require(re.fullmatch(r"[0-9a-f]{64}", expected) is not None, f"{label} SHA format")
    payload = path.read_bytes()
    require(sha256_bytes(payload) == expected, f"{label} SHA drift")
    return payload


def preregister(args: argparse.Namespace) -> dict[str, Any]:
    runner_sha = sha256_file(args.runner)
    original_manifest_sha = sha256_file(args.original_receipt_manifest)
    value = {
        "schema_version": PREREG_SCHEMA,
        "created_before_candidate_outputs": True,
        "original_rr2_run_id": args.original_rr2_run_id,
        "original_rr2_receipt_manifest_sha256": original_manifest_sha,
        "runner_sha256": runner_sha,
        "builder_sha256": sha256_file(Path(__file__).resolve()),
        "imported_rr2_code_ledger_sha256": args.imported_rr2_code_ledger_sha256,
        "mutant_assignment": {str(key): list(value) for key, value in ASSIGNMENT.items()},
        "mutant_ids": list(MUTANT_IDS),
        "expected_gate_ids": EXPECTED_GATES,
        "target_requests": TARGET_REQUESTS,
        "measured_steps": MEASURED_STEPS,
        "mutant_cell": {
            "resident_count": 2,
            "kv_policy": "vllm-q16-shared-document-reuse",
            "gdn_policy": "borrow-immutable-base-functional-rebind",
        },
        "cross_arm_clean_reference": {
            "resident_count": 2,
            "kv_policy": "vllm-q16-fresh-full-copy-control",
            "gdn_policy": "materialize-request-base-functional-rebind",
        },
        "cross_n_clean_reference": {
            "resident_count": 1,
            "kv_policy": "vllm-q16-shared-document-reuse",
            "gdn_policy": "borrow-immutable-base-functional-rebind",
        },
        "target_gate_suppression_rule": (
            "suppress exactly the mutant's preregistered gate; preserve every "
            "other runtime/ForkAudit failure"
        ),
        "detectors": [
            "token_only",
            "full_logit",
            "cross_arm",
            "cross_n",
            "existing_runtime_assertions",
            "each_named_forkaudit_gate",
        ],
        "missingness_policy": "not_evaluated is never converted to pass or not_caught",
    }
    write_json(args.output, value)
    return value


def original_receipts(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    manifest = load_json(args.original_receipt_manifest)
    require(
        manifest.get("schema_version") == "qcomem-forkaudit-detached-receipts-v1",
        "RR2 manifest schema",
    )
    rows: dict[str, dict[str, Any]] = {}
    for shard_ref in manifest["shards"]:
        path = args.original_rr2_root / "raw" / shard_ref["relative_path"]
        payload = path.read_bytes()
        require(len(payload) == shard_ref["bytes"], "RR2 shard bytes")
        require(sha256_bytes(payload) == shard_ref["sha256"], "RR2 shard SHA")
        shard = json.loads(payload)
        for mutant_id, case in shard["fault_campaign"]["mutants"].items():
            outcome = case["outcome"]
            rows[mutant_id] = {
                "rank": shard["rank"],
                "shard_relative_path": shard_ref["relative_path"],
                "shard_sha256": shard_ref["sha256"],
                "classification": outcome["classification"],
                "expected_gate_id": outcome["expected_gate_id"],
                "observed_gate_id": outcome["observed_gate_id"],
                "restoration_verified": outcome["restoration_verified"],
                "matched_clean_classification": case["matched_clean"]["outcome"][
                    "classification"
                ],
            }
    require(set(rows) == set(MUTANT_IDS), "RR2 mutant coverage")
    return rows


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    prereg_raw = check_file(
        args.preregistration, args.expected_preregistration_sha256, "preregistration"
    )
    prereg = json.loads(prereg_raw)
    require(prereg.get("schema_version") == PREREG_SCHEMA, "preregistration schema")
    require(prereg.get("runner_sha256") == args.expected_runner_sha256, "runner binding")
    original = original_receipts(args)
    matrix_rows = []
    rank_receipts = []
    for rank in range(8):
        path = args.rank_root / f"detector-matrix-rank-{rank}.json"
        payload = path.read_bytes()
        shard = json.loads(payload)
        require(shard.get("schema_version") == RANK_SCHEMA, "rank schema")
        require(shard.get("rank") == rank, "rank identity")
        require(
            shard.get("preregistration_sha256")
            == args.expected_preregistration_sha256,
            "rank preregistration binding",
        )
        rank_receipts.append(
            {
                "rank": rank,
                "relative_path": path.name,
                "bytes": len(payload),
                "sha256": sha256_bytes(payload),
            }
        )
        for row in shard["rows"]:
            mutant_id = row["mutant_id"]
            rr2_row = original[mutant_id]
            require(
                rr2_row["classification"] == "detected_expected_gate",
                f"{mutant_id} RR2 classification",
            )
            require(
                rr2_row["expected_gate_id"]
                == rr2_row["observed_gate_id"]
                == row["expected_gate_id"]
                == EXPECTED_GATES[mutant_id],
                f"{mutant_id} gate binding",
            )
            require(
                rr2_row["matched_clean_classification"] == "clean_pass",
                f"{mutant_id} matched clean",
            )
            merged = dict(row)
            merged["original_rr2_forkaudit_receipt"] = rr2_row
            matrix_rows.append(merged)
    require(
        [row["mutant_id"] for row in matrix_rows] == list(MUTANT_IDS),
        "aggregate mutant order/coverage",
    )
    summary = {
        "mutants": len(matrix_rows),
        "token_only_caught": sum(
            row["detectors"]["token_only"].get("caught") is True
            for row in matrix_rows
        ),
        "full_logit_caught": sum(
            row["detectors"]["full_logit"].get("caught") is True
            for row in matrix_rows
        ),
        "existing_runtime_caught": sum(
            row["detectors"]["existing_runtime_assertions"].get("caught") is True
            for row in matrix_rows
        ),
        "output_preserved": sum(
            row["output_preserving_status"]
            == "output_preserved_within_measured_horizon"
            for row in matrix_rows
        ),
        "output_changed": sum(
            row["output_preserving_status"]
            == "output_changed_within_measured_horizon"
            for row in matrix_rows
        ),
        "output_unobservable": sum(
            row["output_preserving_status"].startswith("not_observable")
            for row in matrix_rows
        ),
        "forkaudit_expected_gate_caught": sum(
            row["original_rr2_forkaudit_receipt"]["classification"]
            == "detected_expected_gate"
            for row in matrix_rows
        ),
    }
    result = {
        "schema_version": AGGREGATE_SCHEMA,
        "preregistration": prereg,
        "preregistration_sha256": args.expected_preregistration_sha256,
        "rank_receipts": rank_receipts,
        "rows": matrix_rows,
        "summary": summary,
        "limitations": [
            "Output-preserving labels are bounded to the preregistered per-mutant measured steps.",
            "Non-target ForkAudit gates marked not_separately_evaluated are unknown, not passes.",
            "Cross-N is not applicable when a fault targets request 1 because N=1 has no homologous request.",
            "The original RR2 raw receipt, not this bypass run, is authoritative for named-gate detection.",
        ],
    }
    write_json(args.output, result)
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--stage", choices=("preregister", "aggregate"), required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--runner", type=Path)
    result.add_argument("--original-rr2-run-id", default="372384bd37cf7640ca210537a4360e1a")
    result.add_argument("--original-receipt-manifest", type=Path, required=True)
    result.add_argument("--original-rr2-root", type=Path)
    result.add_argument("--imported-rr2-code-ledger-sha256", default="")
    result.add_argument("--preregistration", type=Path)
    result.add_argument("--expected-preregistration-sha256")
    result.add_argument("--expected-runner-sha256")
    result.add_argument("--rank-root", type=Path)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.stage == "preregister":
        require(args.runner is not None, "preregister runner path")
        preregister(args)
    else:
        require(
            args.original_rr2_root is not None
            and args.preregistration is not None
            and args.rank_root is not None,
            "aggregate paths",
        )
        aggregate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
