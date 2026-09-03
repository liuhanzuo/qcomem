from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = "forkaudit-detector-two-fault-local-comparison-v1"
DEBUG_SCHEMA = "forkaudit-detector-m8-m9-debug-v1"
DEBUG_PREREG_SCHEMA = "forkaudit-detector-m8-m9-debug-preregistration-v1"


class BuildError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BuildError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    require(isinstance(value, dict), f"{path} must contain a JSON object")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write("\n")


def verify_ledger(root: Path, ledger: Path) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        ledger.read_text(encoding="utf-8").splitlines(), start=1
    ):
        require("  " in line, f"malformed artifact ledger line {line_number}")
        expected, relative = line.split("  ", 1)
        relative_path = Path(relative)
        require(
            not relative_path.is_absolute() and ".." not in relative_path.parts,
            f"unsafe artifact ledger path on line {line_number}",
        )
        path = root / relative_path
        require(path.is_file(), f"missing artifact: {relative}")
        actual = sha256_file(path)
        require(actual == expected, f"artifact SHA mismatch: {relative}")
        receipts.append(
            {
                "relative_path": relative,
                "bytes": path.stat().st_size,
                "sha256": actual,
            }
        )
    require(receipts, "artifact ledger is empty")
    return receipts


def verify_debug_row(
    root: Path,
    mutant_id: str,
    rank: int,
    expected_preregistration_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = root / "raw" / f"debug-{mutant_id}-rank-{rank}.json"
    value = load_json(path)
    require(value.get("schema_version") == DEBUG_SCHEMA, f"{mutant_id} schema")
    require(value.get("debug_only") is True, f"{mutant_id} debug-only label")
    require(
        value.get("formal_evidence_eligible") is False,
        f"{mutant_id} formal-evidence boundary",
    )
    require(value.get("debug_mutant") == mutant_id, f"{mutant_id} identity")
    require(value.get("rank") == rank, f"{mutant_id} rank binding")
    require(
        value.get("preregistration_sha256") == expected_preregistration_sha256,
        f"{mutant_id} preregistration binding",
    )
    require(value.get("all_debug_gates_passed") is True, f"{mutant_id} gates")
    gates = value.get("debug_gates")
    require(
        isinstance(gates, dict) and gates and all(item is True for item in gates.values()),
        f"{mutant_id} individual gates",
    )

    clean = value.get("matched_clean")
    require(isinstance(clean, dict), f"{mutant_id} matched-clean object")
    require(clean.get("status") == "completed", f"{mutant_id} matched clean")
    require(clean.get("restoration_verified") is True, f"{mutant_id} clean restore")
    tokens = clean.get("tokens")
    logit_digests = clean.get("full_logit_sha256")
    sidecars = clean.get("logit_sidecars")
    require(isinstance(tokens, list) and len(tokens) == 1, f"{mutant_id} clean token")
    require(
        isinstance(logit_digests, list) and len(logit_digests) == 1,
        f"{mutant_id} clean logit digest",
    )
    require(
        isinstance(sidecars, list) and len(sidecars) == 1,
        f"{mutant_id} clean sidecar",
    )
    sidecar = sidecars[0]
    require(sidecar.get("dtype") == "float32", f"{mutant_id} sidecar dtype")
    require(sidecar.get("shape") == [1, 248320], f"{mutant_id} sidecar shape")
    sidecar_path = (
        root
        / "raw"
        / f"debug-rank-{rank}-sidecars"
        / mutant_id
        / "matched-clean"
        / sidecar["relative_path"]
    )
    require(sidecar_path.is_file(), f"{mutant_id} sidecar file")
    require(sidecar_path.stat().st_size == sidecar["bytes"], f"{mutant_id} sidecar bytes")
    sidecar_sha256 = sha256_file(sidecar_path)
    require(sidecar_sha256 == sidecar["sha256"], f"{mutant_id} sidecar SHA")
    require(sidecar_sha256 == logit_digests[0], f"{mutant_id} logit/sidecar binding")

    mutant = value.get("mutant_target_gate_suppressed")
    require(isinstance(mutant, dict), f"{mutant_id} mutant object")
    require(mutant.get("status") == "runtime_abort", f"{mutant_id} abort status")
    require(mutant.get("restoration_verified") is True, f"{mutant_id} restore")
    suppressed = [
        item.get("gate_id") for item in mutant.get("suppressed_target_gate_events", [])
    ]
    if mutant_id == "M8":
        receipt = mutant.get("mutation_receipt", {})
        require(suppressed == ["KERNEL_CALLABLE_ID"], "M8 target gate suppression")
        require(mutant.get("error_type") == "AssertionError", "M8 abort type")
        require(
            mutant.get("error_message") == "matrix M8 sentinel executed",
            "M8 sentinel message",
        )
        require(receipt.get("sentinel_installed") is True, "M8 sentinel receipt")
        require(
            receipt.get("sentinel_is_mutant_ledger_only") is True,
            "M8 sentinel isolation",
        )
        downstream = {
            "status": "caught_by_debug_instrumentation",
            "detector": "mutant-only sentinel",
            "production_runtime_assertion": "not_evaluated",
            "reason": (
                "The sentinel is experiment instrumentation, not an existing "
                "production runtime detector."
            ),
        }
    else:
        receipt = mutant.get("mutation_receipt", {})
        require(suppressed == ["KV_PAGED_VIEW"], "M9 target gate suppression")
        require(mutant.get("error_type") == "QComemPagedKernelError", "M9 abort type")
        require(
            mutant.get("error_message")
            == "fused backend requires paired Q16 paged views",
            "M9 production assertion message",
        )
        require(receipt.get("cache_slot_after") == "torch.Tensor(dense-key)", "M9 dense key")
        require(
            receipt.get("paired_value") == "Q16KernelPagedTensorView(value)",
            "M9 paged value",
        )
        require(receipt.get("dense_key_bridge_count") == 1, "M9 bridge count")
        require(
            receipt.get("bridged_dense_key_sha256") == receipt.get("dense_key_sha256"),
            "M9 bridge identity",
        )
        downstream = {
            "status": "caught",
            "detector": "production fused-backend paired-view assertion",
            "production_runtime_assertion": "caught",
            "error_type": mutant["error_type"],
            "error_message": mutant["error_message"],
        }

    debug_receipt = {
        "relative_path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "matched_clean": {
            "status": "completed",
            "token": tokens[0],
            "full_logit_sha256": logit_digests[0],
            "sidecar": {
                "relative_path": str(sidecar_path.relative_to(root)),
                "bytes": sidecar["bytes"],
                "sha256": sidecar_sha256,
                "dtype": sidecar["dtype"],
                "shape": sidecar["shape"],
            },
            "restoration_verified": True,
        },
        "mutant": {
            "status": mutant["status"],
            "suppressed_target_gate": suppressed[0],
            "restoration_verified": True,
            "downstream": downstream,
        },
    }
    return value, debug_receipt


def build(args: argparse.Namespace) -> dict[str, Any]:
    require(
        sha256_file(args.artifact_ledger) == args.expected_artifact_ledger_sha256,
        "artifact ledger SHA",
    )
    artifact_receipts = verify_ledger(args.debug_root, args.artifact_ledger)

    require(
        sha256_file(args.preregistration) == args.expected_preregistration_sha256,
        "debug preregistration SHA",
    )
    prereg = load_json(args.preregistration)
    require(prereg.get("schema_version") == DEBUG_PREREG_SCHEMA, "prereg schema")
    require(prereg.get("created_before_debug_outputs") is True, "prereg timing")
    require(prereg.get("debug_only") is True, "prereg debug-only label")
    require(prereg.get("formal_evidence_eligible") is False, "prereg evidence boundary")
    require(prereg.get("path_rank_binding") == {"M8": 7, "M9": 0}, "rank plan")

    require(
        sha256_file(args.scope_amendment) == args.expected_scope_amendment_sha256,
        "scope amendment SHA",
    )
    scope = load_json(args.scope_amendment)
    require(scope.get("applies_to_trial_id") == 1862402, "scope trial binding")
    require(
        scope.get("scope", {}).get("included")
        == ["M8 and its matched-clean control", "M9 and its matched-clean control"],
        "scope included paths",
    )
    require(
        scope.get("scope", {}).get("excluded_now_and_final")
        == ["M1", "M2", "M3", "M4", "M5", "M6", "M7"],
        "scope excluded paths",
    )

    require(
        sha256_file(args.w_mutant_outcomes) == args.expected_w_mutant_outcomes_sha256,
        "W-run mutant outcomes SHA",
    )
    w_outcomes = load_json(args.w_mutant_outcomes)
    require(
        w_outcomes.get("schema_version") == "forkaudit-paper-mutant-extract-v1",
        "W-run extract schema",
    )
    w_rows = {row["mutant_id"]: row for row in w_outcomes.get("rows", [])}
    require({"M8", "M9"}.issubset(w_rows), "W-run M8/M9 rows")

    require(
        sha256_file(args.w_raw_validation) == args.expected_w_raw_validation_sha256,
        "W-run raw validation SHA",
    )
    w_raw_validation = load_json(args.w_raw_validation)
    require(
        w_raw_validation.get("schema_version")
        == "forkaudit-w-run-m8-m9-raw-validation-v1",
        "W-run raw validation schema",
    )
    require(
        w_raw_validation.get("all_shard_hashes_matched_manifest") is True,
        "W-run raw shard replay",
    )
    raw_rows = {
        row["mutant_id"]: row for row in w_raw_validation.get("rows", [])
    }
    require(set(raw_rows) == {"M8", "M9"}, "W-run raw M8/M9 coverage")

    rows: list[dict[str, Any]] = []
    for mutant_id, rank, expected_gate in (
        ("M8", 7, "KERNEL_CALLABLE_ID"),
        ("M9", 0, "KV_PAGED_VIEW"),
    ):
        _, debug_receipt = verify_debug_row(
            args.debug_root,
            mutant_id,
            rank,
            args.expected_preregistration_sha256,
        )
        w_row = w_rows[mutant_id]
        raw_row = raw_rows[mutant_id]
        require(
            w_row.get("mutant_classification") == "detected_expected_gate",
            f"{mutant_id} W-run classification",
        )
        require(
            w_row.get("expected_gate_id")
            == w_row.get("observed_gate_id")
            == expected_gate,
            f"{mutant_id} W-run gate binding",
        )
        require(
            w_row.get("matched_clean_classification") == "clean_pass",
            f"{mutant_id} W-run matched clean",
        )
        require(w_row.get("restoration_verified") is True, f"{mutant_id} W restore")
        require(
            raw_row.get("expected_shard_sha256")
            == raw_row.get("observed_shard_sha256"),
            f"{mutant_id} W raw shard SHA",
        )
        for field in (
            "rank",
            "matched_clean_classification",
            "mutant_classification",
            "expected_gate_id",
            "observed_gate_id",
            "restoration_verified",
        ):
            require(raw_row.get(field) == w_row.get(field), f"{mutant_id} W {field}")
        rows.append(
            {
                "mutant_id": mutant_id,
                "fault_class": (
                    "attention-callable identity drift"
                    if mutant_id == "M8"
                    else "dense-key/paged-value representation mismatch"
                ),
                "current_debug_evidence": debug_receipt,
                "detectors": {
                    "token_only": {
                        "status": "not_evaluated",
                        "caught": None,
                        "reason": "mutant aborted before producing a token",
                    },
                    "full_logit": {
                        "status": "not_evaluated",
                        "caught": None,
                        "reason": "mutant aborted before producing a full-logit sidecar",
                    },
                    "cross_arm": {
                        "status": "not_evaluated",
                        "caught": None,
                        "reason": "cross-arm comparator was outside the frozen two-path scope",
                    },
                    "cross_n": {
                        "status": "not_evaluated",
                        "caught": None,
                        "reason": "cross-N comparator was outside the frozen two-path scope",
                    },
                    "existing_runtime_assertions": (
                        debug_receipt["mutant"]["downstream"]
                        if mutant_id == "M9"
                        else {
                            "status": "not_evaluated",
                            "caught": None,
                            "reason": (
                                "M8 reached an injected mutant-only sentinel; this is not "
                                "an existing production runtime assertion."
                            ),
                        }
                    ),
                    "forkaudit_expected_gate": {
                        "status": "caught_in_prior_w_run",
                        "caught": True,
                        "gate_id": expected_gate,
                        "source_run_id": w_outcomes["source_run_id"],
                        "source_extract_sha256": args.expected_w_mutant_outcomes_sha256,
                        "source_raw_validation_sha256": args.expected_w_raw_validation_sha256,
                        "source_raw_shard": {
                            "relative_path": raw_row["shard_relative_path"],
                            "bytes": raw_row["shard_bytes"],
                            "sha256": raw_row["observed_shard_sha256"],
                        },
                        "matched_clean_classification": w_row[
                            "matched_clean_classification"
                        ],
                        "restoration_verified": w_row["restoration_verified"],
                    },
                },
                "output_preserving_status": "not_observable_due_to_preoutput_abort",
            }
        )

    result = {
        "schema_version": SCHEMA,
        "debug_only": True,
        "formal_evidence_eligible": False,
        "trial_id": 1862402,
        "scope": "two representative fault classes only (M8 and M9)",
        "preregistration_sha256": args.expected_preregistration_sha256,
        "scope_amendment_sha256": args.expected_scope_amendment_sha256,
        "artifact_ledger_sha256": args.expected_artifact_ledger_sha256,
        "w_raw_validation_sha256": args.expected_w_raw_validation_sha256,
        "artifact_receipts": artifact_receipts,
        "rows": rows,
        "summary": {
            "evaluated_fault_classes": 2,
            "matched_clean_controls_completed": 2,
            "matched_clean_fp32_sidecars_validated": 2,
            "forkaudit_expected_gate_caught_in_separate_w_run": 2,
            "production_runtime_assertion_caught_after_target_gate_suppression": 1,
            "debug_instrumentation_caught_after_target_gate_suppression": 1,
            "token_full_logit_cross_arm_cross_n_cells_evaluated": 0,
            "detection_rate": None,
        },
        "reporting_boundary": [
            "This is a local comparison for M8 and M9, not an M1--M9 detector matrix.",
            "The current Trial 1862402 is debug-only and is not retroactively relabeled as confirmatory formal evidence.",
            "M1--M7 detector cells are absent and must not be inferred from the W-run intended-gate receipts.",
            "The W-run named-gate lane and current downstream-failure lane are separate executions and are not pooled into a rate.",
            "M8's downstream sentinel is injected instrumentation, not an existing production runtime detector.",
            "M9's downstream failure is a production paired-view assertion.",
            "Output preservation is unobservable because both mutants abort before producing output.",
        ],
        "missingness_policy": "not_evaluated is unknown and is never converted to pass, not_caught, or caught",
    }
    write_json(args.output, result)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--debug-root", type=Path, required=True)
    value.add_argument("--artifact-ledger", type=Path, required=True)
    value.add_argument("--expected-artifact-ledger-sha256", required=True)
    value.add_argument("--preregistration", type=Path, required=True)
    value.add_argument("--expected-preregistration-sha256", required=True)
    value.add_argument("--scope-amendment", type=Path, required=True)
    value.add_argument("--expected-scope-amendment-sha256", required=True)
    value.add_argument("--w-mutant-outcomes", type=Path, required=True)
    value.add_argument("--expected-w-mutant-outcomes-sha256", required=True)
    value.add_argument("--w-raw-validation", type=Path, required=True)
    value.add_argument("--expected-w-raw-validation-sha256", required=True)
    value.add_argument("--output", type=Path, required=True)
    return value


if __name__ == "__main__":
    build(parser().parse_args())
