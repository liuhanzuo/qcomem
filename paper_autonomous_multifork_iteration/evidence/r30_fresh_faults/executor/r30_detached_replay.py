from __future__ import annotations

"""Candidate-import-free detached replay for the R30 fresh-fault lane."""

import argparse
from array import array
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any


FAULT_F1 = "R30F1_BORROWED_GDN_DETACH_EQUAL"
FAULT_F2 = "R30F2_IMMUTABLE_KV_TRANSIENT_FLIP_RESTORE"
FAULT_F3 = "R30F3_RELEASE_BEFORE_FINAL_OBSERVATION"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def verify_sidecar(run_dir: Path, row: dict[str, Any]) -> bytes:
    path = run_dir / row["path"]
    raw = path.read_bytes()
    require(len(raw) == row["nbytes"], "sidecar nbytes mismatch")
    require(hashlib.sha256(raw).hexdigest() == row["sha256"], "sidecar SHA mismatch")
    require(row["dtype"] == "float32-little-endian", "sidecar dtype drift")
    require(len(raw) % 4 == 0, "sidecar is not float32 aligned")
    values = array("f")
    values.frombytes(raw)
    if sys.byteorder != "little":
        values.byteswap()
    require(all(math.isfinite(value) for value in values), "non-finite sidecar value")
    return raw


def replay_clean(run_dir: Path, result: dict[str, Any]) -> dict[str, Any]:
    require(result["status"] == "fresh_clean_passed", "clean status")
    require(result["classification"] == "clean_pass", "clean classification")
    require(result["all_gates_enabled"] is True, "clean gates disabled")
    require(result["false_positive"] is False, "clean false positive")
    comparisons = result["comparisons"]
    require(comparisons["greedy_token_exact"] is True, "clean token mismatch")
    require(comparisons["canonical_fp32_logits"]["byte_exact"] is True, "clean logit mismatch")
    require(comparisons["terminal_request_0_gdn_exact"] is True, "clean GDN mismatch")
    require(comparisons["terminal_logical_kv_exact"] is True, "clean KV mismatch")
    borrowed = verify_sidecar(run_dir, result["sidecars"]["borrowed"])
    control = verify_sidecar(run_dir, result["sidecars"]["control"])
    require(borrowed == control, "clean sidecars are not byte exact")
    require(
        result["borrowed_repaired"]["detached_in_process_replay"]["storage"]["passed"] is True,
        "clean in-process storage replay failed",
    )
    require(
        result["borrowed_repaired"]["detached_in_process_replay"]["binding"]["passed"] is True,
        "clean in-process binding replay failed",
    )
    require(result["borrowed_repaired"]["persistent_guard"]["passed"] is True, "clean persistent guard failed")
    require(result["allocator"]["exact"] is True, "clean allocator cleanup failed")
    require(all(value is True for key, value in result["reference_equivalence"].items() if key.endswith("_exact")), "reference clean mismatch")
    return {"sidecar_sha256": hashlib.sha256(borrowed).hexdigest(), "sidecar_nbytes": len(borrowed)}


def replay_f1(result: dict[str, Any]) -> dict[str, Any]:
    require(
        result["classification"] in {"existing_validator_rejection", "ordinary_assertion_or_exception"},
        "F1 unexpected classification",
    )
    witness = result["injector_witness"]
    require(witness is not None, "F1 missing injector witness")
    require(witness["nonce"] == "r30f1-borrowed-detach-9f7c2a48b16e03d5", "F1 nonce")
    require(witness["before_content_sha256"] == witness["after_content_sha256"], "F1 content changed")
    require(witness["pre_exact_alias"] is True and witness["post_base_disjoint"] is True, "F1 relation transform")
    require(witness["before_storage_token"] != witness["after_storage_token"], "F1 storage token unchanged")
    require(witness["changed_reference_count"] == 1, "F1 reference count")
    require(all(witness[key] is True for key in ("shape_exact", "stride_exact", "dtype_exact", "device_exact")), "F1 metadata drift")
    require(result["full_horizon_reached"] is False, "F1 unexpectedly reached horizon")
    if result["classification"] == "ordinary_assertion_or_exception":
        require(result["ordinary_exception"] is not None, "F1 ordinary exception absent")
    else:
        require(result["validator_exception"] is not None and result["catch_gate"], "F1 validator receipt absent")
    require(result["allocator_cleanup"]["exact"] is True, "F1 allocator cleanup")
    return {"classification": result["classification"], "witness_valid": True}


def replay_f2(run_dir: Path, result: dict[str, Any]) -> dict[str, Any]:
    witness = result["injector_witness"]
    require(witness is not None, "F2 missing injector witness")
    require(witness["nonce"] == "r30f2-kv-transient-5b81d43e7a20c6f9", "F2 nonce")
    require(witness["mutated_byte"] == (witness["saved_byte"] ^ 1), "F2 XOR transform")
    require(witness["pre_sha256"] != witness["mutated_sha256"], "F2 mutation absent")
    require(witness["pre_sha256"] == witness["restored_sha256"], "F2 restoration mismatch")
    require(
        witness["mutation_start_ns"] <= witness["mutation_observed_ns"] <= witness["restoration_complete_ns"],
        "F2 timestamp order",
    )
    require(
        witness["candidate_capture_count_before"] == witness["candidate_capture_count_after"]
        and witness["model_call_count_before"] == witness["model_call_count_after"],
        "F2 activity inside mutation interval",
    )
    require(witness["device_synchronization_bracketed"] is True, "F2 synchronization witness")
    require(result["allocator_cleanup"]["exact"] is True, "F2 allocator cleanup")
    if result["classification"] == "escape_full_horizon":
        require(result["status"] == "full_horizon_completed" and result["full_horizon_reached"] is True, "F2 horizon")
        semantic = result["semantic_comparisons"]
        require(semantic["greedy_token_exact"] is True, "F2 token mismatch")
        require(semantic["canonical_fp32_logits"]["byte_exact"] is True, "F2 logit mismatch")
        require(semantic["terminal_request_0_gdn_exact"] is True, "F2 GDN mismatch")
        require(semantic["terminal_logical_kv_exact"] is True, "F2 KV mismatch")
        raw = verify_sidecar(run_dir, result["sidecar"])
        clean = (run_dir / "clean" / "raw" / "borrowed-repaired-fp32-logits.bin").read_bytes()
        require(raw == clean, "F2 sidecar differs from clean")
        require(result["in_process_replay"]["storage"]["passed"] is True, "F2 storage replay")
        require(result["in_process_replay"]["binding"]["passed"] is True, "F2 binding replay")
        require(result["persistent_guard"]["passed"] is True, "F2 persistent guard")
    elif result["classification"] == "existing_validator_rejection":
        require(result["validator_exception"] is not None and result["catch_gate"], "F2 validator receipt")
    elif result["classification"] == "ordinary_assertion_or_exception":
        require(result["ordinary_exception"] is not None, "F2 ordinary receipt")
    else:
        raise RuntimeError("F2 unrecognized classification")
    return {"classification": result["classification"], "witness_valid": True}


def replay_f3(result: dict[str, Any]) -> dict[str, Any]:
    require(result["classification"] == "operationally_invalid_precondition", "F3 classification")
    require(result["status"] == "operationally_invalid_precondition", "F3 status")
    require(result["injector_witness"] is None, "F3 should not have an injector witness")
    require(result["full_horizon_reached"] is False, "F3 horizon")
    evidence = result["precondition_evidence"]
    require(evidence["clean_authoritative_release_transition_present"] is False, "F3 release precondition")
    capability = evidence["lifecycle_capability_binding"]
    require(
        capability["authoritative_request_release_operation"] is None
        and capability["authoritative_request_release_receipt_schema"] is None,
        "F3 preregistered release capability is not absent",
    )
    canonical = json.dumps(capability, sort_keys=True, separators=(",", ":")).encode()
    require(
        hashlib.sha256(canonical).hexdigest()
        == evidence["lifecycle_capability_binding_sha256"],
        "F3 lifecycle-capability receipt hash mismatch",
    )
    return {"classification": result["classification"], "precondition_replay_valid": True}


def run(args: argparse.Namespace) -> dict[str, Any]:
    candidate_prefixes = (
        "torch",
        "transformers",
        "vllm",
        "r29_",
        "r30_run_",
        "qcomem_",
        "run_qcomem_",
    )
    imported = sorted(
        name for name in sys.modules if name.startswith(candidate_prefixes)
    )
    require(not imported, f"candidate modules already imported: {imported}")
    prereg_path = args.run_dir / "preregistration.json"
    require(sha256_file(prereg_path) == args.expected_prereg_sha256, "prereg SHA drift")
    prereg = load_json(prereg_path)
    result = load_json(args.result)
    require(result["case_id"] == args.case_id, "case ID drift")
    require(result["run_id"] == prereg["run_id"], "run ID drift")
    require(result["source_bindings"] == prereg["source_bindings"], "source binding drift")
    if args.case_id == "clean":
        detail = replay_clean(args.run_dir, result)
    elif args.case_id == FAULT_F1:
        detail = replay_f1(result)
    elif args.case_id == FAULT_F2:
        detail = replay_f2(args.run_dir, result)
    elif args.case_id == FAULT_F3:
        detail = replay_f3(result)
    else:
        raise RuntimeError("unknown case ID")
    output = {
        "schema_version": "forkaudit-r30-fresh-fault-detached-replay-v1",
        "run_id": prereg["run_id"],
        "case_id": args.case_id,
        "status": "passed",
        "candidate_modules_imported": False,
        "candidate_module_names": [],
        "result_sha256": sha256_file(args.result),
        "preregistration_sha256": args.expected_prereg_sha256,
        "detail": detail,
        "claim_boundary": {
            "candidate_import_free": True,
            "independent_live_recapture": False,
            "trusts_archived_candidate_witness": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    return output


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--run-dir", type=Path, required=True)
    value.add_argument("--case-id", required=True)
    value.add_argument("--result", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--expected-prereg-sha256", required=True)
    return value


if __name__ == "__main__":
    run(parser().parse_args())
