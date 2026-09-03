from __future__ import annotations

"""Build the byte-bound R33 formal execution protocol before GPU execution."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import r33_executor_core as core
import r33_fault_replay as replay


RUN_ID = "R33-FRESH-FAULTS-20260825B"
ROW_SHA256 = {
    replay.FAULT_IDS[0]: "2ee27893a09cc9198f227422ec9fda1de1bebf97cc31b35fc1cfce67f773b8f2",
    replay.FAULT_IDS[1]: "20bbf518f3d2f66577db3e850400407658c8029975e03f6509a3e08f75d18970",
    replay.FAULT_IDS[2]: "6e2b0b4cca4f8a3b72d26e2f13aa6a2a47c5791dd8df44c452fb99bd7d42f282",
    replay.FAULT_IDS[3]: "24c88a88ea2991d16f4e7e63c457fcf92d2a95650ef23232fcf2a1c24d7a64f7",
    replay.FAULT_IDS[4]: "6dfbea24d869efeb4881155dfae1d710109a40017cb25bb3e80f621c266ec80a",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build(repo: Path, execution_input: Path) -> dict[str, Any]:
    paper = repo / "paper_autonomous_multifork_iteration"
    scripts = paper / "scripts"
    author = paper / "evidence" / "r33_fresh_faults" / "author_freeze"
    gpu = repo / "gpu"
    paths = {
        "r33_executor_source_sha256": scripts / "r33_execute_fresh_faults.py",
        "r33_replay_source_sha256": scripts / "r33_fault_replay.py",
        "r33_core_source_sha256": scripts / "r33_executor_core.py",
        "r33_aggregate_source_sha256": scripts / "r33_aggregate_fresh_faults.py",
        "r33_prepare_source_sha256": scripts / "r33_prepare_fresh_faults.py",
        "r33_launcher_source_sha256": scripts / "r33_launch_fresh_faults.sh",
        "r33_mapping_test_source_sha256": scripts / "r33_test_fault_mapping.py",
        "r33_core_test_source_sha256": scripts / "r33_test_executor_core.py",
        "r29_stack_loader_source_sha256": gpu / "r29_execute_heldout_faults.py",
        "single_token_gdn_repair_source_sha256": gpu / "qcomem_single_token_gdn_ownership.py",
        "resident_source_sha256": gpu / "qcomem_vllm_paged_multifork_resident.py",
        "storage_witness_source_sha256": gpu / "qcomem_forkaudit_storage_witness.py",
        "paged_kernel_source_sha256": gpu / "qcomem_vllm_paged_kernel.py",
        "resident_runner_source_sha256": gpu / "run_qcomem_qwen35_vllm_paged_multifork_resident.py",
    }
    source_bindings = {key: sha256_file(path) for key, path in paths.items()}
    fault_bindings = {
        fault_id: {
            "fault_id": fault_id,
            "rank": rank,
            "expected_primary_gate": replay.EXPECTED_PRIMARY_GATES[fault_id],
            "fault_definition_sha256": ROW_SHA256[fault_id],
        }
        for rank, fault_id in enumerate(replay.FAULT_IDS)
    }
    protocol = {
        "schema_version": core.PROTOCOL_SCHEMA,
        "run_id": RUN_ID,
        "mode": "formal_fresh_faults",
        "candidate_output_seen_when_frozen": False,
        "fault_ids": list(replay.FAULT_IDS),
        "execution_input_sha256": sha256_file(execution_input),
        "source_bindings": source_bindings,
        "author_freeze_manifest_sha256": sha256_file(author / "MANIFEST.sha256"),
        "fault_bindings": fault_bindings,
        "lifecycle_capability_binding": {
            "case_disposal_operation": "unregister_backends_clear_live_refs_gc_empty_cache_cuda_synchronize",
            "case_disposal_receipt_schema": "forkaudit-r33-case-cleanup-v1",
            "python_reference_clear_only": False,
        },
        "claim_boundary": {
            "local_dry_run_is_scientific_evidence": False,
            "per_fault_outcomes_only": True,
            "population_detection_rate_allowed": False,
            "fixed_single_stack_only": True,
            "compiled_binary_identity_scope": "partial",
            "autotuning_choice_scope": "partial",
        },
    }
    core.validate_protocol(protocol)
    return protocol


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--repo-root", type=Path, required=True)
    value.add_argument("--execution-input", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    return value


if __name__ == "__main__":
    args = parser().parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite formal R33 protocol")
    protocol = build(args.repo_root.resolve(), args.execution_input.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(protocol, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "raw_sha256": sha256_file(args.output)}, sort_keys=True))
