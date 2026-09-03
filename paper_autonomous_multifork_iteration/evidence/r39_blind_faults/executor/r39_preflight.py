#!/usr/bin/env python3
"""Freeze one fault's source-aware feasibility result before candidate lanes."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import traceback
from typing import Any

import r39_contract as contract


def _manifest_sha(path: Path) -> str:
    return contract.sha256_file(path)


def _bf11_cache_candidates(root: Path) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    if root.is_dir():
        for path in sorted(root.rglob("*.json"), key=lambda item: str(item)):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(value, dict):
                continue
            fields = ("name", "hash", "num_warps", "num_ctas", "num_stages")
            if not all(field in value for field in fields):
                continue
            directory = path.parent
            files = [item for item in directory.iterdir() if item.is_file()]
            if not any(item.suffix == ".cubin" for item in files):
                continue
            if not any(item.suffix == ".ptx" for item in files):
                continue
            signature = contract.sha256_json({field: value[field] for field in fields})
            bundle = [
                {"name": item.name, "sha256": contract.sha256_file(item)}
                for item in sorted(files, key=lambda child: child.name)
            ]
            groups.setdefault(signature, []).append({
                "metadata_path": path.relative_to(root).as_posix(),
                "artifact_id": contract.sha256_json(bundle),
                "kernel_name": value["name"],
            })
    compatible = [
        {"signature_sha256": signature, "artifacts": rows}
        for signature, rows in sorted(groups.items())
        if len({row["artifact_id"] for row in rows}) >= 2
    ]
    return {
        "cache_root": str(root),
        "metadata_bundle_count": sum(len(rows) for rows in groups.values()),
        "signature_count": len(groups),
        "multi_artifact_compatible_groups": compatible,
    }


def _input_rank_for_bf10(execution_input: dict[str, Any]) -> tuple[int, int, list[str]]:
    bank_path = Path(execution_input["data"]["frozen_query_banks_path"])
    banks = json.loads(bank_path.read_text(encoding="utf-8"))
    hashes = [str(row["document_token_ids_sha256"]) for row in banks]
    contract.require(len(hashes) == 8 and len(set(hashes)) == 8, "BF10 document identities")
    order = sorted(range(8), key=lambda index: hashes[index])
    return order[0], order[1], hashes


def _live_probe(
    *, fault_id: str, gpu_index: int, expected_gpu_uuid: str,
    execution_input: dict[str, Any], triton_cache_root: Path,
) -> tuple[bool, dict[str, Any], dict[str, Any] | None]:
    import r39_live_common as live

    input_rank = gpu_index
    bf10_selection = None
    if fault_id == "R39-BF10":
        a_rank, b_rank, hashes = _input_rank_for_bf10(execution_input)
        input_rank = a_rank
        bf10_selection = (a_rank, b_rank, hashes)
    runtime = live.load_runtime(
        input_rank=input_rank, expected_gpu_uuid=expected_gpu_uuid,
        execution_input=execution_input,
    )
    persistent = persistent_b = group = persistent_guard = request_guard = None
    ledgers: list[Any] = []
    backends: list[str] = []
    try:
        warmup = live.discarded_warmup(runtime)
        if fault_id == "R39-BF11":
            scan = _bf11_cache_candidates(triton_cache_root)
            eligible = bool(scan["multi_artifact_compatible_groups"])
            if eligible:
                # A future cache containing two real artifacts is not silently
                # accepted: selection also needs an already loadable alternate
                # CompiledKernel handle, which this fixed stack does not expose.
                return False, {
                    "input_rank": input_rank,
                    "warmup": warmup,
                    "artifact_scan": scan,
                    "distinct_loadable_alternate_handle_count": 0,
                }, {
                    "code": "NO_LOADABLE_NONCANONICAL_COMPILED_HANDLE",
                    "reason": (
                        "Disk bundles were found, but the frozen vLLM/Triton stack "
                        "exposes only the selected CompiledKernel instance and no "
                        "ABI-authenticated alternate executable handle.  Substituting "
                        "cache bytes is forbidden."
                    ),
                }
            return False, {
                "input_rank": input_rank,
                "warmup": warmup,
                "artifact_scan": scan,
                "distinct_loadable_alternate_handle_count": 0,
            }, {
                "code": "NO_DISTINCT_ABI_COMPATIBLE_COMPILED_ARTIFACT",
                "reason": "After warmup, no second distinct ABI-compatible compiled artifact existed for any captured signature.",
            }

        if fault_id == "R39-BF10":
            assert bf10_selection is not None
            a_rank, b_rank, hashes = bf10_selection
            (
                persistent, persistent_b, group, guard_a, guard_b,
                request_guard, identity,
            ) = live.build_two_document_case(
                runtime, execution_input, document_a_rank=a_rank,
                document_b_rank=b_rank, mutant=False,
            )
            a_gdn = live.gdn_digest_map(persistent, runtime.plan.linear_layer_indices)
            b_gdn = live.gdn_digest_map(persistent_b, runtime.plan.linear_layer_indices)
            distinct = a_gdn != b_gdn
            return distinct, {
                "input_rank": a_rank,
                "document_a_rank": a_rank,
                "document_b_rank": b_rank,
                "ordered_document_identity_sha256": [hashes[index] for index in sorted(range(8), key=lambda index: hashes[index])],
                "document_a_gdn_manifest_sha256": contract.sha256_json(a_gdn),
                "document_b_gdn_manifest_sha256": contract.sha256_json(b_gdn),
                "persistent_gdn_bases_distinct": distinct,
                "both_bases_resident_before_candidate_H0": True,
                "identity_probe": identity,
                "probe_semantic_outputs_persisted": False,
            }, None if distinct else {
                "code": "LOWEST_TWO_DOCUMENT_GDN_BASES_NOT_DISTINCT",
                "reason": "The frozen lowest two document identities did not yield distinct persistent GDN base digests.",
            }

        persistent, group, persistent_guard, request_guard = live.build_default_case(runtime)
        target_layer = min(int(item) for item in runtime.plan.full_attention_layer_indices)
        request = group.requests[0]
        if fault_id == "R39-BF01":
            arena = persistent.layers[target_layer].arena
            tail_block = int(arena.document_block_table[0, -1].item())
            preceding_block = int(arena.document_block_table[0, -2].item())
            tail = int(arena.document_length % arena.page_size)
            true_digest = contract.sha256_json({
                "k": live.tensor_sha(arena.key_cache[tail_block, :tail]),
                "v": live.tensor_sha(arena.value_cache[tail_block, :tail]),
            })
            wrong_digest = contract.sha256_json({
                "k": live.tensor_sha(arena.key_cache[preceding_block, :tail]),
                "v": live.tensor_sha(arena.value_cache[preceding_block, :tail]),
            })
            eligible = tail == 127 and true_digest != wrong_digest
            return eligible, {
                "input_rank": input_rank,
                "target_request": 0,
                "target_layer": target_layer,
                "tail_tokens": tail,
                "true_tail_digest": true_digest,
                "preceding_page_prefix_digest": wrong_digest,
                "digests_distinct": true_digest != wrong_digest,
                "target_private_reservation_id": int(request.layers[target_layer].sequence.reservations[0, 0]),
            }, None if eligible else {
                "code": "BF01_FROZEN_SOURCE_DIGEST_EQUAL_OR_GEOMETRY_DRIFT",
                "reason": "The exact 127-token wrong and true sources are not a distinct eligible pair.",
            }
        if fault_id == "R39-BF03":
            sequences = [request.layers[index].sequence for index in runtime.plan.full_attention_layer_indices]
            eligible = all(item.arena.max_append_tokens >= 39 for item in sequences)
            return eligible, {
                "input_rank": input_rank,
                "target_request": 0,
                "target_round": 7,
                "full_attention_layer_count": len(sequences),
                "all_layers_reserve_final_append": eligible,
            }, None if eligible else {
                "code": "BF03_FINAL_APPEND_NOT_RESERVED",
                "reason": "At least one full-attention layer has no real final append to roll back.",
            }
        if fault_id == "R39-BF07":
            selected = live.select_nonzero_base_slice(persistent, runtime.plan)
            return True, {"input_rank": input_rank, "selected_base_slice": selected}, None

        # BF04/BF05/BF06/BF08 resolve against the deterministic H4 state.  The
        # feasibility probe persists no logits and never uses token/logit values
        # for target choice.
        for request_index, item in enumerate(group.requests):
            ledger = live.resident.MultiForkHitLedger(
                runtime.plan, item, request_index=request_index,
                resident_count=2, request_policy=live.SHARED,
                expected_calls_per_layer=1, initial_query_tokens=32,
                kernel=runtime.kernel, strict_position_values=True,
            )
            ledgers.append(ledger)
            backends.append(live.resident.register_multifork_backend(ledger))
        for request_index in range(2):
            output, logits = live.model_call(
                runtime, group.requests[request_index], backends[request_index],
                runtime.queries[request_index],
            )
            del output, logits
        transitions = live.privatize_all_requests(persistent, group, runtime.plan)
        if fault_id == "R39-BF04":
            layer, family, state_index, current = live.first_gdn_coordinate(request, runtime.plan)
            base = getattr(persistent.layers[layer], family)[state_index]
            eligible = live.storage_key(current) != live.storage_key(base) and live.tensor_sha(current) != live.tensor_sha(base)
            return eligible, {
                "input_rank": input_rank,
                "coordinate": f"{layer}:{family}:{state_index}",
                "private_descriptor": live.tensor_descriptor(current),
                "base_descriptor": live.tensor_descriptor(base),
                "storage_disjoint": live.storage_key(current) != live.storage_key(base),
                "correct_and_stale_source_digests_distinct": live.tensor_sha(current) != live.tensor_sha(base),
                "transition_receipts": transitions,
            }, None if eligible else {
                "code": "BF04_STALE_AND_CORRECT_SOURCE_NOT_DISTINCT",
                "reason": "The frozen first GDN coordinate lacks the required disjoint unequal H1/H4 sources.",
            }
        if fault_id == "R39-BF05":
            left, right = live.select_gdn_permutation(request, runtime.plan)
            return True, {
                "input_rank": input_rank,
                "left_coordinate": f"{left[0]}:{left[1]}:{left[2]}",
                "right_coordinate": f"{right[0]}:{right[1]}:{right[2]}",
                "left_descriptor": live.tensor_descriptor(left[3]),
                "right_descriptor": live.tensor_descriptor(right[3]),
                "transition_receipts": transitions,
            }, None
        if fault_id == "R39-BF06":
            sequence = request.layers[target_layer].sequence
            suffix = live.kv_unused_suffix(sequence)
            selected = live.select_cross_family_coordinate(request, runtime.plan, suffix)
            return True, {
                "input_rank": input_rank,
                "kv_layer": target_layer,
                "kv_suffix": suffix,
                "gdn_coordinate": f"{selected[0]}:{selected[1]}:{selected[2]}",
                "gdn_descriptor": live.tensor_descriptor(selected[3]),
                "transition_receipts": transitions,
            }, None
        if fault_id == "R39-BF08":
            base_keys = {live.storage_key(row[3]) for row in live.gdn_coordinates(persistent, runtime.plan.linear_layer_indices)}
            candidates = [row for row in live.gdn_coordinates(request, runtime.plan.linear_layer_indices) if live.storage_key(row[3]) not in base_keys]
            contract.require(bool(candidates), "BF08 no exclusive private GDN storage")
            selected = candidates[0]
            return True, {
                "input_rank": input_rank,
                "gdn_coordinate": f"{selected[0]}:{selected[1]}:{selected[2]}",
                "descriptor": live.tensor_descriptor(selected[3]),
                "transition_receipts": transitions,
            }, None
        raise contract.ContractError(f"unhandled live preflight fault {fault_id}")
    finally:
        if backends:
            try:
                live.rr2._unregister_backends(backends)
            except Exception:
                pass
        ledgers = []
        backends = []
        persistent = persistent_b = group = persistent_guard = request_guard = runtime = None
        gc.collect()
        if "torch" in sys.modules:
            import torch
            if torch.cuda.is_initialized():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()


def run(args: argparse.Namespace) -> dict[str, Any]:
    contract.require(args.fault_id in contract.FAULT_IDS, "unknown frozen fault")
    contract.require(contract.FAULT_TO_GPU[args.fault_id] == args.gpu_index, "GPU/fault map drift")
    freeze = contract.verify_freeze(args.protocol, args.plan)
    contract.require(contract.sha256_file(args.execution_input) == contract.EXECUTION_INPUT_SHA256, "execution-input SHA drift")
    contract.verify_source_manifest(args.source_manifest, args.source_root)
    source_manifest_sha = _manifest_sha(args.source_manifest)
    execution_input = json.loads(args.execution_input.read_text(encoding="utf-8"))
    if args.fault_id in contract.STATIC_INELIGIBLE:
        reason = contract.STATIC_INELIGIBLE[args.fault_id]
        selector = {
            "default_input_rank": args.gpu_index,
            "source_aware_static_preflight": True,
            "exact_frozen_payload_implemented_as_substitute": False,
        }
        eligible = False
    else:
        try:
            eligible, selector, reason = _live_probe(
                fault_id=args.fault_id, gpu_index=args.gpu_index,
                expected_gpu_uuid=args.expected_gpu_uuid,
                execution_input=execution_input,
                triton_cache_root=args.triton_cache_root,
            )
        except BaseException as exc:
            # A probe failure is not an ineligibility amendment: it is an
            # operational-invalid pre-output stop and must remain visible.
            invalid = {
                "schema_version": "forkaudit-r39-preflight-operational-invalid-v1",
                "run_id": contract.RUN_ID,
                "fault_id": args.fault_id,
                "status": "operational_invalid_before_candidate_output",
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
                "outcome_substituted": False,
            }
            contract.atomic_json(args.output.with_name("preflight-operational-invalid.json"), invalid)
            raise
    value = contract.make_feasibility(
        fault_id=args.fault_id, freeze=freeze,
        selector_resolution=selector, eligible=eligible,
        ineligible_reason=reason,
        source_manifest_sha256=source_manifest_sha,
    )
    contract.atomic_json(args.output, value)
    contract.validate_feasibility(value, fault_id=args.fault_id, freeze=freeze)
    return value


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--fault-id", choices=contract.FAULT_IDS, required=True)
    value.add_argument("--gpu-index", type=int, choices=range(8), required=True)
    value.add_argument("--expected-gpu-uuid", required=True)
    value.add_argument("--protocol", type=Path, required=True)
    value.add_argument("--plan", type=Path, required=True)
    value.add_argument("--execution-input", type=Path, required=True)
    value.add_argument("--source-root", type=Path, required=True)
    value.add_argument("--source-manifest", type=Path, required=True)
    value.add_argument("--triton-cache-root", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    return value


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args()), sort_keys=True), flush=True)
