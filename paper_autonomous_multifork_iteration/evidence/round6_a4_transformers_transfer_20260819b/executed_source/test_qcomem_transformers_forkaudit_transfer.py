from __future__ import annotations

import copy
import ast
import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from build_qcomem_transformers_forkaudit_transfer_prereg import model_authority, verify_static_rebuild
from qcomem_transformers_forkaudit_transfer import (
    AGGREGATE_SCHEMA,
    FAULT_CONTRACT,
    PROTOCOL,
    SHARD_SCHEMA,
    STATIC_SCHEMA,
    TARGET_CONTRACT,
    TransferEvidenceError,
    aggregate_shards,
    build_target_rows,
    canonical_json_bytes,
    classify_detector_vector,
    compare_logit_steps,
    disjointness_receipt,
    iter_tensor_slots,
    sha256_bytes,
    sha256_json,
    sha256_file,
    state_content_receipt,
    storage_inventory,
    tensor_receipt,
    tensor_tree_receipt,
    validate_model_authority_receipt,
    validate_gpu_assignment,
    write_canonical_json,
)
from run_qcomem_transformers_forkaudit_transfer import t5_ordinary_exception_receipt


SALT = "cpu-fixture-common-storage-domain"
DOMAIN = sha256_bytes(SALT.encode())
ZERO = "0" * 64
MODEL_ID = "Qwen/Qwen3.5-35B-A3B"
MODEL_REVISION = "59d61f3ce65a6d9863b86d2e96597125219dc754"
LAYER_TYPES = ["full_attention" if index in range(3, 40, 4) else "linear_attention" for index in range(40)]


class Bundle:
    def __init__(self, rank: int, directory: Path) -> None:
        self.rank = rank
        self.directory = directory
        self.payload = bytearray()
        self.records: list[dict[str, Any]] = []
        self.keepalive: list[torch.Tensor] = []

    def add(self, record_id: str, tensor: torch.Tensor) -> str:
        value = tensor.float().contiguous()
        raw = value.numpy().astype("<f4", copy=False).tobytes()
        offset = len(self.payload)
        self.payload.extend(raw)
        self.records.append(
            {
                "record_id": record_id,
                "offset_bytes": offset,
                "nbytes": len(raw),
                "shape": list(value.shape),
                "dtype": "float32-le",
                "content_sha256": sha256_bytes(raw),
            }
        )
        return record_id

    def finish(self) -> dict[str, Any]:
        name = f"forkaudit-transformers-transfer-logits-rank-{self.rank}.bin"
        (self.directory / name).write_bytes(bytes(self.payload))
        return {
            "schema_version": "forkaudit-fp32-logit-bundle-v1",
            "logical_name": name,
            "bytes": len(self.payload),
            "sha256": sha256_bytes(bytes(self.payload)),
            "record_count": len(self.records),
            "records": self.records,
            "terminal_closure": {
                "first_offset_bytes": 0,
                "last_end_offset_bytes": len(self.payload),
                "exact_byte_coverage": True,
            },
        }


def logits(request: int, *, mutant: bool = False) -> list[torch.Tensor]:
    if mutant:
        return [torch.tensor([[3.0, 0.0, 0.0]]), torch.tensor([[0.0, 3.0, 0.0]])]
    return [
        torch.tensor([[0.0, 3.0 + request, 0.0]]),
        torch.tensor([[0.0, 0.0, 3.0 + request]]),
    ]


def semantic(
    bundle: Bundle,
    prefix: str,
    request: int,
    query_sha: str,
    values: list[torch.Tensor],
    *,
    lower_content: str,
    suffix_content: str,
    state_tag: str,
) -> dict[str, Any]:
    ids = [bundle.add(f"{prefix}/request-{request}/step-{step}", value) for step, value in enumerate(values)]
    return {
        "request_index": request,
        "query_token_ids_sha256": query_sha,
        "generated_token_ids": [int(value.argmax()) for value in values],
        "step_logit_sha256": [tensor_receipt(value)["content_sha256"] for value in values],
        "step_logit_record_ids": ids,
        "final_lower_state_sha256": sha256_bytes(state_tag.encode()),
        "final_lower_cache_content_sha256": lower_content,
        "final_suffix_cache_sha256": suffix_content,
    }


def snapshot(tensor: torch.Tensor | None, role: str) -> dict[str, Any]:
    root: list[torch.Tensor] = [] if tensor is None else [tensor]
    tree = tensor_tree_receipt(root)
    return {
        "tensor_count": tree["tensor_count"],
        "content_sha256": tree["content_sha256"],
        "storage": storage_inventory(root, salt=SALT, role=role),
    }


def combined(role: str, inventories: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for inventory in inventories for row in inventory["rows"]]
    return {
        "role": role,
        "storage_salt_domain_sha256": DOMAIN,
        "tensor_rows": len(rows),
        "rows": rows,
        "inventory_sha256": sha256_json(rows),
    }


def comparison(candidate: dict[str, Any], candidate_values: list[torch.Tensor], oracle: dict[str, Any], oracle_values: list[torch.Tensor]) -> dict[str, Any]:
    numeric = compare_logit_steps(candidate_values, oracle_values, relative_l2_threshold=0.005)
    token_match = candidate["generated_token_ids"] == oracle["generated_token_ids"]
    return {
        "request_index": candidate["request_index"],
        "token_match": token_match,
        "numeric": numeric,
        "passed": token_match and numeric["passed"],
    }


def call_row(index: int, request: int, phase: str, kind: str, before_length: int, count: int, before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    return {
        "call_index": index,
        "request_index": request,
        "phase": phase,
        "callable": (
            "TorchSplitCausalLM.continue_lower_replay"
            if kind == "lower"
            else "TorchSplitCausalLM.run_suffix_cached_last_logits"
        ),
        "layer_start": 0 if kind == "lower" else 7,
        "layer_end": 7 if kind == "lower" else 40,
        "position_offset": before_length,
        "current_length_before": before_length,
        "current_length_after": before_length + count,
        "input_tokens": count,
        "append_delta": count,
        "cache_before": before,
        "cache_after": after,
        "completed": True,
    }


def make_arm(bundle: Bundle, rank_input: dict[str, Any], fanout: int, arm_name: str, oracle_rows: list[dict[str, Any]], oracle_values: list[list[torch.Tensor]]) -> tuple[dict[str, Any], list[list[torch.Tensor]]]:
    persistent = arm_name == "persistent_fork"
    forbidden_tensor = torch.tensor([900.0 + fanout])
    bundle.keepalive.append(forbidden_tensor)
    forbidden = [storage_inventory([forbidden_tensor], salt=SALT, role=f"{arm_name}/persistent-base/lower-cache")] if persistent else []
    setup: list[dict[str, Any]] = []
    first: list[dict[str, Any]] = []
    final: list[dict[str, Any]] = []
    residuals: list[dict[str, Any]] = []
    semantics: list[dict[str, Any]] = []
    ledgers_by_request: list[list[dict[str, Any]]] = []
    persistent_residual_tensor = torch.tensor([777.0]) if persistent else None
    if persistent_residual_tensor is not None:
        bundle.keepalive.append(persistent_residual_tensor)
    persistent_residual = (
        storage_inventory([persistent_residual_tensor], salt=SALT, role=f"{arm_name}/persistent-base/document-residual")
        if persistent else None
    )
    for request in range(fanout):
        setup_tensor = torch.tensor([10.0 + request])
        first_lower = torch.tensor([20.0 + request])
        suffix_document = torch.tensor([30.0 + request])
        first_suffix = torch.tensor([40.0 + request])
        final_lower = torch.tensor([50.0 + request])
        final_suffix = torch.tensor([60.0 + request])
        bundle.keepalive.extend([setup_tensor, first_lower, suffix_document, first_suffix, final_lower, final_suffix])
        setup_inv = storage_inventory([setup_tensor], salt=SALT, role=f"{arm_name}/request-{request}/lower-cache/setup")
        first_lower_inv = storage_inventory([first_lower], salt=SALT, role=f"{arm_name}/request-{request}/lower-cache/first")
        first_suffix_inv = storage_inventory([first_suffix], salt=SALT, role=f"{arm_name}/request-{request}/suffix-cache/first")
        final_lower_inv = storage_inventory([final_lower], salt=SALT, role=f"{arm_name}/request-{request}/lower-cache/final")
        final_suffix_inv = storage_inventory([final_suffix], salt=SALT, role=f"{arm_name}/request-{request}/suffix-cache/final")
        setup.append(setup_inv)
        first.append(combined(f"{arm_name}/request-{request}/all-mutable-cache/first-transition", [first_lower_inv, first_suffix_inv]))
        final.append(combined(f"{arm_name}/request-{request}/all-mutable-cache/final", [final_lower_inv, final_suffix_inv]))
        residual_tensor = persistent_residual_tensor if persistent else torch.tensor([70.0 + request])
        bundle.keepalive.append(residual_tensor)
        residuals.append(storage_inventory([residual_tensor], salt=SALT, role=f"{arm_name}/request-{request}/document-residual"))

        lower_setup = snapshot(setup_tensor, f"{arm_name}/r{request}/c0/before")
        lower_first = snapshot(first_lower, f"{arm_name}/r{request}/c0/after")
        suffix_empty = snapshot(None, f"{arm_name}/r{request}/c1/before")
        suffix_doc = snapshot(suffix_document, f"{arm_name}/r{request}/c1/after")
        suffix_doc_again = snapshot(suffix_document, f"{arm_name}/r{request}/c2/before")
        suffix_first = snapshot(first_suffix, f"{arm_name}/r{request}/c2/after")
        lower_first_again = snapshot(first_lower, f"{arm_name}/r{request}/c3/before")
        lower_final = snapshot(final_lower, f"{arm_name}/r{request}/c3/after")
        suffix_first_again = snapshot(first_suffix, f"{arm_name}/r{request}/c4/before")
        suffix_final = snapshot(final_suffix, f"{arm_name}/r{request}/c4/after")
        ledgers_by_request.append(
            [
                call_row(-1, request, "first-query-lower", "lower", 256, 24, lower_setup, lower_first),
                call_row(-1, request, "suffix-document", "suffix", 0, 256, suffix_empty, suffix_doc),
                call_row(-1, request, "first-query-suffix", "suffix", 256, 24, suffix_doc_again, suffix_first),
                call_row(-1, request, "generated-step-0-lower", "lower", 280, 1, lower_first_again, lower_final),
                call_row(-1, request, "generated-step-0-suffix", "suffix", 280, 1, suffix_first_again, suffix_final),
            ]
        )
        values = oracle_values[request]
        row = semantic(
            bundle, f"fanout-{fanout}/{arm_name}", request,
            rank_input["queries"][request]["token_ids_sha256"], values,
            lower_content=lower_final["content_sha256"], suffix_content=suffix_final["content_sha256"],
            state_tag=f"clean-r{request}",
        )
        row.update(
            final_current_length=281,
            lower_cache_storage=final_lower_inv,
            suffix_cache_storage=final_suffix_inv,
        )
        semantics.append(row)
    ledger = []
    for request in range(fanout):
        ledger.extend(ledgers_by_request[request][:3])
    for request in range(fanout):
        ledger.extend(ledgers_by_request[request][3:])
    for index, row in enumerate(ledger):
        row["call_index"] = index
    setup_gate = disjointness_receipt(setup, forbidden=forbidden)
    first_gate = disjointness_receipt(first, forbidden=forbidden)
    final_gate = disjointness_receipt(final, forbidden=forbidden)
    if persistent:
        base_ids = {row["storage_id_sha256"] for row in persistent_residual["rows"]}
        residual_gate = {
            "predicate_id": "READ_ONLY_DOCUMENT_RESIDUAL_ALIASES_PERSISTENT_BASE",
            "tensor_pair_comparison_count": sum(len(item["rows"]) * len(persistent_residual["rows"]) for item in residuals),
            "passed": all({row["storage_id_sha256"] for row in item["rows"]} == base_ids for item in residuals),
        }
    else:
        residual_gate = disjointness_receipt(residuals)
    arm = {
        "arm": arm_name,
        "fanout": fanout,
        "state_construction": "one-persistent-prefix-then-LowerReplayState.fork" if persistent else "independent-write_lower_replay-per-request",
        "scheduler": "single-cuda-stream-request-index-interleaved",
        "setup_storage_inventories": setup,
        "persistent_forbidden_inventories": forbidden,
        "first_transition_combined_storage_inventories": first,
        "final_combined_storage_inventories": final,
        "document_residual_storage_inventories": residuals,
        "persistent_document_residual_inventory": persistent_residual,
        "setup_disjointness": setup_gate,
        "first_transition_disjointness": first_gate,
        "final_disjointness": final_gate,
        "document_residual_ownership": residual_gate,
        "adapter_call_ledger": ledger,
        "allocator_accounting_snapshots": [
            {"phase": phase, "allocated_bytes": 100, "reserved_bytes": 200, "max_allocated_bytes": 150, "max_reserved_bytes": 250}
            for phase in ("setup", "first_transition", "final")
        ],
        "semantics": semantics,
    }
    return arm, [oracle_values[index] for index in range(fanout)]


def fault_semantic(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: row[key]
        for key in (
            "request_index", "query_token_ids_sha256", "generated_token_ids", "step_logit_sha256",
            "step_logit_record_ids", "final_lower_state_sha256", "final_lower_cache_content_sha256",
            "final_suffix_cache_sha256",
        )
    }


def make_faults(bundle: Bundle, rank_input: dict[str, Any], clean_n1: dict[str, Any], oracle_rows: list[dict[str, Any]], oracle_values: list[list[torch.Tensor]]) -> list[dict[str, Any]]:
    contracts = {item["fault_id"]: item for item in FAULT_CONTRACT}
    clean_oracle = True
    clean_cross = True
    mutant_values = logits(0, mutant=True)

    pre_m = torch.tensor([1.0, 2.0])
    pre_b = torch.tensor([1.0, 2.0])
    bundle.keepalive.extend([pre_m, pre_b])
    pre_m_receipt, pre_b_receipt = tensor_receipt(pre_m), tensor_receipt(pre_b)
    m_storage_before = storage_inventory([pre_m], salt=SALT, role="T1-materialized-residual")
    b_storage_before = storage_inventory([pre_b], salt=SALT, role="T1-corrupted-base-residual")
    pre_m.neg_(); pre_b.neg_()
    post_m_receipt, post_b_receipt = tensor_receipt(pre_m), tensor_receipt(pre_b)
    m_storage_after = storage_inventory([pre_m], salt=SALT, role="T1-materialized-residual")
    b_storage_after = storage_inventory([pre_b], salt=SALT, role="T1-corrupted-base-residual")
    persistent_storage = storage_inventory([pre_b], salt=SALT, role="T1-persistent-fork-residual")
    t1_semantics = {}
    t1_comparisons = {}
    for arm in ("deep_materialized", "persistent_fork"):
        dummy_lower = sha256_bytes(b"fault-lower")
        dummy_suffix = sha256_bytes(b"fault-suffix")
        row = semantic(bundle, f"fault-T1/{arm}", 0, rank_input["queries"][0]["token_ids_sha256"], mutant_values, lower_content=dummy_lower, suffix_content=dummy_suffix, state_tag="fault-t1")
        t1_semantics[arm] = [fault_semantic(row)]
        t1_comparisons[arm] = [comparison(row, mutant_values, oracle_rows[0], oracle_values[0])]
    t1_vectors = {
        "matched_clean": {"INDEPENDENT_DENSE_SEMANTIC_ORACLE": True, "DEEP_MATERIALIZED_EQUALS_PERSISTENT_FORK": True},
        "mutant": {"INDEPENDENT_DENSE_SEMANTIC_ORACLE": False, "DEEP_MATERIALIZED_EQUALS_PERSISTENT_FORK": True},
    }
    t1 = {
        **contracts["T1"],
        "exercise_kind": "downstream_runtime_fault",
        "execution_outcome": {"completed": True, "outputs_available": True, "runtime_exception": None, "ordinary_assertion_triggered": False},
        "matched_clean": {"predicate_passed": True, "source": "clean N=1 dense-oracle receipts"},
        "mutant": {
            "injection": "digest-proven common-mode document-boundary residual content mutation in both arms",
            "injection_receipt": {
                "materialized_before": pre_m_receipt, "materialized_after": post_m_receipt,
                "corrupted_base_before": pre_b_receipt, "corrupted_base_after": post_b_receipt,
                "persistent_fork_after": post_b_receipt,
                "materialized_storage_before": m_storage_before, "materialized_storage_after": m_storage_after,
                "corrupted_base_storage_before": b_storage_before, "corrupted_base_storage_after": b_storage_after,
                "persistent_fork_storage": persistent_storage,
                "changed_identically": True, "persistent_aliases_corrupted_base": True,
            },
            "common_mode_cross_arm_exact": True,
            "cross_arm_semantics": t1_semantics,
            "oracle_comparisons": t1_comparisons,
            "predicate_passed": False,
        },
        "classification": classify_detector_vector(expected_predicate=contracts["T1"]["expected_predicate"], **t1_vectors),
        "detector_vector": t1_vectors,
        "fault_case_valid": True,
    }

    clean_a, clean_b, base = torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([3.0])
    bundle.keepalive.extend([clean_a, clean_b, base])
    clean_invs = [storage_inventory([clean_a], salt=SALT, role="T2-clean-request-0"), storage_inventory([clean_b], salt=SALT, role="T2-clean-request-1")]
    base_inv = storage_inventory([base], salt=SALT, role="T2-persistent-base")
    target_old = torch.tensor([4.0])
    bundle.keepalive.append(target_old)
    before = storage_inventory([target_old], salt=SALT, role="alias-target-cache")
    shared = torch.tensor([5.0])
    bundle.keepalive.append(shared)
    mutant_invs = [storage_inventory([shared], salt=SALT, role="T2-mutant-request-0"), storage_inventory([shared], salt=SALT, role="T2-mutant-request-1")]
    after = storage_inventory([shared], salt=SALT, role="alias-target-cache")
    t2_vectors = {"matched_clean": {contracts["T2"]["expected_predicate"]: True}, "mutant": {contracts["T2"]["expected_predicate"]: False}}
    t2 = {
        **contracts["T2"],
        "exercise_kind": "direct_contract_sensitivity",
        "execution_outcome": {"completed": True, "outputs_available": False, "runtime_exception": None, "ordinary_assertion_triggered": False},
        "matched_clean": {"inventories": clean_invs, "forbidden_inventories": [base_inv], "gate": disjointness_receipt(clean_invs, forbidden=[base_inv])},
        "mutant": {
            "binding": {
                "source_path": "root/0", "target_path": "root/0", "source_tensor": tensor_receipt(shared),
                "target_tensor_before": tensor_receipt(target_old),
                "target_inventory_before_sha256": before["inventory_sha256"],
                "target_inventory_after_sha256": after["inventory_sha256"], "mutated": True,
            },
            "inventories": mutant_invs, "forbidden_inventories": [base_inv],
            "gate": disjointness_receipt(mutant_invs, forbidden=[base_inv]),
        },
        "classification": classify_detector_vector(expected_predicate=contracts["T2"]["expected_predicate"], **t2_vectors),
        "detector_vector": t2_vectors, "fault_case_valid": True,
    }
    t3_vectors = {"matched_clean": {contracts["T3"]["expected_predicate"]: True}, "mutant": {contracts["T3"]["expected_predicate"]: False}}
    t3 = {
        **contracts["T3"],
        "exercise_kind": "direct_contract_sensitivity",
        "execution_outcome": {"completed": True, "outputs_available": False, "runtime_exception": None, "ordinary_assertion_triggered": False},
        "matched_clean": {"document_length": 256, "current_length": 256, "next_position": 256, "predicate_passed": True},
        "mutant": {"document_length": 256, "current_length_before": 256, "current_length_after": 257, "next_position_after": 257, "predicate_passed": False},
        "classification": classify_detector_vector(expected_predicate=contracts["T3"]["expected_predicate"], **t3_vectors),
        "detector_vector": t3_vectors, "fault_case_valid": True,
    }
    packed_pre, packed_post = sha256_bytes(b"packed-pre"), sha256_bytes(b"packed-post")
    t4_vectors = {"matched_clean": {contracts["T4"]["expected_predicate"]: True}, "mutant": {contracts["T4"]["expected_predicate"]: False}}
    t4 = {
        **contracts["T4"],
        "exercise_kind": "direct_contract_sensitivity",
        "execution_outcome": {"completed": True, "outputs_available": False, "runtime_exception": None, "ordinary_assertion_triggered": False},
        "matched_clean": {"expected_pre_content_sha256": packed_pre, "observed_clean_content_sha256": packed_pre, "packed_state_content_sha256": packed_pre, "predicate_passed": True},
        "mutant": {"target": "PackedLowerReplayState.document_residual.data[0]", "pre_value": 1.0, "post_value": 2.0, "packed_state_content_sha256": packed_post, "predicate_passed": False},
        "classification": classify_detector_vector(expected_predicate=contracts["T4"]["expected_predicate"], **t4_vectors),
        "detector_vector": t4_vectors, "fault_case_valid": True,
    }

    target = torch.tensor([1.0, 2.0])
    bundle.keepalive.append(target)
    target_pre = tensor_receipt(target)
    storage_pre = storage_inventory([target], salt=SALT, role="T5-mutated-cache-tensor")
    target[0] += 9
    target_post = tensor_receipt(target)
    storage_post = storage_inventory([target], salt=SALT, role="T5-mutated-cache-tensor")
    t5_semantics = {}
    t5_comparisons = {}
    for arm, values, tag in (
        ("deep_materialized", oracle_values[0], "t5-deep"),
        ("persistent_fork", mutant_values, "t5-mutant"),
    ):
        row = semantic(bundle, f"fault-T5/{arm}", 0, rank_input["queries"][0]["token_ids_sha256"], values, lower_content=sha256_bytes(tag.encode()+b"lower"), suffix_content=sha256_bytes(tag.encode()+b"suffix"), state_tag=tag)
        t5_semantics[arm] = [fault_semantic(row)]
        t5_comparisons[arm] = [comparison(row, values, oracle_rows[0], oracle_values[0])]
    t5_vectors = {
        "matched_clean": {"INDEPENDENT_DENSE_SEMANTIC_ORACLE": clean_oracle, "DEEP_MATERIALIZED_EQUALS_PERSISTENT_FORK": clean_cross, "STATE_CROSS_ARM": clean_cross, "DOWNSTREAM_OUTPUT_CONSISTENCY": True},
        "mutant": {"INDEPENDENT_DENSE_SEMANTIC_ORACLE": False, "DEEP_MATERIALIZED_EQUALS_PERSISTENT_FORK": False, "STATE_CROSS_ARM": False, "DOWNSTREAM_OUTPUT_CONSISTENCY": False},
    }
    t5 = {
        **contracts["T5"],
        "exercise_kind": "downstream_runtime_fault",
        "execution_outcome": {"completed": True, "outputs_available": True, "runtime_exception": None, "ordinary_assertion_triggered": False},
        "matched_clean": {"oracle_passed": True, "state_cross_arm_exact": True, "output_cross_arm_exact": True},
        "mutant": {
            "target_path": "root/0", "target_pre": target_pre, "target_post": target_post,
            "target_storage_before": storage_pre, "target_storage_after": storage_post,
            "one_element_delta": 9.0,
            "executions": {
                arm: {"completed": True, "outputs_available": True, "runtime_exception": None, "ordinary_assertion_triggered": False}
                for arm in ("deep_materialized", "persistent_fork")
            },
            "cross_arm_semantics": t5_semantics, "oracle_comparisons": t5_comparisons,
            "state_cross_arm_exact": False, "output_cross_arm_exact": False, "oracle_passed": False,
        },
        "classification": classify_detector_vector(expected_predicate=contracts["T5"]["expected_predicate"], **t5_vectors),
        "detector_vector": t5_vectors, "fault_case_valid": True,
    }
    return [t1, t2, t3, t4, t5]


def make_static() -> dict[str, Any]:
    config = {
        "world_size": 8, "pg19_train_books": 8, "document_tokens": 256, "query_tokens": 24,
        "fanouts": [1, 2], "split_depth": 7, "semantic_steps": 2, "window_stride": 197,
        "query_stride": 32, "candidate_windows_per_book": 8, "seed": 20260819,
        "scheduler": "single-cuda-stream-request-index-interleaved", "arms": ["deep_materialized", "persistent_fork"],
    }
    rank_inputs = []
    for rank in range(8):
        document = list(range(rank, rank + 256))
        queries = []
        for request in range(2):
            tokens = list(range(1000 + rank * 100 + request * 24, 1024 + rank * 100 + request * 24))
            queries.append({"request_index": request, "source_token_offset": rank * 197 + 280 + request * 32, "token_ids": tokens, "token_ids_sha256": tensor_receipt(torch.tensor([tokens], dtype=torch.int64))["content_sha256"]})
        base = {
            "rank": rank, "source_id": f"book-{rank}", "source_object": f"train/book-{rank}.txt",
            "document_start_token": rank * 197, "document_end_token_exclusive": rank * 197 + 256,
            "document_token_ids": document,
            "document_token_ids_sha256": tensor_receipt(torch.tensor([document], dtype=torch.int64))["content_sha256"],
            "queries": queries,
        }
        rank_inputs.append({**base, "rank_input_sha256": sha256_json(base)})
    artifact_names = ["chat_template.jinja", "config.json", "generation_config.json", "merges.txt", "model.safetensors.index.json", "tokenizer_config.json", "vocab.json"]
    weight_names = [f"model.safetensors-{index:05d}-of-00014.safetensors" for index in range(1, 15)]
    artifact_entries = [{"path": name, "sha256": sha256_bytes(name.encode()), "bytes": index + 1} for index, name in enumerate(artifact_names)]
    weight_entries = [{"path": name, "sha256": sha256_bytes(name.encode()), "bytes": index + 1} for index, name in enumerate(weight_names)]
    artifact = {"raw_sha256": "a" * 64, "file_count": 7, "normalized_entries_sha256": sha256_json(artifact_entries), "entries": artifact_entries}
    weight = {"raw_sha256": "c" * 64, "file_count": 14, "normalized_entries_sha256": sha256_json(weight_entries), "entries": weight_entries}
    return {
        "schema_version": STATIC_SCHEMA, "protocol": PROTOCOL, "created_before_gpu_execution": True, "source_manifest_raw_sha256": "e" * 64,
        "formal_config": config, "formal_config_sha256": sha256_json(config), "rank_inputs": rank_inputs,
        "rank_inputs_sha256": sha256_json(rank_inputs), "storage_receipt_salt": SALT,
        "dataset": {"bucket": "deepmind-gutenberg", "prefix": "train/", "records": 64, "data_sha256": "1" * 64, "manifest_sha256": "2" * 64, "test_or_validation_objects_used": False},
        "window_algorithm": {"implementation": "independent-bounded-PG19-raw-token-windows-v1", "selection_key": "sha256(seed|transformers-transfer-book|source_object)", "window_key": "sha256(seed|transformers-transfer-window|source_object)", "raw_int64_token_receipts": True},
        "oracle_contract": {"path": "cpu fixture", "independent_of_ownership_arms": True, "full_vocabulary_cpu_fp32_sidecars_required": True, "top1_exact_required": True, "relative_l2_threshold": 0.005},
        "environment_contract": {"python": "fixture", "torch": "fixture", "cuda": "fixture", "transformers": "fixture"},
        "hardware_contract": {"world_size": 8, "gpu_name": "NVIDIA H20-3e", "compute_capability": [9, 0], "bf16_required": True, "assignment_frozen_pre_output": True},
        "portable_record_mapping": {"identity": "adapted", "ownership": "adapted", "execution": "adapted", "accounting": "adapted", "tail_event": "not_applicable: fixture", "dispatch": "partial: fixture"},
        "target_contract": list(TARGET_CONTRACT),
        "fault_contract": list(FAULT_CONTRACT),
        "model": {"model_id": MODEL_ID, "model_revision": MODEL_REVISION, "model_artifact_ledger_raw_sha256": "a" * 64, "model_weight_ledger_raw_sha256": "c" * 64, "artifact_ledger_receipt": artifact, "weight_ledger_receipt": weight, "artifact_set_sha256": sha256_json(artifact_entries), "layer_types": LAYER_TYPES, "tokenizer_class": "fixture.Tokenizer"},
        "claim_boundary": {"same_model": f"{MODEL_ID}@{MODEL_REVISION}", "different_runtime": "Transformers DynamicCache through qcomem_torch.TorchSplitCausalLM", "tail_target": "not_applicable: fixture", "dispatch_target": "partial: fixture", "not_authorized": ["fixture"]},
    }


def make_gpu_assignment() -> dict[str, Any]:
    rows = [
        {"rank": rank, "visible_index": rank, "uuid": f"GPU-{rank:064x}", "name": "NVIDIA H20-3e", "total_memory_mib": 143000, "compute_capability": [9, 0], "bf16_supported": True}
        for rank in range(8)
    ]
    return {"schema_version": "forkaudit-transformers-gpu-assignment-v1", "world_size": 8, "hardware_contract": "NVIDIA H20-3e / compute capability 9.0 / BF16", "rows": rows, "rows_sha256": sha256_json(rows)}


def make_shard(rank: int, static: dict[str, Any], assignment: dict[str, Any], sidecar_dir: Path) -> dict[str, Any]:
    bundle = Bundle(rank, sidecar_dir)
    rank_input = static["rank_inputs"][rank]
    oracle_values = [logits(0), logits(1)]
    oracle_rows = []
    for request, values in enumerate(oracle_values):
        ids = [bundle.add(f"dense-oracle/request-{request}/step-{step}", value) for step, value in enumerate(values)]
        oracle_rows.append({"oracle_path": "AutoModelForImageTextToText.full_last_logits_dense_recompute", "generated_token_ids": [int(value.argmax()) for value in values], "step_logit_sha256": [tensor_receipt(value)["content_sha256"] for value in values], "step_logit_record_ids": ids, "semantic_steps": 2})
    fanouts = {}
    for fanout in (1, 2):
        arms = {}
        arm_logits = {}
        comps = {}
        for arm_name in ("deep_materialized", "persistent_fork"):
            arms[arm_name], arm_logits[arm_name] = make_arm(bundle, rank_input, fanout, arm_name, oracle_rows, oracle_values)
            comps[arm_name] = [comparison(row, arm_logits[arm_name][index], oracle_rows[index], oracle_values[index]) for index, row in enumerate(arms[arm_name]["semantics"])]
        fanouts[str(fanout)] = {"fanout": fanout, "arms": arms, "oracle_comparisons": comps, "cross_arm_exact": True, "all_storage_ownership_predicates_passed": True}
    base = SimpleNamespace(cache=[torch.tensor([800.0])], document_residual=torch.tensor([801.0]), depth=7, document_length=256, current_length=256)
    bundle.keepalive.extend([base.cache[0], base.document_residual])
    base_receipt = state_content_receipt(base)
    faults = make_faults(bundle, rank_input, fanouts["1"], oracle_rows, oracle_values)
    targets = build_target_rows({"frozen_identity": True, "prefix_immutability": True, "private_ownership": True, "dispatch_provenance": True, "cross_arm_equivalence": True, "cross_n_prefix_consistency": True})
    gpu = assignment["rows"][rank]
    return {
        "schema_version": SHARD_SCHEMA, "protocol": PROTOCOL, "status": "completed", "rank": rank,
        "world_size": 8, "scientific_run_valid": True, "passed": True,
        "static_manifest_raw_sha256": "f" * 64, "source_manifest_raw_sha256": "e" * 64,
        "model_artifact_ledger_raw_sha256": "a" * 64, "model_weight_ledger_raw_sha256": "c" * 64,
        "gpu_assignment_raw_sha256": "9" * 64, "formal_config_sha256": static["formal_config_sha256"],
        "model_identity": {"model_id": MODEL_ID, "model_revision": MODEL_REVISION, "artifact_ledger": static["model"]["artifact_ledger_receipt"], "weight_ledger": static["model"]["weight_ledger_receipt"], "authority_raw_sha256": "8" * 64, "authority_stat_validation": {"verified": True, "file_count": 21, "stat_snapshot_sha256": "7" * 64}},
        "input": {"pg19_train_only": True, "source_object": rank_input["source_object"], "source_id": rank_input["source_id"], "document_start_token": rank_input["document_start_token"], "document_token_ids_sha256": rank_input["document_token_ids_sha256"], "query_token_ids_sha256": [item["token_ids_sha256"] for item in rank_input["queries"]], "rank_input_sha256": rank_input["rank_input_sha256"]},
        "hardware": {"cuda_visible_devices": gpu["uuid"], "uuid": gpu["uuid"], "name": gpu["name"], "total_memory_mib": gpu["total_memory_mib"], "compute_capability": gpu["compute_capability"], "bf16_supported": gpu["bf16_supported"]},
        "environment": {"python": "fixture", "torch": "fixture", "cuda": "fixture", "transformers": "fixture", "model_geometry": {"model_type": "qwen3_5_moe_text", "num_layers": 40, "layer_types": LAYER_TYPES, "split_depth": 7, "matches_frozen": True}},
        "dispatch_provenance": {"adapter": "qcomem_torch.TorchSplitCausalLM", "cache": "transformers.cache_utils.DynamicCache", "manual_suffix_method": "TorchSplitCausalLM.run_suffix_cached_last_logits", "layer_forward_types": ["fixture.Layer"], "same_receipt_for_both_arms": True, "compiled_kernel_fingerprint": None, "autotuning_choice_fingerprint": None},
        "persistent_base": {"before": base_receipt, "after": base_receipt, "storage": storage_inventory(base.cache, salt=SALT, role="persistent-document-lower-cache"), "content_immutable": True},
        "dense_oracle": {"contract": static["oracle_contract"], "semantics": oracle_rows, "all_clean_arms_passed": True},
        "fanouts": fanouts,
        "cross_n": {arm: {"passed": True, "compared_requests": 1} for arm in ("deep_materialized", "persistent_fork")},
        "targets": targets,
        "clean_audit": {"all_applicable_predicates_passed": True, "independent_dense_oracle_passed": True, "target_status_vector": [item["status"] for item in targets]},
        "fault_suite": faults, "claim_boundary": static["claim_boundary"], "logit_sidecar": bundle.finish(),
    }


class TransferFixtureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.sidecars = self.root / "sidecars"; self.sidecars.mkdir()
        self.shards_dir = self.root / "shards"; self.shards_dir.mkdir()
        self.static = make_static(); self.assignment = make_gpu_assignment()
        self.shards = [make_shard(rank, self.static, self.assignment, self.sidecars) for rank in range(8)]
        self.paths = []
        for rank, shard in enumerate(self.shards):
            path = self.shards_dir / f"forkaudit-transformers-transfer-shard-{rank}.json"
            write_canonical_json(path, shard); self.paths.append(path)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def aggregate(self) -> dict[str, Any]:
        return aggregate_shards(self.paths, static_manifest=self.static, sidecar_dir=self.sidecars, static_manifest_raw_sha256="f" * 64, source_manifest_raw_sha256="e" * 64, model_authority_raw_sha256="8" * 64, gpu_assignment=self.assignment, gpu_assignment_raw_sha256="9" * 64)

    def rewrite_rank0(self) -> None:
        write_canonical_json(self.paths[0], self.shards[0])

    def test_full_eight_rank_aggregate_replays_sidecars_and_faults(self) -> None:
        result = self.aggregate()
        self.assertEqual(result["schema_version"], AGGREGATE_SCHEMA)
        self.assertTrue(result["passed"])
        self.assertEqual(result["rank_count"], 8)
        self.assertEqual(result["fault_case_count"], 40)

    def assert_tamper_rejected(self, mutate) -> None:
        mutate(self.shards[0]); self.rewrite_rank0()
        with self.assertRaises(TransferEvidenceError): self.aggregate()

    def test_blind_replay_rejects_redundant_and_typed_tampers(self) -> None:
        cases = [
            lambda s: s["clean_audit"].__setitem__("all_applicable_predicates_passed", False),
            lambda s: s["targets"][0].__setitem__("status", "open"),
            lambda s: s["fault_suite"][0]["classification"].__setitem__("outcome", "escaped"),
            lambda s: s["fanouts"]["2"]["arms"]["deep_materialized"]["setup_disjointness"].__setitem__("passed", 1),
            lambda s: s["fanouts"]["1"]["arms"]["deep_materialized"]["adapter_call_ledger"][0].__setitem__("phase", "renamed"),
            lambda s: s["environment"].__setitem__("torch", "forged"),
            lambda s: s["hardware"].__setitem__("name", "NVIDIA A100"),
            lambda s: s["fanouts"]["2"]["arms"]["deep_materialized"]["setup_storage_inventories"][0].update(tensor_rows=0, rows=[], inventory_sha256=sha256_json([])),
            lambda s: s["fanouts"]["1"]["oracle_comparisons"]["deep_materialized"][0]["numeric"]["rows"][0].__setitem__("relative_l2", 0.1),
        ]
        originals = copy.deepcopy(self.shards[0])
        for mutate in cases:
            self.shards[0] = copy.deepcopy(originals)
            self.assert_tamper_rejected(mutate)

    def test_sidecar_byte_and_orphan_reference_tampers(self) -> None:
        path = self.sidecars / "forkaudit-transformers-transfer-logits-rank-0.bin"
        original = path.read_bytes(); path.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
        with self.assertRaises(TransferEvidenceError): self.aggregate()
        path.write_bytes(original)
        self.shards[0]["fault_suite"][4]["mutant"]["cross_arm_semantics"]["persistent_fork"][0]["step_logit_record_ids"][0] = self.shards[0]["fault_suite"][4]["mutant"]["cross_arm_semantics"]["deep_materialized"][0]["step_logit_record_ids"][0]
        self.rewrite_rank0()
        with self.assertRaises(TransferEvidenceError): self.aggregate()

    def test_tuple_before_replaceable_slot_regression(self) -> None:
        root = SimpleNamespace(first=(torch.tensor([1.0]),), second=[torch.tensor([2.0])])
        slots = list(iter_tensor_slots(root))
        self.assertFalse(slots[0].replaceable)
        replaceable = next(slot for slot in slots if slot.replaceable)
        replaceable.replace(torch.tensor([3.0]))
        self.assertEqual(root.second[0].item(), 3.0)

    def test_noncontiguous_storage_witness_fails_closed(self) -> None:
        base = torch.arange(10)
        with self.assertRaises(TransferEvidenceError):
            storage_inventory([base[::2]], salt=SALT, role="noncontiguous")

    def test_gpu_assignment_accepts_exact_h20_3e_and_rejects_plain_h20(self) -> None:
        assignment = make_gpu_assignment()
        validate_gpu_assignment(assignment)
        wrong_name = copy.deepcopy(assignment)
        wrong_name["rows"][0]["name"] = "NVIDIA H20"
        wrong_name["rows_sha256"] = sha256_json(wrong_name["rows"])
        with self.assertRaises(TransferEvidenceError):
            validate_gpu_assignment(wrong_name)

    def test_self_consistent_forged_static_rejected_by_independent_rebuild(self) -> None:
        forged = copy.deepcopy(self.static)
        forged["rank_inputs"][0]["document_token_ids"][0] += 1
        forged["rank_inputs"][0]["document_token_ids_sha256"] = tensor_receipt(
            torch.tensor([forged["rank_inputs"][0]["document_token_ids"]], dtype=torch.int64)
        )["content_sha256"]
        payload = {key: value for key, value in forged["rank_inputs"][0].items() if key != "rank_input_sha256"}
        forged["rank_inputs"][0]["rank_input_sha256"] = sha256_json(payload)
        forged["rank_inputs_sha256"] = sha256_json(forged["rank_inputs"])
        with self.assertRaises(TransferEvidenceError): verify_static_rebuild(forged, self.static)

    def test_model_authority_parses_ledgers_and_rejects_entry_or_stat_forgery(self) -> None:
        model = self.root / "model"; model.mkdir()
        artifact_names = [
            "chat_template.jinja", "config.json", "generation_config.json", "merges.txt",
            "model.safetensors.index.json", "tokenizer_config.json", "vocab.json",
        ]
        weight_names = [f"model.safetensors-{index:05d}-of-00014.safetensors" for index in range(1, 15)]
        for index, name in enumerate([*artifact_names, *weight_names]):
            path = model / name; path.write_bytes(f"fixture-{index}".encode()); path.chmod(0o444)
        artifact_ledger = self.root / "artifacts.sha256"
        weight_ledger = self.root / "weights.sha256"
        artifact_ledger.write_text("".join(f"{sha256_file(model/name)}  {name}\n" for name in artifact_names), encoding="utf-8")
        weight_ledger.write_text("".join(f"{sha256_file(model/name)}  {name}\n" for name in weight_names), encoding="utf-8")
        artifact_sha, weight_sha = sha256_file(artifact_ledger), sha256_file(weight_ledger)
        authority = model_authority(model, artifact_ledger, weight_ledger, artifact_sha256=artifact_sha, weight_sha256=weight_sha)
        self.assertEqual(authority["schema_version"], "forkaudit-transformers-model-authority-v2")
        self.assertTrue(all(row["regular_file"] and row["no_write_mode_bits"] for row in authority["stat_snapshot"]))
        if os.geteuid() == 0:
            # Root can satisfy os.access(W_OK) even for a 0444 file.  Authority is
            # deliberately based on the immutable mode-bit contract instead.
            self.assertTrue(os.access(model / artifact_names[0], os.W_OK))
        validate_model_authority_receipt(model, authority, artifact_ledger_path=artifact_ledger, weight_ledger_path=weight_ledger, artifact_ledger_raw_sha256=artifact_sha, weight_ledger_raw_sha256=weight_sha)
        writable = model / artifact_names[0]
        writable.chmod(0o644)
        with self.assertRaises(TransferEvidenceError):
            model_authority(model, artifact_ledger, weight_ledger, artifact_sha256=artifact_sha, weight_sha256=weight_sha)
        writable.chmod(0o444)
        forged_entry = copy.deepcopy(authority); forged_entry["weight_ledger"]["entries"][0]["sha256"] = "0" * 64
        with self.assertRaises(TransferEvidenceError):
            validate_model_authority_receipt(model, forged_entry, artifact_ledger_path=artifact_ledger, weight_ledger_path=weight_ledger, artifact_ledger_raw_sha256=artifact_sha, weight_ledger_raw_sha256=weight_sha)
        extra = model / "unledgered.bin"; extra.write_bytes(b"extra"); extra.chmod(0o444)
        stat = extra.stat()
        forged_stat = copy.deepcopy(authority)
        forged_stat["stat_snapshot"][0] = {
            "path": extra.name,
            "bytes": stat.st_size,
            "device": stat.st_dev,
            "inode": stat.st_ino,
            "ctime_ns": stat.st_ctime_ns,
            "regular_file": True,
            "no_write_mode_bits": True,
        }
        with self.assertRaises(TransferEvidenceError):
            validate_model_authority_receipt(model, forged_stat, artifact_ledger_path=artifact_ledger, weight_ledger_path=weight_ledger, artifact_ledger_raw_sha256=artifact_sha, weight_ledger_raw_sha256=weight_sha)

    def test_t5_exception_policy_rejects_evidence_and_resource_failures(self) -> None:
        for error in (TransferEvidenceError("evidence"), RuntimeError("runtime"), MemoryError("memory"), ValueError("value")):
            with self.assertRaises(type(error)):
                t5_ordinary_exception_receipt(error)
        receipt = t5_ordinary_exception_receipt(AssertionError("ordinary model assertion"))
        self.assertFalse(receipt["completed"])
        self.assertTrue(receipt["ordinary_assertion_triggered"])

    def test_ast_has_no_prior_forkaudit_or_vllm_ownership_import(self) -> None:
        gpu = Path(__file__).resolve().parent
        for name in (
            "qcomem_transformers_forkaudit_transfer.py",
            "run_qcomem_transformers_forkaudit_transfer.py",
            "build_qcomem_transformers_forkaudit_transfer_prereg.py",
        ):
            tree = ast.parse((gpu / name).read_text(encoding="utf-8"), filename=name)
            modules = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import): modules.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom): modules.append(node.module or "")
            self.assertFalse(any(module.startswith("qcomem_forkaudit") or module.startswith("vllm") for module in modules), modules)

    def test_detector_classification_has_reachable_negative_outcomes(self) -> None:
        self.assertEqual(
            classify_detector_vector(expected_predicate="expected", matched_clean={"expected": True, "other": True}, mutant={"expected": True, "other": False})["outcome"],
            "detected_wrong_predicate",
        )
        self.assertEqual(
            classify_detector_vector(expected_predicate="expected", matched_clean={"expected": True}, mutant={"expected": True})["outcome"],
            "escaped",
        )
        self.assertEqual(
            classify_detector_vector(expected_predicate="expected", matched_clean={"expected": False}, mutant={"expected": False})["outcome"],
            "clean_false_positive",
        )


if __name__ == "__main__":
    unittest.main()
