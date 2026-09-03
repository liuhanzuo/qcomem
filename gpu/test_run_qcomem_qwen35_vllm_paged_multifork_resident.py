from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import run_qcomem_qwen35_vllm_paged_multifork_resident as runner
from qcomem_vllm_paged_fair_control import FRESH_CONTROL, SHARED_REUSE
from qcomem_vllm_paged_multifork_resident import MULTIFORK_COUNTS, MULTIFORK_PROTOCOL


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _snapshot(
    label: str,
    allocated: int,
    reserved: int,
    peak_allocated: int | None = None,
    peak_reserved: int | None = None,
) -> dict[str, int | str]:
    return {
        "label": label,
        "current_allocated_bytes": allocated,
        "current_reserved_bytes": reserved,
        "peak_allocated_bytes": allocated if peak_allocated is None else peak_allocated,
        "peak_reserved_bytes": reserved if peak_reserved is None else peak_reserved,
    }


def _phase(
    before: dict[str, object], after: dict[str, object], *, seconds: float = 0.1
) -> dict[str, object]:
    return {
        "seconds": seconds,
        "allocator_before": before,
        "allocator_after": after,
        **runner._allocator_delta(before, after),
    }


def _query_rows() -> list[dict[str, object]]:
    return [
        {
            "request_index": index,
            "source_token_offset": 4160 + index * 64,
            "query_tokens": 32,
            "query_token_ids_sha256": _digest(f"query-{index}"),
        }
        for index in range(32)
    ]


def _storage(count: int, policy: str, *, after: bool) -> dict[str, object]:
    rows = []
    for layer_idx in runner.FULL_LAYERS:
        source_private = count * 524288
        fresh_document = count * 8388608 if policy == FRESH_CONTROL else 0
        fresh_private = source_private if policy == FRESH_CONTROL else 0
        row = {
            "layer_idx": layer_idx,
            "resident_count": count,
            "block_bytes": 262144,
            "valid_document_payload_nbytes": 8386560,
            "source_document_allocated_nbytes": 8388608,
            "source_document_padding_nbytes": 2048,
            "source_private_reservation_nbytes": source_private,
            "source_total_arena_allocated_nbytes": 8388608 + source_private,
            "fresh_duplicate_document_allocated_nbytes": fresh_document,
            "fresh_duplicate_document_padding_nbytes": (
                count * 2048 if policy == FRESH_CONTROL else 0
            ),
            "fresh_duplicate_private_reservation_nbytes": fresh_private,
            "active_request_private_payload_nbytes": count * 339968 if after else 0,
            "active_request_private_allocated_page_nbytes": (
                count * 524288 if after else 0
            ),
            "active_request_private_blocks": count * 2 if after else 0,
            "request_private_reserved_unused_nbytes": 0 if after else source_private,
            "active_request_appended_tokens_sum": count * 39 if after else 0,
            "active_request_detached_tail_tokens_sum": count * 127 if after else 0,
            "partial_tail_staging_copy_nbytes": count * 260096 if after else 0,
            "request_block_table_accelerator_nbytes": count * 136,
            "source_document_table_accelerator_nbytes": 128,
            "fresh_document_table_accelerator_nbytes": (
                count * 128 if policy == FRESH_CONTROL else 0
            ),
            "source_cpu_reservation_metadata_nbytes": count * 16,
            "fresh_cpu_reservation_metadata_nbytes": (
                count * 16 if policy == FRESH_CONTROL else 0
            ),
            "physical_document_block_copy_nbytes_including_padding": fresh_document,
        }
        rows.append(row)
    keys = sorted(
        key
        for key in rows[0]
        if key.endswith("_nbytes")
        or key
        in (
            "active_request_private_blocks",
            "active_request_appended_tokens_sum",
            "active_request_detached_tail_tokens_sum",
            "physical_document_block_copy_nbytes_including_padding",
        )
    )
    totals = {key: sum(int(row[key]) for row in rows) for key in keys}
    analytic = (
        totals["source_total_arena_allocated_nbytes"]
        + totals["fresh_duplicate_document_allocated_nbytes"]
        + totals["fresh_duplicate_private_reservation_nbytes"]
    )
    persistent = totals["source_total_arena_allocated_nbytes"]
    requests = analytic - persistent
    return {
        "protocol": MULTIFORK_PROTOCOL,
        "policy": policy,
        "resident_count": count,
        "simultaneous_lifetime": True,
        "full_attention_layer_count": 10,
        "source_private_reservation_is_common_pack_capacity": True,
        "active_private_payload_is_subset_not_additive": True,
        "fresh_duplicate_pool_is_separate_from_source": policy == FRESH_CONTROL,
        "layers": rows,
        "totals": totals,
        "unique_storage": {
            "persistent_total_nbytes": persistent,
            "persistent_accelerator_nbytes": persistent,
            "requests_total_nbytes": requests,
            "requests_accelerator_nbytes": requests,
            "combined_unique_total_nbytes": analytic,
            "combined_unique_accelerator_nbytes": analytic,
        },
    }


def _group_audit(count: int, policy: str) -> dict[str, object]:
    fresh = policy == FRESH_CONTROL
    rows = [
        {
            "request_index": index,
            "document_block_copy_nbytes_including_padding": 83886080 if fresh else 0,
            "allocated_request_pool_nbytes": 89128960 if fresh else 0,
            "source_document_storage_shared": not fresh,
        }
        for index in range(count)
    ]
    return {
        "protocol": MULTIFORK_PROTOCOL,
        "policy": policy,
        "resident_count": count,
        "all_requests_materialized_before_measurement": True,
        "strong_reference_count": count,
        "rows": rows,
        "ownership": {
            "passed": True,
            "resident_count": count,
            "request_object_ids_pairwise_distinct": True,
            "request_sequence_ids_pairwise_distinct": True,
            "all_requests_strongly_referenced": True,
            "fresh_request_arena_storages_pairwise_disjoint": fresh,
            "fresh_private_id_namespace_is_per_arena": fresh,
            "reuse_requests_share_source_arena": not fresh,
            "private_physical_reservation_ids_pairwise_disjoint": not fresh,
        },
        "physical_document_block_copy_nbytes_including_padding": (
            count * 83886080 if fresh else 0
        ),
        "allocated_fresh_request_pool_nbytes": count * 89128960 if fresh else 0,
    }


def _kernel_identity(callable_id: int = 17) -> dict[str, object]:
    return {
        "callable_id": callable_id,
        "module": "vllm.attention.ops.triton_unified_attention",
        "qualname": "unified_attention",
        "signature": "(q, k, v, **kwargs)",
    }


def _intercepts(count: int, policy: str, identity: dict[str, object]) -> list[dict[str, object]]:
    result = []
    for request_index in range(count):
        calls = []
        for round_index in range(runner.FORMAL_NEW_TOKENS):
            query_tokens = 32 if round_index == 0 else 1
            kv_tokens = 4127 + round_index
            for layer_idx in runner.FULL_LAYERS:
                calls.append(
                    {
                        "layer_idx": layer_idx,
                        "request_index": request_index,
                        "resident_count": count,
                        "request_policy": policy,
                        "protocol": MULTIFORK_PROTOCOL,
                        "kernel_identity": identity,
                        "current_append_delta_tokens": query_tokens,
                        "query_tokens": query_tokens,
                        "kv_tokens": kv_tokens,
                        "physical_block_pool_shape": [
                            34 if policy == FRESH_CONTROL else 32 + 2 * count,
                            128,
                            2,
                            256,
                        ],
                        "active_block_table_shape": [1, 33],
                        "kernel_mode": runner.KERNEL_MODE,
                        "quantization": "Q16",
                        "fused_gpu_kernel_calls": 1,
                        "full_kv_concatenations": 0,
                        "full_document_staging_copy_nbytes": 0,
                        "partial_tail_staging_copy_nbytes": 260096,
                        "gqa_groups": 8,
                        "mask_contract": "prevalidated-no-padding-tail-causal",
                        "materialized_attention_mask_nbytes": 0,
                        "mask_validation_host_syncs": 0,
                        "position_ids_contract": "qwen3.5-text-tail-post-rope-v1",
                        "position_ids_validated": True,
                        "position_ids_semantically_consumed_upstream": True,
                        "position_ids_shape": [1, query_tokens],
                        "position_ids_dtype": "torch.int64",
                        "position_ids_expected_tail_start": kv_tokens - query_tokens,
                        "position_ids_expected_tail_end_exclusive": kv_tokens,
                        "position_ids_strict_tail_values_checked": False,
                        "position_ids_validation_host_syncs": 0,
                    }
                )
        result.append(
            {
                "verified": True,
                "request_index": request_index,
                "resident_count": count,
                "request_policy": policy,
                "protocol": MULTIFORK_PROTOCOL,
                "kernel_identity": identity,
                "same_unified_attention_kernel": True,
                "kernel_mode": runner.KERNEL_MODE,
                "initial_query_tokens": 32,
                "counts": {str(index): 8 for index in runner.FULL_LAYERS},
                "round_major_request_local_layer_order_verified": True,
                "total_calls": 80,
                "dense_fallback_calls": 0,
                "full_kv_concatenations": 0,
                "mask_contract": "prevalidated-no-padding-tail-causal",
                "position_ids_contract": "qwen3.5-text-tail-post-rope-v1",
                "calls": calls,
            }
        )
    return result


def _arm(count: int, policy: str, query_rows: list[dict[str, object]]) -> dict[str, object]:
    identity = _kernel_identity()
    baseline = _snapshot("baseline", 1000, 2000)
    prefill_before = _snapshot("prefill-before", 1000, 2000)
    prefill_after = _snapshot("prefill-after", 1100, 2200, 1200, 2400)
    pack_before = _snapshot("pack-before", 1100, 2200)
    pack_after = _snapshot("pack-after", 1200, 2400, 1300, 2600)
    setup_before = _snapshot("setup-before", 1200, 2400)
    setup_after = _snapshot("setup-after", 1300, 2600, 1400, 2800)
    generation_before = _snapshot("generation-before", 1300, 2600)
    production_steps = [
        {
            "round_index": round_index,
            "request_index": request_index,
            **_snapshot(
                f"production-{round_index}-{request_index}",
                1350,
                2700,
                1500 + round_index * count + request_index,
                3000 + round_index * count + request_index,
            ),
        }
        for round_index in range(8)
        for request_index in range(count)
    ]
    production_peak_allocated = max(row["peak_allocated_bytes"] for row in production_steps)
    production_peak_reserved = max(row["peak_reserved_bytes"] for row in production_steps)
    generation_after = _snapshot("generation-after", 1400, 2800)
    generation_only = {
        "allocator_before": generation_before,
        "allocator_after": generation_after,
        "current_allocated_delta_bytes": 100,
        "current_reserved_delta_bytes": 200,
        "peak_allocated_delta_bytes": production_peak_allocated - 1300,
        "peak_reserved_delta_bytes": production_peak_reserved - 2600,
        "production_absolute_peak_allocated_bytes": production_peak_allocated,
        "production_absolute_peak_reserved_bytes": production_peak_reserved,
        "exactness_diagnostics_excluded_from_peak": True,
        "setup_to_generation_current_continuity_verified": True,
        "all_n_objects_alive_at_snapshot": True,
    }
    setup_plus_generation = {
        "allocator_before": setup_before,
        "allocator_after": generation_after,
        "current_allocated_delta_bytes": 200,
        "current_reserved_delta_bytes": 400,
        "peak_allocated_delta_bytes": production_peak_allocated - 1200,
        "peak_reserved_delta_bytes": production_peak_reserved - 2400,
        "combined_absolute_peak_allocated_bytes": production_peak_allocated,
        "combined_absolute_peak_reserved_bytes": production_peak_reserved,
        "all_n_objects_alive_at_snapshot": True,
    }
    trajectories = [
        {
            "request_index": index,
            "query_token_ids_sha256": query_rows[index]["query_token_ids_sha256"],
            "generated_token_ids": [100 + index + step for step in range(8)],
            "full_vocab_step_logit_sha256": [
                _digest(f"logit-{index}-{step}") for step in range(8)
            ],
            "step_seconds": [0.01] * 8,
        }
        for index in range(count)
    ]
    return {
        "protocol": MULTIFORK_PROTOCOL,
        "policy": policy,
        "resident_count": count,
        "quantization": "Q16",
        "batch_per_request": 1,
        "all_requests_simultaneously_resident": True,
        "same_unified_attention_kernel": True,
        "kernel_identity": identity,
        "allocator_frozen_baseline": baseline,
        "baseline_before_common_build": baseline,
        "common_document_prefill": _phase(prefill_before, prefill_after),
        "common_q16_pack": {
            **_phase(pack_before, pack_after),
            "source_document_payload_nbytes": 83865600,
            "source_allocated_pool_nbytes": 83886080 + count * 5242880,
            "max_request_forks": count,
        },
        "resident_setup": {
            **_phase(setup_before, setup_after),
            "all_n_objects_alive_at_snapshot": True,
            "group_audit": _group_audit(count, policy),
        },
        "generation_only": generation_only,
        "setup_plus_generation": setup_plus_generation,
        "storage_before_generation": _storage(count, policy, after=False),
        "storage_after_generation": _storage(count, policy, after=True),
        "source_document_sha256_before": {
            str(index): _digest(f"source-{index}") for index in runner.FULL_LAYERS
        },
        "source_document_sha256_after": {
            str(index): _digest(f"source-{index}") for index in runner.FULL_LAYERS
        },
        "source_document_immutable": True,
        "persistent_gdn_before": {"sha256": _digest("persistent-gdn"), "tensor_count": 60},
        "persistent_gdn_after": {"sha256": _digest("persistent-gdn"), "tensor_count": 60},
        "persistent_gdn_immutable": True,
        "request_gdn_after_generation": [
            {"request_index": index, "sha256": _digest(f"gdn-{index}"), "tensor_count": 60}
            for index in range(count)
        ],
        "request_logical_kv_after_generation": [
            {
                "request_index": index,
                "layer_sha256": {
                    str(layer): _digest(f"kv-{index}-{layer}")
                    for layer in runner.FULL_LAYERS
                },
            }
            for index in range(count)
        ],
        "intercepts": _intercepts(count, policy, identity),
        "generation": {
            "scheduler": "single-cuda-stream-sequential-round-major",
            "concurrent_kernel_execution_claimed": False,
            "all_requests_resident_for_entire_schedule": True,
            "rounds": 8,
            "resident_count": count,
            "total_model_steps": count * 8,
            "wall_seconds": count * 8 * 0.02,
            "schedule": [
                {"round_index": round_index, "request_index": request_index}
                for round_index in range(8)
                for request_index in range(count)
            ],
            "trajectories": trajectories,
            "production_allocator_before_exactness": {
                "steps": production_steps,
                "peak_allocated_bytes": production_peak_allocated,
                "peak_reserved_bytes": production_peak_reserved,
                "exactness_diagnostics_excluded_from_peak": True,
            },
        },
        "claim_boundaries": {
            "single_stream_sequential_round_robin": True,
            "concurrent_kernel_throughput_claimed": False,
            "ttft_speedup_claimed": False,
            "raw_step_timing_is_diagnostic_single_observation_only": True,
            "round_robin_wall_includes_logit_digest_and_cpu_clone": True,
            "nvml_peak_measured": False,
            "downstream_quality_measured": False,
        },
    }


def _frozen_identity() -> dict[str, str]:
    return {field: _digest(field) for field in runner.FROZEN_IDENTITY_FIELDS}


def _query_bank(rank: int) -> dict[str, object]:
    rows = _query_rows()
    return {
        "source_object": f"train/book-{rank}.txt",
        "source_role": "same-pg19-train-book-raw-nonoverlapping-query-chunks",
        "synthetic_markers_used": False,
        "count": 32,
        "query_tokens": 32,
        "query_stride_tokens": 64,
        "document_start_token": rank * 10000,
        "document_end_token_exclusive": rank * 10000 + 4095,
        "query_bank_start_token": rank * 10000 + 4127,
        "pairwise_nonoverlapping": True,
        "pairwise_distinct": True,
        "rows": rows,
        "query_bank_sha256": _digest(f"bank-{rank}"),
    }


def _minimal_arm(count: int, policy: str, query_rows: list[dict[str, object]]) -> dict[str, object]:
    source_total = 83886080 + count * 5242880
    fresh_document = count * 83886080 if policy == FRESH_CONTROL else 0
    fresh_private = count * 5242880 if policy == FRESH_CONTROL else 0
    trajectories = [
        {
            "request_index": index,
            "query_token_ids_sha256": query_rows[index]["query_token_ids_sha256"],
            "generated_token_ids": [index + step for step in range(8)],
            "full_vocab_step_logit_sha256": [
                _digest(f"aggregate-logit-{index}-{step}") for step in range(8)
            ],
        }
        for index in range(count)
    ]
    return {
        "kernel_identity": _kernel_identity(),
        "generation": {"trajectories": trajectories},
        "request_gdn_after_generation": [
            {"request_index": index, "sha256": _digest(f"aggregate-gdn-{index}"), "tensor_count": 60}
            for index in range(count)
        ],
        "request_logical_kv_after_generation": [
            {"request_index": index, "layer_sha256": {"3": _digest(f"aggregate-kv-{index}")}}
            for index in range(count)
        ],
        "source_document_sha256_before": {"3": _digest("aggregate-source")},
        "persistent_gdn_before": {"sha256": _digest("aggregate-persistent"), "tensor_count": 60},
        "resident_setup": {
            "current_allocated_delta_bytes": count,
            "peak_allocated_delta_bytes": count + 1,
            "current_reserved_delta_bytes": count + 2,
            "peak_reserved_delta_bytes": count + 3,
        },
        "setup_plus_generation": {
            "current_allocated_delta_bytes": count + 4,
            "current_reserved_delta_bytes": count + 5,
            "peak_allocated_delta_bytes": count + 6,
            "peak_reserved_delta_bytes": count + 7,
        },
        "generation_only": {
            "allocator_after": {
                "current_allocated_bytes": count + 10,
                "current_reserved_bytes": count + 20,
            },
            "production_absolute_peak_allocated_bytes": count + 30,
            "production_absolute_peak_reserved_bytes": count + 40,
        },
        "storage_after_generation": {
            "totals": {
                "source_total_arena_allocated_nbytes": source_total,
                "source_document_allocated_nbytes": 83886080,
                "source_private_reservation_nbytes": count * 5242880,
                "fresh_duplicate_document_allocated_nbytes": fresh_document,
                "fresh_duplicate_private_reservation_nbytes": fresh_private,
                "active_request_private_payload_nbytes": count * 3399680,
                "physical_document_block_copy_nbytes_including_padding": fresh_document,
                "partial_tail_staging_copy_nbytes": count * 2600960,
            }
        },
    }


def _parity(count: int, query_rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "passed": True,
        "request_count": count,
        "all_request_token_trajectories_exact": True,
        "all_request_full_vocab_step_logits_exact": True,
        "all_request_full_vocab_step_logits_runtime_torch_equal": True,
        "all_request_logical_kv_exact": True,
        "all_request_gdn_state_exact": True,
        "rows": [
            {
                "request_index": index,
                "query_token_ids_sha256": query_rows[index]["query_token_ids_sha256"],
                "generated_tokens_exact": True,
                "full_vocab_step_logits_exact": True,
                "full_vocab_step_logits_runtime_torch_equal": True,
            }
            for index in range(count)
        ],
    }


def _shards() -> tuple[list[dict[str, object]], dict[str, str], list[dict[str, object]]]:
    identity = _frozen_identity()
    banks = [_query_bank(rank) for rank in range(8)]
    shards = []
    for rank in range(8):
        rows = []
        for count in MULTIFORK_COUNTS:
            fresh = _minimal_arm(count, FRESH_CONTROL, banks[rank]["rows"])
            reuse = _minimal_arm(count, SHARED_REUSE, banks[rank]["rows"])
            rows.append(
                {
                    "resident_count": count,
                    "arm_execution_order": list(runner._arm_execution_order(count)),
                    "query_bank_prefix_sha256": runner._sha256_bytes(
                        "".join(
                            row["query_token_ids_sha256"]
                            for row in banks[rank]["rows"][:count]
                        ).encode()
                    ),
                    "fresh": fresh,
                    "reuse": reuse,
                    "parity": _parity(count, banks[rank]["rows"]),
                }
            )
        shards.append(
            {
                "status": "completed_multifork_resident_pg19_shard",
                "passed": True,
                "protocol": MULTIFORK_PROTOCOL,
                "rank": rank,
                "world_size": 8,
                "window_index": rank,
                "query_tokens": 32,
                "generated_tokens": 8,
                "kernel_mode": runner.KERNEL_MODE,
                "quantization": "Q16",
                "resident_counts": list(MULTIFORK_COUNTS),
                "execution_order": list(runner.FORMAL_EXECUTION_ORDER),
                "document_tokens": 4095,
                "document_tail_tokens": 127,
                "windows_sha256": identity["pg19_windows_sha256"],
                "pg19_train_only": True,
                "longbench_consumed": False,
                "source_6_9_consumed": False,
                "source_68_99_consumed": False,
                "test_v2_consumed": False,
                "static": {**identity, "frozen_query_banks": banks},
                "data_audit": {
                    "data_sha256": identity["pg19_data_sha256"],
                    "manifest_sha256": identity["pg19_manifest_sha256"],
                    "records": 8,
                    "bucket": runner.PG19_BUCKET,
                    "prefix": runner.PG19_PREFIX,
                    "data_role": "pg19_train_development_calibration_only",
                    "longbench_labels_used": False,
                    "formal_validation_source_6_35_used": False,
                    "frozen_test_v2_source_68_99_used": False,
                },
                "kernel_identity": _kernel_identity(),
                "query_bank": banks[rank],
                "source_object": banks[rank]["source_object"],
                "rows": rows,
                "allocator_fresh_state": {},
                "claim_boundaries": {
                    "approximately_4k_non_aligned_document": True,
                    "aligned_4096_measured": False,
                    "single_stream_round_robin": True,
                    "real_parallel_kernel_execution": False,
                    "vllm_engine_scheduler_tested": False,
                    "multi_document_tested": False,
                    "downstream_quality_measured": False,
                    "ttft_or_throughput_speedup_claimed": False,
                    "nvml_peak_measured": False,
                },
            }
        )
    return shards, identity, banks


def _full_shards() -> tuple[list[dict[str, object]], dict[str, str], list[dict[str, object]]]:
    shards, identity, banks = _shards()
    query_rows = _query_rows()
    full_rows = []
    for count in MULTIFORK_COUNTS:
        full_rows.append(
            {
                "resident_count": count,
                "arm_execution_order": list(runner._arm_execution_order(count)),
                "query_bank_prefix_sha256": runner._sha256_bytes(
                    "".join(
                        row["query_token_ids_sha256"] for row in query_rows[:count]
                    ).encode()
                ),
                "fresh": _arm(count, FRESH_CONTROL, query_rows),
                "reuse": _arm(count, SHARED_REUSE, query_rows),
                "parity": _parity(count, query_rows),
            }
        )
    frozen = _snapshot("frozen", 1000, 2000)
    cleanup = [
        {"resident_count": count, "policy": policy, "after": frozen}
        for count in runner.FORMAL_EXECUTION_ORDER
        for policy in runner._arm_execution_order(count)
    ]
    for shard in shards:
        shard["rows"] = full_rows
        shard["allocator_fresh_state"] = {
            "max_n_warmup_completed": True,
            "frozen_baseline": frozen,
            "cleanup_after_each_arm": cleanup,
        }
    return shards, identity, banks


class MultiForkFormalRunnerTest(unittest.TestCase):
    def test_validate_arm_replays_raw_and_rejects_tamper_matrix(self) -> None:
        queries = _query_rows()
        baseline = _arm(1, SHARED_REUSE, queries)
        runner._validate_arm(baseline, count=1, policy=SHARED_REUSE, query_rows=queries)

        tamper_cases = {
            "arm-n-bool": lambda value: value.__setitem__("resident_count", True),
            "trajectory-token": lambda value: value["generation"]["trajectories"][0]["generated_token_ids"].__setitem__(0, True),
            "trajectory-logit": lambda value: value["generation"]["trajectories"][0]["full_vocab_step_logit_sha256"].__setitem__(0, "bad"),
            "logical-kv": lambda value: value["request_logical_kv_after_generation"][0]["layer_sha256"].__setitem__("3", "bad"),
            "gdn-empty": lambda value: value["request_gdn_after_generation"][0].__setitem__("tensor_count", 0),
            "query-provenance": lambda value: value["generation"]["trajectories"][0].__setitem__("query_token_ids_sha256", _digest("wrong-query")),
            "allocator-impossible": lambda value: value["resident_setup"]["allocator_after"].__setitem__("current_reserved_bytes", 1),
            "allocator-derived": lambda value: value["resident_setup"].__setitem__("current_allocated_delta_bytes", 999),
            "production-peak": lambda value: value["generation"]["production_allocator_before_exactness"]["steps"][1].__setitem__("peak_allocated_bytes", 9999),
            "call-delta": lambda value: value["intercepts"][0]["calls"][10].__setitem__("current_append_delta_tokens", 7),
            "storage-formula": lambda value: value["storage_after_generation"]["layers"][0].__setitem__("active_request_private_payload_nbytes", 1),
            "storage-n-bool": lambda value: value["storage_after_generation"]["layers"][0].__setitem__("resident_count", True),
            "unique-union": lambda value: value["storage_after_generation"]["unique_storage"].__setitem__("combined_unique_total_nbytes", 0),
            "timing-nan": lambda value: value["generation"]["trajectories"][0]["step_seconds"].__setitem__(0, float("nan")),
            "claim-boundary": lambda value: value["claim_boundaries"].__setitem__("ttft_speedup_claimed", True),
        }
        for name, mutate in tamper_cases.items():
            with self.subTest(name=name):
                candidate = copy.deepcopy(baseline)
                mutate(candidate)
                with self.assertRaises(RuntimeError):
                    runner._validate_arm(candidate, count=1, policy=SHARED_REUSE, query_rows=queries)

    def test_allocator_fresh_state_replays_cleanup_and_rejects_drift(self) -> None:
        frozen = _snapshot("frozen", 1000, 2000)
        rows = []
        cleanup = []
        for count in MULTIFORK_COUNTS:
            fresh = {"allocator_frozen_baseline": frozen, "baseline_before_common_build": frozen}
            reuse = {"allocator_frozen_baseline": frozen, "baseline_before_common_build": frozen}
            rows.append({"resident_count": count, "fresh": fresh, "reuse": reuse})
        for count in runner.FORMAL_EXECUTION_ORDER:
            for policy in runner._arm_execution_order(count):
                cleanup.append({"resident_count": count, "policy": policy, "after": frozen})
        shard = {
            "rows": rows,
            "allocator_fresh_state": {
                "max_n_warmup_completed": True,
                "frozen_baseline": frozen,
                "cleanup_after_each_arm": cleanup,
            },
        }
        runner._validate_shard_allocator(shard)
        drifted = copy.deepcopy(shard)
        drifted["allocator_fresh_state"]["cleanup_after_each_arm"][4]["after"]["current_allocated_bytes"] += 1
        with self.assertRaises(RuntimeError):
            runner._validate_shard_allocator(drifted)

    def test_aggregate_eight_shards_replays_provenance_parity_and_curves(self) -> None:
        shards, identity, banks = _shards()

        def aggregate(candidate: list[dict[str, object]]) -> dict[str, object]:
            with tempfile.TemporaryDirectory() as directory:
                paths = []
                for rank, shard in enumerate(candidate):
                    path = Path(directory) / f"rank-{rank}.json"
                    path.write_text(json.dumps(shard, sort_keys=True))
                    paths.append(path)
                with mock.patch.object(runner, "_validate_arm"), mock.patch.object(
                    runner, "_validate_shard_allocator"
                ):
                    return runner.aggregate_shards(
                        paths,
                        expected_frozen_identity=identity,
                        expected_query_banks=banks,
                    )

        summary = aggregate(shards)
        self.assertTrue(summary["passed"])
        self.assertEqual(summary["rank_count"], 8)
        self.assertEqual(len(summary["raw_shard_artifacts"]), 8)
        self.assertTrue(summary["cross_n_prefix_isolation_exact"])
        self.assertEqual(summary["frozen_identity"], identity)
        self.assertNotIn(
            "diagnostic_storage_inventory_combined_unique_accelerator_median_nbytes",
            summary["capacity_matrix"][0]["fresh"],
        )
        self.assertEqual(
            summary["rank_capacity_curves_and_fits"][0]["fits"]["controlled_pool_bytes_saved"]["slope_nbytes_per_request"],
            89128960.0,
        )

        tamper_cases = {
            "top-schema": lambda value: value[0].__setitem__("query_tokens", 31),
            "rank-bool": lambda value: value[0].__setitem__("rank", False),
            "data-audit": lambda value: value[0]["data_audit"].__setitem__("data_sha256", _digest("wrong-data")),
            "query-bank": lambda value: value[0]["query_bank"].__setitem__("query_bank_sha256", _digest("wrong-bank")),
            "query-prefix": lambda value: value[0]["rows"][0].__setitem__("query_bank_prefix_sha256", _digest("wrong-prefix")),
            "trajectory": lambda value: value[0]["rows"][0]["reuse"]["generation"]["trajectories"][0]["generated_token_ids"].__setitem__(0, 999),
            "logit": lambda value: value[0]["rows"][0]["reuse"]["generation"]["trajectories"][0]["full_vocab_step_logit_sha256"].__setitem__(0, _digest("wrong-logit")),
            "gdn": lambda value: value[0]["rows"][0]["reuse"]["request_gdn_after_generation"][0].__setitem__("sha256", _digest("wrong-gdn")),
            "kv": lambda value: value[0]["rows"][0]["reuse"]["request_logical_kv_after_generation"][0]["layer_sha256"].__setitem__("3", _digest("wrong-kv")),
            "capacity-fit": lambda value: value[0]["rows"][2]["fresh"]["storage_after_generation"]["totals"].__setitem__("fresh_duplicate_document_allocated_nbytes", 1),
            "cross-n-prefix": lambda value: (
                value[0]["rows"][1]["fresh"]["generation"]["trajectories"][0]["generated_token_ids"].__setitem__(0, 777),
                value[0]["rows"][1]["reuse"]["generation"]["trajectories"][0]["generated_token_ids"].__setitem__(0, 777),
            ),
        }
        for name, mutate in tamper_cases.items():
            with self.subTest(name=name):
                candidate = copy.deepcopy(shards)
                mutate(candidate)
                with self.assertRaises(RuntimeError):
                    aggregate(candidate)

    def test_unmocked_eight_shard_json_roundtrip_aggregate_and_call_tamper(self) -> None:
        shards, identity, banks = _full_shards()
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for rank, shard in enumerate(shards):
                path = Path(directory) / f"full-rank-{rank}.json"
                path.write_text(json.dumps(shard))
                paths.append(path)
            summary = runner.aggregate_shards(
                paths,
                expected_frozen_identity=identity,
                expected_query_banks=banks,
            )
            self.assertTrue(summary["passed"])
            self.assertTrue(summary["cross_n_prefix_isolation_exact"])
            self.assertEqual(len(summary["raw_shard_artifacts"]), 8)

            shards[0]["rows"][0]["fresh"]["intercepts"][0]["calls"][1][
                "current_append_delta_tokens"
            ] = 1
            paths[0].write_text(json.dumps(shards[0]))
            with self.assertRaisesRegex(RuntimeError, "append/query delta"):
                runner.aggregate_shards(
                    paths,
                    expected_frozen_identity=identity,
                    expected_query_banks=banks,
                )

    def test_q8_fails_before_any_path_or_cuda_access(self) -> None:
        args = mock.Mock()
        args.bits = 8
        with mock.patch.object(runner, "sha256_file") as file_hash, mock.patch.object(
            runner.torch.cuda, "is_initialized"
        ) as cuda_initialized:
            with self.assertRaisesRegex(RuntimeError, "Q16 only"):
                runner._validate_static(args)
        file_hash.assert_not_called()
        cuda_initialized.assert_not_called()


if __name__ == "__main__":
    unittest.main()
