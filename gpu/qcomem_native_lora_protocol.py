from __future__ import annotations

"""Audited data and step-one contracts for native-cache Q-CoMem LoRA."""

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import torch


VIEW_SCHEMA = "qcomem-native-lora-domain-view-v1"
BOUNDARY_SCHEMA = "qcomem-domain-document-query-boundary-v1"


class NativeLoRAProtocolError(ValueError):
    pass


def stable_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def token_ids_sha256(values: Sequence[int]) -> str:
    return hashlib.sha256(stable_json(list(values)).encode("utf-8")).hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise NativeLoRAProtocolError(f"{label} must be one lowercase SHA256")
    return value


def symmetric_head_tail(values: Sequence[int], maximum: int) -> tuple[list[int], bool]:
    if maximum < 1:
        raise NativeLoRAProtocolError("maximum document tokens must be positive")
    values = list(values)
    if len(values) <= maximum:
        return values, False
    left = maximum // 2
    return values[:left] + values[-(maximum - left) :], True


def domain_view_record(
    row: dict[str, Any],
    *,
    max_document_tokens: int,
    max_query_tokens: int,
) -> dict[str, Any]:
    """Derive one answer-free document/query training view from a parent row."""

    if row.get("schema_version") != "qcomem-deployment-aware-example-v1":
        raise NativeLoRAProtocolError("parent example schema drifted")
    if row.get("source_split") != "train" or row.get("stratum") != "domain":
        raise NativeLoRAProtocolError("native LoRA views require official-train domain rows")
    boundary = row.get("deployment_boundary")
    if not isinstance(boundary, dict) or boundary.get("applicable") is not True:
        raise NativeLoRAProtocolError("domain deployment boundary is unavailable")
    document = boundary.get("document_input_ids")
    query = boundary.get("query_input_ids")
    if (
        not isinstance(document, list)
        or not document
        or not isinstance(query, list)
        or not query
        or not all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in document + query
        )
    ):
        raise NativeLoRAProtocolError("domain document/query token lists are invalid")
    labels = row.get("labels")
    input_ids = row.get("input_ids")
    if not isinstance(labels, list) or not isinstance(input_ids, list):
        raise NativeLoRAProtocolError("parent token/label arrays are missing")
    try:
        first_target = next(index for index, value in enumerate(labels) if value != -100)
    except StopIteration as error:
        raise NativeLoRAProtocolError("parent has no supervised target") from error
    if document + query != input_ids[:first_target]:
        raise NativeLoRAProtocolError("parent boundary does not reconstruct the prompt")
    expected_hashes = {
        "document_input_ids_sha256": token_ids_sha256(document),
        "query_input_ids_sha256": token_ids_sha256(query),
        "prompt_input_ids_sha256": token_ids_sha256(document + query),
    }
    if any(boundary.get(key) != value for key, value in expected_hashes.items()):
        raise NativeLoRAProtocolError("parent boundary token SHA256 drifted")
    if boundary.get("answer_or_eos_tokens_in_query") is not False:
        raise NativeLoRAProtocolError("query continuation may not contain answer/EOS targets")
    if len(query) > max_query_tokens:
        raise NativeLoRAProtocolError(
            f"query has {len(query)} tokens, exceeding the fail-closed limit "
            f"{max_query_tokens}; query semantics are never truncated"
        )
    document_view, truncated = symmetric_head_tail(document, max_document_tokens)
    example_id = _require_sha256(row.get("example_id"), "parent example_id")
    parent_document_sha = _require_sha256(
        boundary.get("document_input_ids_sha256"), "parent document SHA256"
    )
    parent_query_sha = _require_sha256(
        boundary.get("query_input_ids_sha256"), "parent query SHA256"
    )
    return {
        "schema_version": VIEW_SCHEMA,
        # PG19WindowDataset intentionally receives only the two execution
        # arrays. ``source_dataset`` preserves provenance without presenting
        # official-train rows as LongBench evaluation records.
        "id": example_id,
        "document_ids": document_view,
        "query_ids": list(query),
        "source_dataset": row.get("dataset"),
        "source_split": "train",
        "parent_example_id": example_id,
        "parent_source_id_sha256": _require_sha256(
            row.get("source_id_sha256"), "parent source SHA256"
        ),
        "parent_document_id_sha256": row.get("document_id_sha256"),
        "parent_prompt_sha256": _require_sha256(
            row.get("prompt_sha256"), "parent prompt SHA256"
        ),
        "parent_boundary": {
            "schema_version": BOUNDARY_SCHEMA,
            "document_input_ids_sha256": parent_document_sha,
            "query_input_ids_sha256": parent_query_sha,
            "prompt_input_ids_sha256": _require_sha256(
                boundary.get("prompt_input_ids_sha256"),
                "parent boundary prompt SHA256",
            ),
            "document_tokens": len(document),
            "query_tokens": len(query),
            "answer_or_eos_tokens_in_query": False,
        },
        "view": {
            "document_input_ids_sha256": token_ids_sha256(document_view),
            "query_input_ids_sha256": parent_query_sha,
            "document_tokens": len(document_view),
            "query_tokens": len(query),
            "document_truncated": truncated,
            "document_truncation": (
                "symmetric_head_tail_v1" if truncated else "none"
            ),
            "query_truncation": "forbidden",
        },
    }


def sampler_scheduled_records(
    records: Sequence[dict[str, Any]],
    *,
    seed: int,
    world_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Encode a deterministic epoch-zero order with longest examples at step one."""

    if len(records) < world_size or world_size < 1:
        raise NativeLoRAProtocolError("training view must fill the first global step")
    unique_ids = {record["id"] for record in records}
    if len(unique_ids) != len(records):
        raise NativeLoRAProtocolError("domain view repeats a parent example")
    ordered = sorted(
        records,
        key=lambda record: (
            -len(record["document_ids"]) - len(record["query_ids"]),
            record["id"],
        ),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    permutation = torch.randperm(len(records), generator=generator).tolist()
    encoded: list[dict[str, Any] | None] = [None] * len(records)
    for schedule_position, dataset_index in enumerate(permutation):
        encoded[dataset_index] = ordered[schedule_position]
    if any(record is None for record in encoded):
        raise AssertionError("sampler schedule did not fill every dataset index")
    first_step = ordered[:world_size]
    return list(encoded), {
        "kind": "torch_distributed_sampler_epoch0_inverse_permutation_v1",
        "seed": seed,
        "world_size": world_size,
        "shuffle": True,
        "first_step_example_ids": [record["id"] for record in first_step],
        "first_step_sequence_tokens": [
            len(record["document_ids"]) + len(record["query_ids"])
            for record in first_step
        ],
        "epoch0_sampler_permutation_sha256": token_ids_sha256(permutation),
    }


def view_summary(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise NativeLoRAProtocolError("domain view is empty")
    documents = sorted(len(record["document_ids"]) for record in records)
    queries = sorted(len(record["query_ids"]) for record in records)
    sequences = sorted(
        len(record["document_ids"]) + len(record["query_ids"])
        for record in records
    )
    return {
        "rows": len(records),
        "source_dataset_counts": dict(
            sorted(Counter(record["source_dataset"] for record in records).items())
        ),
        "document_tokens": {
            "min": documents[0],
            "median": documents[len(documents) // 2],
            "max": documents[-1],
        },
        "query_tokens": {
            "min": queries[0],
            "median": queries[len(queries) // 2],
            "max": queries[-1],
        },
        "sequence_tokens": {
            "min": sequences[0],
            "median": sequences[len(sequences) // 2],
            "max": sequences[-1],
        },
        "document_truncated_rows": sum(
            record["view"]["document_truncated"] for record in records
        ),
        "all_query_segments_untruncated": all(
            record["view"]["query_truncation"] == "forbidden"
            for record in records
        ),
    }


def evaluate_step1_gate(
    rank_records: Sequence[dict[str, Any]],
    metadata: dict[str, Any],
    *,
    expected_world_size: int,
    expected_modules: int,
    expected_parameter_tensors: int,
    minimum_headroom_bytes: int,
    expected_init_checkpoint_sha256: str,
) -> dict[str, Any]:
    """Make the global post-update decision for the in-job step-one gate."""

    gradient = metadata.get("last_gradient_coverage", {})
    cache = metadata.get("last_detached_capability", {})
    gradient_ranks = gradient.get("by_rank", [])
    cache_ranks = cache.get("by_rank", [])
    warm_start = metadata.get("warm_start", {})
    rank_ids = {record.get("rank") for record in rank_records}
    checks = {
        "world_size": len(rank_records) == expected_world_size
        and rank_ids == set(range(expected_world_size)),
        "warm_start_interface_checkpoint_200": warm_start.get("source_mode")
        == "interface"
        and warm_start.get("source_step") == 200
        and warm_start.get("checkpoint_sha256")
        == expected_init_checkpoint_sha256,
        "metadata_gradient_gate": gradient.get("step") == 1
        and gradient.get("hard_gate_passed") is True
        and len(gradient_ranks) == expected_world_size
        and all(
            row.get("module_count") == expected_modules
            and row.get("finite_module_count") == expected_modules
            and row.get("nonzero_module_count") == expected_modules
            for row in gradient_ranks
        ),
        "metadata_native_cache_gate": cache.get("step") == 1
        and cache.get("hard_gate_passed") is True
        and len(cache_ranks) == expected_world_size
        and all(
            row.get("execution") == "native-functional-cache"
            and row.get("hard_gate_passed") is True
            and row.get("original_cache_versions_unchanged") is True
            and row.get("all_cache_paths_rebound") is True
            and row.get("query_positions_expected", 0) > 0
            and row.get("query_positions_observed")
            == row.get("query_positions_expected")
            for row in cache_ranks
        ),
        "all_adapter_gradients_finite_nonzero": all(
            record.get("parameter_tensors") == expected_parameter_tensors
            and record.get("modules") == expected_modules
            and record.get("finite_gradient_tensors") == expected_parameter_tensors
            and record.get("nonzero_gradient_tensors") == expected_parameter_tensors
            for record in rank_records
        ),
        "all_adapter_updates_finite_nonzero": all(
            record.get("finite_update_tensors") == expected_parameter_tensors
            and record.get("nonzero_update_tensors") == expected_parameter_tensors
            for record in rank_records
        ),
        "memory_headroom": all(
            isinstance(record.get("headroom_bytes"), int)
            and record["headroom_bytes"] >= minimum_headroom_bytes
            for record in rank_records
        ),
        "test_v2_unused": metadata.get("test_v2_used") is False,
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "step": 1,
        "checks": checks,
        "expected_world_size": expected_world_size,
        "expected_modules": expected_modules,
        "expected_parameter_tensors": expected_parameter_tensors,
        "minimum_headroom_bytes": minimum_headroom_bytes,
        "minimum_observed_headroom_bytes": min(
            (int(record.get("headroom_bytes", -1)) for record in rank_records),
            default=-1,
        ),
        "rank_records": list(rank_records),
        "cache_gate_metadata": cache,
        "gradient_gate_metadata": gradient,
        "single_token_autograd_claimed": False,
        "supported_training_continuation": "multi_token_full_query_only",
    }


__all__ = [
    "BOUNDARY_SCHEMA",
    "NativeLoRAProtocolError",
    "VIEW_SCHEMA",
    "domain_view_record",
    "evaluate_step1_gate",
    "sampler_scheduled_records",
    "sha256_file",
    "stable_json",
    "symmetric_head_tail",
    "token_ids_sha256",
    "view_summary",
]
