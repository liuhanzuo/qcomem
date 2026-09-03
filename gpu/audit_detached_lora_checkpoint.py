from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def audit_checkpoint(
    path: Path,
    *,
    expected_sha256: str | None,
    expected_world_size: int,
    expected_modules: int,
    expected_query_positions: int,
) -> dict[str, Any]:
    actual_sha256 = sha256_file(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata", {})
    semantics = metadata.get("semantics", {})
    gradient = metadata.get("last_gradient_coverage", {})
    detached = metadata.get("last_detached_capability", {})
    state = payload.get("lora", {})
    a_tensors = {
        name: value for name, value in state.items() if name.endswith("lora_a")
    }
    b_tensors = {
        name: value for name, value in state.items() if name.endswith("lora_b")
    }
    b_nonzero_tensors = sum(
        int(torch.count_nonzero(value).item() > 0) for value in b_tensors.values()
    )
    b_nonzero_elements = sum(
        int(torch.count_nonzero(value).item()) for value in b_tensors.values()
    )
    b_l1 = sum(float(value.float().abs().sum().item()) for value in b_tensors.values())
    b_max = max(
        (float(value.float().abs().max().item()) for value in b_tensors.values()),
        default=0.0,
    )
    gradient_ranks = gradient.get("by_rank", [])
    cache_ranks = detached.get("by_rank", [])
    checks = {
        "sha256_matches": expected_sha256 is None
        or actual_sha256 == expected_sha256,
        "checkpoint_format": payload.get("format") == "qcomem_suffix_lora_v1",
        "step_is_one": int(payload.get("step", -1)) == 1
        and int(metadata.get("last_step", -1)) == 1,
        "cold_start": "warm_start" not in metadata and "resume" not in metadata,
        "world_size": metadata.get("world_size") == expected_world_size,
        "detached_execution": semantics.get("student_suffix_execution_option")
        == "detached-document-cache",
        "query_only_claim": semantics.get("document_cache_detached_before_query")
        is True
        and semantics.get("document_prefill_parameter_gradients_enabled") is False,
        "gradient_gate": gradient.get("hard_gate_passed") is True
        and gradient.get("gradient_scope") == "query_continuation_only"
        and len(gradient_ranks) == expected_world_size
        and all(
            row.get("module_count") == expected_modules
            and row.get("finite_module_count") == expected_modules
            and row.get("nonzero_module_count") == expected_modules
            for row in gradient_ranks
        ),
        "cache_immutability_gate": detached.get("hard_gate_passed") is True
        and len(cache_ranks) == expected_world_size
        and all(
            row.get("hard_gate_passed") is True
            and row.get("detached_cache_storage_disjoint") is True
            and row.get("detached_cache_all_tensors_grad_free") is True
            and row.get("original_cache_versions_unchanged") is True
            and row.get("query_positions_expected") == expected_query_positions
            and row.get("query_positions_observed") == expected_query_positions
            for row in cache_ranks
        ),
        "adapter_tensor_count": len(a_tensors) == expected_modules
        and len(b_tensors) == expected_modules,
        "adapter_finite": all(torch.isfinite(value).all().item() for value in state.values()),
        "optimizer_update_visible": b_nonzero_tensors == expected_modules
        and b_nonzero_elements > 0
        and bool(payload.get("optimizer", {}).get("state")),
        "test_v2_unused": metadata.get("test_v2_used") is False,
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checkpoint": str(path),
        "checkpoint_sha256": actual_sha256,
        "checks": checks,
        "world_size": metadata.get("world_size"),
        "installed_modules": len(metadata.get("adapter", {}).get("installed_modules", [])),
        "trainable_parameters": metadata.get("adapter", {}).get("trainable_parameters"),
        "gradient_scope": gradient.get("gradient_scope"),
        "finite_modules_by_rank": [
            row.get("finite_module_count") for row in gradient_ranks
        ],
        "nonzero_modules_by_rank": [
            row.get("nonzero_module_count") for row in gradient_ranks
        ],
        "cache_tensor_counts_by_rank": [
            row.get("document_cache_tensor_count") for row in cache_ranks
        ],
        "query_positions_by_rank": [
            row.get("query_positions_observed") for row in cache_ranks
        ],
        "lora_b_update": {
            "nonzero_tensors": b_nonzero_tensors,
            "nonzero_elements": b_nonzero_elements,
            "l1": b_l1,
            "max_abs": b_max,
        },
        "semantics": semantics,
        "test_v2_used": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit a cold-start detached-document-cache LoRA capability checkpoint"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--expected-world-size", type=int, default=8)
    parser.add_argument("--expected-modules", type=int, default=36)
    parser.add_argument("--expected-query-positions", type=int, default=128)
    args = parser.parse_args()
    result = audit_checkpoint(
        args.checkpoint,
        expected_sha256=args.expected_sha256,
        expected_world_size=args.expected_world_size,
        expected_modules=args.expected_modules,
        expected_query_positions=args.expected_query_positions,
    )
    atomic_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
