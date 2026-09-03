#!/usr/bin/env python3
"""Read-only CPU counterexamples for the R39 compiled-dispatch v6 verifier.

The script writes only inside TemporaryDirectory instances.  It imports the
frozen v6 verifier and its tiny fixture without modifying v6.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
from pathlib import Path


AUDIT_ROOT = Path(__file__).resolve().parent
EVIDENCE_ROOT = AUDIT_ROOT.parent
V6_ROOT = EVIDENCE_ROOT / "r39_primary_compiled_dispatch_v6"
sys.path.insert(0, str(V6_ROOT / "executed_source"))
sys.path.insert(0, str(V6_ROOT / "tests"))

import r39_compiled_dispatch_receipts as base  # noqa: E402
from r39_primary_compact_dispatch import (  # noqa: E402
    verify_payload,
    verify_primary_shard,
)
from test_r39_primary_compact_dispatch import (  # noqa: E402
    TINY,
    build_tiny,
    fixture_bindings,
)


def _runner_argv_sha256(argv: list[str]) -> str:
    raw = json.dumps(
        argv, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _tiny_shard(rank: int) -> dict[str, object]:
    calls = [
        {
            "request_index": 0,
            "layer_idx": 1,
            "query_tokens": 32,
            "physical_block_pool_shape": [34, 128, 2, 256],
            "active_block_table_shape": [1, 33],
            "kv_tokens": 4127,
            "softmax_scale": 0.0625,
        },
        {
            "request_index": 0,
            "layer_idx": 1,
            "query_tokens": 1,
            "physical_block_pool_shape": [34, 128, 2, 256],
            "active_block_table_shape": [1, 33],
            "kv_tokens": 4128,
            "softmax_scale": 0.0625,
        },
    ]
    ledger = {
        "request_index": 0,
        "total_calls": 2,
        "verified": True,
        "dense_fallback_calls": 0,
        "calls": calls,
    }
    return {
        "schema_version": "qcomem-forkaudit-review-shard-v1",
        "protocol": "qcomem-qwen35-forkaudit-review-revision-v1",
        "status": "completed_formal_gpu_shard",
        "rank": rank,
        "world_size": 8,
        "protocol_config": {
            "resident_counts": [1],
            "generation_steps": 2,
            "document_tokens": 3,
            "factorial_arm_ids": ["kv=tiny-kv|gdn=tiny-gdn"],
        },
        "factorial": [
            {
                "resident_count": 1,
                "cells": [
                    {
                        "arm_id": "kv=tiny-kv|gdn=tiny-gdn",
                        "memory_kernel_ledgers": [copy.deepcopy(ledger)],
                        "witness_kernel_ledgers": [copy.deepcopy(ledger)],
                    }
                ],
            }
        ],
    }


def _verify(candidate: dict[str, object], cache: Path, code: Path, runtime: Path, rank: int) -> None:
    verify_payload(
        candidate,
        cache_root=cache,
        code_root=code,
        runtime_root=runtime,
        geometry=TINY,
        expected_rank=rank,
        expected_gdn_bindings=fixture_bindings(candidate),
    )


def run() -> dict[str, object]:
    accepted: dict[str, str] = {}

    with tempfile.TemporaryDirectory(prefix="r40-v6-audit-") as temporary:
        cache, code, runtime, payload = build_tiny(Path(temporary))

        # CE1: selected_kwargs is serialized but never compared with either the
        # selected artifact metadata or the compact configuration table.
        candidate = copy.deepcopy(payload)
        config = candidate["tables"]["selected_compile_configurations"][0]
        candidate["tables"]["autotune_observations"][0] = {
            "mode": "triton-autotuner",
            "events": [
                {
                    "selected_kwargs": {
                        "BLOCK_M": 2**31,
                        "AUDIT_SENTINEL": "not-the-selected-config",
                    },
                    "num_warps": config["num_warps"],
                    "num_stages": config["num_stages"],
                    "num_ctas": config["num_ctas"],
                }
            ],
        }
        _verify(candidate, cache, code, runtime, 0)
        accepted["CE1_forged_autotune_selected_kwargs"] = "accepted"

        # CE2: the detached verifier accepts a self-consistent substitution to
        # an unrelated cache bundle.  It checks receipt/cache consistency, not
        # an independently observed selector event.
        decoy_dir = cache / "DECOY"
        decoy_dir.mkdir()
        decoy_metadata = {
            "hash": "b" * 64,
            "name": "unrelated_decoy_kernel",
            "num_warps": 4,
            "num_ctas": 1,
            "num_stages": 3,
        }
        (decoy_dir / "unrelated_decoy_kernel.json").write_text(
            json.dumps(decoy_metadata, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        (decoy_dir / "unrelated_decoy_kernel.cubin").write_bytes(b"decoy-cubin")
        (decoy_dir / "unrelated_decoy_kernel.ptx").write_bytes(b"decoy-ptx")
        decoy = base.TritonArtifact.from_metadata_file(
            cache, decoy_dir / "unrelated_decoy_kernel.json"
        ).as_dict()
        candidate = copy.deepcopy(payload)
        candidate["tables"]["compiled_artifacts"] = [decoy]
        candidate["tables"]["selected_compile_configurations"] = [
            {
                "name": decoy["kernel_name"],
                "hash": decoy["compiler_hash"],
                **decoy["compile_config"],
            }
        ]
        _verify(candidate, cache, code, runtime, 0)
        accepted["CE2_self_consistent_decoy_selector_substitution"] = "accepted"

        # CE3: unknown contradictory provenance fields are ignored because the
        # receipt and scope schemas are not exact at their outer levels.
        candidate = copy.deepcopy(payload)
        candidate["independent_device_launch_count"] = 0
        candidate["execution_backend_observed_elsewhere"] = "dense-fallback"
        candidate["scope"]["vllm_source_sha256"] = "0" * 64
        _verify(candidate, cache, code, runtime, 0)
        accepted["CE3_contradictory_device_and_vllm_provenance_fields"] = "accepted"

        # CE4: the compiled layer can be relabeled to another rank without any
        # process/GPU identity field.  The formal launcher inherits a separate
        # GPU-assignment check, so this is a boundary of the compiled receipt,
        # not evidence that simple rank omission passes the full launcher.
        candidate = copy.deepcopy(payload)
        candidate["rank"] = 1
        for cell in candidate["cells"]:
            cell["rank"] = 1
        argv = candidate["execution_binding"]["runner_argv"]
        argv[argv.index("--rank") + 1] = "1"
        argv[argv.index("--output") + 1] = "/frozen/forkaudit-shard-1.json"
        candidate["execution_binding"]["runner_argv_sha256"] = _runner_argv_sha256(argv)
        candidate["execution_binding"]["primary_shard_path"] = (
            "/frozen/forkaudit-shard-1.json"
        )
        _verify(candidate, cache, code, runtime, 1)
        verify_primary_shard(
            _tiny_shard(1), expected_rank=1, geometry=TINY, receipt=candidate
        )
        accepted["CE4_rank_relabel_without_compiled_layer_gpu_identity"] = "accepted"

        # CE5: exact compact-table cardinality/canonicality is not enforced.
        candidate = copy.deepcopy(payload)
        for table_name in (
            "compiled_artifacts",
            "selected_compile_configurations",
            "call_shapes",
            "autotune_observations",
        ):
            candidate["tables"][table_name].append(
                copy.deepcopy(candidate["tables"][table_name][0])
            )
        _verify(candidate, cache, code, runtime, 0)
        accepted["CE5_unreferenced_duplicate_table_rows"] = "accepted"

    return {
        "schema_version": "forkaudit-r40-compiled-v6-readonly-counterexamples-v1",
        "target": str(V6_ROOT),
        "target_modified": False,
        "gpu_used": False,
        "qs_or_ssh_used": False,
        "accepted_counterexamples": accepted,
        "all_counterexamples_accepted": all(
            result == "accepted" for result in accepted.values()
        ),
    }


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True, indent=2))
