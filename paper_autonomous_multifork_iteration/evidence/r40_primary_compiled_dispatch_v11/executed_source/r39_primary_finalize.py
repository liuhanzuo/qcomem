#!/usr/bin/env python3
"""Detach, adversarially replay, and aggregate all eight primary receipts."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping

import r39_compiled_dispatch_receipts as base
from r40_runtime_smoke import load_and_verify_runtime_preflight
from r39_primary_compact_dispatch import (
    AGGREGATE_SCHEMA_VERSION,
    GDN_COLUMNS,
    PRIMARY_WORLD_SIZE,
    PrimaryDispatchError,
    expected_rank_counts,
    verify_payload,
    verify_gpu_assignment_receipt,
    verify_primary_aggregate,
    verify_primary_shard,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    if path.exists():
        raise PrimaryDispatchError(f"refusing to overwrite finalizer output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n",
        encoding="utf-8",
    )


def _expect_reject(name: str, function: Callable[[], Any], controls: dict[str, str]) -> None:
    try:
        function()
    except (PrimaryDispatchError, base.DispatchReceiptError):
        controls[name] = "rejected"
    else:
        raise PrimaryDispatchError(f"negative control unexpectedly passed: {name}")


def _bound_controls(
    *,
    receipt: Mapping[str, Any],
    primary_shard: Mapping[str, Any],
    receipt_path: Path,
    cache_root: Path,
    snapshot_code: Path,
    snapshot_runtime: Path,
    expected_rank: int,
    expected_source_bindings: Mapping[str, Mapping[str, str]],
    gpu_assignment_receipt: Mapping[str, Any],
    gpu_assignment_raw_sha256: str,
    launcher_identity: Mapping[str, Any],
    launcher_identity_raw_sha256: str,
    runtime_preflight_sha256: str,
) -> dict[str, Any]:
    controls: dict[str, str] = {}

    def replay(candidate: Mapping[str, Any]) -> Any:
        return verify_payload(
            candidate,
            cache_root=cache_root,
            code_root=snapshot_code,
            runtime_root=snapshot_runtime,
            expected_rank=expected_rank,
            expected_source_bindings=expected_source_bindings,
            expected_gpu_assignment_receipt=gpu_assignment_receipt,
            expected_gpu_assignment_raw_sha256=gpu_assignment_raw_sha256,
            expected_launcher_identity=launcher_identity,
            expected_launcher_identity_raw_sha256=launcher_identity_raw_sha256,
            expected_runtime_preflight_sha256=runtime_preflight_sha256,
        )

    def replay_and_bind(candidate: Mapping[str, Any]) -> Any:
        replay(candidate)
        return verify_primary_shard(
            primary_shard,
            expected_rank=expected_rank,
            receipt=candidate,
        )

    def replay_at(
        candidate: Mapping[str, Any], *, cache: Path, code: Path, runtime: Path
    ) -> Any:
        return verify_payload(
            candidate,
            cache_root=cache,
            code_root=code,
            runtime_root=runtime,
            expected_rank=expected_rank,
            expected_source_bindings=expected_source_bindings,
            expected_gpu_assignment_receipt=gpu_assignment_receipt,
            expected_gpu_assignment_raw_sha256=gpu_assignment_raw_sha256,
            expected_launcher_identity=launcher_identity,
            expected_launcher_identity_raw_sha256=launcher_identity_raw_sha256,
            expected_runtime_preflight_sha256=runtime_preflight_sha256,
        )

    candidate = copy.deepcopy(receipt)
    candidate["attention_calls"].pop()
    _expect_reject(
        "missing-primary-attention-call",
        lambda: replay(candidate),
        controls,
    )
    candidate = copy.deepcopy(receipt)
    candidate["gdn_calls"].pop()
    _expect_reject(
        "missing-primary-gdn-call",
        lambda: replay(candidate),
        controls,
    )
    candidate = copy.deepcopy(receipt)
    candidate["cells"][0]["arm_id"] = "tampered-arm"
    _expect_reject(
        "primary-cell-context-relabel",
        lambda: replay(candidate),
        controls,
    )
    candidate = copy.deepcopy(receipt)
    candidate["tables"]["selected_compile_configurations"][0]["num_warps"] += 1
    _expect_reject(
        "selected-compile-config-tamper",
        lambda: replay(candidate),
        controls,
    )
    candidate = copy.deepcopy(receipt)
    candidate["tables"]["compiled_artifacts"][0]["artifact_id"] = "0" * 64
    _expect_reject(
        "selected-artifact-id-tamper",
        lambda: replay(candidate),
        controls,
    )
    candidate = copy.deepcopy(receipt)
    candidate["scope"]["gdn"] = "compiled-superkernel-attested"
    _expect_reject("gdn-scope-overclaim", lambda: replay(candidate), controls)
    candidate = copy.deepcopy(receipt)
    candidate["attention_call_columns"] = ["lies"]
    _expect_reject("attention-column-relabel", lambda: replay(candidate), controls)
    candidate = copy.deepcopy(receipt)
    candidate["gdn_call_columns"] = ["lies"]
    _expect_reject("gdn-column-relabel", lambda: replay(candidate), controls)
    candidate = copy.deepcopy(receipt)
    candidate["gdn_calls"][0][GDN_COLUMNS.index("recurrent_rule_calls")] = 1
    _expect_reject(
        "gdn-multi-route-cross-contamination",
        lambda: replay(candidate),
        controls,
    )
    candidate = copy.deepcopy(receipt)
    single_token_row = next(
        row
        for row in candidate["gdn_calls"]
        if row[GDN_COLUMNS.index("cache_has_previous_state")] is True
        and row[GDN_COLUMNS.index("sequence_length")] == 1
    )
    single_token_row[GDN_COLUMNS.index("chunk_rule_calls")] = 1
    _expect_reject(
        "gdn-single-route-cross-contamination",
        lambda: replay(candidate),
        controls,
    )
    candidate = copy.deepcopy(receipt)
    shape_record = candidate["tables"]["call_shapes"][0]
    shape_record["shape"]["max_seqlen_k"] = (
        4128 if shape_record["shape"]["max_seqlen_k"] == 4127 else 4127
    )
    shape_record["shape_sha256"] = base._sha256_bytes(
        base._canonical_bytes(shape_record["shape"])
    )
    _expect_reject(
        "attention-shape-substitution",
        lambda: replay_and_bind(candidate),
        controls,
    )
    candidate = copy.deepcopy(receipt)
    candidate["rank"] = (expected_rank + 1) % 8
    _expect_reject("cross-rank-receipt-substitution", lambda: replay(candidate), controls)
    candidate = copy.deepcopy(receipt)
    candidate["execution_binding"]["runner_sha256"] = "0" * 64
    _expect_reject("runner-identity-substitution", lambda: replay(candidate), controls)

    # The five counterexamples accepted by v6 are permanent v7 regressions.
    candidate = copy.deepcopy(receipt)
    autotune_index = candidate["attention_calls"][0][6]
    config_index = candidate["attention_calls"][0][5]
    config = candidate["tables"]["selected_compile_configurations"][config_index]
    forged_kwargs = {"AUDIT_SENTINEL": "forged", "BLOCK_M": 2**31}
    candidate["tables"]["autotune_observations"][autotune_index] = {
        "mode": "triton-autotuner",
        "events": [
            {
                "selected_kwargs": forged_kwargs,
                "selected_kwargs_sha256": base._sha256_bytes(
                    base._canonical_bytes(forged_kwargs)
                ),
                "num_warps": config["num_warps"],
                "num_stages": config["num_stages"],
                "num_ctas": config["num_ctas"],
            }
        ],
    }
    _expect_reject(
        "v6-ce1-forged-autotune-selected-kwargs",
        lambda: replay(candidate),
        controls,
    )

    candidate = copy.deepcopy(receipt)
    candidate["independent_device_launch_count"] = 0
    candidate["scope"]["vllm_source_sha256"] = "0" * 64
    _expect_reject(
        "v6-ce3-contradictory-unknown-provenance",
        lambda: replay(candidate),
        controls,
    )

    candidate = copy.deepcopy(receipt)
    candidate["rank"] = (expected_rank + 1) % 8
    for cell in candidate["cells"]:
        cell["rank"] = candidate["rank"]
    candidate["rank_identity"]["rank"] = candidate["rank"]
    _expect_reject(
        "v6-ce4-rank-relabel-without-external-assignment",
        lambda: replay(candidate),
        controls,
    )

    candidate = copy.deepcopy(receipt)
    for table_name in (
        "compiled_artifacts",
        "selected_compile_configurations",
        "call_shapes",
        "autotune_observations",
    ):
        candidate["tables"][table_name].append(
            copy.deepcopy(candidate["tables"][table_name][0])
        )
    _expect_reject(
        "v6-ce5-unreferenced-duplicate-table-rows",
        lambda: replay(candidate),
        controls,
    )

    source_key = "transformers_accelerate_wrapper"
    other_key = "transformers_qwen35_moe_torch_chunk_gated_delta_rule"
    candidate = copy.deepcopy(receipt)
    candidate["dispatch_source_bindings"][source_key], candidate["dispatch_source_bindings"][other_key] = (
        copy.deepcopy(candidate["dispatch_source_bindings"][other_key]),
        copy.deepcopy(candidate["dispatch_source_bindings"][source_key]),
    )
    _expect_reject("gdn-cross-key-whole-binding-substitution", lambda: replay(candidate), controls)
    candidate = copy.deepcopy(receipt)
    candidate["dispatch_source_bindings"][source_key] = copy.deepcopy(
        candidate["dispatch_source_bindings"][other_key]
    )
    _expect_reject("gdn-duplicate-callable-binding", lambda: replay(candidate), controls)
    for name, field, value in (
        ("gdn-binding-root-tamper", "root_kind", "code"),
        ("gdn-binding-path-tamper", "relative_source_path", "qcomem_qwen35_native_cache.py"),
        ("gdn-binding-module-tamper", "module", "qcomem_qwen35_native_cache"),
        ("gdn-binding-qualname-tamper", "qualname", "torch_chunk_gated_delta_rule"),
        ("gdn-binding-source-hash-tamper", "source_sha256", "0" * 64),
    ):
        candidate = copy.deepcopy(receipt)
        candidate["dispatch_source_bindings"][source_key][field] = value
        _expect_reject(name, lambda candidate=candidate: replay(candidate), controls)

    with tempfile.TemporaryDirectory(prefix="r39-primary-controls-") as temporary:
        root = Path(temporary)
        cache_copy = root / "cache"
        source_copy = root / "source"
        shutil.copytree(cache_root, cache_copy)
        shutil.copytree(snapshot_code.parent, source_copy)

        decoy_dir = cache_copy / "R40-DECOY"
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
            cache_copy, decoy_dir / "unrelated_decoy_kernel.json"
        ).as_dict()
        candidate = copy.deepcopy(receipt)
        candidate["tables"]["compiled_artifacts"] = [decoy]
        candidate["tables"]["selected_compile_configurations"] = [
            {
                "name": decoy["kernel_name"],
                "hash": decoy["compiler_hash"],
                **decoy["compile_config"],
            }
        ]
        for row in candidate["attention_calls"]:
            shape = candidate["tables"]["call_shapes"][row[3]]["shape"]
            autotune = candidate["tables"]["autotune_observations"][row[6]]
            row[4] = 0
            row[5] = 0
            row[12] = base._sha256_bytes(
                base._canonical_bytes(
                    {
                        "call_id": row[2],
                        "call_shape": shape,
                        "artifact_id": decoy["artifact_id"],
                        "selected_compile_config": candidate["tables"][
                            "selected_compile_configurations"
                        ][0],
                        "autotune": autotune,
                        "launch_context": {
                            "cuda_visible_devices": row[7],
                            "torch_device_index": row[8],
                            "torch_device_type": "cuda",
                            "torch_stream_id": row[9],
                        },
                        "post_launcher_returned": row[10],
                        "post_return_context_matches": row[11],
                    }
                )
            )
        _expect_reject(
            "v6-ce2-self-consistent-decoy-selector",
            lambda: verify_payload(
                candidate,
                cache_root=cache_copy,
                code_root=source_copy / "code",
                runtime_root=source_copy / "runtime",
                expected_rank=expected_rank,
                expected_source_bindings=expected_source_bindings,
                expected_gpu_assignment_receipt=gpu_assignment_receipt,
                expected_gpu_assignment_raw_sha256=gpu_assignment_raw_sha256,
                expected_launcher_identity=launcher_identity,
                expected_launcher_identity_raw_sha256=launcher_identity_raw_sha256,
                expected_runtime_preflight_sha256=runtime_preflight_sha256,
            ),
            controls,
        )

        shutil.rmtree(cache_copy)
        shutil.copytree(cache_root, cache_copy)
        selected_files = receipt["tables"]["compiled_artifacts"][0]["files"]
        selected_ptx = [row["relative_path"] for row in selected_files if Path(row["relative_path"]).suffix == ".ptx"]
        selected_cubin = [row["relative_path"] for row in selected_files if Path(row["relative_path"]).suffix == ".cubin"]
        if len(selected_ptx) != 1 or len(selected_cubin) != 1:
            raise PrimaryDispatchError("selected artifact PTX/cubin set drift")
        ptx = cache_copy / selected_ptx[0]
        ptx.unlink()
        _expect_reject(
            "missing-selected-ptx",
            lambda: replay_at(
                receipt,
                cache=cache_copy,
                code=source_copy / "code",
                runtime=source_copy / "runtime",
            ),
            controls,
        )

        shutil.rmtree(cache_copy)
        shutil.copytree(cache_root, cache_copy)
        cubin = cache_copy / selected_cubin[0]
        cubin.write_bytes(cubin.read_bytes() + b"substitution")
        _expect_reject(
            "compiled-artifact-byte-substitution",
            lambda: replay_at(
                receipt,
                cache=cache_copy,
                code=source_copy / "code",
                runtime=source_copy / "runtime",
            ),
            controls,
        )

        shutil.rmtree(source_copy)
        shutil.copytree(snapshot_code.parent, source_copy)
        binding = next(iter(receipt["dispatch_source_bindings"].values()))
        source = source_copy / binding["root_kind"] / binding["relative_source_path"]
        source.write_bytes(source.read_bytes() + b"\n# substitution\n")
        _expect_reject(
            "gdn-bound-source-substitution",
            lambda: replay_at(
                receipt,
                cache=cache_root,
                code=source_copy / "code",
                runtime=source_copy / "runtime",
            ),
            controls,
        )

    return {
        "schema_version": "forkaudit-r40-primary-bound-controls-v7",
        "control_basis": "actual-primary-receipt-artifacts-and-snapshotted-sources",
        "receipt_sha256": _sha256_file(receipt_path),
        "negative_controls": controls,
        "all_rejected": all(value == "rejected" for value in controls.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-run-root", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--runtime-preflight-manifest", type=Path, required=True)
    parser.add_argument("--expected-runtime-preflight-sha256", required=True)
    parser.add_argument("--launch-identity-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit("final output must be absent")
    output.mkdir(parents=True)

    runtime_preflight_path = args.runtime_preflight_manifest.resolve()
    _require_runtime_sha = _sha256_file(runtime_preflight_path)
    if _require_runtime_sha != args.expected_runtime_preflight_sha256:
        raise PrimaryDispatchError("runtime preflight raw SHA drift")
    runtime_preflight = load_and_verify_runtime_preflight(runtime_preflight_path)
    expected_source_bindings = runtime_preflight["dispatch_source_bindings"]

    gpu_assignment_path = (
        args.primary_run_root / "receipts" / "gpu-assignment-receipt.json"
    )
    gpu_assignment_raw_sha256 = _sha256_file(gpu_assignment_path)
    gpu_assignment_receipt = verify_gpu_assignment_receipt(_load(gpu_assignment_path))

    primary_summary_path = args.primary_run_root / "forkaudit-summary.json"
    primary_summary_value = _load(primary_summary_path)
    primary_summary = verify_primary_aggregate(primary_summary_value)
    raw_shard_artifacts = primary_summary_value.get("raw_shard_artifacts")
    if not isinstance(raw_shard_artifacts, list) or len(raw_shard_artifacts) != PRIMARY_WORLD_SIZE:
        raise PrimaryDispatchError("primary aggregate does not enumerate exactly eight raw shards")
    for rank, row in enumerate(raw_shard_artifacts):
        expected_relative = f"shards/forkaudit-shard-{rank}.json"
        shard_path = args.primary_run_root / "raw" / expected_relative
        if not (
            isinstance(row, dict)
            and set(row) == {"bytes", "relative_path", "sha256"}
            and row.get("relative_path") == expected_relative
            and row.get("bytes") == shard_path.stat().st_size
            and row.get("sha256") == _sha256_file(shard_path)
        ):
            raise PrimaryDispatchError(f"primary aggregate raw shard {rank} binding drift")
    counts = expected_rank_counts()
    rank_replays: list[dict[str, Any]] = []
    artifact_ids: set[str] = set()
    config_objects: set[str] = set()
    rank_process_ids: set[int] = set()
    rank_gpu_uuids: set[str] = set()
    rank_identity_tuples: set[tuple[int, int, str]] = set()
    for rank in range(PRIMARY_WORLD_SIZE):
        source_rank = args.capture_root / f"rank-{rank}"
        receipt_path = source_rank / "raw" / "primary-compiled-dispatch-receipt.json"
        shard_path = args.primary_run_root / "raw" / "shards" / f"forkaudit-shard-{rank}.json"
        receipt = _load(receipt_path)
        launcher_identity_path = args.launch_identity_root / f"rank-{rank}.json"
        launcher_identity_raw_sha256 = _sha256_file(launcher_identity_path)
        launcher_identity = _load(launcher_identity_path)
        embedded_identity = receipt.get("rank_identity", {})
        if embedded_identity.get("launcher_identity_path") != str(
            launcher_identity_path.resolve()
        ):
            raise PrimaryDispatchError("receipt binds a different proxy launcher identity path")
        binding = receipt.get("execution_binding", {})
        if binding.get("primary_shard_path") != str(shard_path.resolve()):
            raise PrimaryDispatchError("receipt binds a different primary shard path")
        if binding.get("primary_shard_sha256") != _sha256_file(shard_path):
            raise PrimaryDispatchError("receipt binds a different primary shard")
        rank_output = output / f"rank-{rank}"
        snapshot = rank_output / "source-snapshot"
        snapshot_manifest_path = rank_output / "source-snapshot-manifest.json"
        snapshot_manifest = base.snapshot_bound_sources(
            payload=receipt,
            code_root=args.code_root,
            runtime_root=args.runtime_root,
            target=snapshot,
            output=snapshot_manifest_path,
        )
        replay = verify_payload(
            receipt,
            cache_root=source_rank / "runtime-cache" / "triton",
            code_root=snapshot / "code",
            runtime_root=snapshot / "runtime",
            expected_rank=rank,
            expected_source_bindings=expected_source_bindings,
            expected_gpu_assignment_receipt=gpu_assignment_receipt,
            expected_gpu_assignment_raw_sha256=gpu_assignment_raw_sha256,
            expected_launcher_identity=launcher_identity,
            expected_launcher_identity_raw_sha256=launcher_identity_raw_sha256,
            expected_runtime_preflight_sha256=args.expected_runtime_preflight_sha256,
        )
        primary_shard = _load(shard_path)
        shard_replay = verify_primary_shard(
            primary_shard, expected_rank=rank, receipt=receipt
        )
        controls = _bound_controls(
            receipt=receipt,
            primary_shard=primary_shard,
            receipt_path=receipt_path,
            cache_root=source_rank / "runtime-cache" / "triton",
            snapshot_code=snapshot / "code",
            snapshot_runtime=snapshot / "runtime",
            expected_rank=rank,
            expected_source_bindings=expected_source_bindings,
            gpu_assignment_receipt=gpu_assignment_receipt,
            gpu_assignment_raw_sha256=gpu_assignment_raw_sha256,
            launcher_identity=launcher_identity,
            launcher_identity_raw_sha256=launcher_identity_raw_sha256,
            runtime_preflight_sha256=args.expected_runtime_preflight_sha256,
        )
        _write(rank_output / "replay.json", replay)
        _write(rank_output / "primary-shard-replay.json", shard_replay)
        _write(rank_output / "negative-controls.json", controls)
        shutil.copy2(receipt_path, rank_output / "primary-compiled-dispatch-receipt.json")
        shutil.copytree(source_rank / "runtime-cache", rank_output / "runtime-cache")
        artifact_ids.update(
            row["artifact_id"]
            for row in receipt["tables"]["compiled_artifacts"]
        )
        config_objects.update(
            json.dumps(row, sort_keys=True, separators=(",", ":"))
            for row in receipt["tables"]["selected_compile_configurations"]
        )
        identity = receipt["rank_identity"]
        process_id = identity["process_id"]
        gpu_uuid = identity["assigned_gpu_uuid"]
        rank_process_ids.add(process_id)
        rank_gpu_uuids.add(gpu_uuid)
        rank_identity_tuples.add((rank, process_id, gpu_uuid))
        rank_replays.append(
            {
                "rank": rank,
                "receipt_sha256": _sha256_file(receipt_path),
                "primary_shard_sha256": _sha256_file(shard_path),
                "replay": replay,
                "primary_shard_replay": shard_replay,
                "source_snapshot_manifest": snapshot_manifest,
                "negative_controls": controls,
            }
        )

    if len(rank_process_ids) != PRIMARY_WORLD_SIZE:
        raise PrimaryDispatchError("eight rank receipts do not bind eight unique processes")
    if len(rank_gpu_uuids) != PRIMARY_WORLD_SIZE:
        raise PrimaryDispatchError("eight rank receipts do not bind eight unique assigned GPUs")
    if len(rank_identity_tuples) != PRIMARY_WORLD_SIZE:
        raise PrimaryDispatchError("rank/process/GPU identity closure failed")

    aggregate = {
        "schema_version": AGGREGATE_SCHEMA_VERSION,
        "status": "pass",
        "formal_evidence_eligible": True,
        "target_5_status_at_declared_scope": "pass",
        "primary_aggregate": primary_summary,
        "primary_aggregate_sha256": _sha256_file(primary_summary_path),
        "runtime_preflight_manifest_sha256": args.expected_runtime_preflight_sha256,
        "gpu_assignment_receipt_raw_sha256": gpu_assignment_raw_sha256,
        "rank_identity_closure": {
            "exact_rank_count": PRIMARY_WORLD_SIZE,
            "unique_process_count": len(rank_process_ids),
            "unique_gpu_uuid_count": len(rank_gpu_uuids),
            "all_rank_process_gpu_tuples_unique": True,
        },
        "rank_replays": rank_replays,
        "totals": {
            "rank_count": PRIMARY_WORLD_SIZE,
            "primary_configuration_count": 96,
            "primary_execution_cell_count": counts["cell_count"] * PRIMARY_WORLD_SIZE,
            "attention_call_count": counts["attention_call_count"] * PRIMARY_WORLD_SIZE,
            "gdn_document_prefill_call_count": counts["gdn_document_prefill_call_count"] * PRIMARY_WORLD_SIZE,
            "gdn_request_call_count": counts["gdn_request_call_count"] * PRIMARY_WORLD_SIZE,
            "gdn_call_count": counts["gdn_call_count"] * PRIMARY_WORLD_SIZE,
            "distinct_compiled_artifact_ids": sorted(artifact_ids),
            "distinct_selected_compile_configurations": [json.loads(item) for item in sorted(config_objects)],
        },
        "claim_boundary": {
            "established": (
                "Across the fresh rerun of the frozen 96-configuration RR2 primary factorial, "
                "every attention call in both formal-memory and ownership-witness rebuilds has "
                "an exact call ID and is bound before invocation to its selected fully hashed "
                "Triton launcher artifact/configuration, then sealed only after that original "
                "launcher returns successfully on the same assigned device and stream. The "
                "vLLM/Triton/Transformers/qcomem callables are source-bound. Every native "
                "Qwen3.5-MoE GDN call is closed over its mutually exclusive frozen eager route: "
                "multi-token chunk rule plus functional conv rebind, or cached single-token "
                "recurrent rule plus in-place causal-conv update; both routes bind the qcomem "
                "functional recurrent cache rebind."
            ),
            "not_established": [
                "a compiled GDN binary or identity of the eager path's underlying ATen/CUDA operators",
                "driver/device-level binary execution attestation or malicious-runtime resistance",
                "cross-model, cross-runtime, or cross-hardware generality",
            ],
        },
    }
    _write(output / "formal-aggregate.json", aggregate)
    ledger_rows: list[str] = []
    for path in sorted((item for item in output.rglob("*") if item.is_file()), key=lambda item: item.as_posix()):
        relative = path.relative_to(output).as_posix()
        ledger_rows.append(f"{_sha256_file(path)}  {relative}\n")
    (output / "terminal-files.sha256").write_text("".join(ledger_rows), encoding="utf-8")
    (output / "COMPLETE").touch()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
