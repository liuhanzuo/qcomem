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
from r39_primary_compact_dispatch import (
    AGGREGATE_SCHEMA_VERSION,
    PRIMARY_WORLD_SIZE,
    PrimaryDispatchError,
    expected_rank_counts,
    verify_payload,
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
    receipt_path: Path,
    cache_root: Path,
    snapshot_code: Path,
    snapshot_runtime: Path,
) -> dict[str, Any]:
    controls: dict[str, str] = {}

    candidate = copy.deepcopy(receipt)
    candidate["attention_calls"].pop()
    _expect_reject(
        "missing-primary-attention-call",
        lambda: verify_payload(candidate, cache_root=cache_root, code_root=snapshot_code, runtime_root=snapshot_runtime),
        controls,
    )
    candidate = copy.deepcopy(receipt)
    candidate["gdn_calls"].pop()
    _expect_reject(
        "missing-primary-gdn-call",
        lambda: verify_payload(candidate, cache_root=cache_root, code_root=snapshot_code, runtime_root=snapshot_runtime),
        controls,
    )
    candidate = copy.deepcopy(receipt)
    candidate["cells"][0]["arm_id"] = "tampered-arm"
    _expect_reject(
        "primary-cell-context-relabel",
        lambda: verify_payload(candidate, cache_root=cache_root, code_root=snapshot_code, runtime_root=snapshot_runtime),
        controls,
    )
    candidate = copy.deepcopy(receipt)
    candidate["tables"]["selected_compile_configurations"][0]["num_warps"] += 1
    _expect_reject(
        "selected-compile-config-tamper",
        lambda: verify_payload(candidate, cache_root=cache_root, code_root=snapshot_code, runtime_root=snapshot_runtime),
        controls,
    )
    candidate = copy.deepcopy(receipt)
    candidate["tables"]["compiled_artifacts"][0]["artifact_id"] = "0" * 64
    _expect_reject(
        "selected-artifact-id-tamper",
        lambda: verify_payload(candidate, cache_root=cache_root, code_root=snapshot_code, runtime_root=snapshot_runtime),
        controls,
    )

    with tempfile.TemporaryDirectory(prefix="r39-primary-controls-") as temporary:
        root = Path(temporary)
        cache_copy = root / "cache"
        source_copy = root / "source"
        shutil.copytree(cache_root, cache_copy)
        shutil.copytree(snapshot_code.parent, source_copy)
        ptx = next(cache_copy.rglob("*.ptx"))
        ptx.unlink()
        _expect_reject(
            "missing-selected-ptx",
            lambda: verify_payload(receipt, cache_root=cache_copy, code_root=source_copy / "code", runtime_root=source_copy / "runtime"),
            controls,
        )

        shutil.rmtree(cache_copy)
        shutil.copytree(cache_root, cache_copy)
        cubin = next(cache_copy.rglob("*.cubin"))
        cubin.write_bytes(cubin.read_bytes() + b"substitution")
        _expect_reject(
            "compiled-artifact-byte-substitution",
            lambda: verify_payload(receipt, cache_root=cache_copy, code_root=source_copy / "code", runtime_root=source_copy / "runtime"),
            controls,
        )

        shutil.rmtree(source_copy)
        shutil.copytree(snapshot_code.parent, source_copy)
        binding = next(iter(receipt["gdn_source_bindings"].values()))
        source = source_copy / binding["root_kind"] / binding["relative_source_path"]
        source.write_bytes(source.read_bytes() + b"\n# substitution\n")
        _expect_reject(
            "gdn-bound-source-substitution",
            lambda: verify_payload(receipt, cache_root=cache_root, code_root=source_copy / "code", runtime_root=source_copy / "runtime"),
            controls,
        )

    return {
        "schema_version": "forkaudit-r39-primary-bound-controls-v1",
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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit("final output must be absent")
    output.mkdir(parents=True)

    primary_summary_path = args.primary_run_root / "forkaudit-summary.json"
    primary_summary = verify_primary_aggregate(_load(primary_summary_path))
    counts = expected_rank_counts()
    rank_replays: list[dict[str, Any]] = []
    artifact_ids: set[str] = set()
    config_objects: set[str] = set()
    for rank in range(PRIMARY_WORLD_SIZE):
        source_rank = args.capture_root / f"rank-{rank}"
        receipt_path = source_rank / "raw" / "primary-compiled-dispatch-receipt.json"
        shard_path = args.primary_run_root / "raw" / "shards" / f"forkaudit-shard-{rank}.json"
        receipt = _load(receipt_path)
        binding = receipt.get("execution_binding", {})
        if binding.get("primary_shard_sha256") != _sha256_file(shard_path):
            raise PrimaryDispatchError("receipt binds a different primary shard")
        shard_replay = verify_primary_shard(_load(shard_path), expected_rank=rank)
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
        )
        controls = _bound_controls(
            receipt=receipt,
            receipt_path=receipt_path,
            cache_root=source_rank / "runtime-cache" / "triton",
            snapshot_code=snapshot / "code",
            snapshot_runtime=snapshot / "runtime",
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

    aggregate = {
        "schema_version": AGGREGATE_SCHEMA_VERSION,
        "status": "pass",
        "formal_evidence_eligible": True,
        "target_5_status_at_declared_scope": "pass",
        "primary_aggregate": primary_summary,
        "primary_aggregate_sha256": _sha256_file(primary_summary_path),
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
                "every attention call in both formal-memory and ownership-witness rebuilds is "
                "bound to its selected fully hashed Triton artifact/configuration. Every native "
                "Qwen3.5-MoE GDN call is closed over the selected eager torch chunk rule and the "
                "qcomem functional conv/recurrent cache-rebind source bindings."
            ),
            "not_established": [
                "a compiled GDN binary or identity of the eager path's underlying ATen/CUDA operators",
                "runtime attestation or malicious-producer resistance",
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
