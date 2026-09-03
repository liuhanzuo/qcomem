from __future__ import annotations

"""Auditable post-execution repair for the R28 RR2 run-ID projection bug.

The preregistered builder read ``run_id`` from the detached RR2 manifest even
though that manifest schema does not contain such a field.  The same manifest
does bind eight immutable shards, and every shard embeds the identical,
derivation-verifiable run-ID receipt.  This wrapper leaves the frozen builder,
replay, preregistration, candidate ranks, and FP32 sidecars unchanged.  It
validates that pre-existing shard/receipt authority, changes only the generated
in-memory RR2 row ``run_id`` value from null to the verified value, and then
delegates all remaining validation and summary construction to the frozen
builder.
"""

import argparse
import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import build_qcomem_qwen35_forkaudit_detector_matrix_v2 as frozen_builder


AMENDMENT_SCHEMA = "forkaudit-r28-postexecution-rr2-run-binding-correction-v1"
CORRECTION_RECEIPT_SCHEMA = (
    "forkaudit-r28-postexecution-rr2-run-binding-correction-receipt-v1"
)
RR2_RECEIPT_SCHEMA = "qcomem-forkaudit-run-id-receipt-v1"
RR2_MANIFEST_SCHEMA = "qcomem-forkaudit-detached-receipts-v1"
RUN_ID_DERIVATION = "sha256(domain || static_sha256 || protocol_sha256 || nonce)[:16]"
HEX_64 = re.compile(r"^[0-9a-f]{64}$")
HEX_32 = re.compile(r"^[0-9a-f]{32}$")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise frozen_builder.BuildError(message)


def _load_canonical_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    _require(path.is_file() and not path.is_symlink(), f"{label} file integrity")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise frozen_builder.BuildError(f"{label} JSON") from error
    _require(isinstance(value, dict), f"{label} object")
    _require(
        raw == frozen_builder.canonical_bytes(value) + b"\n",
        f"{label} canonical JSON",
    )
    return value, raw


def _validated_relative_path(root: Path, relative: str, label: str) -> Path:
    _require(isinstance(relative, str) and bool(relative), f"{label} relative path")
    relative_path = Path(relative)
    _require(not relative_path.is_absolute(), f"{label} absolute path")
    _require(".." not in relative_path.parts, f"{label} parent traversal")
    candidate = root / relative_path
    _require(candidate.is_file() and not candidate.is_symlink(), f"{label} file integrity")
    resolved_root = root.resolve()
    resolved = candidate.resolve()
    _require(resolved.is_relative_to(resolved_root), f"{label} escaped root")
    return candidate


def validate_raw_artifact_ledger(
    *, ledger_path: Path, rank_root: Path, expected_sha256: str
) -> dict[str, Any]:
    """Verify the pre-correction ledger against every frozen candidate artifact."""

    _require(HEX_64.fullmatch(expected_sha256) is not None, "raw ledger expected SHA")
    _require(ledger_path.is_file() and not ledger_path.is_symlink(), "raw ledger integrity")
    raw = ledger_path.read_bytes()
    _require(
        frozen_builder.sha256_bytes(raw) == expected_sha256,
        "raw ledger SHA binding",
    )
    run_root = rank_root.resolve().parent
    seen: set[str] = set()
    verified: list[dict[str, Any]] = []
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise frozen_builder.BuildError("raw ledger UTF-8") from error
    _require(bool(lines), "raw ledger nonempty")
    for index, line in enumerate(lines, start=1):
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        _require(match is not None, f"raw ledger line {index}")
        expected, relative = match.groups()
        _require(relative not in seen, f"raw ledger duplicate {relative}")
        seen.add(relative)
        path = _validated_relative_path(run_root, relative, f"raw ledger {relative}")
        payload = path.read_bytes()
        _require(
            frozen_builder.sha256_bytes(payload) == expected,
            f"raw ledger artifact SHA {relative}",
        )
        verified.append(
            {"relative_path": relative, "bytes": len(payload), "sha256": expected}
        )
    expected_ranks = {f"raw/detector-matrix-v2-rank-{rank}.json" for rank in range(8)}
    _require(expected_ranks <= seen, "raw ledger eight rank coverage")
    return {
        "raw_sha256": expected_sha256,
        "entry_count": len(verified),
        "rank_entries_verified": 8,
        "all_entries_verified": True,
    }


def validate_rr2_run_binding(
    *,
    original_receipt_manifest: Path,
    original_rr2_root: Path,
    expected_manifest_sha256: str,
    expected_run_id: str,
    expected_run_id_receipt_raw_sha256: str,
) -> dict[str, Any]:
    """Recover the RR2 run ID only from manifest-bound, receipt-bearing shards."""

    _require(HEX_64.fullmatch(expected_manifest_sha256) is not None, "RR2 manifest SHA")
    _require(HEX_32.fullmatch(expected_run_id) is not None, "RR2 expected run ID")
    _require(
        HEX_64.fullmatch(expected_run_id_receipt_raw_sha256) is not None,
        "RR2 run-ID receipt raw SHA",
    )
    manifest, manifest_raw = _load_canonical_json(
        original_receipt_manifest, "RR2 detached manifest"
    )
    _require(
        frozen_builder.sha256_bytes(manifest_raw) == expected_manifest_sha256,
        "RR2 detached manifest SHA binding",
    )
    _require(manifest.get("schema_version") == RR2_MANIFEST_SCHEMA, "RR2 manifest schema")
    _require("run_id" not in manifest, "RR2 legacy manifest must omit top-level run_id")

    receipt_path = original_rr2_root / "receipts" / "run-id-receipt.json"
    receipt, receipt_raw = _load_canonical_json(receipt_path, "RR2 run-ID receipt")
    _require(
        frozen_builder.sha256_bytes(receipt_raw)
        == expected_run_id_receipt_raw_sha256,
        "RR2 run-ID receipt raw SHA binding",
    )
    _require(receipt.get("schema_version") == RR2_RECEIPT_SCHEMA, "RR2 receipt schema")
    _require(receipt.get("derivation") == RUN_ID_DERIVATION, "RR2 receipt derivation label")
    _require(receipt.get("run_id_bits") == 128, "RR2 receipt bit width")
    _require(
        receipt.get("generated_once_after_static_before_candidate_outputs") is True,
        "RR2 receipt prospective timing",
    )
    for field in (
        "domain_hex",
        "static_artifact_sha256",
        "protocol_manifest_sha256",
        "nonce_hex",
    ):
        value = receipt.get(field)
        _require(
            isinstance(value, str)
            and len(value) % 2 == 0
            and re.fullmatch(r"[0-9a-f]+", value) is not None,
            f"RR2 receipt {field}",
        )
    derived = hashlib.sha256(
        bytes.fromhex(receipt["domain_hex"])
        + bytes.fromhex(receipt["static_artifact_sha256"])
        + bytes.fromhex(receipt["protocol_manifest_sha256"])
        + bytes.fromhex(receipt["nonce_hex"])
    ).digest()[:16].hex()
    _require(derived == receipt.get("run_id") == expected_run_id, "RR2 run-ID derivation")
    _require(
        receipt.get("static_artifact_sha256") == manifest.get("static_artifact_sha256"),
        "RR2 receipt/manifest static artifact binding",
    )
    receipt_canonical_sha = frozen_builder.sha256_bytes(
        frozen_builder.canonical_bytes(receipt)
    )

    shard_refs = manifest.get("shards")
    _require(isinstance(shard_refs, list) and len(shard_refs) == 8, "RR2 eight shard refs")
    ranks: set[int] = set()
    verified_shards: list[dict[str, Any]] = []
    for shard_ref in shard_refs:
        _require(isinstance(shard_ref, dict), "RR2 shard ref object")
        relative = shard_ref.get("relative_path")
        path = _validated_relative_path(
            original_rr2_root / "raw", relative, f"RR2 shard {relative}"
        )
        payload = path.read_bytes()
        _require(len(payload) == shard_ref.get("bytes"), f"RR2 shard bytes {relative}")
        _require(
            frozen_builder.sha256_bytes(payload) == shard_ref.get("sha256"),
            f"RR2 shard SHA {relative}",
        )
        try:
            shard = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise frozen_builder.BuildError(f"RR2 shard JSON {relative}") from error
        _require(isinstance(shard, dict), f"RR2 shard object {relative}")
        rank = shard.get("rank")
        _require(type(rank) is int and 0 <= rank < 8, f"RR2 shard rank {relative}")
        _require(rank not in ranks, f"RR2 duplicate shard rank {rank}")
        ranks.add(rank)
        _require(shard.get("run_id") == expected_run_id, f"RR2 shard run ID rank {rank}")
        _require(shard.get("run_id_receipt") == receipt, f"RR2 shard receipt rank {rank}")
        _require(
            shard.get("run_id_receipt_sha256") == receipt_canonical_sha,
            f"RR2 shard receipt SHA rank {rank}",
        )
        _require(
            shard.get("static_artifact_sha256") == receipt["static_artifact_sha256"],
            f"RR2 shard static artifact rank {rank}",
        )
        _require(
            shard.get("protocol") == manifest.get("protocol"),
            f"RR2 shard protocol rank {rank}",
        )
        verified_shards.append(
            {
                "rank": rank,
                "relative_path": relative,
                "bytes": len(payload),
                "sha256": shard_ref.get("sha256"),
                "run_id": expected_run_id,
                "run_id_receipt_sha256": receipt_canonical_sha,
            }
        )
    _require(ranks == set(range(8)), "RR2 exact rank coverage")
    return {
        "manifest_raw_sha256": expected_manifest_sha256,
        "manifest_top_level_run_id_observed": None,
        "manifest_top_level_run_id_field_present": False,
        "run_id_receipt_raw_sha256": expected_run_id_receipt_raw_sha256,
        "run_id_receipt_canonical_json_sha256": receipt_canonical_sha,
        "verified_run_id": expected_run_id,
        "derivation_recomputed": True,
        "verified_shards": sorted(verified_shards, key=lambda row: row["rank"]),
    }


def corrected_original_receipts(
    *,
    original_receipt_manifest: Path,
    original_rr2_root: Path,
    expected_manifest_sha256: str,
    expected_run_id: str,
    expected_run_id_receipt_raw_sha256: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Apply the one-field correction after all frozen RR2 checks pass."""

    legacy_rows = frozen_builder.original_receipts(
        original_receipt_manifest=original_receipt_manifest,
        original_rr2_root=original_rr2_root,
    )
    binding = validate_rr2_run_binding(
        original_receipt_manifest=original_receipt_manifest,
        original_rr2_root=original_rr2_root,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_run_id=expected_run_id,
        expected_run_id_receipt_raw_sha256=expected_run_id_receipt_raw_sha256,
    )
    corrected = copy.deepcopy(legacy_rows)
    for mutant_id in frozen_builder.MUTANT_IDS:
        before = legacy_rows[mutant_id]
        _require(
            before.get("run_id") is None,
            f"{mutant_id} correction is only authorized for null legacy run_id",
        )
        corrected[mutant_id]["run_id"] = expected_run_id
        changed = {
            key
            for key in set(before) | set(corrected[mutant_id])
            if before.get(key) != corrected[mutant_id].get(key)
        }
        _require(changed == {"run_id"}, f"{mutant_id} correction field scope")
    return corrected, binding


def validate_amendment(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    amendment, raw = _load_canonical_json(args.correction_amendment, "correction amendment")
    amendment_sha = frozen_builder.sha256_bytes(raw)
    _require(
        amendment_sha == args.expected_correction_amendment_sha256,
        "correction amendment SHA binding",
    )
    _require(amendment.get("schema_version") == AMENDMENT_SCHEMA, "correction amendment schema")
    _require(amendment.get("workstream_id") == "E-R28-FULL-DETECTOR-MATRIX", "correction workstream")
    _require(
        amendment.get("created_after_candidate_execution") is True
        and amendment.get("candidate_outputs_observed_before_creation") is True,
        "correction post-execution timing",
    )
    _require(
        amendment.get("candidate_outputs_or_preexecution_sources_modified") is False,
        "correction immutability declaration",
    )
    source = amendment.get("correction_source")
    _require(isinstance(source, dict), "correction source binding")
    _require(
        source.get("wrapper_sha256") == frozen_builder.sha256_file(Path(__file__).resolve()),
        "correction wrapper SHA",
    )
    _require(
        source.get("test_sha256") == frozen_builder.sha256_file(args.correction_test_file),
        "correction test SHA",
    )
    frozen = amendment.get("frozen_execution_binding")
    _require(isinstance(frozen, dict), "correction frozen execution binding")
    _require(
        frozen.get("preexecution_builder_sha256")
        == frozen_builder.sha256_file(Path(frozen_builder.__file__).resolve()),
        "correction frozen builder SHA",
    )
    _require(
        frozen.get("preregistration_sha256")
        == frozen_builder.sha256_file(args.preregistration)
        == args.expected_preregistration_sha256,
        "correction preregistration SHA",
    )
    _require(
        frozen.get("original_rr2_manifest_sha256")
        == frozen_builder.sha256_file(args.original_receipt_manifest),
        "correction RR2 manifest SHA",
    )
    _require(
        frozen.get("original_rr2_run_id_receipt_raw_sha256")
        == frozen_builder.sha256_file(
            args.original_rr2_root / "receipts" / "run-id-receipt.json"
        ),
        "correction RR2 receipt raw SHA",
    )
    _require(
        frozen.get("raw_artifacts_ledger_sha256")
        == frozen_builder.sha256_file(args.raw_artifacts_ledger),
        "correction raw ledger SHA",
    )
    _require(
        amendment.get("authorized_transformation")
        == {
            "input": "legacy generated RR2 comparison rows",
            "field": "run_id",
            "before": None,
            "after": frozen.get("original_rr2_run_id"),
            "authority": (
                "derivation-verified canonical run-id receipt identically embedded "
                "in all eight detached-manifest-bound RR2 shards"
            ),
            "all_other_fields_unchanged": True,
        },
        "correction exact transformation",
    )
    return amendment, amendment_sha


def _aggregate_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        output=args.output,
        preregistration=args.preregistration,
        expected_preregistration_sha256=args.expected_preregistration_sha256,
        original_receipt_manifest=args.original_receipt_manifest,
        original_rr2_root=args.original_rr2_root,
        rank_root=args.rank_root,
        expected_runner_sha256=args.expected_runner_sha256,
        runner=args.runner,
        replay=args.replay,
        test_file=args.preexecution_test_file,
        launcher=args.launcher,
        gate_policy=args.gate_policy,
        qs_config=args.qs_config,
        scope_supersession=args.scope_supersession,
        external_pin_payload=args.external_pin_payload,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    amendment, amendment_sha = validate_amendment(args)
    frozen = amendment["frozen_execution_binding"]
    raw_ledger_receipt = validate_raw_artifact_ledger(
        ledger_path=args.raw_artifacts_ledger,
        rank_root=args.rank_root,
        expected_sha256=frozen["raw_artifacts_ledger_sha256"],
    )
    corrected_rows, binding = corrected_original_receipts(
        original_receipt_manifest=args.original_receipt_manifest,
        original_rr2_root=args.original_rr2_root,
        expected_manifest_sha256=frozen["original_rr2_manifest_sha256"],
        expected_run_id=frozen["original_rr2_run_id"],
        expected_run_id_receipt_raw_sha256=frozen[
            "original_rr2_run_id_receipt_raw_sha256"
        ],
    )

    original_function = frozen_builder.original_receipts

    def one_field_corrected_rows(
        *, original_receipt_manifest: Path, original_rr2_root: Path
    ) -> dict[str, dict[str, Any]]:
        _require(
            original_receipt_manifest.resolve()
            == args.original_receipt_manifest.resolve(),
            "correction manifest path drift",
        )
        _require(
            original_rr2_root.resolve() == args.original_rr2_root.resolve(),
            "correction RR2 root path drift",
        )
        return copy.deepcopy(corrected_rows)

    recorded = None
    if args.mode == "replay":
        _require(args.recorded_summary is not None, "replay recorded summary")
        recorded = args.recorded_summary.read_bytes()
        _require(
            recorded
            == frozen_builder.canonical_bytes(
                frozen_builder.load_json(args.recorded_summary)
            )
            + b"\n",
            "recorded summary canonical JSON",
        )
    try:
        frozen_builder.original_receipts = one_field_corrected_rows
        summary = frozen_builder.aggregate_from_paths(_aggregate_args(args))
    finally:
        frozen_builder.original_receipts = original_function

    summary_raw = args.output.read_bytes()
    byte_identical = None
    if recorded is not None:
        _require(summary_raw == recorded, "corrected replay summary is not byte-identical")
        byte_identical = True
    receipt = {
        "schema_version": CORRECTION_RECEIPT_SCHEMA,
        "mode": args.mode,
        "workstream_id": "E-R28-FULL-DETECTOR-MATRIX",
        "correction_amendment_sha256": amendment_sha,
        "correction_wrapper_sha256": frozen_builder.sha256_file(Path(__file__).resolve()),
        "correction_test_sha256": frozen_builder.sha256_file(args.correction_test_file),
        "frozen_preexecution_builder_sha256": frozen[
            "preexecution_builder_sha256"
        ],
        "preregistration_sha256": args.expected_preregistration_sha256,
        "raw_artifact_immutability": raw_ledger_receipt,
        "rr2_run_binding": binding,
        "authorized_in_memory_transformation": amendment[
            "authorized_transformation"
        ],
        "candidate_outputs_rewritten": False,
        "preexecution_sources_rewritten": False,
        "summary_sha256": frozen_builder.sha256_bytes(summary_raw),
        "summary_bytes": len(summary_raw),
        "recorded_summary_sha256": (
            frozen_builder.sha256_bytes(recorded) if recorded is not None else None
        ),
        "byte_identical_to_recorded_summary": byte_identical,
        "scientific_valid": summary["scientific_valid"],
        "scientific_outcome": summary["scientific_outcome"],
    }
    frozen_builder.write_json(args.correction_receipt_output, receipt)
    return receipt


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--mode", choices=("aggregate", "replay"), required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--recorded-summary", type=Path)
    result.add_argument("--correction-receipt-output", type=Path, required=True)
    result.add_argument("--correction-amendment", type=Path, required=True)
    result.add_argument("--expected-correction-amendment-sha256", required=True)
    result.add_argument("--correction-test-file", type=Path, required=True)
    result.add_argument("--raw-artifacts-ledger", type=Path, required=True)
    result.add_argument("--preregistration", type=Path, required=True)
    result.add_argument("--expected-preregistration-sha256", required=True)
    result.add_argument("--rank-root", type=Path, required=True)
    result.add_argument("--original-receipt-manifest", type=Path, required=True)
    result.add_argument("--original-rr2-root", type=Path, required=True)
    result.add_argument("--expected-runner-sha256", required=True)
    result.add_argument("--runner", type=Path, required=True)
    result.add_argument("--replay", type=Path, required=True)
    result.add_argument("--preexecution-test-file", type=Path, required=True)
    result.add_argument("--launcher", type=Path, required=True)
    result.add_argument("--gate-policy", type=Path, required=True)
    result.add_argument("--qs-config", type=Path, required=True)
    result.add_argument("--scope-supersession", type=Path, required=True)
    result.add_argument("--external-pin-payload", type=Path, required=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.mode == "aggregate":
        _require(args.recorded_summary is None, "aggregate forbids recorded summary")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
