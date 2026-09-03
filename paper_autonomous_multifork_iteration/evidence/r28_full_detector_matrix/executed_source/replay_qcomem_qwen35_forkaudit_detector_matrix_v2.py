from __future__ import annotations

"""Replay and byte-verify an R28 detector-matrix summary from raw ranks.

Replay reopens every rank and FP32 sidecar, revalidates the detached RR2
reference, rebuilds the canonical summary, and requires exact byte equality
with the recorded aggregate.  It never executes model code.
"""

import argparse
from pathlib import Path
from typing import Sequence

import build_qcomem_qwen35_forkaudit_detector_matrix_v2 as builder


REPLAY_SCHEMA = "forkaudit-detector-matrix-replay-v2"


def replay(args: argparse.Namespace) -> dict[str, object]:
    recorded = args.summary.read_bytes()
    builder.require(
        recorded == builder.canonical_bytes(builder.load_json(args.summary)) + b"\n",
        "recorded summary is not canonical JSON",
    )
    aggregate_args = argparse.Namespace(
        output=args.output,
        preregistration=args.preregistration,
        expected_preregistration_sha256=args.expected_preregistration_sha256,
        original_receipt_manifest=args.original_receipt_manifest,
        original_rr2_root=args.original_rr2_root,
        rank_root=args.rank_root,
        expected_runner_sha256=args.expected_runner_sha256,
        runner=args.runner,
        replay=Path(__file__).resolve(),
        test_file=args.test_file,
        launcher=args.launcher,
        gate_policy=args.gate_policy,
        qs_config=args.qs_config,
        scope_supersession=args.scope_supersession,
        external_pin_payload=args.external_pin_payload,
    )
    rebuilt = builder.aggregate_from_paths(aggregate_args)
    rebuilt_bytes = args.output.read_bytes()
    builder.require(rebuilt_bytes == recorded, "replayed summary is not byte-identical")
    return {
        "schema_version": REPLAY_SCHEMA,
        "summary_sha256": builder.sha256_bytes(recorded),
        "summary_bytes": len(recorded),
        "rank_receipts_verified": len(rebuilt["rank_receipts"]),
        "sidecar_receipts_verified": len(rebuilt["sidecar_receipts"]),
        "byte_identical": True,
        "scientific_valid": rebuilt["scientific_valid"],
        "scientific_outcome": rebuilt["scientific_outcome"],
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--summary", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--preregistration", type=Path, required=True)
    result.add_argument("--expected-preregistration-sha256", required=True)
    result.add_argument("--rank-root", type=Path, required=True)
    result.add_argument("--original-receipt-manifest", type=Path, required=True)
    result.add_argument("--original-rr2-root", type=Path, required=True)
    result.add_argument("--expected-runner-sha256", required=True)
    result.add_argument("--runner", type=Path, required=True)
    result.add_argument("--test-file", type=Path, required=True)
    result.add_argument("--launcher", type=Path, required=True)
    result.add_argument("--gate-policy", type=Path, required=True)
    result.add_argument("--qs-config", type=Path, required=True)
    result.add_argument("--scope-supersession", type=Path, required=True)
    result.add_argument("--external-pin-payload", type=Path, required=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    receipt = replay(args)
    print(builder.canonical_bytes(receipt).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
