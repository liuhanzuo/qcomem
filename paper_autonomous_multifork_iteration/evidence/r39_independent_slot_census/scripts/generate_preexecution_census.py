from __future__ import annotations

"""Freeze the independently derived census before a fresh H20 producer starts."""

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any, Sequence

from audit_independent_slot_census import (
    derive_expected_census,
    sha256_file,
    sha256_json,
    validate_protocol,
    write_json,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-ledger", type=Path, required=True)
    parser.add_argument("--census-output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    args = parser.parse_args(argv)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    validate_protocol(protocol)
    census = derive_expected_census(protocol)
    slots = [census[slot_id] for slot_id in sorted(census)]
    semantic_sha = sha256_json(slots)
    census_document: dict[str, Any] = {
        "schema_version": "forkaudit-r39-preexecution-slot-census-v1",
        "status": "frozen_before_fresh_h20_producer_start",
        "experiment_id": protocol["experiment_id"],
        "protocol_raw_sha256": sha256_file(args.protocol),
        "source_ledger_raw_sha256": sha256_file(args.source_ledger),
        "producer_manifest_used": False,
        "producer_rows_used": False,
        "slot_count": len(slots),
        "census_semantic_sha256": semantic_sha,
        "slots": slots,
    }
    write_json(args.census_output, census_document)
    receipt = {
        "schema_version": "forkaudit-r39-preexecution-census-receipt-v1",
        "status": "frozen_before_fresh_h20_producer_start",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "experiment_id": protocol["experiment_id"],
        "protocol_raw_sha256": sha256_file(args.protocol),
        "source_ledger_raw_sha256": sha256_file(args.source_ledger),
        "census_file_sha256": sha256_file(args.census_output),
        "census_semantic_sha256": semantic_sha,
        "derived_slot_count": len(slots),
        "producer_started": False,
        "producer_manifest_available_to_derivation": False,
        "producer_rows_available_to_derivation": False,
    }
    write_json(args.receipt_output, receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
