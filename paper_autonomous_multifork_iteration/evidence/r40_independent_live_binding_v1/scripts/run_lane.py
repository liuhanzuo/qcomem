from __future__ import annotations

import argparse
import json
from pathlib import Path

from r40lib.lane import execute_lane
from r40lib.provenance import load_json, sha256_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--expected-preregistration-sha256", required=True)
    parser.add_argument("--source-ledger", type=Path, required=True)
    parser.add_argument("--expected-source-ledger-sha256", required=True)
    parser.add_argument("--fault-id", required=True)
    parser.add_argument("--lane-type", choices=("clean", "mutant"), required=True)
    args = parser.parse_args()
    if sha256_file(args.preregistration) != args.expected_preregistration_sha256:
        raise RuntimeError("preregistration hash drift")
    if sha256_file(args.source_ledger) != args.expected_source_ledger_sha256:
        raise RuntimeError("source ledger hash drift")
    preregistration = load_json(args.preregistration)
    matches = [row for row in preregistration["faults"] if row["fault_id"] == args.fault_id]
    if len(matches) != 1:
        raise RuntimeError("fault id did not resolve exactly once")
    result = execute_lane(
        preregistration,
        matches[0],
        lane_type=args.lane_type,
        preregistration_sha256=args.expected_preregistration_sha256,
        source_ledger_sha256=args.expected_source_ledger_sha256,
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

