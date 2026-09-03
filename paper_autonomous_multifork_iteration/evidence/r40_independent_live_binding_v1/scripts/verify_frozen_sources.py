from __future__ import annotations

import argparse
from pathlib import Path

from r40lib.provenance import sha256_file, verify_source_ledger


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    args = parser.parse_args()
    ledger = verify_source_ledger(args.root.resolve(), args.ledger.resolve())
    print(f"PASS source files={ledger['file_count']}")
    print(f"source_ledger_sha256={sha256_file(args.ledger.resolve())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

