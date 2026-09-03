from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

from r40lib.provenance import SOURCE_LEDGER_SCHEMA, sha256_file, write_json_new


def source_paths(root: Path) -> list[Path]:
    fixed = [
        root / "README.md",
        root / "DESIGN.md",
        root / "CLAIM_BOUNDARY.md",
        root / "preregistration.json",
        root / "formal/run_local_formal.sh",
    ]
    discovered: list[Path] = []
    for directory in ("r40lib", "scripts", "tests"):
        discovered.extend(sorted((root / directory).glob("*.py")))
    paths = sorted(set(fixed + discovered), key=lambda path: path.relative_to(root).as_posix())
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"source files missing: {missing}")
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    entries = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in source_paths(root)
    ]
    preregistration = root / "preregistration.json"
    ledger = {
        "schema_version": SOURCE_LEDGER_SCHEMA,
        "experiment_id": "R40-INDEPENDENT-LIVE-BINDING-20260827A",
        "frozen_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "file_count": len(entries),
        "files": entries,
        "preregistration_sha256": sha256_file(preregistration),
    }
    write_json_new(args.ledger, ledger)
    receipt = {
        "schema_version": "forkaudit-r40-independent-live-binding-freeze-receipt-v1",
        "experiment_id": ledger["experiment_id"],
        "source_ledger_path": args.ledger.resolve().relative_to(root).as_posix(),
        "source_ledger_sha256": sha256_file(args.ledger),
        "preregistration_sha256": sha256_file(preregistration),
        "campaign_outputs_existed_at_freeze": False,
    }
    write_json_new(args.receipt, receipt)
    print(f"frozen {len(entries)} files")
    print(f"source_ledger_sha256={receipt['source_ledger_sha256']}")
    print(f"preregistration_sha256={receipt['preregistration_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

