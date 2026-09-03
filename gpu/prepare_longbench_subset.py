from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_dataset(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise argparse.ArgumentTypeError("dataset must be NAME=PATH")
    return name, Path(raw_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", type=parse_dataset, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--source-repo", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.start < 0 or args.count < 1:
        raise SystemExit("require start >= 0 and count >= 1")

    selected = []
    for dataset, path in args.dataset:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        subset = rows[args.start : args.start + args.count]
        if len(subset) != args.count:
            raise SystemExit(
                f"{dataset}: requested {args.count} rows from {args.start}, found {len(subset)}"
            )
        for source_index, row in enumerate(subset, start=args.start):
            if row.get("dataset") != dataset:
                raise SystemExit(
                    f"{path}: row {source_index} says dataset={row.get('dataset')!r}"
                )
            row["_source_repo"] = args.source_repo
            row["_source_revision"] = args.source_revision
            row["_source_index"] = source_index
            selected.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(selected)} rows to {args.output}")


if __name__ == "__main__":
    main()
