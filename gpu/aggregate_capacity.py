from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_capacity_scaling import aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    result = aggregate(args.run_dir)
    destination = args.run_dir / "capacity_analysis.json"
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
