from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from analyze_validation import comparison
from run_interface_diagnostic import LORA_VALIDATION_CONFIGS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--expected-data-sha256", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    args = parser.parse_args()
    shards = [json.loads(path.read_text()) for path in args.run_dir.glob("shard-*-*.json")]
    expected = 8 * len(LORA_VALIDATION_CONFIGS)
    if len(shards) != expected:
        raise SystemExit(f"expected {expected} shard files, found {len(shards)}")
    merged = {}
    shard_data_hashes = {item.get("data_sha256") for item in shards}
    checkpoint_hashes = {
        item.get("lora", {}).get("checkpoint_sha256") for item in shards
    }
    if shard_data_hashes != {args.expected_data_sha256}:
        raise SystemExit(f"validation data SHA mismatch across shards: {shard_data_hashes}")
    if checkpoint_hashes != {args.expected_checkpoint_sha256}:
        raise SystemExit(f"LoRA checkpoint SHA mismatch across shards: {checkpoint_hashes}")
    for name in LORA_VALIDATION_CONFIGS:
        matching = [item for item in shards if item["config"] == name]
        if len(matching) != 8:
            raise SystemExit(f"{name}: expected 8 shards, found {len(matching)}")
        rows = [row for item in matching for row in item["rows"]]
        keys = {(row["dataset"], row["id"], row["source_index"]) for row in rows}
        if len(rows) != 60 or len(keys) != 60:
            raise SystemExit(f"{name}: expected 60 unique rows, found {len(rows)}/{len(keys)}")
        for dataset in ("qasper", "2wikimqa"):
            indices = {
                int(row["source_index"])
                for row in rows
                if row["dataset"] == dataset
            }
            if indices != set(range(6, 36)):
                raise SystemExit(f"{name}/{dataset}: expected source indices 6-35")
        merged[name] = {
            "config": name,
            "rows": rows,
            "mean_f1": statistics.fmean(row["f1"] for row in rows),
        }
    baseline = merged["chunk-d7"]
    adapted = merged["chunk-lora-d7"]
    result = {
        "status": "completed",
        "protocol": "interface-lora-validation-source-index-6-35",
        "samples": 60,
        "validation_data_sha256": args.expected_data_sha256,
        "lora_checkpoint_sha256": args.expected_checkpoint_sha256,
        "test_v2_used": False,
        "summary": {name: item["mean_f1"] for name, item in merged.items()},
        "chunk_lora_vs_chunk_frozen": comparison(adapted, baseline, seed=20260812),
        "chunk_lora_vs_dense": comparison(adapted, merged["dense"], seed=20260912),
    }
    destination = args.run_dir / "interface_lora_analysis.json"
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
