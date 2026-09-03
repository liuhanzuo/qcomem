#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SYSTEM = "sglang-0.5.17-qwen35-radix-extra-buffer"
PAIRS = [
    ["qasper", 6], ["qasper", 7], ["qasper", 8], ["qasper", 9],
    ["2wikimqa", 6], ["2wikimqa", 7], ["2wikimqa", 8], ["2wikimqa", 9],
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


manifest = json.loads((ROOT / "MANIFEST.json").read_text())
assert manifest["schema"] == "related-serving-reviewer-package-manifest-v1"
for row in manifest["files"]:
    path = ROOT / row["path"]
    assert path.is_file(), row["path"]
    assert path.stat().st_size == row["bytes"], row["path"]
    assert sha256(path) == row["sha256"], row["path"]

phases: dict[str, list[dict]] = {"cache_off": [], "cache_on": []}
for phase in phases:
    for rank, pair in enumerate(PAIRS):
        row = json.loads((ROOT / "raw" / f"{phase}-rank-{rank}.json").read_text())
        assert row["schema"] == "forkaudit-related-serving-shard-v1"
        assert row["status"] == "completed"
        assert row["system"] == SYSTEM
        assert row["phase"] == phase and row["rank"] == rank and row["world_size"] == 8
        assert [row["workload"]["dataset"], row["workload"]["source_index"]] == pair
        measured = row["measured"]
        assert isinstance(measured["prediction"], str)
        assert measured["usage"]["completion_tokens"] > 0
        assert measured["stream_event_count"] > 0
        phases[phase].append(row)

exact = []
hits = []
phase_summaries = {}
for rank in range(8):
    off = phases["cache_off"][rank]["measured"]
    on = phases["cache_on"][rank]["measured"]
    exact.append(off["prediction"] == on["prediction"])
    details = on["usage"].get("prompt_tokens_details") or {}
    counter_hits = phases["cache_on"][rank]["prefix_counters"]["measured_delta"].get("hits")
    hits.append(details.get("cached_tokens", 0) > 0 or (counter_hits or 0) > 0)

for phase, shards in phases.items():
    rows = [row["measured"] for row in shards]
    tpots = [row["median_tpot_seconds"] for row in rows if row["median_tpot_seconds"] is not None]
    phase_summaries[phase] = {
        "mean_f1": statistics.mean(float(row["f1"]) for row in rows),
        "median_ttft_seconds": statistics.median(float(row["ttft_seconds"]) for row in rows),
        "median_tpot_seconds": statistics.median(float(value) for value in tpots),
        "median_generated_tokens_per_second": statistics.median(
            float(row["generated_tokens_per_second"]) for row in rows
        ),
        "predictions": [row["prediction"] for row in rows],
    }

replayed = {
    "schema": "forkaudit-related-serving-summary-v1",
    "scientific_run_valid": True,
    "hypothesis_passed": all(exact) and all(hits),
    "scientific_outcome": "valid_positive" if all(exact) and all(hits) else "valid_negative",
    "system": SYSTEM,
    "comparison_boundary": "same-model-same-slice-openai-streaming-serving-only",
    "not_comparable_to": "in-process CoMem direct-adapter wall-clock",
    "pairs": PAIRS,
    "cache_off_vs_on_prediction_exact": exact,
    "cache_hit_observed": hits,
    "phases": phase_summaries,
    "raw_shards": {
        phase: [f"{phase}-rank-{rank}.json" for rank in range(8)]
        for phase in phases
    },
}
recorded = json.loads((ROOT / "summary.json").read_text())
assert replayed == recorded
assert canonical_bytes(replayed) == (ROOT / "summary.json").read_bytes()
assert sha256(ROOT / "summary.json") == "509b0c6a148313eac1ab7f5d6011bff80bd0e7e8c91fc054b34a20f54a17070d"
assert (ROOT / "stages" / "COMPLETE").is_file()
print("SGLANG_REVIEWER_REPLAY_PASS: 16 shards; 8/8 hits; 8/8 exact predictions")
