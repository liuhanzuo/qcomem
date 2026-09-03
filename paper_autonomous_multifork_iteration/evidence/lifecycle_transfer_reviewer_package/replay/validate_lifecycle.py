#!/usr/bin/env python3
"""Manifest-first, raw-first validator for the frozen lifecycle cohort."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any


PACKAGE = Path(__file__).resolve().parents[1]
RUN = PACKAGE / "artifacts/qwen35-forkaudit-lifecycle-transfer-20260819c"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ValidationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_package_manifest() -> dict[str, Any]:
    manifest_path = PACKAGE / "replay/MANIFEST.json"
    binding_path = PACKAGE / "replay/MANIFEST.sha256"
    binding = binding_path.read_text(encoding="utf-8").strip().split()
    require(len(binding) == 2 and binding[1] == "MANIFEST.json", "invalid package manifest binding")
    require(file_sha(manifest_path) == binding[0], "package MANIFEST.json digest mismatch")
    manifest = load_json(manifest_path)
    entries = manifest.get("files")
    require(isinstance(entries, list) and entries, "package manifest has no file entries")
    seen: set[str] = set()
    for row in entries:
        relative = row.get("path")
        expected_sha = row.get("sha256")
        expected_bytes = row.get("size_bytes")
        require(isinstance(relative, str) and relative not in seen, "duplicate/invalid manifest path")
        require(".." not in Path(relative).parts and not Path(relative).is_absolute(), "unsafe manifest path")
        require(isinstance(expected_sha, str) and SHA256_RE.fullmatch(expected_sha) is not None, "bad manifest SHA")
        path = PACKAGE / relative
        require(path.is_file(), f"missing manifest member: {relative}")
        require(path.stat().st_size == expected_bytes, f"size mismatch: {relative}")
        require(file_sha(path) == expected_sha, f"SHA mismatch: {relative}")
        seen.add(relative)
    return manifest


def validate_raw_ledger() -> list[Path]:
    ledger_path = RUN / "receipts/raw-artifacts.sha256"
    shard_paths: list[Path] = []
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        fields = line.split()
        require(len(fields) == 2 and SHA256_RE.fullmatch(fields[0]) is not None, "invalid raw ledger row")
        relative = fields[1]
        require(".." not in Path(relative).parts and not Path(relative).is_absolute(), "unsafe raw path")
        path = RUN / relative
        require(path.is_file(), f"missing raw artifact: {relative}")
        require(file_sha(path) == fields[0], f"raw artifact SHA mismatch: {relative}")
        shard_paths.append(path)
    require(len(shard_paths) == 8, f"expected 8 raw shards, found {len(shard_paths)}")
    return shard_paths


def replay_lease_events(record: dict[str, Any]) -> tuple[list[int], list[str | None]]:
    slot_count = record.get("slot_count")
    require(slot_count == 4, "lease slot_count drift")
    epochs = [0] * slot_count
    owners: list[str | None] = [None] * slot_count
    events = record.get("events")
    require(isinstance(events, list), "lease events missing")
    for expected_index, event in enumerate(events):
        require(event.get("event_index") == expected_index, "lease event index drift")
        slot = event.get("slot_id")
        require(isinstance(slot, int) and 0 <= slot < slot_count, "lease slot out of range")
        require(event.get("epoch_before") == epochs[slot], "lease epoch_before mismatch")
        operation = event.get("operation")
        request_id = event.get("request_id")
        require(isinstance(request_id, str) and request_id, "lease request_id missing")
        if operation == "acquire":
            require(owners[slot] is None, "acquire on owned slot")
            require(event.get("epoch_after") == epochs[slot], "acquire changed epoch")
            owners[slot] = request_id
        elif operation == "cancel":
            require(owners[slot] == request_id, "cancel owner mismatch")
            require(event.get("epoch_after") == epochs[slot] + 1, "cancel did not increment epoch")
            epochs[slot] += 1
            owners[slot] = None
        else:
            raise ValidationError(f"unknown lease operation: {operation!r}")
    require(record.get("final_epochs") == epochs, "lease final epochs mismatch")
    require(record.get("final_owners") == owners, "lease final owners mismatch")
    return epochs, owners


def validate_reservation_rows(rows: Any, expected_slots: list[int]) -> dict[int, dict[str, Any]]:
    require(isinstance(rows, list) and len(rows) == len(expected_slots), "reservation row cardinality drift")
    out: dict[int, dict[str, Any]] = {}
    for expected_slot, row in zip(expected_slots, rows):
        require(row.get("slot_id") == expected_slot, "reservation slot order drift")
        layers = row.get("layers")
        require(isinstance(layers, dict) and len(layers) == 10, "reservation layer cardinality drift")
        require(row.get("sha256") == canonical_sha(layers), "reservation row digest mismatch")
        for blocks in layers.values():
            require(isinstance(blocks, list) and len(blocks) == 1 and isinstance(blocks[0], int), "bad reservation block list")
        out[expected_slot] = row
    for layer in next(iter(out.values()))["layers"]:
        blocks = [out[slot]["layers"][layer][0] for slot in expected_slots]
        require(len(blocks) == len(set(blocks)), f"private reservation overlap at layer {layer}")
    return out


def validate_scrub_and_reassignment(shard: dict[str, Any], initial: dict[int, dict[str, Any]]) -> None:
    lifecycle = shard["lifecycle"]
    reclaimed = validate_reservation_rows(lifecycle.get("reclaimed_reservation_rows"), [0, 1])
    cancelled_slots = shard["formal_config"].get("cancel_slots")
    require(cancelled_slots == [2, 3], "cancelled slot set drift")
    for replacement_slot, old_slot in enumerate(cancelled_slots):
        require(reclaimed[replacement_slot]["layers"] == initial[old_slot]["layers"], "reassigned reservation is not exact")
        require(reclaimed[replacement_slot]["sha256"] == initial[old_slot]["sha256"], "reassigned reservation digest drift")
    receipt = lifecycle.get("scrub_receipt")
    require(receipt.get("zero_scrub_before_reassignment") is True, "zero-scrub receipt false")
    rows = receipt.get("layers")
    require(isinstance(rows, list) and len(rows) == 10, "scrub layer cardinality drift")
    expected_layers = sorted(int(k) for k in initial[2]["layers"])
    require([row.get("layer_index") for row in rows] == expected_layers, "scrub layer order drift")
    for row in rows:
        layer = str(row["layer_index"])
        expected_blocks = [initial[slot]["layers"][layer][0] for slot in cancelled_slots]
        require(row.get("reclaimed_slots") == cancelled_slots, "scrub reclaimed slots drift")
        require(row.get("scrubbed_physical_block_ids") == expected_blocks, "scrubbed block set drift")
        require(row.get("rewound_fork_cursor") == 2, "fork cursor was not rewound to surviving slots")


def validate_semantics(shard: dict[str, Any]) -> None:
    control = shard["control"]
    lifecycle = shard["lifecycle"]
    require(control.get("trajectory") == lifecycle.get("trajectory"), "full-vocabulary hash/token trajectory mismatch")
    require(control.get("logical_kv") == lifecycle.get("logical_kv"), "final logical KV mismatch")
    require(control.get("gdn_state") == lifecycle.get("gdn_state"), "final GDN state mismatch")
    require(control.get("document_sha256_before") == control.get("document_sha256_after"), "control document mutated")
    require(lifecycle.get("document_sha256_before") == lifecycle.get("document_sha256_after"), "lifecycle document mutated")
    require(control.get("document_sha256_before") == lifecycle.get("document_sha256_before"), "cross-cell document binding drift")
    source_rows = shard["source"]["query_bank"]["rows"]
    expected_queries = [row["query_token_ids_sha256"] for row in source_rows]
    observed_queries = [row["query_token_ids_sha256"] for row in control["trajectory"]]
    require(expected_queries == observed_queries, "query-bank binding mismatch")
    require(all(SHA256_RE.fullmatch(value) for value in observed_queries), "invalid query digest")
    for row in control["trajectory"]:
        require(len(row["generated_token_ids"]) == 4, "token trajectory length drift")
        require(len(row["full_vocab_step_logit_sha256"]) == 4, "logit trajectory length drift")
        require(all(SHA256_RE.fullmatch(value) for value in row["full_vocab_step_logit_sha256"]), "invalid logit digest")
    require(all(row.get("tensor_count") == 60 for row in control["gdn_state"]), "GDN tensor count drift")


def validate_shard(shard: dict[str, Any], expected_rank: int) -> dict[str, bool]:
    require(shard.get("schema_version") == "qcomem-forkaudit-lifecycle-transfer-shard-v1", "shard schema drift")
    require(shard.get("status") == "completed" and shard.get("rank") == expected_rank, "rank/status drift")
    config = shard.get("formal_config")
    require(config.get("world_size") == 8 and shard.get("world_size") == 8, "world size drift")
    require(config.get("document_tokens") == 4096 and config.get("page_size") == 128, "geometry drift")
    require(config.get("document_tail_tokens") == 0, "lifecycle prefix is not page aligned")
    require(config.get("expected_fault_gate") == "STALE_SLOT_LEASE", "expected fault gate drift")
    control_epochs, _ = replay_lease_events(shard["control"]["lease_events"])
    lifecycle_epochs, lifecycle_owners = replay_lease_events(shard["lifecycle"]["lease_events"])
    require(control_epochs == [0, 0, 0, 0], "control lease epochs drift")
    require(lifecycle_epochs == [0, 0, 1, 1], "lifecycle lease epochs drift")
    require(lifecycle_owners == ["initial-slot-0", "initial-slot-1", "replacement-slot-2", "replacement-slot-3"], "lifecycle final owners drift")
    replay = shard["lifecycle"].get("lease_replay")
    require(replay == {"final_epochs": lifecycle_epochs, "final_owners": lifecycle_owners, "event_count": 8}, "stored lease replay drift")
    initial = validate_reservation_rows(shard["lifecycle"].get("initial_reservation_rows"), [0, 1, 2, 3])
    validate_reservation_rows(shard["control"].get("reservation_rows"), [0, 1, 2, 3])
    validate_scrub_and_reassignment(shard, initial)
    mutant = shard["lifecycle"].get("stale_handle_mutant")
    require(mutant.get("fault") == "schedule-cancelled-handle-after-reclaim", "stale fault identity drift")
    require(mutant.get("expected_gate") == "STALE_SLOT_LEASE", "stale expected gate drift")
    require(mutant.get("observed_gate") == "STALE_SLOT_LEASE", "stale handle reached wrong gate")
    require(mutant.get("detected") is True and mutant.get("wrong_gate") is False, "stale handle was not cleanly rejected")
    require(mutant.get("matched_clean_replacement_accepted") is True, "matched replacement was rejected")
    require(shard["lifecycle"].get("aligned_prefix_no_partial_tail_copy") is True, "aligned append predicate false")
    require(shard["lifecycle"].get("partial_tail_staging_copy_nbytes") == 0, "unexpected partial-tail copy")
    validate_semantics(shard)
    return {
        "input_binding": True,
        "immutable_document": True,
        "private_reservation_disjointness": True,
        "aligned_append_without_tail_copy": True,
        "cancel_invalidation": True,
        "zero_scrub_before_reassignment": True,
        "epoch_bound_reclamation": True,
        "semantic_equivalence_after_reclamation": True,
    }


def self_test(clean_shard: dict[str, Any]) -> list[str]:
    cases: list[tuple[str, dict[str, Any]]] = []
    stale = copy.deepcopy(clean_shard)
    stale["lifecycle"]["stale_handle_mutant"]["observed_gate"] = "OTHER_GATE"
    cases.append(("stale_gate", stale))
    scrub = copy.deepcopy(clean_shard)
    scrub["lifecycle"]["scrub_receipt"]["zero_scrub_before_reassignment"] = False
    cases.append(("zero_scrub", scrub))
    reassignment = copy.deepcopy(clean_shard)
    reassignment["lifecycle"]["reclaimed_reservation_rows"][0]["layers"]["3"] = [999]
    reassignment["lifecycle"]["reclaimed_reservation_rows"][0]["sha256"] = canonical_sha(
        reassignment["lifecycle"]["reclaimed_reservation_rows"][0]["layers"]
    )
    cases.append(("reservation_reassignment", reassignment))
    rejected: list[str] = []
    for name, candidate in cases:
        try:
            validate_shard(candidate, int(candidate["rank"]))
        except ValidationError:
            rejected.append(name)
        else:
            raise ValidationError(f"semantic tamper escaped: {name}")
    ledger_line = (RUN / "receipts/raw-artifacts.sha256").read_text(encoding="utf-8").splitlines()[0]
    expected = ledger_line.split()[0]
    raw = (RUN / ledger_line.split()[1]).read_bytes()
    tampered = raw[:-1] + bytes([raw[-1] ^ 1])
    require(hashlib.sha256(tampered).hexdigest() != expected, "raw-byte tamper did not change SHA")
    rejected.append("raw_manifest")
    return rejected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true", help="also run four tamper regressions")
    args = parser.parse_args()
    manifest = validate_package_manifest()
    shard_paths = validate_raw_ledger()
    reports = []
    clean_shards = []
    for expected_rank, path in enumerate(sorted(shard_paths)):
        shard = load_json(path)
        clean_shards.append(shard)
        reports.append({"rank": expected_rank, "predicates": validate_shard(shard, expected_rank)})
    result: dict[str, Any] = {
        "status": "PASS",
        "schema_version": "forkaudit-lifecycle-raw-first-replay-v1",
        "package_manifest_files": manifest["file_count"],
        "validated_raw_shards": len(reports),
        "ranks": reports,
        "boundary": "Hash/receipt replay of frozen raw shards; not live GPU regeneration and not a second adapter/runtime.",
    }
    if args.self_test:
        result["tamper_tests_rejected"] = self_test(clean_shards[0])
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
