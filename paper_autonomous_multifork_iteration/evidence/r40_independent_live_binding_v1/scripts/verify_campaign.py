from __future__ import annotations

"""Standard-library-only independent replay of the R40 campaign."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


COORDINATE_FIELDS = (
    "owner_kind",
    "request_index",
    "layer_index",
    "state_family",
    "state_index",
)
DESCRIPTOR_FIELDS = (
    "shape",
    "stride",
    "storage_offset",
    "dtype",
    "device",
    "tensor_nbytes",
    "storage_nbytes",
    "byte_start",
    "byte_end_exclusive",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_new(path: Path, value: Any) -> None:
    require(not path.exists(), f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def verify_seal(value: Mapping[str, Any], field: str) -> None:
    unsigned = dict(value)
    observed = unsigned.get(field)
    unsigned[field] = None
    require(observed == sha256_json(unsigned), f"{field} drift")


def coordinate(value: Mapping[str, Any]) -> dict[str, Any]:
    return {field: value[field] for field in COORDINATE_FIELDS}


def slot_id(value: Mapping[str, Any]) -> str:
    return "s-" + sha256_json(
        {"domain": "r40-independent-live-binding-slot-v1", **coordinate(value)}
    )[:20]


def verify_source_ledger(root: Path, ledger_path: Path) -> dict[str, Any]:
    ledger = load_json(ledger_path)
    require(
        ledger.get("schema_version")
        == "forkaudit-r40-independent-live-binding-source-ledger-v1",
        "source ledger schema drift",
    )
    require(len(ledger["files"]) == int(ledger["file_count"]), "source count drift")
    for row in ledger["files"]:
        path = root / row["path"]
        require(path.is_file(), f"source missing: {row['path']}")
        require(path.stat().st_size == int(row["bytes"]), f"source size drift: {row['path']}")
        require(sha256_file(path) == row["sha256"], f"source hash drift: {row['path']}")
    return ledger


def verify_observation(value: Mapping[str, Any], expected_role: str) -> None:
    verify_seal(value, "payload_sha256")
    require(value.get("role") == expected_role, "observation role drift")
    require(value.get("live_item_fields_received") == ["slot_id", "tensor"], "wire field drift")
    require(value.get("raw_addresses_serialized") is False, "raw address receipt drift")
    rows = value["rows"]
    relations = value["relations"]
    require(len(rows) == int(value["row_count"]), "row count drift")
    require(len(relations) == int(value["relation_count"]), "relation count drift")
    require(sha256_json(rows) == value["rows_sha256"], "row digest drift")
    require(sha256_json(relations) == value["relations_sha256"], "relation digest drift")
    require(len({row["slot_id"] for row in rows}) == len(rows), "duplicate observation slot")


def replay_detector(
    oracle: Mapping[str, Any], observation: Mapping[str, Any]
) -> dict[str, Any]:
    oracle_rows = {row["slot_id"]: row for row in oracle["rows"]}
    observed_rows = {row["slot_id"]: row for row in observation["rows"]}
    missing = sorted(set(oracle_rows) - set(observed_rows))
    unexpected = sorted(set(observed_rows) - set(oracle_rows))
    codes: list[str] = []
    if missing or unexpected:
        codes.append("slot_set_mismatch")
    common = sorted(set(oracle_rows) & set(observed_rows))
    descriptors = [
        slot
        for slot in common
        if any(oracle_rows[slot][field] != observed_rows[slot][field] for field in DESCRIPTOR_FIELDS)
    ]
    if descriptors:
        codes.append("descriptor_mismatch")
    challenges = [
        slot
        for slot in common
        if oracle_rows[slot]["challenge_response_sha256"]
        != observed_rows[slot]["challenge_response_sha256"]
    ]
    if challenges:
        codes.append("challenge_response_mismatch")
    oracle_relations = {(a, b): r for a, b, r in oracle["relations"]}
    observed_relations = {(a, b): r for a, b, r in observation["relations"]}
    relation_mismatches = [
        [a, b, oracle_relations.get((a, b)), observed_relations.get((a, b))]
        for a, b in sorted(set(oracle_relations) | set(observed_relations))
        if oracle_relations.get((a, b)) != observed_relations.get((a, b))
    ]
    if relation_mismatches:
        codes.append("storage_relation_mismatch")
    return {
        "schema_version": "forkaudit-r40-live-binding-detector-v1",
        "passed": not codes,
        "failure_codes": codes,
        "missing_slot_ids": missing,
        "unexpected_slot_ids": unexpected,
        "descriptor_mismatch_slot_ids": descriptors,
        "challenge_mismatch_slot_ids": challenges,
        "relation_mismatch_pairs": relation_mismatches,
        "numeric_tolerance": 0,
    }


def verify_manifest(manifest: Mapping[str, Any], expected_count: int) -> None:
    unsigned = dict(manifest)
    observed = unsigned.pop("manifest_sha256", None)
    require(sha256_json(unsigned) == observed, "semantic manifest digest drift")
    rows = manifest["slots"]
    require(len(rows) == expected_count, "semantic manifest count drift")
    require(len({row["slot_id"] for row in rows}) == expected_count, "semantic slot duplicate")
    require(all(row["slot_id"] == slot_id(row) for row in rows), "slot binding drift")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--source-ledger", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    preregistration_path = args.preregistration.resolve()
    source_ledger_path = args.source_ledger.resolve()
    campaign_path = args.campaign.resolve()
    prereg = load_json(preregistration_path)
    require(
        prereg.get("schema_version")
        == "forkaudit-r40-independent-live-binding-preregistration-v1",
        "preregistration schema drift",
    )
    source_ledger = verify_source_ledger(root, source_ledger_path)
    campaign = load_json(campaign_path)
    verify_seal(campaign, "campaign_payload_sha256")
    require(
        campaign.get("schema_version")
        == "forkaudit-r40-independent-live-binding-campaign-v1",
        "campaign schema drift",
    )
    prereg_sha = sha256_file(preregistration_path)
    ledger_sha = sha256_file(source_ledger_path)
    require(source_ledger["preregistration_sha256"] == prereg_sha, "ledger prereg binding drift")
    require(campaign["preregistration_sha256"] == prereg_sha, "campaign prereg binding drift")
    require(campaign["source_ledger_sha256"] == ledger_sha, "campaign source binding drift")
    expected_order = [
        f"{fault['fault_id']}/{lane_type}"
        for fault in prereg["faults"]
        for lane_type in ("clean", "mutant")
    ]
    require(campaign["lane_order"] == expected_order, "lane order drift")
    lanes = campaign["lanes"]
    require(len(lanes) == len(expected_order) == 8, "lane count drift")
    by_pair: dict[tuple[str, str], Mapping[str, Any]] = {}
    producer_pids: list[int] = []
    replay_exact = 0
    for lane in lanes:
        verify_seal(lane, "lane_payload_sha256")
        key = (lane["fault_id"], lane["lane_type"])
        require(key not in by_pair, "duplicate lane")
        by_pair[key] = lane
        producer_pids.append(int(lane["producer_pid"]))
        require(lane["preregistration_sha256"] == prereg_sha, "lane prereg drift")
        require(lane["source_ledger_sha256"] == ledger_sha, "lane source drift")
        verify_manifest(lane["semantic_manifest"], int(prereg["fixture"]["expected_slots_per_lane"]))
        oracle = lane["oracle"]
        observation = lane["observation"]
        verify_observation(oracle, "pre_injection_oracle")
        verify_observation(observation, "post_binding_observer")
        require(
            oracle["manifest_sha256"]
            == observation["manifest_sha256"]
            == lane["semantic_manifest"]["manifest_sha256"],
            "lane manifest binding drift",
        )
        require(
            len({int(lane["producer_pid"]), int(lane["oracle_pid"]), int(lane["observer_pid"])}) == 3,
            "lane process separation drift",
        )
        require(oracle["parent_pid"] == lane["producer_pid"], "oracle parent drift")
        require(observation["parent_pid"] == lane["producer_pid"], "observer parent drift")
        require(lane["environment"]["fixture_device"] == "cpu", "fixture device drift")
        require(lane["environment"]["cuda_available"] is False, "unexpected CUDA claim")
        replayed = replay_detector(oracle, observation)
        require(replayed == lane["detector"], "detector replay drift")
        replay_exact += 1
        stable = {
            "manifest_sha256": oracle["manifest_sha256"],
            "rows": oracle["rows"],
            "relations": oracle["relations"],
        }
        require(
            sha256_json(stable) == lane["stable_oracle_projection_sha256"],
            "stable oracle projection drift",
        )
        injection = lane["injection_receipt"]
        require(injection["schema_or_label_row_mutation_used"] is False, "label mutation used")
        require(injection["raw_addresses_serialized"] is False, "raw address receipt drift")
        require(
            injection["semantic_manifest_sha256_before"]
            == injection["semantic_manifest_sha256_after"]
            == lane["semantic_manifest"]["manifest_sha256"],
            "injection manifest drift",
        )
        require(
            injection["live_wire_fields_before"]
            == injection["live_wire_fields_after"]
            == ["slot_id", "tensor"],
            "injection wire drift",
        )
        fault = next(row for row in prereg["faults"] if row["fault_id"] == lane["fault_id"])
        target_slots = sorted(slot_id(row) for row in fault["targets"])
        if lane["lane_type"] == "clean":
            require(replayed["passed"] is True, "clean control failed")
            require(not replayed["failure_codes"], "clean control emitted failure code")
            require(injection["applied"] is False, "clean injection receipt drift")
            require(injection["changed_slot_ids"] == [], "clean live handle changed")
        else:
            require(replayed["passed"] is False, "mutant escaped")
            require(injection["applied"] is True, "mutant injection missing")
            require(injection["actual_live_tensor_references_changed"] is True, "mutant was not live")
            require(injection["changed_slot_ids"] == target_slots, "mutant changed-slot drift")
            require(len(target_slots) == int(fault["expected_changed_slots"]), "target count drift")
            require(
                all(code in replayed["failure_codes"] for code in fault["required_detection_codes"]),
                "required detection code missing",
            )
            require(
                all(row["reference_changed"] for row in injection["object_token_rows"]),
                "object-token reference receipt drift",
            )
    require(len(set(producer_pids)) == len(producer_pids), "producer process reuse")
    for fault in prereg["faults"]:
        clean = by_pair[(fault["fault_id"], "clean")]
        mutant = by_pair[(fault["fault_id"], "mutant")]
        require(
            clean["stable_oracle_projection_sha256"]
            == mutant["stable_oracle_projection_sha256"],
            f"matched oracle drift: {fault['fault_id']}",
        )
    controls = [by_pair[(fault["fault_id"], "clean")] for fault in prereg["faults"]]
    mutants = [by_pair[(fault["fault_id"], "mutant")] for fault in prereg["faults"]]
    expected_aggregate = {
        "matched_clean_controls": 4,
        "clean_controls_accepted": 4,
        "mutants": 4,
        "mutants_failed_closed": 4,
        "pair_oracle_projections_exact": True,
        "producer_pids_all_distinct": True,
        "schema_or_label_mutation_faults": 0,
        "actual_live_handle_faults": 4,
        "campaign_passed": True,
    }
    require(campaign["aggregate"] == expected_aggregate, "campaign aggregate drift")
    require(campaign["status"] == "completed", "campaign status drift")
    require(campaign["claim_boundary"] == prereg["claim_boundary"], "claim boundary drift")
    verification = {
        "schema_version": "forkaudit-r40-independent-live-binding-verification-v1",
        "experiment_id": prereg["experiment_id"],
        "passed": True,
        "preregistration_sha256": prereg_sha,
        "source_ledger_sha256": ledger_sha,
        "campaign_raw_sha256": sha256_file(campaign_path),
        "campaign_payload_sha256": campaign["campaign_payload_sha256"],
        "source_files_verified": int(source_ledger["file_count"]),
        "lanes_replayed_exact": replay_exact,
        "matched_clean_controls_accepted": len(controls),
        "mutants_failed_closed": len(mutants),
        "actual_live_handle_faults": 4,
        "schema_or_label_mutation_faults": 0,
        "fault_detection_codes": {
            lane["fault_id"]: lane["detector"]["failure_codes"] for lane in mutants
        },
        "claim_boundary": prereg["claim_boundary"],
    }
    write_json_new(args.output.resolve(), verification)
    print("PASS replayed=8 clean=4 mutants=4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

