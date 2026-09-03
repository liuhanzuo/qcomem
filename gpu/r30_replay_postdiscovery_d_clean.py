from __future__ import annotations

"""Candidate-import-free detached replay for the R30 D clean regression.

This scorer imports only the Python standard library.  It recomputes raw file
hashes, FP32 sidecar properties, normalized-storage relations, binding-token
relations, and every frozen clean acceptance predicate from serialized bytes.
"""

import argparse
from array import array
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "forkaudit-r30-postdiscovery-d-clean-detached-replay-v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
STORAGE_ID_RE = re.compile(r"^storage-[0-9]{4,}$")
EXPECTED_SHAPE = [1, 248320]
EXPECTED_NBYTES = 993280
EXPECTED_FAMILIES = ("conv_states", "recurrent_states")
WITNESS_FAMILIES = ("conv", "recurrent")
DTYPE_BYTES = {
    "torch.float16": 2,
    "torch.bfloat16": 2,
    "torch.float32": 4,
    "torch.float64": 8,
}


class ReplayError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReplayError(message)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def load_json(path: Path) -> Any:
    return json.loads(path.read_bytes())


def check_sha(value: Any, label: str) -> str:
    require(isinstance(value, str) and SHA256_RE.fullmatch(value) is not None, f"{label} SHA-256 drift")
    return value


def parse_pre_replay_ledger(path: Path, run_dir: Path) -> list[dict[str, Any]]:
    expected = {
        "preregistration.json",
        "raw/borrowed-repaired-fp32-logits.bin",
        "raw/materialized-control-fp32-logits.bin",
        "raw/clean-result.json",
    }
    rows: list[dict[str, Any]] = []
    observed: set[str] = set()
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        match = re.fullmatch(r"([0-9a-f]{64})  ([^\n]+)", line)
        require(match is not None, f"ledger line {line_number} schema drift")
        digest, relative = match.groups()
        require(relative in expected, f"unexpected ledger path {relative}")
        require(relative not in observed, f"duplicate ledger path {relative}")
        artifact = (run_dir / relative).resolve()
        require(artifact.is_relative_to(run_dir.resolve()), f"ledger path escapes run directory: {relative}")
        require(artifact.is_file(), f"ledger artifact missing: {relative}")
        actual = sha256_file(artifact)
        require(actual == digest, f"ledger artifact hash drift: {relative}")
        observed.add(relative)
        rows.append({"path": relative, "sha256": digest, "nbytes": artifact.stat().st_size})
    require(observed == expected, "pre-replay ledger file set drift")
    return rows


def fp32_receipt(path: Path) -> tuple[dict[str, Any], bytes, list[float]]:
    payload = path.read_bytes()
    require(len(payload) == EXPECTED_NBYTES, f"{path.name} byte count drift")
    require(sys.byteorder == "little", "detached replay requires a little-endian host")
    values = array("f")
    values.frombytes(payload)
    require(len(values) == EXPECTED_SHAPE[0] * EXPECTED_SHAPE[1], f"{path.name} shape drift")
    finite = all(math.isfinite(value) for value in values)
    require(finite, f"{path.name} contains non-finite FP32 values")
    maximum = max(range(len(values)), key=values.__getitem__)
    return (
        {
            "path": path.name,
            "sha256": sha256_bytes(payload),
            "dtype": "float32-little-endian",
            "shape": EXPECTED_SHAPE,
            "nbytes": len(payload),
            "finite": finite,
            "argmax_token_id": maximum,
        },
        payload,
        values.tolist(),
    )


def interval_from_row(row: Mapping[str, Any], *, family_style: str) -> tuple[int, int]:
    shape = row.get("shape")
    stride = row.get("stride")
    require(isinstance(shape, list) and isinstance(stride, list) and len(shape) == len(stride) and bool(shape), "row shape/stride drift")
    require(all(type(size) is int and size > 0 for size in shape), "row shape drift")
    require(all(type(step) is int for step in stride), "row stride drift")
    if family_style == "diagnostic":
        dtype = row.get("dtype")
        require(dtype in DTYPE_BYTES, "diagnostic dtype drift")
        element_size = DTYPE_BYTES[dtype]
        # The post-discovery diagnostic emits byte ranges directly but not the
        # element storage offset.  Reconstruct its conservative interval from
        # byte_start and verify width from shape/stride.
        start = row.get("byte_start")
        require(type(start) is int and start >= 0, "diagnostic byte_start drift")
        minimum = 0
        maximum = 0
        for size, step in zip(shape, stride):
            displacement = (size - 1) * step
            minimum += min(displacement, 0)
            maximum += max(displacement, 0)
        require(minimum == 0, "diagnostic negative-stride offset is unsupported")
        expected_end = start + (maximum + 1) * element_size
        end = row.get("byte_end_exclusive")
        require(end == expected_end, "diagnostic byte interval drift")
    else:
        dtype = row.get("dtype")
        require(dtype in DTYPE_BYTES, "witness dtype drift")
        offset = row.get("storage_offset")
        require(type(offset) is int and offset >= 0, "witness storage offset drift")
        minimum = offset
        maximum = offset
        for size, step in zip(shape, stride):
            displacement = (size - 1) * step
            minimum += min(displacement, 0)
            maximum += max(displacement, 0)
        require(minimum >= 0, "witness negative byte interval")
        start = minimum * DTYPE_BYTES[dtype]
        end = (maximum + 1) * DTYPE_BYTES[dtype]
        require(row.get("byte_start") == start and row.get("byte_end_exclusive") == end, "witness byte interval drift")
    storage_nbytes = row.get("storage_nbytes")
    require(type(storage_nbytes) is int and 0 <= start < end <= storage_nbytes, "row interval outside storage")
    tensor_nbytes = row.get("tensor_nbytes")
    product = 1
    for size in shape:
        product *= size
    require(tensor_nbytes == product * DTYPE_BYTES[dtype], "row tensor_nbytes drift")
    return start, end


def overlaps(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return left["storage_id"] == right["storage_id"] and max(left["byte_start"], right["byte_start"]) < min(left["byte_end_exclusive"], right["byte_end_exclusive"])


def exact_alias(left: Mapping[str, Any], right: Mapping[str, Any], *, require_object: bool) -> bool:
    fields = (
        "storage_id",
        "byte_start",
        "byte_end_exclusive",
        "shape",
        "stride",
        "dtype",
        "device",
        "storage_nbytes",
        "tensor_nbytes",
        "content_sha256",
    )
    if require_object:
        fields += ("tensor_id",)
    return all(left.get(field) == right.get(field) for field in fields)


def diagnostic_map(snapshot: Mapping[str, Any], label: str) -> dict[tuple[str, str], Mapping[str, Any]]:
    require(snapshot.get("row_count") == 180, f"{label} row count drift")
    rows = snapshot.get("rows")
    require(isinstance(rows, list) and len(rows) == 180, f"{label} rows drift")
    require(snapshot.get("rows_sha256") == sha256_json(rows), f"{label} row digest drift")
    require(snapshot.get("absolute_pointers_persisted") is False, f"{label} pointer flag drift")
    require(snapshot.get("python_object_ids_persisted") is False, f"{label} object-id flag drift")
    output: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in rows:
        require(row.get("owner") in ("persistent", "request_0", "request_1"), f"{label} owner drift")
        require(row.get("state_family") in EXPECTED_FAMILIES, f"{label} state family drift")
        require(isinstance(row.get("storage_id"), str) and STORAGE_ID_RE.fullmatch(row["storage_id"]) is not None, f"{label} storage ID drift")
        interval_from_row(row, family_style="diagnostic")
        check_sha(row.get("content_sha256"), f"{label} content")
        key = (row["owner"], row["coordinate"])
        require(key not in output, f"{label} duplicate coordinate")
        output[key] = row
    require(len(output) == 180, f"{label} coordinate cardinality drift")
    return output


def stable_owner_row(before: Mapping[str, Any], after: Mapping[str, Any]) -> bool:
    fields = (
        "tensor_id",
        "storage_id",
        "byte_start",
        "byte_end_exclusive",
        "shape",
        "stride",
        "dtype",
        "device",
        "storage_nbytes",
        "tensor_nbytes",
        "content_sha256",
    )
    return all(before.get(field) == after.get(field) for field in fields)


def replay_diagnostic_snapshots(result: Mapping[str, Any]) -> dict[str, Any]:
    borrowed = result["borrowed_repaired"]
    setup = diagnostic_map(borrowed["setup_snapshot"], "setup")
    pre = diagnostic_map(borrowed["pre_kernel_snapshot"], "pre-kernel")
    post = diagnostic_map(borrowed["post_snapshot"], "post")
    coordinates = sorted(coordinate for owner, coordinate in setup if owner == "persistent")
    require(len(coordinates) == 60, "diagnostic coordinate count drift")
    counts = {
        "setup_request0_base_exact_alias": 0,
        "setup_request1_base_exact_alias": 0,
        "setup_request0_peer_exact_alias": 0,
        "pre_request0_conv_disjoint": 0,
        "pre_request0_recurrent_exact_alias": 0,
        "post_request0_base_disjoint": 0,
        "post_request0_peer_disjoint": 0,
        "base_peer_exact_alias_all_phases": 0,
        "persistent_stable_setup_to_post": 0,
        "peer_stable_setup_to_post": 0,
    }
    for coordinate in coordinates:
        s_base, s_r0, s_r1 = setup[("persistent", coordinate)], setup[("request_0", coordinate)], setup[("request_1", coordinate)]
        p_base, p_r0, p_r1 = pre[("persistent", coordinate)], pre[("request_0", coordinate)], pre[("request_1", coordinate)]
        o_base, o_r0, o_r1 = post[("persistent", coordinate)], post[("request_0", coordinate)], post[("request_1", coordinate)]
        require(exact_alias(s_r0, s_base, require_object=True), f"setup request0/base alias drift at {coordinate}")
        require(exact_alias(s_r1, s_base, require_object=True), f"setup request1/base alias drift at {coordinate}")
        require(exact_alias(s_r0, s_r1, require_object=True), f"setup request0/peer alias drift at {coordinate}")
        counts["setup_request0_base_exact_alias"] += 1
        counts["setup_request1_base_exact_alias"] += 1
        counts["setup_request0_peer_exact_alias"] += 1
        family = s_base["state_family"]
        if family == "conv_states":
            require(not overlaps(p_r0, p_base) and not overlaps(p_r0, p_r1), f"pre-kernel conv ownership drift at {coordinate}")
            require(p_r0["content_sha256"] == s_r0["content_sha256"], f"conv clone content drift at {coordinate}")
            counts["pre_request0_conv_disjoint"] += 1
        else:
            require(exact_alias(p_r0, p_base, require_object=True), f"pre-kernel recurrent alias drift at {coordinate}")
            counts["pre_request0_recurrent_exact_alias"] += 1
        require(not overlaps(o_r0, o_base), f"post request0/base overlap at {coordinate}")
        require(not overlaps(o_r0, o_r1), f"post request0/peer overlap at {coordinate}")
        counts["post_request0_base_disjoint"] += 1
        counts["post_request0_peer_disjoint"] += 1
        for base, peer in ((s_base, s_r1), (p_base, p_r1), (o_base, o_r1)):
            require(exact_alias(base, peer, require_object=True), f"base/peer alias drift at {coordinate}")
        counts["base_peer_exact_alias_all_phases"] += 1
        require(stable_owner_row(s_base, p_base) and stable_owner_row(s_base, o_base), f"persistent changed at {coordinate}")
        require(stable_owner_row(s_r1, p_r1) and stable_owner_row(s_r1, o_r1), f"unadvanced peer changed at {coordinate}")
        counts["persistent_stable_setup_to_post"] += 1
        counts["peer_stable_setup_to_post"] += 1
    expected = {
        "setup_request0_base_exact_alias": 60,
        "setup_request1_base_exact_alias": 60,
        "setup_request0_peer_exact_alias": 60,
        "pre_request0_conv_disjoint": 30,
        "pre_request0_recurrent_exact_alias": 30,
        "post_request0_base_disjoint": 60,
        "post_request0_peer_disjoint": 60,
        "base_peer_exact_alias_all_phases": 60,
        "persistent_stable_setup_to_post": 60,
        "peer_stable_setup_to_post": 60,
    }
    require(counts == expected, "diagnostic relation count drift")
    return counts


def content_manifest(rows: Sequence[Mapping[str, Any]]) -> str:
    fields = (
        "layer_index",
        "state_family",
        "state_index",
        "shape",
        "stride",
        "storage_offset",
        "dtype",
        "device",
        "storage_nbytes",
        "tensor_nbytes",
        "byte_start",
        "byte_end_exclusive",
        "content_sha256",
    )
    return sha256_json([{field: row[field] for field in fields} for row in rows])


def witness_coordinate(row: Mapping[str, Any]) -> tuple[int, str, int]:
    return row["layer_index"], row["state_family"], row["state_index"]


def replay_unified_witness(result: Mapping[str, Any]) -> dict[str, Any]:
    phase = result["borrowed_repaired"]["gdn_phase_witness"]
    require(phase.get("capture_protocol") == "unified-live-gdn-phase-v1", "phase capture protocol drift")
    require(phase.get("phase") == "post_transition", "phase label drift")
    storage = phase["storage_witness"]
    require(storage.get("schema_version") == "qcomem-gdn-storage-witness-v1", "storage schema drift")
    require(storage.get("phase") == "post_transition" and storage.get("policy") == "shared-base", "storage policy drift")
    require(storage.get("resident_count") == 2 and storage.get("completed_request_indices") == [0], "storage lifecycle drift")
    indices = storage.get("layer_indices")
    require(isinstance(indices, list) and len(indices) == 30 and len(set(indices)) == 30, "storage layer plan drift")
    rows = storage.get("rows")
    require(isinstance(rows, list) and len(rows) == 180, "storage row count drift")
    require(storage.get("rows_sha256") == sha256_json(rows), "storage row digest drift")
    normalized: list[str] = []
    metadata: dict[str, tuple[str, int]] = {}
    owners: dict[tuple[str, int | None], list[Mapping[str, Any]]] = {}
    for row in rows:
        require(row.get("state_family") in WITNESS_FAMILIES, "storage state family drift")
        interval_from_row(row, family_style="witness")
        storage_id = row.get("storage_id")
        require(isinstance(storage_id, str) and STORAGE_ID_RE.fullmatch(storage_id) is not None, "storage ID drift")
        if storage_id not in metadata:
            require(storage_id == f"storage-{len(normalized):04d}", "storage normalized order drift")
            normalized.append(storage_id)
            metadata[storage_id] = (row["device"], row["storage_nbytes"])
        require(metadata[storage_id] == (row["device"], row["storage_nbytes"]), "storage metadata conflict")
        owner = (row["owner_kind"], row["request_index"])
        owners.setdefault(owner, []).append(row)
    require(set(owners) == {("persistent", None), ("request", 0), ("request", 1)}, "storage owner set drift")
    require(all(len(owner_rows) == 60 for owner_rows in owners.values()), "storage per-owner count drift")
    base = {witness_coordinate(row): row for row in owners[("persistent", None)]}
    r0 = {witness_coordinate(row): row for row in owners[("request", 0)]}
    r1 = {witness_coordinate(row): row for row in owners[("request", 1)]}
    require(set(base) == set(r0) == set(r1) and len(base) == 60, "storage coordinate set drift")
    disjoint_base = disjoint_peer = peer_alias = 0
    for coordinate in sorted(base):
        require(not overlaps(r0[coordinate], base[coordinate]), f"unified request0/base overlap at {coordinate}")
        require(not overlaps(r0[coordinate], r1[coordinate]), f"unified request0/peer overlap at {coordinate}")
        require(exact_alias(r1[coordinate], base[coordinate], require_object=False), f"unified peer/base alias drift at {coordinate}")
        disjoint_base += 1
        disjoint_peer += 1
        peer_alias += 1
    guard = storage.get("persistent_guard")
    require(isinstance(guard, dict), "persistent guard missing")
    require(guard.get("baseline_binding_sha256") == guard.get("observed_binding_sha256"), "persistent binding guard drift")
    require(guard.get("baseline_content_sha256") == guard.get("observed_content_sha256"), "persistent content guard drift")
    require(content_manifest(owners[("persistent", None)]) == guard.get("observed_content_sha256"), "persistent row manifest drift")
    binding = phase["binding_witness"]
    binding_rows = binding.get("rows")
    require(isinstance(binding_rows, list) and len(binding_rows) == 120, "binding row count drift")
    require(binding.get("rows_sha256") == sha256_json(binding_rows), "binding row digest drift")
    require(binding.get("resident_count") == 2 and binding.get("completed_request_indices") == [0], "binding lifecycle drift")
    rebound = unchanged = 0
    for row in binding_rows:
        for field in ("baseline_binding_token", "observed_binding_token", "baseline_storage_token", "observed_storage_token"):
            check_sha(row.get(field), f"binding {field}")
        if row.get("request_index") == 0:
            require(row.get("expected_relation") == "rebound", "completed binding relation drift")
            require(row["baseline_storage_token"] != row["observed_storage_token"], "completed storage token unchanged")
            rebound += 1
        elif row.get("request_index") == 1:
            require(row.get("expected_relation") == "unchanged", "peer binding relation drift")
            require(row["baseline_binding_token"] == row["observed_binding_token"], "peer binding token changed")
            require(row["baseline_storage_token"] == row["observed_storage_token"], "peer storage token changed")
            unchanged += 1
        else:
            raise ReplayError("binding request index drift")
    require((rebound, unchanged) == (60, 60), "binding relation counts drift")
    return {
        "normalized_storage_count": len(normalized),
        "completed_vs_base_disjoint": disjoint_base,
        "completed_vs_peer_disjoint": disjoint_peer,
        "incomplete_peer_vs_base_exact_alias": peer_alias,
        "rebound_binding_count": rebound,
        "unchanged_binding_count": unchanged,
        "persistent_guard_exact": True,
    }


def replay_transition_receipts(result: Mapping[str, Any]) -> dict[str, Any]:
    first = result["borrowed_repaired"]["transition_receipt"]
    repeat = result["borrowed_repaired"]["repeat_helper_receipt"]
    for label, receipt, clones, action in (
        ("first", first, 30, "cloned_borrowed_state"),
        ("repeat", repeat, 0, "already_private_noop"),
    ):
        require(receipt.get("schema_version") == "qcomem-single-token-gdn-conv-privatization-v1", f"{label} repair schema drift")
        require(receipt.get("request_index") == 0 and receipt.get("resident_count") == 2, f"{label} repair ownership drift")
        require(receipt.get("conv_tensor_count") == 30 and receipt.get("cloned_tensor_count") == clones, f"{label} repair clone count drift")
        rows = receipt.get("rows")
        require(isinstance(rows, list) and len(rows) == 30, f"{label} repair row count drift")
        require(receipt.get("rows_sha256") == sha256_json(rows), f"{label} repair row digest drift")
        for row in rows:
            require(row.get("action") == action, f"{label} repair action drift")
            require(row.get("base_disjoint") is True and row.get("all_peers_disjoint") is True, f"{label} repair disjointness drift")
            check_sha(row.get("content_sha256"), f"{label} repair content")
    return {"first_clone_count": 30, "repeat_clone_count": 0, "ownership_only": True}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--pre-replay-ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "detached replay output already exists")
    protocol_sha = sha256_file(args.protocol)
    require(protocol_sha == check_sha(args.expected_protocol_sha256, "expected protocol"), "protocol raw hash drift")
    protocol = load_json(args.protocol)
    require(protocol.get("schema_version") == "forkaudit-r30-postdiscovery-d-clean-prereg-v1", "protocol schema drift")
    require(protocol.get("candidate_output_seen_when_frozen") is False, "protocol outcome-blind flag drift")
    require(protocol.get("faults") == {"loaded": False, "executed": False, "known_r29_faults_excluded": ["H01", "H02", "H03"]}, "protocol fault exclusion drift")
    ledger_rows = parse_pre_replay_ledger(args.pre_replay_ledger, args.run_dir)
    require(sha256_file(args.run_dir / "preregistration.json") == protocol_sha, "run-local preregistration drift")
    result_path = args.run_dir / "raw" / "clean-result.json"
    result = load_json(result_path)
    require(result.get("schema_version") == "forkaudit-r30-postdiscovery-d-clean-v1", "clean result schema drift")
    require(result.get("run_id") == protocol.get("run_id"), "clean result run ID drift")
    require(result.get("status") == "valid_clean_positive", "clean status is not positive")
    require(result.get("post_discovery") is True, "post-discovery flag drift")
    require(result.get("faults_executed") is False and result.get("known_r29_fault_module_loaded") is False, "fault isolation drift")
    require(result.get("heldout_fault_claim_allowed") is False and result.get("paper_import_allowed") is False, "claim boundary flag drift")
    result_bindings = result.get("source_bindings")
    require(isinstance(result_bindings, dict), "result source bindings missing")
    require(result_bindings.get("protocol_raw_sha256") == protocol_sha, "result protocol binding drift")
    require({key: value for key, value in result_bindings.items() if key != "protocol_raw_sha256"} == protocol.get("source_bindings"), "result source bindings drift")
    borrowed_receipt, borrowed_bytes, borrowed_values = fp32_receipt(args.run_dir / "raw" / "borrowed-repaired-fp32-logits.bin")
    control_receipt, control_bytes, control_values = fp32_receipt(args.run_dir / "raw" / "materialized-control-fp32-logits.bin")
    comparisons = result.get("comparisons")
    require(isinstance(comparisons, dict), "comparison receipt missing")
    require(comparisons.get("borrowed_logits_sha256") == borrowed_receipt["sha256"], "borrowed sidecar binding drift")
    require(comparisons.get("control_logits_sha256") == control_receipt["sha256"], "control sidecar binding drift")
    require(borrowed_bytes == control_bytes, "canonical FP32 sidecars differ")
    require(borrowed_values == control_values, "decoded FP32 vectors differ")
    require(borrowed_receipt["argmax_token_id"] == result["borrowed_repaired"]["model_step"]["greedy_token_id"], "borrowed argmax/token drift")
    require(control_receipt["argmax_token_id"] == result["materialized_control"]["model_step"]["greedy_token_id"], "control argmax/token drift")
    require(all(comparisons.get(field) is True for field in ("greedy_token_exact", "canonical_fp32_logits_byte_exact", "terminal_request_0_gdn_exact", "terminal_logical_kv_exact")), "clean exactness predicate drift")
    require(result["borrowed_repaired"]["digests"] == result["materialized_control"]["digests"], "terminal digest comparison drift")
    transition = replay_transition_receipts(result)
    diagnostic = replay_diagnostic_snapshots(result)
    unified = replay_unified_witness(result)
    persistent_guard = result["borrowed_repaired"]["persistent_guard"]
    require(persistent_guard["baseline_binding_sha256"] == persistent_guard["observed_binding_sha256"], "borrowed persistent binding drift")
    require(persistent_guard["baseline_content_sha256"] == persistent_guard["observed_content_sha256"], "borrowed persistent content drift")
    cleanup = result.get("cleanup")
    require(cleanup.get("exact") is True, "allocator cleanup exact flag drift")
    require(cleanup.get("before") == cleanup.get("allocator_baseline") == cleanup.get("after_borrowed") == cleanup.get("after_control"), "allocator cleanup snapshots differ")
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "run_id": result["run_id"],
        "status": "detached_clean_replay_passed",
        "candidate_modules_imported": False,
        "python_modules_used": sorted(name for name in sys.modules if name.split(".")[0] in {"argparse", "array", "hashlib", "json", "math", "pathlib", "re", "sys", "typing"}),
        "protocol_raw_sha256": protocol_sha,
        "pre_replay_ledger_raw_sha256": sha256_file(args.pre_replay_ledger),
        "verified_artifacts": ledger_rows,
        "sidecars": {"borrowed_repaired": borrowed_receipt, "materialized_control": control_receipt, "byte_exact": True},
        "transition_replay": transition,
        "diagnostic_storage_replay": diagnostic,
        "unified_storage_and_binding_replay": unified,
        "terminal_exactness_replay": {
            "greedy_token_exact": True,
            "canonical_fp32_logits_byte_exact": True,
            "terminal_request_0_gdn_exact": True,
            "terminal_logical_kv_exact": True,
        },
        "allocator_cleanup_exact": True,
        "faults_loaded": False,
        "faults_executed": False,
        "heldout_fault_claim_allowed": False,
        "paper_import_allowed": False,
        "claim_boundary": protocol["claim_boundary"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_bytes(receipt) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
