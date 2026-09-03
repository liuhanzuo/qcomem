from __future__ import annotations

"""Standard-library detached replay for one R35 historical-alias rank.

The replay treats producer booleans as receipts, not verdicts.  It reopens the
three FP32 sidecars, reconstructs lane-local storage relations from normalized
pointer-free snapshots, authenticates any reported first rejection, and keeps
operational validity separate from the scientific hypothesis outcomes.
"""

import argparse
from array import array
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence


RANK_SCHEMA = "forkaudit-r35-historical-alias-rank-v1"
UPSTREAM_R29_RUN_ID = "R29-HELDOUT-FAULTS-20260825C"
UPSTREAM_R29_RAW_SHA256 = "5a522e48650e3010621e6e06c7c8bbab67c074bca1b6a6a0c70aa50133b4e98d"
PROTOCOL_SCHEMA = "forkaudit-r35-historical-alias-protocol-v1"
AMENDMENT_SCHEMA = "forkaudit-r35-resource-amendment-v1"
REPLAY_SCHEMA = "forkaudit-r35-historical-alias-rank-replay-v1"
LANES = ("historical_pre_fix", "repaired_borrowed", "materialized_control")
SHARED_LANES = frozenset(("historical_pre_fix", "repaired_borrowed"))
RECEIPT_ORDER = (
    "frozen_input_and_request_provenance",
    "live_kv_ownership_and_construction_binding",
    "gdn_phase_storage_snapshot_and_pointer_free_replay",
    "advertised_scheduler_action_sequence_replay",
    "persistent_kv_and_gdn_immutability",
    "fresh_case_disposal_pending",
)
GATE_COMPLETED_REBOUND = "gdn_completed_binding_rebound"
GATE_PERSISTENT_IMMUTABLE = "gdn_persistent_immutable"
ALLOWED_HISTORICAL_GATES = frozenset((GATE_COMPLETED_REBOUND,))
OWNERS = ("persistent", "request_0", "request_1")
FAMILIES = ("conv_states", "recurrent_states")
DTYPE_NBYTES = {
    "torch.float16": 2,
    "torch.bfloat16": 2,
    "torch.float32": 4,
    "torch.float64": 8,
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
STORAGE_RE = re.compile(r"^storage-[0-9]{4,}$")
TENSOR_RE = re.compile(r"^tensor-[0-9]{4,}$")


class ReplayError(RuntimeError):
    """An evidence-integrity or operational-validity failure."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReplayError(message)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    require(path.is_file(), f"missing JSON artifact: {path}")
    try:
        return json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReplayError(f"invalid JSON artifact {path}: {exc}") from exc


def check_sha(value: Any, label: str) -> str:
    require(
        isinstance(value, str) and SHA256_RE.fullmatch(value) is not None,
        f"{label} must be one lowercase SHA-256",
    )
    return value


def exact_int(value: Any, label: str, *, minimum: int = 0) -> int:
    require(type(value) is int and value >= minimum, f"{label} integer drift")
    return value


def exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    require(set(value) == expected, f"{label} schema drift: {sorted(set(value) ^ expected)}")


def safe_child(root: Path, relative_value: Any, label: str) -> Path:
    require(isinstance(relative_value, str) and relative_value, f"{label} path")
    relative = Path(relative_value)
    require(not relative.is_absolute() and ".." not in relative.parts, f"{label} unsafe path")
    root_resolved = root.resolve()
    path = (root / relative).resolve()
    require(path.is_relative_to(root_resolved), f"{label} path escape")
    return path


def _validate_sha_collection(value: Any, label: str) -> None:
    if isinstance(value, str):
        check_sha(value, label)
    elif isinstance(value, Mapping):
        require(bool(value), f"{label} empty SHA map")
        for key, child in value.items():
            _validate_sha_collection(child, f"{label}.{key}")
    elif isinstance(value, list):
        require(bool(value), f"{label} empty SHA list")
        for index, child in enumerate(value):
            _validate_sha_collection(child, f"{label}[{index}]")
    else:
        raise ReplayError(f"{label} is not a SHA or SHA collection")


def _validate_hash_tree(value: Any, label: str) -> None:
    """Validate all fields advertised as SHA values in a nested receipt."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered = str(key).lower()
            if lowered == "sha256" or lowered.endswith("_sha256"):
                _validate_sha_collection(child, f"{label}.{key}")
            else:
                _validate_hash_tree(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_hash_tree(child, f"{label}[{index}]")


def _validate_module_closure(value: Any, label: str) -> None:
    require(isinstance(value, Mapping), f"{label} object")
    rows = value.get("modules")
    require(isinstance(rows, list) and rows, f"{label} rows")
    require(value.get("module_count") == len(rows), f"{label} module count")
    require(value.get("modules_sha256") == sha256_json(rows), f"{label} digest")
    require(value.get("shadowed_module_count") == 0, f"{label} shadowed modules")
    names: set[str] = set()
    for row in rows:
        require(isinstance(row, Mapping), f"{label} row")
        exact_keys(row, {"module", "source_class", "path", "sha256"}, f"{label} row")
        name = row.get("module")
        require(isinstance(name, str) and name and name not in names, f"{label} module name")
        names.add(name)
        require(
            row.get("source_class") in {"r35_package_override", "imported_rr2_ledger"},
            f"{label} source class",
        )
        require(isinstance(row.get("path"), str) and row["path"], f"{label} module path")
        check_sha(row.get("sha256"), f"{label} module SHA")
    required = {
        "build_qcomem_forkaudit_rr2_input_manifest",
        "qcomem_forkaudit_storage_witness",
        "qcomem_joint_policy",
        "qcomem_qwen35_functional_stack",
        "qcomem_single_token_gdn_ownership",
        "qcomem_vllm_paged_fair_control",
        "qcomem_vllm_paged_kernel",
        "qcomem_vllm_paged_multifork_resident",
        "run_qcomem_qwen35_forkaudit_review_revision",
        "run_qcomem_qwen35_vllm_paged_multifork_resident",
    }
    require(required.issubset(names), f"{label} required module coverage")


def _reject_absolute_identity_fields(value: Any, label: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered = str(key).lower()
            require(
                not (
                    "data_ptr" in lowered
                    or "storage_ptr" in lowered
                    or lowered in {"pointer", "address", "absolute_address", "python_object_id"}
                ),
                f"absolute identity field at {label}.{key}",
            )
            _reject_absolute_identity_fields(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_absolute_identity_fields(child, f"{label}[{index}]")


def validate_protocol(protocol: Any) -> dict[str, Any]:
    require(isinstance(protocol, Mapping), "protocol object")
    exact_keys(
        protocol,
        {
            "schema_version",
            "run_id",
            "rank_count",
            "lanes",
            "lane_order_by_rank",
            "lane_audit_mode",
            "sidecar",
            "receipt_order",
            "historical_first_failure",
            "expected",
            "content_digest_formula",
            "coordinate_classes",
            "comparison_matrix",
            "source_bindings",
            "resource_amendment_binding",
        },
        "protocol",
    )
    require(protocol.get("schema_version") == PROTOCOL_SCHEMA, "protocol schema drift")
    require(isinstance(protocol.get("run_id"), str) and protocol["run_id"], "protocol run id")
    require(protocol.get("rank_count") == 8, "protocol must freeze eight ranks")
    require(tuple(protocol.get("lanes", ())) == LANES, "protocol lane order drift")
    parity = protocol.get("lane_order_by_rank")
    require(isinstance(parity, Mapping), "protocol parity lane order")
    even = tuple(parity.get("even", ()))
    odd = tuple(parity.get("odd", ()))
    require(even == LANES, "protocol even lane order drift")
    require(odd == tuple(reversed(LANES)), "protocol odd lane order drift")
    sidecar = protocol.get("sidecar")
    require(isinstance(sidecar, Mapping), "protocol sidecar contract")
    require(sidecar.get("dtype") == "float32-little-endian", "protocol sidecar dtype")
    shape = sidecar.get("shape")
    require(
        isinstance(shape, list) and shape and all(type(item) is int and item > 0 for item in shape),
        "protocol sidecar shape",
    )
    count = math.prod(shape)
    require(sidecar.get("nbytes") == count * 4, "protocol sidecar byte geometry")
    require(tuple(protocol.get("receipt_order", ())) == RECEIPT_ORDER, "protocol receipt order")
    failure = protocol.get("historical_first_failure")
    require(isinstance(failure, Mapping), "protocol historical first failure")
    require(failure.get("model_step_index") == 0, "historical failure step drift")
    require(failure.get("receipt_id") == RECEIPT_ORDER[2], "historical failure receipt drift")
    require(failure.get("predicate_id") == GATE_COMPLETED_REBOUND, "historical failure gate drift")
    expected = protocol.get("expected")
    require(isinstance(expected, Mapping), "protocol expected geometry")
    require(expected.get("resident_count") == 2, "protocol resident count")
    require(expected.get("linear_state_count") == 60, "protocol GDN state count")
    require(
        protocol.get("lane_audit_mode")
        == {
            "historical_pre_fix": "unified_storage_and_binding",
            "repaired_borrowed": "unified_storage_and_binding",
            "materialized_control": "policy_aware_storage_only",
        },
        "protocol lane audit modes",
    )
    require(
        protocol.get("coordinate_classes")
        == {
            "archived_coordinate_ranks": [0, 1, 2],
            "additional_frozen_input_ranks": [3, 4, 5, 6, 7],
            "statistical_independence_claimed": False,
        },
        "protocol coordinate classes",
    )
    require(
        protocol.get("comparison_matrix")
        == {
            "pair_mappings": [
                "historical_pre_fix_vs_materialized_control",
                "historical_pre_fix_vs_repaired_borrowed",
                "repaired_borrowed_vs_materialized_control",
            ],
            "output_only": ["greedy_token_exact", "full_fp32_logits_exact"],
            "state_differential": [
                "request0_terminal_gdn_content_exact",
                "logical_kv_content_exact",
            ],
            "state_invariant": ["persistent_base_content_only_invariant"],
            "forkaudit": [
                "lane_local_storage_intervals",
                "owner_relations",
                "setup_to_transition_binding",
            ],
            "normalized_storage_ids_comparable_across_lanes": False,
        },
        "protocol comparison matrix",
    )
    source = protocol.get("source_bindings")
    require(isinstance(source, Mapping) and source, "protocol source bindings")
    _validate_hash_tree(source, "protocol.source_bindings")
    require(
        source.get("upstream_r29_execution_input_raw_sha256") == UPSTREAM_R29_RAW_SHA256,
        "protocol historical R29 input binding",
    )
    require(
        protocol.get("resource_amendment_binding") == "external_preexecution",
        "protocol must bind an external preexecution resource amendment",
    )
    require("amendment_raw_sha256" not in protocol, "pre-resource protocol embeds amendment hash")
    require(
        protocol.get("content_digest_formula") == "sha256_json(ordered_content_digests)",
        "protocol content digest formula",
    )
    return dict(protocol)


def validate_amendment(
    amendment: Any,
    *,
    protocol_raw_sha256: str,
    amendment_raw_sha256: str,
    expected_run_id: str,
) -> dict[str, Any]:
    require(isinstance(amendment, Mapping), "resource amendment object")
    exact_keys(
        amendment,
        {
            "schema_version",
            "status",
            "created_at_utc",
            "run_id",
            "preregistration_raw_sha256",
            "execution_input_raw_sha256",
            "source_ledger_raw_sha256",
            "execution_package_sha256",
            "science_design_changed",
            "candidate_output_seen_when_frozen",
            "job_id",
            "trial_id",
            "pod",
            "gpu_assignments",
        },
        "resource amendment",
    )
    require(amendment.get("schema_version") == AMENDMENT_SCHEMA, "resource amendment schema")
    require(
        isinstance(amendment.get("created_at_utc"), str) and amendment["created_at_utc"],
        "resource amendment creation time",
    )
    require(
        amendment.get("preregistration_raw_sha256") == protocol_raw_sha256,
        "resource amendment does not bind preregistration",
    )
    require(
        amendment.get("status") == "frozen_after_resource_creation_before_candidate_outputs",
        "resource amendment freeze status",
    )
    require(amendment.get("run_id") == expected_run_id, "resource amendment run id")
    require(amendment.get("candidate_output_seen_when_frozen") is False, "resource amendment outcome blindness")
    require(amendment.get("science_design_changed") is False, "resource amendment changed science design")
    require(
        isinstance(amendment.get("job_id"), (int, str))
        and not isinstance(amendment.get("job_id"), bool)
        and str(amendment["job_id"]),
        "resource amendment job id",
    )
    require(
        isinstance(amendment.get("trial_id"), (int, str))
        and not isinstance(amendment.get("trial_id"), bool)
        and str(amendment["trial_id"]),
        "resource amendment trial id",
    )
    require(isinstance(amendment.get("pod"), str) and amendment["pod"], "resource amendment pod")
    assignments = amendment.get("gpu_assignments")
    require(
        isinstance(assignments, Mapping) and set(assignments) == {str(index) for index in range(8)},
        "resource amendment GPU assignments",
    )
    observed_uuids: list[str] = []
    for rank, row in assignments.items():
        require(isinstance(row, Mapping), f"resource amendment GPU rank {rank}")
        exact_keys(row, {"physical_index", "uuid"}, f"resource amendment GPU rank {rank}")
        require(
            exact_int(row.get("physical_index"), f"resource amendment GPU rank {rank} index")
            == int(rank),
            f"resource amendment GPU rank {rank} physical mapping",
        )
        require(
            isinstance(row.get("uuid"), str) and row["uuid"].startswith("GPU-"),
            f"resource amendment GPU rank {rank} UUID",
        )
        observed_uuids.append(row["uuid"])
    require(len(set(observed_uuids)) == 8, "resource amendment GPU UUID uniqueness")
    for field in (
        "execution_input_raw_sha256",
        "source_ledger_raw_sha256",
        "execution_package_sha256",
    ):
        check_sha(amendment.get(field), f"resource amendment {field}")
    check_sha(amendment_raw_sha256, "resource amendment")
    _validate_hash_tree(amendment, "resource_amendment")
    return dict(amendment)


def _interval(row: Mapping[str, Any], label: str) -> tuple[int, int]:
    shape = row.get("shape")
    stride = row.get("stride")
    require(
        isinstance(shape, list)
        and isinstance(stride, list)
        and len(shape) == len(stride)
        and shape,
        f"{label} shape/stride",
    )
    require(all(type(item) is int and item > 0 for item in shape), f"{label} shape")
    require(all(type(item) is int for item in stride), f"{label} stride")
    dtype = row.get("dtype")
    require(dtype in DTYPE_NBYTES, f"{label} dtype")
    start = exact_int(row.get("byte_start"), f"{label}.byte_start")
    minimum = maximum = 0
    for size, step in zip(shape, stride):
        displacement = (size - 1) * step
        minimum += min(displacement, 0)
        maximum += max(displacement, 0)
    require(minimum == 0, f"{label} negative-stride view unsupported")
    end = start + (maximum + 1) * DTYPE_NBYTES[dtype]
    require(row.get("byte_end_exclusive") == end, f"{label} byte interval")
    storage_nbytes = exact_int(row.get("storage_nbytes"), f"{label}.storage_nbytes", minimum=1)
    require(0 <= start < end <= storage_nbytes, f"{label} interval outside storage")
    tensor_nbytes = math.prod(shape) * DTYPE_NBYTES[dtype]
    require(row.get("tensor_nbytes") == tensor_nbytes, f"{label} tensor nbytes")
    return start, end


def validate_snapshot(snapshot: Any, label: str) -> dict[tuple[str, str], dict[str, Any]]:
    require(isinstance(snapshot, Mapping), f"{label} snapshot")
    rows = snapshot.get("rows")
    require(snapshot.get("row_count") == 180, f"{label} row count")
    require(isinstance(rows, list) and len(rows) == 180, f"{label} rows")
    require(snapshot.get("rows_sha256") == sha256_json(rows), f"{label} row digest")
    require(snapshot.get("absolute_pointers_persisted") is False, f"{label} pointer flag")
    require(snapshot.get("python_object_ids_persisted") is False, f"{label} object-id flag")
    _reject_absolute_identity_fields(rows, f"{label}.rows")
    output: dict[tuple[str, str], dict[str, Any]] = {}
    owner_counts = {owner: 0 for owner in OWNERS}
    for index, raw in enumerate(rows):
        require(isinstance(raw, Mapping), f"{label} row {index}")
        row = dict(raw)
        owner = row.get("owner")
        family = row.get("state_family")
        require(owner in OWNERS and family in FAMILIES, f"{label} owner/family")
        layer = exact_int(row.get("layer_index"), f"{label}.layer")
        require(row.get("state_index") == 0, f"{label} state index")
        coordinate = f"layer:{layer}/{family}/state:0"
        require(row.get("coordinate") == coordinate, f"{label} coordinate")
        require(
            isinstance(row.get("storage_id"), str) and STORAGE_RE.fullmatch(row["storage_id"]),
            f"{label} normalized storage id",
        )
        require(
            isinstance(row.get("tensor_id"), str) and TENSOR_RE.fullmatch(row["tensor_id"]),
            f"{label} normalized tensor id",
        )
        require(row.get("contains_absolute_pointer") is False, f"{label} row pointer flag")
        require(row.get("contains_python_object_id") is False, f"{label} row object flag")
        check_sha(row.get("content_sha256"), f"{label} content")
        _interval(row, f"{label}[{owner},{coordinate}]")
        key = (owner, coordinate)
        require(key not in output, f"{label} duplicate coordinate")
        output[key] = row
        owner_counts[owner] += 1
    require(owner_counts == {owner: 60 for owner in OWNERS}, f"{label} owner cardinality")
    coordinates = {coordinate for owner, coordinate in output if owner == "persistent"}
    require(len(coordinates) == 60, f"{label} coordinate cardinality")
    require(
        all({(owner, coordinate) for owner in OWNERS}.issubset(output) for coordinate in coordinates),
        f"{label} coordinate coverage",
    )
    layers = {row["layer_index"] for row in output.values()}
    require(len(layers) == 30, f"{label} linear layer count")
    return output


def overlaps(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return (
        left["storage_id"] == right["storage_id"]
        and max(left["byte_start"], right["byte_start"])
        < min(left["byte_end_exclusive"], right["byte_end_exclusive"])
    )


def exact_alias(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
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
    return all(left.get(field) == right.get(field) for field in fields)


def stable_row(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return exact_alias(left, right)


def relation_summary(
    lane: str,
    setup: Mapping[tuple[str, str], Mapping[str, Any]],
    post: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    """Recompute all relevant relations using IDs only within this lane."""

    coordinates = sorted(coordinate for owner, coordinate in setup if owner == "persistent")
    counts = {
        "setup_request0_base_alias": 0,
        "setup_request1_base_alias": 0,
        "setup_request0_peer_alias": 0,
        "setup_request0_base_disjoint": 0,
        "setup_request1_base_disjoint": 0,
        "setup_request0_peer_disjoint": 0,
        "post_request0_base_alias": 0,
        "post_request0_base_disjoint": 0,
        "post_request0_peer_alias": 0,
        "post_request0_peer_disjoint": 0,
        "post_base_peer_alias": 0,
        "post_base_peer_disjoint": 0,
        "request0_binding_changed": 0,
        "persistent_binding_changed": 0,
        "persistent_content_changed": 0,
        "peer_binding_changed": 0,
        "peer_content_changed": 0,
    }
    by_family = {
        family: {
            "post_request0_base_alias": 0,
            "post_request0_base_disjoint": 0,
            "request0_binding_changed": 0,
            "persistent_content_changed": 0,
        }
        for family in FAMILIES
    }
    for coordinate in coordinates:
        s_base = setup[("persistent", coordinate)]
        s_r0 = setup[("request_0", coordinate)]
        s_r1 = setup[("request_1", coordinate)]
        p_base = post[("persistent", coordinate)]
        p_r0 = post[("request_0", coordinate)]
        p_r1 = post[("request_1", coordinate)]
        family = s_base["state_family"]
        for name, condition in (
            ("setup_request0_base_alias", exact_alias(s_r0, s_base)),
            ("setup_request1_base_alias", exact_alias(s_r1, s_base)),
            ("setup_request0_peer_alias", exact_alias(s_r0, s_r1)),
            ("setup_request0_base_disjoint", not overlaps(s_r0, s_base)),
            ("setup_request1_base_disjoint", not overlaps(s_r1, s_base)),
            ("setup_request0_peer_disjoint", not overlaps(s_r0, s_r1)),
            ("post_request0_base_alias", exact_alias(p_r0, p_base)),
            ("post_request0_base_disjoint", not overlaps(p_r0, p_base)),
            ("post_request0_peer_alias", exact_alias(p_r0, p_r1)),
            ("post_request0_peer_disjoint", not overlaps(p_r0, p_r1)),
            ("post_base_peer_alias", exact_alias(p_base, p_r1)),
            ("post_base_peer_disjoint", not overlaps(p_base, p_r1)),
        ):
            counts[name] += int(condition)
            if name in by_family[family]:
                by_family[family][name] += int(condition)
        r0_binding_changed = not (
            s_r0["tensor_id"] == p_r0["tensor_id"]
            and s_r0["storage_id"] == p_r0["storage_id"]
            and s_r0["byte_start"] == p_r0["byte_start"]
            and s_r0["byte_end_exclusive"] == p_r0["byte_end_exclusive"]
        )
        counts["request0_binding_changed"] += int(r0_binding_changed)
        by_family[family]["request0_binding_changed"] += int(r0_binding_changed)
        for owner, before, after in (
            ("persistent", s_base, p_base),
            ("peer", s_r1, p_r1),
        ):
            binding_changed = not (
                before["tensor_id"] == after["tensor_id"]
                and before["storage_id"] == after["storage_id"]
                and before["byte_start"] == after["byte_start"]
                and before["byte_end_exclusive"] == after["byte_end_exclusive"]
            )
            content_changed = before["content_sha256"] != after["content_sha256"]
            counts[f"{owner}_binding_changed"] += int(binding_changed)
            counts[f"{owner}_content_changed"] += int(content_changed)
            if owner == "persistent":
                by_family[family]["persistent_content_changed"] += int(content_changed)
    shared_setup = (
        counts["setup_request0_base_alias"]
        == counts["setup_request1_base_alias"]
        == counts["setup_request0_peer_alias"]
        == 60
    )
    materialized_setup = (
        counts["setup_request0_base_disjoint"]
        == counts["setup_request1_base_disjoint"]
        == counts["setup_request0_peer_disjoint"]
        == 60
    )
    require(shared_setup if lane in SHARED_LANES else materialized_setup, f"{lane} setup policy")
    return {
        "counts": counts,
        "by_family": by_family,
        "shared_setup_contract": shared_setup,
        "materialized_setup_contract": materialized_setup,
        "persistent_base_content_only_invariant": counts["persistent_content_changed"] == 0,
        "persistent_base_binding_invariant": counts["persistent_binding_changed"] == 0,
        "unadvanced_peer_content_invariant": counts["peer_content_changed"] == 0,
        "unadvanced_peer_binding_invariant": counts["peer_binding_changed"] == 0,
    }


def replay_sidecar(
    raw_root: Path,
    lane: str,
    receipt: Any,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    require(isinstance(receipt, Mapping), f"{lane} sidecar receipt")
    expected_name = f"{lane}-full-fp32-logits.bin"
    require(receipt.get("path") == expected_name, f"{lane} sidecar name")
    path = safe_child(raw_root, receipt["path"], f"{lane} sidecar")
    require(path.is_file(), f"{lane} sidecar missing")
    payload = path.read_bytes()
    require(receipt.get("dtype") == contract["dtype"], f"{lane} sidecar dtype")
    require(receipt.get("shape") == contract["shape"], f"{lane} sidecar shape")
    require(receipt.get("nbytes") == contract["nbytes"] == len(payload), f"{lane} sidecar nbytes")
    digest = sha256_bytes(payload)
    require(digest == check_sha(receipt.get("sha256"), f"{lane} sidecar"), f"{lane} sidecar hash")
    require(array("f").itemsize == 4, "host C float is not 32-bit")
    values = array("f")
    values.frombytes(payload)
    if sys.byteorder != "little":
        values.byteswap()
    require(len(values) == math.prod(contract["shape"]), f"{lane} decoded sidecar shape")
    finite = all(math.isfinite(value) for value in values)
    require(finite, f"{lane} sidecar contains non-finite FP32")
    require(receipt.get("finite") is True, f"{lane} producer finite receipt")
    argmax = max(range(len(values)), key=values.__getitem__)
    return {
        "path": receipt["path"],
        "sha256": digest,
        "dtype": contract["dtype"],
        "shape": list(contract["shape"]),
        "nbytes": len(payload),
        "finite": True,
        "argmax_token_id": argmax,
        "payload": payload,
        "values": values,
    }


def _validate_no_mutation(value: Any, lane: str) -> None:
    expected = {
        "r29_heldout_fault_module_loaded": False,
        "generic_mutant_definition_module_passively_loaded": True,
        "mutation_requested": False,
        "mutation_applied": False,
        "mutation_event_count": 0,
    }
    require(value == expected, f"{lane} no-mutation receipt")


def _validate_repair_receipt(value: Any, lane: str) -> None:
    if lane != "repaired_borrowed":
        require(value is None, f"{lane} unexpectedly used the repair helper")
        return
    require(isinstance(value, Mapping), "repaired transition receipt")
    require(
        value.get("schema_version") == "qcomem-single-token-gdn-conv-privatization-v1",
        "repaired transition schema",
    )
    require(value.get("request_index") == 0 and value.get("resident_count") == 2, "repaired transition ownership")
    require(value.get("conv_tensor_count") == 30 and value.get("cloned_tensor_count") == 30, "repaired clone count")
    require(value.get("fault_id_specialization") is False, "repair is fault specialized")
    require(value.get("ownership_only_change") is True, "repair is not ownership only")
    rows = value.get("rows")
    require(isinstance(rows, list) and len(rows) == 30, "repaired transition rows")
    require(value.get("rows_sha256") == sha256_json(rows), "repaired transition row digest")
    for row in rows:
        require(isinstance(row, Mapping), "repaired transition row")
        require(row.get("action") == "cloned_borrowed_state", "repaired transition action")
        require(row.get("base_disjoint") is True and row.get("all_peers_disjoint") is True, "repaired transition disjointness")
        check_sha(row.get("content_sha256"), "repaired transition content")


def _validate_cleanup(lane: Mapping[str, Any], label: str) -> None:
    require(lane.get("fresh_case") is True, f"{label} not fresh")
    require(lane.get("state_reused_from_prior_lane") is False, f"{label} state reused")
    baseline = lane.get("allocator_baseline")
    require(isinstance(baseline, Mapping), f"{label} allocator baseline")
    require(lane.get("allocator_before") == baseline, f"{label} allocator precondition")
    cleanup = lane.get("cleanup")
    require(isinstance(cleanup, Mapping), f"{label} cleanup")
    for field in (
        "fresh_case_disposed",
        "registered_backend_restored",
        "strong_references_released",
        "gc_collect_completed",
        "cuda_empty_cache_completed",
        "cuda_synchronize_completed",
        "allocator_baseline_exact",
    ):
        require(cleanup.get(field) is True, f"{label} cleanup {field}")
    require(cleanup.get("cleanup_error") is None, f"{label} cleanup error")
    require(cleanup.get("allocator_after") == baseline, f"{label} allocator cleanup drift")


def _validate_receipts(audit: Any, label: str) -> tuple[list[str], Mapping[str, Any] | None]:
    require(isinstance(audit, Mapping), f"{label} audit")
    completed = audit.get("completed_receipts")
    require(isinstance(completed, list), f"{label} completed receipts")
    ids: list[str] = []
    for index, row in enumerate(completed):
        require(isinstance(row, Mapping), f"{label} receipt {index}")
        require(row.get("status") == "passed", f"{label} receipt status")
        payload = row.get("payload")
        require(isinstance(payload, Mapping), f"{label} receipt payload")
        require(row.get("payload_sha256") == sha256_json(payload), f"{label} receipt payload hash")
        ids.append(str(row.get("receipt_id")))
    require(tuple(ids) == RECEIPT_ORDER[: len(ids)], f"{label} receipt prefix/order")
    rejection = audit.get("first_authenticated_rejection")
    if rejection is None:
        require(len(ids) == len(RECEIPT_ORDER), f"{label} incomplete clean receipt battery")
        return ids, None
    require(isinstance(rejection, Mapping), f"{label} rejection")
    require(rejection.get("authenticated") is True, f"{label} rejection unauthenticated")
    require(len(ids) < len(RECEIPT_ORDER), f"{label} rejection after complete battery")
    require(rejection.get("receipt_id") == RECEIPT_ORDER[len(ids)], f"{label} rejection position")
    predicate = rejection.get("predicate_id")
    require(isinstance(predicate, str) and predicate, f"{label} rejection predicate")
    exception = rejection.get("exception")
    require(isinstance(exception, Mapping), f"{label} rejection exception")
    require(exception.get("gate_id") == predicate, f"{label} rejection gate binding")
    authority = (exception.get("module"), exception.get("type"))
    allowed_authorities = {
        ("qcomem_forkaudit_storage_witness", "GDNStorageWitnessError"),
        ("qcomem_vllm_paged_multifork_resident", "RuntimeInvariantError"),
        ("r35_run_historical_alias_regression", "ReceiptPredicateRejection"),
        ("__main__", "ReceiptPredicateRejection"),
    }
    require(authority in allowed_authorities, f"{label} rejection authority")
    stack = exception.get("stack")
    require(
        isinstance(stack, list)
        and any(
            isinstance(frame, Mapping)
            and frame.get("filename")
            in {
                "qcomem_forkaudit_storage_witness.py",
                "qcomem_vllm_paged_multifork_resident.py",
                "r35_run_historical_alias_regression.py",
            }
            for frame in stack
        ),
        f"{label} rejection source stack",
    )
    return ids, rejection


def _content_receipt(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label} terminal content")
    _reject_absolute_identity_fields(value, label)
    _validate_hash_tree(value, label)
    require(value.get("storage_or_pointer_fields_persisted") is False, f"{label} storage field flag")
    request_gdn = value.get("request_gdn")
    require(isinstance(request_gdn, list) and len(request_gdn) == 2, f"{label} request GDN")
    require([row.get("request_index") for row in request_gdn] == [0, 1], f"{label} request GDN order")
    for row in request_gdn:
        require(row.get("tensor_count") == 60, f"{label} request GDN tensor count")
        check_sha(row.get("sha256"), f"{label} request GDN")
        ordered = row.get("ordered_content_digests")
        require(isinstance(ordered, list) and len(ordered) == 60, f"{label} request GDN digest list")
        for digest in ordered:
            check_sha(digest, f"{label} request GDN member")
        require(row["sha256"] == sha256_json(ordered), f"{label} request GDN aggregate hash")
    logical_kv = value.get("logical_kv")
    require(isinstance(logical_kv, list) and len(logical_kv) == 2, f"{label} logical KV")
    require([row.get("request_index") for row in logical_kv] == [0, 1], f"{label} logical KV order")
    for row in logical_kv:
        layers = row.get("layer_sha256")
        require(isinstance(layers, Mapping) and len(layers) == 10, f"{label} KV layers")
        for digest in layers.values():
            check_sha(digest, f"{label} KV layer")
    require(value.get("logical_kv_sha256") == sha256_json(logical_kv), f"{label} logical KV aggregate hash")
    persistent = value.get("persistent_gdn")
    require(isinstance(persistent, Mapping) and set(persistent) == {"setup", "post"}, f"{label} persistent GDN")
    for phase in ("setup", "post"):
        row = persistent[phase]
        require(isinstance(row, Mapping) and row.get("tensor_count") == 60, f"{label} persistent {phase}")
        check_sha(row.get("sha256"), f"{label} persistent {phase}")
        ordered = row.get("ordered_content_digests")
        require(isinstance(ordered, list) and len(ordered) == 60, f"{label} persistent {phase} digest list")
        for digest in ordered:
            check_sha(digest, f"{label} persistent {phase} member")
        require(row["sha256"] == sha256_json(ordered), f"{label} persistent {phase} aggregate hash")
    return dict(value)


def _snapshot_owner_content(
    rows: Mapping[tuple[str, str], Mapping[str, Any]], owner: str
) -> dict[str, Any]:
    ordered = [
        row["content_sha256"]
        for (row_owner, _), row in rows.items()
        if row_owner == owner
    ]
    require(len(ordered) == 60, f"{owner} snapshot content count")
    return {
        "sha256": sha256_json(ordered),
        "tensor_count": 60,
        "ordered_content_digests": ordered,
    }


def _request0_gdn_digest(content: Mapping[str, Any]) -> str:
    return str(content["request_gdn"][0]["sha256"])


def _logical_kv_content(content: Mapping[str, Any]) -> Any:
    return content["logical_kv"]


def _persistent_invariant(content: Mapping[str, Any]) -> bool:
    return content["persistent_gdn"]["setup"]["sha256"] == content["persistent_gdn"]["post"]["sha256"]


def _observed_historical_gate(relations: Mapping[str, Any]) -> str | None:
    by_family = relations["by_family"]
    if by_family["conv_states"]["request0_binding_changed"] < 30:
        return GATE_COMPLETED_REBOUND
    if relations["counts"]["persistent_content_changed"] > 0:
        return GATE_PERSISTENT_IMMUTABLE
    return None


def replay_rank(
    *,
    result: Any,
    raw_root: Path,
    protocol: Mapping[str, Any],
    protocol_raw_sha256: str,
    amendment_raw_sha256: str | None,
    amendment: Mapping[str, Any] | None = None,
    result_raw_sha256: str | None = None,
) -> dict[str, Any]:
    protocol = validate_protocol(protocol)
    require(isinstance(result, Mapping), "rank result object")
    require(result.get("schema_version") == RANK_SCHEMA, "rank result schema")
    require(result.get("run_id") == protocol["run_id"], "rank run id")
    rank = exact_int(result.get("rank"), "rank")
    require(rank < 8, "rank range")
    require(result.get("status") in {"completed", "rank_completed"}, "rank operational status")
    require(result.get("operational_invalid") is None, "rank recorded operational invalidity")
    require(result.get("protocol") == protocol, "rank embedded protocol binding")
    require(result.get("preregistration_raw_sha256") == protocol_raw_sha256, "rank preregistration binding")
    require(amendment_raw_sha256 is not None, "rank amendment argument missing")
    check_sha(amendment_raw_sha256, "rank amendment argument")
    require(result.get("amendment_raw_sha256") == amendment_raw_sha256, "rank amendment binding")
    require(isinstance(amendment, Mapping), "validated resource amendment missing")
    require(
        result.get("execution_input_raw_sha256") == amendment["execution_input_raw_sha256"],
        "rank execution-input/amendment binding",
    )
    require(
        protocol["source_bindings"].get("source_ledger_raw_sha256")
        == amendment["source_ledger_raw_sha256"],
        "protocol/amendment source-ledger binding",
    )
    resource = result.get("resource")
    require(isinstance(resource, Mapping), "rank resource receipt")
    assignment = amendment["gpu_assignments"][str(rank)]
    require(
        resource
        == {
            "job_id": amendment["job_id"],
            "trial_id": amendment["trial_id"],
            "pod": amendment["pod"],
            "gpu_assignment": assignment,
            "preregistration_raw_sha256": protocol_raw_sha256,
            "execution_package_sha256": amendment["execution_package_sha256"],
        },
        "rank resource/amendment binding",
    )
    require(result.get("source_bindings") == protocol["source_bindings"], "rank source binding")
    _validate_hash_tree(result.get("source_bindings"), "result.source_bindings")
    upstream = result.get("upstream_r29_execution_input")
    require(
        upstream
        == {
            "raw_sha256": UPSTREAM_R29_RAW_SHA256,
            "run_id": UPSTREAM_R29_RUN_ID,
            "copied_fields_exact": True,
        },
        "rank historical R29 input receipt",
    )
    fault_isolation = result.get("fault_isolation")
    require(
        fault_isolation
        == {
            "r29_heldout_fault_suite_import_blocked": True,
            "r29_heldout_fault_suite_in_sys_modules": False,
            "generic_mutant_definition_module_passively_loaded": True,
            "mutation_requested": False,
            "mutation_applied": False,
        },
        "rank fault isolation",
    )
    lanes_value = result.get("lanes")
    require(isinstance(lanes_value, Mapping) and set(lanes_value) == set(LANES), "rank lane set")
    observed_order = tuple(result.get("lane_order", ()))
    expected_order = tuple(protocol["lane_order_by_rank"]["even" if rank % 2 == 0 else "odd"])
    require(observed_order == expected_order, "rank parity lane order")
    input_receipt = result.get("input_receipt")
    require(isinstance(input_receipt, Mapping), "rank input receipt")
    require(input_receipt.get("rank") == rank, "rank input-receipt binding")
    require(
        input_receipt.get("coordinate_class")
        == ("archived" if rank < 3 else "additional_frozen"),
        "rank coordinate class",
    )
    imported_rr2 = input_receipt.get("imported_rr2_code")
    require(isinstance(imported_rr2, Mapping), "rank imported RR2 receipt")
    check_sha(imported_rr2.get("raw_sha256"), "rank imported RR2 ledger")
    file_sha256 = imported_rr2.get("file_sha256")
    require(isinstance(file_sha256, Mapping) and file_sha256, "rank imported RR2 file map")
    require(imported_rr2.get("file_count") == len(file_sha256), "rank imported RR2 file count")
    check_sha(imported_rr2.get("rows_sha256"), "rank imported RR2 rows")
    require(all(isinstance(path, str) and path and not Path(path).is_absolute() for path in file_sha256), "rank imported RR2 paths")
    for digest in file_sha256.values():
        check_sha(digest, "rank imported RR2 file")
    _validate_module_closure(
        input_receipt.get("loaded_science_module_closure"),
        "rank loaded science module closure",
    )
    process_ids: set[str] = set()
    case_nonces: set[str] = set()
    lane_replays: dict[str, dict[str, Any]] = {}
    sidecar_paths: set[str] = set()
    for lane_name in LANES:
        lane = lanes_value[lane_name]
        require(isinstance(lane, Mapping) and lane.get("lane") == lane_name, f"{lane_name} binding")
        require(lane.get("rank") == rank, f"{lane_name} rank binding")
        require(lane.get("operational_invalid") is None, f"{lane_name} operational invalidity")
        process_id = check_sha(lane.get("process_instance_id"), f"{lane_name} process instance")
        process_ids.add(process_id)
        case_nonce = lane.get("case_nonce")
        require(
            isinstance(case_nonce, str)
            and re.fullmatch(r"[0-9a-f]{32}|[0-9a-f]{64}", case_nonce) is not None
            and case_nonce not in case_nonces,
            f"{lane_name} fresh case nonce",
        )
        case_nonces.add(case_nonce)
        _validate_no_mutation(lane.get("mutation_receipt"), lane_name)
        _validate_repair_receipt(lane.get("repair_transition_receipt"), lane_name)
        _validate_cleanup(lane, lane_name)
        step = lane.get("model_step")
        require(isinstance(step, Mapping), f"{lane_name} model step")
        require(step.get("step_index") == 0, f"{lane_name} model step index")
        require(step.get("semantic_horizon_reached") is True, f"{lane_name} semantic horizon")
        sidecar = replay_sidecar(raw_root, lane_name, lane.get("full_logits"), protocol["sidecar"])
        require(sidecar["path"] not in sidecar_paths, "duplicate sidecar path")
        sidecar_paths.add(sidecar["path"])
        require(step.get("full_logit_sha256") == sidecar["sha256"], f"{lane_name} model/sidecar hash")
        require(step.get("greedy_token_id") == sidecar["argmax_token_id"], f"{lane_name} greedy token")
        setup = validate_snapshot(lane.get("setup_snapshot"), f"{lane_name}.setup")
        post = validate_snapshot(lane.get("post_snapshot"), f"{lane_name}.post")
        relations = relation_summary(lane_name, setup, post)
        content = _content_receipt(lane.get("terminal_content"), lane_name)
        require(
            content["request_gdn"][0]
            == {"request_index": 0, **_snapshot_owner_content(post, "request_0")},
            f"{lane_name} request0 terminal/snapshot binding",
        )
        require(
            content["request_gdn"][1]
            == {"request_index": 1, **_snapshot_owner_content(post, "request_1")},
            f"{lane_name} request1 terminal/snapshot binding",
        )
        require(content["persistent_gdn"]["setup"] == _snapshot_owner_content(setup, "persistent"), f"{lane_name} persistent setup terminal/snapshot binding")
        require(content["persistent_gdn"]["post"] == _snapshot_owner_content(post, "persistent"), f"{lane_name} persistent post terminal/snapshot binding")
        require(
            _persistent_invariant(content) == relations["persistent_base_content_only_invariant"],
            f"{lane_name} persistent content receipt/snapshot disagreement",
        )
        receipt_ids, rejection = _validate_receipts(lane.get("audit"), lane_name)
        audit = lane["audit"]
        audit_mode = protocol["lane_audit_mode"][lane_name]
        require(audit.get("audit_mode") == audit_mode, f"{lane_name} audit mode")
        storage_passed = RECEIPT_ORDER[2] in receipt_ids
        require(audit.get("storage_witness_passed") is storage_passed, f"{lane_name} storage audit flag")
        expected_unified = storage_passed if audit_mode == "unified_storage_and_binding" else None
        require(audit.get("unified_witness_passed") is expected_unified, f"{lane_name} unified audit flag")
        expected_historical_flag = (
            lane_name == "historical_pre_fix"
            and rejection is not None
            and rejection.get("receipt_id") == RECEIPT_ORDER[2]
            and rejection.get("predicate_id") == GATE_COMPLETED_REBOUND
        )
        require(
            audit.get("expected_historical_rejection_observed") is expected_historical_flag,
            f"{lane_name} historical rejection flag",
        )
        expected_status = (
            "authenticated_forkaudit_rejection_after_model_step"
            if rejection is not None
            else "completed_clean"
        )
        require(lane.get("status") == expected_status, f"{lane_name} status/rejection binding")
        observed_gate = _observed_historical_gate(relations) if lane_name == "historical_pre_fix" else None
        if lane_name == "historical_pre_fix":
            if rejection is None:
                require(observed_gate is None, "historical snapshots fail but producer reports no rejection")
            else:
                require(rejection["receipt_id"] == RECEIPT_ORDER[2], "historical rejection receipt")
                require(rejection["predicate_id"] == observed_gate, "historical gate not reproduced from rows")
        lane_replays[lane_name] = {
            "status": lane.get("status"),
            "case_nonce": case_nonce,
            "model_step_index": 0,
            "greedy_token_id": sidecar["argmax_token_id"],
            "full_logits_sha256": sidecar["sha256"],
            "sidecar": {key: sidecar[key] for key in ("path", "sha256", "dtype", "shape", "nbytes", "finite", "argmax_token_id")},
            "relations": relations,
            "terminal_content": content,
            "completed_receipt_ids": receipt_ids,
            "reported_first_gate": None if rejection is None else rejection["predicate_id"],
            "row_recomputed_first_gate": observed_gate,
            "full_logits_payload": sidecar["payload"],
        }
    require(len(process_ids) == 1, "three fresh cases did not share the rank process instance")
    require(process_ids == {result.get("process_instance_id")}, "rank/lane process binding")
    require(len(case_nonces) == 3, "rank fresh case nonce cardinality")
    require(sidecar_paths == {f"{lane}-full-fp32-logits.bin" for lane in LANES}, "rank sidecar coverage")
    historical = lane_replays["historical_pre_fix"]
    repaired = lane_replays["repaired_borrowed"]
    control = lane_replays["materialized_control"]
    expected_gate = protocol["historical_first_failure"]["predicate_id"]
    historical_pair = {
        "greedy_token_exact": historical["greedy_token_id"] == control["greedy_token_id"],
        "full_fp32_logits_exact": historical["full_logits_payload"] == control["full_logits_payload"],
        "request0_terminal_gdn_content_exact": _request0_gdn_digest(historical["terminal_content"]) == _request0_gdn_digest(control["terminal_content"]),
        "logical_kv_content_exact": _logical_kv_content(historical["terminal_content"]) == _logical_kv_content(control["terminal_content"]),
        "persistent_base_content_only_invariant": _persistent_invariant(historical["terminal_content"]),
    }
    repaired_pair = {
        "greedy_token_exact": repaired["greedy_token_id"] == control["greedy_token_id"],
        "full_fp32_logits_exact": repaired["full_logits_payload"] == control["full_logits_payload"],
        "request0_terminal_gdn_content_exact": _request0_gdn_digest(repaired["terminal_content"]) == _request0_gdn_digest(control["terminal_content"]),
        "logical_kv_content_exact": _logical_kv_content(repaired["terminal_content"]) == _logical_kv_content(control["terminal_content"]),
        "persistent_base_content_only_invariant": _persistent_invariant(repaired["terminal_content"]),
    }
    historical_repaired_pair = {
        "greedy_token_exact": historical["greedy_token_id"] == repaired["greedy_token_id"],
        "full_fp32_logits_exact": historical["full_logits_payload"] == repaired["full_logits_payload"],
        "request0_terminal_gdn_content_exact": _request0_gdn_digest(historical["terminal_content"]) == _request0_gdn_digest(repaired["terminal_content"]),
        "logical_kv_content_exact": _logical_kv_content(historical["terminal_content"]) == _logical_kv_content(repaired["terminal_content"]),
        "persistent_base_content_only_invariant": _persistent_invariant(historical["terminal_content"]),
    }
    recomputed_comparisons = {
        "historical_pre_fix_vs_materialized_control": historical_pair,
        "repaired_borrowed_vs_materialized_control": repaired_pair,
        "historical_pre_fix_vs_repaired_borrowed": historical_repaired_pair,
    }
    require(result.get("comparisons") == recomputed_comparisons, "producer comparison receipt drift")
    repair_relations = repaired["relations"]
    control_relations = control["relations"]
    repair_storage_clean = (
        repair_relations["counts"]["post_request0_base_disjoint"] == 60
        and repair_relations["counts"]["post_request0_peer_disjoint"] == 60
        and repair_relations["counts"]["post_base_peer_alias"] == 60
        and repair_relations["persistent_base_binding_invariant"]
        and repair_relations["persistent_base_content_only_invariant"]
    )
    materialized_storage_clean = (
        control_relations["counts"]["post_request0_base_disjoint"] == 60
        and control_relations["counts"]["post_request0_peer_disjoint"] == 60
        and control_relations["counts"]["post_base_peer_disjoint"] == 60
        and control_relations["persistent_base_binding_invariant"]
        and control_relations["persistent_base_content_only_invariant"]
    )
    for lane in lane_replays.values():
        lane.pop("full_logits_payload")
    return {
        "schema_version": REPLAY_SCHEMA,
        "run_id": protocol["run_id"],
        "rank": rank,
        "status": "operationally_valid_detached_replay",
        "operational_valid": True,
        "candidate_modules_imported": False,
        "preregistration_raw_sha256": protocol_raw_sha256,
        "amendment_raw_sha256": amendment_raw_sha256,
        "execution_input_raw_sha256": amendment["execution_input_raw_sha256"],
        "source_ledger_raw_sha256": amendment["source_ledger_raw_sha256"],
        "rank_result_raw_sha256": result_raw_sha256,
        "verified_lane_count": 3,
        "verified_sidecar_count": 3,
        "lanes": lane_replays,
        "fixed_pair_mappings": list(protocol["comparison_matrix"]["pair_mappings"]),
        "conventional_baseline_matrix": recomputed_comparisons,
        "hypothesis_outcomes": {
            "historical_expected_authenticated_first_gate_reproduced": historical["reported_first_gate"] == expected_gate and historical["row_recomputed_first_gate"] == expected_gate,
            "historical_output_only_exact_to_materialized": historical_pair["greedy_token_exact"] and historical_pair["full_fp32_logits_exact"],
            "repaired_storage_contract_clean": repaired["reported_first_gate"] is None and repair_storage_clean,
            "materialized_storage_contract_clean": control["reported_first_gate"] is None and materialized_storage_clean,
            "repaired_semantic_and_terminal_exact_to_materialized": all(
                repaired_pair[key]
                for key in (
                    "greedy_token_exact",
                    "full_fp32_logits_exact",
                    "request0_terminal_gdn_content_exact",
                    "logical_kv_content_exact",
                )
            ),
        },
        "scientific_summary_authorized": True,
        "population_detection_rate_computed": False,
        "claim_boundary": {
            "one_rank_three_fresh_cases": True,
            "normalized_storage_ids_compared_within_lane_only": True,
            "baseline_cells_not_combined_into_a_detector": True,
            "hypothesis_mismatch_is_valid_negative_not_operational_invalid": True,
        },
    }


def _write_new_json(path: Path, value: Any) -> None:
    require(not path.exists(), "replay output already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(canonical_bytes(value) + b"\n")
    temporary.replace(path)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--rank-result", type=Path, required=True)
    value.add_argument("--raw-root", type=Path, required=True)
    value.add_argument("--preregistration", type=Path, required=True)
    value.add_argument("--expected-preregistration-sha256", required=True)
    value.add_argument("--amendment", type=Path, required=True)
    value.add_argument("--expected-amendment-sha256", required=True)
    value.add_argument("--output", type=Path, required=True)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        protocol_sha = sha256_file(args.preregistration)
        require(
            protocol_sha == check_sha(args.expected_preregistration_sha256, "expected preregistration"),
            "preregistration raw hash drift",
        )
        protocol = validate_protocol(load_json(args.preregistration))
        require(args.amendment is not None, "missing external preexecution resource amendment")
        amendment_sha = sha256_file(args.amendment)
        require(
            amendment_sha == check_sha(args.expected_amendment_sha256, "expected amendment"),
            "amendment raw hash drift",
        )
        amendment = validate_amendment(
            load_json(args.amendment),
            protocol_raw_sha256=protocol_sha,
            amendment_raw_sha256=amendment_sha,
            expected_run_id=protocol["run_id"],
        )
        result_sha = sha256_file(args.rank_result)
        receipt = replay_rank(
            result=load_json(args.rank_result),
            raw_root=args.raw_root,
            protocol=protocol,
            protocol_raw_sha256=protocol_sha,
            amendment_raw_sha256=amendment_sha,
            amendment=amendment,
            result_raw_sha256=result_sha,
        )
        _write_new_json(args.output, receipt)
        return 0
    except (OSError, ReplayError) as exc:
        print(f"R35 detached replay invalid: {exc}", file=sys.stderr)
        return 2


__all__ = [
    "AMENDMENT_SCHEMA",
    "ALLOWED_HISTORICAL_GATES",
    "LANES",
    "PROTOCOL_SCHEMA",
    "RANK_SCHEMA",
    "RECEIPT_ORDER",
    "REPLAY_SCHEMA",
    "ReplayError",
    "canonical_bytes",
    "relation_summary",
    "replay_rank",
    "replay_sidecar",
    "sha256_file",
    "sha256_json",
    "validate_protocol",
    "validate_amendment",
    "validate_snapshot",
]


if __name__ == "__main__":
    raise SystemExit(main())
