from __future__ import annotations

"""Aggregate exactly eight detached R35 rank replays without computing rates."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence


REPLAY_SCHEMA = "forkaudit-r35-historical-alias-rank-replay-v1"
AGGREGATE_SCHEMA = "forkaudit-r35-historical-alias-aggregate-v1"
LANES = ("historical_pre_fix", "repaired_borrowed", "materialized_control")
PAIRS = (
    "historical_pre_fix_vs_materialized_control",
    "historical_pre_fix_vs_repaired_borrowed",
    "repaired_borrowed_vs_materialized_control",
)
METRICS = (
    "greedy_token_exact",
    "full_fp32_logits_exact",
    "request0_terminal_gdn_content_exact",
    "logical_kv_content_exact",
    "persistent_base_content_only_invariant",
)
OUTCOMES = (
    "historical_expected_authenticated_first_gate_reproduced",
    "historical_output_only_exact_to_materialized",
    "repaired_storage_contract_clean",
    "materialized_storage_contract_clean",
    "repaired_semantic_and_terminal_exact_to_materialized",
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class AggregateError(RuntimeError):
    """An integrity or cardinality failure in detached replay aggregation."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AggregateError(message)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_sha(value: Any, label: str) -> str:
    require(
        isinstance(value, str) and SHA256_RE.fullmatch(value) is not None,
        f"{label} must be one lowercase SHA-256",
    )
    return value


def load_json(path: Path) -> Any:
    require(path.is_file(), f"missing replay: {path}")
    try:
        return json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AggregateError(f"invalid replay {path}: {exc}") from exc


def _bool_map(value: Any, expected: Sequence[str], label: str) -> dict[str, bool]:
    require(isinstance(value, Mapping) and set(value) == set(expected), f"{label} fields")
    output = dict(value)
    require(all(type(item) is bool for item in output.values()), f"{label} booleans")
    return output


def validate_replay(
    value: Any,
    *,
    expected_rank: int,
    expected_preregistration_sha256: str,
    expected_amendment_sha256: str,
    expected_execution_input_sha256: str,
    expected_source_ledger_sha256: str,
) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"rank {expected_rank} replay object")
    require(value.get("schema_version") == REPLAY_SCHEMA, f"rank {expected_rank} replay schema")
    require(value.get("rank") == expected_rank, f"rank {expected_rank} identity")
    require(
        value.get("status") == "operationally_valid_detached_replay"
        and value.get("operational_valid") is True,
        f"rank {expected_rank} operational validity",
    )
    require(value.get("candidate_modules_imported") is False, f"rank {expected_rank} replay imports")
    require(
        value.get("preregistration_raw_sha256") == expected_preregistration_sha256,
        f"rank {expected_rank} preregistration binding",
    )
    require(
        value.get("amendment_raw_sha256") == expected_amendment_sha256,
        f"rank {expected_rank} amendment binding",
    )
    require(
        value.get("execution_input_raw_sha256") == expected_execution_input_sha256,
        f"rank {expected_rank} execution-input binding",
    )
    require(
        value.get("source_ledger_raw_sha256") == expected_source_ledger_sha256,
        f"rank {expected_rank} source-ledger binding",
    )
    check_sha(value.get("rank_result_raw_sha256"), f"rank {expected_rank} result")
    require(value.get("verified_lane_count") == 3, f"rank {expected_rank} lane count")
    require(value.get("verified_sidecar_count") == 3, f"rank {expected_rank} sidecar count")
    require(value.get("population_detection_rate_computed") is False, f"rank {expected_rank} rate flag")
    require(
        value.get("fixed_pair_mappings") == list(PAIRS),
        f"rank {expected_rank} pair mappings",
    )
    lanes = value.get("lanes")
    require(isinstance(lanes, Mapping) and set(lanes) == set(LANES), f"rank {expected_rank} lane set")
    for lane_name in LANES:
        lane = lanes[lane_name]
        require(isinstance(lane, Mapping), f"rank {expected_rank} {lane_name} lane")
        nonce = lane.get("case_nonce")
        require(
            isinstance(nonce, str) and re.fullmatch(r"[0-9a-f]{32}|[0-9a-f]{64}", nonce),
            f"rank {expected_rank} {lane_name} nonce",
        )
        sidecar = lane.get("sidecar")
        require(isinstance(sidecar, Mapping), f"rank {expected_rank} {lane_name} sidecar")
        require(sidecar.get("path") == f"{lane_name}-full-fp32-logits.bin", f"rank {expected_rank} sidecar path")
        check_sha(sidecar.get("sha256"), f"rank {expected_rank} {lane_name} sidecar")
        require(sidecar.get("finite") is True, f"rank {expected_rank} {lane_name} finite sidecar")
    matrix = value.get("conventional_baseline_matrix")
    require(isinstance(matrix, Mapping) and set(matrix) == set(PAIRS), f"rank {expected_rank} baseline pairs")
    for pair in PAIRS:
        _bool_map(matrix[pair], METRICS, f"rank {expected_rank} {pair}")
    _bool_map(value.get("hypothesis_outcomes"), OUTCOMES, f"rank {expected_rank} outcomes")
    boundary = value.get("claim_boundary")
    require(isinstance(boundary, Mapping), f"rank {expected_rank} claim boundary")
    require(boundary.get("hypothesis_mismatch_is_valid_negative_not_operational_invalid") is True, f"rank {expected_rank} valid-negative boundary")
    require(boundary.get("normalized_storage_ids_compared_within_lane_only") is True, f"rank {expected_rank} storage-ID boundary")
    return dict(value)


def _empty_counts() -> dict[str, Any]:
    return {
        "cell_count": 0,
        "ranks": [],
        "hypothesis_true_counts": {name: 0 for name in OUTCOMES},
        "baseline_true_counts": {
            pair: {metric: 0 for metric in METRICS}
            for pair in PAIRS
        },
        "historical_gate_and_output_only_exact_count": 0,
        "historical_persistent_base_invariant_violation_count": 0,
    }


def aggregate(
    replays: Sequence[Mapping[str, Any]],
    *,
    replay_raw_sha256: Mapping[int, str],
    preregistration_sha256: str,
    amendment_sha256: str,
    execution_input_sha256: str,
    source_ledger_sha256: str,
) -> dict[str, Any]:
    require(len(replays) == 8, "aggregate requires exactly eight replays")
    ordered = sorted(replays, key=lambda row: int(row["rank"]))
    require([row["rank"] for row in ordered] == list(range(8)), "rank coverage")
    run_ids = {row.get("run_id") for row in ordered}
    require(len(run_ids) == 1 and next(iter(run_ids)), "run-id agreement")
    require(len({row["rank_result_raw_sha256"] for row in ordered}) == 8, "rank-result hash uniqueness")
    nonces: set[str] = set()
    sidecars: set[tuple[int, str]] = set()
    classes = {"archived_coordinates": _empty_counts(), "additional_frozen_inputs": _empty_counts()}
    overall = _empty_counts()
    for replay in ordered:
        rank = int(replay["rank"])
        class_name = "archived_coordinates" if rank < 3 else "additional_frozen_inputs"
        targets = (classes[class_name], overall)
        outcomes = replay["hypothesis_outcomes"]
        matrix = replay["conventional_baseline_matrix"]
        for target in targets:
            target["cell_count"] += 1
            target["ranks"].append(rank)
            for name in OUTCOMES:
                target["hypothesis_true_counts"][name] += int(outcomes[name])
            for pair in PAIRS:
                for metric in METRICS:
                    target["baseline_true_counts"][pair][metric] += int(matrix[pair][metric])
            target["historical_gate_and_output_only_exact_count"] += int(
                outcomes["historical_expected_authenticated_first_gate_reproduced"]
                and outcomes["historical_output_only_exact_to_materialized"]
            )
            target["historical_persistent_base_invariant_violation_count"] += int(
                not matrix["historical_pre_fix_vs_materialized_control"][
                    "persistent_base_content_only_invariant"
                ]
            )
        for lane_name in LANES:
            nonce = replay["lanes"][lane_name]["case_nonce"]
            require(nonce not in nonces, "duplicate case nonce across ranks")
            nonces.add(nonce)
            key = (rank, replay["lanes"][lane_name]["sidecar"]["path"])
            require(key not in sidecars, "duplicate rank-qualified sidecar")
            sidecars.add(key)
    require(len(nonces) == 24 and len(sidecars) == 24, "global lane/sidecar cardinality")
    require(classes["archived_coordinates"]["ranks"] == [0, 1, 2], "archived rank split")
    require(classes["additional_frozen_inputs"]["ranks"] == [3, 4, 5, 6, 7], "additional rank split")
    true_counts = overall["hypothesis_true_counts"]
    headline = {
        "all_eight_historical_expected_gates_reproduced": true_counts[
            "historical_expected_authenticated_first_gate_reproduced"
        ] == 8,
        "all_eight_historical_output_only_pairs_exact": true_counts[
            "historical_output_only_exact_to_materialized"
        ] == 8,
        "all_eight_repaired_storage_contracts_clean": true_counts[
            "repaired_storage_contract_clean"
        ] == 8,
        "all_eight_materialized_storage_contracts_clean": true_counts[
            "materialized_storage_contract_clean"
        ] == 8,
        "all_eight_repaired_pairs_semantic_and_terminal_exact": true_counts[
            "repaired_semantic_and_terminal_exact_to_materialized"
        ] == 8,
    }
    headline["registered_positive_headline_rule_satisfied"] = all(headline.values())
    return {
        "schema_version": AGGREGATE_SCHEMA,
        "run_id": next(iter(run_ids)),
        "status": "operationally_valid_aggregate",
        "operational_valid": True,
        "preregistration_raw_sha256": preregistration_sha256,
        "amendment_raw_sha256": amendment_sha256,
        "execution_input_raw_sha256": execution_input_sha256,
        "source_ledger_raw_sha256": source_ledger_sha256,
        "rank_replay_raw_sha256": {str(rank): replay_raw_sha256[rank] for rank in range(8)},
        "operational_cardinality": {
            "rank_replay_count": 8,
            "lane_count": 24,
            "fp32_sidecar_count": 24,
            "unique_case_nonce_count": 24,
        },
        "coordinate_classes": classes,
        "overall_cell_counts": overall,
        "headline_outcomes": headline,
        "population_detection_rate_computed": False,
        "statistical_independence_claimed": False,
        "claim_boundary": {
            "archived_and_additional_coordinates_reported_separately": True,
            "all_values_are_cell_counts_or_exact_all_cell_booleans": True,
            "hypothesis_mismatch_remains_a_valid_scientific_outcome": True,
        },
    }


def write_new_json(path: Path, value: Any) -> None:
    require(not path.exists(), "aggregate output already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".pending")
    require(not pending.exists(), "stale aggregate pending file")
    with pending.open("xb") as handle:
        handle.write(canonical_bytes(value) + b"\n")
    pending.replace(path)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--replay-root", type=Path, required=True)
    value.add_argument("--expected-preregistration-sha256", required=True)
    value.add_argument("--expected-amendment-sha256", required=True)
    value.add_argument("--expected-execution-input-sha256", required=True)
    value.add_argument("--expected-source-ledger-sha256", required=True)
    value.add_argument("--output", type=Path, required=True)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        preregistration_sha = check_sha(args.expected_preregistration_sha256, "preregistration")
        amendment_sha = check_sha(args.expected_amendment_sha256, "amendment")
        execution_input_sha = check_sha(args.expected_execution_input_sha256, "execution input")
        source_ledger_sha = check_sha(args.expected_source_ledger_sha256, "source ledger")
        require(args.replay_root.is_dir(), "replay root is not a directory")
        expected_names = {f"rank-{rank}-replay.json" for rank in range(8)}
        observed_names = {path.name for path in args.replay_root.iterdir()}
        require(observed_names == expected_names, "missing, duplicate, or extra replay-root paths")
        replays = []
        raw_hashes: dict[int, str] = {}
        for rank in range(8):
            path = args.replay_root / f"rank-{rank}-replay.json"
            raw_hashes[rank] = sha256_file(path)
            replays.append(
                validate_replay(
                    load_json(path),
                    expected_rank=rank,
                    expected_preregistration_sha256=preregistration_sha,
                    expected_amendment_sha256=amendment_sha,
                    expected_execution_input_sha256=execution_input_sha,
                    expected_source_ledger_sha256=source_ledger_sha,
                )
            )
        receipt = aggregate(
            replays,
            replay_raw_sha256=raw_hashes,
            preregistration_sha256=preregistration_sha,
            amendment_sha256=amendment_sha,
            execution_input_sha256=execution_input_sha,
            source_ledger_sha256=source_ledger_sha,
        )
        write_new_json(args.output, receipt)
        return 0
    except (OSError, AggregateError) as exc:
        print(f"R35 aggregation invalid: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
