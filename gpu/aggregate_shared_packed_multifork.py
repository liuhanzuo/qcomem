"""Torch-free aggregator and blind re-validator for the C1 multifork shards.

Reads every ``multifork-shard-*.json`` in a run directory, re-runs the shard's
own row validator and contract arithmetic without a GPU stack, and writes one
aggregate.  It fails closed on an incomplete run: a missing shard, a shard that
stopped at its gate, an arm missing from a fanout, a contract row whose status
disagrees with its own coverage and predicate, or a semantic-equivalence
discrepancy are all reported as defects rather than averaged away.

Deliberately *not* enforced here: that any predicate passed.  A run in which the
ownership contract fails is a valid scientific negative and must aggregate; the
aggregate reports the failure explicitly.  What is enforced is that the record
is complete and internally consistent.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from qcomem_multifork_accounting import (
    AGGREGATE_SCHEMA,
    MULTIFORK_ARMS,
    MULTIFORK_TARGET_CONTRACT,
    PROTOCOL,
    SHARD_SCHEMA,
    contract_summary,
    crossover_request_count,
    format_mib,
    packed_entry_obligation_names,
    summarize_multifork_rows,
    validate_multifork_row,
)


class AggregateError(RuntimeError):
    """The archived record is incomplete or internally inconsistent."""


def load_shards(run_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    paths = sorted(run_dir.glob("multifork-shard-*.json"))
    return [(path, json.loads(path.read_text())) for path in paths]


def replay_contract_rows(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Re-derive each target's status from its own coverage and predicate.

    This is the blind replay: the producer's status is not trusted.  A row whose
    coverage is incomplete must be ``open`` with a null predicate; a row whose
    predicate is false must be ``open``; a row whose predicate is true must
    carry its declared maximum status.
    """

    contract_by_target = {row["target"]: row for row in MULTIFORK_TARGET_CONTRACT}
    problems: list[str] = []
    seen = set()
    for row in rows:
        target = row.get("target")
        if target not in contract_by_target:
            problems.append(f"unknown target in shard: {target!r}")
            continue
        if target in seen:
            problems.append(f"duplicate target row: {target}")
        seen.add(target)
        declared = contract_by_target[target]
        for field in ("applicability", "maximum_status", "predicate_id", "family"):
            if row.get(field) != declared[field]:
                problems.append(
                    f"{target}: {field} drifted from the frozen contract"
                )
        coverage = row.get("coverage")
        passed = row.get("predicate_passed")
        status = row.get("status")
        if declared["applicability"] == "not_applicable":
            expected = "not_applicable"
        elif coverage != "complete":
            expected = "open"
            if passed is not None:
                problems.append(
                    f"{target}: incomplete coverage carries a non-null predicate"
                )
        elif passed is True:
            expected = declared["maximum_status"]
        elif passed is False:
            expected = "open"
        else:
            problems.append(f"{target}: complete coverage with no boolean predicate")
            continue
        if status != expected:
            problems.append(
                f"{target}: status {status!r} does not replay to {expected!r}"
            )
    missing = sorted(set(contract_by_target) - seen)
    if missing:
        problems.append(f"contract rows missing for targets: {missing}")
    return problems


def validate_shard(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    problems: list[str] = []
    if payload.get("schema") != SHARD_SCHEMA:
        problems.append(f"schema is {payload.get('schema')!r}")
    if payload.get("protocol") != PROTOCOL:
        problems.append(f"protocol is {payload.get('protocol')!r}")
    status = payload.get("status")
    if status != "completed":
        problems.append(f"status is {status!r}, not 'completed'")
    gates = payload.get("gates") or {}
    if not gates:
        problems.append("shard carries no gate record")
    for name, gate in gates.items():
        if not isinstance(gate, Mapping) or not gate.get("passed"):
            problems.append(f"gate {name} did not pass")
    rows = payload.get("rows") or []
    if not rows:
        problems.append("shard carries no rows")
    row_problems: list[str] = []
    for index, row in enumerate(rows):
        for message in validate_multifork_row(row):
            row_problems.append(f"row {index}: {message}")
        if row["arm"] == "qcomem-shared-packed":
            audit = row.get("forkaudit") or {}
            target_rows = audit.get("target_rows")
            if not target_rows:
                row_problems.append(f"row {index}: shared arm carries no contract rows")
            else:
                for message in replay_contract_rows(target_rows):
                    row_problems.append(f"row {index}: {message}")
                replayed = contract_summary(target_rows)
                declared = audit.get("contract_summary") or {}
                for field in (
                    "status_vector",
                    "coverage_vector",
                    "all_applicable_targets_covered",
                    "all_applicable_predicates_passed",
                ):
                    if declared.get(field) != replayed[field]:
                        row_problems.append(
                            f"row {index}: contract summary field {field} drifted"
                        )
    problems.extend(row_problems)
    return {
        "path": str(path),
        "rank": payload.get("rank"),
        "row_count": len(rows),
        "problems": problems,
        "valid": not problems,
    }


def arm_coverage(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Every declared fanout must carry every arm; report the gaps."""

    by_fanout: dict[int, set[str]] = defaultdict(set)
    for row in rows:
        by_fanout[int(row["request_count"])].add(row["arm"])
    missing = {
        fanout: sorted(set(MULTIFORK_ARMS) - arms)
        for fanout, arms in sorted(by_fanout.items())
        if set(MULTIFORK_ARMS) - arms
    }
    return {
        "fanouts": sorted(by_fanout),
        "arms_per_fanout": {
            fanout: sorted(arms) for fanout, arms in sorted(by_fanout.items())
        },
        "missing_arms_per_fanout": missing,
        "complete": not missing,
    }


def semantic_equivalence_report(
    rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Count and list every token-trace discrepancy, per arm and fanout.

    The Q-CoMem shared and private arms are gating comparisons: they must be
    token-identical to the published N=1 private path.  The full-prefix arm is
    a diagnostic and its discrepancies are counted but never treated as a
    defect, because that comparison crosses the document/query chunk boundary
    the Qwen3.5 recurrence is sensitive to.
    """

    gating_arms = ("qcomem-shared-packed", "qcomem-private-materialize")
    counts: dict[tuple[str, int], dict[str, int]] = defaultdict(
        lambda: {"rows": 0, "identical": 0, "discrepant": 0}
    )
    discrepancies: list[dict[str, Any]] = []
    for row in rows:
        key = (row["arm"], int(row["request_count"]))
        cell = counts[key]
        cell["rows"] += 1
        equivalence = row["semantic_equivalence"]
        if equivalence.get("token_sequences_identical"):
            cell["identical"] += 1
        else:
            cell["discrepant"] += 1
            discrepancies.append(
                {
                    "arm": row["arm"],
                    "request_count": int(row["request_count"]),
                    "workload_id": row["workload_id"],
                    "repeat": row.get("repeat"),
                    "gating": row["arm"] in gating_arms,
                    "discrepancies": equivalence.get("discrepancies"),
                }
            )
    gating_failures = [row for row in discrepancies if row["gating"]]
    return {
        "gating_arms": list(gating_arms),
        "diagnostic_arms": ["full-prefix"],
        "per_arm_fanout": {
            f"{arm}@{fanout}": cell for (arm, fanout), cell in sorted(counts.items())
        },
        "discrepancies": discrepancies,
        "gating_failure_count": len(gating_failures),
        "gating_all_identical": not gating_failures,
    }


def contract_report(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Coverage and verdict across every shared-arm row, per target."""

    per_target: dict[str, dict[str, int]] = {
        row["target"]: {
            "rows": 0,
            "covered": 0,
            "passed": 0,
            "open": 0,
            "uncovered": 0,
        }
        for row in MULTIFORK_TARGET_CONTRACT
    }
    shared_rows = [row for row in rows if row["arm"] == "qcomem-shared-packed"]
    for row in shared_rows:
        for target_row in (row.get("forkaudit") or {}).get("target_rows", ()):
            cell = per_target[target_row["target"]]
            cell["rows"] += 1
            if target_row["coverage"] == "complete":
                cell["covered"] += 1
            else:
                cell["uncovered"] += 1
            if target_row["predicate_passed"] is True:
                cell["passed"] += 1
            if target_row["status"] == "open":
                cell["open"] += 1
    obligations = packed_entry_obligation_names()
    return {
        "shared_arm_row_count": len(shared_rows),
        "per_target": per_target,
        "packed_entry_obligations": list(obligations),
        "all_targets_covered_everywhere": all(
            cell["uncovered"] == 0 and cell["rows"] > 0
            for cell in per_target.values()
        ),
        "all_targets_passed_everywhere": all(
            cell["passed"] == cell["rows"] and cell["rows"] > 0
            for cell in per_target.values()
        ),
        "packed_entry_obligations_passed_everywhere": all(
            per_target[name]["passed"] == per_target[name]["rows"]
            and per_target[name]["rows"] > 0
            for name in obligations
        ),
    }


def working_set_table(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One printable line per (arm, fanout) with the transient-term columns."""

    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["arm"], int(row["request_count"]))].append(row)
    table = []
    for (arm, fanout), cell in sorted(grouped.items()):
        table.append(
            {
                "arm": arm,
                "request_count": fanout,
                "rows": len(cell),
                "entry_retained_mib": format_mib(
                    statistics.median(
                        [row["entry_retained_nbytes"] for row in cell]
                    )
                ),
                "shared_view_mib": format_mib(
                    statistics.median(
                        [row["shared_dequantized_view_nbytes"] for row in cell]
                    )
                ),
                "transient_materialized_total_mib": format_mib(
                    statistics.median(
                        [row["transient_materialized_nbytes_total"] for row in cell]
                    )
                ),
                "peak_transient_allocation_mib": format_mib(
                    statistics.median(
                        [row["peak_transient_allocation_nbytes"] for row in cell]
                    )
                ),
                "steady_state_resident_mib": format_mib(
                    statistics.median(
                        [row["steady_state_resident_nbytes"] for row in cell]
                    )
                ),
                "resident_intercept_mib": format_mib(
                    statistics.median(
                        [row["resident_model"]["intercept_nbytes"] for row in cell]
                    )
                ),
                "resident_slope_mib_per_request": format_mib(
                    statistics.median(
                        [
                            row["resident_model"]["slope_nbytes_per_request"]
                            for row in cell
                        ]
                    )
                ),
            }
        )
    return table


def crossover_report(
    table: Sequence[Mapping[str, Any]], *, search_limit: int
) -> list[dict[str, Any]]:
    """Fitted resident crossover of each Q-CoMem arm against full prefix."""

    by_key = {(row["arm"], row["request_count"]): row for row in table}
    report = []
    for (arm, fanout), row in sorted(by_key.items()):
        if arm == "full-prefix":
            continue
        reference = by_key.get(("full-prefix", fanout))
        if reference is None:
            continue
        report.append(
            {
                "arm": arm,
                "request_count": fanout,
                **crossover_request_count(
                    left={
                        "intercept_nbytes": row["resident_intercept_mib"] * 2**20,
                        "slope_nbytes_per_request": (
                            row["resident_slope_mib_per_request"] * 2**20
                        ),
                    },
                    right={
                        "intercept_nbytes": reference["resident_intercept_mib"]
                        * 2**20,
                        "slope_nbytes_per_request": (
                            reference["resident_slope_mib_per_request"] * 2**20
                        ),
                    },
                    max_request_count=search_limit,
                ),
            }
        )
    return report


def aggregate(
    run_dir: Path, *, expected_shards: int | None, search_limit: int = 4096
) -> dict[str, Any]:
    shards = load_shards(run_dir)
    if not shards:
        raise AggregateError(f"no multifork shards under {run_dir}")
    if expected_shards is not None and len(shards) != expected_shards:
        raise AggregateError(
            f"expected {expected_shards} shards, found {len(shards)}"
        )
    validations = [validate_shard(path, payload) for path, payload in shards]
    rows: list[dict[str, Any]] = []
    for _, payload in shards:
        rows.extend(payload.get("rows") or [])
    protocols = {payload.get("protocol") for _, payload in shards}
    labels = {
        (payload.get("protocol_settings") or {}).get("label")
        for _, payload in shards
    }
    tail_policies = {
        (payload.get("protocol_settings") or {}).get("tail_policy")
        for _, payload in shards
    }
    rebind_policies = {
        (payload.get("protocol_settings") or {}).get("rebind_policy")
        for _, payload in shards
    }
    defects: list[str] = []
    for validation in validations:
        for problem in validation["problems"]:
            defects.append(f"{validation['path']}: {problem}")
    if len(protocols) != 1:
        defects.append(f"shards disagree on protocol: {sorted(map(str, protocols))}")
    if len(labels) != 1:
        defects.append(f"shards disagree on protocol label: {sorted(map(str, labels))}")
    if len(tail_policies) != 1:
        defects.append(
            f"shards disagree on tail policy: {sorted(map(str, tail_policies))}"
        )
    if len(rebind_policies) != 1:
        defects.append(
            f"shards disagree on rebind policy: {sorted(map(str, rebind_policies))}"
        )
    coverage = arm_coverage(rows)
    if not coverage["complete"]:
        defects.append(f"arm coverage is incomplete: {coverage['missing_arms_per_fanout']}")
    table = working_set_table(rows)
    return {
        "schema": AGGREGATE_SCHEMA,
        "protocol": PROTOCOL,
        "run_dir": str(run_dir),
        "shard_count": len(shards),
        "row_count": len(rows),
        "protocol_label": sorted(map(str, labels)),
        "tail_policy": sorted(map(str, tail_policies)),
        "rebind_policy": sorted(map(str, rebind_policies)),
        "shard_validations": validations,
        "record_complete": not defects,
        "defects": defects,
        "arm_coverage": coverage,
        "arm_summaries": summarize_multifork_rows(rows),
        "working_set_table": table,
        "resident_crossover": crossover_report(table, search_limit=search_limit),
        "semantic_equivalence": semantic_equivalence_report(rows),
        "forkaudit_contract": contract_report(rows),
        "scope": (
            "this aggregate covers the shared-packed Read path on the "
            "Transformers split-replay stack only; it establishes nothing "
            "about a paged kernel, a second backbone, throughput, or admitted "
            "serving capacity"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="aggregate and blind-replay the C1 multifork shards"
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--expected-shards", type=int)
    parser.add_argument("--search-limit", type=int, default=4096)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--require-complete-record",
        action="store_true",
        help="exit non-zero when the archived record is incomplete",
    )
    args = parser.parse_args()
    result = aggregate(
        args.run_dir,
        expected_shards=args.expected_shards,
        search_limit=args.search_limit,
    )
    destination = args.output or (args.run_dir / "multifork-aggregate.json")
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in (
        "shard_count",
        "row_count",
        "record_complete",
        "working_set_table",
    )}, indent=2))
    if result["defects"]:
        print("DEFECTS:", flush=True)
        for defect in result["defects"]:
            print(f"  - {defect}", flush=True)
    if args.require_complete_record and not result["record_complete"]:
        raise SystemExit("the archived C1 record is incomplete")
    print(f"SAVED {destination}", flush=True)


if __name__ == "__main__":
    main()
