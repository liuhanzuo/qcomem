from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from qcomem_joint_policy import (
    FROZEN_STATIC_LAYER_BITS,
    component_profile_order,
    frozen_static_policy,
    predicted_policy_cost,
    profile_option,
    q16_exactness_passes,
    q16_policy,
    top_predicted_policies,
    uniform_q8_policy,
)
from run_downstream import atomic_json


def _file_set_digest(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _candidate_payload(
    name: str,
    bits: tuple[int, ...],
    byte_count: int,
    objective: float,
    groups: list[str],
) -> dict[str, Any]:
    return {
        "name": name,
        "residual_bits": bits[0],
        "cache_layer_bits": list(bits[1:]),
        "attention_bits": 16,
        "linear_bits": 16,
        "selection_group": "pg19_joint_automatic_candidate",
        "eligible_budgets": sorted(groups),
        "predicted_component_bytes": byte_count,
        "predicted_component_objective": objective,
        "prediction_semantics": (
            "additive component profile used only to nominate this candidate; "
            "selection requires actual joint-quantization evaluation"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate joint-policy candidates from expanded PG-19 profiles"
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--expected-shards", type=int, default=8)
    parser.add_argument("--candidates-per-budget", type=int, default=6)
    args = parser.parse_args()
    paths = sorted(args.run_dir.glob("joint-profile-*.json"))
    if len(paths) != args.expected_shards:
        raise SystemExit(
            f"expected {args.expected_shards} component profiles, found {len(paths)}"
        )
    profiles = [json.loads(path.read_text()) for path in paths]
    if any(profile.get("status") != "completed" for profile in profiles):
        raise SystemExit("one or more component profiles are incomplete")
    expected_components = {"residual", *(f"cache.{index}" for index in range(7))}
    actual_components = {str(profile.get("component")) for profile in profiles}
    if actual_components != expected_components:
        raise SystemExit(f"component set mismatch: {sorted(actual_components)}")
    if len({json.dumps(item["protocol"], sort_keys=True) for item in profiles}) != 1:
        raise SystemExit("component profiles used different PG-19 protocols/windows")
    profiles.sort(key=component_profile_order)
    for profile in profiles:
        q16 = profile_option(profile, 16)["metrics"]
        if not q16_exactness_passes(q16):
            raise SystemExit(
                f"{profile['component']} failed its Q16 exactness gate: {q16}"
            )

    depth = 7
    q16 = q16_policy(depth)
    frozen = frozen_static_policy(depth)
    uniform_q8 = uniform_q8_policy(depth)
    q16_bits = (q16.residual_bits, *q16.cache_layer_bits)
    frozen_bits = (frozen.residual_bits, *frozen.cache_layer_bits)
    uniform_q8_bits = (uniform_q8.residual_bits, *uniform_q8.cache_layer_bits)
    frozen_budget, frozen_predicted_objective = predicted_policy_cost(
        profiles, frozen_bits
    )
    minimum_budget, _ = predicted_policy_cost(profiles, (2,) * 8)
    minus25_budget = max(round(frozen_budget * 0.75), minimum_budget)

    excluded = (q16_bits, frozen_bits, uniform_q8_bits)
    same = top_predicted_policies(
        profiles,
        budget_bytes=frozen_budget,
        limit=args.candidates_per_budget,
        excluded_bits=excluded,
    )
    minus25 = top_predicted_policies(
        profiles,
        budget_bytes=minus25_budget,
        limit=args.candidates_per_budget,
        excluded_bits=excluded,
    )
    nominated: dict[tuple[int, ...], dict[str, Any]] = {}
    for group, rows in (("same_memory", same), ("minus_25_percent", minus25)):
        for bits, byte_count, objective in rows:
            item = nominated.setdefault(
                bits,
                {
                    "bits": bits,
                    "byte_count": byte_count,
                    "objective": objective,
                    "groups": [],
                },
            )
            item["groups"].append(group)
    ordered = sorted(
        nominated.values(),
        key=lambda item: (item["objective"], item["byte_count"], item["bits"]),
    )
    automatic = [
        _candidate_payload(
            f"pg19-joint-auto-{index:02d}",
            item["bits"],
            item["byte_count"],
            item["objective"],
            item["groups"],
        )
        for index, item in enumerate(ordered)
    ]
    controls = []
    for policy in (q16, frozen, uniform_q8):
        payload = policy.as_dict()
        bits = (policy.residual_bits, *policy.cache_layer_bits)
        predicted_bytes, predicted_objective = predicted_policy_cost(profiles, bits)
        payload.update(
            {
                "eligible_budgets": [],
                "predicted_component_bytes": predicted_bytes,
                "predicted_component_objective": predicted_objective,
            }
        )
        controls.append(payload)

    protocol = profiles[0]["protocol"]
    result = {
        "status": "candidates_frozen_before_joint_evaluation",
        "stage": "expanded_pg19_joint_candidate_generation",
        "protocol": protocol,
        "profile_file_set_sha256": _file_set_digest(paths),
        "profiles": [
            {
                "component": profile["component"],
                "component_kind": profile["component_kind"],
                "options": profile["options"],
            }
            for profile in profiles
        ],
        "budgets": {
            "same_memory": frozen_budget,
            "minus_25_percent": minus25_budget,
            "minimum_all_q2": minimum_budget,
            "frozen_static_predicted_objective": frozen_predicted_objective,
        },
        "controls": controls,
        "automatic_candidates": automatic,
        "evaluation_policies": [*controls, *automatic],
        "selection_contract": {
            "selection_source": "PG-19 official train objects only",
            "joint_metrics": (
                "full combined pack/dequant policy on multi-position teacher logits "
                "and natural PG-19 next-token NLL"
            ),
            "component_additivity_claimed": False,
            "longbench_validation_labels_used": False,
            "formal_validation_source_6_35_may_select_policy": False,
            "frozen_test_v2_source_68_99_used": False,
            "automatic_policy_reuses_formal_validation_label": False,
            "legacy_layer_validation_names_reused": False,
        },
    }
    destination = args.run_dir / "joint-policy-candidates.json"
    atomic_json(destination, result)
    print(
        json.dumps(
            {
                "saved": str(destination),
                "automatic_candidates": len(automatic),
                "evaluation_policies": len(result["evaluation_policies"]),
                "budgets": result["budgets"],
                "frozen_static_layer_bits": list(FROZEN_STATIC_LAYER_BITS),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
