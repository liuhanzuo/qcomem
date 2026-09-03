from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def pareto_prune(states):
    ordered = sorted(states, key=lambda state: (state[0], state[1], state[2]))
    frontier = []
    best = float("inf")
    for state in ordered:
        if state[1] < best:
            frontier.append(state)
            best = state[1]
    return frontier


def optimize(profiles: list[dict[str, Any]], budget: int) -> dict[str, Any]:
    states = [(0, 0.0, ())]
    for profile in profiles:
        candidates = []
        for used, distortion, choices in states:
            for option in profile["options"]:
                new_used = used + option["mean_component_nbytes"]
                if new_used <= budget:
                    candidates.append(
                        (
                            new_used,
                            distortion + option["mean_kl_divergence"],
                            (*choices, (profile["component"], option["bits"])),
                        )
                    )
        if not candidates:
            raise ValueError(f"budget {budget} is infeasible")
        states = pareto_prune(candidates)
    used, distortion, choices = min(
        states, key=lambda state: (state[1], state[0], state[2])
    )
    mapping = dict(choices)
    return {
        "budget_bytes": budget,
        "predicted_bytes": used,
        "predicted_kl_divergence": distortion,
        "residual_bits": mapping["residual"],
        "cache_layer_bits": [
            mapping[f"cache.{index}"] for index in range(len(profiles) - 1)
        ],
    }


def option_bytes(profile: dict[str, Any], bits: int) -> int:
    return next(
        option["mean_component_nbytes"]
        for option in profile["options"]
        if option["bits"] == bits
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    shards = [
        json.loads(path.read_text())
        for path in sorted(args.run_dir.glob("sensitivity-*.json"))
    ]
    if len(shards) != 8:
        raise SystemExit(f"expected 8 sensitivity shards, found {len(shards)}")
    profiles = sorted(
        shards,
        key=lambda shard: (
            shard["component"] != "residual",
            int(shard["component"].split(".")[-1])
            if shard["component"] != "residual"
            else -1,
        ),
    )
    frozen_budget = 0
    for profile in profiles:
        if profile["component"] == "residual":
            bits = 4
        else:
            bits = 8 if profile["is_linear"] else 4
        frozen_budget += option_bytes(profile, bits)
    minimum_budget = sum(option_bytes(profile, 2) for profile in profiles)
    budgets = {
        "same_memory_as_frozen": frozen_budget,
        "minus_25_percent": max(round(frozen_budget * 0.75), minimum_budget),
        "extreme_q2_floor": minimum_budget,
    }
    policies = {
        name: optimize(profiles, budget) for name, budget in budgets.items()
    }
    result = {
        "status": "completed",
        "objective": "mean first-token KL on held-out calibration prompts",
        "components": [
            {
                "component": profile["component"],
                "is_linear": profile["is_linear"],
                "options": profile["options"],
            }
            for profile in profiles
        ],
        "frozen_static_policy": {
            "residual_bits": 4,
            "attention_bits": 4,
            "linear_bits": 8,
            "predicted_bytes": frozen_budget,
        },
        "policies": policies,
    }
    destination = args.run_dir / "layer_policy.json"
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
