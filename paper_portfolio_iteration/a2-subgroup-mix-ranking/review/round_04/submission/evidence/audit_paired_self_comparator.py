#!/usr/bin/env python3
"""CPU-only audit for the nonnegative paired certificate repair.

Reads the frozen r1903 result artifact without modifying it.  The historical JSON stores
uncapped paired UCBs.  For every nonnegative tolerance, replacing u by max(0,u) cannot change
the gate u <= tau; this script checks the stored frontier at its reported tau and exercises the
strictly-dominant two-model boundary where a valid pairwise UCB is negative.
"""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "remote_snapshot/results/SUBGMIX_M25_PAIRED_R1885_5SEED.json"
TAUS = (0.0, 0.02, 0.04, 0.10)
FIELDS = ("UB_paired", "UB_paired_hoef", "UB_paired_mpb")


def gate(value: float, tau: float) -> bool:
    return value <= tau


def main() -> None:
    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    rows = payload["rows"]
    report = {"source": str(SOURCE.relative_to(ROOT)), "rows": len(rows), "tau_checks": {}}
    for field in FIELDS:
        vals = [float(row[field]) for row in rows]
        report["tau_checks"][field] = {
            "min_raw": min(vals),
            "negative_raw_values": sum(x < 0 for x in vals),
            "decisions_changed_by_tau": {
                str(tau): sum(gate(x, tau) != gate(max(0.0, x), tau) for x in vals)
                for tau in TAUS
            },
        }
        assert all(n == 0 for n in report["tau_checks"][field]["decisions_changed_by_tau"].values())

    # Boundary: chosen i strictly dominates j, yet a valid upper bound on R_i-R_j is negative.
    # The old max over j != i would return -0.20 and fail to be a nonnegative regret certificate.
    dominant_pair_ucb = -0.20
    old_value = dominant_pair_ucb
    repaired_value = max(0.0, dominant_pair_ucb)
    assert repaired_value == 0.0
    assert all(gate(old_value, tau) == gate(repaired_value, tau) for tau in TAUS)
    report["boundary_test"] = {
        "scenario": "two models; chosen model strictly dominates its sole comparator",
        "valid_pairwise_ucb": dominant_pair_ucb,
        "old_certificate": old_value,
        "repaired_certificate": repaired_value,
        "regret": 0.0,
        "gate_unchanged_for_nonnegative_tau": True,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
