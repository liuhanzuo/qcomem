#!/usr/bin/env python3
"""Boundary-only audit for Proposition 1's TV-radius status contract.

This script intentionally reads no carrier or LP artifact.  It distinguishes a
feasible critical radius of exactly zero from an empty feasible set at R=0,
which must be represented by an explicit infeasible sentinel rather than 0.
"""
from __future__ import annotations

from fractions import Fraction


def affine_radius_status(v0: Fraction, slope: Fraction, alpha: Fraction) -> dict[str, object]:
    """Exact contract test for a nondecreasing affine stand-in V(R)=v0+slope*R.

    The paper's stored profiles are capacity-constrained piecewise-linear LPs,
    not assumed affine.  This minimal stand-in exercises only the universal
    R=0 feasibility branch in the definition.
    """
    if v0 > alpha:
        return {
            "status": "infeasible_at_R0",
            "feasible_at_R0": False,
            "tau_star": None,
        }
    if slope == 0:
        tau = Fraction(1)
    else:
        tau = min(Fraction(1), (alpha - v0) / slope)
    return {
        "status": "feasible",
        "feasible_at_R0": True,
        "tau_star": tau,
    }


def main() -> None:
    # Empty R_alpha: null/sentinel, never the overloaded numeric value zero.
    infeasible = affine_radius_status(Fraction(3, 10), Fraction(1, 5), Fraction(1, 4))
    assert infeasible == {
        "status": "infeasible_at_R0",
        "feasible_at_R0": False,
        "tau_star": None,
    }

    # R=0 can itself be a valid maximum, so it needs a distinct representation.
    zero_feasible = affine_radius_status(Fraction(1, 4), Fraction(1, 5), Fraction(1, 4))
    assert zero_feasible == {
        "status": "feasible",
        "feasible_at_R0": True,
        "tau_star": Fraction(0),
    }
    print("PASS TV-radius status audit: infeasible sentinel distinct from feasible tau*=0")


if __name__ == "__main__":
    main()
