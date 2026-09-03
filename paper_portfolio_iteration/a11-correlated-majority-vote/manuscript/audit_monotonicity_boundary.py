#!/usr/bin/env python3
"""Exact-rational boundary audit for Lemma 1; no carrier data is read.

It checks the reversal-boundary algebra, the even-N center case, and the
separate no-feasible-odd-budget -> FULL-N fallback contract.
"""
from __future__ import annotations

from fractions import Fraction
from math import comb


def choose(n: int, r: int) -> int:
    return comb(n, r) if 0 <= r <= n else 0


def hg(n: int, K: int, k: int, x: int) -> Fraction:
    return Fraction(choose(K, x) * choose(n - K, k - x), choose(n, k))


def flip(n: int, K: int, k: int) -> Fraction:
    full_positive = K > n / 2
    return sum(
        hg(n, K, k, x)
        for x in range(k + 1)
        if (x > k / 2) != full_positive
    )


def expected_flip(n: int, mixture: dict[int, Fraction], k: int) -> Fraction:
    return sum((weight * flip(n, K, k) for K, weight in mixture.items()), Fraction())


def select_odd_early_or_full_fallback(
    n: int, mixture: dict[int, Fraction], alpha: Fraction
) -> tuple[str, int]:
    """Return an odd early budget only if one is feasible; otherwise FULL-N.

    The actual OMR protocol is a finite subgrid of this ladder.  This generic
    edge audit uses k=3,5,...,N-1 to make the empty-set semantics explicit.
    FULL-N is deliberately outside the early-budget candidate set.
    """
    eligible = tuple(range(3, n, 2))
    feasible = [k for k in eligible if expected_flip(n, mixture, k) <= alpha]
    if feasible:
        return "odd-early-budget", min(feasible)
    return "full-N-fallback", n


def main() -> None:
    checked = 0
    zero_f0 = 0
    center_checked = 0
    fallback_checked = 0
    for n in range(2, 65):
        for K in range(n // 2 + 1, n + 1):
            for k in range(1, n - 1, 2):
                t = (k + 1) // 2
                e2 = hg(n, K, k, t - 1) * Fraction(choose(K - t + 1, 2), choose(n - k, 2))
                f0 = hg(n, K, k, t) * Fraction(choose(n - K - t + 1, 2), choose(n - k, 2))
                assert flip(n, K, k) - flip(n, K, k + 2) == e2 - f0
                assert flip(n, K, k + 2) <= flip(n, K, k)
                if f0 == 0:
                    zero_f0 += 1
                    assert e2 >= 0
                else:
                    # All ratio denominators are now positive; this is the printed case.
                    assert n - K - t > 0
                    assert e2 / f0 == Fraction(K - t, n - K - t)
                checked += 1

    # If N is even and K=N/2, complementing pass/fail maps x to k-x.  Odd k
    # has no tie, so each prefix side has probability exactly 1/2.  At alpha
    # below 1/2 no odd early budget is feasible; the distinct FULL-N fallback
    # has zero replay flip and is not silently named k*.
    for n in range(2, 65):
        if n % 2:
            continue
        K = n // 2
        center_mixture = {K: Fraction(1)}
        for k in range(1, n, 2):
            assert flip(n, K, k) == Fraction(1, 2)
            center_checked += 1
        status, budget = select_odd_early_or_full_fallback(
            n, center_mixture, Fraction(49, 100)
        )
        assert status == "full-N-fallback"
        assert budget == n
        assert flip(n, K, budget) == 0
        fallback_checked += 1

    assert flip(32, 17, 29) >= flip(32, 17, 31)
    print(
        "PASS monotonicity boundary audit: "
        f"{checked} (N,K,k) reversal cases, {zero_f0} zero-F0 support cases; "
        f"{center_checked} even-N center checks, {fallback_checked} alpha<1/2 "
        "no-feasible-odd -> FULL-N fallback checks; includes N=32,K=17,k=29"
    )


if __name__ == "__main__":
    main()
