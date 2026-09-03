#!/usr/bin/env python3
"""Exact-rational boundary audit for Lemma 1; no carrier data is read."""
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


def main() -> None:
    checked = 0
    zero_f0 = 0
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
    assert flip(32, 17, 29) >= flip(32, 17, 31)
    print(
        "PASS monotonicity boundary audit: "
        f"{checked} (N,K,k) reversal cases, {zero_f0} zero-F0 support cases; "
        "includes N=32,K=17,k=29"
    )


if __name__ == "__main__":
    main()
