#!/usr/bin/env python3
"""Stdlib-only replay of the recovered OMR r469 count-level main table.

This verifier deliberately consumes only the reviewer-safe derived manifest:
problem SHA-256, retained source order, count K, and split assignment.  It
never needs, reads, or reconstructs raw problem text or the parquet source.
"""

import argparse
import hashlib
import json
import math
import random
from math import comb, log, sqrt
from pathlib import Path


N = 32
KGRID = list(range(3, N, 2))
AGRID = [0.10, 0.05, 0.02, 0.01]
DELTA_CAL = 0.05
J_FAM = len(KGRID) * len(AGRID) + len(AGRID)
D_TEST = DELTA_CAL / J_FAM
SEED = 20260815


def hyper_pmf(K, k, x):
    if x < max(0, k - (N - K)) or x > min(k, K):
        return 0.0
    return comb(K, x) * comb(N - K, k - x) / comb(N, k)


def side(cnt, k):
    return 1 if cnt > k / 2 else 0


def build_flip_fixed():
    table = {}
    for K in range(N + 1):
        full = side(K, N)
        table[K] = {
            k: sum(
                hyper_pmf(K, k, x)
                for x in range(k + 1)
                if side(x, k) != full
            )
            for k in KGRID
        }
    return table


def build_cert_table(H):
    likelihood = {
        K: [[hyper_pmf(K, k, x) for x in range(k + 1)] for k in range(N + 1)]
        for K in range(N + 1)
    }
    cert = {}
    for k in range(N + 1):
        cert[k] = []
        for x in range(k + 1):
            numerator = 0.0
            denominator = 0.0
            prefix_side = side(x, k)
            for K in range(N + 1):
                weight = H[K] * likelihood[K][k][x]
                denominator += weight
                if side(K, N) != prefix_side:
                    numerator += weight
            cert[k].append(numerator / denominator if denominator > 0 else 0.0)
    return cert


def dp_adaptive_flip(K, cert, alpha):
    full = side(K, N)
    reach = {(0, 0): 1.0}
    flip = 0.0
    expected_k = 0.0
    for k in range(N):
        nxt = {}
        for (kk, x), probability in reach.items():
            if kk != k:
                continue
            if k >= 3 and cert[k][x] <= alpha:
                expected_k += probability * k
                if side(x, k) != full:
                    flip += probability
                continue
            pass_probability = (K - x) / (N - k)
            fail_probability = 1.0 - pass_probability
            if fail_probability > 0:
                nxt[(k + 1, x)] = nxt.get((k + 1, x), 0.0) + probability * fail_probability
            if pass_probability > 0:
                nxt[(k + 1, x + 1)] = nxt.get((k + 1, x + 1), 0.0) + probability * pass_probability
        reach = nxt
    for (kk, x), probability in reach.items():
        assert kk == N
        expected_k += probability * N
        if side(x, N) != full:
            flip += probability
    return flip, expected_k


def eb_ucb(values, delta):
    m = len(values)
    mean = sum(values) / m
    variance = sum((value - mean) ** 2 for value in values) / (m - 1) if m > 1 else 0.0
    return mean + sqrt(2 * variance * log(4 / delta) / m) + 7 * log(4 / delta) / (3 * (m - 1))


def hoef_ucb(values, delta):
    return sum(values) / len(values) + sqrt(log(1 / delta) / (2 * len(values)))


def mean_ci(values, delta):
    mean = sum(values) / len(values)
    radius = sqrt(log(2 / delta) / (2 * len(values)))
    return mean, radius


def load_counts(manifest_path):
    data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    required = {
        "K": N,
        "split_seed": SEED,
        "deduplicated_problem_count": 11607,
        "split_counts": {"FIT": 4000, "CAL": 4000, "TEST": 3607},
    }
    for key, expected in required.items():
        if data.get(key) != expected:
            raise ValueError(f"manifest {key}={data.get(key)!r}, expected {expected!r}")
    rows = data.get("rows")
    if not isinstance(rows, list) or len(rows) != data["deduplicated_problem_count"]:
        raise ValueError("manifest rows do not match the declared deduplicated count")
    ordered = sorted(rows, key=lambda row: row["source_row"])
    if ordered != rows:
        raise ValueError("manifest rows are not in retained source-row order")
    seen_hashes = set()
    for row in ordered:
        problem_hash = row.get("problem_sha256")
        if not isinstance(problem_hash, str) or len(problem_hash) != 64:
            raise ValueError("invalid problem SHA-256")
        if problem_hash in seen_hashes:
            raise ValueError("duplicate retained problem SHA-256")
        seen_hashes.add(problem_hash)
        if not isinstance(row.get("K"), int) or not 0 <= row["K"] <= N:
            raise ValueError("invalid per-problem K")
    indices = list(range(len(ordered)))
    random.Random(SEED).shuffle(indices)
    expected_split = {}
    for rank, index in enumerate(indices):
        expected_split[index] = "FIT" if rank < 4000 else "CAL" if rank < 8000 else "TEST"
    for index, row in enumerate(ordered):
        if row.get("split") != expected_split[index]:
            raise ValueError("stored split does not reproduce seeded problem-level split")
    return [row["K"] for row in ordered], indices


def replay(manifest_path):
    Ks, shuffled_indices = load_counts(manifest_path)
    fit_idx = shuffled_indices[:4000]
    cal_idx = shuffled_indices[4000:8000]
    test_idx = shuffled_indices[8000:]

    H = [0.0] * (N + 1)
    for index in fit_idx:
        H[Ks[index]] += 1.0
    H = [value / len(fit_idx) for value in H]

    cert = build_cert_table(H)
    fixed_flip = build_flip_fixed()
    adaptive = {
        alpha: {K: dp_adaptive_flip(K, cert, alpha) for K in range(N + 1)}
        for alpha in AGRID
    }

    out = {
        "n": len(Ks),
        "n_fit": len(fit_idx),
        "n_cal": len(cal_idx),
        "n_test": len(test_idx),
        "delta_cal": DELTA_CAL,
        "J_family": J_FAM,
        "d_test": D_TEST,
        "seed": SEED,
        "N": N,
    }

    cal = {}
    for k in KGRID:
        cal[("F", k)] = [fixed_flip[Ks[index]][k] for index in cal_idx]
    for alpha in AGRID:
        cal[("B", alpha)] = [adaptive[alpha][Ks[index]][0] for index in cal_idx]

    selection = {}
    for alpha in AGRID:
        row = {}
        fixed_eb = next(
            (k for k in KGRID if eb_ucb(cal[("F", k)], D_TEST) <= alpha),
            None,
        )
        row["FIXED_EB_k"] = fixed_eb
        row["FIXED_EB_cert"] = round(eb_ucb(cal[("F", fixed_eb)], D_TEST), 5) if fixed_eb else None
        fixed_hoef = next(
            (k for k in KGRID if hoef_ucb(cal[("F", k)], D_TEST) <= alpha),
            None,
        )
        row["FIXED_HOEF_k"] = fixed_hoef
        row["FIXED_HOEF_cert"] = round(hoef_ucb(cal[("F", fixed_hoef)], D_TEST), 5) if fixed_hoef else None
        bayesh_cert = eb_ucb(cal[("B", alpha)], D_TEST)
        row["BAYESH_cert"] = round(bayesh_cert, 5)
        row["BAYESH_ok"] = bool(bayesh_cert <= alpha)
        selection[str(alpha)] = row
    out["cal_selection"] = selection

    rng = random.Random(7)
    max_abs_difference = 0.0
    repetitions = 400
    for index in cal_idx[:200]:
        K = Ks[index]
        base = [1] * K + [0] * (N - K)
        flips = 0
        for _ in range(repetitions):
            order = base[:]
            rng.shuffle(order)
            count = 0
            for k in range(1, N + 1):
                count += order[k - 1]
                if k >= 3 and cert[k][count] <= 0.05:
                    flips += int(side(count, k) != side(K, N))
                    break
                if k == N:
                    flips += int(side(count, N) != side(K, N))
        mc = flips / repetitions
        dp = adaptive[0.05][K][0]
        max_abs_difference = max(max_abs_difference, abs(mc - dp))
    tolerance = 3 * sqrt(0.25 / repetitions)
    out["selfcheck_dp_vs_mc"] = {
        "max_abs_diff": round(max_abs_difference, 4),
        "tol_3sigma": round(tolerance, 4),
        "pass": bool(max_abs_difference <= tolerance),
    }
    if max_abs_difference > tolerance:
        raise AssertionError("DP versus Monte-Carlo self-check failed")

    rules = []
    for alpha in AGRID:
        row = selection[str(alpha)]
        if row["FIXED_EB_k"]:
            rules.append((f"FIXED_EB_a{alpha}", "F", row["FIXED_EB_k"], alpha))
        if row["FIXED_HOEF_k"]:
            rules.append((f"FIXED_HOEF_a{alpha}", "F", row["FIXED_HOEF_k"], alpha))
        if row["BAYESH_ok"]:
            rules.append((f"BAYESH_a{alpha}", "B", alpha, alpha))
    rules.append(("FULL32", "FULL", N, None))
    per_rule_delta = 0.05 / len(rules)

    test_readout = {}
    adaptive_ks = {}
    for name, kind, parameter, alpha in rules:
        flips = []
        stopping_counts = []
        for index in test_idx:
            K = Ks[index]
            if kind == "F":
                flips.append(fixed_flip[K][parameter])
                stopping_counts.append(float(parameter))
            elif kind == "B":
                flip, mean_k = adaptive[parameter][K]
                flips.append(flip)
                stopping_counts.append(mean_k)
            else:
                flips.append(0.0)
                stopping_counts.append(float(N))
        mean_flip, flip_radius = mean_ci(flips, per_rule_delta)
        mean_k, k_radius = mean_ci(stopping_counts, per_rule_delta)
        test_readout[name] = {
            "alpha": alpha,
            "realized_flip": round(mean_flip, 5),
            "flip_ci_radius": round(flip_radius, 5),
            "mean_k": round(mean_k, 3),
            "k_ci_radius": round(k_radius, 4),
            "saving_vs_full": round(1 - mean_k / N, 4),
        }
        if kind == "B":
            adaptive_ks[parameter] = stopping_counts
    out["test_readout"] = test_readout
    out["test_ci_bonferroni_rules"] = len(rules)

    gaps = {}
    for alpha in AGRID:
        row = selection[str(alpha)]
        if not (row["BAYESH_ok"] and row["FIXED_HOEF_k"]):
            continue
        bayesh_ks = adaptive_ks[alpha]
        differences = [float(row["FIXED_HOEF_k"]) - value for value in bayesh_ks]
        mean_difference, difference_radius = mean_ci(
            [value / N for value in differences],
            per_rule_delta,
        )
        gaps[str(alpha)] = {
            "bayesh_saving": test_readout[f"BAYESH_a{alpha}"]["saving_vs_full"],
            "hoef_saving": test_readout[f"FIXED_HOEF_a{alpha}"]["saving_vs_full"],
            "abs_gap_bayesh_minus_hoef": round(mean_difference, 4),
            "gap_ci_radius": round(difference_radius, 4),
            "rel_gain_vs_hoef": round(
                mean_difference / (1 - row["FIXED_HOEF_k"] / N),
                4,
            ) if row["FIXED_HOEF_k"] < N else None,
            "significant": bool(mean_difference - difference_radius > 0),
        }
    out["fair_gap_bayesh_vs_hoeffding"] = gaps
    return out


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--output", default="fit_cal_test_r469_replayed.json")
    parser.add_argument("--expected")
    args = parser.parse_args()
    result = replay(args.manifest)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=1)
    result_hash = sha256_file(args.output)
    print(json.dumps({"output": args.output, "sha256": result_hash}, sort_keys=True))
    if args.expected:
        expected = json.loads(Path(args.expected).read_text(encoding="utf-8"))
        if result != expected:
            raise SystemExit("semantic mismatch against expected result JSON")
        expected_hash = sha256_file(args.expected)
        if result_hash != expected_hash:
            raise SystemExit("semantic match but byte SHA-256 mismatch")
        print(json.dumps({"expected": args.expected, "sha256_match": True}, sort_keys=True))


if __name__ == "__main__":
    main()
