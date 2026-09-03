#!/usr/bin/env python3
"""Deterministic staged A11 same-endpoint comparator analysis.

The formal TEST aggregation is gated by immutable preflight, FIT, and CAL
artifacts.  No stage overwrites an existing formal artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from fractions import Fraction
from math import comb, log, sqrt
from pathlib import Path
from typing import Any, Iterable


BASE = Path(__file__).resolve().parent
CONFIG_PATH = BASE / "config.json"
PROTOCOL_PATH = BASE / "protocol.md"
SCHEMA_PATH = BASE / "output_schema.json"
RUNNER_PATH = Path(__file__).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def write_json_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def alpha_fraction(text: str) -> Fraction:
    return Fraction(text)


def side(count: int, total: int) -> int:
    return int(2 * count > total)


def hyper_pmf_fraction(N: int, K: int, k: int, x: int) -> Fraction:
    if not (0 <= K <= N and 0 <= k <= N and 0 <= x <= k):
        return Fraction(0, 1)
    if x < max(0, k - (N - K)) or x > min(k, K):
        return Fraction(0, 1)
    return Fraction(comb(K, x) * comb(N - K, k - x), comb(N, k))


def build_bayes_certificate_table(N: int, weights: list[int]) -> dict[int, list[Fraction]]:
    if len(weights) != N + 1 or any(weight < 0 for weight in weights) or sum(weights) <= 0:
        raise ValueError("Bayes weights must be nonnegative, nonzero, and have length N+1")
    table: dict[int, list[Fraction]] = {}
    for k in range(N + 1):
        row: list[Fraction] = []
        for x in range(k + 1):
            if k == N:
                row.append(Fraction(0, 1))
                continue
            prefix_side = side(x, k)
            denominator = Fraction(0, 1)
            numerator = Fraction(0, 1)
            for K, weight in enumerate(weights):
                likelihood = hyper_pmf_fraction(N, K, k, x)
                mass = weight * likelihood
                denominator += mass
                if side(K, N) != prefix_side:
                    numerator += mass
            row.append(numerator / denominator if denominator else Fraction(0, 1))
        table[k] = row
    return table


def ppr_current_mask(N: int, k: int, x: int, alpha: Fraction) -> int:
    """Theorem-2.1 PPR set for a beta-binomial(1,1) working prior."""
    threshold = alpha / (k + 1)
    mask = 0
    for K in range(N + 1):
        if hyper_pmf_fraction(N, K, k, x) > threshold:
            mask |= 1 << K
    return mask


def mask_decision(mask: int, N: int) -> int | None:
    if mask == 0:
        return None
    negative_mask = (1 << (N // 2 + 1)) - 1
    full_mask = (1 << (N + 1)) - 1
    positive_mask = full_mask ^ negative_mask
    if mask & positive_mask == 0:
        return 0
    if mask & negative_mask == 0:
        return 1
    return None


def transition_probabilities(N: int, K: int, k: int, x: int) -> tuple[float, float]:
    pass_probability = (K - x) / (N - k)
    fail_probability = 1.0 - pass_probability
    if pass_probability < -1e-15 or fail_probability < -1e-15:
        raise AssertionError("invalid hypergeometric transition")
    return max(0.0, fail_probability), max(0.0, pass_probability)


def evaluate_bayes_rule(
    N: int,
    K: int,
    certificate: dict[int, list[Fraction]],
    alpha: Fraction,
    min_stop_k: int,
) -> dict[str, float]:
    full_side = side(K, N)
    reach: dict[int, float] = {0: 1.0}
    flip = 0.0
    expected_k = 0.0
    terminal_probability = 0.0
    for k in range(N + 1):
        nxt: dict[int, float] = defaultdict(float)
        for x, probability in reach.items():
            if k == N:
                expected_k += probability * N
                terminal_probability += probability
                continue
            if k >= min_stop_k and certificate[k][x] <= alpha:
                expected_k += probability * k
                flip += probability * int(side(x, k) != full_side)
                continue
            fail_probability, pass_probability = transition_probabilities(N, K, k, x)
            if fail_probability:
                nxt[x] += probability * fail_probability
            if pass_probability:
                nxt[x + 1] += probability * pass_probability
        reach = dict(nxt)
    return {
        "flip_probability": flip,
        "expected_k": expected_k,
        "terminal_probability": terminal_probability,
    }


def build_ppr_masks(N: int, alpha: Fraction) -> dict[int, list[int]]:
    return {
        k: [ppr_current_mask(N, k, x, alpha) for x in range(k + 1)]
        for k in range(N + 1)
    }


def evaluate_wor_ppr_rule(
    N: int,
    K: int,
    masks: dict[int, list[int]],
    min_stop_k: int,
) -> dict[str, float]:
    full_side = side(K, N)
    initial_mask = masks[0][0]
    reach: dict[tuple[int, int], float] = {(0, initial_mask): 1.0}
    flip = 0.0
    expected_k = 0.0
    terminal_probability = 0.0
    for k in range(N + 1):
        nxt: dict[tuple[int, int], float] = defaultdict(float)
        for (x, running_mask), probability in reach.items():
            if k == N:
                expected_k += probability * N
                terminal_probability += probability
                continue
            decision = mask_decision(running_mask, N) if k >= min_stop_k else None
            if decision is not None:
                expected_k += probability * k
                flip += probability * int(decision != full_side)
                continue
            fail_probability, pass_probability = transition_probabilities(N, K, k, x)
            if fail_probability:
                new_mask = running_mask & masks[k + 1][x]
                nxt[(x, new_mask)] += probability * fail_probability
            if pass_probability:
                new_mask = running_mask & masks[k + 1][x + 1]
                nxt[(x + 1, new_mask)] += probability * pass_probability
        reach = dict(nxt)
    return {
        "flip_probability": flip,
        "expected_k": expected_k,
        "terminal_probability": terminal_probability,
    }


def ppr_running_miscoverage_probability(N: int, K: int, masks: dict[int, list[int]]) -> float:
    """Probability that some current PPR set excludes the fixed true K."""
    reach: dict[tuple[int, int], float] = {(0, masks[0][0]): 1.0}
    for k in range(N):
        nxt: dict[tuple[int, int], float] = defaultdict(float)
        for (x, running_mask), probability in reach.items():
            fail_probability, pass_probability = transition_probabilities(N, K, k, x)
            if fail_probability:
                new_mask = running_mask & masks[k + 1][x]
                nxt[(x, new_mask)] += probability * fail_probability
            if pass_probability:
                new_mask = running_mask & masks[k + 1][x + 1]
                nxt[(x + 1, new_mask)] += probability * pass_probability
        reach = dict(nxt)
    truth_bit = 1 << K
    return sum(probability for (_, running_mask), probability in reach.items() if not running_mask & truth_bit)


def fraction_table_fingerprint(table: dict[int, list[Fraction]]) -> str:
    serial = {
        str(k): [f"{value.numerator}/{value.denominator}" for value in row]
        for k, row in table.items()
    }
    return canonical_sha256(serial)


def masks_fingerprint(masks_by_alpha: dict[str, dict[int, list[int]]]) -> str:
    serial = {
        alpha: {str(k): row for k, row in table.items()}
        for alpha, table in masks_by_alpha.items()
    }
    return canonical_sha256(serial)


def compute_policy_tables(config: dict[str, Any], fit_histogram: list[int]) -> dict[str, Any]:
    N = config["input"]["N"]
    min_stop_k = config["endpoint"]["min_stop_k"]
    alpha_texts = config["alphas"]
    bayes_h_cert = build_bayes_certificate_table(N, fit_histogram)
    bayes_unif_cert = build_bayes_certificate_table(N, [1] * (N + 1))
    ppr_masks = {text: build_ppr_masks(N, alpha_fraction(text)) for text in alpha_texts}
    methods: dict[str, dict[str, Any]] = {
        "BAYES-H": {},
        "BAYES-UNIF": {},
        "WOR-PPR-CS": {},
    }
    cs_coverage: dict[str, Any] = {}
    for alpha_text in alpha_texts:
        alpha = alpha_fraction(alpha_text)
        for method, cert in (("BAYES-H", bayes_h_cert), ("BAYES-UNIF", bayes_unif_cert)):
            methods[method][alpha_text] = [
                {"K": K, **evaluate_bayes_rule(N, K, cert, alpha, min_stop_k)}
                for K in range(N + 1)
            ]
        wor_rows = []
        max_miscoverage = 0.0
        max_flip = 0.0
        max_flip_minus_miscoverage = -1.0
        for K in range(N + 1):
            row = evaluate_wor_ppr_rule(N, K, ppr_masks[alpha_text], min_stop_k)
            miscoverage = ppr_running_miscoverage_probability(N, K, ppr_masks[alpha_text])
            row = {"K": K, **row, "cs_running_miscoverage_probability": miscoverage}
            wor_rows.append(row)
            max_miscoverage = max(max_miscoverage, miscoverage)
            max_flip = max(max_flip, row["flip_probability"])
            max_flip_minus_miscoverage = max(
                max_flip_minus_miscoverage,
                row["flip_probability"] - miscoverage,
            )
        methods["WOR-PPR-CS"][alpha_text] = wor_rows
        cs_coverage[alpha_text] = {
            "max_fixed_K_decision_error": max_flip,
            "max_fixed_K_running_miscoverage": max_miscoverage,
            "max_decision_error_minus_miscoverage": max_flip_minus_miscoverage,
            "alpha": float(alpha),
            "pass": bool(max_miscoverage <= float(alpha) + 1e-12 and max_flip_minus_miscoverage <= 1e-12),
        }
    if not all(row["pass"] for row in cs_coverage.values()):
        raise AssertionError("WOR-PPR-CS fixed-K coverage audit failed")
    return {
        "methods": methods,
        "policy_fingerprints": {
            "BAYES-H_certificate_table": fraction_table_fingerprint(bayes_h_cert),
            "BAYES-UNIF_certificate_table": fraction_table_fingerprint(bayes_unif_cert),
            "WOR-PPR-CS_current_sets_all_alphas": masks_fingerprint(ppr_masks),
        },
        "wor_ppr_cs_fixed_K_audit": cs_coverage,
    }


def eb_ucb(values: list[float], delta: float) -> float:
    m = len(values)
    mean = sum(values) / m
    variance = sum((value - mean) ** 2 for value in values) / (m - 1) if m > 1 else 0.0
    return mean + sqrt(2 * variance * log(4 / delta) / m) + 7 * log(4 / delta) / (3 * (m - 1))


def sample_variance(values: list[float]) -> float:
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / (len(values) - 1)


def hoeffding_radius(delta: float, count: int, value_range: float) -> float:
    return value_range * sqrt(log(2 / delta) / (2 * count))


def config_hashes() -> dict[str, str]:
    return {
        "config_sha256": sha256_file(CONFIG_PATH),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "output_schema_sha256": sha256_file(SCHEMA_PATH),
        "runner_sha256": sha256_file(RUNNER_PATH),
    }


def load_and_validate_config() -> dict[str, Any]:
    config = load_json(CONFIG_PATH)
    if config.get("experiment_id") != "A11-WOR-CS-COMPARISON-OMR-R1":
        raise ValueError("unexpected experiment id")
    if config["input"]["N"] != 32:
        raise ValueError("this frozen protocol requires N=32")
    if config["alphas"] != ["0.10", "0.05", "0.02", "0.01"]:
        raise ValueError("alpha grid differs from the preregistration")
    if config["primary_alpha"] != "0.05":
        raise ValueError("primary alpha differs from the preregistration")
    expected_methods = {"BAYES-H", "BAYES-UNIF", "WOR-PPR-CS"}
    if set(config["methods"]) != expected_methods:
        raise ValueError("method family differs from preregistration")
    if config["calibration"]["family_size"] != len(expected_methods) * len(config["alphas"]):
        raise ValueError("CAL family size mismatch")
    if config["test_readout"]["paired_gap_family_size"] != 8:
        raise ValueError("paired-gap family size mismatch")
    output_dir = (BASE / config["outputs"]["directory"]).resolve()
    if output_dir != (BASE / "outputs").resolve() or BASE.resolve() not in output_dir.parents:
        raise ValueError("unsafe output directory")
    return config


def input_manifest_path(config: dict[str, Any]) -> Path:
    path = (BASE / config["input"]["relative_path"]).resolve()
    expected_parent = (BASE / "../../evidence/repro_bundle_round4").resolve()
    if path.parent != expected_parent or path.name != "omr_problem_manifest.json":
        raise ValueError("input path escapes the frozen reviewer-safe bundle")
    return path


def output_paths(config: dict[str, Any]) -> dict[str, Path]:
    directory = (BASE / config["outputs"]["directory"]).resolve()
    return {
        key: directory / config["outputs"][key]
        for key in ("preflight", "fit_lock", "cal_lock", "formal_result", "verification")
    }


def validate_manifest(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, list[int]], dict[str, Any]]:
    path = input_manifest_path(config)
    actual_hash = sha256_file(path)
    if actual_hash != config["input"]["sha256"]:
        raise ValueError("input manifest SHA-256 mismatch")
    manifest = load_json(path)
    for key, expected in (
        ("K", config["input"]["N"]),
        ("split_seed", config["input"]["split_seed"]),
        ("split_counts", config["input"]["split_counts"]),
    ):
        if manifest.get(key) != expected:
            raise ValueError(f"manifest {key} mismatch")
    rows = manifest.get("rows")
    if not isinstance(rows, list) or len(rows) != sum(config["input"]["split_counts"].values()):
        raise ValueError("manifest row count mismatch")
    if rows != sorted(rows, key=lambda row: row["source_row"]):
        raise ValueError("manifest rows are not in retained source order")
    seen_problem_hashes: set[str] = set()
    seen_source_rows: set[int] = set()
    N = config["input"]["N"]
    for row in rows:
        problem_hash = row.get("problem_sha256")
        source_row = row.get("source_row")
        if not isinstance(problem_hash, str) or len(problem_hash) != 64:
            raise ValueError("invalid problem hash")
        if problem_hash in seen_problem_hashes:
            raise ValueError("duplicate retained problem hash")
        if not isinstance(source_row, int) or source_row in seen_source_rows:
            raise ValueError("invalid or duplicate retained source row")
        if not isinstance(row.get("K"), int) or not 0 <= row["K"] <= N:
            raise ValueError("invalid count K")
        seen_problem_hashes.add(problem_hash)
        seen_source_rows.add(source_row)
    indices = list(range(len(rows)))
    random.Random(config["input"]["split_seed"]).shuffle(indices)
    sizes = config["input"]["split_counts"]
    split_indices = {
        "FIT": indices[: sizes["FIT"]],
        "CAL": indices[sizes["FIT"] : sizes["FIT"] + sizes["CAL"]],
        "TEST": indices[sizes["FIT"] + sizes["CAL"] :],
    }
    for split, split_rows in split_indices.items():
        for index in split_rows:
            if rows[index].get("split") != split:
                raise ValueError("stored split does not reproduce seeded split")
    checks = {
        "manifest_sha256": actual_hash,
        "row_count": len(rows),
        "unique_problem_hash_count": len(seen_problem_hashes),
        "unique_source_row_count": len(seen_source_rows),
        "split_reconstruction": "pass",
        "duplicate_protection": "pass",
    }
    return rows, split_indices, checks


def split_fingerprint(rows: list[dict[str, Any]], indices: Iterable[int]) -> str:
    return canonical_sha256(
        [
            {
                "problem_sha256": rows[index]["problem_sha256"],
                "source_row": rows[index]["source_row"],
                "K": rows[index]["K"],
            }
            for index in indices
        ]
    )


def common_lock_assertions(lock: dict[str, Any], expected_stage: str) -> None:
    config = load_and_validate_config()
    if lock.get("experiment_id") != config["experiment_id"] or lock.get("stage") != expected_stage:
        raise ValueError(f"invalid {expected_stage} lock identity")
    if lock.get("frozen_file_hashes") != config_hashes():
        raise ValueError(f"frozen files changed after {expected_stage}")


def internal_unit_tests(config: dict[str, Any]) -> dict[str, Any]:
    N = config["input"]["N"]
    alpha_texts = config["alphas"]
    hyper_cells = 0
    uniform_marginal_cells = 0
    for K in range(N + 1):
        for k in range(N + 1):
            total = sum(hyper_pmf_fraction(N, K, k, x) for x in range(k + 1))
            if total != 1:
                raise AssertionError(f"hypergeometric normalization failed at K={K}, k={k}")
            hyper_cells += 1
    for k in range(N + 1):
        for x in range(k + 1):
            marginal = sum(hyper_pmf_fraction(N, K, k, x) for K in range(N + 1)) / (N + 1)
            if marginal != Fraction(1, k + 1):
                raise AssertionError(f"uniform-prior marginal identity failed at k={k}, x={x}")
            uniform_marginal_cells += 1
    ppr_membership_cells = 0
    for alpha_text in alpha_texts:
        alpha = alpha_fraction(alpha_text)
        for k in range(N + 1):
            for x in range(k + 1):
                mask = ppr_current_mask(N, k, x, alpha)
                likelihood_sum = sum(hyper_pmf_fraction(N, K, k, x) for K in range(N + 1))
                for K in range(N + 1):
                    likelihood = hyper_pmf_fraction(N, K, k, x)
                    posterior = likelihood / likelihood_sum if likelihood_sum else Fraction(0, 1)
                    ratio = Fraction(1, N + 1) / posterior if posterior else None
                    direct_member = ratio is not None and ratio < 1 / alpha
                    if bool(mask & (1 << K)) != direct_member:
                        raise AssertionError("PPR ratio and hypergeometric threshold disagree")
                    ppr_membership_cells += 1
        for x in range(N + 1):
            if ppr_current_mask(N, N, x, alpha) != 1 << x:
                raise AssertionError("terminal PPR set is not the observed singleton")
    synthetic_rows = [
        {"problem_sha256": "0" * 64, "source_row": 0, "K": 0},
        {"problem_sha256": "1" * 64, "source_row": 1, "K": N},
    ]
    if split_fingerprint(synthetic_rows, [0, 1]) == split_fingerprint(synthetic_rows, [1, 0]):
        raise AssertionError("split fingerprint is not order-sensitive")
    zero_histogram = [0] * (N + 1)
    zero_histogram[0] = 1
    zero_histogram[N] = 1
    policy_audit = compute_policy_tables(config, zero_histogram)["wor_ppr_cs_fixed_K_audit"]
    return {
        "status": "pass",
        "hypergeometric_normalization_cells": hyper_cells,
        "uniform_prior_marginal_identity_cells": uniform_marginal_cells,
        "ppr_ratio_equivalence_cells": ppr_membership_cells,
        "terminal_singleton_cells": (N + 1) * len(alpha_texts),
        "synthetic_duplicate_and_order_guard": "pass",
        "wor_ppr_cs_fixed_K_coverage": policy_audit,
    }


def stage_preflight() -> dict[str, Any]:
    config = load_and_validate_config()
    paths = output_paths(config)
    manifest_path = input_manifest_path(config)
    actual_hash = sha256_file(manifest_path)
    if actual_hash != config["input"]["sha256"]:
        raise ValueError("input manifest SHA-256 mismatch")
    report = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "stage": "preflight",
        "status": "pass",
        "frozen_file_hashes": config_hashes(),
        "input_manifest": {
            "path": str(manifest_path),
            "sha256": actual_hash,
            "bytes": manifest_path.stat().st_size,
            "content_parsed": False,
            "test_aggregated": False,
        },
        "unit_tests": internal_unit_tests(config),
        "configuration_checks": {
            "family_size": config["calibration"]["family_size"],
            "method_count": len(config["methods"]),
            "alpha_count": len(config["alphas"]),
            "primary_alpha": config["primary_alpha"],
            "analysis_rng": config["randomness"]["analysis_rng"],
            "output_path_safety": "pass",
            "overwrite_policy": config["overwrite_policy"],
        },
    }
    write_json_exclusive(paths["preflight"], report)
    return report


def stage_fit() -> dict[str, Any]:
    config = load_and_validate_config()
    paths = output_paths(config)
    preflight = load_json(paths["preflight"])
    common_lock_assertions(preflight, "preflight")
    if preflight.get("status") != "pass":
        raise ValueError("preflight did not pass")
    rows, splits, manifest_checks = validate_manifest(config)
    N = config["input"]["N"]
    fit_histogram = [0] * (N + 1)
    for index in splits["FIT"]:
        fit_histogram[rows[index]["K"]] += 1
    if sum(fit_histogram) != config["input"]["split_counts"]["FIT"]:
        raise AssertionError("FIT histogram count mismatch")
    policy_tables = compute_policy_tables(config, fit_histogram)
    lock = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "stage": "fit",
        "status": "pass",
        "frozen_file_hashes": config_hashes(),
        "preflight_sha256": sha256_file(paths["preflight"]),
        "manifest_checks": manifest_checks,
        "fit": {
            "count": len(splits["FIT"]),
            "split_fingerprint": split_fingerprint(rows, splits["FIT"]),
            "count_histogram_K_0_to_N": fit_histogram,
        },
        "policy_tables": policy_tables,
        "cal_aggregated": False,
        "test_aggregated": False,
    }
    write_json_exclusive(paths["fit_lock"], lock)
    return lock


def rows_by_K(policy_tables: dict[str, Any], method: str, alpha: str) -> list[dict[str, Any]]:
    rows = policy_tables["methods"][method][alpha]
    if [row["K"] for row in rows] != list(range(len(rows))):
        raise ValueError("per-K policy table is malformed")
    return rows


def cal_metrics(config: dict[str, Any], rows: list[dict[str, Any]], indices: list[int], policy_tables: dict[str, Any]) -> dict[str, Any]:
    family_size = config["calibration"]["family_size"]
    delta = 0.05 / family_size
    result: dict[str, Any] = {}
    for method in config["methods"]:
        result[method] = {}
        for alpha_text in config["alphas"]:
            per_K = rows_by_K(policy_tables, method, alpha_text)
            flips = [per_K[rows[index]["K"]]["flip_probability"] for index in indices]
            stopping = [per_K[rows[index]["K"]]["expected_k"] for index in indices]
            ucb = eb_ucb(flips, delta)
            result[method][alpha_text] = {
                "n": len(indices),
                "mean_flip": sum(flips) / len(flips),
                "sample_variance_flip": sample_variance(flips),
                "empirical_bernstein_ucb": ucb,
                "per_rule_delta": delta,
                "passes_named_alpha": bool(ucb <= float(alpha_fraction(alpha_text))),
                "mean_k": sum(stopping) / len(stopping),
                "count_reduction": 1.0 - sum(stopping) / len(stopping) / config["input"]["N"],
            }
    return result


def stage_cal() -> dict[str, Any]:
    config = load_and_validate_config()
    paths = output_paths(config)
    preflight = load_json(paths["preflight"])
    fit_lock = load_json(paths["fit_lock"])
    common_lock_assertions(preflight, "preflight")
    common_lock_assertions(fit_lock, "fit")
    if fit_lock.get("preflight_sha256") != sha256_file(paths["preflight"]):
        raise ValueError("preflight artifact changed before CAL")
    rows, splits, manifest_checks = validate_manifest(config)
    if split_fingerprint(rows, splits["FIT"]) != fit_lock["fit"]["split_fingerprint"]:
        raise ValueError("FIT split changed before CAL")
    metrics = cal_metrics(config, rows, splits["CAL"], fit_lock["policy_tables"])
    lock = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "stage": "cal",
        "status": "pass",
        "frozen_file_hashes": config_hashes(),
        "preflight_sha256": sha256_file(paths["preflight"]),
        "fit_lock_sha256": sha256_file(paths["fit_lock"]),
        "manifest_checks": manifest_checks,
        "cal": {
            "count": len(splits["CAL"]),
            "split_fingerprint": split_fingerprint(rows, splits["CAL"]),
            "family_size": config["calibration"]["family_size"],
            "row_metrics": metrics,
        },
        "test_aggregated": False,
    }
    write_json_exclusive(paths["cal_lock"], lock)
    return lock


def test_metrics(config: dict[str, Any], rows: list[dict[str, Any]], indices: list[int], policy_tables: dict[str, Any]) -> dict[str, Any]:
    row_delta = 0.05 / config["test_readout"]["row_ci_family_size"]
    N = config["input"]["N"]
    min_stop = config["endpoint"]["min_stop_k"]
    result: dict[str, Any] = {}
    for method in config["methods"]:
        result[method] = {}
        for alpha_text in config["alphas"]:
            per_K = rows_by_K(policy_tables, method, alpha_text)
            flips = [per_K[rows[index]["K"]]["flip_probability"] for index in indices]
            stopping = [per_K[rows[index]["K"]]["expected_k"] for index in indices]
            mean_flip = sum(flips) / len(flips)
            mean_k = sum(stopping) / len(stopping)
            result[method][alpha_text] = {
                "n": len(indices),
                "mean_flip": mean_flip,
                "mean_flip_hoeffding_radius": hoeffding_radius(row_delta, len(indices), 1.0),
                "mean_k": mean_k,
                "mean_k_hoeffding_radius": N * hoeffding_radius(row_delta, len(indices), (N - min_stop) / N),
                "count_reduction": 1.0 - mean_k / N,
                "count_reduction_hoeffding_radius": hoeffding_radius(
                    row_delta, len(indices), (N - min_stop) / N
                ),
                "row_delta": row_delta,
            }
    return result


def paired_metrics(
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    indices: list[int],
    policy_tables: dict[str, Any],
) -> dict[str, Any]:
    N = config["input"]["N"]
    min_stop = config["endpoint"]["min_stop_k"]
    delta = 0.05 / config["test_readout"]["paired_gap_family_size"]
    output: dict[str, Any] = {}
    for comparator in config["test_readout"]["paired_comparators"]:
        output[comparator] = {}
        for alpha_text in config["alphas"]:
            bayes_rows = rows_by_K(policy_tables, "BAYES-H", alpha_text)
            comparator_rows = rows_by_K(policy_tables, comparator, alpha_text)
            count_gaps = []
            flip_gaps = []
            for index in indices:
                K = rows[index]["K"]
                count_gaps.append((comparator_rows[K]["expected_k"] - bayes_rows[K]["expected_k"]) / N)
                flip_gaps.append(bayes_rows[K]["flip_probability"] - comparator_rows[K]["flip_probability"])
            count_mean = sum(count_gaps) / len(count_gaps)
            flip_mean = sum(flip_gaps) / len(flip_gaps)
            count_radius = hoeffding_radius(delta, len(indices), 2 * (N - min_stop) / N)
            flip_radius = hoeffding_radius(delta, len(indices), 2.0)
            output[comparator][alpha_text] = {
                "n": len(indices),
                "bayes_h_minus_comparator_count_reduction": count_mean,
                "count_reduction_gap_hoeffding_radius": count_radius,
                "positive_count_gap_favors_bayes_h": True,
                "count_gap_interval_excludes_zero": bool(abs(count_mean) > count_radius),
                "bayes_h_minus_comparator_flip": flip_mean,
                "flip_gap_hoeffding_radius": flip_radius,
                "negative_flip_gap_favors_bayes_h": True,
                "flip_gap_interval_excludes_zero": bool(abs(flip_mean) > flip_radius),
                "paired_gap_delta": delta,
            }
    return output


def build_formal_result(
    config: dict[str, Any],
    paths: dict[str, Path],
    preflight: dict[str, Any],
    fit_lock: dict[str, Any],
    cal_lock: dict[str, Any],
    rows: list[dict[str, Any]],
    splits: dict[str, list[int]],
    manifest_checks: dict[str, Any],
) -> dict[str, Any]:
    test_readout = test_metrics(config, rows, splits["TEST"], fit_lock["policy_tables"])
    paired = paired_metrics(config, rows, splits["TEST"], fit_lock["policy_tables"])
    primary_alpha = config["primary_alpha"]
    primary = {
        "alpha": primary_alpha,
        "BAYES-H": test_readout["BAYES-H"][primary_alpha],
        "comparators": {
            comparator: {
                "test": test_readout[comparator][primary_alpha],
                "paired_vs_BAYES-H": paired[comparator][primary_alpha],
                "cal": cal_lock["cal"]["row_metrics"][comparator][primary_alpha],
            }
            for comparator in config["test_readout"]["paired_comparators"]
        },
        "BAYES-H_cal": cal_lock["cal"]["row_metrics"]["BAYES-H"][primary_alpha],
    }
    return {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": "complete_integrity_valid",
        "provenance": {
            **config_hashes(),
            "manifest_sha256": manifest_checks["manifest_sha256"],
            "preflight_sha256": sha256_file(paths["preflight"]),
            "fit_lock_sha256": sha256_file(paths["fit_lock"]),
            "cal_lock_sha256": sha256_file(paths["cal_lock"]),
            "analysis_rng": config["randomness"]["analysis_rng"],
            "derived_data_boundary": config["input"]["derived_data_boundary"],
        },
        "split_counts": config["input"]["split_counts"],
        "calibration": {
            "family_size": config["calibration"]["family_size"],
            "row_metrics": cal_lock["cal"]["row_metrics"],
            "test_used_for_selection": False,
        },
        "test_readout": {
            "status": config["test_readout"]["status"],
            "rows": test_readout,
            "paired_BAYES-H_gaps": paired,
        },
        "primary_comparison": primary,
        "integrity_checks": {
            "manifest": manifest_checks,
            "preflight_status": preflight["status"],
            "fit_status": fit_lock["status"],
            "cal_status": cal_lock["status"],
            "wor_ppr_cs_fixed_K_audit": fit_lock["policy_tables"]["wor_ppr_cs_fixed_K_audit"],
            "test_split_fingerprint": split_fingerprint(rows, splits["TEST"]),
            "test_aggregated_after_fit_and_cal_locks": True,
            "test_used_for_method_or_alpha_selection": False,
        },
        "interpretation_boundary": [
            "Conditional on the anonymous derived count manifest; not a raw-data reconstruction.",
            "Exact count-exchangeable random-prefix replay; not observed chronological online stopping.",
            "Replay disagreement with the full-count binary side; not gold answer correctness.",
            "Expected replay prefix count; not measured tokens, latency, cancellation, or monetary cost.",
            "Any efficiency statement is empirical for this frozen carrier and operating point, not theorem-level dominance.",
        ],
    }


def validate_formal_schema(result: dict[str, Any]) -> None:
    schema = load_json(SCHEMA_PATH)
    required = set(schema["required"])
    if set(result) != required:
        raise ValueError(f"formal output fields differ from schema: {set(result) ^ required}")
    if result["schema_version"] != "1.0" or result["experiment_id"] != "A11-WOR-CS-COMPARISON-OMR-R1":
        raise ValueError("formal output identity mismatch")
    if result["status"] not in schema["properties"]["status"]["enum"]:
        raise ValueError("formal output status is invalid")


def stage_test() -> dict[str, Any]:
    config = load_and_validate_config()
    paths = output_paths(config)
    preflight = load_json(paths["preflight"])
    fit_lock = load_json(paths["fit_lock"])
    cal_lock = load_json(paths["cal_lock"])
    common_lock_assertions(preflight, "preflight")
    common_lock_assertions(fit_lock, "fit")
    common_lock_assertions(cal_lock, "cal")
    if fit_lock["preflight_sha256"] != sha256_file(paths["preflight"]):
        raise ValueError("preflight changed before TEST")
    if cal_lock["preflight_sha256"] != sha256_file(paths["preflight"]):
        raise ValueError("preflight changed before TEST")
    if cal_lock["fit_lock_sha256"] != sha256_file(paths["fit_lock"]):
        raise ValueError("FIT lock changed before TEST")
    if preflight["input_manifest"]["sha256"] != config["input"]["sha256"]:
        raise ValueError("manifest identity changed since preflight")
    rows, splits, manifest_checks = validate_manifest(config)
    if split_fingerprint(rows, splits["FIT"]) != fit_lock["fit"]["split_fingerprint"]:
        raise ValueError("FIT split changed before TEST")
    if split_fingerprint(rows, splits["CAL"]) != cal_lock["cal"]["split_fingerprint"]:
        raise ValueError("CAL split changed before TEST")
    result = build_formal_result(
        config, paths, preflight, fit_lock, cal_lock, rows, splits, manifest_checks
    )
    validate_formal_schema(result)
    write_json_exclusive(paths["formal_result"], result)
    return result


def stage_verify() -> dict[str, Any]:
    """Verify locks/schema/result without a second aggregation of TEST rows."""
    config = load_and_validate_config()
    paths = output_paths(config)
    preflight = load_json(paths["preflight"])
    fit_lock = load_json(paths["fit_lock"])
    cal_lock = load_json(paths["cal_lock"])
    result = load_json(paths["formal_result"])
    common_lock_assertions(preflight, "preflight")
    common_lock_assertions(fit_lock, "fit")
    common_lock_assertions(cal_lock, "cal")
    validate_formal_schema(result)
    if result["provenance"]["preflight_sha256"] != sha256_file(paths["preflight"]):
        raise ValueError("formal result points to a different preflight")
    if result["provenance"]["fit_lock_sha256"] != sha256_file(paths["fit_lock"]):
        raise ValueError("formal result points to a different FIT lock")
    if result["provenance"]["cal_lock_sha256"] != sha256_file(paths["cal_lock"]):
        raise ValueError("formal result points to a different CAL lock")
    if result["provenance"]["manifest_sha256"] != sha256_file(input_manifest_path(config)):
        raise ValueError("formal result points to a changed manifest")
    if not all(
        row["pass"] for row in result["integrity_checks"]["wor_ppr_cs_fixed_K_audit"].values()
    ):
        raise ValueError("formal result contains a failed fixed-K CS audit")
    report = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "stage": "verification",
        "status": "pass",
        "formal_result_sha256": sha256_file(paths["formal_result"]),
        "formal_result_canonical_sha256": canonical_sha256(result),
        "checks": {
            "frozen_file_hashes": "pass",
            "stage_chain_hashes": "pass",
            "input_manifest_hash_without_content_reaggregation": "pass",
            "output_schema": "pass",
            "fixed_K_CS_audit": "pass",
            "second_TEST_aggregation_performed": False,
        },
    }
    write_json_exclusive(paths["verification"], report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("preflight", "fit", "cal", "test", "verify"))
    args = parser.parse_args()
    stage_fn = {
        "preflight": stage_preflight,
        "fit": stage_fit,
        "cal": stage_cal,
        "test": stage_test,
        "verify": stage_verify,
    }[args.stage]
    result = stage_fn()
    config = load_and_validate_config()
    path = output_paths(config)["verification" if args.stage == "verify" else "formal_result" if args.stage == "test" else "fit_lock" if args.stage == "fit" else "cal_lock" if args.stage == "cal" else "preflight"]
    print(json.dumps({"stage": args.stage, "status": result["status"], "output": str(path), "sha256": sha256_file(path)}, sort_keys=True))


if __name__ == "__main__":
    main()

