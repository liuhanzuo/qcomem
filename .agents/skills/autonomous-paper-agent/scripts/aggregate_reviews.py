#!/usr/bin/env python3
"""Aggregate independent paper-review JSON files into robust panel statistics."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

DIMENSIONS = (
    "soundness",
    "presentation",
    "contribution",
)

ALLOWED_RATINGS = {2, 4, 6, 8, 10}
RATING_TO_RECOMMENDATION = {
    2: "reject",
    4: "marginally_below",
    6: "marginally_above",
    8: "accept",
    10: "strong_accept",
}

REQUIRED_TOP_LEVEL = {
    "reviewer_id",
    "round",
    "snapshot_sha256",
    "role",
    "overall_score",
    "confidence",
    "recommendation",
    "dimension_scores",
    "dimension_justifications",
    "questions",
    "issues",
    "ethics_flag",
    "ethics_concerns",
    "llm_usage_disclosure",
}


def percentile(values: list[float], fraction: float) -> float:
    """Linear-interpolated percentile using inclusive endpoints."""
    if not values:
        raise ValueError("Cannot compute percentile of an empty list")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def validate_review(data: dict[str, Any], path: Path) -> list[str]:
    errors: list[str] = []
    missing = REQUIRED_TOP_LEVEL - data.keys()
    if missing:
        errors.append(f"missing fields: {sorted(missing)}")
        return errors

    score = data.get("overall_score")
    if isinstance(score, bool) or not isinstance(score, int) or score not in ALLOWED_RATINGS:
        errors.append("overall_score must be one of the ICLR ratings [2, 4, 6, 8, 10]")
    elif data.get("recommendation") != RATING_TO_RECOMMENDATION[score]:
        errors.append("recommendation must match the ICLR overall_score label")

    confidence = data.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, int) or not 1 <= confidence <= 5:
        errors.append("confidence must be an integer in [1, 5]")

    dimensions = data.get("dimension_scores")
    if not isinstance(dimensions, dict):
        errors.append("dimension_scores must be an object")
    else:
        for dimension in DIMENSIONS:
            value = dimensions.get(dimension)
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 4:
                errors.append(f"dimension_scores.{dimension} must be an integer in [1, 4]")

    justifications = data.get("dimension_justifications")
    if not isinstance(justifications, dict):
        errors.append("dimension_justifications must be an object")
    else:
        for dimension in DIMENSIONS:
            value = justifications.get(dimension)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"dimension_justifications.{dimension} must be non-empty")

    if not isinstance(data.get("questions"), list):
        errors.append("questions must be an array")
    if not isinstance(data.get("ethics_flag"), bool):
        errors.append("ethics_flag must be boolean")
    if not isinstance(data.get("ethics_concerns"), str):
        errors.append("ethics_concerns must be a string")
    if not isinstance(data.get("llm_usage_disclosure"), str) or not data["llm_usage_disclosure"].strip():
        errors.append("llm_usage_disclosure must be non-empty")

    issues = data.get("issues")
    if not isinstance(issues, list):
        errors.append("issues must be an array")
    else:
        for index, issue in enumerate(issues):
            if not isinstance(issue, dict):
                errors.append(f"issues[{index}] must be an object")
                continue
            if issue.get("severity") not in {"critical", "major", "minor"}:
                errors.append(f"issues[{index}].severity is invalid")

    return [f"{path.name}: {error}" for error in errors]


def load_reviews(directory: Path) -> tuple[list[dict[str, Any]], list[str]]:
    reviews: list[dict[str, Any]] = []
    errors: list[str] = []
    for path in sorted(directory.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{path.name}: invalid JSON: {exc}")
            continue
        if not isinstance(data, dict):
            errors.append(f"{path.name}: top-level JSON must be an object")
            continue
        file_errors = validate_review(data, path)
        if file_errors:
            errors.extend(file_errors)
            continue
        reviews.append(data)
    return reviews, errors


def aggregate(reviews: list[dict[str, Any]]) -> dict[str, Any]:
    rounds = {int(review["round"]) for review in reviews}
    snapshots = {str(review["snapshot_sha256"]) for review in reviews}
    if len(rounds) != 1:
        raise ValueError(f"Reviews span multiple rounds: {sorted(rounds)}")
    if len(snapshots) != 1:
        raise ValueError("Reviews refer to multiple snapshot hashes")

    scores = [float(review["overall_score"]) for review in reviews]
    confidences = [int(review["confidence"]) for review in reviews]
    dimension_medians = {
        dimension: statistics.median(float(review["dimension_scores"][dimension]) for review in reviews)
        for dimension in DIMENSIONS
    }

    severity_counts: Counter[str] = Counter()
    dimension_issue_counts: Counter[str] = Counter()
    for review in reviews:
        for issue in review["issues"]:
            severity_counts[str(issue["severity"])] += 1
            for dimension in issue.get("dimensions", []):
                dimension_issue_counts[str(dimension)] += 1

    recommendation_counts = Counter(str(review["recommendation"]) for review in reviews)
    role_counts = Counter(str(review["role"]) for review in reviews)

    return {
        "schema_version": "2.0.0",
        "venue": "ICLR 2026",
        "rating_scale": [2, 4, 6, 8, 10],
        "dimension_scale": {"soundness": [1, 4], "presentation": [1, 4], "contribution": [1, 4]},
        "round": rounds.pop(),
        "snapshot_sha256": snapshots.pop(),
        "reviewer_count": len(reviews),
        "reviewer_ids": [str(review["reviewer_id"]) for review in reviews],
        "role_counts": dict(sorted(role_counts.items())),
        "overall": {
            "scores": scores,
            "median": statistics.median(scores),
            "lower_quartile": percentile(scores, 0.25),
            "minimum": min(scores),
            "maximum": max(scores),
            "mean": statistics.fmean(scores),
            "population_stddev": statistics.pstdev(scores) if len(scores) > 1 else 0.0,
        },
        "confidence": {
            "scores": confidences,
            "median": statistics.median(confidences),
            "minimum": min(confidences),
        },
        "dimension_medians": dimension_medians,
        "issue_counts": {
            "by_severity": {
                "critical": severity_counts.get("critical", 0),
                "major": severity_counts.get("major", 0),
                "minor": severity_counts.get("minor", 0),
            },
            "by_dimension": dict(sorted(dimension_issue_counts.items())),
        },
        "recommendation_counts": dict(sorted(recommendation_counts.items())),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("review_dir", type=Path, help="Directory containing normalized review JSON files.")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path; defaults beside review_dir.")
    parser.add_argument("--allow-invalid", action="store_true", help="Aggregate valid files even when others are invalid.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.review_dir.is_dir():
        raise FileNotFoundError(f"Review directory not found: {args.review_dir}")

    reviews, errors = load_reviews(args.review_dir)
    if errors and not args.allow_invalid:
        raise ValueError("Review validation failed:\n" + "\n".join(errors))
    if not reviews:
        raise ValueError("No valid review JSON files found")

    result = aggregate(reviews)
    if errors:
        result["ignored_invalid_files"] = errors

    output = args.output or args.review_dir.parent / "panel_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
