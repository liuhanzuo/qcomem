from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Iterable

from analyze_validation import bootstrap_mean_ci
from run_downstream import atomic_json, normalize_answer
from run_native_lora_checkpoint_attribution import (
    ACTIVE_STEP,
    ATTRIBUTION_CONFIGS,
    EXPECTED_STORE_POLICY,
    STORE_CONFIG,
)
from aggregate_replay import token_position_agreement


EXPECTED_RANKS = set(range(8))
EXPECTED_INDICES = set(range(6, 36))
ANSWER_TYPES = ("yes", "no", "entity", "abstain")
ABSTAIN_NORMALIZED = frozenset(
    {
        "",
        "unanswerable",
        "unknown",
        "not enough information",
        "cannot be determined",
        "cannot answer",
        "insufficient information",
    }
)


def row_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return row["dataset"], str(row.get("id")), int(row["source_index"])


def answer_type(text: str) -> str:
    normalized = normalize_answer(str(text))
    if normalized in ABSTAIN_NORMALIZED:
        return "abstain"
    if normalized == "yes":
        return "yes"
    if normalized == "no":
        return "no"
    return "entity"


def reference_answer_type(references: Iterable[str]) -> str:
    categories = {answer_type(reference) for reference in references}
    if not categories:
        raise ValueError("reference answer list is empty")
    if len(categories) != 1:
        raise ValueError(f"reference answer types disagree: {sorted(categories)}")
    return next(iter(categories))


def answer_type_confusion(rows: list[dict[str, Any]]) -> dict[str, Any]:
    matrix = {
        reference: {prediction: 0 for prediction in ANSWER_TYPES}
        for reference in ANSWER_TYPES
    }
    for row in rows:
        reference = reference_answer_type(row["references"])
        prediction = answer_type(row["prediction"])
        matrix[reference][prediction] += 1
    total = sum(sum(values.values()) for values in matrix.values())
    diagonal = sum(matrix[category][category] for category in ANSWER_TYPES)
    return {
        "labels": list(ANSWER_TYPES),
        "rows_are_reference_columns_are_prediction": matrix,
        "reference_counts": {
            category: sum(matrix[category].values()) for category in ANSWER_TYPES
        },
        "prediction_counts": {
            category: sum(matrix[reference][category] for reference in ANSWER_TYPES)
            for category in ANSWER_TYPES
        },
        "type_exact_accuracy": diagonal / total if total else None,
        "samples": total,
        "classification_rule": {
            "normalization": "LongBench answer normalization (lowercase, punctuation/articles removed)",
            "yes": "normalized answer is exactly yes",
            "no": "normalized answer is exactly no",
            "abstain": sorted(ABSTAIN_NORMALIZED),
            "entity": "every other non-abstaining answer",
        },
    }


def paired_comparison(
    candidate: dict[str, Any],
    reference: dict[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    left = {row_key(row): row for row in candidate["rows"]}
    right = {row_key(row): row for row in reference["rows"]}
    if left.keys() != right.keys():
        raise ValueError("paired checkpoint sample keys differ")
    pairs = [(left[key], right[key]) for key in sorted(left)]
    deltas = [candidate_row["f1"] - reference_row["f1"] for candidate_row, reference_row in pairs]
    datasets = sorted({candidate_row["dataset"] for candidate_row, _ in pairs})
    per_dataset: dict[str, Any] = {}
    for offset, dataset in enumerate(datasets, 1):
        selected = [
            (candidate_row, reference_row)
            for candidate_row, reference_row in pairs
            if candidate_row["dataset"] == dataset
        ]
        dataset_deltas = [
            candidate_row["f1"] - reference_row["f1"]
            for candidate_row, reference_row in selected
        ]
        per_dataset[dataset] = {
            "samples": len(selected),
            "mean_f1_delta": statistics.fmean(dataset_deltas),
            "paired_bootstrap_95_ci": bootstrap_mean_ci(
                dataset_deltas, seed=seed + offset
            ),
            "prediction_exact_agreement": statistics.fmean(
                candidate_row["prediction"] == reference_row["prediction"]
                for candidate_row, reference_row in selected
            ),
            "token_sequence_exact_agreement": statistics.fmean(
                candidate_row["generated_token_ids"]
                == reference_row["generated_token_ids"]
                for candidate_row, reference_row in selected
            ),
            "mean_token_position_agreement": statistics.fmean(
                token_position_agreement(
                    reference_row["generated_token_ids"],
                    candidate_row["generated_token_ids"],
                )
                for candidate_row, reference_row in selected
            ),
        }
    return {
        "candidate": candidate["config"],
        "reference": reference["config"],
        "samples": len(pairs),
        "mean_f1_delta": statistics.fmean(deltas),
        "paired_bootstrap_95_ci": bootstrap_mean_ci(deltas, seed=seed),
        "prediction_exact_agreement": statistics.fmean(
            candidate_row["prediction"] == reference_row["prediction"]
            for candidate_row, reference_row in pairs
        ),
        "token_sequence_exact_agreement": statistics.fmean(
            candidate_row["generated_token_ids"]
            == reference_row["generated_token_ids"]
            for candidate_row, reference_row in pairs
        ),
        "mean_token_position_agreement": statistics.fmean(
            token_position_agreement(
                reference_row["generated_token_ids"],
                candidate_row["generated_token_ids"],
            )
            for candidate_row, reference_row in pairs
        ),
        "per_dataset": per_dataset,
    }


def validate_store_contract(shard: dict[str, Any]) -> None:
    actual = {
        "mode": shard.get("mode"),
        "depth": shard.get("depth"),
        "residual_bits": shard.get("residual_bits"),
        "attention_bits": shard.get("attention_bits"),
        "linear_bits": shard.get("linear_bits"),
        "cache_layer_bits": tuple(shard.get("cache_layer_bits") or ()),
    }
    if actual != EXPECTED_STORE_POLICY or shard.get("store_config") != STORE_CONFIG:
        raise ValueError(f"attribution store policy drifted: {actual}")


def merge_condition(
    shards: list[dict[str, Any]],
    condition: str,
    *,
    expected_checkpoint_sha256: dict[int, str],
) -> dict[str, Any]:
    selected = [shard for shard in shards if shard.get("config") == condition]
    if len(selected) != 8 or {int(shard["rank"]) for shard in selected} != EXPECTED_RANKS:
        raise ValueError(f"{condition}: shard rank coverage is not exactly 0--7")
    for shard in selected:
        validate_store_contract(shard)
        checkpoint = shard.get("checkpoint", {})
        step = ACTIVE_STEP[condition]
        if step is None:
            if checkpoint.get("adapter_enabled") is not False:
                raise ValueError("adapter-disabled condition unexpectedly enabled LoRA")
            if checkpoint.get("resident_disabled_adapter_checkpoint_sha256") != expected_checkpoint_sha256[0]:
                raise ValueError("disabled adapter resident step-0 SHA drifted")
            if checkpoint.get("active_checkpoint_sha256") is not None:
                raise ValueError("disabled adapter has an active checkpoint")
        else:
            if checkpoint.get("adapter_enabled") is not True:
                raise ValueError(f"{condition}: adapter is disabled")
            if checkpoint.get("active_checkpoint_step") != step:
                raise ValueError(f"{condition}: active checkpoint step drifted")
            if checkpoint.get("active_checkpoint_sha256") != expected_checkpoint_sha256[step]:
                raise ValueError(f"{condition}: active checkpoint SHA drifted")
    rows = [row for shard in selected for row in shard["rows"]]
    keys = {row_key(row) for row in rows}
    if len(rows) != 60 or len(keys) != 60:
        raise ValueError(f"{condition}: expected 60 unique rows")
    for dataset in ("qasper", "2wikimqa"):
        indices = {
            int(row["source_index"]) for row in rows if row["dataset"] == dataset
        }
        if indices != EXPECTED_INDICES:
            raise ValueError(f"{condition}/{dataset}: source indices are not 6--35")
    return {
        "config": condition,
        "active_checkpoint_step": ACTIVE_STEP[condition],
        "active_checkpoint_sha256": (
            expected_checkpoint_sha256[ACTIVE_STEP[condition]]
            if ACTIVE_STEP[condition] is not None
            else None
        ),
        "rows": rows,
        "shards": selected,
        "mean_f1": statistics.fmean(float(row["f1"]) for row in rows),
        "dataset_mean_f1": {
            dataset: statistics.fmean(
                float(row["f1"]) for row in rows if row["dataset"] == dataset
            )
            for dataset in ("qasper", "2wikimqa")
        },
        "two_wiki_answer_type_confusion": answer_type_confusion(
            [row for row in rows if row["dataset"] == "2wikimqa"]
        ),
    }


def aggregate(
    shards: list[dict[str, Any]],
    *,
    expected_data_sha256: str,
    expected_checkpoint_sha256: dict[int, str],
    source_job_id: int,
    source_trial_id: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    expected_shards = len(ATTRIBUTION_CONFIGS) * 8
    if len(shards) != expected_shards:
        raise ValueError(f"expected {expected_shards} shards, found {len(shards)}")
    if {shard.get("schema_version") for shard in shards} != {
        "qcomem-native-lora-checkpoint-attribution-shard-v1"
    }:
        raise ValueError("attribution shard schema drifted")
    if {shard.get("data_sha256") for shard in shards} != {expected_data_sha256}:
        raise ValueError("validation data SHA drifted")
    if not all(
        shard.get("raw_test_v2_read") is False
        and shard.get("validation_already_consumed") is True
        and shard.get("selection_or_checkpoint_choice_permitted") is False
        and shard.get("attribution_only") is True
        and shard.get("prompt_protocol") == "longbench-v1-official"
        and shard.get("caller") == "run_replay_diagnostic.run_config/full_state_replay"
        and shard.get("decoding") == "greedy_argmax"
        and shard.get("source_index_start") == 6
        and shard.get("source_index_end") == 35
        and shard.get("excluded_source_indices") == [4, 5]
        and shard.get("dataset_max_new_tokens") == {"qasper": 128, "2wikimqa": 32}
        and shard.get("max_input_tokens") == 4096
        and shard.get("max_new_tokens") == 128
        and shard.get("group_size") == 64
        for shard in shards
    ):
        raise ValueError("governance, caller, prompt, greedy, or generation protocol drifted")
    for shard in shards:
        ledger = shard.get("checkpoint_ledger", {})
        if {
            int(step): item.get("sha256") for step, item in ledger.items()
        } != expected_checkpoint_sha256:
            raise ValueError("checkpoint ledger drifted")

    merged = {
        condition: merge_condition(
            shards,
            condition,
            expected_checkpoint_sha256=expected_checkpoint_sha256,
        )
        for condition in ATTRIBUTION_CONFIGS
    }
    expected_keys = {row_key(row) for row in merged["adapter-disabled"]["rows"]}
    if any(
        {row_key(row) for row in merged[condition]["rows"]} != expected_keys
        for condition in ATTRIBUTION_CONFIGS
    ):
        raise ValueError("checkpoint attribution samples are not paired")

    pairs = (
        ("native-lora-step0", "adapter-disabled"),
        ("native-lora-step64", "adapter-disabled"),
        ("native-lora-step128", "adapter-disabled"),
        ("native-lora-step64", "native-lora-step0"),
        ("native-lora-step128", "native-lora-step0"),
        ("native-lora-step128", "native-lora-step64"),
    )
    comparisons = {
        f"{candidate}_vs_{reference}": paired_comparison(
            merged[candidate],
            merged[reference],
            seed=bootstrap_seed + index * 100,
        )
        for index, (candidate, reference) in enumerate(pairs, 1)
    }
    result = {
        "schema_version": "qcomem-native-lora-checkpoint-attribution-v1",
        "status": "completed",
        "experiment": "preregistered_experiment_A_checkpoint_attribution",
        "source_job_id": source_job_id,
        "source_trial_id": source_trial_id,
        "samples": 60,
        "validation_data_sha256": expected_data_sha256,
        "source_indices": [6, 35],
        "excluded_calibration_indices": [4, 5],
        "checkpoint_sha256": {
            str(step): expected_checkpoint_sha256[step] for step in (0, 64, 128)
        },
        "store_policy": {
            **EXPECTED_STORE_POLICY,
            "cache_layer_bits": list(EXPECTED_STORE_POLICY["cache_layer_bits"]),
            "name": STORE_CONFIG,
            "state_scope": "residual plus complete lower-layer KV/recurrent/conv state",
        },
        "protocol": {
            "same_model_process_per_rank": True,
            "same_store_policy": True,
            "same_caller": "run_replay_diagnostic.run_config/full_state_replay",
            "same_prompt": "longbench-v1-official",
            "same_decoding": "greedy_argmax",
            "max_input_tokens": 4096,
            "dataset_max_new_tokens": {"qasper": 128, "2wikimqa": 32},
            "paired_examples": True,
            "bootstrap_seed": bootstrap_seed,
            "bootstrap_repetitions": 10_000,
            "four_conditions_run_sequentially_in_one_8_gpu_job": True,
        },
        "mean_f1": {
            condition: {
                "overall": merged[condition]["mean_f1"],
                **merged[condition]["dataset_mean_f1"],
            }
            for condition in ATTRIBUTION_CONFIGS
        },
        "paired_comparisons": comparisons,
        "two_wiki_answer_type_confusion": {
            condition: merged[condition]["two_wiki_answer_type_confusion"]
            for condition in ATTRIBUTION_CONFIGS
        },
        "claim_boundaries": {
            "attribution_only": True,
            "validation_already_consumed": True,
            "validation_may_select_checkpoint_or_policy": False,
            "checkpoint_selection_was_completed_before_this_experiment": True,
            "raw_test_v2_read": False,
            "test_v2_source_indices_68_99_used": False,
            "inference_only_no_new_training": True,
            "q4_q8_applies_to_persistent_document_state_not_model_weights": True,
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-data-sha256", required=True)
    parser.add_argument("--expected-step0-sha256", required=True)
    parser.add_argument("--expected-step64-sha256", required=True)
    parser.add_argument("--expected-step128-sha256", required=True)
    parser.add_argument("--source-job-id", type=int, default=235749)
    parser.add_argument("--source-trial-id", type=int, default=1834056)
    parser.add_argument("--bootstrap-seed", type=int, default=20260831)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    shards = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(args.run_dir.glob("attribution-shard-*.json"))
    ]
    expected = {
        0: args.expected_step0_sha256,
        64: args.expected_step64_sha256,
        128: args.expected_step128_sha256,
    }
    try:
        result = aggregate(
            shards,
            expected_data_sha256=args.expected_data_sha256,
            expected_checkpoint_sha256=expected,
            source_job_id=args.source_job_id,
            source_trial_id=args.source_trial_id,
            bootstrap_seed=args.bootstrap_seed,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    destination = args.output or args.run_dir / "checkpoint-attribution-analysis.json"
    atomic_json(destination, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
