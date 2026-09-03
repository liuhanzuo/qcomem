from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from analyze_validation import bootstrap_mean_ci
from qcomem_answer_supervised_lora import (
    EXPECTED_ADAPTER_MODULES,
    EXPECTED_ADAPTER_PARAMETERS,
)
from run_answer_lora_full_state_downstream import CONDITIONS, EXPECTED_POLICIES
from run_downstream import atomic_json, normalize_answer


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
    if normalized in {"yes", "no"}:
        return normalized
    return "entity"


def reference_answer_type(references: list[str]) -> str:
    categories = {answer_type(reference) for reference in references}
    if len(categories) != 1:
        raise ValueError(f"reference answer types disagree: {sorted(categories)}")
    return next(iter(categories))


def answer_type_confusion(rows: list[dict[str, Any]]) -> dict[str, Any]:
    matrix = {
        reference: {prediction: 0 for prediction in ANSWER_TYPES}
        for reference in ANSWER_TYPES
    }
    for row in rows:
        matrix[reference_answer_type(row["references"])][
            answer_type(row["prediction"])
        ] += 1
    return {
        "labels": list(ANSWER_TYPES),
        "rows_are_reference_columns_are_prediction": matrix,
        "samples": len(rows),
    }


def answer_type_transition(
    candidate: dict[str, Any], reference: dict[str, Any]
) -> dict[str, Any]:
    left = {
        row_key(row): row for row in candidate["rows"] if row["dataset"] == "2wikimqa"
    }
    right = {
        row_key(row): row for row in reference["rows"] if row["dataset"] == "2wikimqa"
    }
    if left.keys() != right.keys() or len(left) != 30:
        raise ValueError("2Wiki answer-type transition requires 30 paired rows")
    matrix = {
        before: {after: 0 for after in ANSWER_TYPES} for before in ANSWER_TYPES
    }
    reference_yes = {
        "samples": 0,
        "disabled_prediction_types": {label: 0 for label in ANSWER_TYPES},
        "selected_prediction_types": {label: 0 for label in ANSWER_TYPES},
    }
    text_changed = 0
    type_changed = 0
    for key in sorted(left):
        selected = left[key]
        disabled = right[key]
        before = answer_type(disabled["prediction"])
        after = answer_type(selected["prediction"])
        matrix[before][after] += 1
        type_changed += int(before != after)
        text_changed += int(selected["prediction"] != disabled["prediction"])
        if reference_answer_type(selected["references"]) == "yes":
            reference_yes["samples"] += 1
            reference_yes["disabled_prediction_types"][before] += 1
            reference_yes["selected_prediction_types"][after] += 1
    return {
        "rows_are_disabled_prediction_columns_are_selected_prediction": matrix,
        "samples": 30,
        "prediction_text_changed": text_changed,
        "prediction_type_changed": type_changed,
        "reference_yes": reference_yes,
        "purpose": "audit the previously observed yes-to-no failure mode",
    }


def paired_comparison(
    candidate: dict[str, Any], reference: dict[str, Any], *, seed: int
) -> dict[str, Any]:
    left = {row_key(row): row for row in candidate["rows"]}
    right = {row_key(row): row for row in reference["rows"]}
    if left.keys() != right.keys():
        raise ValueError("paired downstream sample keys differ")
    keys = sorted(left)
    deltas = [float(left[key]["f1"]) - float(right[key]["f1"]) for key in keys]
    result = {
        "candidate": candidate["config"],
        "reference": reference["config"],
        "samples": len(keys),
        "mean_f1_delta": statistics.fmean(deltas),
        "paired_bootstrap_95_ci": bootstrap_mean_ci(deltas, seed=seed),
        "prediction_exact_agreement": statistics.fmean(
            left[key]["prediction"] == right[key]["prediction"] for key in keys
        ),
        "per_dataset": {},
    }
    for offset, dataset in enumerate(("qasper", "2wikimqa"), 1):
        selected = [key for key in keys if key[0] == dataset]
        dataset_deltas = [
            float(left[key]["f1"]) - float(right[key]["f1"]) for key in selected
        ]
        result["per_dataset"][dataset] = {
            "samples": len(selected),
            "mean_f1_delta": statistics.fmean(dataset_deltas),
            "paired_bootstrap_95_ci": bootstrap_mean_ci(
                dataset_deltas, seed=seed + offset
            ),
            "prediction_exact_agreement": statistics.fmean(
                left[key]["prediction"] == right[key]["prediction"]
                for key in selected
            ),
        }
    return result


def validate_store(shard: dict[str, Any], condition: str) -> None:
    store = CONDITIONS[condition][0]
    expected = EXPECTED_POLICIES[store]
    actual = {
        "mode": shard.get("mode"),
        "depth": shard.get("depth"),
        "residual_bits": shard.get("residual_bits"),
        "attention_bits": shard.get("attention_bits"),
        "linear_bits": shard.get("linear_bits"),
        "cache_layer_bits": shard.get("cache_layer_bits"),
    }
    if shard.get("store_config") != store or actual != expected:
        raise ValueError(f"{condition}: full-state store policy drifted: {actual}")


def merge_condition(
    shards: list[dict[str, Any]],
    condition: str,
) -> dict[str, Any]:
    selected = [shard for shard in shards if shard.get("config") == condition]
    if len(selected) != 8 or {int(shard["rank"]) for shard in selected} != EXPECTED_RANKS:
        raise ValueError(f"{condition}: shard rank coverage is not exactly 0--7")
    expected_enabled = CONDITIONS[condition][1]
    expected_step = CONDITIONS[condition][2]
    for shard in selected:
        validate_store(shard, condition)
        if shard.get("adapter_enabled") is not expected_enabled:
            raise ValueError(f"{condition}: adapter activation drifted")
        if shard.get("active_checkpoint_step") != expected_step:
            raise ValueError(f"{condition}: active checkpoint step drifted")
        if bool(shard.get("active_checkpoint_sha256")) is not (expected_step is not None):
            raise ValueError(f"{condition}: active checkpoint SHA presence drifted")
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
    persistent = [
        int(row["stored_persistent_nbytes"])
        for row in rows
        if row.get("stored_persistent_nbytes") is not None
    ]
    return {
        "config": condition,
        "store_config": CONDITIONS[condition][0],
        "adapter_enabled": expected_enabled,
        "active_checkpoint_step": expected_step,
        "rows": rows,
        "mean_f1": statistics.fmean(float(row["f1"]) for row in rows),
        "dataset_mean_f1": {
            dataset: statistics.fmean(
                float(row["f1"]) for row in rows if row["dataset"] == dataset
            )
            for dataset in ("qasper", "2wikimqa")
        },
        "mean_stored_persistent_nbytes": (
            statistics.fmean(persistent) if persistent else None
        ),
        "two_wiki_answer_type_confusion": answer_type_confusion(
            [row for row in rows if row["dataset"] == "2wikimqa"]
        ),
    }


def aggregate(
    shards: list[dict[str, Any]],
    *,
    expected_data_sha256: str,
    bootstrap_seed: int,
) -> dict[str, Any]:
    if len(shards) != len(CONDITIONS) * 8:
        raise ValueError(
            f"expected {len(CONDITIONS) * 8} downstream shards, found {len(shards)}"
        )
    if {shard.get("schema_version") for shard in shards} != {
        "qcomem-answer-lora-full-state-shard-v1"
    }:
        raise ValueError("answer downstream shard schema drifted")
    if {shard.get("data_sha256") for shard in shards} != {expected_data_sha256}:
        raise ValueError("answer downstream validation SHA drifted")
    checkpoint_steps = {shard.get("selected_checkpoint_step") for shard in shards}
    checkpoint_shas = {shard.get("selected_checkpoint_sha256") for shard in shards}
    selection_shas = {shard.get("best_checkpoint_record_sha256") for shard in shards}
    suite_ledgers = {
        json.dumps(shard.get("checkpoint_suite_sha256"), sort_keys=True)
        for shard in shards
    }
    if len(checkpoint_steps) != 1 or checkpoint_steps.pop() not in {0, 64, 128}:
        raise ValueError("answer downstream selected checkpoint step drifted")
    selected_step = next(iter({shard["selected_checkpoint_step"] for shard in shards}))
    if len(checkpoint_shas) != 1 or len(selection_shas) != 1 or len(suite_ledgers) != 1:
        raise ValueError("answer downstream checkpoint/selection SHA drifted")
    checkpoint_suite = json.loads(next(iter(suite_ledgers)))
    if set(checkpoint_suite) != {"0", "64", "128"} or any(
        not isinstance(checkpoint_suite[key], str) or len(checkpoint_suite[key]) != 64
        for key in checkpoint_suite
    ):
        raise ValueError("answer downstream full checkpoint suite ledger drifted")
    if next(iter(checkpoint_shas)) != checkpoint_suite[str(selected_step)]:
        raise ValueError("heldout-selected checkpoint is outside the frozen suite")
    for shard in shards:
        condition = shard.get("config")
        if condition not in CONDITIONS:
            raise ValueError("answer downstream contains an unknown condition")
        active_step = CONDITIONS[condition][2]
        expected_active_sha = (
            checkpoint_suite[str(active_step)] if active_step is not None else None
        )
        if shard.get("active_checkpoint_sha256") != expected_active_sha:
            raise ValueError(f"{condition}: active checkpoint SHA drifted")
        expected_resident_step = active_step if active_step is not None else 0
        if (
            shard.get("resident_checkpoint_step") != expected_resident_step
            or shard.get("resident_checkpoint_sha256")
            != checkpoint_suite[str(expected_resident_step)]
        ):
            raise ValueError(f"{condition}: resident checkpoint state drifted")
    if not all(
        shard.get("raw_test_v2_read") is False
        and shard.get("validation_already_consumed") is True
        and shard.get("selection_or_checkpoint_choice_permitted") is False
        and shard.get("checkpoint_selection_frozen_before_validation_read") is True
        and shard.get("checkpoint_selection_source")
        == "independent_official_train_heldout_domain_only"
        and shard.get("all_checkpoint_steps_evaluated_unconditionally") is True
        and shard.get("validation_step_results_may_reselect_checkpoint") is False
        and shard.get("prompt_protocol") == "longbench-v1-official"
        and shard.get("caller")
        == "run_replay_diagnostic.run_config/full_state_replay"
        and shard.get("decoding") == "greedy_argmax"
        and shard.get("source_index_start") == 6
        and shard.get("source_index_end") == 35
        and shard.get("excluded_source_indices") == [4, 5]
        and shard.get("dataset_max_new_tokens") == {"qasper": 128, "2wikimqa": 32}
        and shard.get("max_input_tokens") == 4096
        and shard.get("max_new_tokens") == 128
        and shard.get("group_size") == 64
        and shard.get("resident_adapter_modules") == EXPECTED_ADAPTER_MODULES
        and shard.get("resident_adapter_parameters") == EXPECTED_ADAPTER_PARAMETERS
        and shard.get("resident_adapter_parameter_bytes") == 106_758_144
        and shard.get("resident_adapter_memory_scope")
        == "shared_model_resident_per_process_not_per_document"
        and shard.get("adapter_config", {}).get("rank") == 32
        and shard.get("adapter_config", {}).get("alpha") == 64.0
        and shard.get("adapter_config", {}).get("dropout") == 0.0
        for shard in shards
    ):
        raise ValueError("answer downstream governance/surface/protocol drifted")

    merged = {
        condition: merge_condition(shards, condition) for condition in CONDITIONS
    }
    reference_keys = {
        row_key(row) for row in merged["frozen-static-adapter-disabled"]["rows"]
    }
    if any(
        {row_key(row) for row in merged[condition]["rows"]} != reference_keys
        for condition in CONDITIONS
    ):
        raise ValueError("answer downstream conditions are not paired")
    selected_condition = f"frozen-static-answer-lora-step{selected_step}"
    if selected_condition not in merged:
        raise ValueError("heldout-selected alias does not name a frozen step condition")
    for shard in shards:
        expected_alias = shard.get("config") == selected_condition
        if shard.get("condition_is_heldout_selected_alias") is not expected_alias:
            raise ValueError("heldout-selected condition alias drifted")
    pairs = (
        ("frozen-static-answer-lora-step0", "frozen-static-adapter-disabled"),
        ("frozen-static-answer-lora-step64", "frozen-static-adapter-disabled"),
        ("frozen-static-answer-lora-step128", "frozen-static-adapter-disabled"),
        ("frozen-static-answer-lora-step64", "frozen-static-answer-lora-step0"),
        ("frozen-static-answer-lora-step128", "frozen-static-answer-lora-step64"),
        ("frozen-static-answer-lora-step128", "frozen-static-answer-lora-step0"),
        (selected_condition, "q16-adapter-disabled-control"),
        (selected_condition, "dense-adapter-disabled-control"),
        ("frozen-static-adapter-disabled", "q16-adapter-disabled-control"),
        ("q16-adapter-disabled-control", "dense-adapter-disabled-control"),
    )
    comparisons = {
        f"{candidate}_vs_{reference}": paired_comparison(
            merged[candidate],
            merged[reference],
            seed=bootstrap_seed + index * 100,
        )
        for index, (candidate, reference) in enumerate(pairs, 1)
    }
    return {
        "schema_version": "qcomem-answer-lora-full-state-downstream-v1",
        "status": "completed",
        "experiment": "answer_supervised_lora_b_post_selection_attribution",
        "samples": 60,
        "conditions": list(CONDITIONS),
        "validation_data_sha256": expected_data_sha256,
        "source_indices": [6, 35],
        "excluded_calibration_indices": [4, 5],
        "selected_checkpoint_step": selected_step,
        "selected_checkpoint_sha256": next(iter(checkpoint_shas)),
        "checkpoint_suite_sha256": checkpoint_suite,
        "best_checkpoint_record_sha256": next(iter(selection_shas)),
        "heldout_selected_alias": {
            "condition": selected_condition,
            "selection_source": "independent_official_train_heldout_domain_only",
            "additional_forward_executed": False,
            "validation_results_used_to_change_alias": False,
        },
        "mean_f1": {
            condition: {
                "overall": result["mean_f1"],
                **result["dataset_mean_f1"],
            }
            for condition, result in merged.items()
        },
        "mean_stored_persistent_nbytes": {
            condition: result["mean_stored_persistent_nbytes"]
            for condition, result in merged.items()
        },
        "memory_accounting": {
            "shared_model_resident_answer_adapter": {
                "parameters": EXPECTED_ADAPTER_PARAMETERS,
                "dtype": "float32",
                "bytes": 106_758_144,
                "mib": 101.8125,
                "scope": "one shared model process; amortized across documents",
            },
            "per_document_persistent_state": {
                "source": "mean of the 60 measured full-state rows",
                "bytes_by_condition": {
                    condition: result["mean_stored_persistent_nbytes"]
                    for condition, result in merged.items()
                },
                "mib_by_condition": {
                    condition: (
                        result["mean_stored_persistent_nbytes"] / 2**20
                        if result["mean_stored_persistent_nbytes"] is not None
                        else None
                    )
                    for condition, result in merged.items()
                },
                "historical_frozen_static_reference_mib": 9.66,
                "historical_reference_is_not_a_gate": True,
                "scope": "one stored document; scales with resident documents",
            },
            "adapter_bytes_included_in_per_document_state": False,
        },
        "paired_comparisons": comparisons,
        "heldout_selected_vs_disabled": comparisons[
            f"{selected_condition}_vs_frozen-static-adapter-disabled"
        ],
        "two_wiki_answer_type_confusion": {
            condition: result["two_wiki_answer_type_confusion"]
            for condition, result in merged.items()
        },
        "two_wiki_step_vs_disabled_type_transition": {
            str(step): answer_type_transition(
                merged[f"frozen-static-answer-lora-step{step}"],
                merged["frozen-static-adapter-disabled"],
            )
            for step in (0, 64, 128)
        },
        "two_wiki_selected_vs_disabled_type_transition": answer_type_transition(
            merged[selected_condition],
            merged["frozen-static-adapter-disabled"],
        ),
        "protocol": {
            "same_model_process_per_rank": True,
            "same_full_state_caller": True,
            "same_prompt": "longbench-v1-official",
            "same_decoding": "greedy_argmax",
            "paired_examples": True,
            "bootstrap_seed": bootstrap_seed,
            "bootstrap_repetitions": 10_000,
            "checkpoint_frozen_before_validation_read": True,
            "all_checkpoint_steps_evaluated_regardless_of_validation": True,
            "heldout_selected_alias_reuses_one_of_step_0_64_128_forwards": True,
            "same_8_gpu_job_after_training_stage": True,
        },
        "claim_boundaries": {
            "downstream_is_post_selection_attribution_only": True,
            "validation_already_consumed": True,
            "validation_may_select_checkpoint_or_policy": False,
            "validation_step_trajectory_may_reselect_heldout_checkpoint": False,
            "raw_test_v2_read": False,
            "test_v2_source_indices_68_99_used": False,
            "inference_only_after_training": True,
            "q4_q8_q16_apply_to_persistent_document_state_not_model_weights": True,
            "dense_control_has_resident_disabled_adapter": True,
            "shared_adapter_overhead_is_separate_from_per_document_state": True,
            "shared_adapter_overhead_bytes": 106_758_144,
            "shared_adapter_overhead_mib": 101.8125,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-data-sha256", required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=20260814)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    shards = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(args.run_dir.glob("answer-downstream-shard-*.json"))
    ]
    try:
        result = aggregate(
            shards,
            expected_data_sha256=args.expected_data_sha256,
            bootstrap_seed=args.bootstrap_seed,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    destination = args.output or args.run_dir / "answer-full-state-downstream-analysis.json"
    atomic_json(destination, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
