from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from supervised_sft import (
    assert_train_only_path,
    validate_formal_integrity_ledgers,
    validate_prepared_training_manifest,
    validate_supervised_row,
)
from train_supervised_sft import (
    FROZEN_SMOKE_DATASET_COUNTS,
    FROZEN_TEXT_LAYER_COUNT,
    FROZEN_TEXT_PARAMETER_COUNT,
    FROZEN_WORLD_SIZE,
    validate_formal_smoke_dataset_counts,
)


FROZEN_CONFIG = {
    "training_scope": "dense_full_model_sft_smoke",
    "steps": 1,
    "gradient_accumulation": 1,
    "dataset_limit": FROZEN_WORLD_SIZE,
    "expected_trainable_params": FROZEN_TEXT_PARAMETER_COUNT,
    "expected_num_layers": FROZEN_TEXT_LAYER_COUNT,
    "expected_world_size": FROZEN_WORLD_SIZE,
    "checkpoint_mode": "metadata-only",
    "optimizer_parameter_dtype": "float32",
    "gradient_reduce_dtype": "float32",
    "loss_weighting": "global-token-weighted",
    "require_parameter_delta_gate": True,
}


def preflight_formal_supervised_smoke(
    *,
    data_path: Path,
    manifest_path: Path,
    config_path: Path,
    expected_data_sha256: str,
    expected_manifest_sha256: str,
    tokenizer: Any,
    model_path: Path,
    code_ledger_path: Path,
    expected_code_ledger_sha256: str,
    model_ledger_path: Path,
    expected_model_ledger_sha256: str,
) -> dict[str, Any]:
    assert_train_only_path(data_path)
    integrity_audit = validate_formal_integrity_ledgers(
        code_ledger_path=code_ledger_path,
        expected_code_ledger_sha256=expected_code_ledger_sha256,
        model_ledger_path=model_ledger_path,
        expected_model_ledger_sha256=expected_model_ledger_sha256,
    )
    manifest_audit = validate_prepared_training_manifest(
        manifest_path,
        data_path,
        expected_data_sha256=expected_data_sha256,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    counts: dict[str, int] = {}
    records = 0
    with data_path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"line {line_number}: JSONL record must be an object")
            validate_supervised_row(row, line_number=line_number)
            records += 1
            counts[row["dataset"]] = counts.get(row["dataset"], 0) + 1
    if records != FROZEN_WORLD_SIZE:
        raise ValueError(
            f"formal smoke requires exactly {FROZEN_WORLD_SIZE} rows, got {records}"
        )
    validate_formal_smoke_dataset_counts(counts)
    if counts != manifest_audit["dataset_written_examples"]:
        raise ValueError(
            "JSONL dataset counts differ from converter manifest: "
            f"jsonl={counts}, manifest={manifest_audit['dataset_written_examples']}"
        )

    config = json.loads(config_path.read_text())
    if not isinstance(config, dict):
        raise ValueError("supervised smoke config must be an object")
    for key, value in FROZEN_CONFIG.items():
        if config.get(key) != value:
            raise ValueError(
                f"frozen supervised smoke config mismatch for {key}: "
                f"expected={value!r}, actual={config.get(key)!r}"
            )
    result = {
        "status": "passed",
        "train_only_records": records,
        "dataset_counts": counts,
        "manifest_audit": manifest_audit,
        "frozen_config": FROZEN_CONFIG,
        "labels_rebuilt_at_runtime": True,
        "integrity": integrity_audit,
    }
    from supervised_sft import validate_runtime_tokenizer_against_manifest

    result["runtime_tokenizer"] = validate_runtime_tokenizer_against_manifest(
        tokenizer, manifest_path, model_path=model_path
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CPU-only formal data/config preflight for dense supervised SFT"
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--expected-data-sha256", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--code-ledger", type=Path, required=True)
    parser.add_argument("--expected-code-ledger-sha256", required=True)
    parser.add_argument("--model-artifact-ledger", type=Path, required=True)
    parser.add_argument("--expected-model-artifact-ledger-sha256", required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    result = preflight_formal_supervised_smoke(
        data_path=args.data,
        manifest_path=args.manifest,
        config_path=args.config,
        expected_data_sha256=args.expected_data_sha256,
        expected_manifest_sha256=args.expected_manifest_sha256,
        tokenizer=tokenizer,
        model_path=args.model,
        code_ledger_path=args.code_ledger,
        expected_code_ledger_sha256=args.expected_code_ledger_sha256,
        model_ledger_path=args.model_artifact_ledger,
        expected_model_ledger_sha256=args.expected_model_artifact_ledger_sha256,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
