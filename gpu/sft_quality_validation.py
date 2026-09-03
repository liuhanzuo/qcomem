from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from split_supervised_sft_scale import (
    DATASETS,
    MANIFEST_SCHEMA,
    SplitContractError,
    stable_json,
    validate_parent_row,
)


FROZEN_WORLD_SIZE = 8
FROZEN_TEXT_PARAMETER_COUNT = 34_660_610_688
FROZEN_LONGBENCH_REVISION = "5e628be450b7e67fb7ae6e201bd6d8f7056f7672"
FROZEN_VALIDATION_SOURCE_START = 6
FROZEN_VALIDATION_SOURCE_END = 35
HASH_RE_LENGTH = 64


class QualityContractError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != HASH_RE_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise QualityContractError(f"{label} must be one lowercase SHA256")
    return value


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise QualityContractError(f"{label} must contain one JSON object")
    return payload


def validate_runtime_tokenizer_metadata(
    tokenizer: Any, split_manifest: dict[str, Any]
) -> dict[str, Any]:
    expected = split_manifest.get("tokenizer")
    if not isinstance(expected, dict):
        raise QualityContractError("split manifest tokenizer metadata is missing")
    chat_template = getattr(tokenizer, "chat_template", None)
    actual = {
        "class": type(tokenizer).__name__,
        "vocab_size": getattr(tokenizer, "vocab_size", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "chat_template_sha256": (
            hashlib.sha256(chat_template.encode("utf-8")).hexdigest()
            if isinstance(chat_template, str)
            else None
        ),
    }
    for key, value in actual.items():
        if expected.get(key) != value:
            raise QualityContractError(
                f"runtime tokenizer {key} differs from split manifest: "
                f"expected={expected.get(key)!r}, actual={value!r}"
            )
    return actual


@dataclass(frozen=True)
class PreparedQualityExample:
    input_ids: torch.Tensor
    labels: torch.Tensor
    dataset: str
    source_id: str
    parent_row_index: int

    @property
    def target_tokens(self) -> int:
        return int((self.labels != -100).sum().item())

    @property
    def sequence_tokens(self) -> int:
        return int(self.input_ids.numel())

    @property
    def key(self) -> tuple[str, str]:
        return self.dataset, self.source_id


class PreparedQualityDataset(Dataset[PreparedQualityExample]):
    """A frozen, fingerprint-disjoint official-train CE-heldout split.

    This reader consumes only the split artifact produced from official *train*
    rows. It does not know a test-v2 path and therefore cannot open test-v2.
    """

    def __init__(
        self,
        heldout_path: Path,
        split_manifest_path: Path,
        *,
        expected_heldout_sha256: str,
        expected_split_manifest_sha256: str,
        expected_train_sha256: str,
    ) -> None:
        expected_heldout_sha256 = require_sha256(
            expected_heldout_sha256, "expected_heldout_sha256"
        )
        expected_split_manifest_sha256 = require_sha256(
            expected_split_manifest_sha256, "expected_split_manifest_sha256"
        )
        expected_train_sha256 = require_sha256(
            expected_train_sha256, "expected_train_sha256"
        )
        actual_heldout_sha256 = sha256_file(heldout_path)
        actual_manifest_sha256 = sha256_file(split_manifest_path)
        if actual_heldout_sha256 != expected_heldout_sha256:
            raise QualityContractError(
                "CE-heldout JSONL SHA256 mismatch: "
                f"expected={expected_heldout_sha256}, actual={actual_heldout_sha256}"
            )
        if actual_manifest_sha256 != expected_split_manifest_sha256:
            raise QualityContractError(
                "split manifest SHA256 mismatch: "
                f"expected={expected_split_manifest_sha256}, "
                f"actual={actual_manifest_sha256}"
            )
        manifest = _read_json_object(split_manifest_path, "split manifest")
        self._validate_manifest(
            manifest,
            heldout_path=heldout_path,
            heldout_sha256=actual_heldout_sha256,
            expected_train_sha256=expected_train_sha256,
        )

        tokenizer = manifest["tokenizer"]
        eos_token_id = tokenizer.get("eos_token_id")
        if isinstance(eos_token_id, bool) or not isinstance(eos_token_id, int):
            raise QualityContractError("split tokenizer EOS must be one integer")
        max_sequence_tokens = manifest["prompt_protocol"].get(
            "max_sequence_tokens"
        )
        if (
            isinstance(max_sequence_tokens, bool)
            or not isinstance(max_sequence_tokens, int)
            or max_sequence_tokens < 1
        ):
            raise QualityContractError("split max_sequence_tokens is invalid")

        examples: list[PreparedQualityExample] = []
        seen: set[tuple[str, str]] = set()
        dataset_counts: Counter[str] = Counter()
        with heldout_path.open(encoding="utf-8") as handle:
            for row_index, raw_line in enumerate(handle):
                try:
                    row = json.loads(raw_line)
                except json.JSONDecodeError as error:
                    raise QualityContractError(
                        f"CE-heldout row {row_index} is invalid JSON"
                    ) from error
                if not isinstance(row, dict):
                    raise QualityContractError(
                        f"CE-heldout row {row_index} must be an object"
                    )
                if raw_line != stable_json(row) + "\n":
                    raise QualityContractError(
                        f"CE-heldout row {row_index} is not canonical JSONL"
                    )
                try:
                    validate_parent_row(
                        row,
                        row_index=row_index,
                        eos_token_id=eos_token_id,
                        max_sequence_tokens=max_sequence_tokens,
                    )
                except SplitContractError as error:
                    raise QualityContractError(str(error)) from error
                key = (row["dataset"], row["source_id"])
                if key in seen:
                    raise QualityContractError(f"duplicate CE-heldout source key {key}")
                seen.add(key)
                dataset_counts[row["dataset"]] += 1
                example = PreparedQualityExample(
                    input_ids=torch.tensor(row["input_ids"], dtype=torch.long),
                    labels=torch.tensor(row["labels"], dtype=torch.long),
                    dataset=row["dataset"],
                    source_id=row["source_id"],
                    parent_row_index=row_index,
                )
                if example.target_tokens < 1:
                    raise QualityContractError(
                        f"CE-heldout row {row_index} has no answer/EOS targets"
                    )
                examples.append(example)
        expected_summary = manifest["outputs"]["heldout_ce_jsonl"]
        if len(examples) != expected_summary.get("count"):
            raise QualityContractError("CE-heldout row count differs from manifest")
        if dict(dataset_counts) != expected_summary.get("dataset_counts"):
            raise QualityContractError("CE-heldout dataset counts differ from manifest")
        if len(examples) % FROZEN_WORLD_SIZE:
            raise QualityContractError(
                "CE-heldout count must divide exactly across eight ranks; padding or "
                "duplicate evaluation is forbidden"
            )
        self.examples = examples
        self.manifest = manifest
        self.audit = {
            "heldout_jsonl": str(heldout_path),
            "heldout_jsonl_sha256": actual_heldout_sha256,
            "split_manifest": str(split_manifest_path),
            "split_manifest_sha256": actual_manifest_sha256,
            "train_jsonl_sha256": expected_train_sha256,
            "examples": len(examples),
            "dataset_counts": dict(dataset_counts),
            "source_split": "train",
            "final_downstream_evaluation": False,
            "raw_test_v2_read": False,
        }

    @staticmethod
    def _validate_manifest(
        manifest: dict[str, Any],
        *,
        heldout_path: Path,
        heldout_sha256: str,
        expected_train_sha256: str,
    ) -> None:
        if manifest.get("schema_version") != MANIFEST_SCHEMA:
            raise QualityContractError("unexpected scale split manifest schema")
        if manifest.get("status") != "passed":
            raise QualityContractError("scale split manifest did not pass")
        outputs = manifest.get("outputs")
        if not isinstance(outputs, dict):
            raise QualityContractError("scale split outputs are missing")
        heldout = outputs.get("heldout_ce_jsonl")
        train = outputs.get("train_jsonl")
        if not isinstance(heldout, dict) or not isinstance(train, dict):
            raise QualityContractError("scale split train/heldout outputs are missing")
        if heldout.get("basename") != heldout_path.name:
            raise QualityContractError("CE-heldout basename differs from manifest")
        if heldout.get("sha256") != heldout_sha256:
            raise QualityContractError("CE-heldout SHA differs from manifest")
        if train.get("sha256") != expected_train_sha256:
            raise QualityContractError("train JSONL SHA differs from split manifest")
        audit = manifest.get("disjoint_audit")
        if not isinstance(audit, dict) or audit.get("all_zero") is not True:
            raise QualityContractError("train/CE-heldout disjoint audit did not pass")
        if audit.get("source_id_intersection_count") != 0:
            raise QualityContractError("train/CE-heldout source IDs overlap")
        if audit.get("component_intersection_count") != 0:
            raise QualityContractError("train/CE-heldout fingerprint groups overlap")
        fingerprint_counts = audit.get("fingerprint_intersection_counts")
        if not isinstance(fingerprint_counts, dict) or any(
            value != 0 for value in fingerprint_counts.values()
        ):
            raise QualityContractError("train/CE-heldout fingerprints overlap")
        governance = manifest.get("data_governance")
        expected_governance = {
            "all_rows_top_level_source_split": "train",
            "all_rows_provenance_source_split": "train",
            "validation_or_test_rows_used": False,
            "raw_test_v2_read": False,
            "heldout_ce_is_final_downstream_evaluation": False,
        }
        if not isinstance(governance, dict):
            raise QualityContractError("split data governance is missing")
        for key, expected in expected_governance.items():
            if governance.get(key) != expected:
                raise QualityContractError(
                    f"split data governance {key} must be {expected!r}"
                )
        parent = manifest.get("parent")
        if not isinstance(parent, dict):
            raise QualityContractError("split parent audit is missing")
        if parent.get("raw_test_v2_read_by_converter") is not False:
            raise QualityContractError("parent converter read raw test-v2")
        if parent.get("test_v2_content_hash_check") != "deferred_not_read":
            raise QualityContractError("test-v2 must remain deferred and unread")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> PreparedQualityExample:
        return self.examples[index]


def single_quality_example_collate(
    batch: Sequence[PreparedQualityExample],
) -> dict[str, Any]:
    if len(batch) != 1:
        raise QualityContractError("quality evaluation requires batch size one")
    example = batch[0]
    return {
        "input_ids": example.input_ids.unsqueeze(0),
        "labels": example.labels.unsqueeze(0),
        "attention_mask": torch.ones_like(example.input_ids).unsqueeze(0),
        "dataset": example.dataset,
        "source_id": example.source_id,
        "parent_row_index": example.parent_row_index,
        "sequence_tokens": example.sequence_tokens,
        "target_tokens": example.target_tokens,
    }


def _summarize_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise QualityContractError("cannot summarize zero quality rows")
    target_tokens = sum(int(row["target_tokens"]) for row in rows)
    nll_sum = sum(float(row["nll_sum"]) for row in rows)
    if target_tokens < 1 or not math.isfinite(nll_sum):
        raise QualityContractError("quality NLL aggregate is not finite and positive")
    token_weighted_ce = nll_sum / target_tokens
    return {
        "examples": len(rows),
        "target_tokens": target_tokens,
        "nll_sum": nll_sum,
        "token_weighted_ce": token_weighted_ce,
        "perplexity": math.exp(min(token_weighted_ce, 80.0)),
        "mean_example_ce": sum(float(row["ce"]) for row in rows) / len(rows),
    }


def summarize_quality_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    keys = [(row["dataset"], row["source_id"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise QualityContractError("quality result contains duplicate source keys")
    if any(row["dataset"] not in DATASETS for row in rows):
        raise QualityContractError("quality result contains an unknown dataset")
    return {
        "overall": _summarize_rows(rows),
        "by_dataset": {
            dataset: _summarize_rows(
                [row for row in rows if row["dataset"] == dataset]
            )
            for dataset in DATASETS
        },
    }


def paired_quality_comparison(
    before_rows: Sequence[dict[str, Any]],
    after_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    before = {(row["dataset"], row["source_id"]): row for row in before_rows}
    after = {(row["dataset"], row["source_id"]): row for row in after_rows}
    if len(before) != len(before_rows) or len(after) != len(after_rows):
        raise QualityContractError("paired quality rows contain duplicate keys")
    if set(before) != set(after):
        raise QualityContractError("before/after quality source keys differ")
    paired_rows = []
    for key in sorted(before):
        left, right = before[key], after[key]
        if left["target_tokens"] != right["target_tokens"]:
            raise QualityContractError(f"paired target count changed for {key}")
        delta = float(right["ce"]) - float(left["ce"])
        paired_rows.append(
            {
                "dataset": key[0],
                "source_id": key[1],
                "target_tokens": int(left["target_tokens"]),
                "before_ce": float(left["ce"]),
                "after_ce": float(right["ce"]),
                "after_minus_before_ce": delta,
                "improved": delta < 0.0,
            }
        )
    before_summary = summarize_quality_rows(list(before.values()))
    after_summary = summarize_quality_rows(list(after.values()))
    before_ce = before_summary["overall"]["token_weighted_ce"]
    after_ce = after_summary["overall"]["token_weighted_ce"]
    finite = math.isfinite(before_ce) and math.isfinite(after_ce)
    improved = finite and after_ce < before_ce
    return {
        "schema_version": "qcomem-sft-paired-heldout-ce-v1",
        "before": before_summary,
        "after": after_summary,
        "paired": {
            "examples": len(paired_rows),
            "improved_examples": sum(row["improved"] for row in paired_rows),
            "mean_after_minus_before_example_ce": sum(
                row["after_minus_before_ce"] for row in paired_rows
            )
            / len(paired_rows),
            "token_weighted_after_minus_before_ce": after_ce - before_ce,
            "relative_token_weighted_ce_change": (after_ce - before_ce) / before_ce,
            "rows": paired_rows,
        },
        "conditional_model_only_checkpoint_gate": {
            "finite": finite,
            "heldout_token_weighted_ce_improved": improved,
            "passed": improved,
            "criterion": "after finite token-weighted CE < step-0 finite token-weighted CE",
        },
    }


def validate_fsdp_parameter_gate(
    model: torch.nn.Module,
    device: torch.device,
    *,
    expected_parameters: int = FROZEN_TEXT_PARAMETER_COUNT,
    expected_world_size: int = FROZEN_WORLD_SIZE,
) -> dict[str, Any]:
    if not dist.is_initialized() or dist.get_world_size() != expected_world_size:
        raise QualityContractError(
            f"quality model gate requires exactly {expected_world_size} ranks"
        )
    local = sum(parameter.numel() for parameter in model.parameters())
    count = torch.tensor(local, dtype=torch.int64, device=device)
    dist.all_reduce(count)
    if int(count.item()) != expected_parameters:
        raise QualityContractError(
            f"FSDP parameter shards sum to {int(count.item()):,}, expected "
            f"{expected_parameters:,}"
        )
    dtypes = sorted({str(parameter.dtype) for parameter in model.parameters()})
    if dtypes != ["torch.float32"]:
        raise QualityContractError(
            f"persistent eval FSDP shards must be FP32, got {dtypes}"
        )
    return {
        "world_size": expected_world_size,
        "global_parameter_count": expected_parameters,
        "persistent_shard_dtype": "torch.float32",
        "passed": True,
    }


@torch.no_grad()
def evaluate_answer_eos_ce_distributed(
    model: torch.nn.Module,
    dataset: PreparedQualityDataset,
    device: torch.device,
    *,
    phase: str,
) -> dict[str, Any]:
    """Evaluate every heldout row once across equal eight-rank FSDP shards.

    The supplied model must use the same forward contract as
    ``DenseSupervisedCausalLM``: ``model(input_ids, labels, attention_mask)``
    returns mean answer+EOS CE for the local example.
    """

    if not phase or not isinstance(phase, str):
        raise QualityContractError("quality phase must be a non-empty string")
    if not dist.is_initialized() or dist.get_world_size() != FROZEN_WORLD_SIZE:
        raise QualityContractError("quality CE requires exactly eight distributed ranks")
    rank = dist.get_rank()
    sampler = DistributedSampler(
        dataset,
        num_replicas=FROZEN_WORLD_SIZE,
        rank=rank,
        shuffle=False,
        drop_last=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        collate_fn=single_quality_example_collate,
        num_workers=0,
    )
    was_training = model.training
    model.eval()
    local_rows = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        loss = model(input_ids, labels, attention_mask)
        value = float(loss.detach().float().item())
        if not math.isfinite(value) or value < 0.0:
            raise QualityContractError(
                f"{phase} produced invalid CE {value!r} for "
                f"{batch['dataset']}/{batch['source_id']}"
            )
        targets = int(batch["target_tokens"])
        local_rows.append(
            {
                "phase": phase,
                "rank": rank,
                "dataset": batch["dataset"],
                "source_id": batch["source_id"],
                "parent_row_index": int(batch["parent_row_index"]),
                "sequence_tokens": int(batch["sequence_tokens"]),
                "target_tokens": targets,
                "ce": value,
                "nll_sum": value * targets,
            }
        )
    gathered: list[Any] = [None] * FROZEN_WORLD_SIZE
    dist.all_gather_object(gathered, local_rows)
    if was_training:
        model.train()
    rows = [row for rank_rows in gathered for row in rank_rows]
    if len(rows) != len(dataset):
        raise QualityContractError(
            f"quality evaluation covered {len(rows)} rows, expected {len(dataset)}"
        )
    rows.sort(key=lambda row: (row["dataset"], row["source_id"]))
    return {
        "schema_version": "qcomem-sft-heldout-ce-phase-v1",
        "phase": phase,
        "data": dataset.audit,
        "summary": summarize_quality_rows(rows),
        "rows": rows,
    }


def validate_longbench_validation_rows(
    path: Path, *, expected_sha256: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read exactly LongBench validation source 6--35; reject every other row."""

    expected_sha256 = require_sha256(expected_sha256, "expected_validation_sha256")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise QualityContractError(
            f"LongBench validation SHA256 mismatch: expected={expected_sha256}, "
            f"actual={actual}"
        )
    rows = []
    keys: set[tuple[str, int]] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            if not isinstance(row, dict):
                raise QualityContractError(
                    f"LongBench validation line {line_number} is not an object"
                )
            dataset = row.get("dataset")
            source_index = row.get("_source_index")
            if dataset not in DATASETS:
                raise QualityContractError("LongBench validation dataset is not frozen")
            if (
                isinstance(source_index, bool)
                or not isinstance(source_index, int)
                or not (
                    FROZEN_VALIDATION_SOURCE_START
                    <= source_index
                    <= FROZEN_VALIDATION_SOURCE_END
                )
            ):
                raise QualityContractError(
                    "LongBench quality evaluation permits only source indices 6--35"
                )
            if row.get("_source_revision") != FROZEN_LONGBENCH_REVISION:
                raise QualityContractError("LongBench validation revision drifted")
            key = dataset, source_index
            if key in keys:
                raise QualityContractError(f"duplicate LongBench validation row {key}")
            keys.add(key)
            rows.append(row)
    expected_keys = {
        (dataset, source_index)
        for dataset in DATASETS
        for source_index in range(
            FROZEN_VALIDATION_SOURCE_START, FROZEN_VALIDATION_SOURCE_END + 1
        )
    }
    if keys != expected_keys:
        raise QualityContractError(
            "LongBench validation must contain exactly both datasets' source 6--35"
        )
    return rows, {
        "path": str(path),
        "sha256": actual,
        "source_revision": FROZEN_LONGBENCH_REVISION,
        "source_index_start": FROZEN_VALIDATION_SOURCE_START,
        "source_index_end": FROZEN_VALIDATION_SOURCE_END,
        "rows": len(rows),
        "dataset_counts": dict(Counter(row["dataset"] for row in rows)),
        "raw_test_v2_read": False,
    }


def paired_generation_comparison(
    before_rows: Iterable[dict[str, Any]], after_rows: Iterable[dict[str, Any]]
) -> dict[str, Any]:
    before = {
        (row["dataset"], int(row["source_index"])): row for row in before_rows
    }
    after = {
        (row["dataset"], int(row["source_index"])): row for row in after_rows
    }
    if set(before) != set(after):
        raise QualityContractError("base/checkpoint LongBench paired slices differ")
    rows = []
    for key in sorted(before):
        left, right = before[key], after[key]
        rows.append(
            {
                "dataset": key[0],
                "source_index": key[1],
                "base_prediction": left["prediction"],
                "checkpoint_prediction": right["prediction"],
                "base_f1": float(left["f1"]),
                "checkpoint_f1": float(right["f1"]),
                "checkpoint_minus_base_f1": float(right["f1"])
                - float(left["f1"]),
            }
        )

    def mean(selected: Sequence[dict[str, Any]], field: str) -> float:
        return sum(float(row[field]) for row in selected) / len(selected)

    by_dataset = {}
    for dataset in DATASETS:
        selected = [row for row in rows if row["dataset"] == dataset]
        by_dataset[dataset] = {
            "examples": len(selected),
            "base_mean_f1": mean(selected, "base_f1"),
            "checkpoint_mean_f1": mean(selected, "checkpoint_f1"),
            "checkpoint_minus_base_mean_f1": mean(
                selected, "checkpoint_minus_base_f1"
            ),
        }
    return {
        "schema_version": "qcomem-sft-paired-longbench-validation-v1",
        "examples": len(rows),
        "base_mean_f1": mean(rows, "base_f1"),
        "checkpoint_mean_f1": mean(rows, "checkpoint_f1"),
        "checkpoint_minus_base_mean_f1": mean(
            rows, "checkpoint_minus_base_f1"
        ),
        "by_dataset": by_dataset,
        "rows": rows,
        "source_revision": FROZEN_LONGBENCH_REVISION,
        "source_indices": [
            FROZEN_VALIDATION_SOURCE_START,
            FROZEN_VALIDATION_SOURCE_END,
        ],
        "raw_test_v2_read": False,
    }
