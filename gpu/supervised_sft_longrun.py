from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from torch.utils.data import Dataset

from split_supervised_sft_scale import (
    DATASETS,
    MANIFEST_SCHEMA,
    SplitContractError,
    sha256_file,
    validate_parent_row,
)


SCALE_DATASETS = tuple(DATASETS)
SCALE_SPLIT_MANIFEST_SCHEMA = MANIFEST_SCHEMA
FORMAL_FORMAT = "qcomem_dense_full_sft_formal_v1"


@dataclass(frozen=True)
class PreparedScaleExample:
    input_ids: torch.Tensor
    labels: torch.Tensor
    dataset: str
    source_id_sha256: str
    prompt_tokens: int
    target_tokens: int

    @property
    def sequence_tokens(self) -> int:
        return int(self.input_ids.numel())


class PreparedScaleDataset(Dataset[PreparedScaleExample]):
    """Pre-tokenized train-only rows bound by the scale split manifest.

    The scale builder has already rebuilt every sequence with the frozen model
    tokenizer and downstream prompt.  The trainer nevertheless validates every
    token part, answer/EOS label and train provenance before retaining tensors.
    Raw source IDs are deliberately converted to hashes at this boundary.
    """

    def __init__(
        self,
        path: Path,
        *,
        eos_token_id: int,
        max_sequence_tokens: int,
        expected_counts: dict[str, int],
    ) -> None:
        examples: list[PreparedScaleExample] = []
        counts: Counter[str] = Counter()
        source_keys: set[tuple[str, str]] = set()
        with path.open(encoding="utf-8") as stream:
            for row_index, line in enumerate(stream):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise SplitContractError(
                        f"{path.name} row {row_index} must be an object"
                    )
                validate_parent_row(
                    row,
                    row_index=row_index,
                    eos_token_id=eos_token_id,
                    max_sequence_tokens=max_sequence_tokens,
                )
                source_id = row["source_id"]
                key = (row["dataset"], source_id)
                if key in source_keys:
                    raise SplitContractError(
                        f"{path.name} repeats dataset/source_id at row {row_index}"
                    )
                source_keys.add(key)
                source_id_sha256 = hashlib.sha256(source_id.encode("utf-8")).hexdigest()
                counts[row["dataset"]] += 1
                token_counts = row["token_counts"]
                examples.append(
                    PreparedScaleExample(
                        input_ids=torch.tensor(row["input_ids"], dtype=torch.long),
                        labels=torch.tensor(row["labels"], dtype=torch.long),
                        dataset=row["dataset"],
                        source_id_sha256=source_id_sha256,
                        prompt_tokens=int(token_counts["prompt"]),
                        target_tokens=int(token_counts["answer_with_eos"]),
                    )
                )
        if dict(counts) != expected_counts:
            raise SplitContractError(
                f"{path.name} dataset counts differ from split manifest: "
                f"expected={expected_counts}, actual={dict(counts)}"
            )
        if len(examples) != sum(expected_counts.values()):
            raise SplitContractError(f"{path.name} row count is inconsistent")
        self.examples = examples
        self.indices_by_dataset = {
            dataset: tuple(
                index
                for index, example in enumerate(examples)
                if example.dataset == dataset
            )
            for dataset in SCALE_DATASETS
        }
        self.audit = {
            "path": str(path),
            "sha256": sha256_file(path),
            "rows": len(examples),
            "dataset_counts": dict(counts),
            "all_rows_token_and_train_provenance_validated": True,
            "raw_source_ids_retained_in_runtime_metadata": False,
        }

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> PreparedScaleExample:
        return self.examples[index]


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SplitContractError(f"{label} must be one lowercase SHA256")
    return value


def validate_scale_split_manifest(
    manifest_path: Path,
    *,
    expected_manifest_sha256: str,
    train_path: Path,
    expected_train_sha256: str,
    heldout_path: Path,
    expected_heldout_sha256: str,
) -> dict[str, Any]:
    """Fail closed on the immutable train/CE-heldout split contract.

    The launcher additionally runs the independent full partition audit.  This
    lightweight trainer-side check pins the three supplied files and all
    governance facts needed before any 35B weights are loaded.
    """

    expected_manifest_sha256 = _require_sha256(
        expected_manifest_sha256, "expected_manifest_sha256"
    )
    expected_train_sha256 = _require_sha256(
        expected_train_sha256, "expected_train_sha256"
    )
    expected_heldout_sha256 = _require_sha256(
        expected_heldout_sha256, "expected_heldout_sha256"
    )
    actual = {
        "manifest": sha256_file(manifest_path),
        "train": sha256_file(train_path),
        "heldout_ce": sha256_file(heldout_path),
    }
    expected = {
        "manifest": expected_manifest_sha256,
        "train": expected_train_sha256,
        "heldout_ce": expected_heldout_sha256,
    }
    if actual != expected:
        raise SplitContractError(
            f"scale split SHA256 mismatch: expected={expected}, actual={actual}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise SplitContractError("scale split manifest must be an object")
    if manifest.get("schema_version") != SCALE_SPLIT_MANIFEST_SCHEMA:
        raise SplitContractError("scale split manifest schema drifted")
    if manifest.get("status") != "passed":
        raise SplitContractError("scale split manifest did not pass")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise SplitContractError("scale split outputs are missing")
    bindings = {
        "train_jsonl": (train_path, expected_train_sha256),
        "heldout_ce_jsonl": (heldout_path, expected_heldout_sha256),
    }
    counts: dict[str, dict[str, int]] = {}
    for name, (path, digest) in bindings.items():
        item = outputs.get(name)
        if not isinstance(item, dict):
            raise SplitContractError(f"manifest.outputs.{name} is missing")
        if item.get("basename") != path.name or item.get("sha256") != digest:
            raise SplitContractError(f"manifest.outputs.{name} artifact binding drifted")
        dataset_counts = item.get("dataset_counts")
        if (
            not isinstance(dataset_counts, dict)
            or set(dataset_counts) != set(SCALE_DATASETS)
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 4
                for value in dataset_counts.values()
            )
        ):
            raise SplitContractError(
                f"manifest.outputs.{name}.dataset_counts must contain both datasets"
            )
        if item.get("count") != sum(dataset_counts.values()):
            raise SplitContractError(f"manifest.outputs.{name}.count is inconsistent")
        if item.get("source_split_values") != ["train"]:
            raise SplitContractError(f"manifest.outputs.{name} is not train-only")
        counts[name] = {dataset: int(dataset_counts[dataset]) for dataset in SCALE_DATASETS}
    disjoint = manifest.get("disjoint_audit")
    if not isinstance(disjoint, dict) or disjoint.get("all_zero") is not True:
        raise SplitContractError("train/CE-heldout disjoint audit did not pass")
    if disjoint.get("source_id_intersection_count") != 0 or disjoint.get(
        "component_intersection_count"
    ) != 0:
        raise SplitContractError("train/CE-heldout identity components overlap")
    fingerprint_counts = disjoint.get("fingerprint_intersection_counts")
    if not isinstance(fingerprint_counts, dict) or any(fingerprint_counts.values()):
        raise SplitContractError("train/CE-heldout fingerprint sets overlap")
    governance = manifest.get("data_governance")
    if not isinstance(governance, dict):
        raise SplitContractError("data governance is missing")
    for key in (
        "validation_or_test_rows_used",
        "raw_test_v2_read",
        "heldout_ce_is_final_downstream_evaluation",
    ):
        if governance.get(key) is not False:
            raise SplitContractError(f"data_governance.{key} must remain false")
    if governance.get("all_rows_top_level_source_split") != "train" or governance.get(
        "all_rows_provenance_source_split"
    ) != "train":
        raise SplitContractError("scale artifacts must use official train sources only")
    parent = manifest.get("parent")
    if not isinstance(parent, dict):
        raise SplitContractError("parent leakage audit binding is missing")
    if (
        parent.get("output_overlap_count") != 0
        or parent.get("raw_test_v2_read_by_converter") is not False
        or parent.get("test_v2_content_hash_check") != "deferred_not_read"
        or parent.get("full_train_scan_completed") is not True
    ):
        raise SplitContractError("parent leakage/test-v2 governance drifted")
    prompt = manifest.get("prompt_protocol")
    if not isinstance(prompt, dict) or prompt.get("max_sequence_tokens") != 1024:
        raise SplitContractError("prompt protocol or sequence length drifted")
    tokenizer = manifest.get("tokenizer")
    if not isinstance(tokenizer, dict):
        raise SplitContractError("tokenizer binding is missing")
    eos = tokenizer.get("eos_token_id")
    if isinstance(eos, bool) or not isinstance(eos, int) or eos < 0:
        raise SplitContractError("tokenizer EOS binding is invalid")
    return {
        "schema_version": manifest["schema_version"],
        "manifest_sha256": actual["manifest"],
        "train_sha256": actual["train"],
        "heldout_ce_sha256": actual["heldout_ce"],
        "counts": counts,
        "tokenizer": tokenizer,
        "prompt_protocol": prompt,
        "disjoint_audit": disjoint,
        "data_governance": governance,
        "parent": parent,
        "split_config_sha256": manifest.get("split_config", {}).get("sha256"),
    }


def _epoch_order(indices: Sequence[int], *, seed: int, dataset: str, epoch: int) -> list[int]:
    order = list(indices)
    if not order:
        raise ValueError(f"dataset {dataset} has no examples")
    dataset_offset = sum((index + 1) * ord(character) for index, character in enumerate(dataset))
    random.Random(seed + 1_000_003 * epoch + dataset_offset).shuffle(order)
    return order


def _cyclic_shuffled_take(
    indices: Sequence[int],
    *,
    start: int,
    count: int,
    seed: int,
    dataset: str,
) -> list[int]:
    values: list[int] = []
    cursor = start
    while len(values) < count:
        epoch, offset = divmod(cursor, len(indices))
        order = _epoch_order(indices, seed=seed, dataset=dataset, epoch=epoch)
        available = min(count - len(values), len(order) - offset)
        values.extend(order[offset : offset + available])
        cursor += available
    return values


def balanced_global_indices(
    indices_by_dataset: dict[str, Sequence[int]],
    *,
    micro_batch_index: int,
    seed: int,
    per_dataset: int = 4,
) -> list[int]:
    """Return one frozen 4+4 global batch, deterministically shuffled by rank."""

    if micro_batch_index < 0 or per_dataset < 1:
        raise ValueError("micro_batch_index must be non-negative and per_dataset positive")
    if set(indices_by_dataset) != set(SCALE_DATASETS):
        raise ValueError("balanced schedule requires exactly qasper and 2wikimqa")
    selected: list[int] = []
    for dataset in SCALE_DATASETS:
        indices = indices_by_dataset[dataset]
        if len(indices) < per_dataset:
            raise ValueError(f"dataset {dataset} has fewer than {per_dataset} examples")
        selected.extend(
            _cyclic_shuffled_take(
                indices,
                start=micro_batch_index * per_dataset,
                count=per_dataset,
                seed=seed,
                dataset=dataset,
            )
        )
    random.Random(seed + 10_000_019 * micro_batch_index).shuffle(selected)
    return selected


def schedule_audit(
    examples: Sequence[PreparedScaleExample], indices: Sequence[int]
) -> dict[str, Any]:
    datasets = Counter(examples[index].dataset for index in indices)
    hashes = [examples[index].source_id_sha256 for index in indices]
    digest = hashlib.sha256("\n".join(hashes).encode("ascii")).hexdigest()
    return {
        "dataset_counts": dict(sorted(datasets.items())),
        "source_id_sha256": hashes,
        "ordered_source_id_sha256_digest": digest,
        "raw_source_ids_recorded": False,
    }


def global_token_weighted_rank_scale(
    *, local_target_tokens: int, global_step_target_tokens: int, world_size: int
) -> float:
    if local_target_tokens < 1 or global_step_target_tokens < local_target_tokens:
        raise ValueError("target-token counts are inconsistent")
    if world_size < 1:
        raise ValueError("world_size must be positive")
    return world_size * local_target_tokens / global_step_target_tokens


def summarize_loss_rows(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    if not rows:
        raise ValueError("cannot summarize zero loss rows")
    result: dict[str, Any] = {}
    for dataset in ("overall", *SCALE_DATASETS):
        selected = rows if dataset == "overall" else [row for row in rows if row["dataset"] == dataset]
        if not selected:
            raise ValueError(f"no rows for dataset {dataset}")
        sample_count = len(selected)
        target_tokens = sum(int(row["target_tokens"]) for row in selected)
        losses = [float(row["mean_ce"]) for row in selected]
        if target_tokens < 1 or not all(math.isfinite(value) for value in losses):
            raise ValueError(f"non-finite or empty CE rows for {dataset}")
        result[dataset] = {
            "samples": sample_count,
            "target_tokens": target_tokens,
            "sample_equal_mean_ce": sum(losses) / sample_count,
            "token_weighted_ce": sum(
                loss * int(row["target_tokens"])
                for loss, row in zip(losses, selected)
            )
            / target_tokens,
        }
    return result


def cosine_warmup_factor(step_index: int, *, warmup_steps: int, total_steps: int) -> float:
    """LR factor used *for* zero-based optimizer step ``step_index``."""

    if step_index < 0 or warmup_steps < 0 or total_steps < 1:
        raise ValueError("invalid scheduler arguments")
    if warmup_steps > total_steps:
        raise ValueError("warmup_steps may not exceed total_steps")
    if warmup_steps and step_index < warmup_steps:
        return (step_index + 1) / warmup_steps
    progress = min(
        max((step_index - warmup_steps) / max(total_steps - warmup_steps, 1), 0.0),
        1.0,
    )
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def heldout_quality_gate(
    evaluations: dict[int, dict[str, Any]], *, final_step: int
) -> dict[str, Any]:
    if 0 not in evaluations or final_step not in evaluations:
        raise ValueError("quality gate requires step-0 and final heldout evaluations")
    baseline = float(evaluations[0]["overall"]["token_weighted_ce"])
    final = float(evaluations[final_step]["overall"]["token_weighted_ce"])
    finite = math.isfinite(baseline) and math.isfinite(final)
    improved = finite and final < baseline
    return {
        "metric": "heldout_ce.overall.token_weighted_ce",
        "baseline_step": 0,
        "final_step": final_step,
        "baseline": baseline,
        "final": final,
        "absolute_change": final - baseline if finite else None,
        "finite": finite,
        "strictly_improved": improved,
        "passed": improved,
        "long_run_recommended": improved,
        "automatic_early_stop_used": False,
        "claim_boundary": (
            "train-split heldout CE diagnostic only; final downstream quality remains untested"
        ),
    }
