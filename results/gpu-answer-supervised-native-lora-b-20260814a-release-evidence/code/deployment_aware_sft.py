from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


SCHEMA_VERSION = "qcomem-deployment-aware-sft-v1"
EXAMPLE_SCHEMA = "qcomem-deployment-aware-example-v1"
IGNORE_INDEX = -100
STRATA = ("domain", "general_replay", "teacher_preservation")
DATASETS = ("qasper", "2wikimqa", "tulu3_persona_if")
TRAIN_COUNTS = {"domain": 410, "general_replay": 307, "teacher_preservation": 307}
HELDOUT_COUNTS = {"domain": 26, "general_replay": 19, "teacher_preservation": 19}


class DeploymentSFTContractError(ValueError):
    pass


def stable_json(value: Any, *, pretty: bool = False) -> str:
    if pretty:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise DeploymentSFTContractError(f"{label} must be one lowercase SHA256")
    return value


@dataclass(frozen=True)
class DeploymentExample:
    input_ids: torch.Tensor
    labels: torch.Tensor
    example_id: str
    dataset: str
    stratum: str
    source_id_sha256: str
    document_id_sha256: str | None
    prompt_sha256: str
    context_sha256: str | None
    schedule_index: int | None

    @property
    def sequence_tokens(self) -> int:
        return int(self.input_ids.numel())

    @property
    def target_tokens(self) -> int:
        return int((self.labels != IGNORE_INDEX).sum().item())

    @property
    def teacher_required(self) -> bool:
        return self.stratum == "teacher_preservation"


def validate_example_row(
    row: dict[str, Any],
    *,
    split: str,
    max_sequence_tokens: int,
    row_index: int,
) -> None:
    location = f"{split} row {row_index}"
    if row.get("schema_version") != EXAMPLE_SCHEMA:
        raise DeploymentSFTContractError(f"{location} has the wrong schema")
    if row.get("source_split") != "train":
        raise DeploymentSFTContractError(f"{location} is not official-train sourced")
    if row.get("dataset") not in DATASETS or row.get("stratum") not in STRATA:
        raise DeploymentSFTContractError(f"{location} has an invalid dataset/stratum")
    if row["stratum"] == "domain" and row["dataset"] not in {"qasper", "2wikimqa"}:
        raise DeploymentSFTContractError(f"{location} domain source is invalid")
    if row["stratum"] != "domain" and row["dataset"] != "tulu3_persona_if":
        raise DeploymentSFTContractError(f"{location} replay source is invalid")
    for key in ("example_id", "source_id_sha256", "prompt_sha256"):
        require_sha256(row.get(key), f"{location}.{key}")
    for key in ("document_id_sha256", "context_sha256"):
        value = row.get(key)
        if value is not None:
            require_sha256(value, f"{location}.{key}")
    input_ids, labels = row.get("input_ids"), row.get("labels")
    if (
        not isinstance(input_ids, list)
        or not input_ids
        or not all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in input_ids)
    ):
        raise DeploymentSFTContractError(f"{location}.input_ids is invalid")
    if not isinstance(labels, list) or len(labels) != len(input_ids):
        raise DeploymentSFTContractError(f"{location}.labels is invalid")
    if len(input_ids) > max_sequence_tokens:
        raise DeploymentSFTContractError(f"{location} exceeds max sequence length")
    active = [index for index, value in enumerate(labels) if value != IGNORE_INDEX]
    if not active or active != list(range(active[0], len(labels))):
        raise DeploymentSFTContractError(f"{location} labels are not one target suffix")
    if labels[active[0] :] != input_ids[active[0] :]:
        raise DeploymentSFTContractError(f"{location} target labels differ from input IDs")
    counts = row.get("token_counts")
    if not isinstance(counts, dict) or counts != {
        "prompt": active[0],
        "target": len(active),
        "total": len(input_ids),
    }:
        raise DeploymentSFTContractError(f"{location} token counts are inconsistent")
    expected_schedule = split == "train"
    schedule = row.get("schedule_index")
    if expected_schedule:
        if isinstance(schedule, bool) or not isinstance(schedule, int) or schedule < 0:
            raise DeploymentSFTContractError(f"{location} lacks a schedule index")
    elif schedule is not None:
        raise DeploymentSFTContractError(f"{location} heldout row has a schedule index")
    if bool(row.get("teacher_target_required")) != (
        row["stratum"] == "teacher_preservation"
    ):
        raise DeploymentSFTContractError(f"{location} teacher flag is inconsistent")
    boundary = row.get("deployment_boundary")
    if not isinstance(boundary, dict):
        raise DeploymentSFTContractError(f"{location} deployment boundary is missing")
    if row["stratum"] == "domain":
        if boundary.get("applicable") is not True:
            raise DeploymentSFTContractError(f"{location} domain boundary is not applicable")
        document_ids = boundary.get("document_input_ids")
        query_ids = boundary.get("query_input_ids")
        if (
            not isinstance(document_ids, list)
            or not document_ids
            or not isinstance(query_ids, list)
            or not query_ids
            or not all(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0
                for value in document_ids + query_ids
            )
        ):
            raise DeploymentSFTContractError(f"{location} domain boundary token IDs are invalid")
        if document_ids + query_ids != input_ids[: active[0]]:
            raise DeploymentSFTContractError(
                f"{location} document+query does not exactly reconstruct prompt"
            )
        expected_boundary = {
            "document_tokens": len(document_ids),
            "query_tokens": len(query_ids),
            "prompt_tokens": active[0],
            "document_input_ids_sha256": hashlib.sha256(
                stable_json(document_ids).encode("utf-8")
            ).hexdigest(),
            "query_input_ids_sha256": hashlib.sha256(
                stable_json(query_ids).encode("utf-8")
            ).hexdigest(),
            "prompt_input_ids_sha256": hashlib.sha256(
                stable_json(input_ids[: active[0]]).encode("utf-8")
            ).hexdigest(),
            "answer_or_eos_tokens_in_query": False,
        }
        if any(boundary.get(key) != value for key, value in expected_boundary.items()):
            raise DeploymentSFTContractError(f"{location} domain boundary metadata drifted")
    elif boundary != {"applicable": False, "reason": "non_domain_replay_row"}:
        raise DeploymentSFTContractError(f"{location} non-domain boundary must be inapplicable")


class DeploymentDataset(Dataset[DeploymentExample]):
    def __init__(
        self,
        path: Path,
        *,
        split: str,
        max_sequence_tokens: int,
        expected_sha256: str,
        expected_counts: dict[str, int],
    ) -> None:
        expected_sha256 = require_sha256(expected_sha256, "expected data SHA256")
        actual = sha256_file(path)
        if actual != expected_sha256:
            raise DeploymentSFTContractError(
                f"{split} SHA256 mismatch: expected={expected_sha256}, actual={actual}"
            )
        examples: list[DeploymentExample] = []
        counts: Counter[str] = Counter()
        ids: set[str] = set()
        schedules: list[int] = []
        with path.open(encoding="utf-8") as handle:
            for row_index, line in enumerate(handle):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict) or line != stable_json(row) + "\n":
                    raise DeploymentSFTContractError(
                        f"{split} row {row_index} is not canonical JSONL"
                    )
                validate_example_row(
                    row,
                    split=split,
                    max_sequence_tokens=max_sequence_tokens,
                    row_index=row_index,
                )
                if row["example_id"] in ids:
                    raise DeploymentSFTContractError(f"{split} repeats an example ID")
                ids.add(row["example_id"])
                counts[row["stratum"]] += 1
                if row["schedule_index"] is not None:
                    schedules.append(int(row["schedule_index"]))
                examples.append(
                    DeploymentExample(
                        input_ids=torch.tensor(row["input_ids"], dtype=torch.long),
                        labels=torch.tensor(row["labels"], dtype=torch.long),
                        example_id=row["example_id"],
                        dataset=row["dataset"],
                        stratum=row["stratum"],
                        source_id_sha256=row["source_id_sha256"],
                        document_id_sha256=row["document_id_sha256"],
                        prompt_sha256=row["prompt_sha256"],
                        context_sha256=row["context_sha256"],
                        schedule_index=row["schedule_index"],
                    )
                )
        if dict(counts) != expected_counts:
            raise DeploymentSFTContractError(
                f"{split} stratum counts differ: expected={expected_counts}, actual={dict(counts)}"
            )
        if split == "train":
            if sorted(schedules) != list(range(len(examples))):
                raise DeploymentSFTContractError("train schedule is not a permutation")
            examples.sort(key=lambda example: int(example.schedule_index))
        self.examples = examples
        self.counts = dict(counts)
        self.audit = {
            "path": str(path),
            "sha256": actual,
            "rows": len(examples),
            "stratum_counts": dict(counts),
            "max_observed_sequence_tokens": max(example.sequence_tokens for example in examples),
            "min_observed_sequence_tokens": min(example.sequence_tokens for example in examples),
        }

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> DeploymentExample:
        return self.examples[index]


def validate_manifest(
    path: Path,
    *,
    expected_sha256: str,
    train_path: Path,
    train_sha256: str,
    heldout_path: Path,
    heldout_sha256: str,
) -> dict[str, Any]:
    if sha256_file(path) != require_sha256(expected_sha256, "manifest SHA256"):
        raise DeploymentSFTContractError("manifest SHA256 mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("status") != "passed":
        raise DeploymentSFTContractError("manifest did not pass the v1 contract")
    if payload.get("model_initialization") != "post_trained_qwen3.5_35b_a3b":
        raise DeploymentSFTContractError("training must initialize from the post-trained checkpoint")
    outputs = payload.get("outputs", {})
    expected = {
        "train": (train_path.name, require_sha256(train_sha256, "train SHA256"), TRAIN_COUNTS),
        "heldout": (
            heldout_path.name,
            require_sha256(heldout_sha256, "heldout SHA256"),
            HELDOUT_COUNTS,
        ),
    }
    for key, (basename, digest, counts) in expected.items():
        item = outputs.get(key, {})
        if (
            item.get("basename") != basename
            or item.get("sha256") != digest
            or item.get("stratum_counts") != counts
        ):
            raise DeploymentSFTContractError(f"manifest output binding failed for {key}")
    governance = payload.get("data_governance", {})
    required_false = (
        "longbench_validation_rows_read",
        "longbench_legacy_rows_read",
        "longbench_test_v2_rows_read",
        "validation_or_test_rows_used_for_training",
    )
    if any(governance.get(key) is not False for key in required_false):
        raise DeploymentSFTContractError("LongBench blind-set governance failed")
    if governance.get("heldout_ledger_hash_only") is not True:
        raise DeploymentSFTContractError("heldout exclusion was not hash-only")
    if payload.get("prompt_protocol", {}).get("max_sequence_tokens") != 4096:
        raise DeploymentSFTContractError("formal max sequence length is not 4096")
    audit = payload.get("audit", {})
    if audit.get("passed") is not True or audit.get("train_heldout_overlap_counts") != {
        "context_sha256": 0,
        "document_id_sha256": 0,
        "example_id": 0,
        "prompt_sha256": 0,
        "source_id_sha256": 0,
    }:
        raise DeploymentSFTContractError("manifest leakage audit failed")
    if audit.get("qasper_min_queries_per_document") != 2:
        raise DeploymentSFTContractError("QASPER document grouping contract failed")
    boundary = payload.get("prompt_protocol", {}).get("deployment_boundary_schema", {})
    if boundary != {
        "version": "qcomem-domain-document-query-boundary-v1",
        "field": "deployment_boundary",
        "applicable_exactly_when_stratum": "domain",
        "document_field": "document_input_ids",
        "query_field": "query_input_ids",
        "reconstruction": (
            "document_input_ids + query_input_ids == input_ids[:first_non_ignore_label]"
        ),
        "answer_or_eos_in_query": False,
        "per_segment_sha256": True,
        "token_list_sha256_definition": (
            "sha256(stable_json(list_ids).encode('utf-8')); stable_json uses "
            "ensure_ascii=False,sort_keys=True,separators=(',',':'), no newline"
        ),
        "all_domain_segments_nonempty": True,
    }:
        raise DeploymentSFTContractError("deployment document/query boundary schema drifted")
    if audit.get("domain_boundary_rows") != TRAIN_COUNTS["domain"] or audit.get(
        "domain_boundary_reconstruction_checked_per_row"
    ) is not True:
        raise DeploymentSFTContractError("domain deployment boundary audit failed")
    return payload


def _active_targets(labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if labels.ndim != 2 or labels.shape[1] < 2:
        raise ValueError("labels must be [batch, sequence>=2]")
    shifted = labels[:, 1:]
    active = shifted != IGNORE_INDEX
    if not torch.any(active):
        raise ValueError("labels have no active target")
    return shifted, active


def log1mexp(value: torch.Tensor) -> torch.Tensor:
    """Stable log(1-exp(x)) for x <= 0."""

    value = torch.minimum(value, torch.full_like(value, -1e-7))
    split = -math.log(2.0)
    return torch.where(value < split, torch.log1p(-torch.exp(value)), torch.log(-torch.expm1(value)))


class DeploymentAwareCausalLM(nn.Module):
    """Full-model CE plus frozen-teacher top-k/tail KL and hidden cosine matching."""

    def __init__(self, language_model: nn.Module, lm_head: nn.Module) -> None:
        super().__init__()
        self.language_model = language_model
        self.lm_head = lm_head

    @classmethod
    def from_conditional_generation(cls, model: nn.Module) -> "DeploymentAwareCausalLM":
        language_model = (
            model.model.language_model if hasattr(model.model, "language_model") else model.model
        )
        return cls(language_model, model.lm_head)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        teacher_topk_ids: torch.Tensor | None = None,
        teacher_topk_logprobs: torch.Tensor | None = None,
        teacher_tail_logprob: torch.Tensor | None = None,
        teacher_normalized_hidden: torch.Tensor | None = None,
        hard_weight: float = 0.45,
        kl_weight: float = 0.35,
        hidden_weight: float = 0.20,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        output = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        shifted, active = _active_targets(labels)
        selected_hidden = output.last_hidden_state[:, :-1, :][active]
        targets = shifted[active]
        logits = self.lm_head(selected_hidden)
        ce = F.cross_entropy(logits.float(), targets, reduction="mean")
        teacher_values = (
            teacher_topk_ids,
            teacher_topk_logprobs,
            teacher_tail_logprob,
            teacher_normalized_hidden,
        )
        if all(value is None for value in teacher_values):
            zero = ce.detach().new_zeros(())
            return ce, ce.detach(), zero, zero
        if any(value is None for value in teacher_values):
            raise ValueError("teacher targets must be supplied together")
        assert teacher_topk_ids is not None
        assert teacher_topk_logprobs is not None
        assert teacher_tail_logprob is not None
        assert teacher_normalized_hidden is not None
        if teacher_topk_ids.shape[0] != targets.numel():
            raise ValueError("teacher target positions differ from labels")
        student_logz = torch.logsumexp(logits.float(), dim=-1)
        student_topk_logits = logits.gather(1, teacher_topk_ids.long()).float()
        student_topk_logprobs = student_topk_logits - student_logz[:, None]
        student_topk_logmass = torch.logsumexp(student_topk_logprobs, dim=-1)
        student_tail_logprob = log1mexp(student_topk_logmass)
        teacher_topk_logprobs = teacher_topk_logprobs.float()
        teacher_tail_logprob = teacher_tail_logprob.float()
        topk_prob = torch.exp(teacher_topk_logprobs)
        tail_prob = torch.exp(teacher_tail_logprob)
        kl = (
            (
                topk_prob * (teacher_topk_logprobs - student_topk_logprobs)
            ).sum(dim=-1)
            + tail_prob * (teacher_tail_logprob - student_tail_logprob)
        ).mean()
        student_normalized = F.normalize(selected_hidden.float(), dim=-1)
        hidden = (
            1.0
            - (student_normalized * teacher_normalized_hidden.float()).sum(dim=-1)
        ).mean()
        loss = hard_weight * ce + kl_weight * kl + hidden_weight * hidden
        return loss, ce.detach(), kl.detach(), hidden.detach()


@torch.no_grad()
def frozen_teacher_targets(
    language_model: nn.Module,
    lm_head: nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    topk: int,
    projection_chunk_tokens: int = 32,
) -> dict[str, torch.Tensor]:
    output = language_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )
    shifted, active = _active_targets(labels)
    hidden = output.last_hidden_state[:, :-1, :][active]
    targets = shifted[active]
    if not 1 <= topk < int(lm_head.weight.shape[0]):
        raise ValueError("teacher top-k is invalid")
    ids, logprobs, tails = [], [], []
    for start in range(0, hidden.shape[0], projection_chunk_tokens):
        selected = hidden[start : start + projection_chunk_tokens]
        logits = lm_head(selected).float()
        values, indices = torch.topk(logits, k=topk, dim=-1, sorted=True)
        logz = torch.logsumexp(logits, dim=-1)
        selected_logprobs = values - logz[:, None]
        tail = log1mexp(torch.logsumexp(selected_logprobs, dim=-1))
        ids.append(indices.to(torch.int32).cpu())
        logprobs.append(selected_logprobs.cpu())
        tails.append(tail.cpu())
    return {
        "target_ids": targets.to(torch.int32).cpu(),
        "topk_ids": torch.cat(ids),
        "topk_logprobs": torch.cat(logprobs),
        "tail_logprob": torch.cat(tails),
        "normalized_hidden": F.normalize(hidden.float(), dim=-1).to(torch.bfloat16).cpu(),
    }


def summarize_example_equal(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    if not rows:
        raise ValueError("cannot summarize zero rows")

    def one(selected: Sequence[dict[str, Any]]) -> dict[str, Any]:
        values = [float(row["ce"]) for row in selected]
        if not values or not all(math.isfinite(value) for value in values):
            raise ValueError("CE rows are empty or non-finite")
        return {
            "examples": len(values),
            "target_tokens": sum(int(row["target_tokens"]) for row in selected),
            "example_equal_mean_ce": sum(values) / len(values),
        }

    return {
        "overall": one(rows),
        "by_stratum": {
            stratum: one([row for row in rows if row["stratum"] == stratum])
            for stratum in STRATA
        },
        "by_dataset": {
            dataset: one([row for row in rows if row["dataset"] == dataset])
            for dataset in DATASETS
            if any(row["dataset"] == dataset for row in rows)
        },
        "selection_metric": "overall.example_equal_mean_ce",
        "target_token_weighting_used": False,
    }


def schedule_audit(examples: Sequence[DeploymentExample], world_size: int = 8) -> dict[str, Any]:
    if len(examples) % world_size:
        raise DeploymentSFTContractError("schedule must divide exactly across ranks")
    steps = []
    for start in range(0, len(examples), world_size):
        batch = examples[start : start + world_size]
        steps.append(
            {
                "step": start // world_size + 1,
                "stratum_counts": dict(sorted(Counter(row.stratum for row in batch).items())),
                "sequence_tokens": [row.sequence_tokens for row in batch],
                "example_ids": [row.example_id for row in batch],
            }
        )
    return {
        "steps": len(steps),
        "world_size": world_size,
        "global_examples": len(examples),
        "first_step_max_sequence_tokens": max(steps[0]["sequence_tokens"]),
        "ordered_example_id_sha256": hashlib.sha256(
            "\n".join(row.example_id for row in examples).encode("ascii")
        ).hexdigest(),
        "step_records": steps,
    }
