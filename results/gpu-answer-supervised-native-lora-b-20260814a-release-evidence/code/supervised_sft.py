from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset

from run_downstream import DATASET_MAX_NEW_TOKENS, prompt_parts


IGNORE_INDEX = -100
SUPPORTED_DATASETS = frozenset({"qasper", "2wikimqa"})
FORMAL_CODE_LEDGER_FILENAMES = frozenset(
    {
        "supervised_sft.py",
        "train_supervised_sft.py",
        "preflight_supervised_sft.py",
        "launch_supervised_sft_8gpu.sh",
        "dense_full_model_sft_smoke_1.json",
        "run_downstream.py",
    }
)
FORMAL_MODEL_LEDGER_FILENAMES = frozenset(
    {
        "config.json",
        "model.safetensors.index.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "chat_template.jinja",
    }
)
REQUIRED_FIELDS = frozenset(
    {
        "dataset",
        "source_split",
        "source_id",
        "context",
        "input",
        "answers",
        "selected_answer",
        "provenance",
    }
)


class AnswerTargetOverGenerationCapError(ValueError):
    """The complete selected answer plus EOS exceeds the eval generation cap."""

    def __init__(self, *, dataset: str, target_tokens: int, generation_cap: int) -> None:
        self.dataset = dataset
        self.target_tokens = target_tokens
        self.generation_cap = generation_cap
        super().__init__(
            f"answer plus EOS has {target_tokens} tokens, over the frozen "
            f"{dataset} generation cap {generation_cap}"
        )


@dataclass(frozen=True)
class SupervisedSFTExample:
    input_ids: torch.Tensor
    labels: torch.Tensor
    dataset: str
    source_id: str
    prompt_tokens: int
    target_tokens: int
    prefix_tokens: int
    context_tokens: int
    original_context_tokens: int
    answer_tokens: int
    eos_token_id: int

    @property
    def sequence_tokens(self) -> int:
        return int(self.input_ids.numel())

    @property
    def context_was_truncated(self) -> bool:
        return self.context_tokens < self.original_context_tokens


def _path_tokens(path: Path) -> list[str]:
    return [
        token
        for token in re.split(r"[^a-z0-9]+", str(path).lower())
        if token
    ]


def assert_train_only_path(path: Path) -> None:
    """Reject paths whose components explicitly name held-out data.

    Row-level ``source_split`` and provenance checks below are authoritative;
    this path check catches accidental use of the frozen benchmark files before
    any tokenization begins.
    """

    forbidden = {"validation", "valid", "dev", "test", "testv2", "eval"}
    tokens = set(_path_tokens(path))
    if tokens & forbidden:
        raise ValueError(
            f"supervised SFT accepts train-only JSONL; held-out path rejected: {path}"
        )


def _require_nonempty_string(row: dict[str, Any], key: str, line_number: int) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"line {line_number}: {key} must be a non-empty string")
    return value


def validate_supervised_row(row: dict[str, Any], *, line_number: int) -> None:
    missing = sorted(REQUIRED_FIELDS - row.keys())
    if missing:
        raise ValueError(f"line {line_number}: missing schema fields: {missing}")

    dataset = _require_nonempty_string(row, "dataset", line_number)
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(
            f"line {line_number}: unsupported dataset {dataset!r}; "
            f"expected one of {sorted(SUPPORTED_DATASETS)}"
        )
    source_split = _require_nonempty_string(row, "source_split", line_number)
    if source_split != "train":
        raise ValueError(
            f"line {line_number}: source_split must be exactly 'train', got "
            f"{source_split!r}"
        )
    _require_nonempty_string(row, "source_id", line_number)
    _require_nonempty_string(row, "context", line_number)
    _require_nonempty_string(row, "input", line_number)
    selected_answer = _require_nonempty_string(row, "selected_answer", line_number)

    answers = row.get("answers")
    if (
        not isinstance(answers, list)
        or not answers
        or any(not isinstance(answer, str) or not answer.strip() for answer in answers)
    ):
        raise ValueError(
            f"line {line_number}: answers must be a non-empty list of non-empty strings"
        )
    if selected_answer not in answers:
        raise ValueError(
            f"line {line_number}: selected_answer must occur verbatim in answers"
        )

    provenance = row.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"line {line_number}: provenance must be an object")
    provenance_split = provenance.get("source_split")
    if provenance_split != "train":
        raise ValueError(
            f"line {line_number}: provenance.source_split must be exactly 'train', "
            f"got {provenance_split!r}"
        )


def resolve_single_eos_token_id(tokenizer: Any) -> int:
    eos = getattr(tokenizer, "eos_token_id", None)
    if not isinstance(eos, int) or isinstance(eos, bool) or eos < 0:
        raise ValueError("tokenizer.eos_token_id must be one non-negative integer")
    return eos


def build_supervised_example(
    tokenizer: Any,
    row: dict[str, Any],
    *,
    max_sequence_tokens: int,
    line_number: int = 1,
) -> SupervisedSFTExample:
    """Build ``document + query + answer + EOS`` with answer-only labels.

    ``prompt_parts`` is the single source of truth for the downstream chat
    template and symmetric context truncation.  We reserve the answer and EOS
    first, so truncation can only remove context and never target tokens.
    """

    validate_supervised_row(row, line_number=line_number)
    if max_sequence_tokens < 1:
        raise ValueError("max_sequence_tokens must be positive")
    eos_token_id = resolve_single_eos_token_id(tokenizer)
    answer_ids = list(
        tokenizer.encode(row["selected_answer"], add_special_tokens=False)
    )
    if not answer_ids:
        raise ValueError(f"line {line_number}: selected_answer tokenized to zero tokens")
    if eos_token_id in answer_ids:
        raise ValueError(
            f"line {line_number}: selected_answer already contains EOS; exactly one "
            "trainer-appended EOS is required"
        )

    target_ids = answer_ids + [eos_token_id]
    generation_cap = DATASET_MAX_NEW_TOKENS[row["dataset"]]
    if len(target_ids) > generation_cap:
        raise AnswerTargetOverGenerationCapError(
            dataset=row["dataset"],
            target_tokens=len(target_ids),
            generation_cap=generation_cap,
        )
    prompt_budget = max_sequence_tokens - len(target_ids)
    if prompt_budget < 1:
        raise ValueError(
            f"line {line_number}: answer plus EOS leaves no prompt token budget"
        )
    try:
        (
            document_ids,
            query_ids,
            prefix_tokens,
            context_tokens,
            original_context_tokens,
        ) = prompt_parts(tokenizer, row, prompt_budget)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"line {line_number}: cannot fit the frozen downstream prompt while "
            "preserving the complete answer and EOS"
        ) from error

    prompt_ids = torch.cat([document_ids, query_ids]).to(dtype=torch.long)
    if prompt_ids.numel() < 1:
        raise ValueError(f"line {line_number}: downstream prompt is empty")
    target_tensor = torch.tensor(target_ids, dtype=torch.long)
    input_ids = torch.cat([prompt_ids, target_tensor])
    if input_ids.numel() > max_sequence_tokens:
        raise RuntimeError("prompt_parts exceeded its reserved token budget")
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    labels[prompt_ids.numel() :] = target_tensor
    if int((labels != IGNORE_INDEX).sum().item()) != len(target_ids):
        raise RuntimeError("answer-only label mask has the wrong target count")
    if int(labels[-1].item()) != eos_token_id:
        raise RuntimeError("EOS must be the final supervised target")

    return SupervisedSFTExample(
        input_ids=input_ids,
        labels=labels,
        dataset=row["dataset"],
        source_id=row["source_id"],
        prompt_tokens=int(prompt_ids.numel()),
        target_tokens=len(target_ids),
        prefix_tokens=int(prefix_tokens),
        context_tokens=int(context_tokens),
        original_context_tokens=int(original_context_tokens),
        answer_tokens=len(answer_ids),
        eos_token_id=eos_token_id,
    )


class SupervisedSFTDataset(Dataset[SupervisedSFTExample]):
    """Validated train-only unified JSONL, tokenized with the eval prompt."""

    def __init__(
        self,
        path: Path,
        tokenizer: Any,
        *,
        max_sequence_tokens: int,
        limit: int,
    ) -> None:
        if limit < 1:
            raise ValueError("limit must be positive")
        assert_train_only_path(path)
        examples: list[SupervisedSFTExample] = []
        dataset_counts: Counter[str] = Counter()
        source_keys: set[tuple[str, str]] = set()
        total_records = 0
        with path.open() as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                total_records += 1
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"line {line_number}: each JSONL record must be an object")
                # Validate every row even after ``limit`` so a mixed train/test
                # file cannot hide held-out records beyond the smoke slice.
                validate_supervised_row(row, line_number=line_number)
                key = (row["dataset"], row["source_id"])
                if key in source_keys:
                    raise ValueError(f"line {line_number}: duplicate source key {key}")
                source_keys.add(key)
                dataset_counts[row["dataset"]] += 1
                if len(examples) < limit:
                    examples.append(
                        build_supervised_example(
                            tokenizer,
                            row,
                            max_sequence_tokens=max_sequence_tokens,
                            line_number=line_number,
                        )
                    )
        if total_records < limit:
            raise ValueError(
                f"training file has {total_records} records, fewer than smoke limit {limit}"
            )
        self.examples = examples
        self.audit = {
            "schema": "qcomem-supervised-qa-v1",
            "required_fields": sorted(REQUIRED_FIELDS),
            "source_split": "train",
            "provenance_source_split": "train",
            "total_records_validated": total_records,
            "examples_tokenized": len(examples),
            "dataset_counts_all_records": dict(sorted(dataset_counts.items())),
            "unique_source_keys": len(source_keys),
            "all_rows_validated_before_limit": True,
            "prompt_builder": "run_downstream.prompt_parts",
            "optional_training_target_ignored_and_regenerated": True,
        }

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> SupervisedSFTExample:
        return self.examples[index]


def single_example_collate(
    batch: Sequence[SupervisedSFTExample],
) -> dict[str, Any]:
    if len(batch) != 1:
        raise ValueError("dense full-model SFT smoke requires per-rank batch size 1")
    example = batch[0]
    return {
        "input_ids": example.input_ids.unsqueeze(0),
        "labels": example.labels.unsqueeze(0),
        "attention_mask": torch.ones_like(example.input_ids).unsqueeze(0),
        "dataset": example.dataset,
        "source_id": example.source_id,
        "prompt_tokens": example.prompt_tokens,
        "target_tokens": example.target_tokens,
        "sequence_tokens": example.sequence_tokens,
        "context_was_truncated": example.context_was_truncated,
    }


def _shifted_active_targets(
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if labels.ndim != 2:
        raise ValueError("labels must have shape [batch, sequence]")
    if labels.shape[1] < 2:
        raise ValueError("causal CE requires at least two sequence positions")
    if torch.any(labels[:, 0] != IGNORE_INDEX):
        raise ValueError("position zero cannot be supervised by a causal predecessor")
    shifted_labels = labels[:, 1:].contiguous()
    active = shifted_labels != IGNORE_INDEX
    if not torch.any(active):
        raise ValueError("labels contain no answer/EOS targets")
    return shifted_labels, active


def answer_only_causal_ce(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Reference causal shift used by tiny alignment tests."""

    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, sequence, vocabulary]")
    if logits.shape[:2] != labels.shape:
        raise ValueError("logits and labels batch/sequence dimensions must match")
    shifted_labels, active = _shifted_active_targets(labels)
    selected_logits = logits[:, :-1, :][active]
    selected_targets = shifted_labels[active]
    return (
        F.cross_entropy(selected_logits.float(), selected_targets, reduction="mean"),
        int(selected_targets.numel()),
    )


def answer_only_hidden_ce(
    hidden_states: torch.Tensor,
    lm_head: nn.Module,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Project only positions that predict answer/EOS, avoiding prompt logits."""

    if hidden_states.ndim != 3:
        raise ValueError("hidden_states must have shape [batch, sequence, hidden]")
    if hidden_states.shape[:2] != labels.shape:
        raise ValueError("hidden_states and labels dimensions must match")
    shifted_labels, active = _shifted_active_targets(labels)
    selected_hidden = hidden_states[:, :-1, :][active]
    selected_targets = shifted_labels[active]
    selected_logits = lm_head(selected_hidden)
    return (
        F.cross_entropy(selected_logits.float(), selected_targets, reduction="mean"),
        int(selected_targets.numel()),
    )


class DenseSupervisedCausalLM(nn.Module):
    """Text-only full-model supervised CE core, with no LoRA/distillation path."""

    def __init__(self, language_model: nn.Module, lm_head: nn.Module) -> None:
        super().__init__()
        self.language_model = language_model
        self.lm_head = lm_head

    @classmethod
    def from_conditional_generation(cls, model: nn.Module) -> "DenseSupervisedCausalLM":
        if hasattr(model.model, "language_model"):
            language_model = model.model.language_model
        else:
            language_model = model.model
        return cls(language_model, model.lm_head)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        loss, _ = answer_only_hidden_ce(output.last_hidden_state, self.lm_head, labels)
        return loss


def _unique_parameters(module: nn.Module) -> list[nn.Parameter]:
    seen: set[int] = set()
    values: list[nn.Parameter] = []
    for parameter in module.parameters():
        if id(parameter) not in seen:
            seen.add(id(parameter))
            values.append(parameter)
    return values


def configure_dense_full_model_trainability(model: nn.Module) -> dict[str, Any]:
    """Enable every remaining text-model parameter and return an exact ledger."""

    adapter_like_modules = [
        name
        for name, module in model.named_modules()
        if "lora" in type(module).__name__.lower()
        or hasattr(module, "lora_A")
        or hasattr(module, "lora_B")
    ]
    if adapter_like_modules:
        raise ValueError(
            "dense_full_model_sft_smoke refuses LoRA/adapter modules: "
            f"{adapter_like_modules[:8]}"
        )
    parameters = _unique_parameters(model)
    if not parameters:
        raise ValueError("model contains no parameters")
    for parameter in parameters:
        parameter.requires_grad_(True)
    frozen = [name for name, parameter in model.named_parameters() if not parameter.requires_grad]
    if frozen:
        raise RuntimeError(f"full-model SFT left frozen parameters: {frozen[:8]}")
    dtype_counts: Counter[str] = Counter()
    logical_bytes = 0
    for parameter in parameters:
        dtype_counts[str(parameter.dtype)] += parameter.numel()
        logical_bytes += parameter.numel() * parameter.element_size()
    count = sum(parameter.numel() for parameter in parameters)
    return {
        "training_scope": "dense_full_model_sft_smoke",
        "objective_family": "supervised_token_cross_entropy",
        "all_model_parameters_trainable": True,
        "trainable_parameters": count,
        "total_parameters": count,
        "trainable_matches_total": True,
        "trainable_logical_bytes": logical_bytes,
        "parameter_dtype_counts": dict(sorted(dtype_counts.items())),
        "lora_used": False,
        "distillation_used": False,
        "quantization_used": False,
    }


def qcomem_suffix_supervised_sft_capability_gate() -> dict[str, Any]:
    """Fail-closed boundary for the not-yet-valid cached suffix SFT path."""

    capabilities = {
        "teacher_forced_full_query_cached_equivalence_validated": False,
        "answer_chunk_then_stepwise_decode_equivalence_validated": False,
        "gated_delta_recurrent_cache_autograd_validated": False,
        "mutable_attention_cache_backward_replay_validated": False,
        "answer_label_alignment_across_split_cache_validated": False,
        "real_model_token_and_logit_gate_passed": False,
    }
    return {
        "requested_scope": "qcomem_suffix_supervised_sft",
        "implemented": False,
        "fail_closed": True,
        "capability_gate_passed": all(capabilities.values()),
        "capabilities": capabilities,
        "forbidden_claim": (
            "Dense teacher-forced CE cannot be relabeled as Q-CoMem suffix SFT until "
            "cached query and answer/decode chunk semantics pass real-model gates."
        ),
        "observed_blocker": {
            "trial_id": 1830867,
            "result": "failed",
            "scope": "cached-two-stage mutable-cache autograd smoke",
            "implication": (
                "Qwen3.5 mutable suffix cache state did not support the attempted "
                "training backward path; cached suffix supervised CE remains closed"
            ),
            "dense_full_model_sft_affected": False,
        },
        "required_before_implementation": [
            (
                "compare uncached document+query+answer teacher forcing against "
                "cached document prefill plus full-query continuation at every "
                "query position"
            ),
            (
                "extend that comparison through answer teacher forcing and "
                "one-token-at-a-time deployment decode"
            ),
            (
                "validate backward through Qwen3.5 full-attention and "
                "GatedDeltaNet mutable cache state without duplicate mutation "
                "under checkpoint recomputation; Trial 1830867 failed this class "
                "of cached-two-stage autograd smoke"
            ),
            "prove the first-answer and EOS causal label positions across the split boundary",
            "pass strict token/logit agreement gates before any downstream quality claim",
        ],
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{field} must be one lowercase 64-hex SHA256")
    return value


def validate_sha256_ledger(
    ledger_path: Path,
    *,
    expected_ledger_sha256: str,
    required_filenames: frozenset[str],
    ledger_name: str,
) -> dict[str, Any]:
    """Verify a pinned sha256sum ledger and every artifact it names."""

    expected_ledger_sha256 = _require_sha256(
        expected_ledger_sha256,
        field=f"expected_{ledger_name}_ledger_sha256",
    )
    actual_ledger_sha256 = file_sha256(ledger_path)
    if actual_ledger_sha256 != expected_ledger_sha256:
        raise ValueError(
            f"{ledger_name} ledger SHA256 mismatch: "
            f"expected={expected_ledger_sha256}, actual={actual_ledger_sha256}"
        )
    entries: list[dict[str, str]] = []
    paths: set[Path] = set()
    filenames: set[str] = set()
    for line_number, line in enumerate(ledger_path.read_text().splitlines(), 1):
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None:
            raise ValueError(
                f"{ledger_name} ledger line {line_number} is not sha256sum format"
            )
        expected_file_sha256, raw_path = match.groups()
        artifact_path = Path(raw_path)
        if not artifact_path.is_absolute():
            raise ValueError(
                f"{ledger_name} ledger paths must be absolute: {artifact_path}"
            )
        if artifact_path in paths:
            raise ValueError(
                f"{ledger_name} ledger repeats artifact path {artifact_path}"
            )
        paths.add(artifact_path)
        filenames.add(artifact_path.name)
        if not artifact_path.is_file():
            raise ValueError(
                f"{ledger_name} ledger artifact is missing: {artifact_path}"
            )
        actual_file_sha256 = file_sha256(artifact_path)
        if actual_file_sha256 != expected_file_sha256:
            raise ValueError(
                f"{ledger_name} artifact SHA256 drift for {artifact_path}: "
                f"expected={expected_file_sha256}, actual={actual_file_sha256}"
            )
        entries.append(
            {
                "filename": artifact_path.name,
                "path": str(artifact_path),
                "sha256": actual_file_sha256,
            }
        )
    if filenames != set(required_filenames):
        raise ValueError(
            f"{ledger_name} ledger filenames mismatch: "
            f"expected={sorted(required_filenames)}, actual={sorted(filenames)}"
        )
    return {
        "ledger_path": str(ledger_path),
        "ledger_sha256": actual_ledger_sha256,
        "entries": entries,
        "required_filenames": sorted(required_filenames),
        "all_artifacts_exist_and_match": True,
    }


def validate_formal_integrity_ledgers(
    *,
    code_ledger_path: Path,
    expected_code_ledger_sha256: str,
    model_ledger_path: Path,
    expected_model_ledger_sha256: str,
) -> dict[str, Any]:
    return {
        "code": validate_sha256_ledger(
            code_ledger_path,
            expected_ledger_sha256=expected_code_ledger_sha256,
            required_filenames=FORMAL_CODE_LEDGER_FILENAMES,
            ledger_name="code",
        ),
        "model_artifacts": validate_sha256_ledger(
            model_ledger_path,
            expected_ledger_sha256=expected_model_ledger_sha256,
            required_filenames=FORMAL_MODEL_LEDGER_FILENAMES,
            ledger_name="model_artifact",
        ),
    }


def _validate_manifest_tokenizer_metadata(manifest: dict[str, Any]) -> dict[str, Any]:
    expected = manifest.get("tokenizer")
    if not isinstance(expected, dict):
        raise ValueError("formal manifest.tokenizer must be an object")
    requested = expected.get("requested_name_or_path")
    if not isinstance(requested, str) or not requested.strip():
        raise ValueError(
            "manifest.tokenizer.requested_name_or_path must be non-empty"
        )
    requested_revision = expected.get("requested_revision")
    if not isinstance(requested_revision, str) or not requested_revision.strip():
        raise ValueError("manifest.tokenizer.requested_revision must be non-empty")
    if "59d61f3" not in requested and "59d61f3" not in requested_revision:
        raise ValueError(
            "manifest tokenizer path/revision must bind frozen model identifier "
            "59d61f3"
        )
    tokenizer_class = expected.get("class")
    if not isinstance(tokenizer_class, str) or not tokenizer_class.strip():
        raise ValueError("manifest.tokenizer.class must be non-empty")
    vocab_size = expected.get("vocab_size")
    if isinstance(vocab_size, bool) or not isinstance(vocab_size, int) or vocab_size < 1:
        raise ValueError("manifest.tokenizer.vocab_size must be a positive integer")
    eos_token_id = expected.get("eos_token_id")
    if (
        isinstance(eos_token_id, bool)
        or not isinstance(eos_token_id, int)
        or eos_token_id < 0
    ):
        raise ValueError(
            "manifest.tokenizer.eos_token_id must be a non-negative integer"
        )
    chat_template_sha256 = _require_sha256(
        expected.get("chat_template_sha256"),
        field="manifest.tokenizer.chat_template_sha256",
    )
    resolved = expected.get("resolved_commit_hash")
    if resolved is not None and (
        not isinstance(resolved, str) or not resolved.strip()
    ):
        raise ValueError(
            "manifest.tokenizer.resolved_commit_hash must be null or non-empty"
        )
    return {
        "requested_name_or_path": requested,
        "requested_revision": requested_revision,
        "resolved_commit_hash": resolved,
        "class": tokenizer_class,
        "vocab_size": vocab_size,
        "eos_token_id": eos_token_id,
        "chat_template_sha256": chat_template_sha256,
    }


def validate_prepared_training_manifest(
    manifest_path: Path,
    data_path: Path,
    *,
    expected_data_sha256: str,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    """Bind a formal train JSONL to its converter/leakage-audit manifest."""

    expected_data_sha256 = _require_sha256(
        expected_data_sha256, field="expected_data_sha256"
    )
    expected_manifest_sha256 = _require_sha256(
        expected_manifest_sha256, field="expected_manifest_sha256"
    )
    actual_data_sha256 = file_sha256(data_path)
    actual_manifest_sha256 = file_sha256(manifest_path)
    if actual_data_sha256 != expected_data_sha256:
        raise ValueError(
            "prepared train JSONL SHA256 mismatch: "
            f"expected={expected_data_sha256}, actual={actual_data_sha256}"
        )
    if actual_manifest_sha256 != expected_manifest_sha256:
        raise ValueError(
            "converter manifest SHA256 mismatch: "
            f"expected={expected_manifest_sha256}, actual={actual_manifest_sha256}"
        )

    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict):
        raise ValueError("converter manifest must be a JSON object")
    if manifest.get("schema_version") != "qcomem-supervised-qa-v1":
        raise ValueError("unexpected supervised converter schema_version")
    if manifest.get("status") != "passed":
        raise ValueError(
            f"converter manifest status must be passed, got {manifest.get('status')!r}"
        )
    if manifest.get("mode") != "build":
        raise ValueError("formal supervised training requires a build-mode manifest")
    tokenizer_metadata = _validate_manifest_tokenizer_metadata(manifest)
    prompt_protocol = manifest.get("prompt_protocol")
    if not isinstance(prompt_protocol, dict):
        raise ValueError("manifest.prompt_protocol must be an object")
    expected_prompt_protocol = {
        "function": "run_downstream.prompt_parts",
        "target_builder": "supervised_sft.build_supervised_example",
        "max_sequence_tokens": 1024,
        "answer_tokens_reserved_before_prompt_truncation": True,
        "label_ignore_index": IGNORE_INDEX,
        "answer_eos_appended": True,
    }
    for key, expected_value in expected_prompt_protocol.items():
        if prompt_protocol.get(key) != expected_value:
            raise ValueError(
                f"manifest.prompt_protocol.{key} must be {expected_value!r}, got "
                f"{prompt_protocol.get(key)!r}"
            )
    prompt_source_sha256 = _require_sha256(
        prompt_protocol.get("source_file_sha256"),
        field="manifest.prompt_protocol.source_file_sha256",
    )
    target_builder_sha256 = _require_sha256(
        prompt_protocol.get("target_builder_source_file_sha256"),
        field="manifest.prompt_protocol.target_builder_source_file_sha256",
    )
    current_prompt_sha256 = file_sha256(Path(__file__).with_name("run_downstream.py"))
    current_target_builder_sha256 = file_sha256(Path(__file__))
    if prompt_source_sha256 != current_prompt_sha256:
        raise ValueError(
            "converter/evaluation prompt source drift: manifest run_downstream.py "
            "SHA does not match runtime"
        )
    if target_builder_sha256 != current_target_builder_sha256:
        raise ValueError(
            "converter/trainer target builder drift: manifest supervised_sft.py "
            "SHA does not match runtime"
        )
    output_sha256 = _require_sha256(
        manifest.get("output_jsonl_sha256"),
        field="manifest.output_jsonl_sha256",
    )
    if output_sha256 != actual_data_sha256:
        raise ValueError(
            "manifest.output_jsonl_sha256 does not bind the supplied train JSONL"
        )
    output_jsonl = manifest.get("output_jsonl")
    if not isinstance(output_jsonl, str) or not output_jsonl.strip():
        raise ValueError("manifest.output_jsonl must be a non-empty path")
    if Path(output_jsonl).name != data_path.name:
        raise ValueError(
            "manifest.output_jsonl basename does not match the supplied train JSONL"
        )
    detected_overlap_count = manifest.get("detected_overlap_count")
    output_overlap_count = manifest.get("output_overlap_count")
    if (
        isinstance(detected_overlap_count, bool)
        or not isinstance(detected_overlap_count, int)
        or detected_overlap_count < 0
    ):
        raise ValueError(
            "manifest.detected_overlap_count must be a non-negative integer"
        )
    if isinstance(output_overlap_count, bool) or output_overlap_count != 0:
        raise ValueError(
            "manifest.output_overlap_count must equal integer zero for the "
            "published train JSONL"
        )
    overlap_report = manifest.get("overlap_report")
    if not isinstance(overlap_report, list) or len(overlap_report) != detected_overlap_count:
        raise ValueError(
            "manifest overlap_report length must equal detected_overlap_count"
        )
    report_dataset_counts: Counter[str] = Counter()
    allowed_fingerprint_kinds = {
        "id_sha256",
        "context_input_sha256",
        "context_sha256",
        "input_sha256",
    }
    for index, report in enumerate(overlap_report):
        if not isinstance(report, dict) or set(report) != {
            "dataset",
            "train_source_id_sha256",
            "matches",
        }:
            raise ValueError(
                f"manifest.overlap_report[{index}] must contain hash/reference fields only"
            )
        dataset = report["dataset"]
        if dataset not in SUPPORTED_DATASETS:
            raise ValueError(f"invalid overlap-report dataset {dataset!r}")
        _require_sha256(
            report["train_source_id_sha256"],
            field=f"manifest.overlap_report[{index}].train_source_id_sha256",
        )
        matches = report["matches"]
        if not isinstance(matches, list) or not matches:
            raise ValueError(f"manifest.overlap_report[{index}].matches must be non-empty")
        for match_index, match in enumerate(matches):
            if not isinstance(match, dict) or set(match) != {
                "fingerprint_kind",
                "dataset",
                "split",
                "source_index",
            }:
                raise ValueError(
                    f"manifest overlap match {index}/{match_index} must be a reference only"
                )
            if match["fingerprint_kind"] not in allowed_fingerprint_kinds:
                raise ValueError("manifest overlap match has an unknown fingerprint kind")
            if match["dataset"] not in SUPPORTED_DATASETS:
                raise ValueError("manifest overlap match has an unknown dataset")
            if not isinstance(match["split"], str) or not match["split"]:
                raise ValueError("manifest overlap match split must be non-empty")
            if isinstance(match["source_index"], bool) or not isinstance(
                match["source_index"], int
            ):
                raise ValueError("manifest overlap match source_index must be an integer")
        report_dataset_counts[dataset] += 1

    heldout = manifest.get("heldout_protocol")
    if not isinstance(heldout, dict):
        raise ValueError("manifest.heldout_protocol must be an object")
    content_hash_check = heldout.get("test_v2_content_hash_check")
    if content_hash_check not in {"deferred_not_read", "blind_hash_manifest"}:
        raise ValueError(
            "heldout test-v2 status must be deferred_not_read or blind_hash_manifest"
        )
    if heldout.get("raw_test_v2_read_by_converter") is not False:
        raise ValueError("converter must attest raw_test_v2_read_by_converter=false")
    overlap_policy = heldout.get("overlap_policy")
    if overlap_policy not in {"fail", "drop"}:
        raise ValueError("manifest heldout overlap_policy must be fail or drop")

    output_selection = manifest.get("output_selection")
    expected_selection = {
        "strategy": "first_n_target_valid_eligible_in_official_source_order-v1",
        "requested_max_output_per_dataset": 4,
        "max_output_per_dataset": 4,
        "full_train_scan_completed": True,
        "selection_applied_after_overlap_filter": True,
        "target_validity_checked_before_selection": True,
        "answer_over_cap_policy": "skip_complete_answer_without_truncation",
        "written_smoke_count": 8,
        "written_jsonl_count": 8,
    }
    if not isinstance(output_selection, dict):
        raise ValueError("manifest.output_selection must be an object")
    for key, expected in expected_selection.items():
        if output_selection.get(key) != expected:
            raise ValueError(
                f"manifest.output_selection.{key} must be {expected!r}, got "
                f"{output_selection.get(key)!r}"
            )

    source_spec = manifest.get("source_spec")
    datasets = source_spec.get("datasets") if isinstance(source_spec, dict) else None
    if not isinstance(datasets, dict):
        raise ValueError("manifest.source_spec.datasets must be an object")
    dataset_stats = manifest.get("dataset_stats")
    if not isinstance(dataset_stats, dict):
        raise ValueError("manifest.dataset_stats must be an object")
    written_examples: dict[str, int] = {}
    eligible_examples: dict[str, int] = {}
    full_eligible_examples: dict[str, int] = {}
    selected_for_output_examples: dict[str, int] = {}
    dropped_examples: dict[str, int] = {}
    overlap_examples: dict[str, int] = {}
    skipped_answer_over_cap: dict[str, int] = {}
    skipped_answer_source_hashes: dict[str, list[str]] = {}
    source_fingerprints: dict[str, dict[str, str]] = {}
    for dataset in sorted(SUPPORTED_DATASETS):
        spec = datasets.get(dataset)
        if not isinstance(spec, dict) or spec.get("source_split") != "train":
            raise ValueError(
                f"manifest source_spec.datasets.{dataset}.source_split must be train"
            )
        fingerprints: dict[str, str] = {}
        for key in ("source_revision", "archive_sha256", "extracted_file_sha256", "license"):
            value = spec.get(key)
            if key.endswith("sha256"):
                value = _require_sha256(
                    value, field=f"manifest.source_spec.datasets.{dataset}.{key}"
                )
            elif not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"manifest.source_spec.datasets.{dataset}.{key} must be non-empty"
                )
            fingerprints[key] = value
        source_fingerprints[dataset] = fingerprints
        stats = dataset_stats.get(dataset)
        if not isinstance(stats, dict):
            raise ValueError(f"manifest.dataset_stats.{dataset} must be an object")
        values: dict[str, int] = {}
        for key in (
            "parsed_examples",
            "overlap_examples",
            "dropped_examples",
            "eligible_examples",
            "full_eligible_examples",
            "selected_for_output_examples",
            "written_examples",
        ):
            value = stats.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"manifest.dataset_stats.{dataset}.{key} must be non-negative"
                )
            values[key] = value
        if values["written_examples"] < 1:
            raise ValueError(
                f"manifest.dataset_stats.{dataset}.written_examples must be positive"
            )
        if values["eligible_examples"] != values["full_eligible_examples"]:
            raise ValueError(
                f"manifest {dataset} eligible_examples must equal the separately "
                "audited full_eligible_examples"
            )
        if values["parsed_examples"] != (
            values["full_eligible_examples"] + values["overlap_examples"]
        ):
            raise ValueError(
                f"manifest {dataset} parsed count does not partition into "
                "eligible plus overlap examples"
            )
        skipped_count = stats.get("output_selection_skipped_answer_over_cap")
        if (
            isinstance(skipped_count, bool)
            or not isinstance(skipped_count, int)
            or skipped_count < 0
            or skipped_count > values["full_eligible_examples"]
        ):
            raise ValueError(
                f"manifest {dataset} output_selection_skipped_answer_over_cap "
                "must be between zero and full_eligible_examples"
            )
        skipped_hashes = stats.get(
            "output_selection_skipped_answer_over_cap_source_id_sha256"
        )
        if not isinstance(skipped_hashes, list) or len(skipped_hashes) != skipped_count:
            raise ValueError(
                f"manifest {dataset} skipped-answer hash list length must equal "
                "output_selection_skipped_answer_over_cap"
            )
        for index, digest in enumerate(skipped_hashes):
            _require_sha256(
                digest,
                field=(
                    f"manifest.dataset_stats.{dataset}."
                    "output_selection_skipped_answer_over_cap_source_id_sha256"
                    f"[{index}]"
                ),
            )
        if len(set(skipped_hashes)) != len(skipped_hashes):
            raise ValueError(f"manifest {dataset} skipped-answer hashes must be unique")
        expected_written = min(
            values["full_eligible_examples"] - skipped_count, 4
        )
        if values["selected_for_output_examples"] != expected_written:
            raise ValueError(
                f"manifest {dataset} selected_for_output_examples must equal "
                f"min(full_eligible_examples, 4)={expected_written}"
            )
        if values["written_examples"] != values["selected_for_output_examples"]:
            raise ValueError(
                f"manifest {dataset} written_examples must equal "
                "selected_for_output_examples, not full eligible count"
            )
        written_examples[dataset] = values["written_examples"]
        eligible_examples[dataset] = values["eligible_examples"]
        full_eligible_examples[dataset] = values["full_eligible_examples"]
        selected_for_output_examples[dataset] = values[
            "selected_for_output_examples"
        ]
        dropped_examples[dataset] = values["dropped_examples"]
        overlap_examples[dataset] = values["overlap_examples"]
        skipped_answer_over_cap[dataset] = skipped_count
        skipped_answer_source_hashes[dataset] = skipped_hashes
    manifest_skipped = output_selection.get("skipped_answer_over_cap")
    expected_skipped = {
        dataset: {
            "count": skipped_answer_over_cap[dataset],
            "source_id_sha256": skipped_answer_source_hashes[dataset],
        }
        for dataset in sorted(SUPPORTED_DATASETS)
    }
    if manifest_skipped != expected_skipped:
        raise ValueError(
            "manifest.output_selection.skipped_answer_over_cap must exactly mirror "
            "per-dataset count/hash audit"
        )
    if sum(overlap_examples.values()) != detected_overlap_count:
        raise ValueError(
            "sum(dataset_stats.*.overlap_examples) must equal detected_overlap_count"
        )
    if dict(report_dataset_counts) != {
        dataset: count for dataset, count in overlap_examples.items() if count
    }:
        raise ValueError(
            "overlap_report per-dataset counts must match dataset_stats overlap_examples"
        )
    if overlap_policy == "fail":
        if detected_overlap_count != 0 or sum(dropped_examples.values()) != 0:
            raise ValueError(
                "passed fail-policy manifest requires zero detected/dropped overlaps"
            )
    else:
        if sum(dropped_examples.values()) != detected_overlap_count:
            raise ValueError(
                "drop-policy manifest must prove sum(dropped_examples) equals "
                "detected_overlap_count"
            )
        if dropped_examples != overlap_examples:
            raise ValueError(
                "drop-policy manifest must drop every per-dataset overlap example"
            )
    if sum(written_examples.values()) != 8:
        raise ValueError("formal smoke manifest must publish exactly eight examples")

    return {
        "schema_version": manifest["schema_version"],
        "manifest_path": str(manifest_path),
        "manifest_sha256": actual_manifest_sha256,
        "output_jsonl": str(data_path),
        "output_jsonl_sha256": actual_data_sha256,
        "detected_overlap_count": detected_overlap_count,
        "output_overlap_count": 0,
        "overlap_report_entries": len(overlap_report),
        "dataset_written_examples": written_examples,
        "dataset_eligible_examples": eligible_examples,
        "dataset_full_eligible_examples": full_eligible_examples,
        "dataset_selected_for_output_examples": selected_for_output_examples,
        "dataset_dropped_examples": dropped_examples,
        "dataset_overlap_examples": overlap_examples,
        "dataset_output_selection_skipped_answer_over_cap": skipped_answer_over_cap,
        "dataset_output_selection_skipped_answer_source_id_sha256": (
            skipped_answer_source_hashes
        ),
        "source_splits": {dataset: "train" for dataset in sorted(SUPPORTED_DATASETS)},
        "source_fingerprints": source_fingerprints,
        "test_v2_content_hash_check": content_hash_check,
        "raw_test_v2_read_by_converter": False,
        "overlap_policy": overlap_policy,
        "tokenizer": tokenizer_metadata,
        "prompt_protocol": {
            **expected_prompt_protocol,
            "source_file_sha256": prompt_source_sha256,
            "target_builder_source_file_sha256": target_builder_sha256,
            "runtime_sources_match_manifest": True,
        },
        "formal_build_not_smoke_manifest": True,
    }


def tokenizer_runtime_metadata(tokenizer: Any) -> dict[str, Any]:
    chat_template = getattr(tokenizer, "chat_template", None)
    return {
        "class": type(tokenizer).__name__,
        "vocab_size": getattr(tokenizer, "vocab_size", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "chat_template_sha256": (
            hashlib.sha256(chat_template.encode("utf-8")).hexdigest()
            if isinstance(chat_template, str)
            else None
        ),
    }


def validate_runtime_tokenizer_against_manifest(
    tokenizer: Any,
    manifest_path: Path,
    *,
    model_path: Path,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict):
        raise ValueError("formal converter manifest must be an object")
    expected = _validate_manifest_tokenizer_metadata(manifest)
    requested_revision = expected["requested_revision"]
    model_identifier = model_path.name
    model_revision_token = model_identifier.rsplit("-", 1)[-1]
    if model_revision_token != "59d61f3":
        raise ValueError(
            "formal smoke model path must retain the frozen 59d61f3 identifier"
        )
    requested = expected["requested_name_or_path"]
    resolved = expected.get("resolved_commit_hash")
    revision_locked = (
        model_revision_token in requested_revision
        or (isinstance(requested, str) and model_revision_token in requested)
        or (isinstance(resolved, str) and model_revision_token in resolved)
    )
    if not revision_locked:
        raise ValueError(
            "manifest tokenizer revision/name/hash must bind frozen model identifier "
            f"{model_revision_token}"
        )
    actual = tokenizer_runtime_metadata(tokenizer)
    for field in ("class", "vocab_size", "eos_token_id", "chat_template_sha256"):
        if expected.get(field) != actual[field]:
            raise ValueError(
                f"runtime tokenizer {field} drift: expected={expected.get(field)!r}, "
                f"actual={actual[field]!r}"
            )
    return {
        **actual,
        "requested_name_or_path": requested,
        "requested_revision": requested_revision,
        "resolved_commit_hash": resolved,
        "frozen_model_identifier": model_revision_token,
        "runtime_matches_manifest": True,
    }


def example_metadata(example: SupervisedSFTExample) -> dict[str, Any]:
    value = asdict(example)
    value.pop("input_ids")
    value.pop("labels")
    value["sequence_tokens"] = example.sequence_tokens
    value["context_was_truncated"] = example.context_was_truncated
    return value


class TinyLanguageModel(nn.Module):
    """Dependency-free helper used only by the unit gradient smoke."""

    def __init__(self, vocabulary: int, width: int) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocabulary, width)
        self.proj = nn.Linear(width, width)

    def forward(self, input_ids: torch.Tensor, **_: Any) -> SimpleNamespace:
        hidden = torch.tanh(self.proj(self.embed_tokens(input_ids)))
        return SimpleNamespace(last_hidden_state=hidden)
