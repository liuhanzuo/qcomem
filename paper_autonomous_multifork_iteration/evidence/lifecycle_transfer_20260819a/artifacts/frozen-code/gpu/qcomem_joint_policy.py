from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F

from qcomem_torch import (
    LowerReplayState,
    PackedLowerReplayState,
    TorchSplitCausalLM,
    active_cache_layer_indices,
    cache_nbytes,
)


SUPPORTED_BITS = (2, 4, 8, 16)
FROZEN_VALIDATION_SHA256 = (
    "1553739496b3c209988de56a4ccf574d407379b6b7937ddfafacbe66925069fe"
)
FROZEN_TEST_V2_SHA256 = (
    "fe046477bf5b39629e9f66fd4def7a55c2d5d1f073c8bb601ee3833f08eaaa5f"
)
PG19_BUCKET = "deepmind-gutenberg"
PG19_PREFIX = "train/"
FROZEN_STATIC_LAYER_BITS = (8, 8, 8, 4, 8, 8, 8)


@dataclass(frozen=True)
class JointPolicy:
    name: str
    residual_bits: int
    cache_layer_bits: tuple[int, ...]
    selection_group: str
    predicted_component_bytes: int | None = None
    predicted_component_objective: float | None = None

    def __post_init__(self) -> None:
        if self.residual_bits not in SUPPORTED_BITS:
            raise ValueError(f"unsupported residual bits: {self.residual_bits}")
        if not self.cache_layer_bits:
            raise ValueError("cache_layer_bits must not be empty")
        if any(bits not in SUPPORTED_BITS for bits in self.cache_layer_bits):
            raise ValueError(f"unsupported cache bits: {self.cache_layer_bits}")

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["cache_layer_bits"] = list(self.cache_layer_bits)
        # Explicit per-layer bits cover every active layer. These two values are
        # merely fail-closed fallbacks for cache implementations with extra slots.
        payload["attention_bits"] = 16
        payload["linear_bits"] = 16
        return payload


@dataclass(frozen=True)
class PG19CalibrationWindow:
    source_id: str
    source_object: str
    start_token: int
    document_ids: torch.Tensor
    query_ids: torch.Tensor


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _forbidden_calibration_path(path: Path) -> bool:
    normalized = str(path).lower().replace("_", "-")
    return "longbench" in normalized or "qasper" in normalized or "2wikimqa" in normalized


def audit_pg19_train_calibration(
    data_path: Path,
    manifest_path: Path,
    *,
    expected_data_sha256: str,
    expected_manifest_sha256: str,
    minimum_books: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fail closed unless the input is provenance-locked PG-19 ``train/``.

    LongBench validation labels are deliberately unusable here: names, known
    digests, row schema, GCS bucket/prefix and the independent manifest must all
    agree before a tokenizer or model is loaded.
    """

    if minimum_books < 1:
        raise ValueError("minimum_books must be positive")
    if _forbidden_calibration_path(data_path) or _forbidden_calibration_path(
        manifest_path
    ):
        raise ValueError("joint policy calibration accepts PG-19 train paths only")
    data_sha256 = sha256_file(data_path)
    manifest_sha256 = sha256_file(manifest_path)
    if data_sha256 in {FROZEN_VALIDATION_SHA256, FROZEN_TEST_V2_SHA256}:
        raise ValueError("refusing a known LongBench validation/test digest")
    if data_sha256 != expected_data_sha256:
        raise ValueError(
            f"PG-19 data SHA256 mismatch: {data_sha256} != {expected_data_sha256}"
        )
    if manifest_sha256 != expected_manifest_sha256:
        raise ValueError(
            "PG-19 manifest SHA256 mismatch: "
            f"{manifest_sha256} != {expected_manifest_sha256}"
        )

    manifest = json.loads(manifest_path.read_text())
    if manifest.get("bucket") != PG19_BUCKET or manifest.get("prefix") != PG19_PREFIX:
        raise ValueError("manifest is not the official PG-19 train bucket/prefix")
    if manifest.get("test_or_validation_objects_used") is not False:
        raise ValueError("manifest does not explicitly exclude PG-19 test/validation")
    if manifest.get("jsonl_sha256") != data_sha256:
        raise ValueError("manifest JSONL digest does not match calibration data")
    objects = manifest.get("objects")
    if not isinstance(objects, list) or len(objects) < minimum_books:
        raise ValueError("manifest has too few PG-19 train objects")
    manifest_objects = {
        str(item.get("name")): item
        for item in objects
        if isinstance(item, dict) and item.get("name") is not None
    }

    records: list[dict[str, Any]] = []
    seen_sources: set[str] = set()
    with data_path.open() as stream:
        for line_index, line in enumerate(stream):
            if not line.strip():
                continue
            record = json.loads(line)
            if any(
                key in record
                for key in ("dataset", "_source_index", "answers", "input", "context")
            ):
                raise ValueError(
                    f"record {line_index} has an evaluation/QA schema, not PG-19"
                )
            if record.get("_source_bucket") != PG19_BUCKET:
                raise ValueError(f"record {line_index} has a non-PG-19 source bucket")
            source_object = str(record.get("_source_object", ""))
            if not source_object.startswith(PG19_PREFIX) or not source_object.endswith(
                ".txt"
            ):
                raise ValueError(f"record {line_index} is not a PG-19 train object")
            if source_object in seen_sources:
                raise ValueError(f"duplicate PG-19 source object: {source_object}")
            seen_sources.add(source_object)
            listed = manifest_objects.get(source_object)
            if listed is None:
                raise ValueError(f"{source_object} is missing from the manifest")
            if listed.get("md5_base64") != record.get("_source_md5_base64"):
                raise ValueError(f"{source_object} GCS MD5 provenance mismatch")
            if not isinstance(record.get("text"), str):
                raise ValueError(f"record {line_index} needs raw PG-19 text")
            records.append(record)
    if len(records) < minimum_books:
        raise ValueError(f"need {minimum_books} PG-19 books, found {len(records)}")
    if set(manifest_objects) != seen_sources:
        raise ValueError("manifest and JSONL PG-19 object sets differ")
    return records, {
        "data_sha256": data_sha256,
        "manifest_sha256": manifest_sha256,
        "records": len(records),
        "bucket": PG19_BUCKET,
        "prefix": PG19_PREFIX,
        "data_role": "pg19_train_development_calibration_only",
        "longbench_labels_used": False,
        "formal_validation_source_6_35_used": False,
        "frozen_test_v2_source_68_99_used": False,
    }


def _selection_key(seed: int, record: dict[str, Any]) -> str:
    source = str(record["_source_object"])
    return hashlib.sha256(f"{seed}|book|{source}".encode()).hexdigest()


def _window_choice(seed: int, source: str, choices: int) -> int:
    value = hashlib.sha256(f"{seed}|window|{source}".encode()).digest()
    return int.from_bytes(value[:8], "big") % choices


def build_pg19_calibration_windows(
    records: Sequence[dict[str, Any]],
    tokenizer: Any,
    *,
    books: int,
    document_tokens: int,
    query_tokens: int,
    stride: int,
    candidate_windows_per_book: int,
    seed: int,
) -> tuple[list[PG19CalibrationWindow], str]:
    if min(books, document_tokens, query_tokens, stride, candidate_windows_per_book) < 1:
        raise ValueError("window parameters must be positive")
    window_length = document_tokens + query_tokens
    max_length = window_length + stride * (candidate_windows_per_book - 1)
    windows: list[PG19CalibrationWindow] = []
    for record in sorted(records, key=lambda item: _selection_key(seed, item)):
        # Fast tokenizers still scan the entire multi-megabyte book before
        # applying ``max_length``. Grow a bounded character prefix until it has
        # a 64-token guard beyond the last usable token. This preserves every
        # selected token while making eight-rank startup deterministic and cheap.
        text = record["text"]
        guarded_tokens = max_length + 64
        character_limit = min(len(text), max(guarded_tokens * 6, 8192))
        while True:
            ids = tokenizer.encode(
                text[:character_limit], add_special_tokens=False
            )
            if len(ids) >= guarded_tokens or character_limit == len(text):
                break
            character_limit = min(len(text), character_limit * 2)
        ids = ids[:max_length]
        complete = min(
            candidate_windows_per_book,
            max(0, (len(ids) - window_length) // stride + 1),
        )
        if complete < 1:
            continue
        source = str(record["_source_object"])
        window_index = _window_choice(seed, source, complete)
        start = window_index * stride
        split = start + document_tokens
        end = split + query_tokens
        windows.append(
            PG19CalibrationWindow(
                source_id=str(record.get("id", Path(source).stem)),
                source_object=source,
                start_token=start,
                document_ids=torch.tensor(ids[start:split], dtype=torch.long),
                query_ids=torch.tensor(ids[split:end], dtype=torch.long),
            )
        )
        if len(windows) == books:
            break
    if len(windows) != books:
        raise ValueError(f"requested {books} complete PG-19 windows, found {len(windows)}")
    digest = hashlib.sha256()
    for window in windows:
        digest.update(window.source_object.encode())
        digest.update(str(window.start_token).encode())
        digest.update(window.document_ids.numpy().tobytes())
        digest.update(window.query_ids.numpy().tobytes())
    return windows, digest.hexdigest()


def selected_query_positions(query_length: int, positions: int) -> tuple[int, ...]:
    """Evenly sample positions whose next token is available as a PG-19 target."""

    if query_length < 2:
        raise ValueError("query_length must be at least two")
    if positions < 1:
        raise ValueError("positions must be positive")
    available = query_length - 1
    count = min(positions, available)
    if count == 1:
        return (available - 1,)
    values = {
        round(index * (available - 1) / (count - 1)) for index in range(count)
    }
    if len(values) != count:
        raise RuntimeError("query position selection unexpectedly produced duplicates")
    return tuple(sorted(values))


def policy_for_component(
    component_index: int, bits: int, *, depth: int
) -> JointPolicy:
    if component_index < 0 or component_index > depth:
        raise ValueError("component index is outside residual plus lower cache")
    layer_bits = [16] * depth
    if component_index == 0:
        residual_bits = bits
        component = "residual"
    else:
        residual_bits = 16
        layer_bits[component_index - 1] = bits
        component = f"cache.{component_index - 1}"
    return JointPolicy(
        name=f"profile-{component}-q{bits}",
        residual_bits=residual_bits,
        cache_layer_bits=tuple(layer_bits),
        selection_group="component_profile",
    )


def q16_policy(depth: int) -> JointPolicy:
    return JointPolicy(
        name="q16-control",
        residual_bits=16,
        cache_layer_bits=(16,) * depth,
        selection_group="control",
    )


def frozen_static_policy(depth: int) -> JointPolicy:
    if depth != len(FROZEN_STATIC_LAYER_BITS):
        raise ValueError("the preregistered frozen-static policy is depth 7")
    return JointPolicy(
        name="frozen-static-control",
        residual_bits=4,
        cache_layer_bits=FROZEN_STATIC_LAYER_BITS,
        selection_group="control",
    )


def uniform_q8_policy(depth: int) -> JointPolicy:
    return JointPolicy(
        name="uniform-q8-control",
        residual_bits=8,
        cache_layer_bits=(8,) * depth,
        selection_group="control",
    )


def policy_from_dict(payload: dict[str, Any]) -> JointPolicy:
    return JointPolicy(
        name=str(payload["name"]),
        residual_bits=int(payload["residual_bits"]),
        cache_layer_bits=tuple(int(value) for value in payload["cache_layer_bits"]),
        selection_group=str(payload["selection_group"]),
        predicted_component_bytes=(
            int(payload["predicted_component_bytes"])
            if payload.get("predicted_component_bytes") is not None
            else None
        ),
        predicted_component_objective=(
            float(payload["predicted_component_objective"])
            if payload.get("predicted_component_objective") is not None
            else None
        ),
    )


@torch.inference_mode()
def replay_selected_logits(
    adapter: TorchSplitCausalLM,
    state: LowerReplayState | PackedLowerReplayState,
    query_ids: torch.Tensor,
    positions: Sequence[int],
) -> torch.Tensor:
    """Return deployment-boundary logits at selected PG-19 query positions."""

    query_ids = adapter._batch_tokens(query_ids)
    local = state.fork()
    query_residual = adapter.continue_lower_replay(local, query_ids)
    suffix_cache = adapter.make_cache()
    adapter._run_layers(
        local.document_residual,
        state.depth,
        adapter.num_layers,
        past_key_values=suffix_cache,
        position_offset=0,
    )
    query_hidden = adapter._run_layers(
        query_residual,
        state.depth,
        adapter.num_layers,
        past_key_values=suffix_cache,
        position_offset=state.document_length,
    )
    selected = query_hidden[:, list(positions), :]
    selected = adapter.language_model.norm(selected)
    return adapter.lm_head(selected)[0]


@torch.inference_mode()
def quantized_policy_state(
    raw_state: LowerReplayState,
    policy: JointPolicy,
    *,
    group_size: int,
) -> PackedLowerReplayState:
    if len(policy.cache_layer_bits) != raw_state.depth:
        raise ValueError(
            f"policy has {len(policy.cache_layer_bits)} cache bits for depth {raw_state.depth}"
        )
    return raw_state.quantize(
        bits=policy.residual_bits,
        attention_bits=16,
        linear_bits=16,
        cache_layer_bits=policy.cache_layer_bits,
        group_size=group_size,
    )


def component_nbytes(
    packed: PackedLowerReplayState, component_index: int
) -> int:
    if component_index == 0:
        return packed.document_residual.nbytes
    active = active_cache_layer_indices(packed.cache.cache)
    cache_index = component_index - 1
    if cache_index < 0 or cache_index >= len(active):
        raise ValueError("cache component index is outside active packed layers")
    return cache_nbytes(packed.cache.cache.layers[active[cache_index]])


def logit_metric_sums(
    teacher_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, float | int]:
    if teacher_logits.shape != candidate_logits.shape:
        raise ValueError("teacher/candidate logit shapes differ")
    if teacher_logits.ndim != 2 or targets.shape != teacher_logits.shape[:1]:
        raise ValueError("expected logits [positions,vocab] and targets [positions]")
    teacher_logp = F.log_softmax(teacher_logits.float(), dim=-1)
    candidate_logp = F.log_softmax(candidate_logits.float(), dim=-1)
    teacher_probability = teacher_logp.exp()
    candidate_probability = candidate_logp.exp()
    forward = torch.sum(
        teacher_probability * (teacher_logp - candidate_logp), dim=-1
    )
    reverse = torch.sum(
        candidate_probability * (candidate_logp - teacher_logp), dim=-1
    )
    indices = targets.to(device=teacher_logits.device).view(-1, 1)
    teacher_nll = -torch.gather(teacher_logp, 1, indices).squeeze(1)
    candidate_nll = -torch.gather(candidate_logp, 1, indices).squeeze(1)
    nll_delta = candidate_nll - teacher_nll
    top1 = teacher_logits.argmax(dim=-1) == candidate_logits.argmax(dim=-1)
    difference = candidate_logits.float() - teacher_logits.float()
    return {
        "positions": int(targets.numel()),
        "forward_kl_sum": float(forward.sum().item()),
        "reverse_kl_sum": float(reverse.sum().item()),
        "teacher_nll_sum": float(teacher_nll.sum().item()),
        "candidate_nll_sum": float(candidate_nll.sum().item()),
        "nll_delta_sum": float(nll_delta.sum().item()),
        "positive_nll_delta_sum": float(torch.relu(nll_delta).sum().item()),
        "top1_matches": int(top1.sum().item()),
        "max_abs_logit_error": float(difference.abs().max().item()),
        "squared_logit_error_sum": float(torch.square(difference).sum().item()),
        "teacher_squared_logit_sum": float(
            torch.square(teacher_logits.float()).sum().item()
        ),
    }


def merge_metric_sums(rows: Iterable[dict[str, float | int]]) -> dict[str, Any]:
    rows = list(rows)
    if not rows:
        raise ValueError("cannot summarize empty metrics")
    additive = (
        "positions",
        "forward_kl_sum",
        "reverse_kl_sum",
        "teacher_nll_sum",
        "candidate_nll_sum",
        "nll_delta_sum",
        "positive_nll_delta_sum",
        "top1_matches",
        "squared_logit_error_sum",
        "teacher_squared_logit_sum",
    )
    sums = {key: sum(row[key] for row in rows) for key in additive}
    positions = int(sums["positions"])
    if positions < 1:
        raise ValueError("metric summary has no positions")
    mean_forward = float(sums["forward_kl_sum"]) / positions
    mean_positive_nll = float(sums["positive_nll_delta_sum"]) / positions
    agreement = int(sums["top1_matches"]) / positions
    # Pre-registered calibration-only scalar: distribution shift + actual PG-19
    # continuation harm + a small discrete trajectory penalty. It never uses QA
    # answers or LongBench validation labels.
    objective = mean_forward + mean_positive_nll + 0.1 * (1.0 - agreement)
    return {
        **sums,
        "windows": len(rows),
        "mean_forward_kl": mean_forward,
        "mean_reverse_kl": float(sums["reverse_kl_sum"]) / positions,
        "mean_teacher_nll": float(sums["teacher_nll_sum"]) / positions,
        "mean_candidate_nll": float(sums["candidate_nll_sum"]) / positions,
        "mean_nll_delta": float(sums["nll_delta_sum"]) / positions,
        "mean_positive_nll_delta": mean_positive_nll,
        "top1_agreement": agreement,
        "relative_logit_l2": math.sqrt(
            float(sums["squared_logit_error_sum"])
            / max(float(sums["teacher_squared_logit_sum"]), 1e-30)
        ),
        "max_abs_logit_error": max(
            float(row["max_abs_logit_error"]) for row in rows
        ),
        "joint_objective": objective,
        "joint_objective_definition": (
            "mean_forward_KL + mean_positive_PG19_next_token_NLL_delta + "
            "0.1*(1-top1_agreement)"
        ),
    }


def q16_exactness_passes(summary: dict[str, Any]) -> bool:
    return (
        float(summary["mean_forward_kl"]) <= 1e-6
        and float(summary["top1_agreement"]) == 1.0
        and float(summary["max_abs_logit_error"]) <= 1e-5
    )


def component_profile_order(profile: dict[str, Any]) -> int:
    component = str(profile["component"])
    if component == "residual":
        return 0
    if not component.startswith("cache."):
        raise ValueError(f"unknown component name: {component}")
    return int(component.split(".", 1)[1]) + 1


def profile_option(profile: dict[str, Any], bits: int) -> dict[str, Any]:
    try:
        return next(item for item in profile["options"] if int(item["bits"]) == bits)
    except StopIteration as error:
        raise ValueError(f"{profile['component']} has no Q{bits} profile") from error


def predicted_policy_cost(
    profiles: Sequence[dict[str, Any]], bits: Sequence[int]
) -> tuple[int, float]:
    if len(profiles) != len(bits):
        raise ValueError("policy bits must cover residual plus every cache component")
    byte_count = 0
    objective = 0.0
    for profile, value in zip(profiles, bits):
        option = profile_option(profile, int(value))
        byte_count += int(option["mean_component_nbytes"])
        objective += float(option["metrics"]["joint_objective"])
    return byte_count, objective


def top_predicted_policies(
    profiles: Sequence[dict[str, Any]],
    *,
    budget_bytes: int,
    limit: int,
    excluded_bits: Iterable[Sequence[int]] = (),
) -> list[tuple[tuple[int, ...], int, float]]:
    """Enumerate the depth-7 4^8 space and retain calibration candidates.

    The additive component score is used only to propose candidates. Final
    selection is made from actual *jointly quantized* PG-19 logits/NLL in the
    second stage, so no component-independence claim leaks into the result.
    """

    if budget_bytes < 1 or limit < 1:
        raise ValueError("budget and limit must be positive")
    profiles = sorted(profiles, key=component_profile_order)
    excluded = {tuple(int(value) for value in row) for row in excluded_bits}
    candidates: list[tuple[tuple[int, ...], int, float]] = []
    for choices in itertools.product(SUPPORTED_BITS, repeat=len(profiles)):
        if choices in excluded:
            continue
        byte_count, objective = predicted_policy_cost(profiles, choices)
        if byte_count <= budget_bytes:
            candidates.append((choices, byte_count, objective))
    candidates.sort(key=lambda row: (row[2], row[1], row[0]))
    if len(candidates) < limit:
        raise ValueError(
            f"budget {budget_bytes} produced only {len(candidates)} candidates"
        )
    return candidates[:limit]
