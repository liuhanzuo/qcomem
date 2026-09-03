from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


CONFIG_SCHEMA = "qcomem-supervised-sft-scale-split-config-v1"
MANIFEST_SCHEMA = "qcomem-supervised-sft-scale-split-v1"
PARENT_SCHEMA = "qcomem-supervised-qa-v1"
DATASETS = ("qasper", "2wikimqa")
FINGERPRINT_FIELDS = (
    "id_sha256",
    "context_input_sha256",
    "context_sha256",
    "input_sha256",
)
ASSIGNMENT_ALGORITHM = (
    "sorted_component_hash_first_reachable_exact_2d_subset_sum-v1"
)
HASH_RE = re.compile(r"[0-9a-f]{64}")
IGNORE_INDEX = -100


class SplitContractError(ValueError):
    pass


def stable_json(payload: Any, *, pretty: bool = False) -> str:
    if pretty:
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_text(value: str) -> str:
    if not isinstance(value, str):
        raise SplitContractError(f"expected text, found {type(value).__name__}")
    return " ".join(unicodedata.normalize("NFKC", value).split())


def example_fingerprints(source_id: str, context: str, input_text: str) -> dict[str, str]:
    normalized_id = normalize_text(source_id)
    normalized_context = normalize_text(context)
    normalized_input = normalize_text(input_text)
    return {
        "id_sha256": sha256_text(normalized_id),
        "context_input_sha256": sha256_text(
            stable_json([normalized_context, normalized_input])
        ),
        "context_sha256": sha256_text(normalized_context),
        "input_sha256": sha256_text(normalized_input),
    }


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or HASH_RE.fullmatch(value) is None:
        raise SplitContractError(f"{label} must be a lowercase SHA256")
    return value


def _require_nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SplitContractError(f"{label} must be a non-negative integer")
    return value


def _atomic_write(path: Path, text: str, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise SplitContractError(f"refusing to replace existing output {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SplitContractError("split config root must be an object")
    if payload.get("schema_version") != CONFIG_SCHEMA:
        raise SplitContractError(f"split config schema must be {CONFIG_SCHEMA}")
    if payload.get("assignment_algorithm") != ASSIGNMENT_ALGORITHM:
        raise SplitContractError("split assignment algorithm is not frozen")
    if payload.get("component_scope") != "global_across_datasets":
        raise SplitContractError("fingerprint components must span both datasets")
    if payload.get("component_fingerprint_fields") != list(FINGERPRINT_FIELDS):
        raise SplitContractError("component fingerprint fields are not frozen")
    if payload.get("output_order") != "parent_jsonl_official_source_order-v1":
        raise SplitContractError("output order is not frozen")
    if payload.get("forbid_non_train_sources") is not True:
        raise SplitContractError("split config must forbid non-train sources")
    if payload.get("require_test_v2_status") != "deferred_not_read":
        raise SplitContractError("split config must leave test-v2 unread")
    salt = payload.get("split_salt")
    if not isinstance(salt, str) or not salt.strip():
        raise SplitContractError("split salt must be non-empty")
    parent_selection = payload.get("parent_selection")
    expected_parent = {
        "answer_over_cap_policy": "skip_complete_answer_without_truncation",
        "full_train_scan_completed": True,
        "max_sequence_tokens": 1024,
        "strategy": "first_n_target_valid_eligible_in_official_source_order-v1",
        "target_validity_checked_before_selection": True,
    }
    if parent_selection != expected_parent:
        raise SplitContractError("parent selection contract is not frozen")
    counts = payload.get("dataset_counts")
    if not isinstance(counts, dict) or set(counts) != set(DATASETS):
        raise SplitContractError("dataset_counts must contain exactly qasper and 2wikimqa")
    for dataset in DATASETS:
        values = counts[dataset]
        if not isinstance(values, dict) or set(values) != {
            "pool",
            "train",
            "heldout_ce",
        }:
            raise SplitContractError(f"invalid count contract for {dataset}")
        for key, value in values.items():
            _require_nonnegative_int(value, f"dataset_counts.{dataset}.{key}")
        if values["pool"] != values["train"] + values["heldout_ce"]:
            raise SplitContractError(f"{dataset} pool must partition into train/heldout")
        if values["train"] < 1 or values["heldout_ce"] < 1:
            raise SplitContractError(f"{dataset} train/heldout counts must be positive")
    return payload


def validate_parent_manifest(
    manifest: dict[str, Any],
    *,
    parent_jsonl: Path,
    parent_jsonl_sha256: str,
    config: dict[str, Any],
) -> None:
    if manifest.get("schema_version") != PARENT_SCHEMA:
        raise SplitContractError("parent converter manifest schema mismatch")
    if manifest.get("status") != "passed" or manifest.get("mode") != "build":
        raise SplitContractError("parent must be a passed build-mode conversion")
    if manifest.get("output_jsonl_sha256") != parent_jsonl_sha256:
        raise SplitContractError("parent manifest does not bind parent JSONL SHA256")
    output_path = manifest.get("output_jsonl")
    if not isinstance(output_path, str) or Path(output_path).name != parent_jsonl.name:
        raise SplitContractError("parent manifest output basename mismatch")
    heldout = manifest.get("heldout_protocol")
    if not isinstance(heldout, dict):
        raise SplitContractError("parent heldout protocol is missing")
    if heldout.get("raw_test_v2_read_by_converter") is not False:
        raise SplitContractError("parent converter must attest test-v2 was not read")
    if heldout.get("test_v2_content_hash_check") != config[
        "require_test_v2_status"
    ]:
        raise SplitContractError("parent test-v2 status must remain deferred_not_read")
    if heldout.get("overlap_policy") != "drop":
        raise SplitContractError("parent pool must drop every consumed heldout overlap")
    if manifest.get("output_overlap_count") != 0:
        raise SplitContractError("parent output_overlap_count must be zero")
    detected = _require_nonnegative_int(
        manifest.get("detected_overlap_count"), "parent.detected_overlap_count"
    )
    reports = manifest.get("overlap_report")
    if not isinstance(reports, list) or len(reports) != detected:
        raise SplitContractError("parent overlap report/count mismatch")
    selection = manifest.get("output_selection")
    if not isinstance(selection, dict):
        raise SplitContractError("parent output_selection is missing")
    expected = config["parent_selection"]
    for key in (
        "strategy",
        "full_train_scan_completed",
        "target_validity_checked_before_selection",
        "answer_over_cap_policy",
    ):
        if selection.get(key) != expected[key]:
            raise SplitContractError(f"parent output_selection.{key} drifted")
    prompt = manifest.get("prompt_protocol")
    if not isinstance(prompt, dict) or prompt.get("max_sequence_tokens") != expected[
        "max_sequence_tokens"
    ]:
        raise SplitContractError("parent max-sequence contract drifted")
    stats = manifest.get("dataset_stats")
    if not isinstance(stats, dict) or set(stats) != set(DATASETS):
        raise SplitContractError("parent dataset stats are missing")
    dropped_total = 0
    for dataset in DATASETS:
        desired = config["dataset_counts"][dataset]["pool"]
        dataset_stats = stats[dataset]
        if not isinstance(dataset_stats, dict):
            raise SplitContractError(f"parent stats for {dataset} are invalid")
        for key in ("selected_for_output_examples", "written_examples"):
            if dataset_stats.get(key) != desired:
                raise SplitContractError(
                    f"parent {dataset} {key} must equal frozen pool count {desired}"
                )
        if selection.get("max_output_per_dataset") != desired or selection.get(
            "requested_max_output_per_dataset"
        ) != desired:
            raise SplitContractError("parent selected pool size does not match config")
        if dataset_stats.get("eligible_examples") != dataset_stats.get(
            "full_eligible_examples"
        ):
            raise SplitContractError(f"parent {dataset} full scan eligibility mismatch")
        if dataset_stats.get("overlap_examples") != dataset_stats.get(
            "dropped_examples"
        ):
            raise SplitContractError(f"parent {dataset} did not drop every overlap")
        dropped_total += _require_nonnegative_int(
            dataset_stats.get("dropped_examples"),
            f"parent.dataset_stats.{dataset}.dropped_examples",
        )
    if dropped_total != detected:
        raise SplitContractError("parent detected/dropped overlap totals differ")


def validate_parent_row(
    row: dict[str, Any],
    *,
    row_index: int,
    eos_token_id: int,
    max_sequence_tokens: int,
) -> dict[str, str]:
    label = f"parent row {row_index}"
    dataset = row.get("dataset")
    if dataset not in DATASETS:
        raise SplitContractError(f"{label} has invalid dataset")
    if row.get("source_split") != "train":
        raise SplitContractError(f"{label} is not from official train")
    provenance = row.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("source_split") != "train":
        raise SplitContractError(f"{label} provenance is not official train")
    source_id = row.get("source_id")
    context = row.get("context")
    input_text = row.get("input")
    if not all(isinstance(value, str) for value in (source_id, context, input_text)):
        raise SplitContractError(f"{label} source_id/context/input must be text")
    observed = provenance.get("fingerprints")
    if not isinstance(observed, dict) or set(observed) != set(FINGERPRINT_FIELDS):
        raise SplitContractError(f"{label} fingerprints are incomplete")
    expected = example_fingerprints(source_id, context, input_text)
    if observed != expected:
        raise SplitContractError(f"{label} fingerprints do not match raw train row")
    for field in FINGERPRINT_FIELDS:
        _require_sha256(observed[field], f"{label}.{field}")

    input_ids = row.get("input_ids")
    labels = row.get("labels")
    document_ids = row.get("document_input_ids")
    query_ids = row.get("query_input_ids")
    answer_ids = row.get("answer_input_ids")
    if not all(
        isinstance(value, list)
        for value in (input_ids, labels, document_ids, query_ids, answer_ids)
    ):
        raise SplitContractError(f"{label} token arrays are missing")
    if not answer_ids or answer_ids[-1] != eos_token_id:
        raise SplitContractError(f"{label} does not retain the complete answer plus EOS")
    if len(input_ids) != len(labels) or len(input_ids) > max_sequence_tokens:
        raise SplitContractError(f"{label} sequence/label length contract failed")
    prompt_length = len(document_ids) + len(query_ids)
    if document_ids + query_ids + answer_ids != input_ids:
        raise SplitContractError(f"{label} token parts do not reconstruct input_ids")
    if labels[:prompt_length] != [IGNORE_INDEX] * prompt_length:
        raise SplitContractError(f"{label} prompt labels are not fully masked")
    if labels[prompt_length:] != answer_ids:
        raise SplitContractError(f"{label} answer/EOS labels drifted")
    counts = row.get("token_counts")
    if not isinstance(counts, dict):
        raise SplitContractError(f"{label} token_counts are missing")
    expected_counts = {
        "total": len(input_ids),
        "prompt": prompt_length,
        "answer_with_eos": len(answer_ids),
        "query": len(query_ids),
    }
    for key, value in expected_counts.items():
        if counts.get(key) != value:
            raise SplitContractError(f"{label} token_counts.{key} drifted")
    answer_count = counts.get("answer")
    if answer_count != len(answer_ids) - 1:
        raise SplitContractError(f"{label} answer count must exclude one EOS token")
    return observed


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left = self.find(left)
        right = self.find(right)
        if left == right:
            return
        if self.rank[left] < self.rank[right]:
            left, right = right, left
        self.parent[right] = left
        if self.rank[left] == self.rank[right]:
            self.rank[left] += 1


def fingerprint_components(
    rows: list[dict[str, Any]], fingerprints: list[dict[str, str]], salt: str
) -> list[dict[str, Any]]:
    union_find = UnionFind(len(rows))
    first_owner: dict[tuple[str, str], int] = {}
    for row_index, values in enumerate(fingerprints):
        for field in FINGERPRINT_FIELDS:
            key = (field, values[field])
            owner = first_owner.setdefault(key, row_index)
            union_find.union(owner, row_index)
    members: dict[int, list[int]] = defaultdict(list)
    for row_index in range(len(rows)):
        members[union_find.find(row_index)].append(row_index)
    components = []
    for indices in members.values():
        identities = sorted(
            f"{rows[index]['dataset']}:{fingerprints[index]['id_sha256']}"
            for index in indices
        )
        digest = sha256_text(salt + "\0" + stable_json(identities))
        counts = Counter(rows[index]["dataset"] for index in indices)
        components.append(
            {
                "digest": digest,
                "indices": tuple(indices),
                "counts": (counts["qasper"], counts["2wikimqa"]),
            }
        )
    return sorted(components, key=lambda value: value["digest"])


def select_heldout_components(
    components: list[dict[str, Any]], *, qasper_count: int, twowiki_count: int
) -> set[str]:
    target = (qasper_count, twowiki_count)
    # The first path reaching a state is retained. Since components are sorted by
    # their frozen salted digest, this is deterministic and cannot use loss values.
    paths: dict[tuple[int, int], tuple[str, ...]] = {(0, 0): ()}
    for component in components:
        qasper_delta, twowiki_delta = component["counts"]
        for (qasper_so_far, twowiki_so_far), path in list(paths.items()):
            state = (
                qasper_so_far + qasper_delta,
                twowiki_so_far + twowiki_delta,
            )
            if state[0] > target[0] or state[1] > target[1] or state in paths:
                continue
            paths[state] = path + (component["digest"],)
        if target in paths:
            break
    if target not in paths:
        raise SplitContractError(
            "fingerprint-connected groups cannot produce the exact frozen heldout counts"
        )
    return set(paths[target])


def _length_summary(values: Iterable[int]) -> dict[str, int | float]:
    values = list(values)
    if not values:
        raise SplitContractError("cannot summarize an empty split")
    return {
        "count": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def split(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.split_config)
    expected_parent_jsonl_sha = _require_sha256(
        args.expected_parent_jsonl_sha256, "expected_parent_jsonl_sha256"
    )
    expected_parent_manifest_sha = _require_sha256(
        args.expected_parent_manifest_sha256, "expected_parent_manifest_sha256"
    )
    actual_parent_jsonl_sha = sha256_file(args.parent_jsonl)
    actual_parent_manifest_sha = sha256_file(args.parent_manifest)
    if actual_parent_jsonl_sha != expected_parent_jsonl_sha:
        raise SplitContractError("parent JSONL SHA256 does not match pre-registered value")
    if actual_parent_manifest_sha != expected_parent_manifest_sha:
        raise SplitContractError(
            "parent manifest SHA256 does not match pre-registered value"
        )
    parent_manifest = json.loads(args.parent_manifest.read_text(encoding="utf-8"))
    if not isinstance(parent_manifest, dict):
        raise SplitContractError("parent manifest must be an object")
    validate_parent_manifest(
        parent_manifest,
        parent_jsonl=args.parent_jsonl,
        parent_jsonl_sha256=actual_parent_jsonl_sha,
        config=config,
    )
    tokenizer = parent_manifest.get("tokenizer")
    if not isinstance(tokenizer, dict):
        raise SplitContractError("parent tokenizer metadata is missing")
    eos_token_id = _require_nonnegative_int(
        tokenizer.get("eos_token_id"), "parent.tokenizer.eos_token_id"
    )
    max_sequence_tokens = config["parent_selection"]["max_sequence_tokens"]

    rows: list[dict[str, Any]] = []
    fingerprints: list[dict[str, str]] = []
    canonical_lines: list[str] = []
    seen_source_ids: set[tuple[str, str]] = set()
    with args.parent_jsonl.open("r", encoding="utf-8") as handle:
        for row_index, raw_line in enumerate(handle):
            row = json.loads(raw_line)
            if not isinstance(row, dict):
                raise SplitContractError(f"parent row {row_index} must be an object")
            canonical = stable_json(row) + "\n"
            if raw_line != canonical:
                raise SplitContractError(f"parent row {row_index} is not canonical JSONL")
            observed = validate_parent_row(
                row,
                row_index=row_index,
                eos_token_id=eos_token_id,
                max_sequence_tokens=max_sequence_tokens,
            )
            source_key = (row["dataset"], row["source_id"])
            if source_key in seen_source_ids:
                raise SplitContractError(f"duplicate parent source ID {source_key}")
            seen_source_ids.add(source_key)
            rows.append(row)
            fingerprints.append(observed)
            canonical_lines.append(canonical)
    observed_counts = Counter(row["dataset"] for row in rows)
    expected_pool_counts = {
        dataset: config["dataset_counts"][dataset]["pool"] for dataset in DATASETS
    }
    if dict(observed_counts) != expected_pool_counts:
        raise SplitContractError(
            f"parent pool counts {dict(observed_counts)} != {expected_pool_counts}"
        )

    components = fingerprint_components(rows, fingerprints, config["split_salt"])
    heldout_components = select_heldout_components(
        components,
        qasper_count=config["dataset_counts"]["qasper"]["heldout_ce"],
        twowiki_count=config["dataset_counts"]["2wikimqa"]["heldout_ce"],
    )
    component_by_index: dict[int, str] = {}
    for component in components:
        for row_index in component["indices"]:
            component_by_index[row_index] = component["digest"]

    split_indices = {"train": [], "heldout_ce": []}
    assignment_lines: list[str] = []
    for row_index, row in enumerate(rows):
        component_digest = component_by_index[row_index]
        split_name = (
            "heldout_ce" if component_digest in heldout_components else "train"
        )
        split_indices[split_name].append(row_index)
        assignment = {
            "component_sha256": component_digest,
            "dataset": row["dataset"],
            "parent_row_index": row_index,
            "row_sha256": sha256_text(canonical_lines[row_index].rstrip("\n")),
            "source_id_sha256": fingerprints[row_index]["id_sha256"],
            "split": split_name,
        }
        assignment_lines.append(stable_json(assignment) + "\n")

    counts_by_split = {
        split_name: Counter(rows[index]["dataset"] for index in indices)
        for split_name, indices in split_indices.items()
    }
    expected_train = {
        dataset: config["dataset_counts"][dataset]["train"] for dataset in DATASETS
    }
    expected_heldout = {
        dataset: config["dataset_counts"][dataset]["heldout_ce"]
        for dataset in DATASETS
    }
    if dict(counts_by_split["train"]) != expected_train:
        raise SplitContractError("derived train counts do not match frozen config")
    if dict(counts_by_split["heldout_ce"]) != expected_heldout:
        raise SplitContractError("derived heldout counts do not match frozen config")

    fingerprint_intersections: dict[str, int] = {}
    for field in FINGERPRINT_FIELDS:
        train_values = {
            fingerprints[index][field] for index in split_indices["train"]
        }
        heldout_values = {
            fingerprints[index][field] for index in split_indices["heldout_ce"]
        }
        fingerprint_intersections[field] = len(train_values & heldout_values)
    train_sources = {
        (rows[index]["dataset"], rows[index]["source_id"])
        for index in split_indices["train"]
    }
    heldout_sources = {
        (rows[index]["dataset"], rows[index]["source_id"])
        for index in split_indices["heldout_ce"]
    }
    train_groups = {component_by_index[index] for index in split_indices["train"]}
    heldout_groups = {
        component_by_index[index] for index in split_indices["heldout_ce"]
    }
    if any(fingerprint_intersections.values()):
        raise SplitContractError("a fingerprint crosses train and CE-heldout")
    if train_sources & heldout_sources or train_groups & heldout_groups:
        raise SplitContractError("source/group leakage crosses train and CE-heldout")

    train_text = "".join(canonical_lines[index] for index in split_indices["train"])
    heldout_text = "".join(
        canonical_lines[index] for index in split_indices["heldout_ce"]
    )
    assignment_text = "".join(assignment_lines)
    output_shas = {
        "train_jsonl": sha256_text(train_text),
        "heldout_ce_jsonl": sha256_text(heldout_text),
        "assignment_ledger_jsonl": sha256_text(assignment_text),
    }
    split_summaries: dict[str, Any] = {}
    for split_name, indices in split_indices.items():
        split_summaries[split_name] = {
            "count": len(indices),
            "dataset_counts": dict(counts_by_split[split_name]),
            "sequence_tokens": _length_summary(
                len(rows[index]["input_ids"]) for index in indices
            ),
            "answer_with_eos_tokens": _length_summary(
                len(rows[index]["answer_input_ids"]) for index in indices
            ),
            "source_split_values": sorted(
                {rows[index]["source_split"] for index in indices}
            ),
        }

    component_size_histogram = Counter(
        len(component["indices"]) for component in components
    )
    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA,
        "status": "passed",
        "split_config": {
            "basename": args.split_config.name,
            "sha256": sha256_file(args.split_config),
            "payload": config,
        },
        "parent": {
            "jsonl_basename": args.parent_jsonl.name,
            "jsonl_sha256": actual_parent_jsonl_sha,
            "manifest_basename": args.parent_manifest.name,
            "manifest_sha256": actual_parent_manifest_sha,
            "converter_source_file_sha256": parent_manifest.get(
                "converter_source_file_sha256"
            ),
            "heldout_ledger_sha256": parent_manifest.get("heldout_ledger_sha256"),
            "detected_overlap_count": parent_manifest.get(
                "detected_overlap_count"
            ),
            "output_overlap_count": 0,
            "full_train_scan_completed": True,
            "raw_test_v2_read_by_converter": False,
            "test_v2_content_hash_check": "deferred_not_read",
        },
        "assignment": {
            "algorithm": ASSIGNMENT_ALGORITHM,
            "salt_sha256": sha256_text(config["split_salt"]),
            "uses_loss_or_model_outputs": False,
            "component_scope": "global_across_datasets",
            "fingerprint_fields": list(FINGERPRINT_FIELDS),
            "output_order": config["output_order"],
            "component_count": len(components),
            "heldout_component_count": len(heldout_components),
            "component_size_histogram": {
                str(size): count
                for size, count in sorted(component_size_histogram.items())
            },
        },
        "outputs": {
            "train_jsonl": {
                "basename": args.train_output.name,
                "sha256": output_shas["train_jsonl"],
                **split_summaries["train"],
            },
            "heldout_ce_jsonl": {
                "basename": args.heldout_output.name,
                "sha256": output_shas["heldout_ce_jsonl"],
                **split_summaries["heldout_ce"],
            },
            "assignment_ledger_jsonl": {
                "basename": args.assignment_ledger.name,
                "sha256": output_shas["assignment_ledger_jsonl"],
                "count": len(assignment_lines),
                "hash_only_no_raw_text": True,
            },
        },
        "disjoint_audit": {
            "source_id_intersection_count": len(train_sources & heldout_sources),
            "component_intersection_count": len(train_groups & heldout_groups),
            "fingerprint_intersection_counts": fingerprint_intersections,
            "all_zero": True,
        },
        "data_governance": {
            "all_rows_top_level_source_split": "train",
            "all_rows_provenance_source_split": "train",
            "validation_or_test_rows_used": False,
            "raw_test_v2_read": False,
            "heldout_ce_usage": config["heldout_usage"],
            "heldout_ce_is_final_downstream_evaluation": False,
        },
        "tokenizer": tokenizer,
        "prompt_protocol": parent_manifest.get("prompt_protocol"),
        "answer_target_policy": {
            "complete_answer_never_truncated": True,
            "eos_required": True,
            "prompt_labels_ignore_index": IGNORE_INDEX,
            "validated_rows": len(rows),
        },
    }
    manifest["manifest_preimage_sha256"] = sha256_text(stable_json(manifest))

    outputs = (
        args.train_output,
        args.heldout_output,
        args.assignment_ledger,
        args.manifest,
    )
    if not args.overwrite:
        existing = [str(path) for path in outputs if path.exists()]
        if existing:
            raise SplitContractError(f"refusing existing outputs: {existing}")
    _atomic_write(args.train_output, train_text, overwrite=args.overwrite)
    _atomic_write(args.heldout_output, heldout_text, overwrite=args.overwrite)
    _atomic_write(args.assignment_ledger, assignment_text, overwrite=args.overwrite)
    _atomic_write(args.manifest, stable_json(manifest, pretty=True), overwrite=args.overwrite)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split a leakage-audited, target-valid official-train SFT pool into "
            "fingerprint-disjoint multi-step train and CE-heldout artifacts"
        )
    )
    parser.add_argument("--split-config", type=Path, required=True)
    parser.add_argument("--parent-jsonl", type=Path, required=True)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--expected-parent-jsonl-sha256", required=True)
    parser.add_argument("--expected-parent-manifest-sha256", required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--heldout-output", type=Path, required=True)
    parser.add_argument("--assignment-ledger", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    try:
        manifest = split(parse_args())
    except (OSError, json.JSONDecodeError, SplitContractError) as error:
        raise SystemExit(f"scale split contract failed: {error}") from error
    print(
        stable_json(
            {
                "manifest_preimage_sha256": manifest["manifest_preimage_sha256"],
                "outputs": manifest["outputs"],
                "disjoint_audit": manifest["disjoint_audit"],
                "data_governance": manifest["data_governance"],
            },
            pretty=True,
        ),
        end="",
    )


if __name__ == "__main__":
    main()
