from __future__ import annotations

import argparse
import contextlib
import io
import itertools
import json
import os
import zipfile
from pathlib import Path
from typing import Any, Iterator, TextIO, Union

from prepare_supervised_qa_train import (
    DATASETS,
    EXPECTED_HELDOUT_INDICES,
    FINGERPRINT_FIELDS,
    FROZEN_LONGBENCH_REVISION,
    FROZEN_TEST_V2_FILE_SHA256,
    LEDGER_SCHEMA_VERSION,
    DataContractError,
    _reject_raw_heldout_content,
    example_fingerprints,
    stable_json,
    validate_heldout_ledger,
)


BLIND_SCHEMA_VERSION = "qcomem-frozen-test-v2-blind-fingerprints-v1"


DatasetSource = Union[Path, tuple[Path, str]]


def parse_dataset(value: str) -> tuple[str, DatasetSource]:
    name, separator, raw_path = value.partition("=")
    if not separator or name not in DATASETS or not raw_path:
        raise argparse.ArgumentTypeError(
            "dataset must be qasper=PATH, 2wikimqa=PATH, or NAME=ZIP::MEMBER"
        )
    archive_path, member_separator, member = raw_path.partition("::")
    if member_separator:
        if not archive_path or not member:
            raise argparse.ArgumentTypeError("ZIP::MEMBER source is incomplete")
        return name, (Path(archive_path), member)
    return name, Path(raw_path)


@contextlib.contextmanager
def open_dataset_source(source: DatasetSource) -> Iterator[TextIO]:
    if isinstance(source, tuple):
        archive_path, member = source
        with zipfile.ZipFile(archive_path) as archive:
            with archive.open(member) as raw:
                with io.TextIOWrapper(raw, encoding="utf-8") as handle:
                    yield handle
    else:
        with source.open(encoding="utf-8") as handle:
            yield handle


def consumed_entries(source: DatasetSource, dataset: str) -> list[dict[str, Any]]:
    """Read only source indices 0--67; index 68/test-v2 is never consumed."""

    rows = []
    with open_dataset_source(source) as handle:
        for physical_index, line in enumerate(itertools.islice(handle, 68)):
            row = json.loads(line)
            if not isinstance(row, dict):
                raise DataContractError(f"{dataset} row {physical_index} is not an object")
            source_index = int(row.get("_source_index", physical_index))
            if source_index > 67:
                raise DataContractError(
                    f"{dataset} input unexpectedly contains source index {source_index}; "
                    "this builder refuses test-v2"
                )
            if row.get("dataset") != dataset:
                raise DataContractError(
                    f"{dataset} source index {source_index} has wrong dataset field"
                )
            source_id = row.get("_id")
            context = row.get("context")
            input_text = row.get("input")
            if not all(isinstance(value, str) for value in (source_id, context, input_text)):
                raise DataContractError(
                    f"{dataset} source index {source_index} lacks _id/context/input"
                )
            rows.append(
                {
                    "dataset": dataset,
                    "source_index": source_index,
                    **example_fingerprints(source_id, context, input_text),
                }
            )
    return rows


def load_blind_test_entries(path: Path | None) -> tuple[str, list[dict[str, Any]]]:
    if path is None:
        return "deferred_not_read", []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DataContractError("blind test-v2 fingerprint manifest must be an object")
    _reject_raw_heldout_content(payload, "blind_test_v2")
    if payload.get("schema_version") != BLIND_SCHEMA_VERSION:
        raise DataContractError(f"blind manifest schema must be {BLIND_SCHEMA_VERSION}")
    if payload.get("source_revision") != FROZEN_LONGBENCH_REVISION:
        raise DataContractError("blind manifest source revision mismatch")
    if payload.get("source_file_sha256") != FROZEN_TEST_V2_FILE_SHA256:
        raise DataContractError("blind manifest test-v2 file SHA256 mismatch")
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise DataContractError("blind manifest entries must be a list")
    allowed = {"dataset", "source_index", *FINGERPRINT_FIELDS}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or set(entry) != allowed:
            raise DataContractError(f"blind manifest entry {index} has invalid schema")
    return "blind_hash_manifest", entries


def build_ledger(
    dataset_paths: dict[str, DatasetSource], blind_test_v2: Path | None
) -> dict[str, Any]:
    all_entries = []
    for dataset in DATASETS:
        all_entries.extend(consumed_entries(dataset_paths[dataset], dataset))
    status, frozen_entries = load_blind_test_entries(blind_test_v2)
    splits: dict[str, Any] = {}
    for split_name, source_indices in EXPECTED_HELDOUT_INDICES.items():
        entries = (
            frozen_entries
            if split_name == "frozen_test_v2"
            else [
                entry
                for entry in all_entries
                if entry["source_index"] in source_indices
            ]
        )
        split = {
            "source_revision": FROZEN_LONGBENCH_REVISION,
            "content_scope": (
                "hash_only_external_manifest"
                if split_name == "frozen_test_v2" and status == "blind_hash_manifest"
                else (
                    "deferred_without_reading_test_v2"
                    if split_name == "frozen_test_v2"
                    else "consumed_longbench_rows_context_input_id_only"
                )
            ),
            "entries": sorted(
                entries, key=lambda row: (row["dataset"], row["source_index"])
            ),
        }
        if split_name == "frozen_test_v2":
            split["status"] = status
            split["source_file_sha256"] = FROZEN_TEST_V2_FILE_SHA256
        splits[split_name] = split
    payload = {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "source_repo": "zai-org/LongBench",
        "source_revision": FROZEN_LONGBENCH_REVISION,
        "raw_answers_used": False,
        "raw_test_v2_read": False,
        "splits": splits,
    }
    validate_heldout_ledger(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build SFT overlap ledger from already-consumed LongBench rows 0--67. "
            "This command cannot read test-v2."
        )
    )
    parser.add_argument("--dataset", action="append", type=parse_dataset, required=True)
    parser.add_argument("--blind-test-v2-hashes", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dataset_paths: dict[str, DatasetSource] = dict(args.dataset)
    if set(dataset_paths) != set(DATASETS) or len(args.dataset) != len(DATASETS):
        parser.error("provide each of qasper=PATH and 2wikimqa=PATH exactly once")
    try:
        payload = build_ledger(dataset_paths, args.blind_test_v2_hashes)
    except DataContractError as error:
        raise SystemExit(f"ledger contract failed: {error}") from error
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(stable_json(payload, pretty=True), encoding="utf-8")
    os.replace(temporary, args.output)
    print(
        stable_json(
            {
                "output": str(args.output),
                "test_v2_content_hash_check": payload["splits"]["frozen_test_v2"][
                    "status"
                ],
                "counts": {
                    split: len(body["entries"])
                    for split, body in payload["splits"].items()
                },
            },
            pretty=True,
        ),
        end="",
    )


if __name__ == "__main__":
    main()
