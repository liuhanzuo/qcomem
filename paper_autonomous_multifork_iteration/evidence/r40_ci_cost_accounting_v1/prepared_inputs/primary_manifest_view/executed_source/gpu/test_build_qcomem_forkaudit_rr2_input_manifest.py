from __future__ import annotations

import copy
import hashlib
import json
import shutil
import socket
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import torch

import build_qcomem_forkaudit_rr2_input_manifest as builder


class PositionTokenizer:
    vocab_size = 20_000_000
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = None
    unk_token_id = 3
    is_fast = True

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        if add_special_tokens:
            raise AssertionError("window builder must disable special tokens")
        return [index * 2048 + ord(character) for index, character in enumerate(text)]


class ConstantTokenizer(PositionTokenizer):
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        if add_special_tokens:
            raise AssertionError("window builder must disable special tokens")
        return [7] * len(text)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def synthetic_train64() -> tuple[bytes, bytes]:
    records = []
    objects = []
    for book_index in range(builder.FORMAL_TRAIN_RECORDS):
        source = f"train/{10000 + book_index}.txt"
        md5 = f"synthetic-md5-{book_index:02d}"
        # Each source and every position have deterministic token content.  The
        # >8K length covers the latest possible document/query-bank offset.
        text = "".join(
            chr(0x1000 + book_index * 64 + position % 61)
            for position in range(9000)
        )
        records.append(
            {
                "id": str(10000 + book_index),
                "_source_bucket": "deepmind-gutenberg",
                "_source_object": source,
                "_source_md5_base64": md5,
                "text": text,
            }
        )
        objects.append({"name": source, "md5_base64": md5})
    data = b"".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n"
        for row in records
    )
    manifest = {
        "bucket": "deepmind-gutenberg",
        "prefix": "train/",
        "test_or_validation_objects_used": False,
        "jsonl_sha256": _sha(data),
        "objects": objects,
    }
    manifest_bytes = builder.canonical_json_bytes(manifest) + b"\n"
    return data, manifest_bytes


def prior_manifest_bytes(
    coordinates: list[tuple[str, int]],
    *,
    windows_sha256: str = "a" * 64,
) -> bytes:
    banks = []
    for source, start in coordinates:
        banks.append(
            {
                "source_object": source,
                "document_start_token": start,
                "document_end_token_exclusive": start + builder.FORMAL_DOCUMENT_TOKENS,
            }
        )
    value = {
        "schema_version": 1,
        "protocol": "same-vllm-unified-attention-q16-multifork-resident-v1",
        "frozen_identity": {"pg19_windows_sha256": windows_sha256},
        "frozen_query_banks": banks,
    }
    return builder.canonical_json_bytes(value) + b"\n"


def nonoverlapping_prior(*, windows_sha256: str = "a" * 64) -> bytes:
    coordinates = [
        (f"train/{30000 + rank}.txt", rank * builder.FORMAL_WINDOW_STRIDE)
        for rank in range(builder.FORMAL_BOOKS)
    ]
    return prior_manifest_bytes(coordinates, windows_sha256=windows_sha256)


def write_model_artifacts(root: Path, *, tokenizer_layout: str = "vocab-merges") -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name in builder.REQUIRED_MODEL_ARTIFACTS:
        (root / name).write_bytes(f"synthetic-{name}\n".encode("utf-8"))
    if tokenizer_layout == "vocab-merges":
        for name in builder.TOKENIZER_BPE_LAYOUT_FILES:
            (root / name).write_bytes(f"synthetic-{name}\n".encode("utf-8"))
    elif tokenizer_layout == "tokenizer-json":
        (root / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    else:
        raise AssertionError(tokenizer_layout)
    (root / "chat_template.jinja").write_text("synthetic-template\n", encoding="utf-8")


def write_inputs(root: Path, *, prior: bytes | None = None) -> tuple[Path, Path, Path, Path, builder.InputExpectations]:
    data, manifest = synthetic_train64()
    prior = nonoverlapping_prior() if prior is None else prior
    data_path = root / "inputs" / "pg19.jsonl"
    manifest_path = root / "inputs" / "pg19-manifest.json"
    prior_path = root / "prior" / "capacity.json"
    model_dir = root / "model"
    data_path.parent.mkdir(parents=True)
    prior_path.parent.mkdir(parents=True)
    data_path.write_bytes(data)
    manifest_path.write_bytes(manifest)
    prior_path.write_bytes(prior)
    write_model_artifacts(model_dir)
    expectations = builder.InputExpectations(
        pg19_data_sha256=_sha(data),
        pg19_manifest_sha256=_sha(manifest),
        prior_manifest_sha256=_sha(prior),
        prior_windows_sha256=json.loads(prior)["frozen_identity"]["pg19_windows_sha256"],
        rr2_windows_sha256=None,
        rr2_coordinates=None,
    )
    return data_path, manifest_path, prior_path, model_dir, expectations


def build_at(root: Path, *, tokenizer: object | None = None, prior: bytes | None = None):
    data, manifest, prior_path, model_dir, expectations = write_inputs(root, prior=prior)
    value = builder.build_from_paths(
        pg19_data=data,
        pg19_manifest=manifest,
        prior_capacity_manifest=prior_path,
        model_dir=model_dir,
        tokenizer=PositionTokenizer() if tokenizer is None else tokenizer,
        expectations=expectations,
    )
    return value, expectations, (data, manifest, prior_path, model_dir)


class RR2InputManifestTest(unittest.TestCase):
    def setUp(self) -> None:
        self.assertFalse(torch.cuda.is_initialized())

    def tearDown(self) -> None:
        self.assertFalse(torch.cuda.is_initialized())

    def test_deterministic_path_relocation_and_no_network(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_root = root / "tree-a"
            second_root = root / "unrelated" / "relocated-tree-b"
            first_paths = write_inputs(first_root)
            shutil.copytree(first_root, second_root)
            data_a, manifest_a, prior_a, model_a, expectations = first_paths
            with mock.patch.object(socket, "create_connection", side_effect=AssertionError("network forbidden")):
                first = builder.build_from_paths(
                    pg19_data=data_a,
                    pg19_manifest=manifest_a,
                    prior_capacity_manifest=prior_a,
                    model_dir=model_a,
                    tokenizer=PositionTokenizer(),
                    expectations=expectations,
                )
                repeated = builder.build_from_paths(
                    pg19_data=data_a,
                    pg19_manifest=manifest_a,
                    prior_capacity_manifest=prior_a,
                    model_dir=model_a,
                    tokenizer=PositionTokenizer(),
                    expectations=expectations,
                )
                relocated = builder.build_from_paths(
                    pg19_data=second_root / "inputs/pg19.jsonl",
                    pg19_manifest=second_root / "inputs/pg19-manifest.json",
                    prior_capacity_manifest=second_root / "prior/capacity.json",
                    model_dir=second_root / "model",
                    tokenizer=PositionTokenizer(),
                    expectations=expectations,
                )
            self.assertEqual(first, repeated)
            self.assertEqual(first, relocated)
            self.assertNotIn(str(first_root), json.dumps(first))
            self.assertNotIn(str(second_root), json.dumps(first))
            self.assertFalse(first["build_audit"]["network_access_required"])

    def test_complete_window_bank_prefix_and_oracle_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            value, expectations, _ = build_at(Path(temporary))
        builder.validate_rr2_input_manifest(value, expectations=expectations)
        self.assertEqual(len(value["windows"]), 8)
        self.assertEqual(len(value["frozen_query_banks"]), 8)
        self.assertEqual(len(value["n_prefixes_by_rank"]), 24)
        self.assertEqual(len(value["oracle_selection_plan"]), 8)
        self.assertEqual(
            [row["resident_count"] for row in value["n_prefixes_by_rank"][:3]],
            [1, 8, 32],
        )
        bank = value["frozen_query_banks"][0]
        self.assertEqual(bank["query_bank_start_token"], bank["document_end_token_exclusive"] + 32)
        self.assertEqual(
            [row["source_token_offset"] for row in bank["rows"][:3]],
            [bank["query_bank_start_token"], bank["query_bank_start_token"] + 64, bank["query_bank_start_token"] + 128],
        )
        selection = value["oracle_selection_plan"][0]
        self.assertEqual(selection["arm_id"], builder.ORACLE_ARM_ID)
        self.assertEqual(selection["oracle_cell_id"], f"rank-0-N-1-{builder.ORACLE_ARM_ID}-ownership-witness")
        self.assertEqual(selection["document_token_ids_sha256"], bank["document_token_ids_sha256"])
        self.assertTrue(selection["locked_before_candidate_outputs"])

    def test_query_bank_tamper_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            value, expectations, _ = build_at(Path(temporary))
        tampered = copy.deepcopy(value)
        tampered["frozen_query_banks"][0]["rows"][0]["query_token_ids_sha256"] = "f" * 64
        with self.assertRaisesRegex(builder.RR2InputManifestError, "self hash"):
            builder.validate_rr2_input_manifest(tampered, expectations=expectations)

    def test_prior_byte_tamper_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, manifest, prior, model, expectations = write_inputs(root)
            prior.write_bytes(prior.read_bytes() + b"\n")
            with self.assertRaisesRegex(builder.RR2InputManifestError, "prior capacity manifest byte"):
                builder.build_from_paths(
                    pg19_data=data,
                    pg19_manifest=manifest,
                    prior_capacity_manifest=prior,
                    model_dir=model,
                    tokenizer=PositionTokenizer(),
                    expectations=expectations,
                )

    def test_pg19_byte_tamper_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, manifest, prior, model, expectations = write_inputs(root)
            data.write_bytes(data.read_bytes() + b"\n")
            with self.assertRaisesRegex(builder.RR2InputManifestError, "data SHA-256"):
                builder.build_from_paths(
                    pg19_data=data,
                    pg19_manifest=manifest,
                    prior_capacity_manifest=prior,
                    model_dir=model,
                    tokenizer=PositionTokenizer(),
                    expectations=expectations,
                )

    def test_source_start_overlap_with_prior_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            first, _, _ = build_at(Path(temporary) / "first")
            selected = first["windows"][0]
            coordinates = [
                (selected["source_object"], selected["document_start_token"]),
                *[(f"train/{40000 + rank}.txt", rank * builder.FORMAL_WINDOW_STRIDE) for rank in range(1, 8)],
            ]
            overlapping = prior_manifest_bytes(coordinates)
            root = Path(temporary) / "overlap"
            data, manifest, prior, model, expectations = write_inputs(root, prior=overlapping)
            with self.assertRaisesRegex(builder.RR2InputManifestError, "overlaps the prior"):
                builder.build_from_paths(
                    pg19_data=data,
                    pg19_manifest=manifest,
                    prior_capacity_manifest=prior,
                    model_dir=model,
                    tokenizer=PositionTokenizer(),
                    expectations=expectations,
                )

    def test_repeated_prior_windows_digest_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            first, _, _ = build_at(Path(temporary) / "first")
            repeated = nonoverlapping_prior(windows_sha256=first["pg19_windows_sha256"])
            root = Path(temporary) / "repeat"
            data, manifest, prior, model, expectations = write_inputs(root, prior=repeated)
            with self.assertRaisesRegex(builder.RR2InputManifestError, "repeats the prior"):
                builder.build_from_paths(
                    pg19_data=data,
                    pg19_manifest=manifest,
                    prior_capacity_manifest=prior,
                    model_dir=model,
                    tokenizer=PositionTokenizer(),
                    expectations=expectations,
                )

    def test_nonunique_queries_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, manifest, prior, model, expectations = write_inputs(root)
            with self.assertRaisesRegex(Exception, "pairwise distinct"):
                builder.build_from_paths(
                    pg19_data=data,
                    pg19_manifest=manifest,
                    prior_capacity_manifest=prior,
                    model_dir=model,
                    tokenizer=ConstantTokenizer(),
                    expectations=expectations,
                )

    def test_exact_archived_prior_coordinates_are_parsed(self) -> None:
        path = Path(__file__).with_name("qcomem_qwen35_vllm_paged_multifork_resident_protocol_manifest.json")
        prior = builder.parse_prior_capacity_manifest(
            path.read_bytes(), expectations=builder.FORMAL_EXPECTATIONS
        )
        observed = [
            (row["source_object"], row["document_start_token"], row["document_length"])
            for row in prior["coordinates"]
        ]
        self.assertEqual(
            observed,
            [
                ("train/10034.txt", 514, 4095),
                ("train/10.txt", 1028, 4095),
                ("train/10005.txt", 1285, 4095),
                ("train/10017.txt", 1285, 4095),
                ("train/10008.txt", 0, 4095),
                ("train/10016.txt", 257, 4095),
                ("train/10010.txt", 771, 4095),
                ("train/1004.txt", 257, 4095),
            ],
        )

    def test_formal_preoutput_probe_constants_are_exact(self) -> None:
        self.assertEqual(
            builder.FORMAL_RR2_WINDOWS_SHA256,
            "39bc36bb2eb04d51122e66caaebfa72367c02b43b073f072a2da240ed068c166",
        )
        self.assertEqual(
            builder.FORMAL_RR2_COORDINATES,
            (
                ("train/10.txt", 1542),
                ("train/10043.txt", 1028),
                ("train/10021.txt", 514),
                ("train/10009.txt", 514),
                ("train/10026.txt", 1542),
                ("train/10031.txt", 514),
                ("train/10045.txt", 1285),
                ("train/10059.txt", 1799),
            ),
        )

    def test_mutually_exclusive_real_tokenizer_layouts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            legacy = root / "legacy"
            write_model_artifacts(legacy, tokenizer_layout="vocab-merges")
            legacy_audit = builder.audit_model_tokenizer_artifacts(legacy)
            self.assertEqual(
                legacy_audit["selected_tokenizer_layout"]["layout_id"],
                "vocab-merges-bpe-v1",
            )
            self.assertEqual(
                {row["logical_name"] for row in legacy_audit["artifacts"]}
                & {"tokenizer.json", "vocab.json", "merges.txt"},
                {"vocab.json", "merges.txt"},
            )

            consolidated = root / "consolidated"
            write_model_artifacts(consolidated, tokenizer_layout="tokenizer-json")
            consolidated_audit = builder.audit_model_tokenizer_artifacts(consolidated)
            self.assertEqual(
                consolidated_audit["selected_tokenizer_layout"]["layout_id"],
                "consolidated-tokenizer-json-v1",
            )
            self.assertEqual(
                {row["logical_name"] for row in consolidated_audit["artifacts"]}
                & {"tokenizer.json", "vocab.json", "merges.txt"},
                {"tokenizer.json"},
            )

            (consolidated / "vocab.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(builder.RR2InputManifestError, "must not be mixed"):
                builder.audit_model_tokenizer_artifacts(consolidated)

            missing = root / "missing"
            write_model_artifacts(missing, tokenizer_layout="vocab-merges")
            (missing / "merges.txt").unlink()
            with self.assertRaisesRegex(builder.RR2InputManifestError, "complete vocab.json"):
                builder.audit_model_tokenizer_artifacts(missing)


if __name__ == "__main__":
    unittest.main()
