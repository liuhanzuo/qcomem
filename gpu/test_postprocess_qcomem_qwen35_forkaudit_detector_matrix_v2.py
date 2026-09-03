from __future__ import annotations

import argparse
import copy
import hashlib
import tempfile
import unittest
from pathlib import Path

import build_qcomem_qwen35_forkaudit_detector_matrix_v2 as frozen_builder
import postprocess_qcomem_qwen35_forkaudit_detector_matrix_v2 as correction


class Rr2RunBindingCorrectionTest(unittest.TestCase):
    @staticmethod
    def _receipt() -> dict[str, object]:
        value: dict[str, object] = {
            "derivation": correction.RUN_ID_DERIVATION,
            "domain_hex": "746573742d646f6d61696e00",
            "generated_once_after_static_before_candidate_outputs": True,
            "nonce_hex": "11" * 32,
            "protocol_manifest_sha256": "22" * 32,
            "run_id_bits": 128,
            "schema_version": correction.RR2_RECEIPT_SCHEMA,
            "static_artifact_sha256": "33" * 32,
        }
        value["run_id"] = hashlib.sha256(
            bytes.fromhex(value["domain_hex"])
            + bytes.fromhex(value["static_artifact_sha256"])
            + bytes.fromhex(value["protocol_manifest_sha256"])
            + bytes.fromhex(value["nonce_hex"])
        ).digest()[:16].hex()
        return value

    def _fixture(self, root: Path) -> tuple[Path, Path, dict[str, object]]:
        rr2_root = root / "rr2"
        raw_root = rr2_root / "raw" / "shards"
        receipt_root = rr2_root / "receipts"
        raw_root.mkdir(parents=True)
        receipt_root.mkdir(parents=True)
        receipt = self._receipt()
        receipt_path = receipt_root / "run-id-receipt.json"
        frozen_builder.write_json(receipt_path, receipt)
        receipt_sha = frozen_builder.sha256_bytes(
            frozen_builder.canonical_bytes(receipt)
        )
        refs: list[dict[str, object]] = []
        for rank in range(8):
            mutants: dict[str, object] = {}
            for mutant_id in frozen_builder.ASSIGNMENT[rank]:
                mutants[mutant_id] = {
                    "outcome": {
                        "classification": "detected_expected_gate",
                        "expected_gate_id": frozen_builder.EXPECTED_GATES[mutant_id],
                        "observed_gate_id": frozen_builder.EXPECTED_GATES[mutant_id],
                        "restoration_verified": True,
                    },
                    "matched_clean": {"outcome": {"classification": "clean_pass"}},
                }
            shard = {
                "schema_version": "qcomem-forkaudit-review-shard-v1",
                "protocol": "qcomem-qwen35-forkaudit-review-revision-v1",
                "rank": rank,
                "run_id": receipt["run_id"],
                "run_id_receipt": receipt,
                "run_id_receipt_sha256": receipt_sha,
                "static_artifact_sha256": receipt["static_artifact_sha256"],
                "fault_campaign": {"mutants": mutants},
            }
            path = raw_root / f"forkaudit-shard-{rank}.json"
            frozen_builder.write_json(path, shard)
            payload = path.read_bytes()
            refs.append(
                {
                    "relative_path": f"shards/forkaudit-shard-{rank}.json",
                    "bytes": len(payload),
                    "sha256": frozen_builder.sha256_bytes(payload),
                }
            )
        manifest = {
            "schema_version": correction.RR2_MANIFEST_SCHEMA,
            "protocol": "qcomem-qwen35-forkaudit-review-revision-v1",
            "static_artifact_sha256": receipt["static_artifact_sha256"],
            "shards": refs,
        }
        manifest_path = receipt_root / "detached-receipt-manifest.json"
        frozen_builder.write_json(manifest_path, manifest)
        return rr2_root, manifest_path, receipt

    def _correct(self, root: Path) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
        rr2_root, manifest_path, receipt = self._fixture(root)
        return correction.corrected_original_receipts(
            original_receipt_manifest=manifest_path,
            original_rr2_root=rr2_root,
            expected_manifest_sha256=frozen_builder.sha256_file(manifest_path),
            expected_run_id=receipt["run_id"],
            expected_run_id_receipt_raw_sha256=frozen_builder.sha256_file(
                rr2_root / "receipts" / "run-id-receipt.json"
            ),
        )

    def test_only_null_generated_run_id_is_corrected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rr2_root, manifest_path, receipt = self._fixture(root)
            legacy = frozen_builder.original_receipts(
                original_receipt_manifest=manifest_path,
                original_rr2_root=rr2_root,
            )
            corrected, binding = correction.corrected_original_receipts(
                original_receipt_manifest=manifest_path,
                original_rr2_root=rr2_root,
                expected_manifest_sha256=frozen_builder.sha256_file(manifest_path),
                expected_run_id=receipt["run_id"],
                expected_run_id_receipt_raw_sha256=frozen_builder.sha256_file(
                    rr2_root / "receipts" / "run-id-receipt.json"
                ),
            )
            self.assertEqual(set(corrected), set(frozen_builder.MUTANT_IDS))
            self.assertTrue(all(row["run_id"] is None for row in legacy.values()))
            self.assertTrue(
                all(row["run_id"] == receipt["run_id"] for row in corrected.values())
            )
            for mutant_id in frozen_builder.MUTANT_IDS:
                before = copy.deepcopy(legacy[mutant_id])
                after = copy.deepcopy(corrected[mutant_id])
                before.pop("run_id")
                after.pop("run_id")
                self.assertEqual(before, after)
            self.assertEqual(binding["verified_run_id"], receipt["run_id"])
            self.assertTrue(binding["derivation_recomputed"])
            self.assertEqual(len(binding["verified_shards"]), 8)

    def test_rejects_unexpected_manifest_run_id_field(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rr2_root, manifest_path, receipt = self._fixture(root)
            manifest = frozen_builder.load_json(manifest_path)
            manifest["run_id"] = receipt["run_id"]
            frozen_builder.write_json(manifest_path, manifest)
            with self.assertRaisesRegex(
                frozen_builder.BuildError,
                "legacy manifest must omit top-level run_id",
            ):
                correction.validate_rr2_run_binding(
                    original_receipt_manifest=manifest_path,
                    original_rr2_root=rr2_root,
                    expected_manifest_sha256=frozen_builder.sha256_file(manifest_path),
                    expected_run_id=receipt["run_id"],
                    expected_run_id_receipt_raw_sha256=frozen_builder.sha256_file(
                        rr2_root / "receipts" / "run-id-receipt.json"
                    ),
                )

    def test_rejects_one_manifest_bound_shard_run_id_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rr2_root, manifest_path, receipt = self._fixture(root)
            manifest = frozen_builder.load_json(manifest_path)
            shard_path = rr2_root / "raw" / manifest["shards"][0]["relative_path"]
            shard = frozen_builder.load_json(shard_path)
            shard["run_id"] = "00" * 16
            frozen_builder.write_json(shard_path, shard)
            payload = shard_path.read_bytes()
            manifest["shards"][0]["bytes"] = len(payload)
            manifest["shards"][0]["sha256"] = frozen_builder.sha256_bytes(payload)
            frozen_builder.write_json(manifest_path, manifest)
            with self.assertRaisesRegex(frozen_builder.BuildError, "shard run ID rank 0"):
                correction.validate_rr2_run_binding(
                    original_receipt_manifest=manifest_path,
                    original_rr2_root=rr2_root,
                    expected_manifest_sha256=frozen_builder.sha256_file(manifest_path),
                    expected_run_id=receipt["run_id"],
                    expected_run_id_receipt_raw_sha256=frozen_builder.sha256_file(
                        rr2_root / "receipts" / "run-id-receipt.json"
                    ),
                )

    def test_raw_artifact_ledger_detects_postexecution_rewrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory)
            rank_root = run_root / "raw"
            rank_root.mkdir()
            lines: list[str] = []
            for rank in range(8):
                path = rank_root / f"detector-matrix-v2-rank-{rank}.json"
                path.write_bytes(f"rank-{rank}\n".encode("utf-8"))
                lines.append(
                    f"{frozen_builder.sha256_file(path)}  raw/{path.name}\n"
                )
            ledger = run_root / "raw-artifacts.sha256"
            ledger.write_text("".join(lines), encoding="utf-8")
            receipt = correction.validate_raw_artifact_ledger(
                ledger_path=ledger,
                rank_root=rank_root,
                expected_sha256=frozen_builder.sha256_file(ledger),
            )
            self.assertEqual(receipt["entry_count"], 8)
            (rank_root / "detector-matrix-v2-rank-3.json").write_bytes(b"rewritten\n")
            with self.assertRaisesRegex(
                frozen_builder.BuildError, "raw ledger artifact SHA"
            ):
                correction.validate_raw_artifact_ledger(
                    ledger_path=ledger,
                    rank_root=rank_root,
                    expected_sha256=frozen_builder.sha256_file(ledger),
                )

    def test_amendment_binds_frozen_and_postexecution_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rr2_root, manifest_path, receipt = self._fixture(root)
            prereg = root / "prereg.json"
            frozen_builder.write_json(prereg, {"fixture": True})
            raw_ledger = root / "raw-artifacts.sha256"
            raw_ledger.write_text("fixture\n", encoding="utf-8")
            amendment = {
                "schema_version": correction.AMENDMENT_SCHEMA,
                "workstream_id": "E-R28-FULL-DETECTOR-MATRIX",
                "created_after_candidate_execution": True,
                "candidate_outputs_observed_before_creation": True,
                "candidate_outputs_or_preexecution_sources_modified": False,
                "correction_source": {
                    "wrapper_sha256": frozen_builder.sha256_file(
                        Path(correction.__file__).resolve()
                    ),
                    "test_sha256": frozen_builder.sha256_file(Path(__file__).resolve()),
                },
                "frozen_execution_binding": {
                    "preexecution_builder_sha256": frozen_builder.sha256_file(
                        Path(frozen_builder.__file__).resolve()
                    ),
                    "preregistration_sha256": frozen_builder.sha256_file(prereg),
                    "original_rr2_manifest_sha256": frozen_builder.sha256_file(
                        manifest_path
                    ),
                    "original_rr2_run_id": receipt["run_id"],
                    "original_rr2_run_id_receipt_raw_sha256": frozen_builder.sha256_file(
                        rr2_root / "receipts" / "run-id-receipt.json"
                    ),
                    "raw_artifacts_ledger_sha256": frozen_builder.sha256_file(
                        raw_ledger
                    ),
                },
                "authorized_transformation": {
                    "input": "legacy generated RR2 comparison rows",
                    "field": "run_id",
                    "before": None,
                    "after": receipt["run_id"],
                    "authority": (
                        "derivation-verified canonical run-id receipt identically embedded "
                        "in all eight detached-manifest-bound RR2 shards"
                    ),
                    "all_other_fields_unchanged": True,
                },
            }
            amendment_path = root / "amendment.json"
            frozen_builder.write_json(amendment_path, amendment)
            args = argparse.Namespace(
                correction_amendment=amendment_path,
                expected_correction_amendment_sha256=frozen_builder.sha256_file(
                    amendment_path
                ),
                correction_test_file=Path(__file__).resolve(),
                preregistration=prereg,
                expected_preregistration_sha256=frozen_builder.sha256_file(prereg),
                original_receipt_manifest=manifest_path,
                original_rr2_root=rr2_root,
                raw_artifacts_ledger=raw_ledger,
            )
            observed, observed_sha = correction.validate_amendment(args)
            self.assertEqual(observed, amendment)
            self.assertEqual(observed_sha, frozen_builder.sha256_file(amendment_path))


if __name__ == "__main__":
    unittest.main()
