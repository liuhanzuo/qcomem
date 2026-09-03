from __future__ import annotations

import copy
import hashlib
import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path

import build_qcomem_forkaudit_fp32_calibration_manifest as builder


GPU_DIR = Path(__file__).resolve().parent
REPO_ROOT = GPU_DIR.parent
REAL_ARCHIVE = (
    REPO_ROOT
    / "results"
    / "gpu-qwen35-vllm-paged-fair-v2-20260814c"
)


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"


def metric(value: float) -> dict[str, object]:
    return {
        "bitwise_exact": False,
        "finite": True,
        "max_abs": value * 2.0,
        "mean_abs": value / 2.0,
        "relative_l2": value,
    }


def shard(rank: int, *, maximum: float = 0.0019) -> dict[str, object]:
    rows = []
    for offset, layer_idx in enumerate(builder.EXPECTED_LAYERS):
        value = maximum if rank == 7 and layer_idx == builder.EXPECTED_LAYERS[-1] else 0.001 + rank * 1e-5 + offset * 1e-6
        comparison = metric(value)
        rows.append(
            {
                "layer_idx": layer_idx,
                "passed": True,
                "same_post_rope_query_object": True,
                "query_sha256": digest(f"query-{rank}-{layer_idx}"),
                "position_ids_sha256": digest(f"position-{rank}"),
                "mask_sha256": digest(f"mask-{rank}"),
                "same_scale": True,
                "scaling": 0.0625,
                "backend_compatibility_nonblocking": {
                    "vllm_fresh_vs_fp32_dense": copy.deepcopy(comparison),
                    "vllm_reuse_vs_fp32_dense": copy.deepcopy(comparison),
                },
            }
        )
    identity = {
        "pg19_data_sha256": digest("pg19-data"),
        "pg19_manifest_sha256": digest("pg19-manifest"),
        "pg19_windows_sha256": digest("pg19-windows"),
        "protocol_config": {
            "pg19_document_tokens": builder.PRIOR_CONTEXT_DOCUMENT_TOKENS,
            "pg19_query_tokens": builder.PRIOR_CONTEXT_QUERY_TOKENS,
        },
    }
    return {
        "static": identity,
        "status": "completed_pg19_fair_v2_gate_shard",
        "passed": True,
        "rank": rank,
        "world_size": builder.EXPECTED_WORLD_SIZE,
        "fair_protocol": builder.FAIR_PROTOCOL,
        "quantization": "Q16",
        "single_request_only": True,
        "windows_sha256": digest("pg19-windows"),
        "rows": [
            {
                "window_index": rank,
                "source_object": f"train/{10000 + rank}.txt",
                "document_tokens": builder.PRIOR_CONTEXT_DOCUMENT_TOKENS,
                "query_tokens": builder.PRIOR_CONTEXT_QUERY_TOKENS,
                "isolated_same_kernel": {
                    "passed": True,
                    "fair_protocol": builder.FAIR_PROTOCOL,
                    "layer_count": len(builder.EXPECTED_LAYERS),
                    "layer_indices": list(builder.EXPECTED_LAYERS),
                    "rows": rows,
                },
            }
        ],
    }


def write_fixture(root: Path, *, maximum: float = 0.0019) -> None:
    shard_dir = root / builder.RAW_SHARD_DIR
    shard_dir.mkdir(parents=True)
    ledger_lines = []
    for rank in range(builder.EXPECTED_WORLD_SIZE):
        name = builder.RAW_SHARD_NAME.format(rank=rank)
        payload = json_bytes(shard(rank, maximum=maximum))
        (shard_dir / name).write_bytes(payload)
        ledger_lines.append(
            f"{hashlib.sha256(payload).hexdigest()}  /relocated/archive/{builder.RAW_SHARD_DIR}/{name}\n"
        )
    (root / builder.SCIENTIFIC_LEDGER_NAME).write_text(
        "".join(ledger_lines), encoding="utf-8"
    )


class ForkAuditFP32CalibrationManifestTest(unittest.TestCase):
    def test_real_archive_checks_prefixed_threshold_margin_and_binds_all_raw_bytes(self) -> None:
        manifest = builder.build_manifest(REAL_ARCHIVE)
        self.assertEqual(manifest["schema_version"], builder.SCHEMA_VERSION)
        self.assertEqual(manifest["diagnostic_definition"]["diagnostic_count"], 80)
        self.assertEqual(len(manifest["archive"]["raw_shards"]), 8)
        self.assertEqual(
            manifest["pre_fixed_threshold_margin_check"]["maximum_observed_prior_relative_l2"],
            0.001977153355255723,
        )
        margin = manifest["pre_fixed_threshold_margin_check"]
        self.assertEqual(margin["fixed_preregistered_threshold"], 0.005)
        self.assertFalse(margin["prior_rows_selected_or_tuned_threshold"])
        self.assertTrue(margin["fixed_threshold_at_least_twice_prior_maximum"])
        disjointness = manifest["rr2_disjointness_from_prior_context"]
        self.assertTrue(disjointness["document_length_disjoint"])
        self.assertEqual(
            disjointness["prior_context_document_token_values"],
            [1025],
        )
        for receipt in manifest["archive"]["raw_shards"]:
            raw = REAL_ARCHIVE / receipt["logical_name"]
            self.assertEqual(receipt["bytes"], len(raw.read_bytes()))
            self.assertEqual(receipt["sha256"], hashlib.sha256(raw.read_bytes()).hexdigest())

    def test_relocated_archive_produces_byte_identical_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            first = base / "host-a" / "archive"
            second = base / "different" / "deep" / "host-b" / "archive"
            write_fixture(first)
            shutil.copytree(first, second)
            second_ledger = second / builder.SCIENTIFIC_LEDGER_NAME
            second_ledger.write_text(
                second_ledger.read_text(encoding="utf-8").replace(
                    "/relocated/archive/", "/a/different/mounted/location/"
                ),
                encoding="utf-8",
            )
            first_bytes = builder._canonical_json_bytes(builder.build_manifest(first))
            second_bytes = builder._canonical_json_bytes(builder.build_manifest(second))
            self.assertEqual(first_bytes, second_bytes)
            lowered = first_bytes.lower()
            self.assertNotIn(str(first).encode("utf-8").lower(), lowered)
            self.assertNotIn(b"/relocated/archive", lowered)

    def test_duplicate_json_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_fixture(root)
            path = root / builder.RAW_SHARD_DIR / builder.RAW_SHARD_NAME.format(rank=0)
            path.write_text('{"rank":0,"rank":0}\n', encoding="utf-8")
            self._rewrite_ledger(root, path)
            with self.assertRaisesRegex(builder.CalibrationManifestError, "repeats key"):
                builder.build_manifest(root)

    def test_nonfinite_metric_is_rejected_before_margin_check(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_fixture(root)
            path = root / builder.RAW_SHARD_DIR / builder.RAW_SHARD_NAME.format(rank=0)
            value = json.loads(path.read_text(encoding="utf-8"))
            value["rows"][0]["isolated_same_kernel"]["rows"][0][
                "backend_compatibility_nonblocking"
            ]["vllm_reuse_vs_fp32_dense"]["relative_l2"] = math.inf
            path.write_text(json.dumps(value, allow_nan=True) + "\n", encoding="utf-8")
            self._rewrite_ledger(root, path)
            with self.assertRaisesRegex(builder.CalibrationManifestError, "non-finite constant"):
                builder.build_manifest(root)

    def test_fixed_threshold_margin_check_fails_closed_if_context_is_too_large(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_fixture(root, maximum=0.003)
            with self.assertRaisesRegex(builder.CalibrationManifestError, "lacks the required"):
                builder.build_manifest(root)

    def test_manifest_does_not_claim_prior_rows_derived_or_tuned_threshold(self) -> None:
        manifest = builder.build_manifest(REAL_ARCHIVE)
        serialized = builder._canonical_json_bytes(manifest).decode("utf-8").lower()
        for forbidden in (
            "threshold_derivation",
            "selected_threshold",
            "threshold_floor",
            "max(0.005",
            "derived threshold",
            "data-derived threshold",
            "calibrated threshold",
            "threshold calibration",
            "used to preregister",
        ):
            self.assertNotIn(forbidden, serialized)
        self.assertIn('"prior_archive_role": "contextual_validation_only"', serialized)
        self.assertIn('"prior_rows_selected_or_tuned_threshold": false', serialized)

    def test_scientific_ledger_digest_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_fixture(root)
            ledger = root / builder.SCIENTIFIC_LEDGER_NAME
            text = ledger.read_text(encoding="utf-8")
            ledger.write_text("0" * 64 + text[64:], encoding="utf-8")
            with self.assertRaisesRegex(builder.CalibrationManifestError, "digest mismatch"):
                builder.build_manifest(root)

    def test_boolean_numeric_field_and_duplicate_coordinate_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_fixture(root)
            path = root / builder.RAW_SHARD_DIR / builder.RAW_SHARD_NAME.format(rank=0)
            value = json.loads(path.read_text(encoding="utf-8"))
            value["rows"][0]["document_tokens"] = True
            path.write_bytes(json_bytes(value))
            self._rewrite_ledger(root, path)
            with self.assertRaisesRegex(builder.CalibrationManifestError, "integer >= 1"):
                builder.build_manifest(root)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_fixture(root)
            path = root / builder.RAW_SHARD_DIR / builder.RAW_SHARD_NAME.format(rank=1)
            value = json.loads(path.read_text(encoding="utf-8"))
            value["rows"][0]["source_object"] = "train/10000.txt"
            # Query hashes remain distinct, so the stricter per-window source gate catches this.
            path.write_bytes(json_bytes(value))
            self._rewrite_ledger(root, path)
            with self.assertRaisesRegex(builder.CalibrationManifestError, "source object is duplicated"):
                builder.build_manifest(root)

    @staticmethod
    def _rewrite_ledger(root: Path, changed_path: Path) -> None:
        ledger = root / builder.SCIENTIFIC_LEDGER_NAME
        name = changed_path.name
        digest_value = hashlib.sha256(changed_path.read_bytes()).hexdigest()
        lines = []
        for line in ledger.read_text(encoding="utf-8").splitlines():
            if line.endswith("/" + name):
                line = digest_value + line[64:]
            lines.append(line)
        ledger.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
