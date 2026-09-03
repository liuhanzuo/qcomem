from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import r39_compiled_dispatch_receipts as subject


class CompiledDispatchReceiptTests(unittest.TestCase):
    def test_fixture_replays_and_all_negative_controls_reject(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache, code, runtime, payload = subject._demo_fixture(root)
            result = subject.verify_payload(
                payload, cache_root=cache, code_root=code, runtime_root=runtime
            )
            self.assertEqual(result["replay_verdict"], "pass")
            self.assertEqual(result["gdn_document_prefill_call_count"], 1)
            report = root / "negative-controls.json"
            negative = subject.run_negative_controls(report)
            self.assertTrue(negative["all_rejected"])
            self.assertEqual(json.loads(report.read_text()), negative)

    def test_missing_ptx_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache, code, runtime, payload = subject._demo_fixture(root)
            next(cache.rglob("*.ptx")).unlink()
            with self.assertRaises(subject.DispatchReceiptError):
                subject.verify_payload(
                    payload, cache_root=cache, code_root=code, runtime_root=runtime
                )

    def test_bound_negative_controls_mutate_actual_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache, code, runtime, payload = subject._demo_fixture(root)
            receipt = root / "receipt.json"
            subject._write_json(receipt, payload)
            report = root / "bound-negative-controls.json"
            controls = subject.run_bound_negative_controls(
                receipt=receipt,
                cache_root=cache,
                code_root=code,
                runtime_root=runtime,
                output=report,
            )
            self.assertTrue(controls["all_rejected"])
            self.assertEqual(
                set(controls["negative_controls"]),
                {
                    "receipt-config-tamper",
                    "receipt-artifact-id-tamper",
                    "missing-required-ptx",
                    "extra-unreceipted-artifact",
                    "compiled-artifact-substitution",
                    "gdn-runtime-source-substitution",
                    "gdn-cache-rebind-source-substitution",
                },
            )

    def test_source_snapshot_is_self_contained_and_substitution_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache, code, runtime, payload = subject._demo_fixture(root / "fixture")
            snapshot = root / "snapshot"
            manifest = root / "snapshot.json"
            observed = subject.snapshot_bound_sources(
                payload=payload,
                code_root=code,
                runtime_root=runtime,
                target=snapshot,
                output=manifest,
            )
            self.assertEqual(observed["source_file_count"], 3)
            subject.verify_payload(
                payload,
                cache_root=cache,
                code_root=snapshot / "code",
                runtime_root=snapshot / "runtime",
            )
            selected = (
                snapshot
                / "runtime/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py"
            )
            selected.write_text("# substituted\n", encoding="utf-8")
            with self.assertRaises(subject.DispatchReceiptError):
                subject.verify_payload(
                    payload,
                    cache_root=cache,
                    code_root=snapshot / "code",
                    runtime_root=snapshot / "runtime",
                )


if __name__ == "__main__":
    unittest.main()
